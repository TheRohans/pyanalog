#!/usr/bin/env python3
#
# Copyright (c) 2026 The Rohans Limited
# Added in a fork of anabrid's pyanalog: https://github.com/anabrid/pyanalog
# Licensed under the same terms as the rest of this repository (see LICENSE.GPL3).
#

"""
Tests for :mod:`dda.falstad`, the CircuitJS1 export.

Most of these check *electrical* correctness, not just "did it produce
text" -- every op-amp/resistor/capacitor/ground/rail post referenced by a
wire has to actually be reachable, or CircuitJS1 will refuse to simulate
the circuit ("bad connection" / "wire loop detected"). That's exactly the
class of bug this exporter had during development (op-amp "+"/"-" pins are
*derived* by CircuitJS1 from a centerline anchor, not literal coordinates;
a wire endpoint touching another wire's midpoint isn't a connection). The
`_endpoints` helper below reconstructs the same real-post model CircuitJS1
uses and asserts every non-rail-stub post has at least 2 references, which
is a cheap, fast proxy for "would this actually load."
"""

import collections
import pytest

from dda import State, Symbol, symbols, dda, export
from dda.falstad import to_falstad


def _endpoints(text):
    """
    Reconstructs CircuitJS1's real electrical posts from a dump and returns
    {coordinate: reference_count}. Mirrors CircuitJS1's own post model:
    - 'g' / 'R': first coordinate pair is the real post.
    - 'a' (OpAmpElm): (x1,y1) is a centerline anchor, NOT a post -- the real
      posts are (x1,y1-16) ["-"], (x1,y1+16) ["+"], and (x2,y2) [output].
    - everything else ('w','r','c'): both coordinate pairs are real posts.
    """
    counts = collections.Counter()
    for line in text.splitlines()[1:]:
        if not line.strip():
            continue
        t = line.split()
        kind = t[0]
        x1, y1, x2, y2 = int(t[1]), int(t[2]), int(t[3]), int(t[4])
        if kind in ("g", "R"):
            counts[(x1, y1)] += 1
        elif kind == "x":
            continue
        elif kind == "a":
            counts[(x1, y1 - 16)] += 1
            counts[(x1, y1 + 16)] += 1
            counts[(x2, y2)] += 1
        else:
            counts[(x1, y1)] += 1
            counts[(x2, y2)] += 1
    return counts


def _assert_fully_connected(circuit):
    """Every real post must be referenced at least twice, or it's dangling."""
    counts = _endpoints(circuit.text)
    dangling = [xy for xy, n in counts.items() if n < 2]
    assert not dangling, f"dangling post(s) with <2 references: {dangling}\n{circuit.text}"


def test_export_dispatch():
    x, y = symbols("x, y")
    s = State({y: dda.neg(x)})
    circuit = export(s, to="falstad")
    assert isinstance(circuit, to_falstad)
    assert circuit.text.startswith("$ ")


def test_static_mx_plus_b_is_fully_connected():
    "y = m*x + b, both m and x live -- both mult stages are unsupported (no analog multiplier)."
    x, m, b, y = symbols("x, m, b, y")
    s = State({y: dda.neg(dda.sum(dda.mult(m, x), b))})
    circuit = to_falstad(s)
    assert sorted(circuit.unsupported) == ["mult_1"]


def test_decay_self_loop_is_fully_connected():
    "dx/dt = -x: a single integrator feeding back on itself."
    xx = Symbol("xx")
    s = State({xx: dda.int(xx, 1, 1)})
    circuit = to_falstad(s)
    assert circuit.unsupported == []
    _assert_fully_connected(circuit)


def test_oscillator_is_fully_connected():
    "y'' = -y via two cross-coupled integrators -- exercises the feedback (back-edge) wiring."
    yv, myv, mdyv = symbols("yv, myv, mdyv")
    s = State()
    s[myv] = dda.neg(yv)
    s[yv] = dda.int(mdyv, 1, 0)
    s[mdyv] = dda.int(myv, 1, 1)
    circuit = to_falstad(s)
    assert circuit.unsupported == []
    _assert_fully_connected(circuit)


def test_literal_coefficient_both_signs_is_fully_connected():
    "Exercises the 2-stage (sign-fixing) realization for k>0, the 1-stage for k<0, and a bare literal in sum()."
    xx, y = symbols("xx, y")
    s = State({y: dda.neg(dda.sum(dda.mult(2, xx), -3))})
    circuit = to_falstad(s)
    assert circuit.unsupported == []
    _assert_fully_connected(circuit)


def test_negative_literal_coefficient_needs_only_one_stage():
    "k<0 doesn't need the extra sign-fixing inverter (-|k|*x is already k*x)."
    xx, y = symbols("xx, y")
    s = State({y: dda.mult(-2, xx)})
    circuit = to_falstad(s)
    n_opamps = sum(1 for l in circuit.text.splitlines() if l.startswith("a "))
    assert n_opamps == 1
    _assert_fully_connected(circuit)


def test_positive_literal_coefficient_needs_two_stages():
    "k>0 needs a chained sign-fixing inverter, since mult() itself isn't inverting but one op-amp stage always is."
    xx, y = symbols("xx, y")
    s = State({y: dda.mult(2, xx)})
    circuit = to_falstad(s)
    n_opamps = sum(1 for l in circuit.text.splitlines() if l.startswith("a "))
    assert n_opamps == 2
    _assert_fully_connected(circuit)


def test_topological_order_independent_of_dict_insertion_order():
    """
    name_computing_elements() names the *outermost* call first (e.g. the
    final output var can appear before its own dependency in dict order).
    A naive dict-order walk wires a node before the thing it depends on
    exists yet -- this must not happen.
    """
    xx, y = symbols("xx2, y2")
    s = State({y: dda.neg(dda.sum(dda.mult(2, xx), -3))})
    circuit = to_falstad(s)
    _assert_fully_connected(circuit)


def test_two_variable_mult_is_flagged_unsupported_not_faked():
    p, q, r = symbols("p, q, r")
    s = State({r: dda.mult(p, q)})
    circuit = to_falstad(s)
    assert circuit.unsupported == ["r"]
    assert "no\\ analog\\ multiplier" in circuit.text  # _label() escapes spaces for CircuitJS1's text element


def test_empty_state_raises():
    with pytest.raises(ValueError):
        to_falstad(State())


def test_save_writes_file(tmp_path):
    x, y = symbols("x, y")
    s = State({y: dda.neg(x)})
    circuit = to_falstad(s)
    target = tmp_path / "out.txt"
    returned = circuit.save(str(target))
    assert returned == str(target)
    assert target.read_text() == circuit.text


def test_url_is_a_valid_falstad_link():
    x, y = symbols("x, y")
    s = State({y: dda.neg(x)})
    circuit = to_falstad(s)
    url = circuit.url()
    assert url.startswith("https://www.falstad.com/circuit/circuitjs.html?cct=")
