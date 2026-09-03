#!/usr/bin/env python3
#
# Copyright (c) 2026 The Rohans Limited
# Added in a fork of anabrid's pyanalog: https://github.com/anabrid/pyanalog
# Licensed under the same terms as the rest of this repository (see LICENSE.GPL3).
#

"""
Tests for :mod:`dda.schematic`, the SVG block-schematic exporter.
"""

import pytest

from dda import State, Symbol, symbols, dda, export
from dda.schematic import to_schematic


def test_export_dispatch():
    "export(state, to='schematic') should resolve to the schematic exporter."
    x, y = symbols("x, y")
    s = State({y: dda.neg(x)})
    pic = export(s, to="schematic")
    assert isinstance(pic, to_schematic)
    assert "<svg" in pic.svg


def test_static_circuit_layering():
    "y = m*x + b: no feedback, and inputs come out strictly before y."
    x, m, b, y = symbols("x, m, b, y")
    s = State({y: dda.neg(dda.sum(dda.mult(m, x), b))})
    pic = to_schematic(s)

    assert pic.back_edges == set()
    assert pic.final_outputs == ["y"]
    for free_var in ("x", "m", "b"):
        assert pic.layer[free_var] == 0
    assert pic.layer["y"] == max(pic.layer.values())


def test_self_loop_integrator_is_a_back_edge():
    "dx/dt = -x is a single integrator feeding back on itself."
    xx = Symbol("xx")
    s = State({xx: dda.int(xx, 1, 1)})
    pic = to_schematic(s)

    assert ("xx", "xx") in pic.back_edges
    assert pic.layer["xx"] == 0
    assert "dt=1" in pic.svg and "ic=1" in pic.svg


def test_oscillator_cycle_breaks_into_a_dag_plus_one_back_edge():
    "y'' = -y via two cross-coupled integrators is a 3-cycle at the graph level."
    yv, myv, mdyv = symbols("yv, myv, mdyv")
    s = State()
    s[myv] = dda.neg(yv)
    s[yv] = dda.int(mdyv, 1, 0)
    s[mdyv] = dda.int(myv, 1, 1)
    pic = to_schematic(s)

    assert len(pic.back_edges) == 1
    # the layered (non-feedback) part must be a strict chain of 3 distinct layers
    assert sorted(pic.layer.values()) == [0, 1, 2]


def test_variable_times_variable_is_a_multiplier_ic_not_a_coefficient():
    "mult(p, q) with two signal operands can't be a fixed coefficient pot."
    p, q, r = symbols("p, q, r")
    s = State({r: dda.mult(p, q)})
    pic = to_schematic(s)
    assert "IC" in pic.svg


def test_literal_coefficient_multiply_is_labeled_with_its_value():
    "mult(2, x) has one literal operand -> render the coefficient value."
    x, r = symbols("x, r")
    s = State({r: dda.mult(2, x)})
    pic = to_schematic(s)
    assert "×2" in pic.svg


def test_unmapped_primitive_is_flagged_generic():
    "sqrt has no canonical op-amp circuit; it must show up as the generic/warn box."
    p, z = symbols("p, z")
    s = State({z: dda.sqrt(p)})
    pic = to_schematic(s)
    assert pic.warn in pic.svg
    assert "sqrt" in pic.svg


def test_empty_state_raises():
    with pytest.raises(ValueError):
        to_schematic(State())


def test_save_writes_file(tmp_path):
    x, y = symbols("x, y")
    s = State({y: dda.neg(x)})
    pic = to_schematic(s)
    target = tmp_path / "out.svg"
    returned = pic.save(str(target))
    assert returned == str(target)
    assert target.read_text() == pic.svg


def test_repr_svg_matches_svg_attribute():
    x, y = symbols("x, y")
    s = State({y: dda.neg(x)})
    pic = to_schematic(s)
    assert pic._repr_svg_() == pic.svg
