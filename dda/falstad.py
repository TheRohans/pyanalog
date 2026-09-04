#
# Copyright (c) 2026 The Rohans Limited
# Added in a fork of anabrid's pyanalog: https://github.com/anabrid/pyanalog
# Licensed under the same terms as the rest of this repository (see LICENSE.GPL3).
#

"""
This module exports a :class:`dda.State` to the plain-text save format used by
`CircuitJS1 <https://www.falstad.com/circuit/circuitjs.html>`_ (source:
`github.com/pfalstad/circuitjs1 <https://github.com/pfalstad/circuitjs1>`_, GPL) --
an open-source, browser-based circuit simulator. Unlike a static schematic
picture, the output of this module is a *real, simulatable circuit*: paste it
into CircuitJS1 (File > Import From Text, or open the generated ``?cct=`` URL)
and it runs.

Component mapping
------------------

Same primitive vocabulary as the DDA algebra, now realized as real op-amp
stages with real resistor/capacitor values instead of abstract blocks:

* ``sum``, ``neg`` -- an inverting op-amp stage (``Rin`` per input, one
  ``Rf`` feedback resistor, all equal to ``ref_r`` by default).
* ``int(..., dt, ic)`` -- the same stage with the feedback resistor replaced
  by a capacitor, sized so that ``Rin * Cf = dt * tau`` for a chosen
  ``tau`` (real seconds per DDA time-unit -- this is a genuine engineering
  choice, see the ``tau`` parameter). The capacitor's initial voltage is
  set to ``ic``.
* ``mult(k, x)`` with a literal ``k`` -- a dedicated gain stage
  (``Rf = ref_r * |k|``). Since ``mult`` itself is *not* inverting in the
  DDA algebra (only ``sum``/``neg``/``int`` are) but a single op-amp stage
  always is, a positive ``k`` gets a second unity-gain inverter chained on
  to cancel the spurious sign flip; a negative ``k`` needs only the one
  stage. Both cases are algebraically exact, not approximations.
* ``mult(a, b)`` of two live signals -- **CircuitJS1 has no analog
  four-quadrant multiplier element.** This is flagged honestly with a text
  label in the output rather than faked with something that won't behave
  like a real multiplier; see ``self.unsupported``.
* ``const(v)`` -- a fixed DC voltage rail at exactly ``v``.
* a free (undefined) variable -- a DC voltage rail, default ``default_input_v``
  volts, labeled with the variable's name so you know which physical input
  it is. Swap it for whatever real source you want once it's loaded.

Layout is a simple grid placement (16-unit CircuitJS1 grid), columns by
longest-path layer, same feedback-cycle detection as needed for integrator
loops (oscillators etc.) -- correctness of connectivity is what matters here,
not routing aesthetics; CircuitJS1 doesn't care whether wires cross visually,
only whether they share endpoints.

Usage
-----

>>> from dda import State, symbols, dda, export
>>> x, m, b, y = symbols("x, m, b, y")
>>> s = State()
>>> s[y] = dda.neg(dda.sum(dda.mult(m, x), b))
>>> circuit = export(s, to="falstad")   # doctest: +SKIP
>>> circuit.save("y_mx_b.txt")          # doctest: +SKIP
>>> circuit.url()                       # doctest: +SKIP
'https://www.falstad.com/circuit/circuitjs.html?cct=...'
"""

import collections
import urllib.parse
from . import Symbol, State, clean

GRID = 16


def _snap(v):
    return int(round(v / GRID) * GRID)


class to_falstad:
    """
    Export a :class:`dda.State` to a CircuitJS1 plain-text circuit.

    Parameters
    ----------
    state : dda.State
    ref_r : float
        Reference resistor value in ohms (default 10000 = 10k), used for
        every ``Rin``/``Rf`` unless a literal coefficient scales it.
    tau : float
        Real seconds represented by one DDA time-unit, used to turn each
        integrator's ``dt`` into an actual capacitance (``Rin*Cf = dt*tau``).
        This is a genuine choice you're making, not something derived from
        the circuit -- the default (1.0) is arbitrary.
    default_input_v : float
        DC voltage given to free (externally-driven) inputs' rails.
    op_amp_swing : float
        Symmetric output clipping voltage for every op-amp stage (maxOut =
        +this, minOut = -this).
    values : dict
        Current numeric value for any free variable, e.g. ``{"w1": 0.73}``.
        A trainable weight set by a digital pot isn't really "two live
        signals multiplied" -- the pot only changes occasionally (when the
        training loop updates it), not constantly like a real signal -- so
        it's electrically a *coefficient*, the same as writing a literal
        number directly in the DDA code. Naming a variable here treats it
        exactly that way: ``mult(w1, x1)`` with ``values={"w1": 0.73}``
        builds the same resistor-ratio gain stage as ``mult(0.73, x1)``
        would, instead of being flagged as needing a real analog multiplier.
        Substitution applies everywhere that variable appears, not just
        inside ``mult`` -- it's a statement about the variable itself.
    """

    _OPAMP = {"sum", "neg", "int"}

    def __init__(self, state, ref_r=10000.0, tau=1.0, default_input_v=2.0, op_amp_swing=15.0,
                 values=None):
        self.ref_r = ref_r
        self.tau = tau
        self.default_input_v = default_input_v
        self.op_amp_swing = op_amp_swing
        self.values = dict(values) if values else {}

        self.state = clean(state, target="python").name_computing_elements()
        self._build_graph()
        self._layer_nodes()
        self._order_rows()

        self.lines = []
        self.out_port = {}      # name -> (x, y) electrical output coordinate
        self.minus_pin = {}     # name -> (x, y) of the stage's own "-" input pin (feedback target)
        self.unsupported = []   # names with no real CircuitJS1 realization
        self._emit_all()
        self.text = self._render()

    # ---------------------------------------------------------- graph build
    # (same linearized-dependency approach as the schematic exporter: every
    # computing element becomes exactly one node; free variables are inputs)

    def _classify(self, t):
        """
        One tail entry -> ("signal", Symbol) or ("literal", number). A
        variable named in self.values is treated as a literal (its given
        value), not a live signal -- see the `values` constructor parameter.
        """
        if isinstance(t, Symbol) and t.is_variable():
            if t.head in self.values:
                return "literal", self.values[t.head]
            return "signal", t
        return "literal", t

    def _tail_signals(self, head, tail):
        "Returns (signal_symbols, literal_numbers, extra) -- extra is (dt, ic) for int, else None."
        body, extra = tail, None
        if head == "int" and len(tail) >= 2:
            body, dt, ic = tail[:-2], tail[-2], tail[-1]
            extra = (dt, ic)
        sig, lits = [], []
        for t in body:
            kind, val = self._classify(t)
            (sig if kind == "signal" else lits).append(val)
        return sig, lits, extra

    def _build_graph(self):
        self.defs = {}
        self.signals = {}   # name -> [Symbol,...] forward+back signal deps (unfiltered)
        self.literals = {}  # name -> [number,...] bare literal operands (e.g. the -3 in sum(x,-3))
        self.extra = {}     # name -> (dt,ic) for int, else None
        self.free_inputs = set()

        for name in self.state:
            term = self.state[name]
            if not isinstance(term, Symbol):
                term = Symbol("const", term)
            elif term.is_variable():
                term = Symbol("id", term)
            self.defs[name] = term
            sig, lits, extra = self._tail_signals(term.head, term.tail)
            self.signals[name] = sig
            self.literals[name] = lits
            self.extra[name] = extra

        for name, sig in self.signals.items():
            for s in sig:
                if s.head not in self.defs:
                    self.free_inputs.add(s.head)

        if not self.defs:
            raise ValueError("Empty state -- nothing to export.")

    def _fwd_signals(self, name):
        "This node's signal deps, excluding any that are feedback (back-edge) connections."
        return [s for s in self.signals[name] if (s.head, name) not in self.back_edges]

    def _layer_nodes(self):
        WHITE, GRAY, BLACK = 0, 1, 2
        all_names = list(self.free_inputs) + list(self.defs.keys())
        color = {n: WHITE for n in all_names}
        layer = {}
        self.back_edges = set()

        def deps_of(name):
            return [s.head for s in self.signals.get(name, [])]

        def visit(name):
            color[name] = GRAY
            best = -1
            for dep in deps_of(name):
                if color.get(dep, WHITE) == GRAY:
                    self.back_edges.add((dep, name))
                    continue
                if color.get(dep, WHITE) == WHITE:
                    visit(dep)
                best = max(best, layer.get(dep, 0))
            layer[name] = best + 1
            color[name] = BLACK

        for n in all_names:
            if color[n] == WHITE:
                visit(n)
        self.layer = layer

        self.fwd_preds = collections.defaultdict(list)
        for name in all_names:
            for dep in deps_of(name):
                if (dep, name) not in self.back_edges:
                    self.fwd_preds[name].append(dep)

    def _order_rows(self):
        by_layer = collections.defaultdict(list)
        for n, l in self.layer.items():
            by_layer[l].append(n)
        self.row = {}
        for l in sorted(by_layer):
            nodes = by_layer[l]
            if l == 0:
                nodes.sort()
            else:
                def score(n):
                    rows = [self.row[p] for p in self.fwd_preds.get(n, []) if p in self.row]
                    return sum(rows) / len(rows) if rows else 0
                nodes.sort(key=lambda n: (score(n), n))
            for i, n in enumerate(nodes):
                self.row[n] = i
        self.by_layer = by_layer

    def _colrow_xy(self, name):
        # col_w has to be wide enough for the *widest* thing that can land in a single
        # column -- a positive-literal-coefficient mult is two chained op-amp stages,
        # not one, and they must not run into the next layer's column.
        col_w, row_h = 1000, 176
        x = 96 + self.layer[name] * col_w
        y = 96 + self.row[name] * row_h
        return _snap(x), _snap(y)

    # -------------------------------------------------------------- emit

    def _line(self, *fields):
        self.lines.append(" ".join(str(f) for f in fields))

    def _ground(self, x, y):
        self._line("g", x, y, x, y + GRID * 2, 0)

    def _wire(self, x1, y1, x2, y2):
        """
        Orthogonal routing between two posts. A same-row/same-column wire is
        one straight segment. Anything else routes through an *off-grid*
        private row (y1+4) rather than travelling at the literal source row
        for the whole x-distance -- every real coordinate in this file is
        produced by _snap() (a multiple of 16), so y1+4 can never collide
        with another node's row, wire, or bus, no matter how long the run
        is or what it passes under/over. (An earlier version routed at the
        source's exact row, which was fine until two *unrelated* things
        legitimately shared that same row elsewhere in the circuit -- their
        wires would then overlap on the same coordinates for the whole
        shared span, silently merging two different electrical nodes into
        one. This is the same failure mode CircuitJS1 flagged as "path to
        ground with no resistance" when a free input's row happened to
        match another stage's internal row it had to cross under.)
        """
        if x1 == x2 or y1 == y2:
            self._line("w", x1, y1, x2, y2, 0)
            return
        my = y1 + 4
        self._line("w", x1, y1, x1, my, 0)
        self._line("w", x1, my, x2, my, 0)
        self._line("w", x2, my, x2, y2, 0)

    def _label(self, x, y, text):
        self._line("x", x, y, x + len(text) * 8 + 16, y + GRID, 0, 14, text.replace(" ", "\\ "))

    def _local_rail(self, x, y, voltage):
        """
        A small DC rail at an exact voltage, for a bare literal used inline (e.g. the -3
        in sum(x, -3)). The *first* coordinate pair on an 'R' line is the actual electrical
        post -- confirmed against CircuitJS1's own shipped amp-sum.txt, where a resistor
        connects to a rail's first pair, not its second (which is just the drawn stub's
        far end and carries no connection).
        """
        x2 = x - 64
        self._line("R", x, y, x2, y, 0, 0, 40.0, float(voltage), 0.0)
        return x, y

    def _emit_inverting_stage(self, x0, y0, inputs, feedback, ic=0.0, name=None):
        """
        inputs: list of (source_xy, rin_ohms)
        feedback: either a float (Rf ohms) or ("cap", farads)
        Builds an N-input inverting op-amp stage; returns its output (x, y).
        If `name` is given, registers this stage's "-" pin in self.minus_pin[name]
        so a feedback (back-edge) wire can be routed straight into it later.

        CircuitJS1's 'a' line ("a x1 y1 x2 y2 ...") does NOT give the "-"/"+"
        pins as literal coordinates the way it looks like it might. Per
        OpAmpElm's own setPoints(), (x1,y1) and (x2,y2) must share the same y
        (a horizontal element) -- (x1,y1) is only a *centerline anchor*, the
        "-" pin sits exactly `opheight` (16px at normal size) above it, "+"
        exactly 16px below it, and (x2,y2) is the real, literal output post.
        An earlier version of this code invented its own "-"/"+" offsets and
        let (x1,y1)/(x2,y2) differ in y (a "leaning" element); CircuitJS1 was
        silently computing completely different pin positions for it than the
        ones this code was wiring to, which is what the "bad connection" /
        "wire loop" verification failures upstream of this fix turned out to
        be. Verified against CircuitJS1's own shipped amp-sum.txt.
        """
        op_x = x0 + 176
        step = GRID * 2
        n = len(inputs)
        amp_y = _snap(y0 + GRID + n * step)
        minus_y, plus_y = amp_y - GRID, amp_y + GRID
        out_x = op_x + 160

        bus_x = op_x - GRID
        rows = [_snap(minus_y - (n - i) * step) for i in range(n)]

        for (sx, sy), rin, ry in zip([i[0] for i in inputs], [i[1] for i in inputs], rows):
            rx0 = x0
            rx1 = rx0 + 96
            self._wire(sx, sy, rx0, ry)
            self._line("r", rx0, ry, rx1, ry, 0, rin)
            self._wire(rx1, ry, bus_x, ry)

        # Consecutive segments, not one long span -- a wire endpoint landing on
        # another wire's *midpoint* (rather than a shared endpoint) isn't a
        # guaranteed connection, only a visual crossing. This was the source of
        # a real "1 bad connection" case (a 2-input stage's bus, where the
        # first input's connector met the bus segment's middle instead of an
        # endpoint of it).
        bus_ys = sorted(set(rows + [minus_y]))
        for a, b in zip(bus_ys, bus_ys[1:]):
            self._wire(bus_x, a, bus_x, b)
        self._wire(bus_x, minus_y, op_x, minus_y)

        if name is not None:
            self.minus_pin[name] = (op_x, minus_y)

        self._line("a", op_x, amp_y, out_x, amp_y, 0, self.op_amp_swing, -self.op_amp_swing)

        fb_y = (rows[0] if rows else minus_y) - step
        self._wire(out_x, amp_y, out_x, fb_y)
        self._wire(out_x, fb_y, op_x, fb_y)
        self._wire(op_x, fb_y, op_x, minus_y)
        if isinstance(feedback, tuple) and feedback[0] == "cap":
            self._line("c", op_x, fb_y, out_x, fb_y, 0, feedback[1], ic, ic)
        else:
            self._line("r", op_x, fb_y, out_x, fb_y, 0, feedback)

        gx = op_x - GRID * 3
        self._wire(gx, plus_y, op_x, plus_y)
        self._ground(gx, plus_y)

        return out_x, amp_y

    def _emit_all(self):
        # free inputs first: DC rails, one per row in column 0
        for name in sorted(self.free_inputs):
            x, y = 96, 96 + sorted(self.free_inputs).index(name) * 112
            x, y = _snap(x), _snap(y)
            x2 = x - GRID * 3
            self._line("R", x, y, x2, y, 0, 0, 40.0, self.default_input_v, 0.0)
            self._label(x2 - 90, y - 24, name)
            self.out_port[name] = (x, y)

        # process in topological (layer) order, NOT self.defs' raw dict order --
        # name_computing_elements() names the outermost call first (e.g. "y2"
        # before its own dependency "sum_1"), so a plain dict-order walk would
        # try to wire a node before the thing it depends on has been placed.
        for name in sorted(self.defs.keys(), key=lambda n: self.layer[n]):
            term = self.defs[name]
            x0, y0 = self._colrow_xy(name)
            head = term.head
            fwd_sig = self._fwd_signals(name)
            literals = self.literals[name]
            extra = self.extra[name]

            def src(s):
                return self.out_port.get(s.head, (x0 - GRID, y0))

            def literal_inputs(base_y):
                ins = []
                for i, lit in enumerate(literals):
                    rxy = self._local_rail(x0 - 176, base_y + i * 48, lit)
                    ins.append((rxy, self.ref_r))
                return ins

            if head == "id":
                self.out_port[name] = src(term.tail[0])

            elif head == "const":
                x, y = x0, y0
                self._line("R", x, y, x - GRID * 3, y, 0, 0, 40.0, float(literals[0]), 0.0)
                self._label(x - GRID * 3 - 90, y - 24, name)
                self.out_port[name] = (x, y)

            elif head in ("sum", "neg"):
                inputs = [(src(s), self.ref_r) for s in fwd_sig] + literal_inputs(y0 + 260)
                self.out_port[name] = self._emit_inverting_stage(x0, y0, inputs, self.ref_r, name=name)

            elif head == "int":
                dt, ic = extra
                cf = (float(dt) * self.tau) / self.ref_r
                inputs = [(src(s), self.ref_r) for s in fwd_sig] + literal_inputs(y0 + 260)
                self.out_port[name] = self._emit_inverting_stage(
                    x0, y0, inputs, ("cap", cf), ic=float(ic), name=name)

            elif head == "mult" and len(term.tail) >= 1 and len(literals) == len(term.tail):
                # every operand resolved to a plain number (via a literal or a
                # substituted value) -- this is just a constant, not a multiply.
                k = 1.0
                for lit in literals:
                    k *= float(lit)
                x, y = x0, y0
                self._line("R", x, y, x - GRID * 3, y, 0, 0, 40.0, k, 0.0)
                self._label(x - GRID * 3 - 90, y - 24, f"{name} = {k:g}")
                self.out_port[name] = (x, y)

            elif head == "mult" and len(literals) == 1 and len(term.tail) == 2:
                k = float(literals[0])
                if k == 0:
                    x, y = x0, y0
                    self._ground(x, y)
                    self._label(x0, y0 - 24, f"{name} = 0 (tied to ground)")
                    self.out_port[name] = (x, y + GRID * 2)
                else:
                    a_inputs = [(src(s), self.ref_r) for s in fwd_sig]
                    a_out = self._emit_inverting_stage(
                        x0, y0, a_inputs, self.ref_r * abs(k), name=name)
                    if k > 0:
                        x1 = x0 + 480  # within the same (widened) column, well clear of stage A
                        b_out = self._emit_inverting_stage(
                            x1, y0, [(a_out, self.ref_r)], self.ref_r)
                        self.out_port[name] = b_out
                    else:
                        self.out_port[name] = a_out

            elif head == "mult":
                x, y = x0, y0
                self._label(x, y, f"{name}: no analog multiplier in CircuitJS1 for "
                                   f"{name} = {'*'.join(s.head for s in fwd_sig)} -- "
                                   f"wire a VCCS with expression a*b by hand")
                self.unsupported.append(name)
                self.out_port[name] = (x, y)

            else:
                x, y = x0, y0
                self._label(x, y, f"{name}: no canonical circuit for '{head}' -- "
                                   f"needs a hand-designed stage")
                self.unsupported.append(name)
                self.out_port[name] = (x, y)

        # feedback (back-edges): wire straight from the source's output,
        # through its own input resistor, into the *bus* point of the stage
        # it feeds (registered in minus_pin when that stage was built above)
        # -- not directly into the "-" pin itself, because the feedback
        # (Rf/Cf) loop already runs a wire between that exact pin and a
        # point directly above it; landing a second, independent wire path
        # on those same two coordinates creates a zero-resistance loop
        # ("wire loop detected"), even though the intent is a single node.
        # Entering via the bus point one grid step to the left avoids ever
        # duplicating that exact two-point path. The approach row is well
        # below the stage (past its ground symbol) so it can't clip the
        # ground wire either -- CircuitJS1 doesn't care about wires crossing
        # visually elsewhere, only about shared endpoints.
        for src_name, dst_name in self.back_edges:
            sx, sy = self.out_port.get(src_name, (0, 0))
            if dst_name not in self.minus_pin:
                continue
            px, py = self.minus_pin[dst_name]
            bus_x = px - GRID
            rx0, ry = bus_x - 240, py + 176
            self._wire(sx, sy, rx0, ry)
            self._line("r", rx0, ry, rx0 + 96, ry, 0, self.ref_r)
            self._wire(rx0 + 96, ry, bus_x, ry)
            self._wire(bus_x, ry, bus_x, py)

    # ------------------------------------------------------------- output

    def _render(self):
        header = f"$ 1 5.0E-6 10 57 {self.op_amp_swing} 50"
        return "\n".join([header] + self.lines) + "\n"

    def __str__(self):
        return self.text

    def save(self, filename):
        "Write the CircuitJS1 text file. Returns the filename."
        with open(filename, "w") as f:
            f.write(self.text)
        return filename

    def url(self):
        "A shareable https://www.falstad.com/circuit/... URL that opens this exact circuit."
        return "https://www.falstad.com/circuit/circuitjs.html?cct=" + urllib.parse.quote(self.text)
