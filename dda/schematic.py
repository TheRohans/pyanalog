#
# Copyright (c) 2026 The Rohans Limited
# Added in a fork of anabrid's pyanalog: https://github.com/anabrid/pyanalog
# Licensed under the same terms as the rest of this repository (see LICENSE.GPL3).
#

"""
This module renders a :class:`dda.State` as an *analog computer block
schematic*: an SVG picture made of the same circuit primitives an actual
analog computer has (summing/inverting amplifiers, integrators, coefficient
potentiometers, multiplier ICs), wired left-to-right by data dependency.

It complements the two drawing routines already in :mod:`ast`:

* :meth:`ast.Symbol.draw_graph` draws the abstract expression *tree* of a
  single symbol (good for checking algebra, not for wiring a breadboard).
* :meth:`ast.State.draw_dependency_graph` draws the variable dependency
  graph with generic Graphviz nodes.

Neither produces something you could hand to a breadboard: they draw the
*algebra*, not the *hardware*. :class:`to_schematic` closes that gap by
mapping each linearized computing element (see
:meth:`ast.State.name_computing_elements`) onto the canonical op-amp circuit
that implements it.

Design choice -- one block per primitive
-----------------------------------------

The renderer keeps a strict invariant: **every computing element in the
linearized state becomes exactly one schematic block.** It does *not* try to
fold a ``mult(k, x)`` feeding a ``sum(...)`` into a single weighted input
resistor the way a human circuit designer would -- that is a manual
optimization left to you. The payoff is that the picture stays a faithful,
traceable 1:1 rendering of the DDA code that produced it, and the algorithm
stays simple enough to work on arbitrary circuits, not just hand-picked ones.

Primitives are drawn as:

* ``sum``, ``neg``, ``int`` -- an op-amp triangle (these are exactly the
  DDA primitives that are physically inverting). ``int`` gets an extra
  ``dt``/``ic`` caption.
* ``mult`` -- a circle. If one operand is a literal number, it is drawn as
  a *coefficient* (a potentiometer/fixed-gain stage feeding a real,
  size-able resistor ratio); if both operands are signals, it is drawn as
  an analog multiplier IC (e.g. an AD633), since that is genuinely
  different hardware.
* ``const`` -- a small reference/terminal node.
* anything else (``div``, ``sqrt``, ``min``, ``max``, the comparators,
  ...) -- a dashed, amber "generic function" box. These *don't* have a
  single canonical op-amp circuit, so the picture honestly flags them as
  needing a hand-designed shaping/diode network instead of inventing one.

Feedback (an ``int`` closing a loop back on itself or through other
elements, e.g. an oscillator) is detected and drawn as an arcing wire over
the top of the block it feeds, the same way a feedback resistor/capacitor
is normally drawn.

Layout is a simple, from-scratch layered graph drawing (longest-path
layering + a single barycenter ordering pass) -- good enough to keep small
and medium circuits (a handful to a few dozen elements) readable, but it is
not a full schematic-capture-quality auto-router. Big circuits will still
benefit from manual tidying.

Usage
-----

>>> from dda import State, symbols, dda, export
>>> x, m, b, y = symbols("x, m, b, y")
>>> s = State()
>>> s[y] = dda.neg(dda.sum(dda.mult(m, x), b))
>>> pic = export(s, to="schematic")   # doctest: +SKIP
>>> pic.save("y_mx_b.svg")            # doctest: +SKIP
>>> pic                               # doctest: +SKIP
... # in Jupyter, this renders inline via _repr_svg_
"""

import collections
from . import Symbol, State, clean

is_number = lambda v: isinstance(v, (int, float))


class to_schematic:
    """
    Render a :class:`dda.State` as an SVG analog-computer block schematic.

    Parameters
    ----------
    state : dda.State
        The circuit to draw.
    hspace, vspace : int
        Horizontal spacing between layers / vertical spacing between rows,
        in SVG units (pixels).
    node_w, node_h : int
        Nominal block size.
    background, ink, accent, warn : str
        Colors. ``ink`` is used for wires/standard block outlines,
        ``accent`` marks adjustable/standard components (coefficients,
        multiplier ICs, final outputs), ``warn`` flags generic/unmapped
        primitives that have no canonical op-amp circuit.
    """

    #: primitives that are physically inverting op-amp stages
    _OPAMP = {"sum", "neg", "int"}

    def __init__(self, state, hspace=190, vspace=100, node_w=120, node_h=56,
                 background="#ffffff", ink="#1b2733", accent="#2f6fed", warn="#c0621a",
                 margin=50):
        self.hspace, self.vspace = hspace, vspace
        self.node_w, self.node_h = node_w, node_h
        self.background, self.ink, self.accent, self.warn = background, ink, accent, warn
        self.margin = margin

        self.state = clean(state, target="python").name_computing_elements()
        self._build_graph()
        self._layer_nodes()
        self._order_rows()
        self.svg = self._render()

    # ---------------------------------------------------------- graph build

    def _tail_signals(self, head, tail):
        """
        Split a computing element's tail into (signal inputs, caption).
        ``int(...)`` treats its last two (numeric) tail entries as dt/ic,
        everything else keeps only the Symbol-variable entries as signals
        and turns any literal numbers into a short caption string.
        """
        if head == "int" and len(tail) >= 2:
            body, dt, ic = tail[:-2], tail[-2], tail[-1]
            sig = [t for t in body if isinstance(t, Symbol) and t.is_variable()]
            return sig, f"dt={dt}  ic={ic}"
        sig = [t for t in tail if isinstance(t, Symbol) and t.is_variable()]
        lits = [t for t in tail if not (isinstance(t, Symbol) and t.is_variable())]
        if not lits:
            return sig, ""
        prefix = "×" if head == "mult" else ""
        return sig, "  ".join(f"{prefix}{l}" for l in lits)

    def _build_graph(self):
        self.defs = {}       # name -> Symbol term (one primitive call)
        self.signals = {}    # name -> [Symbol,...] forward signal inputs
        self.captions = {}   # name -> literal-value caption string
        self.free_inputs = set()

        for name in self.state:
            term = self.state[name]
            if not isinstance(term, Symbol):
                term = Symbol("const", term)
            elif term.is_variable():
                term = Symbol("id", term)
            self.defs[name] = term
            sig, caption = self._tail_signals(term.head, term.tail)
            self.signals[name] = sig
            self.captions[name] = caption

        for name, sig in self.signals.items():
            for s in sig:
                if s.head not in self.defs:
                    self.free_inputs.add(s.head)

        if not self.defs:
            raise ValueError("Empty state -- nothing to draw.")

    # -------------------------------------------------------------- layout

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

        consumed = {dep for preds in self.fwd_preds.values() for dep in preds}
        consumed |= {dep for dep, _ in self.back_edges}
        self.final_outputs = [n for n in self.defs if n not in consumed]

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

    # ---------------------------------------------------------------- geo

    def _pos(self, name):
        "Top-left corner of a node's box, in SVG coordinates."
        x = self.margin + self.layer[name] * self.hspace
        y = self.margin + self.row[name] * self.vspace
        return x, y

    def _port_y(self, y0, index, count):
        if count <= 0:
            return y0 + self.node_h / 2
        return y0 + (index + 1) * self.node_h / (count + 1)

    def _kind(self, head):
        if head in self._OPAMP:
            return head
        if head == "mult":
            return "mult"
        if head == "const":
            return "const"
        if head == "id":
            return "id"
        return "generic"

    # ------------------------------------------------------------- render

    def _render(self):
        w, esc = [], _xml_escape

        max_layer = max(self.layer.values()) if self.layer else 0
        max_row = max((len(v) for v in self.by_layer.values()), default=1)
        width = self.margin * 2 + (max_layer + 1) * self.hspace + self.node_w
        height = self.margin * 2 + max(max_row, 1) * self.vspace + self.node_h + 40

        w.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
                  f'role="img" aria-label="Analog computer block schematic">')
        w.append(f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="{self.background}"/>')
        w.append(f'<defs><marker id="dda-arrow" viewBox="0 0 10 10" refX="8" refY="5" '
                  f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                  f'<path d="M0,0 L10,5 L0,10 Z" fill="{self.ink}"/></marker></defs>')
        w.append(f'<g font-family="Menlo, Consolas, monospace" font-size="12.5" fill="{self.ink}">')

        # --- free-input terminals ---
        term_x = {}
        for name in sorted(self.free_inputs):
            x, y = self._pos(name)
            cy = y + self.node_h / 2
            cx = x + self.node_w - 14
            w.append(f'<text x="{x:.0f}" y="{cy+4:.0f}" text-anchor="start">{esc(name)}</text>')
            w.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="3.5" fill="none" stroke="{self.ink}" stroke-width="1.8"/>')
            term_x[name] = (cx + 3.5, cy)

        # --- output port lookup, filled after we know each node's box ---
        out_port = dict(term_x)
        in_ports = {}   # name -> {signal_head: (x,y)}
        box = {}        # name -> (x,y,w,h)

        for name, term in self.defs.items():
            x, y = self._pos(name)
            box[name] = (x, y, self.node_w, self.node_h)
            kind = self._kind(term.head)
            sig = self.signals[name]
            ports = {}
            for i, s in enumerate(sig):
                py = self._port_y(y, i, len(sig))
                ports[s.head] = (x, py)
            in_ports[name] = ports
            out_port[name] = (x + self.node_w, y + self.node_h / 2)

        # --- draw wires first (so blocks sit on top) ---
        for name, ports in in_ports.items():
            for src_head, (px, py) in ports.items():
                sx, sy = out_port.get(src_head, (px - 20, py))
                w.append(self._wire(sx, sy, px, py))

        for src, dst in self.back_edges:
            sx, sy = out_port.get(src, (0, 0))
            # feedback re-enters from the top of the destination block
            dx, dy0, dw, dh = box.get(dst, (sx, sy, self.node_w, self.node_h))
            tx = dx + dw * 0.25
            top = min(sy, dy0) - 34
            w.append(f'<path d="M{sx:.0f},{sy:.0f} V{top:.0f} H{tx:.0f} V{dy0:.0f}" '
                      f'fill="none" stroke="{self.accent}" stroke-width="1.8" stroke-dasharray="1,0"/>')
            w.append(f'<circle cx="{tx:.0f}" cy="{dy0:.0f}" r="3" fill="{self.accent}"/>')

        # --- draw blocks ---
        for name, term in self.defs.items():
            x, y, bw, bh = box[name]
            kind = self._kind(term.head)
            caption = self.captions.get(name, "")
            w.append(self._draw_block(x, y, bw, bh, name, term.head, kind, caption, len(self.signals[name])))

        # --- final outputs get an arrow + label past their box ---
        for name in self.final_outputs:
            sx, sy = out_port[name]
            ex = sx + 46
            w.append(f'<path d="M{sx:.0f},{sy:.0f} H{ex:.0f}" fill="none" stroke="{self.accent}" '
                      f'stroke-width="2.2" marker-end="url(#dda-arrow)"/>')
            w.append(f'<text x="{ex+8:.0f}" y="{sy+4:.0f}" fill="{self.accent}" '
                      f'font-weight="600">{esc(name)}</text>')

        w.append("</g></svg>")
        return "\n".join(w)

    def _wire(self, sx, sy, px, py):
        """
        Orthogonal routing. A same-row wire is a straight line. Anything else
        is routed through the *gutter* -- the empty gap between two node rows
        (``vspace`` is kept larger than ``node_h`` precisely so this gap
        exists) -- rather than straight across at the source's row height,
        which would otherwise cut straight through any block sitting between
        source and target in an intermediate column.
        """
        if abs(sy - py) < 0.5:
            return f'<path d="M{sx:.0f},{sy:.0f} H{px:.0f}" fill="none" stroke="{self.ink}" stroke-width="1.8"/>'
        gutter = sy + (self.vspace / 2 if py > sy else -self.vspace / 2)
        stub = min(24, max(6, abs(px - sx) / 3))
        d = (f"M{sx:.0f},{sy:.0f} H{sx+stub:.0f} V{gutter:.0f} "
             f"H{px-stub:.0f} V{py:.0f} H{px:.0f}")
        return f'<path d="{d}" fill="none" stroke="{self.ink}" stroke-width="1.8"/>'

    def _draw_block(self, x, y, bw, bh, name, head, kind, caption, n_in):
        w = []
        cx, cy = x + bw / 2, y + bh / 2

        if kind in self._OPAMP:
            glyph = {"sum": "Σ", "neg": "−1", "int": "∫"}[head]
            w.append(f'<polygon points="{x:.0f},{y:.0f} {x:.0f},{y+bh:.0f} {x+bw:.0f},{cy:.0f}" '
                      f'fill="none" stroke="{self.accent if head=="int" else self.ink}" stroke-width="2.2"/>')
            w.append(f'<text x="{x+bw*0.32:.0f}" y="{cy+6:.0f}" font-size="17" text-anchor="middle">{glyph}</text>')
            w.append(f'<text x="{cx:.0f}" y="{y+bh+16:.0f}" text-anchor="middle" fill="#6b7686">{_xml_escape(name)}</text>')
            if caption:
                w.append(f'<text x="{cx:.0f}" y="{y-8:.0f}" text-anchor="middle" fill="#6b7686" font-size="11">{_xml_escape(caption)}</text>')

        elif kind == "mult":
            r = min(bw, bh) / 2 - 6
            is_coeff = n_in == 1 and caption
            color = self.accent
            w.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="{r:.0f}" fill="none" stroke="{color}" stroke-width="2.2"/>')
            w.append(f'<text x="{cx:.0f}" y="{cy+5:.0f}" text-anchor="middle" font-size="15" fill="{color}">×</text>')
            tag = caption if is_coeff else "IC"
            w.append(f'<text x="{cx:.0f}" y="{y+bh+16:.0f}" text-anchor="middle" fill="{color}">{_xml_escape(tag)}</text>')
            w.append(f'<text x="{cx:.0f}" y="{y-8:.0f}" text-anchor="middle" fill="#6b7686" font-size="11">{_xml_escape(name)}</text>')

        elif kind == "const":
            w.append(f'<circle cx="{x+16:.0f}" cy="{cy:.0f}" r="6" fill="{self.ink}"/>')
            val = caption or "const"
            w.append(f'<text x="{x+30:.0f}" y="{cy+4:.0f}" text-anchor="start">{_xml_escape(val)}</text>')

        elif kind == "id":
            w.append(f'<path d="M{x:.0f},{cy:.0f} H{x+bw:.0f}" stroke="{self.ink}" stroke-width="1.8"/>')
            w.append(f'<text x="{cx:.0f}" y="{y-8:.0f}" text-anchor="middle" fill="#6b7686" font-size="11">{_xml_escape(name)}</text>')

        else:  # generic / unmapped primitive -- flagged, not invented
            w.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw:.0f}" height="{bh:.0f}" rx="6" '
                      f'fill="none" stroke="{self.warn}" stroke-width="2" stroke-dasharray="5,4"/>')
            w.append(f'<text x="{cx:.0f}" y="{cy+4:.0f}" text-anchor="middle" fill="{self.warn}">{_xml_escape(head)}</text>')
            w.append(f'<text x="{cx:.0f}" y="{y+bh+16:.0f}" text-anchor="middle" fill="#6b7686">{_xml_escape(name)}</text>')
            if caption:
                w.append(f'<text x="{cx:.0f}" y="{y-8:.0f}" text-anchor="middle" fill="#6b7686" font-size="11">{_xml_escape(caption)}</text>')

        return "\n".join(w)

    # ------------------------------------------------------------- output

    def _repr_svg_(self):
        "IPython/Jupyter hook: display this object and get the schematic inline."
        return self.svg

    def __str__(self):
        return self.svg

    def save(self, filename):
        "Write the SVG to a file. Returns the filename for convenience."
        with open(filename, "w") as f:
            f.write(self.svg)
        return filename


def _xml_escape(s):
    s = str(s)
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
