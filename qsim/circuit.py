from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from . import gates as G
from .state import QuantumState


@dataclass
class Op:
    name: str
    qubits: tuple
    params: tuple = ()
    label: str = None   # override for diagram


class QuantumCircuit:
    """fluent circuit builder. every method returns self so calls chain."""

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = QuantumState(num_qubits)
        self.ops = []

    def _apply(self, op, matrix):
        self.state.apply(matrix, op.qubits)
        self.ops.append(op)
        return self

    # single-qubit
    def i(self, q):  return self._apply(Op("I", (q,)), G.I)
    def x(self, q):  return self._apply(Op("X", (q,)), G.X)
    def y(self, q):  return self._apply(Op("Y", (q,)), G.Y)
    def z(self, q):  return self._apply(Op("Z", (q,)), G.Z)
    def h(self, q):  return self._apply(Op("H", (q,)), G.H)
    def s(self, q):  return self._apply(Op("S", (q,)), G.S)
    def sdg(self, q): return self._apply(Op("Sdg", (q,)), G.SDG)
    def t(self, q):  return self._apply(Op("T", (q,)), G.T)
    def tdg(self, q): return self._apply(Op("Tdg", (q,)), G.TDG)

    def rx(self, theta, q):
        return self._apply(
            Op("RX", (q,), (theta,), label=f"Rx({theta:.2f})"), G.rx(theta))

    def ry(self, theta, q):
        return self._apply(
            Op("RY", (q,), (theta,), label=f"Ry({theta:.2f})"), G.ry(theta))

    def rz(self, theta, q):
        return self._apply(
            Op("RZ", (q,), (theta,), label=f"Rz({theta:.2f})"), G.rz(theta))

    def p(self, theta, q):
        return self._apply(
            Op("P", (q,), (theta,), label=f"P({theta:.2f})"), G.phase(theta))

    def u3(self, theta, phi, lam, q):
        return self._apply(
            Op("U3", (q,), (theta, phi, lam),
               label=f"U3({theta:.2f},{phi:.2f},{lam:.2f})"),
            G.u3(theta, phi, lam))

    def custom(self, matrix, qubits, name="U"):
        return self._apply(Op(name, tuple(qubits)), matrix)

    # two- and three-qubit
    def cnot(self, control, target):
        return self._apply(Op("CNOT", (control, target)), G.CNOT)
    cx = cnot

    def cz(self, control, target):
        return self._apply(Op("CZ", (control, target)), G.CZ)

    def swap(self, a, b):
        return self._apply(Op("SWAP", (a, b)), G.SWAP)

    def cu(self, gate, control, target, name="CU"):
        return self._apply(Op(name, (control, target)), G.controlled(gate))

    def crz(self, theta, control, target):
        return self._apply(
            Op("CRZ", (control, target), (theta,), label=f"CRz({theta:.2f})"),
            G.controlled(G.rz(theta)))

    def cp(self, theta, control, target):
        return self._apply(
            Op("CP", (control, target), (theta,), label=f"CP({theta:.2f})"),
            G.controlled(G.phase(theta)))

    def ccx(self, c1, c2, target):
        return self._apply(Op("CCX", (c1, c2, target)), G.TOFFOLI)
    toffoli = ccx

    def cswap(self, control, a, b):
        return self._apply(Op("CSWAP", (control, a, b)), G.FREDKIN)

    # state / sampling
    def reset(self):
        self.state = QuantumState(self.num_qubits)
        return self

    def probabilities(self):
        return self.state.probabilities()

    def probability_dict(self, threshold=1e-12):
        return self.state.probability_dict(threshold)

    def sample(self, shots=1024, rng=None):
        return self.state.sample(shots, rng)

    def measure_all(self, rng=None):
        return self.state.measure_all(rng)

    def measure(self, qubit, rng=None):
        return self.state.measure(qubit, rng)

    def __str__(self):
        return self.draw()

    def __repr__(self):
        return (f"QuantumCircuit(num_qubits={self.num_qubits}, "
                f"ops={len(self.ops)})")

    # ascii-only glyphs so windows cp1252 console doesn't choke
    _WIRE = "-"
    _VBAR = "|"
    _CTRL = "@"
    _TARGET = "X"

    def draw(self):
        n = self.num_qubits
        columns = []
        for op in self.ops:
            col = [self._WIRE] * n
            qs = op.qubits
            label = op.label or op.name

            if op.name in ("CNOT", "CX"):
                col[qs[0]] = self._CTRL
                col[qs[1]] = self._TARGET
                self._draw_vertical(col, qs[0], qs[1])
            elif op.name == "CZ":
                col[qs[0]] = self._CTRL
                col[qs[1]] = self._CTRL
                self._draw_vertical(col, qs[0], qs[1])
            elif op.name == "SWAP":
                col[qs[0]] = "x"
                col[qs[1]] = "x"
                self._draw_vertical(col, qs[0], qs[1])
            elif op.name == "CCX":
                col[qs[0]] = self._CTRL
                col[qs[1]] = self._CTRL
                col[qs[2]] = self._TARGET
                self._draw_vertical(col, min(qs), max(qs))
            elif op.name == "CSWAP":
                col[qs[0]] = self._CTRL
                col[qs[1]] = "x"
                col[qs[2]] = "x"
                self._draw_vertical(col, min(qs), max(qs))
            elif op.name in ("CP", "CRZ", "CU"):
                col[qs[0]] = self._CTRL
                col[qs[1]] = label
                self._draw_vertical(col, qs[0], qs[1])
            elif len(qs) == 1:
                col[qs[0]] = label
            else:
                # generic multi-qubit gate: label on top, bars on rest
                lo, hi = min(qs), max(qs)
                col[lo] = label
                for r in range(lo + 1, hi + 1):
                    if r in qs:
                        col[r] = "#"
                    else:
                        col[r] = self._VBAR
            columns.append(col)

        # pad columns
        rendered_cols = []
        for col in columns:
            width = max(len(c) for c in col)
            new_col = []
            for cell in col:
                if len(cell) < width:
                    extra = width - len(cell)
                    left = extra // 2
                    right = extra - left
                    if cell == self._WIRE:
                        new_col.append(self._WIRE * width)
                    elif cell == self._VBAR:
                        new_col.append(" " * left + self._VBAR + " " * right)
                    elif cell in (self._CTRL, self._TARGET, "x", "#"):
                        new_col.append(self._WIRE * left + cell
                                       + self._WIRE * right)
                    else:
                        new_col.append(" " * left + cell + " " * right)
                else:
                    new_col.append(cell)
            rendered_cols.append(new_col)

        lines = []
        for q in range(n):
            parts = [f"q{q}: "]
            for col in rendered_cols:
                parts.append(self._WIRE)
                parts.append(col[q])
            parts.append(self._WIRE)
            lines.append("".join(parts))
        return "\n".join(lines)

    @staticmethod
    def _draw_vertical(col, top, bot):
        lo, hi = sorted((top, bot))
        for r in range(lo + 1, hi):
            if col[r] == QuantumCircuit._WIRE:
                col[r] = QuantumCircuit._VBAR

    def compose(self, other, qubit_map=None):
        """append other's ops to self, with optional qubit remap."""
        if qubit_map is None:
            qubit_map = list(range(other.num_qubits))
        if len(qubit_map) != other.num_qubits:
            raise ValueError("qubit_map length must equal other.num_qubits")
        if any(q >= self.num_qubits for q in qubit_map):
            raise ValueError("qubit_map target out of range")
        for op in other.ops:
            mapped = tuple(qubit_map[q] for q in op.qubits)
            new_op = Op(op.name, mapped, op.params, op.label)
            self._apply(new_op, _gate_matrix_for(op))
        return self


def _gate_matrix_for(op):
    name = op.name
    if name in G.NAMED_GATES:
        return G.NAMED_GATES[name]
    if name == "RX":  return G.rx(op.params[0])
    if name == "RY":  return G.ry(op.params[0])
    if name == "RZ":  return G.rz(op.params[0])
    if name == "P":   return G.phase(op.params[0])
    if name == "U3":  return G.u3(*op.params)
    if name == "CP":  return G.controlled(G.phase(op.params[0]))
    if name == "CRZ": return G.controlled(G.rz(op.params[0]))
    if name == "Sdg": return G.SDG
    if name == "Tdg": return G.TDG
    raise ValueError(f"don't know how to rebuild gate for op {name!r}")
