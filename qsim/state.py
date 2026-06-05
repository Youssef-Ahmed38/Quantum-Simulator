from __future__ import annotations
import numpy as np

from . import gates


# qubit 0 is the MSB. for n qubits, index i = |b0 b1 ... b_{n-1}> with
# b_k = (i >> (n-1-k)) & 1.


class QuantumState:
    def __init__(self, num_qubits, vec=None):
        if num_qubits < 1:
            raise ValueError("num_qubits must be >= 1")
        self.num_qubits = num_qubits
        if vec is None:
            self.vec = np.zeros(2 ** num_qubits, dtype=complex)
            self.vec[0] = 1.0
        else:
            v = np.asarray(vec, dtype=complex)
            if v.shape != (2 ** num_qubits,):
                raise ValueError(
                    f"vec must have shape ({2**num_qubits},), got {v.shape}")
            self.vec = v.copy()

    def copy(self):
        return QuantumState(self.num_qubits, self.vec)

    def normalize(self):
        norm = np.linalg.norm(self.vec)
        if norm == 0:
            raise ValueError("cannot normalize zero vector")
        self.vec = self.vec / norm
        return self

    def apply(self, gate, target_qubits):
        self.vec = gates.apply_gate(self.vec, gate, target_qubits,
                                    self.num_qubits)
        return self

    def probabilities(self):
        return np.abs(self.vec) ** 2

    def probability_dict(self, threshold=1e-12):
        probs = self.probabilities()
        out = {}
        for i, p in enumerate(probs):
            if p > threshold:
                out[format(i, f"0{self.num_qubits}b")] = float(p)
        return out

    def inner(self, other):
        if other.num_qubits != self.num_qubits:
            raise ValueError("qubit count mismatch")
        return complex(np.vdot(self.vec, other.vec))

    def fidelity(self, other):
        return float(np.abs(self.inner(other)) ** 2)

    def sample(self, shots=1, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        probs = self.probabilities()
        probs = probs / probs.sum()  # tiny FP drift
        outcomes = rng.choice(2 ** self.num_qubits, size=shots, p=probs)
        counts = {}
        for idx in outcomes:
            key = format(int(idx), f"0{self.num_qubits}b")
            counts[key] = counts.get(key, 0) + 1
        return counts

    def measure_all(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        probs = self.probabilities()
        probs = probs / probs.sum()
        idx = int(rng.choice(2 ** self.num_qubits, p=probs))
        new_vec = np.zeros_like(self.vec)
        new_vec[idx] = 1.0
        self.vec = new_vec
        return format(idx, f"0{self.num_qubits}b")

    def measure(self, qubit, rng=None):
        if not 0 <= qubit < self.num_qubits:
            raise ValueError("qubit out of range")
        if rng is None:
            rng = np.random.default_rng()

        n = self.num_qubits
        tensor = self.vec.reshape([2] * n)
        slice0 = [slice(None)] * n
        slice0[qubit] = 0
        slice1 = [slice(None)] * n
        slice1[qubit] = 1
        amp0 = tensor[tuple(slice0)]
        amp1 = tensor[tuple(slice1)]
        p0 = float(np.sum(np.abs(amp0) ** 2))
        p1 = float(np.sum(np.abs(amp1) ** 2))
        total = p0 + p1
        p0, p1 = p0 / total, p1 / total

        outcome = 0 if rng.random() < p0 else 1
        new_tensor = np.zeros_like(tensor)
        if outcome == 0:
            new_tensor[tuple(slice0)] = amp0 / np.sqrt(p0)
        else:
            new_tensor[tuple(slice1)] = amp1 / np.sqrt(p1)
        self.vec = new_tensor.reshape(2 ** n)
        return outcome

    def __repr__(self):
        return (f"QuantumState(num_qubits={self.num_qubits}, "
                f"|psi>={self.pretty()})")

    def pretty(self, threshold=1e-9, precision=4):
        parts = []
        for i, amp in enumerate(self.vec):
            if abs(amp) < threshold:
                continue
            label = format(i, f"0{self.num_qubits}b")
            if abs(amp.imag) < threshold:
                coef = f"{amp.real:+.{precision}f}"
            elif abs(amp.real) < threshold:
                coef = f"{amp.imag:+.{precision}f}j"
            else:
                coef = f"({amp.real:+.{precision}f}{amp.imag:+.{precision}f}j)"
            parts.append(f"{coef}|{label}>")
        return " ".join(parts) if parts else "0"
