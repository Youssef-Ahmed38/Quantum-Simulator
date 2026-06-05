from __future__ import annotations
import numpy as np


# Single-qubit gate matrices

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
TDG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    em = np.exp(-1j * theta / 2)
    ep = np.exp(1j * theta / 2)
    return np.array([[em, 0], [0, ep]], dtype=complex)


def phase(theta: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


def u3(theta: float, phi: float, lam: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c,                       -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s,     np.exp(1j * (phi + lam)) * c],
    ], dtype=complex)


# ---------------------------------------------------------------------------
# 2- and 3-qubit gate matrices (first qubit is MSB / control)
# ---------------------------------------------------------------------------
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
], dtype=complex)

CZ = np.diag([1, 1, 1, -1]).astype(complex)

SWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=complex)


def controlled(gate: np.ndarray) -> np.ndarray:
    """Controlled-U: first qubit control, second qubit target."""
    g = np.asarray(gate, dtype=complex)
    out = np.eye(4, dtype=complex)
    out[2:, 2:] = g
    return out


def multi_controlled(gate: np.ndarray, num_controls: int) -> np.ndarray:
    """Multi-controlled 1-qubit gate. First num_controls qubits are controls,
    last qubit is target. Gate is applied only when all controls are 1."""
    g = np.asarray(gate, dtype=complex)
    dim = 2 ** (num_controls + 1)
    out = np.eye(dim, dtype=complex)
    out[dim - 2:dim, dim - 2:dim] = g
    return out


# Toffoli (CCX): two controls, one target.
TOFFOLI = multi_controlled(X, 2)

# Fredkin (CSWAP): one control, two-qubit SWAP target.
FREDKIN = np.eye(8, dtype=complex)
FREDKIN[5, 5] = 0
FREDKIN[6, 6] = 0
FREDKIN[5, 6] = 1
FREDKIN[6, 5] = 1


# ---------------------------------------------------------------------------
# General gate application
# ---------------------------------------------------------------------------
def apply_gate(state: np.ndarray, gate: np.ndarray,
               target_qubits, num_qubits: int) -> np.ndarray:
    """Apply a k-qubit `gate` (shape 2**k x 2**k) to `target_qubits` of an
    n-qubit `state` (shape 2**n). Qubit 0 is the most-significant bit.

    Implementation: reshape state to a tensor with one axis per qubit, then
    contract the gate's input axes with the state's target axes using einsum.
    Cost is O(2**n) memory, O(2**(n+k)) flops.
    """
    n = num_qubits
    targets = list(target_qubits)
    k = len(targets)

    if gate.shape != (2 ** k, 2 ** k):
        raise ValueError(
            f"gate shape {gate.shape} does not match {k} targets")
    if any(q < 0 or q >= n for q in targets):
        raise ValueError(f"target qubit out of range for n={n}: {targets}")
    if len(set(targets)) != k:
        raise ValueError(f"target qubits must be distinct: {targets}")

    state_tensor = state.reshape([2] * n)
    gate_tensor = gate.reshape([2] * (2 * k))

    # Distinct integer labels for einsum's interleaved form.
    state_axes = list(range(n))                       # axes 0..n-1
    gate_out_axes = list(range(n, n + k))             # axes n..n+k-1 (new)
    gate_in_axes = list(targets)                      # match state at targets

    output_axes = list(state_axes)
    for i, q in enumerate(targets):
        output_axes[q] = gate_out_axes[i]

    gate_axes = gate_out_axes + gate_in_axes

    result = np.einsum(
        gate_tensor, gate_axes,
        state_tensor, state_axes,
        output_axes,
    )
    return result.reshape(2 ** n)


# ---------------------------------------------------------------------------
# Lookup for the diagram renderer / circuit code
# ---------------------------------------------------------------------------
NAMED_GATES = {
    "I": I, "X": X, "Y": Y, "Z": Z, "H": H,
    "S": S, "SDG": SDG, "T": T, "TDG": TDG,
    "CNOT": CNOT, "CZ": CZ, "SWAP": SWAP,
    "CCX": TOFFOLI, "CSWAP": FREDKIN,
}
