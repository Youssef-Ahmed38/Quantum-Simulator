import numpy as np
import pytest

from qsim import gates as G


ALL_1Q = [G.I, G.X, G.Y, G.Z, G.H, G.S, G.SDG, G.T, G.TDG]
ALL_2Q = [G.CNOT, G.CZ, G.SWAP]


def is_unitary(U, atol=1e-10):
    n = U.shape[0]
    return np.allclose(U.conj().T @ U, np.eye(n), atol=atol)


@pytest.mark.parametrize("U", ALL_1Q + ALL_2Q + [G.TOFFOLI, G.FREDKIN])
def test_unitary(U):
    assert is_unitary(U)


@pytest.mark.parametrize("theta", [0.0, 0.1, np.pi / 2, np.pi, 2 * np.pi])
def test_rotation_unitary(theta):
    assert is_unitary(G.rx(theta))
    assert is_unitary(G.ry(theta))
    assert is_unitary(G.rz(theta))
    assert is_unitary(G.phase(theta))
    assert is_unitary(G.u3(theta, 0.7, -0.3))


def test_pauli_squares_to_identity():
    for P in (G.X, G.Y, G.Z, G.H):
        assert np.allclose(P @ P, np.eye(2))


def test_S_dagger_inverts_S():
    assert np.allclose(G.S @ G.SDG, np.eye(2))


def test_T_squared_is_S():
    assert np.allclose(G.T @ G.T, G.S)


def test_HXH_equals_Z():
    assert np.allclose(G.H @ G.X @ G.H, G.Z)


def test_apply_X_on_q0():
    # |00> -> |10>
    v = np.array([1, 0, 0, 0], dtype=complex)
    out = G.apply_gate(v, G.X, [0], 2)
    assert np.allclose(out, [0, 0, 1, 0])


def test_apply_X_on_q1():
    # |00> -> |01>
    v = np.array([1, 0, 0, 0], dtype=complex)
    out = G.apply_gate(v, G.X, [1], 2)
    assert np.allclose(out, [0, 1, 0, 0])


def test_cnot_flips_target_when_control_is_one():
    v = np.array([0, 0, 1, 0], dtype=complex)   # |10>
    out = G.apply_gate(v, G.CNOT, [0, 1], 2)
    assert np.allclose(out, [0, 0, 0, 1])       # |11>


def test_cnot_does_not_flip_when_control_is_zero():
    v = np.array([0, 1, 0, 0], dtype=complex)   # |01>
    out = G.apply_gate(v, G.CNOT, [0, 1], 2)
    assert np.allclose(out, [0, 1, 0, 0])


def test_toffoli_flips_only_when_both_controls_one():
    v = np.zeros(8, dtype=complex); v[6] = 1     # |110>
    out = G.apply_gate(v, G.TOFFOLI, [0, 1, 2], 3)
    assert np.allclose(out, np.eye(8)[7])        # |111>

    v = np.zeros(8, dtype=complex); v[3] = 1     # |011>
    out = G.apply_gate(v, G.TOFFOLI, [0, 1, 2], 3)
    assert np.allclose(out, np.eye(8)[3])


def test_swap_swaps_qubits():
    v = np.array([0, 0, 1, 0], dtype=complex)   # |10>
    out = G.apply_gate(v, G.SWAP, [0, 1], 2)
    assert np.allclose(out, [0, 1, 0, 0])       # |01>


def test_apply_preserves_norm():
    rng = np.random.default_rng(0)
    n = 4
    v = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    v = v / np.linalg.norm(v)
    out = G.apply_gate(v, G.H, [2], n)
    assert np.isclose(np.linalg.norm(out), 1.0)
    out = G.apply_gate(out, G.CNOT, [1, 3], n)
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_apply_non_adjacent_qubits():
    # CNOT control q0, target q3 in a 4-qubit system. |1000> -> |1001>.
    v = np.zeros(16, dtype=complex); v[8] = 1.0
    out = G.apply_gate(v, G.CNOT, [0, 3], 4)
    expected = np.zeros(16, dtype=complex); expected[9] = 1.0
    assert np.allclose(out, expected)


def test_apply_reversed_targets():
    # target_qubits=[1, 0] means q1 is control, q0 is target.
    # |01>: q0=0, q1=1 -> ctrl=1 flips q0 -> |11>.
    v = np.array([0, 1, 0, 0], dtype=complex)
    out = G.apply_gate(v, G.CNOT, [1, 0], 2)
    assert np.allclose(out, [0, 0, 0, 1])
