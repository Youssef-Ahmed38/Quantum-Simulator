import numpy as np
import pytest

from qsim import QuantumCircuit, algorithms


# Bell pairs
@pytest.mark.parametrize("kind,expected", [
    ("phi+", {"00": 0.5, "11": 0.5}),
    ("phi-", {"00": 0.5, "11": 0.5}),
    ("psi+", {"01": 0.5, "10": 0.5}),
    ("psi-", {"01": 0.5, "10": 0.5}),
])
def test_bell_pair_outcomes(kind, expected):
    qc = algorithms.bell_pair(kind)
    probs = qc.probability_dict()
    for k, v in expected.items():
        assert np.isclose(probs.get(k, 0), v), (kind, k, probs)


def test_phi_minus_has_negative_relative_phase():
    qc = algorithms.bell_pair("phi-")
    assert np.isclose(qc.state.vec[0], 1 / np.sqrt(2))
    assert np.isclose(qc.state.vec[3], -1 / np.sqrt(2))


# GHZ
def test_ghz_3_qubits():
    qc = algorithms.ghz(3)
    assert set(qc.probability_dict().keys()) == {"000", "111"}


def test_ghz_5_qubits():
    qc = algorithms.ghz(5)
    assert set(qc.probability_dict().keys()) == {"00000", "11111"}


# Deutsch-Jozsa
def test_dj_constant_zero():
    verdict, _ = algorithms.deutsch_jozsa(lambda x: 0, num_input_qubits=4)
    assert verdict == "constant"


def test_dj_constant_one():
    verdict, _ = algorithms.deutsch_jozsa(lambda x: 1, num_input_qubits=4)
    assert verdict == "constant"


def test_dj_balanced_parity():
    verdict, _ = algorithms.deutsch_jozsa(
        lambda x: bin(x).count("1") & 1, num_input_qubits=4)
    assert verdict == "balanced"


def test_dj_balanced_first_bit():
    verdict, _ = algorithms.deutsch_jozsa(
        lambda x: (x >> 3) & 1, num_input_qubits=4)
    assert verdict == "balanced"


# Grover
def test_grover_single_marked_n3():
    qc = algorithms.grover(marked=[5], num_qubits=3)
    probs = qc.probability_dict()
    top = max(probs, key=probs.get)
    assert top == "101"          # 5
    assert probs[top] > 0.9


def test_grover_two_marked_n4():
    qc = algorithms.grover(marked=[3, 12], num_qubits=4)
    probs = qc.probability_dict()
    p3 = probs.get(format(3, "04b"), 0)
    p12 = probs.get(format(12, "04b"), 0)
    assert p3 + p12 > 0.9


# QFT
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_qft_matches_exact(n):
    M = algorithms.qft_matrix(n)
    for i in range(2 ** n):
        v = np.zeros(2 ** n, dtype=complex); v[i] = 1
        qc = QuantumCircuit(n)
        qc.state.vec = v.copy()
        qc.compose(algorithms.qft(n))
        assert np.allclose(qc.state.vec, M @ v), f"QFT mismatch on |{i}>"


@pytest.mark.parametrize("n", [2, 3])
def test_qft_then_inverse_is_identity(n):
    rng = np.random.default_rng(0)
    v = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    v = v / np.linalg.norm(v)
    qc = QuantumCircuit(n)
    qc.state.vec = v.copy()
    qc.compose(algorithms.qft(n))
    qc.compose(algorithms.qft(n, inverse=True))
    assert np.allclose(qc.state.vec, v)


# Teleportation
@pytest.mark.parametrize("payload", [
    np.array([1, 0], dtype=complex),
    np.array([0, 1], dtype=complex),
    np.array([1, 1], dtype=complex) / np.sqrt(2),
    np.array([1, 1j], dtype=complex) / np.sqrt(2),
    np.array([0.6, 0.8], dtype=complex),
])
def test_teleport_preserves_payload(payload):
    rng = np.random.default_rng(0)
    _, m0, m1, out = algorithms.teleport(payload, rng)
    fid = abs(np.vdot(payload, out)) ** 2
    assert np.isclose(fid, 1.0, atol=1e-8), (m0, m1, payload, out)
