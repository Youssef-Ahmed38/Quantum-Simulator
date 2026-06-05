import numpy as np
import pytest

from qsim import QuantumState, gates as G


def test_default_is_all_zeros():
    s = QuantumState(3)
    assert s.vec[0] == 1
    assert np.allclose(s.vec[1:], 0)


def test_probabilities_sum_to_one():
    s = QuantumState(3)
    s.apply(G.H, [0]).apply(G.H, [1]).apply(G.H, [2])
    p = s.probabilities()
    assert np.isclose(p.sum(), 1.0)


def test_bell_pair_only_correlated_outcomes():
    s = QuantumState(2)
    s.apply(G.H, [0]).apply(G.CNOT, [0, 1])
    probs = s.probability_dict()
    assert set(probs.keys()) == {"00", "11"}
    assert np.isclose(probs["00"], 0.5)
    assert np.isclose(probs["11"], 0.5)


def test_sample_distribution():
    s = QuantumState(2).apply(G.H, [0]).apply(G.CNOT, [0, 1])
    rng = np.random.default_rng(42)
    counts = s.sample(10000, rng)
    assert set(counts.keys()) <= {"00", "11"}
    p00 = counts.get("00", 0) / 10000
    assert 0.45 < p00 < 0.55


def test_measure_collapses_state():
    rng = np.random.default_rng(0)
    s = QuantumState(2).apply(G.H, [0]).apply(G.CNOT, [0, 1])
    bit0 = s.measure(0, rng)
    p = s.probability_dict()
    # after measuring q0, q1 has to match q0
    assert all(bits[0] == str(bit0) and bits[1] == str(bit0)
               for bits in p.keys())


def test_measure_all_returns_basis_state():
    rng = np.random.default_rng(1)
    s = QuantumState(3).apply(G.H, [0]).apply(G.H, [1]).apply(G.H, [2])
    outcome = s.measure_all(rng)
    assert len(outcome) == 3
    probs = s.probabilities()
    assert np.isclose(probs.max(), 1.0)


def test_fidelity_self_is_one():
    s = QuantumState(2).apply(G.H, [0]).apply(G.CNOT, [0, 1])
    assert np.isclose(s.fidelity(s), 1.0)


def test_fidelity_orthogonal_states():
    s0 = QuantumState(1)
    s1 = QuantumState(1)
    s1.vec = np.array([0, 1], dtype=complex)
    assert np.isclose(s0.fidelity(s1), 0.0)


def test_invalid_qubit_count():
    with pytest.raises(ValueError):
        QuantumState(0)


def test_normalize_after_partial_zero():
    s = QuantumState(1)
    s.vec = np.array([3, 4], dtype=complex)
    s.normalize()
    assert np.isclose(np.linalg.norm(s.vec), 1.0)
