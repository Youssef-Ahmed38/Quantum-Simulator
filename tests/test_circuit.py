import numpy as np

from qsim import QuantumCircuit


def test_chain_returns_self():
    qc = QuantumCircuit(2)
    out = qc.h(0).cnot(0, 1).x(1)
    assert out is qc


def test_bell_pair_via_circuit():
    qc = QuantumCircuit(2).h(0).cnot(0, 1)
    probs = qc.probability_dict()
    assert set(probs) == {"00", "11"}


def test_rx_pi_equals_minus_i_X_up_to_phase():
    # Rx(pi)|0> = -i|1>
    qc = QuantumCircuit(1).rx(np.pi, 0)
    assert np.allclose(np.abs(qc.state.vec), [0, 1])


def test_ry_pi_over_2_creates_plus():
    qc = QuantumCircuit(1).ry(np.pi / 2, 0)
    assert np.allclose(qc.state.vec, [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_reset_returns_zero_state():
    qc = QuantumCircuit(2).h(0).cnot(0, 1)
    qc.reset()
    assert qc.state.vec[0] == 1


def test_diagram_includes_qubit_labels():
    qc = QuantumCircuit(3).h(0).cnot(0, 1).x(2)
    diagram = str(qc)
    assert "q0:" in diagram
    assert "q1:" in diagram
    assert "q2:" in diagram
    assert "H" in diagram
    assert "X" in diagram


def test_compose_preserves_unitary_action():
    sub = QuantumCircuit(2).h(0).cnot(0, 1)
    qc = QuantumCircuit(3)
    qc.compose(sub, qubit_map=[0, 2])  # Bell on q0, q2
    probs = qc.probability_dict()
    assert set(probs) == {"000", "101"}


def test_ccx_acts_as_toffoli():
    qc = QuantumCircuit(3).x(0).x(1).ccx(0, 1, 2)
    probs = qc.probability_dict()
    assert "111" in probs and np.isclose(probs["111"], 1.0)


def test_custom_gate_application():
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    qc = QuantumCircuit(1).custom(H, [0], name="myH")
    assert np.allclose(qc.state.vec, [1 / np.sqrt(2), 1 / np.sqrt(2)])
