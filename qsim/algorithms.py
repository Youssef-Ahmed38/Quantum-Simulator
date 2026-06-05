import numpy as np

from . import gates as G
from .circuit import QuantumCircuit


# Bell pairs and GHZ

def bell_pair(kind="phi+"):
    """one of {phi+, phi-, psi+, psi-}."""
    qc = QuantumCircuit(2)
    if kind in ("phi-", "psi-"):
        qc.x(0)
    if kind in ("psi+", "psi-"):
        qc.x(1)
    qc.h(0).cnot(0, 1)
    return qc


def ghz(num_qubits=3):
    if num_qubits < 2:
        raise ValueError("GHZ needs >= 2 qubits")
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for q in range(1, num_qubits):
        qc.cnot(0, q)
    return qc


# teleportation

def teleport(payload_state, rng=None):
    """teleport a 1-qubit payload from q0 to q2 via Bell pair on (q1, q2).
    returns (circuit, m0, m1, q2_state_after_correction)."""
    if rng is None:
        rng = np.random.default_rng()

    payload = np.asarray(payload_state, dtype=complex)
    if payload.shape != (2,):
        raise ValueError("payload must have shape (2,)")
    payload = payload / np.linalg.norm(payload)

    qc = QuantumCircuit(3)
    # inject payload into q0
    qc.state.vec = np.kron(payload, np.kron([1, 0], [1, 0])).astype(complex)
    # Bell pair on q1, q2
    qc.h(1).cnot(1, 2)
    # Alice's Bell measurement on q0, q1
    qc.cnot(0, 1).h(0)

    m0 = qc.measure(0, rng)
    m1 = qc.measure(1, rng)
    if m1 == 1:
        qc.x(2)
    if m0 == 1:
        qc.z(2)

    # pull out q2's reduced state (q0, q1 already collapsed to |m0 m1>)
    tensor = qc.state.vec.reshape(2, 2, 2)
    q2_vec = tensor[m0, m1, :].copy()
    nrm = np.linalg.norm(q2_vec)
    if nrm > 0:
        q2_vec /= nrm
    return qc, m0, m1, q2_vec


# Deutsch-Jozsa

def deutsch_jozsa(oracle_fn, num_input_qubits):
    """single-query test: is f constant or balanced?
    returns (verdict_str, circuit). circuit has 1 ancilla at index n."""
    n = num_input_qubits
    qc = QuantumCircuit(n + 1)

    # ancilla |1>, then H everywhere
    qc.x(n)
    for q in range(n + 1):
        qc.h(q)

    # U_f: |x>|y> -> |x>|y XOR f(x)>
    U = _phase_oracle_from_fn(oracle_fn, n)
    qc.custom(U, list(range(n + 1)), name="U_f")

    # H on inputs
    for q in range(n):
        qc.h(q)

    # if input bits all 0 -> constant, else balanced
    probs = qc.probabilities()
    p_all_zero_input = 0.0
    for i, p in enumerate(probs):
        bits = format(i, f"0{n+1}b")
        if bits[:n] == "0" * n:
            p_all_zero_input += p
    verdict = "constant" if p_all_zero_input > 0.5 else "balanced"
    return verdict, qc


def _phase_oracle_from_fn(fn, n):
    # U_f on n+1 qubits, qubit 0 is MSB of x, qubit n is ancilla y.
    dim = 2 ** (n + 1)
    U = np.zeros((dim, dim), dtype=complex)
    for x in range(2 ** n):
        fx = fn(x) & 1
        for y in (0, 1):
            in_idx = (x << 1) | y
            out_idx = (x << 1) | (y ^ fx)
            U[out_idx, in_idx] = 1
    return U


# Grover

def grover(marked, num_qubits, iterations=None):
    n = num_qubits
    N = 2 ** n
    M = len(set(marked))
    if M == 0 or M >= N:
        raise ValueError("need 1 <= |marked| <= N-1")

    if iterations is None:
        iterations = max(1, int(np.floor(np.pi / 4 * np.sqrt(N / M))))

    # oracle: flip sign of marked basis states
    oracle = np.eye(N, dtype=complex)
    for m in marked:
        oracle[m, m] = -1

    # diffuser = H^n (2|0><0| - I) H^n. middle = diag(1,-1,...,-1).
    diffuser_inner = -np.eye(N, dtype=complex)
    diffuser_inner[0, 0] = 1
    Hn = G.H
    for _ in range(n - 1):
        Hn = np.kron(Hn, G.H)
    diffuser = Hn @ diffuser_inner @ Hn

    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)

    for _ in range(iterations):
        qc.custom(oracle, list(range(n)), name="Oracle")
        qc.custom(diffuser, list(range(n)), name="Diffuser")
    return qc


# QFT

def qft(num_qubits, inverse=False):
    n = num_qubits
    qc = QuantumCircuit(n)
    _qft_into(qc, list(range(n)), inverse)
    return qc


def _qft_into(qc, qubits, inverse):
    n = len(qubits)
    if not inverse:
        for j in range(n):
            qc.h(qubits[j])
            for k in range(j + 1, n):
                theta = np.pi / (2 ** (k - j))
                qc.cp(theta, qubits[k], qubits[j])
        # bit-reversal swaps
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - 1 - i])
    else:
        for i in range(n // 2):
            qc.swap(qubits[i], qubits[n - 1 - i])
        for j in reversed(range(n)):
            for k in reversed(range(j + 1, n)):
                theta = -np.pi / (2 ** (k - j))
                qc.cp(theta, qubits[k], qubits[j])
            qc.h(qubits[j])


def qft_matrix(num_qubits):
    """exact QFT unitary, useful for testing."""
    N = 2 ** num_qubits
    omega = np.exp(2j * np.pi / N)
    M = np.array([[omega ** (j * k) for k in range(N)] for j in range(N)],
                 dtype=complex) / np.sqrt(N)
    return M
