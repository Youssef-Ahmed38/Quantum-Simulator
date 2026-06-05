import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from qsim import QuantumCircuit, algorithms


def main():
    n = 3
    print(f"== QFT(n={n}) on every basis state ==")
    M = algorithms.qft_matrix(n)
    for k in range(2 ** n):
        v = np.zeros(2 ** n, dtype=complex)
        v[k] = 1
        qc = QuantumCircuit(n)
        qc.state.vec = v.copy()
        qc.compose(algorithms.qft(n))
        ok = np.allclose(qc.state.vec, M @ v)
        print(f"  |{k}> ({format(k, f'0{n}b')}): match={ok}")

    print("\n== QFT diagram ==")
    print(algorithms.qft(n))

    print("\n== QFT then iQFT on a random state ==")
    rng = np.random.default_rng(7)
    v = rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)
    v = v / np.linalg.norm(v)
    qc = QuantumCircuit(n)
    qc.state.vec = v.copy()
    qc.compose(algorithms.qft(n))
    qc.compose(algorithms.qft(n, inverse=True))
    print("returns original?", np.allclose(qc.state.vec, v))


if __name__ == "__main__":
    main()
