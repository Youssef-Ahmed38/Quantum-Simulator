import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from qsim import algorithms


def demo(payload, label):
    rng = np.random.default_rng()
    qc, m0, m1, bob = algorithms.teleport(payload, rng)
    fid = abs(np.vdot(payload, bob)) ** 2
    print(f"  payload : {payload}")
    print(f"  m0, m1  : {m0}, {m1}")
    print(f"  bob     : {bob}")
    print(f"  fidelity: {fid:.6f}    ({label})")


def main():
    print("== |0> ==")
    demo(np.array([1, 0], dtype=complex), "trivial")

    print("\n== |1> ==")
    demo(np.array([0, 1], dtype=complex), "trivial")

    print("\n== |+> ==")
    demo(np.array([1, 1], dtype=complex) / np.sqrt(2), "plus state")

    print("\n== 0.6|0> + 0.8i|1> ==")
    p = np.array([0.6, 0.8j], dtype=complex)
    p = p / np.linalg.norm(p)
    demo(p, "complex amplitude")


if __name__ == "__main__":
    main()
