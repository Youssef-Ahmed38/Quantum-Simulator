import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qsim import algorithms


def main():
    print("== Bell |Phi+> = (|00> + |11>) / sqrt(2) ==")
    qc = algorithms.bell_pair("phi+")
    print(qc)
    print("amplitudes:", qc.state.pretty())
    print("1024 shots:", qc.sample(1024))

    print("\n== All four Bell states ==")
    for kind in ("phi+", "phi-", "psi+", "psi-"):
        qc = algorithms.bell_pair(kind)
        print(f"{kind:5s} -> {qc.state.pretty()}")

    print("\n== GHZ(4) ==")
    qc = algorithms.ghz(4)
    print(qc)
    print("amplitudes:", qc.state.pretty())
    print("probabilities:", qc.probability_dict())


if __name__ == "__main__":
    main()
