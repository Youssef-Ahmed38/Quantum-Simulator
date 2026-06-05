import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qsim import QuantumCircuit


def main():
    print("== |0> ==")
    qc = QuantumCircuit(1)
    print("state:", qc.state.pretty())

    print("\n== H|0> ==")
    qc = QuantumCircuit(1).h(0)
    print("state:        ", qc.state.pretty())
    print("probabilities:", qc.probability_dict())
    print("circuit:")
    print(qc)

    # HZH = X (sanity)
    print("\n== HZH|0> ==")
    qc = QuantumCircuit(1).h(0).z(0).h(0)
    print("state:        ", qc.state.pretty())

    print("\n== 1024 shots of H|0> ==")
    qc = QuantumCircuit(1).h(0)
    print(qc.sample(1024))

    print("\n== T H|0> (probabilities still 50/50, but phase differs) ==")
    qc = QuantumCircuit(1).h(0).t(0)
    print("state:", qc.state.pretty())
    print("probs:", qc.probability_dict())


if __name__ == "__main__":
    main()
