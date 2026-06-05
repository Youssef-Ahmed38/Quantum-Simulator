import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qsim import algorithms


def main():
    n = 4
    N = 2 ** n
    marked = [11]   # 1011

    print(f"== Searching {N} items for {marked} ==")
    qc = algorithms.grover(marked=marked, num_qubits=n)
    probs = qc.probability_dict(threshold=1e-3)
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
    for bits, p in top:
        flag = " <-- marked" if int(bits, 2) in marked else ""
        print(f"  |{bits}>: {p:.4f}{flag}")

    print("\n2048 shots:")
    counts = qc.sample(2048)
    for bits, c in sorted(counts.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {bits}: {c}")

    print("\n== Two marked ==")
    marked = [3, 12]
    qc = algorithms.grover(marked=marked, num_qubits=n)
    probs = qc.probability_dict(threshold=1e-3)
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
    print(f"marked = {marked}")
    for bits, p in top:
        flag = " <-- marked" if int(bits, 2) in marked else ""
        print(f"  |{bits}>: {p:.4f}{flag}")


if __name__ == "__main__":
    main()
