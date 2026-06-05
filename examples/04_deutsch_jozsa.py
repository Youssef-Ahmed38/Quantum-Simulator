import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qsim import algorithms


def show(name, fn, n):
    verdict, _ = algorithms.deutsch_jozsa(fn, num_input_qubits=n)
    print(f"  {name:32s} -> {verdict}")


def main():
    n = 4
    print(f"== Deutsch-Jozsa, n={n} ==\n")

    show("constant 0",           lambda x: 0,                                 n)
    show("constant 1",           lambda x: 1,                                 n)
    show("first-bit (balanced)", lambda x: (x >> (n - 1)) & 1,                n)
    show("last-bit  (balanced)", lambda x: x & 1,                             n)
    show("parity    (balanced)", lambda x: bin(x).count("1") & 1,             n)
    show("x >= N/2  (balanced)", lambda x: 1 if x >= (1 << (n - 1)) else 0,   n)


if __name__ == "__main__":
    main()
