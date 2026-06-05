import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers 3d projection


def plot_histogram(data, title="Measurement outcomes", ax=None):
    if not data:
        raise ValueError("data is empty")
    keys = sorted(data.keys())
    values = [data[k] for k in keys]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, 0.4 * len(keys) + 2), 3.5))
    bars = ax.bar(keys, values, color="#4c72b0", edgecolor="black")
    ax.set_xlabel("basis state")
    ax.set_ylabel("count" if isinstance(values[0], (int, np.integer))
                  else "probability")
    ax.set_title(title)
    if len(keys) > 8:
        ax.tick_params(axis="x", rotation=45)
    for b, v in zip(bars, values):
        txt = f"{v}" if isinstance(v, (int, np.integer)) else f"{v:.3f}"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                txt, ha="center", va="bottom", fontsize=8)
    if fig is not None:
        fig.tight_layout()
    return fig if fig is not None else ax.figure


def bloch_vector(state, qubit=0):
    """(x, y, z) of the reduced single-qubit Bloch vector."""
    n = state.num_qubits
    if not 0 <= qubit < n:
        raise ValueError("qubit out of range")

    # partial trace over the other qubits
    tensor = state.vec.reshape([2] * n)
    t = np.moveaxis(tensor, qubit, 0)
    flat = t.reshape(2, -1)
    rho = flat @ flat.conj().T  # 2x2

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    x = float(np.real(np.trace(rho @ sx)))
    y = float(np.real(np.trace(rho @ sy)))
    z = float(np.real(np.trace(rho @ sz)))
    return x, y, z


def plot_bloch(state, qubit=0, title=None):
    x, y, z = bloch_vector(state, qubit)

    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111, projection="3d")

    # wireframe sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="lightgray", linewidth=0.4)

    # axes
    ax.quiver(0, 0, 0, 1.2, 0, 0, color="k", arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, 1.2, 0, color="k", arrow_length_ratio=0.05)
    ax.quiver(0, 0, 0, 0, 0, 1.2, color="k", arrow_length_ratio=0.05)
    ax.text(1.3, 0, 0, "x")
    ax.text(0, 1.3, 0, "y")
    ax.text(0, 0, 1.3, "|0>")
    ax.text(0, 0, -1.4, "|1>")

    ax.quiver(0, 0, 0, x, y, z, color="#c44e52", linewidth=2,
              arrow_length_ratio=0.12)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.set_title(title or f"qubit {qubit}: ({x:+.2f}, {y:+.2f}, {z:+.2f})")
    fig.tight_layout()
    return fig


def plot_state_city(state, title="State amplitudes"):
    n = state.num_qubits
    dim = 2 ** n
    labels = [format(i, f"0{n}b") for i in range(dim)]
    real = np.real(state.vec)
    imag = np.imag(state.vec)

    fig = plt.figure(figsize=(max(5, 0.35 * dim + 2), 4))
    ax_re = fig.add_subplot(121, projection="3d")
    ax_im = fig.add_subplot(122, projection="3d")
    xs = np.arange(dim)

    for ax, vals, ttl in [(ax_re, real, "Re"), (ax_im, imag, "Im")]:
        ax.bar3d(xs, np.zeros(dim), np.zeros(dim),
                 0.7, 0.7, vals, color="#4c72b0", shade=True)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=60, fontsize=7)
        ax.set_zlim(-1, 1)
        ax.set_title(ttl)
        ax.set_yticks([])

    fig.suptitle(title)
    fig.tight_layout()
    return fig
