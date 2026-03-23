---
title: Applied Quantum Computing
category: "QUANTUM COMPUTING"
semester: 2023 S
---

# 1. Fundamentals

## Postulates of quantum mechanics

### 1) State space

$$|\psi\rangle \in \mathcal{H}, \qquad \lVert \psi \rVert = 1$$

- A quantum system is described by a **normalized** vector in a complex Hilbert space.
- For one qubit,

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle,\qquad |\alpha|^2+|\beta|^2=1$$

### 2) Observables

$$A = A^\dagger$$

- Physical observables are **Hermitian** operators.
- Measurement outcomes are **eigenvalues**.

### 3) Measurement

$$\mathbb{P}(a_i)=|\langle a_i|\psi\rangle|^2$$

- The Born rule turns amplitudes into probabilities.
- After measurement, the state **collapses** into the observed eigenspace.

### 4) Time evolution

$$|\psi(t)\rangle = U(t)|\psi(0)\rangle,\qquad U^\dagger U = I$$

- Closed systems evolve **unitarily**.
- Quantum gates are finite-dimensional unitaries.

### 5) Composite systems

$$\mathcal{H}_{AB}=\mathcal{H}_A\otimes \mathcal{H}_B$$

- Multi-qubit systems use **tensor products**.
- **Entanglement** means the joint state is not a product of subsystem states.

### Canonical entangled state

$$|\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt{2}}$$

- The step from classical correlated bits to **non-separable** quantum states.

## Gate-based quantum computing

Single-qubit gates:

$$X=\begin{bmatrix}0&1\\1&0\end{bmatrix},\quad
Z=\begin{bmatrix}1&0\\0&-1\end{bmatrix},\quad
H=\frac{1}{\sqrt{2}}\begin{bmatrix}1&1\\1&-1\end{bmatrix}$$

Two-qubit entangling gate:

$$\mathrm{CNOT}\,|a,b\rangle = |a, a\oplus b\rangle$$

- Circuits are sequences of unitaries and measurements.
- **Universality**: arbitrary one-qubit gates + one entangling two-qubit gate.

## Quantum errors and error correction

- Hardware noise: bit-flip, phase-flip, decoherence, gate/readout error.

Encoding intuition:

$$\alpha|0\rangle+\beta|1\rangle
\;\mapsto\;
\alpha|000\rangle+\beta|111\rangle$$

- Logical information is spread over many physical qubits.
- **Syndromes** detect errors without directly measuring the logical qubit.

## Adiabatic quantum computing

$$H(s)=(1-s)H_0+sH_1,\qquad s\in[0,1]$$

- Start in the ground state of a simple $H_0$; sweep slowly to $H_1$.
- If the sweep is slow enough, the system **tracks** the instantaneous ground state.

---

# 2. Hardware

## Mapping postulates to devices

A platform must implement:

- state preparation  
- coherent control  
- measurement  
- multi-qubit coupling  
- noise mitigation and scaling  

## Superconducting qubits

- Nonlinear superconducting circuits; microwave pulses for gates.  
- **Pros**: fast gates, fabrication, ecosystem.  
- **Cons**: cryogenics, crosstalk, calibration drift, coherence limits.

## Atomic and ionic platforms

- Internal atomic/ionic levels; laser control and readout.  
- **Pros**: long coherence, high fidelity.  
- **Cons**: optical complexity, slower gates, scaling cost.

## Spin-based platforms

- Electron or nuclear spin in dots, donors, or defects.  
- **Pros**: semiconductor compatibility, small footprint.  
- **Cons**: readout difficulty, variability, control precision.

## Bottlenecks (all platforms)

- coherence, gate fidelity, connectivity, control noise  
- error-correction overhead  
- scaling toward **fault-tolerant** systems  

---

# 3. Algorithms and software

## Quantum Fourier transform and search

### QFT

For $N=2^n$,

$$|x\rangle \mapsto \frac{1}{\sqrt{N}}\sum_{k=0}^{N-1} e^{2\pi i xk/N}|k\rangle$$

- Structured **interference**; core of phase estimation and period finding.

### Search

- Grover: $O(N)\to O(\sqrt{N})$ queries via **amplitude amplification**.

## Hybrid quantum–classical algorithms

$$\theta^*=\arg\min_\theta \langle \psi(\theta)|H|\psi(\theta)\rangle$$

- Parameterized circuits + **classical** optimizers on measurement data.  
- Examples: **VQE**, **QAOA**, variational classifiers.

## Quantum annealing and optimization

$$E(z)=\sum_i h_i z_i+\sum_{i<j}J_{ij}z_i z_j,\qquad z_i\in\{-1,1\}$$

- Encode problems as **energy** landscapes; seek low-energy states.

## Quantum chemistry

$$E_0 \approx \min_\theta \langle \psi(\theta)|H_{\text{mol}}|\psi(\theta)\rangle$$

- Natural fit for **VQE**-style ground-state estimation.

## Quantum machine learning

- Feature maps, variational models, quantum kernels, hybrid training.

$$\hat y = f\!\left(\langle O\rangle_{\psi(x,\theta)}\right)$$

- Question: does quantum structure yield **useful** advantage, not just runnable models?

---

# 4. Circuit and state visualization

Visualizations below are meant to **connect symbols to geometry**: time flows left-to-right on circuits, amplitudes and probabilities for entanglement, and the Bloch sphere for where a single-qubit state lives under unitaries.

## Do you need Qiskit here?

**No for in-browser Run.** This site executes Python in the browser (Pyodide). **Qiskit** is heavy and not bundled, so every plot below is built with **NumPy + Matplotlib** (+ NetworkX for chip graphs). The diagrams follow the **same gate semantics** you would draw with `QuantumCircuit.draw("mpl")` locally—install Qiskit on your machine when you want transpiler integration, device backends, or publication exports from the same circuit object.

## Bell-state preparation circuit

```python {run}
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle

plt.rcParams.update({"font.family": "sans-serif"})

fig, ax = plt.subplots(figsize=(11.2, 4.0), facecolor="#080a0c")
fig.patch.set_facecolor("#080a0c")
ax.set_facecolor("#080a0c")

# Vignette + subtle grid
yy, xx = np.mgrid[0:1:200j, 0:1:300j]
r = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
ax.imshow(1 - np.clip(r * 1.35, 0, 1), extent=[0, 1, 0, 1], aspect="auto", cmap="gray", alpha=0.35, zorder=0)

for tx in np.linspace(0.12, 0.9, 9):
    ax.axvline(tx, color="#1a2535", lw=0.6, alpha=0.5, zorder=0)
ax.text(0.5, 0.96, "time →", ha="center", va="top", fontsize=9, color="#5a6570", style="italic")

y0, y1 = 0.72, 0.28
for y, lab in [(y0, r"$q_0$"), (y1, r"$q_1$")]:
    for off, al in [(0.004, 0.12), (0, 0.85)]:
        ax.plot([0.06, 0.93], [y - off, y - off], color="#2a3848", lw=3.2, solid_capstyle="round", zorder=1, alpha=al)
    ax.text(0.012, y, lab, fontsize=15, color="#e8c97a", va="center", fontweight="bold")

def gate_shadow(x, y, w, h):
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.008, y - 0.012), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.07",
            facecolor="#000000",
            alpha=0.45,
            zorder=2,
        )
    )

def gate_box(x, y, w, h, text, fc="#1c2a3a", ec="#7eb8ff"):
    gate_shadow(x, y, w, h)
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.07",
            facecolor=fc,
            edgecolor=ec,
            linewidth=2.2,
            zorder=4,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=14, color="#f8fafc", fontweight="bold")

gate_box(0.15, y0 - 0.1, 0.12, 0.2, "H", fc="#243548", ec="#6b9ac4")
ax.plot([0.27, 0.38], [y0, y0], color="#8a96a8", lw=2.4, zorder=3)

# Control + target (CNOT)
cx, cy = 0.42, y0
ax.add_patch(Circle((cx, cy), 0.034, facecolor="#c9a961", edgecolor="#fff0cc", linewidth=2, zorder=6))
ax.plot([cx, cx], [y0, y1], color="#c9a961", lw=2.6, zorder=5)
r_tgt = 0.055
circ = plt.Circle((cx, y1), r_tgt, fill=False, edgecolor="#c9a961", linewidth=2.8, zorder=6)
ax.add_patch(circ)
ax.plot([cx - r_tgt * 0.72, cx + r_tgt * 0.72], [y1, y1], color="#c9a961", lw=2.2, zorder=7)
ax.plot([cx, cx], [y1 - r_tgt * 0.72, y1 + r_tgt * 0.72], color="#c9a961", lw=2.2, zorder=7)

ax.annotate(
    "CNOT",
    xy=(0.56, (y0 + y1) / 2),
    fontsize=11,
    color="#a8e0b8",
    va="center",
    fontweight="bold",
)

ax.plot([0.66, 0.74], [y0, y0], color="#8a96a8", lw=2.4, zorder=3)
ax.plot([0.66, 0.74], [y1, y1], color="#8a96a8", lw=2.4, zorder=3)

for y in (y0, y1):
    gate_shadow(0.76, y - 0.09, 0.13, 0.18)
    ax.add_patch(Rectangle((0.76, y - 0.09), 0.13, 0.18, facecolor="#1a1f28", edgecolor="#606878", linewidth=1.8, zorder=4))
    ax.plot([0.825, 0.875, 0.825], [y + 0.04, y, y - 0.04], color="#d0d8e8", lw=1.3)

ax.text(0.825, y0 + 0.14, "⌁", fontsize=11, color="#8899aa", ha="center")
ax.text(0.825, y1 + 0.14, "⌁", fontsize=11, color="#8899aa", ha="center")

ax.text(0.91, 0.5, r"$|\Phi^+\rangle$", fontsize=13, color="#c9a961", va="center", ha="center")
ax.text(0.91, 0.36, r"$\frac{|00\rangle+|11\rangle}{\sqrt{2}}$", fontsize=10, color="#8899aa", va="center", ha="center")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title(
    r"Bell preparation: $H \otimes I$, CNOT — textbook-equivalent to any framework diagram",
    color="#eef2f6",
    fontsize=12,
    fontweight="medium",
    pad=16,
)
plt.tight_layout()
plt.show()
```

### What this figure shows

- **Time reads left to right**: superposition on $q_0$, then CNOT entanglement, then measurement symbols.
- **CNOT** in standard form: filled **control** on $q_0$, **⊕ target** on $q_1$.
- **Right margin** states the output $|\Phi^+\rangle$ explicitly—same state any SDK prepares from this pattern.
- Drawn in **pure Matplotlib** so it runs in-browser; Qiskit’s `draw("mpl")` would depict the same sequence.

---

## QFT structure ($n=4$ schematic)

```python {run}
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

fig, ax = plt.subplots(figsize=(11.5, 4.5), facecolor="#080a0c")
ax.set_facecolor("#080a0c")
g = np.linspace(0.08, 0.22, 256).reshape(1, -1)
ax.imshow(g, extent=[0, 1, 0, 1], aspect="auto", cmap="bone", alpha=0.55, zorder=0)

n = 4
ys = [0.86 - i * 0.195 for i in range(n)]
cols = [0.11, 0.29, 0.47, 0.65]
for j, xc in enumerate(cols):
    ax.text(xc, 0.94, f"$t_{j}$", ha="center", fontsize=9, color="#6b7a8c")

for i, y in enumerate(ys):
    ax.plot([0.04, 0.93], [y, y], color="#2c3a4c", lw=2.4, solid_capstyle="round", zorder=1)
    ax.text(0.012, y, f"$q_{i}$", fontsize=13, color="#d4c4a8", va="center", fontweight="bold")

for j, x in enumerate(cols):
    for i in range(n):
        if i == j:
            ax.add_patch(
                FancyBboxPatch(
                    (x - 0.058, ys[i] - 0.068),
                    0.116,
                    0.136,
                    boxstyle="round,pad=0.025,rounding_size=0.06",
                    facecolor="#152a40",
                    edgecolor="#7eb0e8",
                    linewidth=2,
                    zorder=4,
                )
            )
            ax.text(x, ys[i], "H", ha="center", va="center", fontsize=11, color="#f0f6ff", fontweight="bold")
        elif i > j:
            ax.add_patch(Circle((x, ys[i]), 0.024, facecolor="#c9a961", edgecolor="#fff4d0", linewidth=1.3, zorder=5))
            ax.plot([x, x], [ys[j], ys[i]], color="#a08050", lw=1.8, alpha=0.85, zorder=3)

ax.text(0.795, 0.5, "controlled\nphase\nstack", ha="left", va="center", fontsize=9, color="#7a8a9c", linespacing=1.15)

ax.text(0.88, 0.5, "⋯\n+ swap\nlayer", ha="center", va="center", fontsize=10, color="#7a8a9c", linespacing=1.25)
ax.set_xlim(0, 1)
ax.set_ylim(0.02, 0.98)
ax.axis("off")
ax.set_title(
    r"QFT structure ($n{=}4$): diagonal Hadamards + controlled phases (schematic)",
    color="#e8ecf0",
    fontsize=12,
    pad=12,
)
plt.tight_layout()
plt.show()
```

### What this figure shows

- **Diagonal $H$ pattern**: each qubit gets a Hadamard at a different stage of the transform.
- **Gold dots + vertical ties**: schematic controlled phase dependencies (detail omitted—focus is **structure**).
- **Swaps** are essential in the canonical QFT layout; label reminds you the 2D diagram is incomplete without them.
- Matches how you read QFT complexity: $O(n^2)$ gates from nested phases.

---

## Bloch sphere — state trajectory (static)

A pure qubit state lives on the **unit sphere**. Here $R_y(\cdot)$ and $R_z(\cdot)$ are composed along $t$; the figure shows the **full path** with a **color gradient** along the curve (early $t$ → late $t$), a faint **equator**, and a **fixed** camera—no rotation, no GIF.

```python {run}
import numpy as np
import matplotlib.pyplot as plt

def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def rz(phi):
    return np.array(
        [[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]],
        dtype=complex,
    )

def bloch(psi):
    a, b = psi[0], psi[1]
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = np.abs(a) ** 2 - np.abs(b) ** 2
    return x, y, z

t_vals = np.linspace(0, 2 * np.pi, 160)
points = []
for t in t_vals:
    U = ry(0.82 * np.sin(t) + np.pi / 2.15) @ rz(1.35 * t)
    p = U @ np.array([1.0, 0.0], dtype=complex)
    p /= np.linalg.norm(p)
    points.append(bloch(p))
points = np.array(points)

u = np.linspace(0, 2 * np.pi, 72)
v = np.linspace(0, np.pi, 36)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))

fig = plt.figure(figsize=(8.2, 7.0), facecolor="#06080c")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#06080c")

ax.plot_surface(
    xs, ys, zs,
    rstride=3,
    cstride=3,
    alpha=0.22,
    color="#3d6a9e",
    edgecolor="#1a3048",
    linewidth=0.08,
    shade=True,
)

th = np.linspace(0, 2 * np.pi, 200)
ax.plot(np.cos(th), np.sin(th), np.zeros_like(th), color="#5a5040", lw=1.0, ls="--", alpha=0.65)
ax.plot(np.cos(th), np.zeros_like(th), np.sin(th), color="#3a4550", lw=0.7, ls=":", alpha=0.45)

for i in range(len(points) - 1):
    tcol = plt.cm.inferno(0.15 + 0.75 * i / (len(points) - 2))
    ax.plot(
        points[i : i + 2, 0],
        points[i : i + 2, 1],
        points[i : i + 2, 2],
        color=tcol,
        lw=3.0,
        solid_capstyle="round",
    )

ax.scatter([0], [0], [1], s=120, c="#5eb0ff", edgecolors="#c8e4ff", linewidths=2, zorder=20, label=r"$|0\rangle$ start")
ax.scatter(
    [points[-1, 0]], [points[-1, 1]], [points[-1, 2]],
    s=140,
    c="#ffd080",
    edgecolors="#c9a961",
    linewidths=2,
    zorder=21,
    label=r"$t=2\pi$",
)

for v0, lab in [([1.08, 0, 0], "X"), ([0, 1.08, 0], "Y"), ([0, 0, 1.08], "Z")]:
    ax.plot([0, v0[0]], [0, v0[1]], [0, v0[2]], color="#3a4558", lw=1.2)
    ax.text(v0[0], v0[1], v0[2], lab, color="#a8b0c0", fontsize=11)

ax.set_box_aspect([1, 1, 1])
ax.grid(False)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False
    axis.pane.set_edgecolor("#1a1e28")
    axis.line.set_color("#2a3040")
ax.view_init(elev=20, azim=38)
ax.set_title(
    "Bloch sphere: unitary path (color = progression along $t$)",
    color="#e8ecf4",
    fontsize=12,
    pad=14,
)
ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.0, 1.0),
    frameon=True,
    facecolor="#12161c",
    edgecolor="#3a4558",
    labelcolor="#d0d8e4",
    fontsize=9,
)
plt.tight_layout()
plt.show()
```

### What this figure shows

- The **mesh** is the unit sphere where **pure** one-qubit states live.
- **Dashed equator** ($z=0$) and a **meridian** in the $xz$-plane give orientation like a textbook figure.
- **Color along the curve** encodes progression in $t$—the path is one continuous $R_y$–$R_z$ family, not random jumps.
- **Fixed viewpoint** keeps the graphic readable; same geometry you would see in a lecture slide or Qiskit Bloch view, drawn without Qiskit.

---

## $|\Phi^+\rangle$: amplitudes, probabilities, and density matrix

```python {run}
import numpy as np
import matplotlib.pyplot as plt

labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
real = np.real(psi)
imag = np.imag(psi)
probs = np.abs(psi) ** 2

rho = np.outer(psi, np.conj(psi))
rho_r = np.real(rho)

fig = plt.figure(figsize=(12.5, 5.2), facecolor="#080a0c")
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.28)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])
for ax in (ax1, ax2, ax3):
    ax.set_facecolor("#0e1218")

x = np.arange(4)
w = 0.34
ax1.bar(x - w / 2, real, w, color="#4a7ec8", edgecolor="#1a3050", linewidth=1.3, label=r"$\mathrm{Re}$", alpha=0.92)
ax1.bar(x + w / 2, imag, w, color="#d4a050", edgecolor="#503010", linewidth=1.3, label=r"$\mathrm{Im}$", alpha=0.92)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, color="#d8e0ec", fontsize=11)
ax1.set_ylabel("Amplitude", color="#98a8b8")
ax1.legend(frameon=True, facecolor="#141c28", edgecolor="#3a4860", labelcolor="#ccc", loc="upper right")
ax1.set_title(r"Statevector $|\Phi^+\rangle$", color="#f0f4f8", fontsize=12)
ax1.tick_params(colors="#8899aa")
ax1.axhline(0, color="#3a4558", lw=1)
ax1.grid(axis="y", alpha=0.12)
ax1.set_ylim(-0.8, 0.8)

ax2.bar(x, probs, color=["#3d5a80", "#2a4a60", "#2a4a60", "#3d5a80"], edgecolor="#1a2838", linewidth=1.4)
for i, p in enumerate(probs):
    if p > 0.01:
        ax2.text(i, p + 0.03, f"{p:.2f}", ha="center", fontsize=10, color="#c9a961")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, color="#d8e0ec", fontsize=11)
ax2.set_ylabel("Probability", color="#98a8b8")
ax2.set_ylim(0, 0.7)
ax2.set_title(r"Born probabilities $|\alpha|^2$", color="#f0f4f8", fontsize=12)
ax2.tick_params(colors="#8899aa")
ax2.grid(axis="y", alpha=0.12)

im = ax3.imshow(rho_r, cmap="cividis", vmin=-0.6, vmax=0.6, aspect="equal")
ax3.set_xticks(range(4))
ax3.set_yticks(range(4))
ax3.set_xticklabels(labels, fontsize=10, color="#b8c4d0")
ax3.set_yticklabels(labels, fontsize=10, color="#b8c4d0")
ax3.set_title(r"Real part of density matrix $\rho = |\Phi^+\rangle\langle\Phi^+|$ (computational basis)", color="#f0f4f8", fontsize=11)
for i in range(4):
    for j in range(4):
        ax3.text(j, i, f"{rho_r[i, j]:.2f}", ha="center", va="center", color="#f0f0f0" if abs(rho_r[i, j]) > 0.2 else "#8899aa", fontsize=10)
cbar = plt.colorbar(im, ax=ax3, fraction=0.035, pad=0.02)
cbar.ax.tick_params(colors="#8899aa")
for tl in cbar.ax.get_yticklabels():
    tl.set_color("#8899aa")

fig.suptitle("Two-qubit Bell state — same object, three views", color="#c9a961", fontsize=11, y=1.02)
plt.tight_layout()
plt.show()
```

### What this figure shows

- **Top left**: complex **amplitudes**; only $|00\rangle$ and $|11\rangle$ components are nonzero here.
- **Top right**: **Born** probabilities—two outcomes each with probability $1/2$.
- **Bottom**: **real part** of $\rho=|\psi\rangle\langle\psi|$; off-diagonal $\frac{1}{2}$ terms witness **quantum** coherence (classical mixtures would lack them).
- Together this matches what Qiskit’s `Statevector` and `DensityMatrix` plots summarize—rendered here without Qiskit.

---

# 5. Hardware-aware and algorithm-aware visualization

## Device connectivity

```python {run}
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

edges = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (5, 6), (4, 7)]
G = nx.Graph()
G.add_edges_from(edges)

pos = {
    0: (0, 1),
    1: (1, 1),
    2: (2, 1),
    3: (3, 1),
    4: (1, 0),
    5: (2, 0),
    6: (3, 0),
    7: (1, -1),
}

fig, ax = plt.subplots(figsize=(8.6, 5.5), facecolor="#080a0c")
ax.set_facecolor("#080a0c")
ax.set_xlim(-0.55, 3.55)
ax.set_ylim(-1.45, 1.35)

panel = FancyBboxPatch(
    (-0.48, -1.38),
    4.0,
    2.68,
    boxstyle="round,pad=0.04",
    facecolor="#0c1018",
    edgecolor="#2a384c",
    linewidth=1.35,
    zorder=0,
)
ax.add_patch(panel)

nx.draw_networkx_edges(
    G,
    pos,
    ax=ax,
    edge_color="#3a5070",
    width=4.2,
    alpha=0.45,
    zorder=1,
)
nx.draw_networkx_edges(
    G,
    pos,
    ax=ax,
    edge_color="#6a9cc8",
    width=2.2,
    alpha=0.95,
    zorder=2,
)
nx.draw_networkx_nodes(
    G,
    pos,
    ax=ax,
    node_color="#152230",
    node_size=1750,
    edgecolors="#d4b060",
    linewidths=2.2,
    zorder=3,
)
nx.draw_networkx_labels(
    G,
    pos,
    ax=ax,
    font_size=12,
    font_color="#f2f6fa",
    font_weight="bold",
    zorder=4,
)

ax.set_title(
    "Coupling map — physical neighbors only",
    color="#f0f4f8",
    fontsize=13,
    pad=12,
    fontweight="600",
)
ax.text(
    0.5,
    1.02,
    "Same abstraction as `coupling_map` in Qiskit / backend topology",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
    color="#8899aa",
)
ax.axis("off")
plt.tight_layout()
plt.show()
```

### What this figure shows

- **Nodes** are physical qubits; **edges** are native two-qubit couplings.
- Circuits must be **routed** onto this graph—non-local gates cost extra SWAP depth.
- Sparse connectivity interacts with **noise**: longer decompositions accumulate more error.
- This is the same object transpilers and schedulers use as input.

---

## Variational optimization trace

```python {run}
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

np.random.seed(5)
steps = np.arange(1, 61)
base = -0.54 - 0.42 * (1 - np.exp(-steps / 14))
noise = 0.008 * np.random.randn(len(steps))
energy = base + noise

fig, ax = plt.subplots(figsize=(9.2, 5), facecolor="#080a0c")
ax.set_facecolor("#080a0c")

panel = FancyBboxPatch(
    (0.02, 0.06),
    0.96,
    0.86,
    transform=ax.transAxes,
    boxstyle="round,pad=0.008",
    facecolor="#0e1218",
    edgecolor="#2a3448",
    linewidth=1.2,
    zorder=0,
)
ax.add_patch(panel)

ax.fill_between(steps, base - 0.02, base + 0.02, color="#3a5080", alpha=0.18, zorder=1)
ax.plot(steps, base, color="#6a88b0", lw=1.6, ls="--", alpha=0.85, zorder=2, label="Noiseless trend")
ax.fill_between(steps, energy - 0.018, energy + 0.018, alpha=0.22, color="#4a7ab0", zorder=2)
ax.plot(steps, energy, color="#e8c066", lw=2.35, zorder=4, solid_capstyle="round")
ax.scatter(
    steps,
    energy,
    c=steps,
    cmap="inferno",
    s=32,
    zorder=5,
    edgecolors="#1a1410",
    linewidths=0.45,
    alpha=0.95,
)

ax.set_title(
    "Hybrid loop — estimated energy vs classical iteration",
    color="#f0f4f8",
    fontsize=13,
    pad=10,
    fontweight="600",
)
ax.text(
    0.5,
    1.02,
    "Schematic VQE-style trace (shots + optimizer noise)",
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
    color="#8899aa",
)
ax.set_xlabel("Iteration", color="#98a8b8", fontsize=11)
ax.set_ylabel("Estimated energy", color="#98a8b8", fontsize=11)
ax.tick_params(colors="#8899aa")
ax.grid(alpha=0.12, color="#4a5568")
ax.set_axisbelow(True)
ax.legend(
    loc="lower right",
    frameon=True,
    facecolor="#141a24",
    edgecolor="#3a4860",
    labelcolor="#c8d0dc",
    fontsize=9,
)
plt.tight_layout()
plt.show()
```

### What this figure shows

- Each point is one **classical update** after quantum measurements fed an objective (e.g. VQE energy).
- **Downward trend** is what you hope for; **jitter** stands in for shot noise and device variability.
- The shaded band suggests **uncertainty** in the estimate—not a smooth deterministic descent.
- The object of study is the **loop**, not a single circuit diagram.

---

# 6. Applied perspective

- **Three layers together**: quantum formalism, imperfect hardware, algorithms and software stacks.  
- **Flow**: postulates → device physics → superposition, interference, entanglement in algorithms → compilation onto **noisy, sparse** graphs.  
- **Visualization** ties abstract symbols to **circuits**, **Bloch geometry**, **probabilities**, and **hardware constraints**—the usual interfaces for interpreting quantum experiments and simulations.
