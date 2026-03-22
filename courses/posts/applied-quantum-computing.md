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

Visualizations below are meant to **connect symbols to geometry**: time flows left-to-right on circuits, amplitudes and probabilities for entanglement, and the Bloch sphere for single-qubit unitary motion.

## Bell-state preparation circuit

```python {run}
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap

fig, ax = plt.subplots(figsize=(10.5, 3.6), facecolor="#0f0f0f")
ax.set_facecolor("#0f0f0f")
grad = LinearSegmentedColormap.from_list("bg", ["#1a2230", "#0f0f0f"])
ax.imshow([[0, 1]], extent=[0, 1, 0, 1], aspect="auto", cmap=grad, zorder=0)

y0, y1 = 0.72, 0.28
for y, lab in [(y0, r"$q_0$"), (y1, r"$q_1$")]:
    ax.plot([0.06, 0.93], [y, y], color="#3a4555", lw=2.2, solid_capstyle="round", zorder=1)
    ax.text(0.018, y, lab, fontsize=14, color="#d4b976", va="center", fontweight="medium")

def gate_box(x, y, w, h, text, fc="#243044", ec="#7eb8e8"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            facecolor=fc,
            edgecolor=ec,
            linewidth=2,
            zorder=3,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=13, color="#f0f0f0", fontweight="bold")

gate_box(0.16, y0 - 0.095, 0.11, 0.19, "H", fc="#2a3850", ec="#6b9ac4")
ax.plot([0.27, 0.4], [y0, y0], color="#5a6575", lw=2, zorder=2)
ax.add_patch(Circle((0.44, y0), 0.032, facecolor="#c9a961", edgecolor="#f0e0b0", linewidth=1.5, zorder=5))
ax.plot([0.44, 0.44], [y0, y1], color="#c9a961", lw=2.2, zorder=4)
ax.add_patch(
    FancyBboxPatch((0.405, y1 - 0.11), 0.07, 0.22, boxstyle="square,pad=0", facecolor="#0f0f0f", edgecolor="#c9a961", linewidth=2.5, zorder=4)
)
ax.plot([0.405, 0.475], [y1, y1], color="#c9a961", lw=2.5)
ax.annotate(
    "CNOT",
    xy=(0.52, (y0 + y1) / 2),
    fontsize=11,
    color="#a8d4a8",
    va="center",
    fontweight="bold",
)

for y in (y0, y1):
    ax.add_patch(Rectangle((0.76, y - 0.085), 0.12, 0.17, facecolor="#252525", edgecolor="#707080", linewidth=1.5, zorder=3))
    ax.plot([0.82, 0.875, 0.82], [y + 0.045, y, y - 0.045], color="#c0c0c0", lw=1.2)

ax.text(0.82, y0 + 0.16, "measure", fontsize=8, color="#888", ha="center")
ax.text(0.82, y1 + 0.16, "measure", fontsize=8, color="#888", ha="center")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title(
    r"Bell pair: $H$ on $q_0$, then CNOT $(q_0 \rightarrow q_1)$ — prepares $|\Phi^+\rangle$",
    color="#e8e8e8",
    fontsize=12,
    pad=14,
)
plt.tight_layout()
plt.show()
```

### What this figure shows

- **Time reads left to right**: first superposition on $q_0$, then entanglement with $q_1$.
- **CNOT** uses control (solid dot) and target (boxed $\oplus$ line)—the standard diagram idiom.
- **Meters** denote projective measurement in the computational basis after the unitary stage.
- Same logic as any framework: only the drawing style changes.

---

## QFT structure ($n=4$ schematic)

```python {run}
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

fig, ax = plt.subplots(figsize=(11, 4.2), facecolor="#0f0f0f")
ax.set_facecolor("#0f0f0f")
grad = LinearSegmentedColormap.from_list("g", ["#152028", "#0a0a0a"])
ax.imshow(np.linspace(0, 1, 100).reshape(1, -1), extent=[0, 1, 0, 1], aspect="auto", cmap=grad, zorder=0)

n = 4
ys = [0.84 - i * 0.2 for i in range(n)]
for i, y in enumerate(ys):
    ax.plot([0.05, 0.92], [y, y], color="#3d4a5c", lw=1.8, zorder=1)
    ax.text(0.015, y, f"$q_{i}$", fontsize=12, color="#b8c4d4", va="center")

cols = [0.12, 0.30, 0.48, 0.66]
for j, x in enumerate(cols):
    for i in range(n):
        if i == j:
            ax.add_patch(
                FancyBboxPatch(
                    (x - 0.055, ys[i] - 0.065),
                    0.11,
                    0.13,
                    boxstyle="round,pad=0.02,rounding_size=0.05",
                    facecolor="#1e3a50",
                    edgecolor="#6b9ac4",
                    linewidth=1.8,
                    zorder=3,
                )
            )
            ax.text(x, ys[i], "H", ha="center", va="center", fontsize=10, color="#e0e8f0", fontweight="bold")
        elif i > j:
            ax.add_patch(Circle((x, ys[i]), 0.022, facecolor="#c9a961", edgecolor="#f5e6c0", linewidth=1, zorder=4))
            ax.plot([x, x], [ys[j], ys[i]], color="#8a7a60", lw=1.4, linestyle=(0, (2, 3)), zorder=2)

ax.text(0.84, 0.5, "⋯\n+ swaps", ha="center", va="center", fontsize=11, color="#6a7585", linespacing=1.3)
ax.set_xlim(0, 1)
ax.set_ylim(0.02, 0.98)
ax.axis("off")
ax.set_title("QFT($n{=}4$): Hadamard on each line, controlled phases to lower qubits, then swap layer", color="#ddd", fontsize=11)
plt.tight_layout()
plt.show()
```

### What this figure shows

- **Diagonal $H$ pattern**: each qubit gets a Hadamard at a different stage of the transform.
- **Gold dots + vertical ties**: schematic controlled phase dependencies (detail omitted—focus is **structure**).
- **Swaps** are essential in the canonical QFT layout; label reminds you the 2D diagram is incomplete without them.
- Matches how you read QFT complexity: $O(n^2)$ gates from nested phases.

---

## Bloch sphere — animated state trajectory

A single qubit $|\psi\rangle$ maps to a point on the **unit sphere**; applying $R_y(\cdot)$ and $R_z(\cdot)$ moves that point. The animation traces a **continuous unitary path** from $|0\rangle$ and slowly **rotates the camera** so depth and curvature are visible.

```python {run}
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import base64
import os

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

t_vals = np.linspace(0, 2 * np.pi, 48)
points = []
for t in t_vals:
    U = ry(0.82 * np.sin(t) + np.pi / 2.15) @ rz(1.35 * t)
    p = U @ np.array([1.0, 0.0], dtype=complex)
    p /= np.linalg.norm(p)
    points.append(bloch(p))
points = np.array(points)

u = np.linspace(0, 2 * np.pi, 48)
v = np.linspace(0, np.pi, 24)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))

fig = plt.figure(figsize=(7.2, 6.4), facecolor="#0f0f0f")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#0f0f0f")


def draw_frame(k):
    ax.clear()
    ax.set_facecolor("#0f0f0f")
    ax.plot_surface(
        xs, ys, zs,
        alpha=0.18,
        color="#4a7ab0",
        edgecolor="#2a4058",
        linewidth=0.15,
        rstride=2,
        cstride=2,
    )
    trail = points[: k + 1]
    if len(trail) > 1:
        ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color="#c9a961", lw=2.8, zorder=10)
    ax.scatter(
        [points[k, 0]], [points[k, 1]], [points[k, 2]],
        s=85,
        c="#fff4d0",
        edgecolors="#c9a961",
        linewidths=1.5,
        zorder=11,
    )
    for v0, lab in [([1, 0, 0], "X"), ([0, 1, 0], "Y"), ([0, 0, 1], "Z")]:
        ax.plot([0, v0[0]], [0, v0[1]], [0, v0[2]], color="#555", lw=1.1)
        ax.text(v0[0] * 1.12, v0[1] * 1.12, v0[2] * 1.12, lab, color="#999", fontsize=10)
    ax.set_box_aspect([1, 1, 1])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.line.set_color("#333")
    ax.view_init(elev=18, azim=25 + k * 5.5)
    ax.set_title("Bloch trajectory + rotating view", color="#e0e0e0", fontsize=11, pad=10)


anim = animation.FuncAnimation(fig, draw_frame, frames=len(t_vals), interval=75, repeat=True)
path = "_bloch_q_anim.gif"
anim.save(path, writer="pillow", fps=12)
plt.close("all")
with open(path, "rb") as f:
    _ANIM_GIF = base64.b64encode(f.read()).decode()
try:
    os.remove(path)
except OSError:
    pass
print("Frames:", len(t_vals))
```

### What this animation shows

- The **state vector** of one qubit is a point on the **surface** of the Bloch sphere (pure states).
- **Unitary gates** are rigid motions of that point; here $R_y$ and $R_z$ combine into a smooth path.
- The **highlighted dot** is the current state; the **gold curve** is the history—continuity of evolution.
- **Rotating camera** stresses 3D structure; without motion, depth is easy to misread on a flat screen.

---

## $|\Phi^+\rangle$: amplitudes and measurement statistics

```python {run}
import numpy as np
import matplotlib.pyplot as plt
labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
real = np.real(psi)
imag = np.imag(psi)
probs = np.abs(psi) ** 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor="#0f0f0f")
for ax in (ax1, ax2):
    ax.set_facecolor("#141820")

x = np.arange(4)
w = 0.36
colors_r = ["#4a7ab8", "#5a8ac8", "#5a8ac8", "#4a7ab8"]
colors_i = ["#b8945a", "#c9a45a", "#c9a45a", "#b8945a"]
ax1.bar(x - w / 2, real, w, color=colors_r, edgecolor="#2a3545", linewidth=1.2, label="Re")
ax1.bar(x + w / 2, imag, w, color=colors_i, edgecolor="#2a3545", linewidth=1.2, label="Im")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, color="#d0d8e0", fontsize=11)
ax1.set_ylabel("Amplitude", color="#a8b0c0")
ax1.legend(frameon=True, facecolor="#1c2430", edgecolor="#3a4555", labelcolor="#ccc", loc="upper right")
ax1.set_title(r"$|\Phi^+\rangle$: complex amplitudes", color="#e8e8e8", fontsize=12)
ax1.tick_params(colors="#888")
ax1.axhline(0, color="#4a5565", lw=0.9)
ax1.grid(axis="y", alpha=0.15)
ax1.set_ylim(-0.75, 0.75)

colors_p = ["#414487", "#2a788e", "#22a884", "#7ad151"]
ax2.bar(x, probs, color=colors_p, edgecolor="#2a3545", linewidth=1.2)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, color="#d0d8e0", fontsize=11)
ax2.set_ylabel("Probability", color="#a8b0c0")
ax2.set_ylim(0, 0.65)
ax2.set_title("Computational-basis measurement (Born rule)", color="#e8e8e8", fontsize=12)
ax2.tick_params(colors="#888")
ax2.grid(axis="y", alpha=0.15)

plt.tight_layout()
plt.show()
```

### What this figure shows

- **Left**: only $|00\rangle$ and $|11\rangle$ have nonzero amplitude; real parts are equal, imaginaries zero for this standard Bell state.
- **Right**: squaring magnitudes gives **50/50** for those two outcomes—no $|01\rangle$ or $|10\rangle$ probability.
- Together, the panels link **complex amplitudes** (interference) to **observed frequencies** (measurement).
- This is the same information a density-matrix or histogram view would emphasize for a pure state.

---

# 5. Hardware-aware and algorithm-aware visualization

## Device connectivity

```python {run}
import networkx as nx
import matplotlib.pyplot as plt

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

fig, ax = plt.subplots(figsize=(8, 5.2), facecolor="#0f0f0f")
ax.set_facecolor("#0f0f0f")

nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#5a8ab0", width=2.8, alpha=0.9)
nx.draw_networkx_nodes(
    G,
    pos,
    ax=ax,
    node_color="#1c2a38",
    node_size=1600,
    edgecolors="#c9a961",
    linewidths=2,
)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_color="#f0f0f0", font_weight="bold")

ax.set_title("Coupling map: physical neighbors only (not all-to-all)", color="#e8e8e8", fontsize=12)
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

np.random.seed(5)
steps = np.arange(1, 61)
base = -0.54 - 0.42 * (1 - np.exp(-steps / 14))
noise = 0.008 * np.random.randn(len(steps))
energy = base + noise

fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="#0f0f0f")
ax.set_facecolor("#141820")
ax.plot(steps, energy, color="#c9a961", lw=2.4, zorder=2)
ax.scatter(steps, energy, c=steps, cmap="plasma", s=28, zorder=3, edgecolors="#2a2030", linewidths=0.4)
ax.fill_between(steps, energy - 0.018, energy + 0.018, alpha=0.2, color="#4a7ab0", zorder=1)
ax.set_title("Hybrid loop: estimated energy vs classical iteration (schematic)", color="#e8e8e8", fontsize=12)
ax.set_xlabel("Iteration", color="#a0a8b8")
ax.set_ylabel("Estimated energy", color="#a0a8b8")
ax.tick_params(colors="#888")
ax.grid(alpha=0.18)
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
