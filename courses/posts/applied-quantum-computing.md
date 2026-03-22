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

## Bell-state circuit (Qiskit — local)

아래는 **로컬에 Qiskit이 설치된 환경**에서 `qc.draw("mpl")`로 얻는 방식입니다. 브라우저(Pyodide)에는 Qiskit이 없으므로, 바로 다음 블록에서 **동일 회로를 matplotlib로 그린** 버전을 실행할 수 있습니다.

```python
from qiskit import QuantumCircuit

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

qc.draw("mpl")
```

- Minimal circuit: product state → **entangled** Bell pair; ties together gates, measurement, and entanglement.

### Bell circuit — matplotlib (browser)

```python {run}
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(9, 3.2), facecolor="#1a1a1a")
ax.set_facecolor("#1a1a1a")

y0, y1 = 0.72, 0.28
for y, lab in [(y0, r"$q_0$"), (y1, r"$q_1$")]:
    ax.plot([0.05, 0.92], [y, y], color="#555", lw=1.8, zorder=1)
    ax.text(0.02, y, lab, fontsize=13, color="#c9a961", va="center")

def box(x, y, w, h, text, fc="#2d3a4a", ec="#6b9ac4"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", facecolor=fc, edgecolor=ec, linewidth=1.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12, color="#e8e8e8", fontweight="bold")

box(0.18, y0 - 0.09, 0.1, 0.18, "H")
ax.plot([0.28, 0.42], [y0, y0], color="#888", lw=1.8)
ax.add_patch(Circle((0.46, y0), 0.028, facecolor="#c9a961", edgecolor="#e8d5a3", zorder=4))
ax.plot([0.46, 0.46], [y0, y1], color="#c9a961", lw=1.8, zorder=3)
ax.add_patch(
    FancyBboxPatch((0.43, y1 - 0.1), 0.06, 0.2, boxstyle="square,pad=0", facecolor="#1a1a1a", edgecolor="#c9a961", linewidth=2)
)
ax.plot([0.43, 0.49], [y1, y1], color="#c9a961", lw=2)
ax.text(0.52, (y0 + y1) / 2, "CNOT", fontsize=10, color="#97c4a0", va="center")

for y in (y0, y1):
    ax.add_patch(Rectangle((0.78, y - 0.08), 0.1, 0.16, facecolor="#333", edgecolor="#888"))
    ax.plot([0.83, 0.88, 0.83], [y + 0.04, y, y - 0.04], color="#ccc", lw=1)

ax.text(0.835, y0 + 0.14, "meas", fontsize=8, color="#888", ha="center")
ax.text(0.835, y1 + 0.14, "meas", fontsize=8, color="#888", ha="center")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title(r'Bell: $H \otimes I$ then CNOT — $|\Phi^+\rangle$ preparation', color="#ddd", fontsize=12)
plt.tight_layout()
plt.show()
```

## QFT circuit (Qiskit — local)

```python
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT

n = 4
qc = QFT(num_qubits=n, do_swaps=True).decompose()
qc.draw("mpl", fold=-1)
```

- QFT diagrams show repeated **Hadamards**, **controlled phases**, and **swaps**.

### QFT structure — schematic ($n=4$)

```python {run}
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(10, 4), facecolor="#1a1a1a")
ax.set_facecolor("#1a1a1a")
n = 4
ys = [0.82 - i * 0.2 for i in range(n)]
for i, y in enumerate(ys):
    ax.plot([0.06, 0.94], [y, y], color="#444", lw=1.5)
    ax.text(0.02, y, f"q{i}", fontsize=11, color="#aaa", va="center")

cols = [0.14, 0.32, 0.5, 0.68]
for j, x in enumerate(cols):
    for i in range(n):
        if i == j:
            ax.add_patch(FancyBboxPatch((x - 0.05, ys[i] - 0.06), 0.1, 0.12, boxstyle="round,pad=0.02", facecolor="#2a3545", edgecolor="#6b9ac4"))
            ax.text(x, ys[i], "H", ha="center", va="center", fontsize=9, color="#ddd")
        elif i > j:
            ax.text(x, ys[i], "•", ha="center", va="center", fontsize=14, color="#c9a961")
            ax.plot([x, x], [ys[j], ys[i]], color="#666", lw=1, linestyle=":")

ax.text(0.86, 0.5, "⋯\n+ swaps", ha="center", va="center", fontsize=10, color="#777")
ax.set_xlim(0, 1)
ax.set_ylim(0.05, 0.95)
ax.axis("off")
ax.set_title("QFT(n=4): Hadamard on diagonal + controlled phases below", color="#ccc", fontsize=11)
plt.tight_layout()
plt.show()
```

## Bloch sphere — pure state

$$|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$$

$$x=\sin\theta\cos\phi,\qquad y=\sin\theta\sin\phi,\qquad z=\cos\theta$$

### Qiskit (local)

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt

qc = QuantumCircuit(1)
qc.h(0)
qc.s(0)

sv = Statevector.from_instruction(qc)
fig = plot_bloch_multivector(sv)
plt.show()
```

### 3D Bloch trajectory (browser — numpy unitaries)

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

def bloch_from_state(psi):
    a, b = psi[0], psi[1]
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = np.abs(a) ** 2 - np.abs(b) ** 2
    return x, y, z

num_steps = 72
points = []
psi = np.array([1.0, 0.0], dtype=complex)
for t in np.linspace(0, 2 * np.pi, num_steps):
    U = ry(0.85 * np.sin(t) + np.pi / 2) @ rz(1.4 * t)
    psi = U @ np.array([1.0, 0.0], dtype=complex)
    psi /= np.linalg.norm(psi)
    points.append(bloch_from_state(psi))
points = np.array(points)

fig = plt.figure(figsize=(7, 6.5), facecolor="#1a1a1a")
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#1a1a1a")

u = np.linspace(0, 2 * np.pi, 64)
v = np.linspace(0, np.pi, 32)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(xs, ys, zs, alpha=0.14, color="#6b9ac4", edgecolor="none", shade=False)

ax.plot(points[:, 0], points[:, 1], points[:, 2], color="#c9a961", lw=2.4, zorder=10)
ax.scatter([points[-1, 0]], [points[-1, 1]], [points[-1, 2]], color="#e8d5a3", s=60, zorder=11)

for v0, lab, col in [([1, 0, 0], "X", "#888"), ([0, 1, 0], "Y", "#888"), ([0, 0, 1], "Z", "#888")]:
    ax.plot([0, v0[0]], [0, v0[1]], [0, v0[2]], color=col, lw=1)
    ax.text(v0[0] * 1.15, v0[1] * 1.15, v0[2] * 1.15, lab, color="#aaa", fontsize=11)

ax.set_title("Single-qubit trajectory on the Bloch sphere", color="#e0e0e0", fontsize=12, pad=12)
ax.set_box_aspect([1, 1, 1])
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.line.set_color("#333")
plt.tight_layout()
plt.show()
```

## Bell state — amplitudes and probabilities

### Qiskit `plot_state_city` / histogram (local)

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_city, plot_histogram
import matplotlib.pyplot as plt

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

sv = Statevector.from_instruction(qc)

fig1 = plot_state_city(sv)
plt.show()

probs = sv.probabilities_dict()
fig2 = plot_histogram(probs)
plt.show()
```

### Browser — $|\Phi^+\rangle$ real/imag bars + measurement probabilities

```python {run}
import numpy as np
import matplotlib.pyplot as plt

labels = [r"$|00\rangle$", r"$|01\rangle$", r"$|10\rangle$", r"$|11\rangle$"]
psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
real = np.real(psi)
imag = np.imag(psi)
probs = np.abs(psi) ** 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor="#1a1a1a")
for ax in (ax1, ax2):
    ax.set_facecolor("#1a1a1a")

x = np.arange(4)
w = 0.35
ax1.bar(x - w / 2, real, w, label="Re", color="#6b9ac4", edgecolor="#444")
ax1.bar(x + w / 2, imag, w, label="Im", color="#c9a961", edgecolor="#444")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, color="#ccc")
ax1.set_ylabel("Amplitude", color="#aaa")
ax1.legend(facecolor="#252525", labelcolor="#ccc", edgecolor="#444")
ax1.set_title(r"$|\Phi^+\rangle$: amplitude components", color="#ddd")
ax1.tick_params(colors="#888")
ax1.axhline(0, color="#555", lw=0.8)
ax1.grid(axis="y", alpha=0.2)

ax2.bar(x, probs, color="#97c4a0", edgecolor="#444")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, color="#ccc")
ax2.set_ylabel("Probability", color="#aaa")
ax2.set_ylim(0, 0.6)
ax2.set_title("Born rule: measurement in computational basis", color="#ddd")
ax2.tick_params(colors="#888")
ax2.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.show()
```

---

# 5. Hardware-aware and algorithm-aware visualization

## Device connectivity (coupling graph)

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

fig, ax = plt.subplots(figsize=(7.5, 5), facecolor="#1a1a1a")
ax.set_facecolor("#1a1a1a")
nx.draw_networkx(
    G,
    pos=pos,
    ax=ax,
    with_labels=True,
    node_color="#2a3545",
    node_size=1100,
    font_color="#e8e8e8",
    font_size=11,
    edge_color="#6b9ac4",
    width=2,
)
ax.set_title("Example hardware connectivity (not a complete graph)", color="#ddd", fontsize=12)
ax.axis("off")
plt.tight_layout()
plt.show()
```

- Real chips constrain **routing**, **SWAP depth**, and **effective noise**.

## Variational optimization trace

```python {run}
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4)
steps = np.arange(1, 51)
energy = -0.52 - 0.44 * (1 - np.exp(-steps / 12)) + 0.012 * np.random.randn(len(steps))

fig, ax = plt.subplots(figsize=(8.5, 4.5), facecolor="#1a1a1a")
ax.set_facecolor("#1a1a1a")
ax.plot(steps, energy, marker="o", ms=4, lw=2, color="#c9a961", markerfacecolor="#e8d5a3")
ax.fill_between(steps, energy - 0.02, energy + 0.02, alpha=0.15, color="#6b9ac4")
ax.set_title("VQE / QAOA style energy trace (schematic)", color="#ddd", fontsize=12)
ax.set_xlabel("Iteration", color="#aaa")
ax.set_ylabel("Estimated energy", color="#aaa")
ax.tick_params(colors="#888")
ax.grid(alpha=0.22)
plt.tight_layout()
plt.show()
```

---

# 6. Applied perspective

- **Three layers together**: quantum formalism, imperfect hardware, algorithms and software stacks.  
- **Flow**: postulates → device physics → superposition, interference, entanglement in algorithms → compilation onto **noisy, sparse** graphs.  
- **Visualization** is not decoration: circuits, states, connectivity, and optimization traces are how we **interpret** what the machine is doing.
