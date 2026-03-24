---
title: Mathematics of Quantum Computing
category: QUANTUM
semester: 2024 F
---

# 1. Quantum States and the Hilbert-Space View

## Qubits and Superposition
A qubit is a normalized state in a two-dimensional complex Hilbert space:
$$
|\psi\rangle = \alpha |0\rangle + \beta |1\rangle,
\qquad
|\alpha|^2+|\beta|^2=1.
$$

- Classical bits live in $\{0,1\}$.
- Qubits live in a complex vector space and allow coherent superposition.

For $L$ qubits, the state space becomes
$$
(\mathbb C^2)^{\otimes L},
\qquad
\dim = 2^L.
$$

- Multi-qubit state spaces grow exponentially with system size.

## Interference
Quantum amplitudes add before probabilities are computed.

For two alternatives with amplitudes $\phi_1,\phi_2$:
$$
P_1=|\phi_1|^2,\qquad P_2=|\phi_2|^2
$$
but jointly
$$
P_{12}=|\phi_1+\phi_2|^2
=|\phi_1|^2+|\phi_2|^2+2\operatorname{Re}(\phi_1\overline{\phi_2}).
$$

- The cross term is the interference term.
- Quantum evolution is organized at the amplitude level, not directly at the probability level.

### Figure — interference from a relative phase
Fix equal amplitudes $\varphi_1=\varphi_2=1/\sqrt2$ and vary a relative phase $\delta$ in $\varphi_1+e^{i\delta}\varphi_2$. The detection probability is $|\varphi_1+e^{i\delta}\varphi_2|^2$, oscillating between constructive ($\delta=0$) and destructive ($\delta=\pi$) interference.

```python {run}
import numpy as np
import matplotlib.pyplot as plt

delta = np.linspace(0, 2 * np.pi, 800)
phi1 = 1 / np.sqrt(2)
phi2 = 1 / np.sqrt(2)
prob = np.abs(phi1 + np.exp(1j * delta) * phi2) ** 2

fig, ax = plt.subplots(figsize=(9.2, 4.6), facecolor="#080a0c")
ax.set_facecolor("#0e1218")
ax.plot(delta, prob, color="#e8c066", lw=2.4, zorder=3)
ax.fill_between(delta, prob, color="#3d6a9a", alpha=0.18, zorder=1)
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"], color="#b8c4d0")
ax.set_xlabel(r"Relative phase $\delta$", color="#98a8b8", fontsize=11)
ax.set_ylabel(r"$|\varphi_1 + e^{i\delta}\varphi_2|^2$", color="#98a8b8", fontsize=11)
ax.set_title("Two-path interference at the amplitude level", color="#f0f4f8", fontsize=12)
ax.tick_params(colors="#8899aa")
ax.grid(alpha=0.12, color="#4a5568")
plt.tight_layout()
plt.show()
```

## Hilbert Space Structure
Quantum states are modeled in a complex inner product space.

Linearity:
$$
\langle x, z_1 y_1 + z_2 y_2\rangle
=
z_1\langle x,y_1\rangle + z_2\langle x,y_2\rangle
$$

Conjugate symmetry:
$$
\langle x,y\rangle = \langle y,x\rangle^*
$$

Positivity:
$$
\langle x,x\rangle \ge 0,
\qquad
\langle x,x\rangle = 0 \iff x=0
$$

Norm:
$$
\|x\|=\sqrt{\langle x,x\rangle}
$$

Orthogonality:
$$
\langle x,y\rangle=0
$$

Cauchy–Schwarz:
$$
|\langle v,w\rangle|^2 \le \langle v,v\rangle\langle w,w\rangle
$$

- A Hilbert space is a complete complex inner product space.
- The inner product controls probability amplitudes, norms, and orthogonality.

## Bra–Ket Notation
States are written as kets
$$
|\psi\rangle
$$
and dual vectors as bras
$$
\langle \psi |.
$$

The inner product is
$$
\langle \psi|\phi\rangle.
$$

- Ket = vector in state space.
- Bra = corresponding dual linear functional.

---

# 2. Operators, Measurement, and Time Evolution

## Linear Operators and Projectors
A linear operator $A$ satisfies
$$
A(\lambda_1|\psi_1\rangle+\lambda_2|\psi_2\rangle)
=
\lambda_1A|\psi_1\rangle+\lambda_2A|\psi_2\rangle.
$$

In general,
$$
AB\ne BA,
\qquad
[A,B]=AB-BA.
$$

For a normalized state $|u\rangle$, the projector onto its span is
$$
P_u = |u\rangle\langle u|,
\qquad
P_u^2=P_u.
$$

- Projectors encode measurement onto subspaces.
- Noncommutativity is a central quantum feature.

## Completeness Relation
If $\{|i\rangle\}$ is an orthonormal basis and
$$
|\psi\rangle=\sum_i \psi_i |i\rangle,
$$
then
$$
\sum_i |i\rangle\langle i|=I.
$$

- This is the resolution of the identity.
- It expresses basis completeness in operator form.

## Postulates of Quantum Mechanics
A closed quantum system is described by a normalized state
$$
|\psi\rangle,
\qquad
\langle\psi|\psi\rangle=1.
$$

An observable is represented by an operator with orthonormal eigenbasis, and measurement outcomes are its eigenvalues.

For an eigenbasis $\{|u_n\rangle\}$, the Born rule gives
$$
P(a_n)=|\langle u_n|\psi\rangle|^2.
$$

For a degenerate eigenspace with projector $P_n$:
$$
P(a_n)=\langle \psi|P_n|\psi\rangle.
$$

After observing $a_n$, the state collapses to
$$
\frac{P_n|\psi\rangle}{\sqrt{\langle \psi|P_n|\psi\rangle}}.
$$

For a qubit
$$
|\psi\rangle=a|0\rangle+b|1\rangle,
$$
measurement in the computational basis gives
$$
P(0)=|a|^2,\qquad P(1)=|b|^2.
$$

- Probabilities come from squared amplitudes.
- Measurement both reveals an outcome and changes the state.

### Figure — Born probabilities and the Bloch direction
Take $|\psi\rangle=\cos(\theta/2)|0\rangle+e^{i\phi}\sin(\theta/2)|1\rangle$ with $(\theta,\phi)=(2\pi/3,\pi/4)$. The bars are $P(0)=|a|^2$ and $P(1)=|b|^2$; the arrow is the corresponding Bloch vector $\mathbf{n}=(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta)$.

```python {run}
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

theta = 2 * np.pi / 3
phi = np.pi / 4
a = np.cos(theta / 2)
b = np.exp(1j * phi) * np.sin(theta / 2)
p0, p1 = np.abs(a) ** 2, np.abs(b) ** 2
nx, ny, nz = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)

fig = plt.figure(figsize=(10, 4.6), facecolor="#080a0c")
axb = fig.add_subplot(1, 2, 1)
axb.set_facecolor("#0e1218")
axb.bar([0, 1], [p0, p1], color=["#4a7ec8", "#d4a050"], edgecolor="#1a2838", linewidth=1.3, width=0.55)
axb.set_xticks([0, 1])
axb.set_xticklabels([r"$|0\rangle$", r"$|1\rangle$"], color="#d8e0ec")
axb.set_ylabel("Probability", color="#98a8b8")
axb.set_title("Computational-basis measurement", color="#f0f4f8", fontsize=11)
axb.set_ylim(0, 1.05)
axb.tick_params(colors="#8899aa")
axb.grid(axis="y", alpha=0.12)

u = np.linspace(0, 2 * np.pi, 48)
v = np.linspace(0, np.pi, 24)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones(np.size(u)), np.cos(v))

ax3 = fig.add_subplot(1, 2, 2, projection="3d", facecolor="#0e1218")
ax3.plot_surface(xs, ys, zs, rstride=1, cstride=1, color="#1c2434", edgecolor="#2a3448", linewidth=0.15, alpha=0.55)
ax3.plot([0, nx], [0, ny], [0, nz], color="#e8c066", lw=2.8)
ax3.scatter([0], [0], [1], s=70, c="#5eb0ff", edgecolors="#c8e4ff", linewidths=1.2, zorder=10)
ax3.scatter([nx], [ny], [nz], s=90, c="#ffd080", edgecolors="#c9a961", linewidths=1.4, zorder=11)
for v0, lab in [([1.15, 0, 0], "X"), ([0, 1.15, 0], "Y"), ([0, 0, 1.15], "Z")]:
    ax3.plot([0, v0[0]], [0, v0[1]], [0, v0[2]], color="#3a4558", lw=1)
    ax3.text(v0[0], v0[1], v0[2], lab, color="#a8b0c0", fontsize=9)
ax3.set_box_aspect([1, 1, 1])
ax3.grid(False)
for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
    axis.pane.fill = False
    axis.pane.set_edgecolor("#1a1e28")
ax3.view_init(elev=18, azim=42)
ax3.set_title("Bloch vector for the same pure state", color="#f0f4f8", fontsize=11, pad=10)

plt.tight_layout()
plt.show()
```

## Schrödinger Evolution
Time evolution is governed by
$$
i\hbar \frac{d}{dt}|\psi(t)\rangle = H(t)|\psi(t)\rangle,
$$
where $H$ is the Hamiltonian.

For time-independent $H$,
$$
|\psi(t_2)\rangle
=
\exp\!\left[-\frac{i}{\hbar}H(t_2-t_1)\right]|\psi(t_1)\rangle
=
U(t_1,t_2)|\psi(t_1)\rangle.
$$

- Hermitian Hamiltonians generate unitary evolution.
- Norm and total probability are preserved under time evolution.

---

# 3. Hermitian Geometry, Tensor Products, and Quantum Gates

## Adjoint, Hermitian, and Unitary Operators
The adjoint $A^\dagger$ is defined by
$$
\langle v|A|w\rangle = \langle A^\dagger v|w\rangle.
$$

Important classes:
- Hermitian:
$$
A=A^\dagger
$$
- Unitary:
$$
A^\dagger=A^{-1}
$$
- Normal:
$$
AA^\dagger=A^\dagger A
$$

For Hermitian operators:
- eigenvalues are real,
- eigenvectors from distinct eigenvalues are orthogonal,
- the operator is diagonalizable in an orthonormal basis.

- Observables are Hermitian because measured values must be real.
- Unitaries represent physically valid gates and time evolution.

## Commuting Observables
If
$$
[A,B]=0,
$$
then $A$ and $B$ can be simultaneously diagonalized.

- Commuting observables admit a common eigenbasis.
- Noncommuting observables encode incompatibility.

## Tensor Product Structure
Two quantum systems combine via tensor product:
$$
\mathbb C^2\otimes \mathbb C^2.
$$

More generally,
$$
\dim(V\otimes W)=\dim(V)\dim(W).
$$

A general bipartite state has the form
$$
\sum_{i,j} c_{ij}\, |v_i\rangle\otimes |w_j\rangle.
$$

Inner products factor:
$$
\langle v_i\otimes w_j,\; v'_\ell\otimes w'_k\rangle
=
\langle v_i,v'_\ell\rangle\langle w_j,w'_k\rangle.
$$

- Tensor product is the basic composition law for quantum systems.
- Entanglement becomes possible only in the tensor-product setting.

## Pauli Matrices
The Pauli matrices are
$$
I=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix},
\qquad
X=
\begin{bmatrix}
0&1\\
1&0
\end{bmatrix},
$$
$$
Y=
\begin{bmatrix}
0&-i\\
i&0
\end{bmatrix},
\qquad
Z=
\begin{bmatrix}
1&0\\
0&-1
\end{bmatrix}.
$$

Their commutator structure is
$$
[\sigma_i,\sigma_j]=2i\,\varepsilon_{ijk}\sigma_k.
$$

Spin-$\tfrac12$ observables are
$$
S_x=\frac{\hbar}{2}\sigma_x,\qquad
S_y=\frac{\hbar}{2}\sigma_y,\qquad
S_z=\frac{\hbar}{2}\sigma_z.
$$

- Pauli matrices form the basic operator basis for one-qubit systems.
- They encode the Lie algebraic structure behind qubit rotations.

## Quantum Gates
Quantum gates are unitary operators.

Hadamard gate:
$$
H=\frac{1}{\sqrt2}
\begin{bmatrix}
1&1\\
1&-1
\end{bmatrix}
$$
with
$$
|0\rangle \mapsto \frac{|0\rangle+|1\rangle}{\sqrt2},
\qquad
|1\rangle \mapsto \frac{|0\rangle-|1\rangle}{\sqrt2}.
$$

Phase gates:
$$
S=
\begin{bmatrix}
1&0\\
0&i
\end{bmatrix},
\qquad
T=
\begin{bmatrix}
1&0\\
0&e^{i\pi/4}
\end{bmatrix}.
$$

Multi-qubit gates include CNOT, controlled phase, SWAP, and Toffoli.

- Single-qubit gates change local amplitudes and phases.
- CNOT supplies nontrivial two-qubit interaction.
- Single-qubit gates together with CNOT form a universal gate set.

### Figure — Bloch picture of $H|0\rangle$ and the Pauli axes
The Bloch vector of $|0\rangle$ is $+Z$. The Hadamard gate sends it to the $+X$ axis (the $|+\rangle$ state). The colored arcs suggest infinitesimal rotations generated by $\sigma_x,\sigma_y,\sigma_z$ (Bloch–sphere picture of one-qubit unitaries).

```python {run}
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def arc_on_sphere(axis, t0, t1, n=80):
    t = np.linspace(t0, t1, n)
    if axis == "x":
        return np.column_stack([np.zeros_like(t), np.cos(t), np.sin(t)])
    if axis == "y":
        return np.column_stack([np.cos(t), np.zeros_like(t), np.sin(t)])
    return np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])

u = np.linspace(0, 2 * np.pi, 40)
v = np.linspace(0, np.pi, 20)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones(np.size(u)), np.cos(v))

fig = plt.figure(figsize=(7.2, 6.2), facecolor="#080a0c")
ax = fig.add_subplot(111, projection="3d", facecolor="#0e1218")
ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, color="#141c28", edgecolor="#2a3448", linewidth=0.12, alpha=0.55)

ax.plot([0, 0], [0, 0], [0, 1], color="#c9a961", lw=2.6, label=r"$|0\rangle$ $(+Z)$")
ax.plot([0, 1], [0, 0], [0, 0], color="#e8c066", lw=2.6, label=r"$H|0\rangle=|+\rangle$ $(+X)$")

arc_x = arc_on_sphere("x", 0, np.pi / 2)
arc_y = arc_on_sphere("y", 0, np.pi / 2)
arc_z = arc_on_sphere("z", 0, np.pi / 2)
ax.plot(arc_x[:, 0], arc_x[:, 1], arc_x[:, 2], color="#6b9ac4", lw=1.8, alpha=0.9)
ax.plot(arc_y[:, 0], arc_y[:, 1], arc_y[:, 2], color="#97c4a0", lw=1.8, alpha=0.9)
ax.plot(arc_z[:, 0], arc_z[:, 1], arc_z[:, 2], color="#d4a5a5", lw=1.8, alpha=0.9)

for v0, lab in [([1.12, 0, 0], "X"), ([0, 1.12, 0], "Y"), ([0, 0, 1.12], "Z")]:
    ax.plot([0, v0[0]], [0, v0[1]], [0, v0[2]], color="#3a4558", lw=1)
    ax.text(v0[0], v0[1], v0[2], lab, color="#a8b0c0", fontsize=10)

ax.set_box_aspect([1, 1, 1])
ax.grid(False)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False
    axis.pane.set_edgecolor("#1a1e28")
ax.view_init(elev=16, azim=38)
ax.set_title(r"Hadamard as a rotation on $S^2$ (schematic)", color="#f0f4f8", fontsize=12, pad=12)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=True, facecolor="#12161c", edgecolor="#3a4558", labelcolor="#d0d8e4", fontsize=8)
plt.tight_layout()
plt.show()
```

---

# 4. Quantum Information Protocols

## Superdense Coding
Using a shared Bell state
$$
|\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt2},
$$
one party applies one of
$$
I,\; Z,\; X,\; iY
$$
to encode two classical bits.

This produces four orthogonal Bell-type states:
$$
\psi_0=\frac{|00\rangle+|11\rangle}{\sqrt2},
\qquad
\psi_1=\frac{|00\rangle-|11\rangle}{\sqrt2},
$$
$$
\psi_2=\frac{|10\rangle+|01\rangle}{\sqrt2},
\qquad
\psi_3=\frac{|01\rangle-|10\rangle}{\sqrt2}.
$$

- Entanglement allows one transmitted qubit to carry two classical bits of information.

### Figure — mutual orthogonality of the four Bell-type states
Let the computational basis of $\mathbb C^4$ be $(|00\rangle,|01\rangle,|10\rangle,|11\rangle)^T$. The four encoding states used above are orthogonal; the heatmap shows $|\langle\psi_i|\psi_j\rangle|^2$ (identity).

```python {run}
import numpy as np
import matplotlib.pyplot as plt

psi0 = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
psi1 = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
psi2 = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
psi3 = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
B = np.column_stack([psi0, psi1, psi2, psi3])
gram = np.abs(np.conj(B).T @ B) ** 2

fig, ax = plt.subplots(figsize=(5.8, 5), facecolor="#080a0c")
ax.set_facecolor("#0e1218")
im = ax.imshow(gram, cmap="viridis", vmin=0, vmax=1, aspect="equal")
labels = [r"$\psi_0$", r"$\psi_1$", r"$\psi_2$", r"$\psi_3$"]
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(labels, color="#b8c4d0", fontsize=10)
ax.set_yticklabels(labels, color="#b8c4d0", fontsize=10)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{gram[i, j]:.2f}", ha="center", va="center", color="#f0f0f0" if gram[i, j] > 0.5 else "#8899aa", fontsize=10)
ax.set_title(r"$|\langle\psi_i|\psi_j\rangle|^2$ for superdense encodings", color="#f0f4f8", fontsize=11)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(colors="#8899aa")
plt.tight_layout()
plt.show()
```

## Quantum Teleportation
Given an unknown state
$$
|\phi\rangle=a|0\rangle+b|1\rangle
$$
and a shared Bell pair
$$
|\Phi^+\rangle=\frac{|00\rangle+|11\rangle}{\sqrt2},
$$
the total initial state is
$$
|\phi\rangle\otimes |\Phi^+\rangle.
$$

Protocol:
1. apply CNOT,
2. apply Hadamard,
3. measure two qubits,
4. send two classical bits,
5. apply one of
$$
I,\; X,\; Z,\; XZ
$$
to recover the original state.

- Teleportation transfers an unknown state using entanglement plus classical communication.
- The quantum state itself is not copied or physically sent as a classical description.

---

# 5. Global Phase, Projective Space, and the Bloch Sphere

## Global Phase
Two normalized vectors differing by a global phase represent the same physical state:
$$
|\psi\rangle \sim e^{i\alpha}|\psi\rangle.
$$

This is because probabilities are unchanged:
$$
|\langle \chi|e^{i\alpha}\psi\rangle|^2
=
|e^{i\alpha}\langle \chi|\psi\rangle|^2
=
|\langle \chi|\psi\rangle|^2.
$$

- Physical states are rays, not individual vectors.

## Projective Hilbert Space
If nonzero vectors are identified up to nonzero scalar multiples,
$$
z\sim \lambda z \qquad (\lambda\ne 0),
$$
the resulting space is projective space.

For qubits, physical pure states live in
$$
\mathbb CP^1.
$$

In homogeneous coordinates:
$$
[z_0:z_1]=[\lambda z_0:\lambda z_1].
$$

- Projective geometry removes the physically irrelevant overall phase.

## Bloch Sphere
A normalized qubit state can be written as
$$
|\psi\rangle = z|0\rangle + w|1\rangle,
\qquad
|z|^2+|w|^2=1.
$$

Normalized vectors form $S^3$, and modding out by $U(1)$ phase gives
$$
S^3/U(1)\cong \mathbb CP^1\cong S^2.
$$

- The Bloch sphere is the geometric space of pure one-qubit states.
- Relative phase survives; global phase does not.

### Figure — Bloch sphere as $\mathbb{C}P^1$
Stereographic coordinates identify the north pole with $|0\rangle$; states approaching the equator carry nontrivial relative phase between $|0\rangle$ and $|1\rangle$. The curve is a latitude circle (fixed $\theta$) illustrating that **relative** phase is visible on $S^2$ even though **global** phase is not.

```python {run}
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

u = np.linspace(0, 2 * np.pi, 56)
v = np.linspace(0, np.pi, 28)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones(np.size(u)), np.cos(v))

theta = np.deg2rad(52)
phi_ring = np.linspace(0, 2 * np.pi, 200)
ring = np.column_stack(
    [
        np.sin(theta) * np.cos(phi_ring),
        np.sin(theta) * np.sin(phi_ring),
        np.cos(theta) * np.ones_like(phi_ring),
    ]
)
cmap = plt.cm.inferno
cols = cmap(np.linspace(0.15, 0.92, len(phi_ring) - 1))

fig = plt.figure(figsize=(7.4, 6.4), facecolor="#080a0c")
ax = fig.add_subplot(111, projection="3d", facecolor="#0e1218")
ax.plot_surface(
    xs,
    ys,
    zs,
    rstride=1,
    cstride=1,
    color="#1a2230",
    edgecolor="#2f3d52",
    linewidth=0.12,
    alpha=0.62,
    shade=True,
)
ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#8899aa", lw=1.1, alpha=0.35)
for i in range(len(phi_ring) - 1):
    ax.plot(
        ring[i : i + 2, 0],
        ring[i : i + 2, 1],
        ring[i : i + 2, 2],
        color=cols[i],
        lw=2.6,
        solid_capstyle="round",
    )
ax.scatter([0], [0], [1], s=95, c="#5eb0ff", edgecolors="#c8e4ff", linewidths=1.3, zorder=15, label=r"$|0\rangle$ (north pole)")
ax.scatter([0], [0], [-1], s=95, c="#ffd080", edgecolors="#c9a961", linewidths=1.3, zorder=15, label=r"$|1\rangle$ (south pole)")

theta_eq = np.pi / 2
phi_m = np.linspace(0, 2 * np.pi, 120)
ax.plot(np.cos(phi_m), np.sin(phi_m), np.zeros_like(phi_m), color="#6a7a90", ls="--", lw=1.2, alpha=0.85)
phi_m2 = np.linspace(-np.pi / 2, np.pi / 2, 80)
ax.plot(np.cos(theta_eq) * np.cos(phi_m2), np.sin(theta_eq) * np.cos(phi_m2), np.sin(theta_eq) * np.sin(phi_m2), color="#5a6a80", ls=":", lw=1.1, alpha=0.75)

for v0, lab in [([1.12, 0, 0], "X"), ([0, 1.12, 0], "Y"), ([0, 0, 1.12], "Z")]:
    ax.plot([0, v0[0]], [0, v0[1]], [0, v0[2]], color="#3a4558", lw=1.1)
    ax.text(v0[0], v0[1], v0[2], lab, color="#a8b0c0", fontsize=10)

ax.set_box_aspect([1, 1, 1])
ax.grid(False)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.fill = False
    axis.pane.set_edgecolor("#1a1e28")
    axis.line.set_color("#2a3040")
ax.view_init(elev=18, azim=32)
ax.set_title(r"Bloch sphere: relative phase visible as geometry on $S^2$", color="#f0f4f8", fontsize=11, pad=14)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=True, facecolor="#12161c", edgecolor="#3a4558", labelcolor="#d0d8e4", fontsize=8)
plt.tight_layout()
plt.show()
```

---

# 6. Geometry of Qubits: Hopf Fibration, Quaternions, and Rotations

## Hopf Fibration
The qubit state geometry fits into
$$
S^1 \longrightarrow S^3 \longrightarrow S^2.
$$

- $S^3$ = normalized qubit vectors,
- $S^1$ = global phase fiber,
- $S^2$ = physical pure states.

This is the Hopf fibration:
$$
S^3/U(1)\cong \mathbb CP^1\cong S^2.
$$

## Projective Unitary Symmetry
Since global phase is physically irrelevant, state symmetry is naturally projective:
$$
PU(n)=U(n)/U(1).
$$

Likewise,
$$
PSU(n)=SU(n)/Z_n,
$$
where
$$
Z_n=\{\xi_n^k I_n : k=0,\dots,n-1\},
\qquad
\xi_n=e^{2\pi i/n}.
$$

- Projective unitary groups describe physical transformations up to global phase.

## Quaternions and $SU(2)$
A quaternion has the form
$$
q=a+bi+cj+dk,
$$
with
$$
i^2=j^2=k^2=-1,\qquad ij=k
$$
and cyclic relations.

Its norm is
$$
|q|^2=a^2+b^2+c^2+d^2.
$$

Unit quaternions form
$$
S^3\subset \mathbb R^4.
$$

The key identification is
$$
SU(2)\cong S^3.
$$

For imaginary quaternions identified with $\mathbb R^3$,
$$
vw=-v\cdot w + v\times w.
$$

- Quaternion multiplication packages 3D dot and cross products into one algebraic rule.
- Unit quaternions give a concrete model for $SU(2)$.

## Double Cover of $SO(3)$
A unit quaternion acts on imaginary quaternions by conjugation:
$$
v\mapsto qvq^{-1}.
$$

This gives a group homomorphism
$$
SU(2)\to SO(3)
$$
with kernel
$$
\{\pm 1\}\cong \mathbb Z_2.
$$

Hence
$$
SO(3)\cong SU(2)/\{\pm I\}.
$$

- $SU(2)$ is the double cover of the 3D rotation group.
- This is the geometric origin of spin-$\tfrac12$ behavior.

---

# 7. Lie Groups, Lie Algebras, and the Exponential Map

## Lie Groups
A Lie group is both a smooth manifold and a group such that multiplication and inversion are smooth:
$$
\mu:G\times G\to G,
\qquad
i:G\to G,\quad i(g)=g^{-1}.
$$

Main examples:
$$
U(n),\ SU(n),\ GL_n,\ SO(n).
$$

- Lie groups encode continuous symmetries.

## Tangent Space and Lie Algebra
The tangent space at the identity is the Lie algebra:
$$
\mathfrak g = T_eG.
$$

A Lie algebra is a vector space with bracket
$$
[\cdot,\cdot]:\mathfrak g\times\mathfrak g\to\mathfrak g
$$
satisfying
$$
[x,y]=-[y,x]
$$
and
$$
[x,[y,z]]+[y,[z,x]]+[z,[x,y]]=0.
$$

For associative algebras, the commutator
$$
[x,y]=xy-yx
$$
defines a Lie bracket.

- Lie algebras describe infinitesimal symmetry.

## Matrix Lie Algebra Examples
For $SO(n)$, differentiating
$$
A^TA=I
$$
gives
$$
\mathfrak{so}(n)=\{X:X^T=-X\}.
$$

For $SL_n$,
$$
\mathfrak{sl}_n=\{X:\operatorname{tr}(X)=0\}.
$$

For $SU(n)$, the Lie algebra consists of traceless skew-Hermitian matrices.

- Matrix Lie algebras are obtained by linearizing the group constraints.

## Exponential Map
The matrix exponential is
$$
\exp(A)=\sum_{n\ge 0}\frac{A^n}{n!}.
$$

It defines
$$
\exp:\mathfrak g\to G.
$$

If $[A,B]=0$, then
$$
\exp(A+B)=\exp(A)\exp(B).
$$

One-parameter subgroups are given by
$$
\gamma(t)=\exp(tA).
$$

- The exponential map turns infinitesimal generators into finite transformations.

---

# 8. Adjoint Action, Invariant Fields, and Core Lie Algebras

## Left-Invariant Vector Fields
For $x\in T_eG=\mathfrak g$, define a left-invariant vector field by
$$
X(g)=(\lambda_g)_e(x),
$$
where
$$
\lambda_g(h)=gh.
$$

This gives the correspondence
$$
\mathfrak g
\cong
\{\text{left-invariant vector fields on }G\}.
$$

- A Lie algebra element extends canonically to a vector field on the whole group.

## Conjugation and Adjoint Representation
Conjugation is
$$
C_g(h)=ghg^{-1}.
$$

Differentiating at the identity gives
$$
\operatorname{Ad}(g):\mathfrak g\to\mathfrak g.
$$

For matrix groups,
$$
\operatorname{Ad}(g)(X)=gXg^{-1}.
$$

Differentiating once more gives
$$
ad:\mathfrak g\to \operatorname{End}(\mathfrak g),
\qquad
ad(x)(y)=[x,y].
$$

- The Lie bracket is the infinitesimal version of conjugation.

## Important Lie Algebras
General linear:
$$
\mathfrak{gl}_n = M_n
$$

Special linear:
$$
\mathfrak{sl}_n=\{A:\operatorname{tr}(A)=0\}
$$

Orthogonal:
$$
\mathfrak{so}(n)=\{A:A^T=-A\}
$$

For $\mathfrak{sl}(2,\mathbb C)$, a standard basis is
$$
H=
\begin{bmatrix}
1&0\\
0&-1
\end{bmatrix},
\qquad
E=
\begin{bmatrix}
0&1\\
0&0
\end{bmatrix},
\qquad
F=
\begin{bmatrix}
0&0\\
1&0
\end{bmatrix},
$$
with
$$
[H,E]=2E,\qquad [H,F]=-2F,\qquad [E,F]=H.
$$

- $\mathfrak{so}(3)$, $\mathfrak{su}(2)$, and $\mathfrak{sl}(2,\mathbb C)$ are the core algebras behind spin and qubit symmetry.

---

# 9. Representation Theory of $\mathfrak{sl}(2)$ and Quantum Spin

## Representations
A group representation is a homomorphism
$$
\rho:G\to \operatorname{Aut}(V).
$$

A Lie algebra representation is
$$
\rho:\mathfrak g\to \operatorname{End}(V)
$$
satisfying
$$
\rho([x,y])=[\rho(x),\rho(y)].
$$

- Representation theory turns abstract symmetry into linear operators on vector spaces.

## Irreducible Representations of $\mathfrak{sl}(2,\mathbb C)$
Finite-dimensional irreducible representations are classified by highest weight.

For each $m\in\mathbb N_0$, there is a unique irreducible representation $V_m$ with
$$
\dim V_m = m+1.
$$

Choose a highest weight vector $v_0$ with
$$
Hv_0 = mv_0,
\qquad
Ev_0=0,
$$
and generate
$$
v_k = F^k v_0.
$$

The weights are
$$
m,\; m-2,\; m-4,\;\dots,\;-m.
$$

- Every finite-dimensional irreducible $\mathfrak{sl}(2)$-representation is determined by one highest weight.

## Spin Notation
In physics one writes
$$
j=\frac{m}{2},
\qquad
\dim V_m = 2j+1.
$$

Basis vectors are
$$
|j,m\rangle,
\qquad
m=-j,-j+1,\dots,j.
$$

They satisfy
$$
J^2|j,m\rangle = \hbar^2 j(j+1)|j,m\rangle,
$$
$$
J_z|j,m\rangle = \hbar m |j,m\rangle.
$$

- $j$ labels the irreducible representation.
- $m$ labels the weight / magnetic quantum number.

## Ladder Operators
Define
$$
J_\pm = J_x \pm iJ_y.
$$

Their action is
$$
J_\pm |j,m\rangle
=
\hbar\,\sqrt{j(j+1)-m(m\pm1)}\,|j,m\pm 1\rangle.
$$

- $J_+$ raises $m$,
- $J_-$ lowers $m$.

These formulas determine the normalization of the standard spin basis.

### Figure — ladder moves on the $j=1$ multiplet
For $j=1$, the matrix elements $\langle 1,m\pm1|J_\pm|1,m\rangle$ pick up factors $\hbar\sqrt{2}$ along the allowed steps. The diagram is a bookkeeping picture of the same algebra as the square-root formula above.

```python {run}
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(8.2, 4.2), facecolor="#080a0c")
ax.set_facecolor("#0e1218")
ax.set_xlim(-0.2, 1.2)
ax.set_ylim(-0.35, 1.05)
ax.axis("off")

levels = [(0.5, 0.85, r"$|1,1\rangle$"), (0.5, 0.5, r"$|1,0\rangle$"), (0.5, 0.15, r"$|1,-1\rangle$")]
for x, y, lab in levels:
    ax.plot([0.35, 0.65], [y, y], color="#c9a961", lw=2.4)
    ax.text(0.72, y, lab, va="center", color="#f0f4f8", fontsize=12)

arr_up1 = FancyArrowPatch(
    (0.5, 0.17),
    (0.5, 0.48),
    arrowstyle="-|>",
    mutation_scale=14,
    lw=1.8,
    color="#6b9ac4",
)
arr_up2 = FancyArrowPatch(
    (0.5, 0.52),
    (0.5, 0.83),
    arrowstyle="-|>",
    mutation_scale=14,
    lw=1.8,
    color="#6b9ac4",
)
arr_dn1 = FancyArrowPatch(
    (0.5, 0.83),
    (0.5, 0.52),
    arrowstyle="-|>",
    mutation_scale=14,
    lw=1.8,
    color="#d4a5a5",
)
arr_dn2 = FancyArrowPatch(
    (0.5, 0.48),
    (0.5,0.17),
    arrowstyle="-|>",
    mutation_scale=14,
    lw=1.8,
    color="#d4a5a5",
)
for a in (arr_up1, arr_up2, arr_dn1, arr_dn2):
    ax.add_patch(a)

ax.text(0.08, 0.32, r"$J_+$", color="#6b9ac4", fontsize=11)
ax.text(0.08, 0.68, r"$J_+$", color="#6b9ac4", fontsize=11)
ax.text(0.92, 0.32, r"$J_-$", color="#d4a5a5", fontsize=11)
ax.text(0.92, 0.68, r"$J_-$", color="#d4a5a5", fontsize=11)
ax.text(0.5, -0.08, r"Coefficients: $\hbar\sqrt{2}$ for each allowed step (here $j=1$)", ha="center", color="#8899aa", fontsize=10)
ax.set_title(r"Ladder operators on $|j,m\rangle$ (schematic)", color="#f0f4f8", fontsize=12, pad=8)
plt.tight_layout()
plt.show()
```

## Casimir / Total Angular Momentum
The total angular momentum operator is
$$
J^2=J_x^2+J_y^2+J_z^2.
$$

It commutes with all generators:
$$
[J^2,J_i]=0.
$$

- $J^2$ is a central invariant on irreducible representations.
- By Schur’s lemma, it acts as a scalar on each irreducible spin sector.

---

# 10. Tensor Products, Clebsch–Gordan, and Coupled Systems

## Tensor Product of Representations
If $\rho_1,\rho_2$ are group representations, then
$$
\rho(g)=\rho_1(g)\otimes \rho_2(g)
$$
is the tensor product representation.

For Lie algebras, the induced action is
$$
\rho(x)=\rho_1(x)\otimes I + I\otimes \rho_2(x).
$$

- On a tensor product, generators act on both factors and add.

## Clebsch–Gordan Decomposition
For $\mathfrak{sl}(2)$ / $SU(2)$,
$$
V_m\otimes V_n
\cong
V_{m+n}\oplus V_{m+n-2}\oplus\cdots\oplus V_{m-n}
\qquad (m\ge n).
$$

In physics notation,
$$
V_{j_1}\otimes V_{j_2}
\cong
\bigoplus_{J=|j_1-j_2|}^{j_1+j_2} V_J.
$$

- Coupling two spins produces all total spins between the minimum and maximum allowed values.

## Clebsch–Gordan Coefficients
The uncoupled basis is
$$
|j_1,m_1\rangle\otimes |j_2,m_2\rangle,
$$
while the coupled basis is
$$
|J,M\rangle.
$$

The basis change is
$$
|J,M\rangle
=
\sum_{m_1,m_2}
\langle j_1,j_2;m_1,m_2 \mid J,M\rangle\;
|j_1,m_1\rangle |j_2,m_2\rangle.
$$

- Clebsch–Gordan coefficients are the transition amplitudes between uncoupled and coupled spin bases.

## Wigner Symbols
Wigner $3j$ symbols repackage Clebsch–Gordan coefficients in a more symmetric form.

For three-fold recoupling, $6j$ symbols compare
$$
(V_{j_1}\otimes V_{j_2})\otimes V_{j_3}
\qquad\text{and}\qquad
V_{j_1}\otimes (V_{j_2}\otimes V_{j_3}).
$$

- Tensor products are associative abstractly, but basis changes between coupling schemes are nontrivial.

---

# 11. Bilinear Forms, Duality, and Tensor Constructions

## Tensor Product as a Universal Object
For any bilinear map
$$
\beta:V\times W\to Z,
$$
there exists a unique linear map
$$
\widetilde\beta:V\otimes W\to Z
$$
factoring it.

- Tensor product is the universal linearization of bilinear maps.

If $\{e_i\}$ and $\{f_j\}$ are bases, then
$$
\{e_i\otimes f_j\}
$$
is a basis of $V\otimes W$, so
$$
\dim(V\otimes W)=\dim V\cdot \dim W.
$$

## Dual Space and Internal Hom
The dual space is
$$
V^*=\operatorname{Hom}(V,k).
$$

For a basis $\{e_i\}$, the dual basis $\{e^i\}$ satisfies
$$
e^i(e_j)=\delta^i_j.
$$

In finite dimensions,
$$
\operatorname{Hom}(V,W)\cong W\otimes V^*.
$$

- Linear maps can be viewed as tensors with one output index and one dual input index.

## Bilinear Forms and Metric Identification
A bilinear form is
$$
B:V\times V\to k.
$$

If nondegenerate, it gives an isomorphism
$$
V\to V^*,
\qquad
v\mapsto B(v,-).
$$

- A nondegenerate bilinear form identifies vectors with covectors.
- This underlies adjoints, contractions, and metric-dependent constructions.

## Casimir Element
Given a nondegenerate invariant bilinear form with inverse coefficients $g^{ij}$, the Casimir element is schematically
$$
C=\sum_{i,j} g^{ij}\, e_i\otimes e_j.
$$

- Casimir elements produce central operators and representation invariants.

---

# 12. Exact Sequences, Projectors, and Algebraic Structure

## Kernel, Image, and Cokernel
For a linear map
$$
\varphi:V\to W,
$$
the kernel and image are
$$
\ker\varphi=\{v\in V:\varphi(v)=0\},
\qquad
\operatorname{im}\varphi=\{\varphi(v):v\in V\}.
$$

The cokernel is
$$
\operatorname{coker}\varphi = W/\operatorname{im}\varphi.
$$

Rank–nullity:
$$
\dim V = \dim(\ker\varphi)+\dim(\operatorname{im}\varphi).
$$

- Kernel captures what maps to zero.
- Cokernel captures what remains outside the image.

## Exact Sequences
A sequence is exact when image equals kernel at each stage.

Short exact sequence:
$$
0\to U\to V\to W\to 0.
$$

For vector spaces, it splits:
$$
V\cong U\oplus W.
$$

- The splitting need not be canonical.
- Exactness organizes subspaces, quotients, and decomposition.

## Projectors
A projector satisfies
$$
p^2=p.
$$

Then
$$
V=\ker p \oplus \operatorname{im} p.
$$

- Idempotents correspond to direct-sum splittings.
- Decomposition and projection are two views of the same structure.

## Universal Enveloping Algebra and Hopf Structure
The universal enveloping algebra $U(\mathfrak g)$ carries a coproduct
$$
\Delta(x)=1\otimes x + x\otimes 1
\qquad (x\in\mathfrak g)
$$
and antipode
$$
S(x)=-x.
$$

- This is why Lie algebra actions extend naturally to tensor products.
- The Hopf viewpoint packages symmetry actions on composite systems.

---

# 13. Qudits, Spin Chains, and Category-Theoretic Language

## Qudits and Many-Body Systems
A single $d$-level quantum system has state space
$$
\mathbb C^d.
$$

An $n$-site chain has
$$
(\mathbb C^d)^{\otimes n}.
$$

For qubits,
$$
(\mathbb C^2)^{\otimes n}.
$$

- Multi-qubit systems are tensor powers of the fundamental qubit space.
- Spin chains are the representation-theoretic version of many-body qubit systems.

## Physical Gates and Projective Unitaries
Because global phase is unphysical, gates are naturally considered projectively:
$$
PU(d)=U(d)/U(1).
$$

- In computation one often uses representatives in $U(d)$ or $SU(d)$.
- Physically, only the projective action matters.

## Category Language
A category consists of:
- objects,
- morphisms,
- composition,
- identity morphisms.

The axioms are
$$
(f\circ g)\circ h = f\circ (g\circ h),
\qquad
f\circ \operatorname{id} = f = \operatorname{id}\circ f.
$$

- Categories abstract the structure of composable transformations.

## Functors and Natural Transformations
A functor
$$
F:\mathcal C\to\mathcal D
$$
preserves identities and composition:
$$
F(\operatorname{id}_X)=\operatorname{id}_{F(X)},
\qquad
F(f\circ g)=F(f)\circ F(g).
$$

A natural transformation $\eta:F\Rightarrow G$ is a compatible family of maps
$$
\eta_X:F(X)\to G(X)
$$
for each object $X$.

- Functors move entire structures between categories.
- Natural transformations compare functors coherently.

## Equivalence and Monoidal Structure
A functor is an equivalence if it is:
- full,
- faithful,
- essentially surjective.

Tensor product leads to monoidal categories, where associativity is controlled by coherent isomorphisms rather than strict equality.

- This language abstracts tensor product, duality, and composition beyond coordinates.
- It is the categorical shadow of quantum composition and symmetry.
