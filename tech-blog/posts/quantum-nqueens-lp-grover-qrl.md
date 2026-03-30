---
title: Quantum N-Queens as LP, Grover Search, and QRL
category: ALGORITHM
date: 2026.03.29
readtime: 28 min
---

This note connects a **continuous (relaxation) N-Queens** formulation as a linear program, a **Grover-style** view of searching good queen moves with an oracle, and a **variational quantum** sketch for Q-learning on a chess-like MDP.

# 1. Quantum N-Queen problem

We interpret an N-Queen variant as a **continuous probabilistic assignment** on an $N \times N$ board: each cell $(i,j)$ holds a fraction $x_{ij} \in [0,1]$ of a queen. This generalizes the classic discrete problem to a **superposition-like** picture where constraints bound cumulative queen “mass” per row, column, and diagonal.

## 1.1 Problem formulation as a linear program (LP)

### 1.1.1 Variables

Decision variables:

$$
\mathbf{x} = \{x_{ij} \mid 0 \leq i,j < N\}, \quad x_{ij} \in [0,1].
$$

Flatten to $\mathbf{x} \in \mathbb{R}^n$ with $n = N^2$:

$$
\mathbf{x} = [x_0, x_1, \dots, x_{n-1}]^\top, \quad x_{k} = x_{\lfloor k/N \rfloor,\, k \bmod N}.
$$

### 1.1.2 Constraints

**(i) Row constraints**

$$
\sum_{j=0}^{N-1} x_{ij} = 1 \quad \forall i \in [0,N).
$$

In code, split for numerical stability:

$$
\sum_j x_{ij} \leq 1, \quad -\sum_j x_{ij} \leq -1.
$$

**(ii) Column constraints**

$$
\sum_{i=0}^{N-1} x_{ij} = 1 \quad \forall j \in [0,N),
$$

and similarly:

$$
\sum_i x_{ij} \leq 1, \quad -\sum_i x_{ij} \leq -1.
$$

**(iii) Diagonal constraints ($\leq 1$)**

For each diagonal $d$:

$$
\sum_{(i,j) \in d} x_{ij} \leq 1.
$$

Cover all ↘ and ↙ diagonals: $2N-1$ per direction, **$4N-2$** inequalities in total.

**(iv) Fixed cells**

If $B_{ij} \in [0,1]$ is prescribed, then $x_{ij} = B_{ij}$, i.e. $x_k = B_{ij}$ for $k = i \cdot N + j$. Fixed values are absorbed into the RHS of other constraints in implementation.

### 1.1.3 Objective function

The model is **feasibility-first**:

$$
\text{maximize } 0 \quad \text{(equivalently minimize } 0\text{)}.
$$

Dummy objective in code:

```cpp
vector<double> c(size_total, 0.0); // zero objective vector
```

## 1.2 Simplex tableau structure

Solve the LP with a **two-phase simplex** method. The tableau is a matrix

$$
D \in \mathbb{R}^{(m + 2) \times (n + 2)},
$$

where $m$ is the number of constraints, $n$ the number of decision variables; the last two rows hold the true objective and the **Phase 1** artificial objective; the last column is the RHS. Slack and artificial variables augment the system.

**Symbolic layout:**

$$
\begin{bmatrix}
A & I & \mathbf{b} \\
-\mathbf{c}^{\top} & \mathbf{0}^{\top} & 0 \\
-\mathbf{e}^{\top} & \mathbf{0}^{\top} & -s
\end{bmatrix}
$$

($A$: constraint matrix; $I$: slack block; $\mathbf{c}$: objective; $\mathbf{b}$: RHS; $\mathbf{e}$: artificial objective row in Phase 1; the third row is optional / Phase 1.)

**Implementation sketch:**

```cpp
container_d.assign(m + 2, vector<double>(n + 2, 0.0));
```

Each constraint row:

```cpp
for (int i = 0; i < m; ++i) {
    container_b[i] = n + i;
    container_d[i][n] = -1.0;       // artificial variable
    container_d[i][n + 1] = b[i];   // RHS
}
```

Phase 1 uses the artificial objective row; the real objective lives in row `m`:

```cpp
container_d[m][j] = -c[j]; // j = 0 .. n-1
```

## 1.3 Pivoting operation

Pivot at $(r,s)$: normalize row $r$, then eliminate other rows.

$$
D_{rj} \leftarrow \frac{D_{rj}}{D_{rs}}, \qquad
D_{ij} \leftarrow D_{ij} - D_{is} \cdot \frac{D_{rj}}{D_{rs}} \quad (i \neq r).
$$

```cpp
double inv = 1.0 / container_d[r][s];
for (int i = 0; i < m + 2; ++i) {
    if (i != r && fabs(container_d[i][s]) > EPS) {
        double inv2 = container_d[i][s] * inv;
        for (int j = 0; j < n + 2; ++j)
            container_d[i][j] -= container_d[r][j] * inv2;
        container_d[i][s] = container_d[r][s] * inv2;
    }
}
```

## 1.4 Simplex iteration per phase

**Entering variable:** choose column $s$ with the most negative coefficient in the active objective row:

$$
s = \arg\min_j \{ D_{xj} \mid D_{xj} < 0 \}.
$$

**Leaving variable:** ratio test $\theta_i = b_i / a_{is}$, minimize over feasible $i$.

```cpp
for (int j = 0; j <= n; ++j) {
    if (container_n[j] != -phase) {
        if (s == -1 || make_pair(container_d[x][j], container_n[j]) < make_pair(container_d[x][s], container_n[s]))
            s = j;
    }
}
```

## 1.5 Feasibility test (Phase 1)

Minimize the artificial objective; if the optimum is $> \varepsilon$, the original LP is **infeasible**.

```cpp
if (!simplex(2) || container_d[m + 1][n + 1] < -EPS)
    return -INF; // infeasible
```

## 1.6 Solution extraction

For basic variables in $B$: $x_i = b_i$; nonbasic variables are zero.

```cpp
for (int i = 0; i < m; ++i)
    if (container_b[i] < n)
        x[container_b[i]] = container_d[i][n + 1];
```

## 1.7 Output tolerance

Numerical checks:

- Row and column sums in $[1 - 10^{-6},\, 1 + 10^{-6}]$.
- Each diagonal sum $\leq 1 + 10^{-6}$.
- Fixed cells: $|x_{ij} - B_{ij}| < 10^{-9}$.

```cpp
if (fabs(e) <= EPS) e = 0.0;
cout << fixed << setprecision(9) << e;
```

# 2. Grover's algorithm for optimal queen move

## 2.1 Problem setup

Let $\mathcal{A} = \{a_1, a_2, \dots, a_n\}$ be all **legal queen moves** from state $s$. Each move $a_i$ has utility $f(a_i) \in \{0,1\}$ under an oracle: $1$ means “optimal” (or target) for the current decision goal.

We seek $a^* \in \mathcal{A}$ with $f(a^*) = 1$.

## 2.2 Quantum state preparation

Uniform superposition over moves:

$$
|\psi_0\rangle = \frac{1}{\sqrt{n}} \sum_{i=1}^{n} |a_i\rangle.
$$

## 2.3 Oracle with backtracking

Phase oracle $O_f$:

$$
O_f |a_i\rangle = (-1)^{f(a_i)} |a_i\rangle.
$$

Mark $f(a_i) = 1$ when a **depth-limited backtracking / minimax** evaluation says the move is good enough. Let $V(s,a_i)$ be a classical value after search truncated at depth $d$:

$$
V(s, a_i) = \max_{\text{depth } \leq d} \mathrm{eval}(s'),
$$

$$
f(a_i) = \begin{cases}
1 & \text{if } V(s, a_i) \geq \tau, \\
0 & \text{otherwise.}
\end{cases}
$$

This is a **hybrid** picture: classical tree search defines the oracle; quantum amplitude amplification searches over labeled moves.

## 2.4 Grover iteration

$$
G = D O_f, \quad D = 2|\psi_0\rangle\langle\psi_0| - I.
$$

## 2.5 Number of iterations

With $k = |\{a_i : f(a_i)=1\}|$,

$$
R = \left\lfloor \frac{\pi}{4} \sqrt{\frac{n}{k}} \right\rfloor
$$

iterations (approximately) maximize success probability.

## 2.6 Measurement

$$
|\psi_R\rangle = G^R |\psi_0\rangle \;\Rightarrow\; \text{measure to obtain } a^*.
$$

## 2.7 Summary

Grover amplification with a **backtracking-defined** oracle ties tactical classical search to quantum query complexity for “find a good move among $n$.”

# 3. Quantum reinforcement learning with variational circuits

## 3.1 Problem definition

States $s \in \mathcal{S}$, actions $a \in \mathcal{A}$ (legal queen actions). Model an MDP with transition $P(s'|s,a)$, reward $R(s,a)$, discount $\gamma \in [0,1]$. Target:

$$
Q^*(s, a) = \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t r_t \,\middle|\, s_0=s,\, a_0=a \right].
$$

Learn $Q_\theta(s,a) \approx Q^*(s,a)$ with a **quantum-parametrized** ansatz.

## 3.2 Circuit-based Q-function approximation

**Input encoding** (angle encoding):

$$
|\phi(s)\rangle = \bigotimes_{i=1}^{d} \left( \cos(\theta_i(s))|0\rangle + \sin(\theta_i(s))|1\rangle \right), \quad \theta_i(s) = \alpha s_i.
$$

**Parametrized unitary:**

$$
|\psi_\theta(s)\rangle = U(\theta)\,|\phi(s)\rangle.
$$

**Q-value from observables:**

$$
Q_\theta(s,a) = \langle \psi_\theta(s) |\, O_a \,| \psi_\theta(s) \rangle,
$$

e.g. Pauli-$Z$ on a qubit tagged to action $a$.

## 3.3 Bellman error objective

TD loss:

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s')}\left[ \left( r + \gamma \max_{a'} Q_\theta(s',a') - Q_\theta(s,a) \right)^2 \right].
$$

**Parameter-shift** gradient (example):

$$
\frac{\partial Q_\theta(s,a)}{\partial \theta_i} \approx \tfrac{1}{2}\left[ Q_{\theta + \frac{\pi}{2} e_i}(s,a) - Q_{\theta - \frac{\pi}{2} e_i}(s,a) \right].
$$

## 3.4 Policy extraction

After training:

$$
a^* = \arg\max_a Q_\theta(s,a).
$$

This template extends to actor–critic and policy-gradient QRL; multi-agent variants follow the same MDP + VQC pattern at larger scale.
