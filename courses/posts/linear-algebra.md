---
title: Linear Algebra
category: MATH
semester: 2023 S
---

# Linear Algebra

## 1. Systems of Linear Equations

A system of linear equations can be written as

$$
A\mathbf{x}=\mathbf{b}
$$

where

$$
A \in \mathbb{R}^{m \times n}, \qquad 
\mathbf{x} \in \mathbb{R}^{n}, \qquad 
\mathbf{b} \in \mathbb{R}^{m}
$$

The main question is whether the system has

- a unique solution,
- no solution,
- or infinitely many solutions.

A linear system is solved by transforming the augmented matrix into a simpler form using elimination.

$$
\left[ A \mid \mathbf{b} \right]
$$

---

## 2. Linear Filters and Matrices

A linear filter is a linear transformation that maps an input vector to an output vector.

$$
T(\mathbf{x}) = A\mathbf{x}
$$

Linearity means

$$
T(\mathbf{u}+\mathbf{v})=T(\mathbf{u})+T(\mathbf{v})
$$

and

$$
T(c\mathbf{u})=cT(\mathbf{u})
$$

Matrices are the standard representation of linear filters.

For example,

$$
A=
\begin{bmatrix}
2 & 1 \\
0 & 3
\end{bmatrix}
$$

acts on a vector by stretching, compressing, rotating, or shearing the plane.

```python {run}
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[2, 1],
              [0, 3]])

vectors = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [-1, 1]
])

transformed = vectors @ A.T

plt.figure(figsize=(6, 6))
for v in vectors:
    plt.arrow(0, 0, v[0], v[1], head_width=0.08, length_includes_head=True)
for v in transformed:
    plt.arrow(0, 0, v[0], v[1], head_width=0.08, length_includes_head=True)

plt.axhline(0)
plt.axvline(0)
plt.xlim(-3, 4)
plt.ylim(-3, 5)
plt.gca().set_aspect('equal')
plt.title("Original vectors and their images under a linear transformation")
plt.show()
```

---

## 3. Determinants and Inverses of $2 \times 2$ Matrices

For

$$
A=
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

the determinant is

$$
\det(A)=ad-bc
$$

The determinant measures whether the matrix is invertible and how area scales under the transformation.

- $\det(A)\neq 0$: invertible

- $\det(A)=0$: singular

If $\det(A)\neq 0$, then

$$
A^{-1}=
\frac{1}{ad-bc}
\begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}
$$

and

$$
A^{-1}A=AA^{-1}=I
$$

---

## 4. Linear Coordinate Systems

A vector can be expressed in different bases.

In the standard basis,

$$
\mathbf{x}=
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
=
x_1\mathbf{e}_1+x_2\mathbf{e}_2
$$

If $\mathcal{B}=\{\mathbf{v}_1,\mathbf{v}_2\}$ is another basis, then

$$
\mathbf{x}=c_1\mathbf{v}_1+c_2\mathbf{v}_2
$$

and its coordinate vector in that basis is

$$
[\mathbf{x}]_{\mathcal{B}}=
\begin{bmatrix}
c_1 \\
c_2
\end{bmatrix}
$$

If

$$
P=[\mathbf{v}_1 \ \mathbf{v}_2]
$$

then

$$
\mathbf{x}=P[\mathbf{x}]_{\mathcal{B}}
\qquad \text{and} \qquad
[\mathbf{x}]_{\mathcal{B}}=P^{-1}\mathbf{x}
$$

---

## 5. Diagonalization

A matrix is diagonalizable if it can be written as

$$
A=PDP^{-1}
$$

where

- $P$ contains eigenvectors,
- $D$ is diagonal with eigenvalues on the diagonal.

An eigenvector $\mathbf{v}\neq \mathbf{0}$ and eigenvalue $\lambda$ satisfy

$$
A\mathbf{v}=\lambda \mathbf{v}
$$

The eigenvalues are found from

$$
\det(A-\lambda I)=0
$$

### Why diagonalization matters

It simplifies powers of matrices:

$$
A^k=PD^kP^{-1}
$$

and since $D$ is diagonal,

$$
D^k=
\begin{bmatrix}
\lambda_1^k & 0 & \cdots & 0 \\
0 & \lambda_2^k & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \lambda_n^k
\end{bmatrix}
$$

### $2 \times 2$ filters

Repeated filtering is modeled by

$$
\mathbf{x}_{k+1}=A\mathbf{x}_k
$$

so

$$
\mathbf{x}_k=A^k\mathbf{x}_0
$$

```python {run}
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[3, 1],
              [1, 3]])

eigvals, eigvecs = np.linalg.eig(A)

grid_x = np.linspace(-2, 2, 9)
grid_y = np.linspace(-2, 2, 9)

plt.figure(figsize=(6, 6))

for x in grid_x:
    for y in grid_y:
        p = np.array([x, y])
        q = A @ p
        plt.plot([p[0], q[0]], [p[1], q[1]], alpha=0.25)

for i in range(2):
    v = eigvecs[:, i]
    plt.arrow(0, 0, v[0], v[1], head_width=0.08, length_includes_head=True)
    plt.arrow(0, 0, -v[0], -v[1], head_width=0.08, length_includes_head=True)

plt.axhline(0)
plt.axvline(0)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.gca().set_aspect('equal')
plt.title("Eigenvector directions under the transformation")
plt.show()
```

### Differential equations

For the system

$$
\frac{d\mathbf{x}}{dt}=A\mathbf{x}
$$

the solution is

$$
\mathbf{x}(t)=e^{At}\mathbf{x}(0)
$$

If $A=PDP^{-1}$, then

$$
e^{At}=Pe^{Dt}P^{-1}
$$

### Recurrence relations

Many recurrences can be rewritten as

$$
\mathbf{x}_{k+1}=A\mathbf{x}_k
$$

so the long-term behavior depends on eigenvalues and eigenvectors.

```python {run}
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0.85, 0.20],
              [-0.10, 0.95]])

x0 = np.array([2.0, 0.5])
points = [x0]

x = x0.copy()
for _ in range(20):
    x = A @ x
    points.append(x.copy())

points = np.array(points)

plt.figure(figsize=(6, 6))
plt.plot(points[:, 0], points[:, 1], marker='o')
plt.axhline(0)
plt.axvline(0)
plt.gca().set_aspect('equal')
plt.title(r"Iterated system: $x_{k+1} = Ax_k$")
plt.show()
```

### Higher-dimensional examples

The same idea extends to $3\times 3$ and $4\times 4$ matrices:

1. find eigenvalues,
2. find eigenvectors,
3. build $P$ and $D$,
4. use $A=PDP^{-1}$ when enough independent eigenvectors exist.

---

## 6. Solving $n \times n$ Systems, Inverses, and Determinants

For a square system

$$
A\mathbf{x}=\mathbf{b}, \qquad A \in \mathbb{R}^{n \times n}
$$

a unique solution exists when $A$ is invertible.

$$
\mathbf{x}=A^{-1}\mathbf{b}
$$

Invertibility is equivalent to

$$
\det(A)\neq 0
$$

The determinant also describes geometric scaling:

- in $\mathbb{R}^2$: area scaling,
- in $\mathbb{R}^3$: volume scaling.

---

## 7. Linearity and Independence

A transformation is linear if

$$
T(a\mathbf{u}+b\mathbf{v})=aT(\mathbf{u})+bT(\mathbf{v})
$$

Vectors $\mathbf{v}_1,\dots,\mathbf{v}_k$ are linearly independent if

$$
c_1\mathbf{v}_1+\cdots+c_k\mathbf{v}_k=\mathbf{0}
$$

implies

$$
c_1=\cdots=c_k=0
$$

If this is not true, the vectors are linearly dependent.

Linear independence means no vector is redundant.

---

## 8. Subspaces

A subset $W\subseteq \mathbb{R}^n$ is a subspace if

1. $\mathbf{0}\in W$
2. $\mathbf{u},\mathbf{v}\in W \Rightarrow \mathbf{u}+\mathbf{v}\in W$
3. $c\in\mathbb{R}, \mathbf{u}\in W \Rightarrow c\mathbf{u}\in W$

Important examples include

- null spaces,
- column spaces,
- lines through the origin,
- planes through the origin.

---

## 9. $m \times n$ Systems of Equations

For

$$
A\mathbf{x}=\mathbf{b}, \qquad A\in\mathbb{R}^{m\times n}
$$

the geometry depends on the relationship between equations and unknowns.

- $m>n$: overdetermined
- $m<n$: underdetermined
- $m=n$: square system

The number of pivot columns is the rank:

$$
\operatorname{rank}(A)
$$

The number of free variables is

$$
n-\operatorname{rank}(A)
$$

A solution exists only if $\mathbf{b}$ lies in the column space of $A$.

---

## 10. The Range of a Filter

For a linear filter

$$
T(\mathbf{x})=A\mathbf{x}
$$

the range is the set of all possible outputs:

$$
\operatorname{Range}(T)=\{A\mathbf{x}:\mathbf{x}\in\mathbb{R}^n\}
$$

This is exactly the column space of $A$:

$$
\operatorname{Range}(A)=\operatorname{Col}(A)
$$

So the system

$$
A\mathbf{x}=\mathbf{b}
$$

is solvable if and only if

$$
\mathbf{b}\in\operatorname{Range}(A)
$$

---

## Core Formulas

$$
A\mathbf{x}=\mathbf{b}
$$

$$
\det
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
=ad-bc
$$

$$
A^{-1}=
\frac{1}{ad-bc}
\begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}
$$

$$
A\mathbf{v}=\lambda \mathbf{v}
$$

$$
\det(A-\lambda I)=0
$$

$$
A=PDP^{-1}
$$

$$
A^k=PD^kP^{-1}
$$

$$
\frac{d\mathbf{x}}{dt}=A\mathbf{x}
\quad \Rightarrow \quad
\mathbf{x}(t)=e^{At}\mathbf{x}(0)
$$

$$
\operatorname{Range}(A)=\operatorname{Col}(A)
$$
