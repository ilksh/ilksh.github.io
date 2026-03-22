---
title: Multivariable Calculus and Vector Analysis
category: MATH
semester: 2022 F
---

# Multivariable Calculus and Vector Analysis

## 1. The Geometry of Euclidean Space

Euclidean space extends familiar 2D geometry to $\mathbb{R}^n$.

A point in $\mathbb{R}^3$ is written as

$$
\mathbf{x} = (x,y,z)
$$

The dot product measures angle and projection:

$$
\mathbf{u}\cdot\mathbf{v} = \|\mathbf{u}\|\,\|\mathbf{v}\|\cos\theta
$$

The norm gives length:

$$
\|\mathbf{u}\|=\sqrt{\mathbf{u}\cdot\mathbf{u}}
$$

The cross product gives an oriented normal vector in $\mathbb{R}^3$:

$$
\mathbf{u}\times\mathbf{v}
=
\begin{bmatrix}
u_2v_3-u_3v_2 \\\\
u_3v_1-u_1v_3 \\\\
u_1v_2-u_2v_1
\end{bmatrix}
$$

Planes in $\mathbb{R}^3$ are described by

$$
ax+by+cz=d
$$

and lines can be written parametrically as

$$
\mathbf{r}(t)=\mathbf{r}_0+t\mathbf{v}
$$

---

## 2. Differentiation

For a scalar-valued function $f(x,y,z)$, the derivative in multivariable calculus is encoded by the gradient:

$$
\nabla f =
\begin{bmatrix}
\frac{\partial f}{\partial x} \\\\
\frac{\partial f}{\partial y} \\\\
\frac{\partial f}{\partial z}
\end{bmatrix}
$$

The directional derivative in the unit direction $\mathbf{u}$ is

$$
D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u}
$$

The gradient points in the direction of steepest ascent.

The tangent plane to $z=f(x,y)$ at $(x_0,y_0)$ is

$$
z \approx f(x_0,y_0)
+ f_x(x_0,y_0)(x-x_0)
+ f_y(x_0,y_0)(y-y_0)
$$

### Visualization — surface and tangent plane

```python {run}
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

x0, y0 = 0.8, -0.6
z0 = np.sin(x0) * np.cos(y0)
fx = np.cos(x0) * np.cos(y0)
fy = -np.sin(x0) * np.sin(y0)

Xt, Yt = np.meshgrid(np.linspace(x0 - 1, x0 + 1, 40),
                     np.linspace(y0 - 1, y0 + 1, 40))
Zt = z0 + fx * (Xt - x0) + fy * (Yt - y0)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.85)
ax.plot_surface(Xt, Yt, Zt, linewidth=0, alpha=0.65)
ax.scatter([x0], [y0], [z0], s=80)
ax.set_title("Surface and local tangent plane")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
```

---

## 3. Higher-Order Derivatives, Maxima, and Minima

### Iterated Partial Derivatives

Second-order partials include

$$
f_{xx}, \quad f_{xy}, \quad f_{yx}, \quad f_{yy}
$$

Under regularity conditions,

$$
f_{xy}=f_{yx}
$$

### Taylor's Theorem

A second-order Taylor approximation of $f(x,y)$ near $(a,b)$ is

$$
f(x,y)\approx f(a,b)
+ f_x(a,b)(x-a)+f_y(a,b)(y-b)
+ \frac{1}{2}
\begin{bmatrix}
x-a & y-b
\end{bmatrix}
H_f(a,b)
\begin{bmatrix}
x-a \\\\
y-b
\end{bmatrix}
$$

where $H_f$ is the Hessian matrix.

### Visualization — second-order Taylor approximation

```python {run}
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return np.exp(-(x**2 + y**2)) * (1 + 0.5 * x - 0.3 * y)

a, b = 0.4, -0.5
fa = f(a, b)

fx = (-2*a) * np.exp(-(a**2 + b**2)) * (1 + 0.5*a - 0.3*b) + 0.5*np.exp(-(a**2 + b**2))
fy = (-2*b) * np.exp(-(a**2 + b**2)) * (1 + 0.5*a - 0.3*b) - 0.3*np.exp(-(a**2 + b**2))

fxx = np.exp(-(a**2+b**2)) * ((4*a*a-2)*(1+0.5*a-0.3*b) - 2*a)
fyy = np.exp(-(a**2+b**2)) * ((4*b*b-2)*(1+0.5*a-0.3*b) + 1.2*b)
fxy = np.exp(-(a**2+b**2)) * (4*a*b*(1+0.5*a-0.3*b) + 0.6*a - b)

x = np.linspace(a - 1.2, a + 1.2, 180)
y = np.linspace(b - 1.2, b + 1.2, 180)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

dx = X - a
dy = Y - b
T2 = fa + fx*dx + fy*dy + 0.5*(fxx*dx**2 + 2*fxy*dx*dy + fyy*dy**2)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.82)
ax.plot_surface(X, Y, T2, linewidth=0, alpha=0.55)
ax.set_title("Function surface and second-order Taylor approximation")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
```

### Extrema of Real-Valued Functions

Critical points satisfy

$$
\nabla f = \mathbf{0}
$$

For functions of two variables, the second derivative test uses

$$
D=f_{xx}f_{yy}-f_{xy}^2
$$

- $D>0$ and $f_{xx}>0$: local minimum
- $D>0$ and $f_{xx}<0$: local maximum
- $D<0$: saddle point

### Constrained Extrema and Lagrange Multipliers

To optimize $f(x,y,z)$ subject to $g(x,y,z)=c$, solve

$$
\nabla f = \lambda \nabla g
$$

together with

$$
g(x,y,z)=c
$$

---

## 4. Vector-Valued Functions

A curve in space is represented by

$$
\mathbf{r}(t)=\langle x(t),y(t),z(t)\rangle
$$

Its derivatives describe motion:

$$
\mathbf{v}(t)=\mathbf{r}'(t), \qquad
\mathbf{a}(t)=\mathbf{r}''(t)
$$

### Acceleration and Newton's Second Law

Newton's second law is

$$
\mathbf{F}=m\mathbf{a}
$$

### Arc Length

The arc length of a curve is

$$
L=\int_a^b \|\mathbf{r}'(t)\|\,dt
$$

### Visualization — helix, velocity, and acceleration

```python {run}
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 6*np.pi, 500)
x = np.cos(t)
y = np.sin(t)
z = 0.15 * t

vx = -np.sin(t)
vy = np.cos(t)
vz = 0.15 * np.ones_like(t)

axv = -np.cos(t)
ayv = -np.sin(t)
azv = np.zeros_like(t)

k = 240

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, linewidth=2)
ax.quiver(x[k], y[k], z[k], vx[k], vy[k], vz[k], length=0.9, normalize=True)
ax.quiver(x[k], y[k], z[k], axv[k], ayv[k], azv[k], length=0.9, normalize=True)
ax.scatter([x[k]], [y[k]], [z[k]], s=70)
ax.set_title("Helix with velocity and acceleration vectors")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
```

### Vector Fields

A vector field in $\mathbb{R}^3$ has the form

$$
\mathbf{F}(x,y,z)=\langle P(x,y,z),Q(x,y,z),R(x,y,z)\rangle
$$

### Divergence and Curl

Divergence measures local expansion:

$$
\nabla\cdot\mathbf{F}
=
\frac{\partial P}{\partial x}
+
\frac{\partial Q}{\partial y}
+
\frac{\partial R}{\partial z}
$$

Curl measures local rotation:

$$
\nabla\times\mathbf{F}
=
\begin{bmatrix}
\frac{\partial R}{\partial y}-\frac{\partial Q}{\partial z} \\\\
\frac{\partial P}{\partial z}-\frac{\partial R}{\partial x} \\\\
\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}
\end{bmatrix}
$$

### Visualization — 3D vector field

```python {run}
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1.5, 1.5, 7)
y = np.linspace(-1.5, 1.5, 7)
z = np.linspace(-1.5, 1.5, 5)
X, Y, Z = np.meshgrid(x, y, z)

U = X - Y
V = X + Y
W = 0.8 * Z

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W, length=0.12, normalize=False)
ax.set_title("Vector field: expansion + rotation")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
```

---

## 5. Double and Triple Integrals

A double integral accumulates values over a planar region:

$$
\iint_D f(x,y)\,dA
$$

A triple integral accumulates over a spatial region:

$$
\iiint_E f(x,y,z)\,dV
$$

These can represent

- mass,
- area-weighted averages,
- volume,
- charge,
- probability.

For constant density,

$$
\iiint_E 1\,dV = \text{Volume}(E)
$$

### Visualization — region in the plane and height over it

```python {run}
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-1.2, 1.2, 220)
y = np.linspace(-1.2, 1.2, 220)
X, Y = np.meshgrid(x, y)

mask = X**2 + Y**2 <= 1
Z = 1 - X**2 - Y**2
Z[~mask] = np.nan

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.92)
ax.contour(X, Y, X**2 + Y**2, levels=[1], zdir='z', offset=-0.25)
ax.set_zlim(-0.25, 1.05)
ax.set_title("Surface above a circular integration region")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
```

---

## 6. The Change of Variables Formula

### The Geometry of Maps from $\mathbb{R}^2 \to \mathbb{R}^2$

A transformation

$$
T(u,v)=(x(u,v),y(u,v))
$$

deforms grids and regions.

The Jacobian matrix is

$$
DT=
\begin{bmatrix}
\frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\\\
\frac{\partial y}{\partial u} & \frac{\partial y}{\partial v}
\end{bmatrix}
$$

and the Jacobian determinant is

$$
J=\frac{\partial(x,y)}{\partial(u,v)}
$$

### Change of Variables Theorem

If $T$ is smooth and one-to-one, then

$$
\iint_D f(x,y)\,dA
=
\iint_{D^\ast} f(x(u,v),y(u,v))
\left|
\frac{\partial(x,y)}{\partial(u,v)}
\right|
\,du\,dv
$$

Likewise in three dimensions,

$$
\iiint_E f(x,y,z)\,dV
=
\iiint_{E^\ast} f(T(u,v,w))
\left|
\frac{\partial(x,y,z)}{\partial(u,v,w)}
\right|
\,du\,dv\,dw
$$

### Visualization — $(u,v)$ grid and its image in $(x,y)$

```python {run}
import numpy as np
import matplotlib.pyplot as plt

u = np.linspace(-1.5, 1.5, 16)
v = np.linspace(-1.5, 1.5, 16)

def T(u, v):
    x = u + 0.35 * u * v
    y = v + 0.25 * (u**2 - v**2)
    return x, y

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for c in u:
    vv = np.linspace(-1.5, 1.5, 300)
    uu = np.full_like(vv, c)
    axes[0].plot(uu, vv, linewidth=1)
    x, y = T(uu, vv)
    axes[1].plot(x, y, linewidth=1)

for c in v:
    uu = np.linspace(-1.5, 1.5, 300)
    vv = np.full_like(uu, c)
    axes[0].plot(uu, vv, linewidth=1)
    x, y = T(uu, vv)
    axes[1].plot(x, y, linewidth=1)

axes[0].set_title("Original grid in $(u,v)$")
axes[1].set_title("Deformed grid in $(x,y)$")
for ax in axes:
    ax.axhline(0)
    ax.axvline(0)
    ax.set_aspect('equal')
plt.show()
```

---

## 7. Integrals over Paths and Surfaces

### Path / Line Integrals

For a scalar field,

$$
\int_C f\,ds
=
\int_a^b f(\mathbf{r}(t))\|\mathbf{r}'(t)\|\,dt
$$

For a vector field,

$$
\int_C \mathbf{F}\cdot d\mathbf{r}
=
\int_a^b \mathbf{F}(\mathbf{r}(t))\cdot \mathbf{r}'(t)\,dt
$$

### Parametrized Surfaces and Surface Area

A surface is parametrized by

$$
\mathbf{X}(u,v)
=
\langle x(u,v),y(u,v),z(u,v)\rangle
$$

Its area element is

$$
dS = \|\mathbf{X}_u\times \mathbf{X}_v\|\,du\,dv
$$

So the surface area is

$$
\iint_S 1\,dS
=
\iint_D \|\mathbf{X}_u\times \mathbf{X}_v\|\,du\,dv
$$

### Visualization — parametrized surface and normal vectors

```python {run}
import numpy as np
import matplotlib.pyplot as plt

u = np.linspace(0, 2*np.pi, 120)
v = np.linspace(0.2, 1.0, 80)
U, V = np.meshgrid(u, v)

X = V * np.cos(U)
Y = V * np.sin(U)
Z = 0.7 * (X**2 - Y**2)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.88)

uq = np.linspace(0, 2*np.pi, 10)
vq = np.linspace(0.3, 0.95, 5)

for uu in uq:
    for vv in vq:
        x = vv * np.cos(uu)
        y = vv * np.sin(uu)
        z = 0.7 * (x**2 - y**2)

        Xu = np.array([-vv*np.sin(uu), vv*np.cos(uu), 0.7 * (2*x*(-vv*np.sin(uu)) - 2*y*(vv*np.cos(uu)) )])
        Xv = np.array([np.cos(uu), np.sin(uu), 0.7 * (2*x*np.cos(uu) - 2*y*np.sin(uu))])
        n = np.cross(Xu, Xv)
        n = n / np.linalg.norm(n)

        ax.quiver(x, y, z, n[0], n[1], n[2], length=0.18, normalize=True)

ax.set_title("Parametrized surface with normal directions")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
```

### Surface Integrals of Vector Fields

Flux through a surface is

$$
\iint_S \mathbf{F}\cdot \mathbf{n}\,dS
$$

or, in parametrized form,

$$
\iint_D \mathbf{F}(\mathbf{X}(u,v))\cdot(\mathbf{X}_u\times \mathbf{X}_v)\,du\,dv
$$

---

## 8. The Integral Theorems of Vector Analysis

### Green's Theorem

For a positively oriented simple closed curve $C$ bounding a region $D$,

$$
\oint_C P\,dx+Q\,dy
=
\iint_D
\left(
\frac{\partial Q}{\partial x}
-
\frac{\partial P}{\partial y}
\right)\,dA
$$

### Stokes' Theorem

For an oriented surface $S$ with boundary $\partial S$,

$$
\oint_{\partial S}\mathbf{F}\cdot d\mathbf{r}
=
\iint_S (\nabla\times\mathbf{F})\cdot\mathbf{n}\,dS
$$

### Conservative Fields

A vector field is conservative if

$$
\mathbf{F}=\nabla \phi
$$

for some potential function $\phi$.

Then line integrals depend only on endpoints:

$$
\int_C \mathbf{F}\cdot d\mathbf{r}
=
\phi(B)-\phi(A)
$$

### Gauss' Theorem (Divergence Theorem)

For a closed surface $S=\partial E$,

$$
\iint_S \mathbf{F}\cdot\mathbf{n}\,dS
=
\iiint_E \nabla\cdot\mathbf{F}\,dV
$$

This connects outward flux through a surface to total divergence inside the volume.

### Visualization — outward field on a sphere (flux intuition)

```python {run}
import numpy as np
import matplotlib.pyplot as plt

phi = np.linspace(0, np.pi, 80)
theta = np.linspace(0, 2*np.pi, 160)
Phi, Theta = np.meshgrid(phi, theta)

X = np.sin(Phi) * np.cos(Theta)
Y = np.sin(Phi) * np.sin(Theta)
Z = np.cos(Phi)

U = X
V = Y
W = Z

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.35)

step_t, step_p = 12, 8
ax.quiver(X[::step_t, ::step_p], Y[::step_t, ::step_p], Z[::step_t, ::step_p],
          U[::step_t, ::step_p], V[::step_t, ::step_p], W[::step_t, ::step_p],
          length=0.15, normalize=True)

ax.set_title("Outward flux field on a sphere")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
```

---

## Core Formulas

$$
\nabla f=
\begin{bmatrix}
f_x\\\\f_y\\\\f_z
\end{bmatrix}
$$

$$
D_{\mathbf{u}}f=\nabla f\cdot\mathbf{u}
$$

$$
\nabla\cdot\mathbf{F}
=
P_x+Q_y+R_z
$$

$$
\nabla\times\mathbf{F}
=
\begin{bmatrix}
R_y-Q_z\\\\P_z-R_x\\\\Q_x-P_y
\end{bmatrix}
$$

$$
L=\int_a^b \|\mathbf{r}'(t)\|\,dt
$$

$$
\int_C \mathbf{F}\cdot d\mathbf{r}
=
\int_a^b \mathbf{F}(\mathbf{r}(t))\cdot \mathbf{r}'(t)\,dt
$$

$$
dS=\|\mathbf{X}_u\times \mathbf{X}_v\|\,du\,dv
$$

$$
\iint_D f(x,y)\,dA
=
\iint_{D^\ast} f(x(u,v),y(u,v))
\left|
\frac{\partial(x,y)}{\partial(u,v)}
\right|
\,du\,dv
$$

$$
\oint_C P\,dx+Q\,dy
=
\iint_D
\left(
\frac{\partial Q}{\partial x}
-
\frac{\partial P}{\partial y}
\right)\,dA
$$

$$
\oint_{\partial S}\mathbf{F}\cdot d\mathbf{r}
=
\iint_S (\nabla\times\mathbf{F})\cdot\mathbf{n}\,dS
$$

$$
\iint_{\partial E}\mathbf{F}\cdot\mathbf{n}\,dS
=
\iiint_E \nabla\cdot\mathbf{F}\,dV
$$

---

