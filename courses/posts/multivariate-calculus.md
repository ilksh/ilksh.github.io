---
title: Multivariate Calculus
category: MATH
semester: 2022 F
---

## 1. Vectors & Geometry in $\mathbb{R}^3$

### Norms, Dot, Cross
$$
\|\mathbf{v}\|=\sqrt{v_1^2+v_2^2+v_3^2}
$$

$$
\mathbf{a}\cdot\mathbf{b}=\sum_{i=1}^3 a_i b_i=\|\mathbf{a}\|\|\mathbf{b}\|\cos\theta
$$

Areas/volumes:
$$
\|\mathbf{a}\times\mathbf{b}\|=\text{area of parallelogram}
$$
$$
\mathbf{a}\cdot(\mathbf{b}\times\mathbf{c})=\det[\mathbf{a}\ \mathbf{b}\ \mathbf{c}]
$$

### Projections
$$
\mathrm{proj}_{\mathbf{b}}\mathbf{a}=\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{b}\|^2}\mathbf{b}
$$

$$
\mathrm{comp}_{\mathbf{b}}\mathbf{a}=\frac{\mathbf{a}\cdot\mathbf{b}}{\|\mathbf{b}\|}
$$

### Lines & Planes
Line (vector/parametric):
$$
\mathbf{r}(t)=\mathbf{r}_0+t\mathbf{v}
$$

Plane (normal form):
$$
\mathbf{n}\cdot(\mathbf{r}-\mathbf{r}_0)=0
$$
Plane through 3 points $A,B,C$:
$$
\mathbf{n}=(\overrightarrow{AB})\times(\overrightarrow{AC}),\quad \mathbf{n}\cdot(\mathbf{r}-\mathbf{r}_A)=0
$$

### Distances
Point $P$ to plane $\mathbf{n}\cdot(\mathbf{r}-\mathbf{r}_0)=0$:
$$
d=\frac{|\mathbf{n}\cdot(\mathbf{r}_P-\mathbf{r}_0)|}{\|\mathbf{n}\|}
$$

### Quadratic Surfaces (canonical)
- Ellipsoid: $\frac{x^2}{a^2}+\frac{y^2}{b^2}+\frac{z^2}{c^2}=1$
- Hyperboloid (1 sheet): $\frac{x^2}{a^2}+\frac{y^2}{b^2}-\frac{z^2}{c^2}=1$
- Hyperboloid (2 sheets): $-\frac{x^2}{a^2}-\frac{y^2}{b^2}+\frac{z^2}{c^2}=1$
- Elliptic paraboloid: $z=\frac{x^2}{a^2}+\frac{y^2}{b^2}$
- Hyperbolic paraboloid: $z=\frac{x^2}{a^2}-\frac{y^2}{b^2}$
- Elliptic cone: $\frac{x^2}{a^2}+\frac{y^2}{b^2}-\frac{z^2}{c^2}=0$
- Cylinders: variable missing (e.g., $x^2+y^2=1$)

---

## 2. Vector-Valued Functions & Space Curves

Curve:
$$
\mathbf{r}(t)=\langle x(t),y(t),z(t)\rangle
$$

Velocity / speed / acceleration:
$$
\mathbf{v}(t)=\mathbf{r}'(t),\quad |\mathbf{v}(t)|=\|\mathbf{r}'(t)\|,\quad \mathbf{a}(t)=\mathbf{r}''(t)
$$

Arc length:
$$
L=\int_a^b \|\mathbf{r}'(t)\|\,dt
$$

Unit tangent:
$$
\mathbf{T}=\frac{\mathbf{r}'(t)}{\|\mathbf{r}'(t)\|}
$$

Curvature:
$$
\kappa=\left\|\frac{d\mathbf{T}}{ds}\right\|
$$
Equivalent:
$$
\kappa=\frac{\|\mathbf{r}'(t)\times\mathbf{r}''(t)\|}{\|\mathbf{r}'(t)\|^3}
$$

Normal/binormal (optional but standard in many Calc III syllabi):
$$
\mathbf{N}=\frac{\mathbf{T}'(s)}{\|\mathbf{T}'(s)\|},\qquad \mathbf{B}=\mathbf{T}\times\mathbf{N}
$$

---

## 3. Multivariable Limits, Continuity, Derivatives

Limit:
$$
\lim_{(x,y)\to(a,b)} f(x,y)
$$

Partials:
$$
f_x=\frac{\partial f}{\partial x},\quad f_y=\frac{\partial f}{\partial y},\quad f_z=\frac{\partial f}{\partial z}
$$

Higher-order partials / Clairaut:
$$
f_{xy}=f_{yx}\quad (\text{when continuous near the point})
$$

Gradient:
$$
\nabla f=\left\langle f_x,f_y,f_z\right\rangle
$$

Directional derivative (unit $\mathbf{u}$):
$$
D_{\mathbf{u}}f=\nabla f\cdot \mathbf{u}
$$

Tangent plane / linearization for $z=f(x,y)$ at $(a,b)$:
$$
z\approx f(a,b)+f_x(a,b)(x-a)+f_y(a,b)(y-b)
$$

Total differential:
$$
dz=f_x\,dx+f_y\,dy\quad (\text{and } df=\nabla f\cdot d\mathbf{r})
$$

Level sets:
$$
f(x,y,z)=c,\qquad \nabla f \perp \text{level surface}
$$

### Implicit Differentiation (2 variables)
If $F(x,y)=0$ and $F_y\neq 0$:
$$
\frac{dy}{dx}=-\frac{F_x}{F_y}
$$

---

## 4. **Multivariable Chain Rule (Complete)**

### A) Scalar output, one parameter
If $z=f(x,y)$ with $x=x(t),y=y(t)$:
$$
\frac{dz}{dt}=f_x(x(t),y(t))\frac{dx}{dt}+f_y(x(t),y(t))\frac{dy}{dt}
$$
If $z=f(x,y,z)$ with $(x,y,z)=(x(t),y(t),z(t))$:
$$
\frac{df}{dt}=f_x x'(t)+f_y y'(t)+f_z z'(t)
$$

### B) Scalar output, several parameters
If $w=f(x,y)$ with $x=x(s,t),y=y(s,t)$:
$$
\frac{\partial w}{\partial s}=f_x\frac{\partial x}{\partial s}+f_y\frac{\partial y}{\partial s},\qquad
\frac{\partial w}{\partial t}=f_x\frac{\partial x}{\partial t}+f_y\frac{\partial y}{\partial t}
$$

### C) Jacobian / matrix form (the clean statement)
Let $f:\mathbb{R}^m\to\mathbb{R}^p$ and $g:\mathbb{R}^n\to\mathbb{R}^m$ be differentiable.
Then:
$$
D(f\circ g)(\mathbf{x}) = Df(g(\mathbf{x}))\,Dg(\mathbf{x})
$$
In terms of Jacobians:
$$
J_{f\circ g}(\mathbf{x}) = J_f(g(\mathbf{x}))\,J_g(\mathbf{x})
$$

### D) Gradient form (common special case)
If $f:\mathbb{R}^m\to\mathbb{R}$ and $g:\mathbb{R}^n\to\mathbb{R}^m$:
$$
\nabla (f\circ g)(\mathbf{x}) = (J_g(\mathbf{x}))^\top \nabla f(g(\mathbf{x}))
$$

---

## 5. Optimization (Unconstrained & Constrained)

Critical points:
$$
\nabla f(\mathbf{x})=\mathbf{0}
$$

Hessian (2D):
$$
H_f=
\begin{bmatrix}
f_{xx} & f_{xy}\\
f_{yx} & f_{yy}
\end{bmatrix}
$$

Second derivative test (2D):
$$
D=f_{xx}f_{yy}-f_{xy}^2
$$
- If $D>0$ and $f_{xx}>0$: local min  
- If $D>0$ and $f_{xx}<0$: local max  
- If $D<0$: saddle

### Lagrange Multipliers
One constraint $g(\mathbf{x})=c$:
$$
\nabla f=\lambda \nabla g
$$

Multiple constraints $g_i(\mathbf{x})=c_i$:
$$
\nabla f=\sum_{i=1}^k \lambda_i \nabla g_i
$$

---

## 6. Multiple Integrals & Coordinate Systems

### Double / Triple Integrals
$$
\iint_R f(x,y)\,dA,\qquad \iiint_E f(x,y,z)\,dV
$$

Iterated integrals (Type I region):
$$
\iint_R f\,dA=\int_a^b\int_{g(x)}^{h(x)} f(x,y)\,dy\,dx
$$

### Polar Coordinates
$$
x=r\cos\theta,\quad y=r\sin\theta,\quad dA=r\,dr\,d\theta
$$

### Cylindrical Coordinates
$$
x=r\cos\theta,\quad y=r\sin\theta,\quad z=z,\quad dV=r\,dr\,d\theta\,dz
$$

### Spherical Coordinates
$$
x=\rho\sin\phi\cos\theta,\quad y=\rho\sin\phi\sin\theta,\quad z=\rho\cos\phi
$$
$$
dV=\rho^2\sin\phi\,d\rho\,d\phi\,d\theta
$$

### Change of Variables / Jacobian
If $(x,y)=(x(u,v),y(u,v))$:
$$
\iint_R f(x,y)\,dA = \iint_S f(x(u,v),y(u,v))\,\left|\frac{\partial(x,y)}{\partial(u,v)}\right|\,du\,dv
$$
where
$$
\frac{\partial(x,y)}{\partial(u,v)}=
\begin{vmatrix}
x_u & x_v\\
y_u & y_v
\end{vmatrix}
$$

---

## 7. Vector Fields, Line Integrals, Flux

Vector field:
$$
\mathbf{F}=\langle P,Q,R\rangle
$$

### Line Integrals
Scalar line integral (with arc length):
$$
\int_C f\,ds
$$

Vector line integral (work):
$$
\int_C \mathbf{F}\cdot d\mathbf{r}
$$
If $\mathbf{r}(t)$, $t\in[a,b]$:
$$
\int_C \mathbf{F}\cdot d\mathbf{r}=\int_a^b \mathbf{F}(\mathbf{r}(t))\cdot \mathbf{r}'(t)\,dt
$$

### Conservative Fields
$\mathbf{F}$ is conservative if $\exists \phi$ such that:
$$
\mathbf{F}=\nabla \phi
$$
Then:
$$
\int_C \mathbf{F}\cdot d\mathbf{r}=\phi(B)-\phi(A)
$$
Curl test (simply connected domain):
$$
\nabla\times\mathbf{F}=\mathbf{0}\ \Longrightarrow\ \mathbf{F}\ \text{conservative}
$$
In 2D ($\mathbf{F}=\langle P,Q\rangle$):
$$
\frac{\partial P}{\partial y}=\frac{\partial Q}{\partial x}
$$


### Flux / Surface Integrals
Flux through oriented surface $S$:

$$
\iint_S \mathbf{F}\cdot \mathbf{n}\,dS
$$

Parametrized surface $\mathbf{r}(u,v)$, $(u,v)\in D$:

---
## 8. Import Theorems

### A. Green’s Theorem

Green’s Theorem relates a line integral around a closed curve in the plane to a double integral over the region it encloses.
It is the 2D special case connecting circulation / flux to partial derivatives inside the region.

### B. Stokes’ Theorem

Stokes’ Theorem generalizes Green’s Theorem to surfaces in 3D.
It states that the circulation along the boundary curve equals the surface integral of the curl over the surface.

### C. Divergence Theorem

The Divergence Theorem relates the flux through a closed surface to a triple integral of divergence over the enclosed volume.
It converts a boundary surface integral into a volume integral.

---

## 9. Common Parameterizations (handy in practice)

Plane $ax+by+cz=d$ (solve for $z$):$
\mathbf{r}(x,y)=\langle x,y,\tfrac{d-ax-by}{c}\rangle
$

Graph surface $z=g(x,y)$:
$$
\mathbf{r}(x,y)=\langle x,y,g(x,y)\rangle,\quad
\mathbf{r}_x\times\mathbf{r}_y=\langle -g_x,-g_y,1\rangle
$$

Cylinder $x^2+y^2=a^2$:
$$
\mathbf{r}(\theta,z)=\langle a\cos\theta,a\sin\theta,z\rangle
$$

Sphere $x^2+y^2+z^2=R^2$:
$$
\mathbf{r}(\phi,\theta)=\langle R\sin\phi\cos\theta,\ R\sin\phi\sin\theta,\ R\cos\phi\rangle
$$

---

## 10. Quick Identity Block (often used)

Product rules (componentwise):
$$
\nabla(fg)=f\nabla g+g\nabla f
$$

Chain rule (scalar):
$$
\nabla (f\circ g) = (J_g)^\top \nabla f\circ g
$$

Fundamental theorem for line integrals:
$$
\int_C \nabla \phi\cdot d\mathbf{r}=\phi(B)-\phi(A)
$$
