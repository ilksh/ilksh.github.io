---
title: Probability
category: STATISTICS
semester: 2023 S
---

# Probability

## 1. Basis of Probability
If one experiment can result in $m$  possible outcomes and another experiment can result in $n$  possible outcomes, then there are $m \times n $ possible outcomes from the two experiments.

### Permutation
A permutation is an arrangement of $r$ objects selected from a set of $n$ distinct objects in a specific order.
$$P(n, r) = \frac{n!}{(n - r)!} $$

### Combination
 A combination is the number of ways to choose $r$ elements from a set of $n$ elements where the order of selection does not matter.
 $$ C(n, r) = \frac{n!}{r!(n-r)!}$$

### Notations & Axioms
- Sample Space ($S$): The set of all possible outcomes of a random experiment. 

  For a die roll, $S = \{1, 2, 3, 4, 5, 6\}$
- Event ($E$):  A subset of the sample space. For rolling an even number, $E = \{2, 4, 6\}$.
- Union (A $\cup$ B), Intersection (A $\cap$ B), Complement ($A^c$)
- Axioms of Probability: Non-negativity, Normalization, and Additivity

### Bayes' Theorem
Given hypotheses $H_1$, $H_2$, $H_3$, the posterior probability of $R_i$ given event $E$ is:
$$ P(R_i | E) = \frac{P(R_i) P(E | R_i)}{\sum_j P(R_j) P(E | R_j)} $$

### Law of Total Probability
- The law of total probability states:
  $$ P(C) = P(C \mid A) P(A) + P(C \mid B) P(B) + P(C \mid E) P(E) $$
- Independence: If events $A$ and $B$ are independent, then:
  $$P(A \cap B) = P(A) P(B)$$

### Random Variables
Random variable is a function from the sample space $S$ to the real numbers $\mathbb{R}$

### Cumulative Distribution Function (CDF)
The CDF $F(x)$ of a random variable $X$ is defined as: **$F(x) = P(X \leq x)$**
1. $F(x)$ is non-decreasing & right-continuous
2. $0 \leq F(x) \leq 1$
3. $\lim\limits_{x \to \infty} F(x) = 1$ and $\lim\limits_{x \to -\infty} F(x) = 0$

```python {run}
import numpy as np
import matplotlib.pyplot as plt

# parameter
p = 0.3

# x range (support of geometric RV)
x = np.arange(0, 21)

# CDF
F = 1 - (1 - p) ** x

# plot
plt.figure(figsize=(12,8))
plt.step(x, F, where="post")
plt.scatter(x, F)

plt.xlabel("x")
plt.ylabel("F(x)")
plt.title(f"Geometric CDF (p = {p})")
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)

plt.show()
```

---

## 2. Discrete Random Variable
A random variable that can take on a countable number of possible values is called a **discrete random variable**

### Probability Mass Function (PMF)
For a discrete random variable $X$, the probability mass function is given by: $P(X = x_i)$
- $0 \leq P(X = x_i) \leq 1 $
- $\sum P(X = x_i)$ = 1

### Expectation & Variance
$$ E[X] = \sum x_i \cdot P(X = x_i) $$
$$ \text{Var}(X) = E[(X - E[X])^2] $$
$$ \text{Var}(X) = E[X^2] - (E[X])^2 $$

### Bernoulli and Binomial Random Variables
A random variable $X$ is called $\textbf{Bernoulli}$ with parameter $p$ if it takes on values 0 and 1 with probabilities:
$$ P(X = 0) = 1 - p \quad P(X = 1) = p $$

For a $\textbf{binomial}$ random variable, $X \sim \text{Binomial}(n, p)$:
$$ P(X = k) = \binom{n}{k} \cdot p^k \cdot (1 - p)^{n - k} $$  

$$ E[X] = n \cdot p $$
$$ \text{Var}[X] = n \cdot p \cdot (1 - p) $$ 

```python {run}
import numpy as np
import matplotlib.pyplot as plt
from math import comb

# ---------------------------
# Parameters
# ---------------------------
p = 0.3
n = 10

# ---------------------------
# Bernoulli Distribution
# ---------------------------
x_bern = np.array([0, 1])
pmf_bern = np.array([1 - p, p])

# ---------------------------
# Binomial Distribution
# ---------------------------
x_bin = np.arange(0, n + 1)
pmf_bin = np.array([comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in x_bin])

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(10, 4))

# Bernoulli
plt.subplot(1, 2, 1)
plt.bar(x_bern, pmf_bern)
plt.xticks([0, 1])
plt.xlabel("X")
plt.ylabel("P(X = x)")
plt.title(f"Bernoulli(p = {p})")
plt.axvline(p, linestyle="--")  # E[X] = p

# Binomial
plt.subplot(1, 2, 2)
plt.bar(x_bin, pmf_bin)
plt.xlabel("k")
plt.ylabel("P(X = k)")
plt.title(f"Binomial(n = {n}, p = {p})")
plt.axvline(n * p, linestyle="--")  # E[X] = np

plt.tight_layout()
plt.show()
```

### Poisson Random Variables
$$ P(X = k) = \frac{\lambda^k \cdot e^{-\lambda}}{k!} $$
$$ E(X) = \lambda \quad \text{Var}(X) = \lambda $$

```python {run}
import numpy as np
import matplotlib.pyplot as plt
from math import factorial, exp

# ---------------------------
# Poisson PMF 함수
# ---------------------------
def poisson_pmf(k, lam):
    return (lam ** k) * exp(-lam) / factorial(k)

# ---------------------------
# Parameters
# ---------------------------
lambdas = [2, 5, 10]
x = np.arange(0, 20)

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(10, 5))

for lam in lambdas:
    pmf = [poisson_pmf(k, lam) for k in x]
    plt.plot(x, pmf, marker='o', linewidth=2, label=f'λ = {lam}')

plt.title("Poisson Distributions with Different λ Values")
plt.xlabel("X (Number of Events)")
plt.ylabel("P(X)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()
```

### Geometric Distribution
A random variable $X$ follows a \textbf{Geometric distribution} if it models the number of trials needed to get the first success in a series of independent Bernoulli trials, each with probability $p$ of success

$$ P(X = k) = (1 - p)^{k-1} \cdot p $$ 
$$ E(X) = \frac{1}{p} \quad \text{Var}(X) = \frac{1 - p}{p^2} $$

```python {run}
import numpy as np
import matplotlib.pyplot as plt

def geom_pmf(k, p):
    # k starts at 1
    return (1 - p) ** (k - 1) * p

def geom_cdf(k, p):
    # P(X <= k) = 1 - (1-p)^k
    return 1 - (1 - p) ** k

# ---------------------------
# Parameters
# ---------------------------
ps = [0.2, 0.5, 0.8]          # 비교할 p들
K = 20                        # 표시 범위
k = np.arange(1, K + 1)

# ---------------------------
# Plot
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# (1) PMF
ax = axes[0]
for p in ps:
    pmf = geom_pmf(k, p)
    ax.plot(k, pmf, marker='o', linewidth=2, label=f"p = {p}")
    ax.axvline(1 / p, linestyle="--", linewidth=1)  # E[X] = 1/p

ax.set_title("Geometric Distribution PMF  (X = trials until first success)")
ax.set_xlabel("k")
ax.set_ylabel("P(X = k)")
ax.grid(True, alpha=0.3)
ax.legend(title="Parameter")

# (2) CDF (step)
ax = axes[1]
for p in ps:
    cdf = geom_cdf(k, p)
    ax.step(k, cdf, where='post', linewidth=2, label=f"p = {p}")

ax.set_title("Geometric Distribution CDF")
ax.set_xlabel("k")
ax.set_ylabel("P(X ≤ k)")
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)
ax.legend(title="Parameter")

plt.show()

# ---------------------------
# Memoryless property demo (single p)
# ---------------------------
p0 = 0.3
m, n = 5, 3  # P(X > m+n | X > m) = P(X > n)
lhs = (1 - p0) ** (m + n) / ((1 - p0) ** m)
rhs = (1 - p0) ** n
print(f"[Memoryless check, p={p0}]  P(X>m+n | X>m)={lhs:.6f},  P(X>n)={rhs:.6f}")
```

---

## 3. Continuous Random Variable
A continuous variable is a random variable if there exists some $\underline{non-negative function}$ $ f(x)$  defined for all $x$, such that for all $a$:
$$P(a \leq X \leq b) = \int_a^b f(x) \, dx$$
3 Rules of Continuous Variable
- $\int_{-\infty}^{\infty} f(x) \, dx = 1$
- $P(a < X < b) = P(a \leq X \leq b) = \int_a^b f(x) \, dx$
- $F(b) = \int_{-\infty}^{b} f(x)dx, \quad f(x) = \frac{d}{dx} F(x)$

### Definition of PDF and CDF
$$ F(x) = P(X \leq x) = \int_{-\infty}^{x} f(t) \, dt $$
The main difference between the PDF and CDF is that the PDF provides the probability density $\underline{at a specific point}$, while the CDF provides the cumulative probability $\underline{up to a specific point}$.

```python {run}
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, exp
from scipy.stats import norm

# ---------------------------
# Parameters
# ---------------------------
mu = 0
sigma = 1
x = np.linspace(-4, 4, 1000)

# ---------------------------
# PDF & CDF
# ---------------------------
pdf = norm.pdf(x, mu, sigma)
cdf = norm.cdf(x, mu, sigma)

# ---------------------------
# Plot
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# (1) PDF
ax = axes[0]
ax.plot(x, pdf, linewidth=2)
ax.fill_between(x, pdf, where=(x <= 1), alpha=0.3)
ax.axvline(1, linestyle="--", linewidth=1)
ax.set_title("PDF: Normal Distribution")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.grid(True, alpha=0.3)

# (2) CDF
ax = axes[1]
ax.plot(x, cdf, linewidth=2)
ax.axvline(1, linestyle="--", linewidth=1)
ax.axhline(norm.cdf(1), linestyle="--", linewidth=1)
ax.set_title("CDF: Normal Distribution")
ax.set_xlabel("x")
ax.set_ylabel("F(x)")
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)

plt.show()
```

### Expectation and Variance
$$ E(X) = \int_{-\infty}^{\infty} x \cdot f(x) \, dx $$
$$  \text{Var}(X) = E[(X - E(X))^2] = \int_{-\infty}^{\infty} (x - E(X))^2 \cdot f(x) \, dx $$
$$ \text{Var}(X) = E(X^2) - [E(X)]^2$$

### Uniform Random Variable
A $\textbf{uniform random variable}$ is a continuous random variable where all outcomes in a given interval are equally likely. The probability density function (PDF) of a uniform random variable is constant within a specified range $\([a, b]\)$, and zero outside of that range

$$ X \sim \text{Uniform}(a, b) $$

PDF & CDF 

$$
f(x) = \frac{1}{b-a}\,\mathbf{1}_{[a,b]}(x)
$$

$$
F(x)=0,\quad x\in(-\infty,a) \newline
F(x)=\frac{x-a}{b-a},\quad x\in(a,b) \newline
F(x)=1,\quad x\in(b,\infty)
$$

Expectation & Variance
$$ E(X) = \frac{a + b}{2} $$
$$ \text{Var}(X) = \frac{(b - a)^2}{12} $$ 

```python{run}
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Parameters
# ---------------------------
a, b = 2, 6
x = np.linspace(a - 2, b + 2, 1000)

# PDF
pdf = np.where((x >= a) & (x <= b), 1 / (b - a), 0)

# CDF
cdf = np.piecewise(
    x,
    [x < a, (x >= a) & (x <= b), x > b],
    [0,
     lambda x: (x - a) / (b - a),
     1]
)

# Expectation
EX = (a + b) / 2

# ---------------------------
# Plot
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# ---- PDF ----
ax = axes[0]
ax.plot(x, pdf, linewidth=2)
ax.fill_between(x, pdf, where=(x >= a) & (x <= b), alpha=0.3)
ax.axvline(a, linestyle="--", linewidth=1)
ax.axvline(b, linestyle="--", linewidth=1)
ax.axvline(EX, linestyle=":", linewidth=2, label=r"$E[X]$")
ax.set_title(r"PDF: Uniform$(a,b)$")
ax.set_xlabel("x")
ax.set_ylabel(r"$f(x)$")
ax.legend()
ax.grid(True, alpha=0.3)

# ---- CDF ----
ax = axes[1]
ax.plot(x, cdf, linewidth=2)
ax.axvline(a, linestyle="--", linewidth=1)
ax.axvline(b, linestyle="--", linewidth=1)
ax.axvline(EX, linestyle=":", linewidth=2)
ax.axhline(0, linewidth=1)
ax.axhline(1, linewidth=1)
ax.set_ylim(-0.05, 1.05)
ax.set_title(r"CDF: Uniform$(a,b)$")
ax.set_xlabel("x")
ax.set_ylabel(r"$F(x)$")
ax.grid(True, alpha=0.3)

plt.show()
```

### Normal Random Variable
Normal random variables are widely used in statistics and probability theory due to the $\underline{Central Limit Theorem}$, which states that the sum of many independent and identically distributed random variables tends to follow a normal distribution, regardless of the original distribution.

$$ X \sim \text{Normal}(\mu, \sigma^2) $$

PDF & CDF 

$$ f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) $$
$$ F(x) = P(X \leq x) = \int_{-\infty}^{x} f(t) \, dt $$

Expectation & Variance
$$ E(X) = \mu \newline \text{Var}(X) = \sigma^2 $$

```python {run}
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------
# Parameters
# ---------------------------
mu = 0
sigma = 1
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)

pdf = norm.pdf(x, mu, sigma)
cdf = norm.cdf(x, mu, sigma)

# ---------------------------
# Plot
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# ---------- PDF ----------
ax = axes[0]
ax.plot(x, pdf, linewidth=2, label="PDF")
ax.fill_between(
    x, pdf,
    where=(x >= mu - sigma) & (x <= mu + sigma),
    alpha=0.3,
    label="Within one standard deviation"
)

ax.axvline(mu, linestyle=":", linewidth=2, label="Mean")
ax.axvline(mu - sigma, linestyle="--", linewidth=1, label="±1 Std Dev")
ax.axvline(mu + sigma, linestyle="--", linewidth=1)

ax.set_title("Normal Distribution (PDF)")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, alpha=0.3)

# ---------- CDF ----------
ax = axes[1]
ax.plot(x, cdf, linewidth=2, label="CDF")
ax.axvline(mu, linestyle=":", linewidth=2, label="Mean")
ax.axvline(mu + sigma, linestyle="--", linewidth=1, label="+1 Std Dev")

ax.set_title("Normal Distribution (CDF)")
ax.set_xlabel("x")
ax.set_ylabel("Cumulative Probability")
ax.set_ylim(0, 1.02)
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()
```

### Exponential Random Variable

$$ X \sim \text{Exponential}(\lambda) $$

$$
f(x)=\lambda e^{-\lambda x},\quad x\in[0,\infty) \newline
f(x)=0,\quad x\in(-\infty,0)
$$

$$
F(x)=1-e^{-\lambda x},\quad x\in[0,\infty) \newline
F(x)=0,\quad x\in(-\infty,0)
$$

Expectation & Variance
$$ E(X) = \frac{1}{\lambda} \newline  \text{Var}(X) = \frac{1}{\lambda^2} $$

```python {run}
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Parameters
# ---------------------------
lambdas = [0.5, 1.0, 2.0]
x = np.linspace(0, 6, 1000)

# ---------------------------
# Plot
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# ---------- PDF ----------
ax = axes[0]
for lam in lambdas:
    pdf = lam * np.exp(-lam * x)
    ax.plot(x, pdf, linewidth=2, label=f"λ = {lam}")
    ax.axvline(1 / lam, linestyle="--", linewidth=1)

ax.set_title("Exponential Distribution (PDF)")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend(title="Rate")
ax.grid(True, alpha=0.3)

# ---------- CDF ----------
ax = axes[1]
for lam in lambdas:
    cdf = 1 - np.exp(-lam * x)
    ax.plot(x, cdf, linewidth=2, label=f"λ = {lam}")

ax.set_title("Exponential Distribution (CDF)")
ax.set_xlabel("x")
ax.set_ylabel("Cumulative Probability")
ax.set_ylim(0, 1.02)
ax.legend(title="Rate")
ax.grid(True, alpha=0.3)

plt.show()
```

### Memorylessness Property of Exponential Distribution
The exponential distribution has a unique property called $\textbf{memorylessness}$.This property states that the probability of an event occurring in the next $t$ units of time is $\textbf{independent}$ of how much time has already elapsed. Formally, for $X \sim \text{Exponential}(\lambda)$, the memorylessness property is expressed as:

$$
P(X > s + t \mid X > s) = P(X > t)
$$

```python {run}
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Parameters
# ---------------------------
lam = 1.0
s = 2.0
t = np.linspace(0, 6, 1000)

# Survival functions
survival_original = np.exp(-lam * t)
survival_conditional = np.exp(-lam * t)  # identical by memorylessness

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(8, 4))

plt.plot(t, survival_original, linewidth=2,
         label="P(X > t)")
plt.plot(t, survival_conditional, linestyle="--", linewidth=2,
         label="P(X > s + t | X > s)")

plt.axvline(0, linewidth=1)
plt.axvline(s, linestyle=":", linewidth=1, label="Elapsed time s")

plt.title("Memorylessness of Exponential Distribution")
plt.xlabel("Additional time t")
plt.ylabel("Survival Probability")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()
```

---

## 4. Continuous Joint Distribution
$$ P((\underline{\overline{X}}, \underline{\overline{Y}}) \in D) = \int \int_D f(x, y) \, dy \, dx $$ 

```python {run}
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# Joint PDF: 2D Normal (independent)
# ---------------------------
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

f = (1 / (2 * np.pi)) * np.exp(-(X**2 + Y**2) / 2)

# ---------------------------
# Region D (rectangle example)
# ---------------------------
x1, x2 = -1, 1
y1, y2 = -0.5, 1.5

mask = (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)

# ---------------------------
# Plot
# ---------------------------
fig = plt.figure(figsize=(12, 4))

# ---- 3D Surface ----
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, f, cmap='viridis', alpha=0.9)
ax1.set_title("Joint PDF f(x, y)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("Density")

# ---- 2D Contour + Region D ----
ax2 = fig.add_subplot(1, 2, 2)
ax2.contour(X, Y, f, levels=15)
ax2.contourf(X, Y, mask, levels=[0.5, 1], alpha=0.3)
ax2.set_title("Region D and Probability Mass")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

plt.tight_layout()
plt.show()
```

### Important Rules of Continuous Joint Distribution
1. The joint PDF $f(x, y)$ must be non-negative for all $(x, y)$
$$ f(x, y) \geq 0 $$ 
2. The total probability over the entire space must be 1
$$   \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x, y) \, dy \, dx = 1 $$
3. The marginal PDFs can be obtained by integrating the joint PDF over the other variable
$$     f_X(x) = \int_{-\infty}^{\infty} f(x, y) \, dy, \quad f_Y(y) = \int_{-\infty}^{\infty} f(x, y) \, dx $$
4.  If $\underline{\overline{X}}$ and $\underline{\overline{Y}}$ are independent, then the joint PDF factorizes as:
$$ f(x, y) = f_X(x) \cdot f_Y(y) $$

### Independent Random Variables in Joint Distributions
$$ f(x, y) = f_X(x) \cdot f_Y(y) $$
1. Joint Cumulative Distribution Function (CDF): 
$$ F(a, b) = P(\underline{\overline{X}} \leq a, \underline{\overline{Y}} \leq b) $$
2. Joint Probability Density Function (PDF):
$$     P((\underline{\overline{X}}, \underline{\overline{Y}}) \in D) = \int \int_D f(x, y) \, dy \, dx $$
3. Independence:
$$ F(a, b) = F_X(a) \cdot F_Y(b) \newline f(x, y) = f_X(x) \cdot f_Y(y) $$ 

### Gamma Distribution
The Gamma distribution is a continuous probability distribution that models the waiting time until the occurrence of $k$ events in a Poisson process.  
$$ X \sim \text{Gamma}(k, \lambda) $$ 

Probability Density Function (PDF)
$$ f(x; k, \lambda) = \frac{\lambda^k x^{k-1} e^{-\lambda x}}{\Gamma(k)}, \quad x > 0 $$ 
$$ \Gamma(k) = \int_0^\infty t^{k-1} e^{-t} \, dt $$
$$ E(X) = \frac{k}{\lambda}, \quad \text{Var}(X) = \frac{k}{\lambda^2} $$ 

```python {run}
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# ---------------------------
# Parameters
# ---------------------------
lam = 1.0           # rate
ks = [1, 2, 5, 9]   # shape parameters
x = np.linspace(0, 15, 1000)

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(8, 4))

for k in ks:
    pdf = gamma.pdf(x, a=k, scale=1/lam)
    plt.plot(x, pdf, linewidth=2, label=f"k = {k}")
    plt.axvline(k / lam, linestyle="--", linewidth=1)

plt.title("Gamma Distribution (PDF)")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend(title="Shape Parameter")
plt.grid(True, alpha=0.3)

plt.show()
```

### Formulas
1. Expectation
Linearity of Expectation
$$ E\left(\sum_{i=1}^n \underline{\overline{X}}_i\right) \text{=} \sum E(\underline{\overline{X}}_i) $$ 
2. Variance
$$ \text{Var}\left(\sum_{i=1}^n \underline{\overline{X}}_i\right) = n \cdot \text{Var}(\underline{\overline{X}}_i) $$ 
3. Covariance
$$ \text{Cov}(\underline{\overline{X}}, \underline{\overline{Y}}) = E[(\underline{\overline{X}} - E(\underline{\overline{X}}))(\underline{\overline{Y}} - E(\underline{\overline{Y}}))] $$
$$ \text{Cov}(\underline{\overline{X}}, \underline{\overline{Y}}) = E(\underline{\overline{X}}\underline{\overline{Y}}) - E(\underline{\overline{X}})E(\underline{\overline{Y}}) $$
4. Correlation
$$ \rho(\underline{\overline{X}}, \underline{\overline{Y}}) = \frac{\text{Cov}(\underline{\overline{X}}, \underline{\overline{Y}})}{\sqrt{\text{Var}(\underline{\overline{X}}) \cdot \text{Var}(\underline{\overline{Y}})}} $$ 

### Moment Generating Function (MGF)
$$ M_X(t) = E\left(e^{t\underline{\overline{X}}}\right) $$ 
The MGF has the property that the  $n$-th moment of $ \underline{\overline{X}}$ can be obtained by differentiating the MGF $n$ times with respect to $t$ and evaluating at $t = 0$:
$$ E(\underline{\overline{X}}^n) = M_X^{(n)}(0) $$ 

```python {run}
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Parameters
# ---------------------------
lam = 1.0
x = np.linspace(0, 6, 1000)
t = np.linspace(-0.9, 0.9, 1000)

# PDF
pdf = lam * np.exp(-lam * x)

# MGF
mgf = lam / (lam - t)

# First and second derivatives at t=0
mean = 1 / lam
second_moment = 2 / (lam**2)

# ---------------------------
# Plot
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# ---------- PDF ----------
ax = axes[0]
ax.plot(x, pdf, linewidth=2)
ax.axvline(mean, linestyle="--", linewidth=1, label="Mean")
ax.set_title("Original Distribution (PDF)")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, alpha=0.3)

# ---------- MGF ----------
ax = axes[1]
ax.plot(t, mgf, linewidth=2, label="MGF")
ax.scatter(0, 1, zorder=5, label="MGF at t = 0")

# Tangent at t=0 (first derivative)
t_tan = np.linspace(-0.5, 0.5, 100)
tangent = 1 + mean * t_tan
ax.plot(t_tan, tangent, linestyle="--", linewidth=2,
        label="Slope at t = 0 (First Moment)")

ax.set_title("Moment Generating Function")
ax.set_xlabel("t")
ax.set_ylabel("Value")
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()
```

### Definition of i.i.d. (Independent and Identically Distributed) 
A sequence of random variables $\underline{\overline{X}}_1, \underline{\overline{X}}_2, \dots, \underline{\overline{X}}_n$ is said to be $\textbf{independent and identically}$ $\textbf{distributed (i.i.d.)}$ if:
1. Each random variable $\underline{\overline{X}}_i$ has the same probability distribution.
2. The random variables are mutually independent.