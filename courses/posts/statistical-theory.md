---
title: Statistical Theory
category: STATISTICS
semester: 2023 S
---
# 1. Statistics / Sampling

## Statistical Model
$$\mathcal{M}=\{f(x\mid\theta):\theta\in\Theta\}$$
- A statistical model is a family of distributions indexed by an unknown parameter.

## Data-Generating Assumption
$$X \sim f(x\mid\theta)$$
- The data are random; inference targets the parameter $\theta$.

### Sample
$$X_1,\dots,X_n \overset{i.i.d.}{\sim} f(x\mid\theta)$$
- Observations are assumed independently and generated from the same distribution.

### Statistic
$$T=T(X_1,\dots,X_n)$$
- A statistic is a parameter-free function of the sample.

## Sampling Distribution
$$T(X_1,\dots,X_n)\sim G_\theta$$
$$\mathrm{Var}(\bar X)=\frac{\sigma^2}{n}$$
- The sampling distribution describes the variability of a statistic under repeated sampling.

## Law of Large Numbers & Central Limit Theorem
$$\bar X_n \xrightarrow{P} \mu$$
- Sample averages converge in probability to the population mean.
  
$$\sqrt{n}(\bar X-\mu)\xrightarrow{d}N(0,\sigma^2)$$
- Scaled sample averages converge in distribution to a normal law.

```python {run}
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------
# Parameters
# ---------------------------
np.random.seed(0)

lam = 1.0                     # Exponential rate
mu = 1 / lam
sigma = 1 / lam
N = 10000                     # total samples
ns = np.arange(1, N + 1)

# ---------------------------
# Generate samples
# ---------------------------
X = np.random.exponential(scale=1/lam, size=N)

# Sample means
sample_means = np.cumsum(X) / ns

# ---------------------------
# LLN Plot
# ---------------------------
plt.figure(figsize=(10, 4))

plt.plot(ns, sample_means, linewidth=1, label="Sample Mean")
plt.axhline(mu, linestyle="--", linewidth=2, label="True Mean")

plt.title("Law of Large Numbers")
plt.xlabel("Sample Size n")
plt.ylabel("Sample Mean")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

# ---------------------------
# CLT Plot
# ---------------------------
n_fixed = 50
num_trials = 5000

samples = np.random.exponential(scale=1/lam, size=(num_trials, n_fixed))
means = samples.mean(axis=1)

# CLT scaling
Z = np.sqrt(n_fixed) * (means - mu) / sigma

# Plot histogram
x = np.linspace(-4, 4, 500)

plt.figure(figsize=(8, 4))
plt.hist(Z, bins=40, density=True, alpha=0.5, label="Empirical Distribution")
plt.plot(x, norm.pdf(x), linewidth=2, label="Standard Normal")

plt.title("Central Limit Theorem")
plt.xlabel("Scaled Sample Mean")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()
```
---
# 2. Point Estimation Theory 

Estimator
$$\hat{\theta} = T(X_1,\dots,X_n)$$
- An estimator is a statistic used to approximate an unknown parameter $\theta$.

Unbiasedness
$$\mathbb{E}_\theta[\hat{\theta}] = \theta$$

Consistency
$$\hat{\theta}_n \xrightarrow{P} \theta$$
- An estimator is consistent if it converges in probability to the true parameter as $$n \to \infty$$.

## Mean Squared Error (MSE)
$$\mathrm{MSE}(\hat{\theta}) = \mathbb{E}\\left[(\hat{\theta}-\theta)^2\right] =
\mathrm{Var}(\hat{\theta})+\mathrm{Bias}(\hat{\theta})^2$$
- MSE captures the bias–variance tradeoff.

## Fisher Information

$$I(\theta) = \mathbb{E}\\left[ \left( \frac{\partial}{\partial\theta}\log f(X\mid\theta) \right)^2 \right]$$

## Cramér–Rao Lower Bound (CRLB)
$$\mathrm{Var}(\hat{\theta}) \ge \frac{1}{I(\theta)}$$

Efficiency
$$\mathrm{Var}(\hat{\theta}) = \frac{1}{I(\theta)}$$

Sufficiency
$$f(x_1,\dots,x_n\mid\theta) = g(T(x),\theta)\,h(x)$$

## Rao–Blackwell Theorem
$$\hat{\theta}_{RB} = \mathbb{E}(\hat{\theta}\mid T)$$
- Conditioning an estimator on a sufficient statistic never increases variance.

## Method of Moments
$$\frac{1}{n}\sum_{i=1}^n X_i^k = \mathbb{E}_\theta[X^k],\quad k=1,\dots,m$$
- Parameters are estimated by matching sample moments to population moments.

---
# 3. Likelihood Inference

## Likelihood Function
$$L(\theta \mid x_1,\dots,x_n) = \prod_{i=1}^n f(x_i\mid\theta)$$

$$\ell(\theta)=\log L(\theta\mid x) = \sum_{i=1}^n \log f(x_i\mid\theta)$$

## MLE via Differentiation (Estimation Process)
$$\hat\theta = \arg\max_\theta \ell(\theta)$$
$$\frac{\partial}{\partial\theta}\ell(\theta)=0 \quad\Rightarrow\quad \hat\theta$$
$$\frac{\partial^2}{\partial\theta^2}\ell(\hat\theta)<0$$
- Solve first-order condition  
- Verify local maximum via second derivative  

```python {run}
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Data (fixed)
# ---------------------------
np.random.seed(0)
n = 50
true_mu = 2.0
sigma = 1.0
data = np.random.normal(true_mu, sigma, n)

# ---------------------------
# Log-likelihood for mu
# ---------------------------
theta = np.linspace(-1, 5, 500)

def log_likelihood(mu, x, sigma):
    return -0.5 * np.sum((x - mu)**2) / sigma**2

ll = np.array([log_likelihood(t, data, sigma) for t in theta])

# MLE
mle = data.mean()

# First derivative (numerical)
dll = np.gradient(ll, theta)

# ---------------------------
# Plot
# ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

# ---- Log-likelihood ----
ax = axes[0]
ax.plot(theta, ll, linewidth=2, label="Log-Likelihood")
ax.axvline(mle, linestyle="--", linewidth=2, label="MLE")
ax.set_title("Log-Likelihood as a Function of Parameter")
ax.set_xlabel("Parameter value")
ax.set_ylabel("Log-Likelihood")
ax.legend()
ax.grid(True, alpha=0.3)

# ---- First derivative ----
ax = axes[1]
ax.plot(theta, dll, linewidth=2, label="First Derivative")
ax.axhline(0, linewidth=1)
ax.axvline(mle, linestyle="--", linewidth=2, label="MLE")
ax.set_title("First-Order Condition")
ax.set_xlabel("Parameter value")
ax.set_ylabel("Derivative")
ax.legend()
ax.grid(True, alpha=0.3)

plt.show()
```

## Score Function
$$U(\theta)=\frac{\partial}{\partial\theta}\ell(\theta)$$
$$U(\hat\theta)=0$$

## Information
Observed Information
$$J(\theta) = -\frac{\partial^2}{\partial\theta^2}\ell(\theta)$$

Fisher Information
$$I(\theta) = \mathbb{E}\\left[ \left(\frac{\partial}{\partial\theta}\log f(X\mid\theta)\right)^2 \right] \newline
= -\mathbb{E}\\left[ \frac{\partial^2}{\partial\theta^2}\log f(X\mid\theta) \right]$$

- Observed Information $J(\theta)$: computed from the actual realized sample via the second derivative of the log-likelihood; it is data-dependent and varies from sample to sample.
  
- Fisher Information $I(\theta)$: the expectation of the observed information under the model; it is model-based, deterministic given $\theta$, and represents the average curvature of the log-likelihood.

Likelihood Ratio
$$\Lambda = \frac{L(\theta_0\mid x)}{L(\hat\theta\mid x)}$$
$$-2\log\Lambda = 2\{\ell(\hat\theta)-\ell(\theta_0)\}$$

## MLE (Differentiation) vs Likelihood Ratio
### MLE via Differentiation
- Produces a point estimator $\hat\theta$
- Based on local optimality of $\ell(\theta)$
- Requires differentiability
### Likelihood Ratio
- Compares global support between $\theta_0$ and $\hat\theta$
- Used for hypothesis testing
- Invariant to reparameterization
  
## Asymptotic Normality of the MLE
$$\sqrt{n}(\hat\theta-\theta_0) \xrightarrow{d} N\\left(0,I(\theta_0)^{-1}\right)$$

Delta Method
$$\sqrt{n}\big(g(\hat\theta)-g(\theta_0)\big) \xrightarrow{d} N\\left(0,(g'(\theta_0))^2 I(\theta_0)^{-1}\right)$$

Wald Statistic
$$W = (\hat\theta-\theta_0)^\top \big[I(\hat\theta)^{-1}\big]^{-1} (\hat\theta-\theta_0) \;\xrightarrow{d}\; \chi^2_k$$
Score Statistic
$$S = U(\theta_0)^\top I(\theta_0)^{-1} U(\theta_0) \;\xrightarrow{d}\; \chi^2_k$$

Confidence Interval (MLE-based)
$$\hat\theta \pm z_{\alpha/2}\sqrt{I(\hat\theta)^{-1}}$$

Bootstrap

$$X_{1}^{\*} \text{,}\dots\text{,}X_n^* \sim \widehat{F}_n \quad \Rightarrow \quad \hat\theta^*$$ 

Asymptotic Equivalence
$$W \\approx\ S \\approx\ -2\log\Lambda \quad (n\to\infty)$$

```python {run}
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------------
# Data
# -----------------------------
np.random.seed(0)
n = 60
true_mu = 1.0
sigma = 1.0
x = np.random.normal(true_mu, sigma, n)

theta = np.linspace(-0.5, 2.5, 600)

def loglik(mu):
    return -0.5 * np.sum((x - mu)**2) / sigma**2

ll = np.array([loglik(t) for t in theta])

# MLE and Fisher information
mle = x.mean()
I = n / sigma**2

# Quadratic approximation
ll_quad = ll.max() - 0.5 * I * (theta - mle)**2

# Wald CI
z = 1.96
ci_low = mle - z / np.sqrt(I)
ci_high = mle + z / np.sqrt(I)

# LR cutoff (chi-square df=1, alpha=0.05)
lr_cut = ll.max() - 0.5 * 3.84

# -----------------------------
# Figure
# -----------------------------
fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

# (1) Likelihood surface (1D)
ax = axes[0, 0]
ax.plot(theta, ll, linewidth=2)
ax.axvline(mle, linestyle="--", label="MLE")
ax.set_title("Log-likelihood as a landscape")
ax.set_xlabel("Parameter value")
ax.set_ylabel("Log-likelihood")
ax.grid(alpha=0.3)
ax.legend()

# (2) Local quadratic approximation
ax = axes[0, 1]
ax.plot(theta, ll, linewidth=2, label="Log-likelihood")
ax.plot(theta, ll_quad, linestyle="--", linewidth=2,
        label="Local quadratic approximation")
ax.axvline(mle, linestyle=":")
ax.set_title("Local quadratic (Taylor) approximation")
ax.grid(alpha=0.3)
ax.legend()

# (3) Wald vs Likelihood Ratio geometry
ax = axes[1, 0]
ax.plot(theta, ll_quad, linewidth=2)
ax.axhline(lr_cut, linestyle="--", label="Likelihood ratio cutoff")
ax.axvspan(ci_low, ci_high, alpha=0.25, label="Wald confidence interval")
ax.axvline(mle, linestyle=":")
ax.set_title("Wald CI and LR test from same geometry")
ax.set_xlabel("Parameter value")
ax.set_ylabel("Approx. log-likelihood")
ax.grid(alpha=0.3)
ax.legend()

# (4) Asymptotic normality (horizontal slice)
ax = axes[1, 1]
z_grid = np.linspace(-4, 4, 600)
ax.plot(z_grid, norm.pdf(z_grid), linewidth=2)
ax.set_title("Asymptotic normality of MLE")
ax.set_xlabel("Standardized parameter")
ax.set_ylabel("Density")
ax.grid(alpha=0.3)

# (5) Delta method (parameter transformation)
ax = axes[2, 0]
g = lambda t: np.log(t)
theta_pos = theta[theta > 0]
ll_trans = ll[theta > 0]
ax.plot(g(theta_pos), ll_trans, linewidth=2)
ax.axvline(g(mle), linestyle="--")
ax.set_title("Delta method = reparameterized geometry")
ax.set_xlabel("Transformed parameter")
ax.set_ylabel("Log-likelihood")
ax.grid(alpha=0.3)

# (6) Bootstrap intuition
ax = axes[2, 1]
boot = []
B = 4000
for _ in range(B):
    xb = np.random.choice(x, size=n, replace=True)
    boot.append(xb.mean())
boot = np.array(boot)

ax.hist(boot, bins=40, density=True, alpha=0.6, label="Bootstrap")
grid = np.linspace(boot.min(), boot.max(), 500)
ax.plot(grid, norm.pdf(grid, mle, 1/np.sqrt(I)),
        linewidth=2, label="Asymptotic normal")
ax.axvline(mle, linestyle="--")
ax.set_title("Bootstrap approximates local geometry")
ax.set_xlabel("Estimator value")
ax.grid(alpha=0.3)
ax.legend()

plt.show()
```
---
# 4. Bayesian Inference

## The Prior and Posterior Distributions
Prior Distribution
$$\pi(\theta)$$
Posterior Distribution
$$\pi(\theta\mid x) = \frac{f(x\mid\theta)\,\pi(\theta)} {\int f(x\mid\theta)\,\pi(\theta)\,d\theta}$$
$$\pi(\theta\mid x)\propto f(x\mid\theta)\pi(\theta)$$

Conjugate Prior
$$\pi(\theta\mid x)\in\text{same family as }\pi(\theta)$$
- Closed-form posterior  
- Algebraic updating  

```python {run}
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# -----------------------------
# Data (Bernoulli observations)
# -----------------------------
np.random.seed(0)
n = 20
x = np.random.binomial(1, 0.6, size=n)
successes = x.sum()
failures = n - successes

# -----------------------------
# Prior (Beta)
# -----------------------------
a0, b0 = 2, 2        # prior hyperparameters
a_post = a0 + successes
b_post = b0 + failures

theta = np.linspace(0, 1, 500)

prior = beta.pdf(theta, a0, b0)
likelihood = theta**successes * (1-theta)**failures
likelihood /= likelihood.max()   # scale for visualization
posterior = beta.pdf(theta, a_post, b_post)

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

# (1) Prior
ax = axes[0, 0]
ax.plot(theta, prior, linewidth=2)
ax.set_title("Prior distribution")
ax.set_xlabel("Parameter value")
ax.set_ylabel("Density")
ax.grid(alpha=0.3)

# (2) Likelihood
ax = axes[0, 1]
ax.plot(theta, likelihood, linewidth=2)
ax.set_title("Likelihood (scaled)")
ax.set_xlabel("Parameter value")
ax.grid(alpha=0.3)

# (3) Posterior
ax = axes[1, 0]
ax.plot(theta, posterior, linewidth=2)
ax.set_title("Posterior distribution")
ax.set_xlabel("Parameter value")
ax.set_ylabel("Density")
ax.grid(alpha=0.3)

# (4) Prior → Posterior (overlay)
ax = axes[1, 1]
ax.plot(theta, prior, linestyle="--", linewidth=2, label="Prior")
ax.plot(theta, posterior, linewidth=2, label="Posterior")
ax.set_title("Bayesian updating (Conjugacy)")
ax.set_xlabel("Parameter value")
ax.legend()
ax.grid(alpha=0.3)

plt.show()
```

Posterior Predictive Distribution
$$p(x_{\text{new}}\mid x) = \int f(x_{\text{new}}\mid\theta)\,\pi(\theta\mid x)\,d\theta$$

Posterior Mean (Bayes Estimator)
$$\hat\theta_{\text{Bayes}} = \mathbb{E}[\theta\mid x]$$

Posterior Mode (MAP)
$$\hat\theta_{\text{MAP}} = \arg\max_\theta \pi(\theta\mid x)$$
$$\hat\theta_{\text{MAP}} = \arg\max_\theta \{\ell(\theta)+\log\pi(\theta)\}$$

Credible Interval
$$\mathbb{P}(\theta\in C_\alpha\mid x)=1-\alpha$$

Bayesian Hypothesis Testing
$$H_0:\theta\in\Theta_0, \quad H_1:\theta\in\Theta_1$$
Bayesian Hypothesis Testing (core ideas)

- Bayes Factor
$$BF_{01} = \frac{p(x\mid H_0)}{p(x\mid H_1)} = \frac{\int_{\Theta_0} f(x\mid\theta)\pi_0(\theta)\,d\theta}{\int_{\Theta_1} f(x\mid\theta)\pi_1(\theta)\,d\theta}$$
- Posterior Odds
$$\frac{\mathbb{P}(H_0\mid x)}{\mathbb{P}(H_1\mid x)} = BF_{01} \times \frac{\mathbb{P}(H_0)}{\mathbb{P}(H_1)}$$
Decision is based on posterior odds (or Bayes factor), not p-values.

## Loss Function
$$L(\theta,a)$$
Posterior Risk
$$R(a\mid x)=\mathbb{E}[L(\theta,a)\mid x]$$

Bayes Rule (Optimal Decision)
$$a^*(x) = \arg\min_a R(a\mid x)$$

## Error & Loss
Squared Error
$$L(\theta,a)=(\theta-a)^2 \quad\Rightarrow \quad a^*=\mathbb{E}[\theta\mid x]$$

Absolute Error
$$L(\theta,a)=|\theta-a| \quad\Rightarrow \quad a^*=\text{posterior median}$$

0–1 Loss
$$L(\theta,a)=\mathbf{1}(\theta\neq a) \quad\Rightarrow \quad a^*=\text{posterior mode}$$

---
# 5. Model Checking
- Inference is conditional on the assumed model.  
- Model checking evaluates whether the assumed model is compatible with the observed data.

## Pearson Chi-Square Test
$$\chi^2 = \sum_{i=1}^k \frac{(O_i-E_i)^2}{E_i}$$
$$\chi^2 \xrightarrow{d} \chi^2_{k-p-1}$$
- Tests global goodness-of-fit using observed vs expected frequencies.

## Q–Q Plot
$$F^{-1}\\left(\frac{i-0.5}{n}\right) \quad \text{vs} \quad X_{(i)}$$
- Compares empirical quantiles to theoretical quantiles.
- Diagnoses distributional misspecification.
```python {run}
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(0)
n = 200

def qq_plot(ax, data, title):
    data = np.sort(data)
    probs = (np.arange(1, n+1) - 0.5) / n
    theo = norm.ppf(probs)

    ax.scatter(theo, data, s=15, alpha=0.7)
    # reference line
    slope = np.std(data)
    intercept = np.mean(data)
    ax.plot(theo, intercept + slope*theo, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Empirical quantiles")
    ax.grid(alpha=0.3)

# -----------------------------
# Data
# -----------------------------
x_normal = np.random.normal(0, 1, n)            # correct
x_heavy = np.random.standard_t(df=3, size=n)    # heavy tails
x_skew = np.random.exponential(scale=1, size=n) # skewed
x_var = np.random.normal(0, 2, n)                # wrong variance

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

qq_plot(axes[0,0], x_normal, "Correct specification (Normal)")
qq_plot(axes[0,1], x_heavy, "Heavy tails")
qq_plot(axes[1,0], x_skew, "Skewness")
qq_plot(axes[1,1], x_var, "Variance misspecification")

plt.show()
```

## Cross-Validation
$$\text{CV} = \frac{1}{K} \sum_{k=1}^K \mathcal{L}\big(x^{(k)}_{\text{test}},\hat\theta^{(-k)}\big)$$
- Evaluates predictive performance on held-out data.
- Detects overfitting and model instability.

## Checking
Residual-Based Checking
$$r_i = X_i - \hat{\mu}_i$$
- Checks independence, homoscedasticity, and distributional assumptions.

Posterior Predictive Check (Bayesian)
$$x^{\text{rep}} \sim p(x^{\text{rep}}\mid x)$$
$$T(x^{\text{rep}})\quad \text{vs} \quad T(x)$$
- Compares replicated data with observed data under the fitted model.
  
## Prior–Data Conflict
Prior Predictive Distribution
$$p(x) = \int f(x\mid\theta)\pi(\theta)\,d\theta$$
Prior–Data Conflict Check
$$\mathbb{P}\big(T(X)\le T(x_{\text{obs}})\big)$$
