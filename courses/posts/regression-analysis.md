---
title: Regression Analysis
category: STATISTICS
semester: 2023 F
---

# Regression Analysis

## 1. What this course is about

Regression studies how a response $Y$ changes with predictors $X$. The course emphasizes four goals: **explanation** (which $X$ matter), **inference** (whether effects are real), **prediction** (generalization), and **diagnosis** (model adequacy).

Flow: simple linear regression → inference and prediction → fit and diagnostics → multiple regression → variable contribution and comparison → extensions (nonlinearity, interactions, categorical predictors).

---

## 2. The core idea of regression

The mean of $Y$ moves with predictors; observations scatter around that mean because of error.

$$
Y_i=\beta_0+\beta_1X_i+\varepsilon_i, \qquad \varepsilon_i \overset{iid}{\sim} N(0,\sigma^2)
$$

Hence $E(Y_i)=\beta_0+\beta_1X_i$ and $\mathrm{Var}(Y_i)=\sigma^2$. Everything extends **mean structure + random error**.

---

## 3. Estimation: fitting the model

Unknown $(\beta_0,\beta_1)$ are estimated by data. Fitted values $\hat Y_i=b_0+b_1X_i$, residuals $e_i=Y_i-\hat Y_i$. Least squares minimizes $\sum_i (Y_i-\hat Y_i)^2$.

$$
b_1=\frac{\sum (X_i-\bar X)(Y_i-\bar Y)}{\sum (X_i-\bar X)^2},
\qquad
b_0=\bar Y-b_1\bar X
$$

Error variance: $\mathrm{MSE}=\mathrm{SSE}/(n-2)$, $s=\sqrt{\mathrm{MSE}}$.

### Animation: least squares “locks in” the slope

```python {run}
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import base64
import os

np.random.seed(7)
n = 28
x = np.linspace(0.5, 9.5, n)
true_b0, true_b1 = 1.2, 0.85
y = true_b0 + true_b1 * x + np.random.randn(n) * 0.9

xbar, ybar = x.mean(), y.mean()
Sxx = np.sum((x - xbar) ** 2)
Sxy = np.sum((x - xbar) * (y - ybar))
b1_ols = Sxy / Sxx
b0_ols = ybar - b1_ols * xbar

b1_grid = np.linspace(-0.2, b1_ols, 18)
fig, ax = plt.subplots(figsize=(8, 5), facecolor='#1a1a1a')
ax.set_facecolor('#1a1a1a')


def sse_for_slope(b1_try):
    b0_try = ybar - b1_try * xbar
    yhat = b0_try + b1_try * x
    return np.sum((y - yhat) ** 2)


def draw_frame(k):
    ax.clear()
    ax.set_facecolor('#1a1a1a')
    b1 = b1_grid[k]
    b0 = ybar - b1 * xbar
    yhat = b0 + b1 * x
    sse = sse_for_slope(b1)
    ax.scatter(x, y, c='#6b9ac4', s=36, zorder=3)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, b0 + b1 * xs, color='#c9a961', lw=2.2, zorder=2)
    for i in range(n):
        ax.plot([x[i], x[i]], [y[i], yhat[i]], color='#444', lw=0.8, zorder=1)
    ax.set_title(f'Slope → OLS   SSE = {sse:.2f}   (final SSE ≈ {sse_for_slope(b1_ols):.2f})', color='#ccc', fontsize=11)
    ax.set_xlabel('X', color='#888')
    ax.set_ylabel('Y', color='#888')
    ax.tick_params(colors='#888')
    ax.grid(True, alpha=0.2)


anim = animation.FuncAnimation(fig, draw_frame, frames=len(b1_grid), interval=120, repeat=True)
_gif_path = '_reg_ols_anim.gif'
anim.save(_gif_path, writer='pillow', fps=8)
plt.close('all')
with open(_gif_path, 'rb') as f:
    _ANIM_GIF = base64.b64encode(f.read()).decode()
try:
    os.remove(_gif_path)
except OSError:
    pass
print('OLS:', f'b0={b0_ols:.3f}, b1={b1_ols:.3f}')
```

---

## 4. Inference: uncertainty in coefficients

Point estimates are not enough—we need uncertainty. For the slope,

$$
t=\frac{b_1-\beta_{1,0}}{s\{b_1\}} \sim t_{n-2},
\qquad
\text{CI: } b_1 \pm t_{1-\alpha/2,n-2}\, s\{b_1\}.
$$

Overall model significance: $F=\mathrm{MSR}/\mathrm{MSE}\sim F_{1,n-2}$. In simple regression, $F=t^2$ for the slope test.

---

## 5. Explaining variation

Total variation in $Y$ splits into **explained** (model) and **unexplained** (residuals):

$$
\mathrm{SST}=\mathrm{SSR}+\mathrm{SSE}.
$$

This underlies ANOVA, $R^2$, $F$-tests, extra sums of squares, and partial $R^2$.

### Visualization: SST, SSR, SSE

```python {run}
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3)
n = 40
x = np.linspace(0, 10, n)
y = 1.0 + 0.7 * x + np.random.randn(n) * 1.1
ybar = y.mean()
b1 = np.sum((x - x.mean()) * (y - ybar)) / np.sum((x - x.mean()) ** 2)
b0 = ybar - b1 * x.mean()
yhat = b0 + b1 * x

SST = np.sum((y - ybar) ** 2)
SSR = np.sum((yhat - ybar) ** 2)
SSE = np.sum((y - yhat) ** 2)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor='#1a1a1a')
for ax in axes:
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='#888')

axes[0].bar(['SST', 'SSR', 'SSE'], [SST, SSR, SSE], color=['#6b9ac4', '#c9a961', '#97c4a0'], edgecolor='#333')
axes[0].set_title('Variation decomposition', color='#ccc')
axes[0].set_ylabel('Sum of squares', color='#888')

axes[1].scatter(x, y, c='#6b9ac4', s=30, label='Data', zorder=3)
axes[1].axhline(ybar, color='#888', ls='--', lw=1, label=r'$\bar{Y}$')
axes[1].plot(x, yhat, color='#c9a961', lw=2, label='Fitted line')
for i in range(0, n, 5):
    axes[1].plot([x[i], x[i]], [ybar, yhat[i]], color='#555', lw=1)
axes[1].set_title('SSR: fitted − Ȳ  (vertical ticks)', color='#ccc')
axes[1].legend(facecolor='#222', edgecolor='#444', labelcolor='#ccc')
plt.tight_layout()
```

---

## 6. Correlation and explanatory power

Sample correlation:

$$
r=
\frac{\sum (X_i-\bar X)(Y_i-\bar Y)}
{\sqrt{\sum (X_i-\bar X)^2\sum (Y_i-\bar Y)^2}}.
$$

Explanatory power:

$$
R^2=\frac{\mathrm{SSR}}{\mathrm{SST}}=1-\frac{\mathrm{SSE}}{\mathrm{SST}}.
$$

In simple regression, $R^2=r^2$. Because $R^2$ rises when predictors are added, use adjusted $R^2$:

$$
R^2_{\text{adj}}=
1-\frac{(1-R^2)(n-1)}{n-p}.
$$

### Visualization: $r$ and $R^2$

```python {run}
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(11)
n = 35
x = np.linspace(0, 8, n)
y = 2 + 0.4 * x + np.random.randn(n) * 1.0
ybar, xbar = y.mean(), x.mean()
Sxx = np.sum((x - xbar) ** 2)
Syy = np.sum((y - ybar) ** 2)
Sxy = np.sum((x - xbar) * (y - ybar))
r = Sxy / np.sqrt(Sxx * Syy)
b1 = Sxy / Sxx
b0 = ybar - b1 * xbar
yhat = b0 + b1 * x
SSR = np.sum((yhat - ybar) ** 2)
SST = np.sum((y - ybar) ** 2)
R2 = SSR / SST

fig, ax = plt.subplots(figsize=(6.5, 5), facecolor='#1a1a1a')
ax.set_facecolor('#1a1a1a')
ax.scatter(x, y, c='#6b9ac4', s=40, zorder=3)
ax.plot(x, yhat, color='#c9a961', lw=2)
ax.set_title(f'Simple linear fit   r = {r:.3f}   R² = {R2:.3f}   (r² = {r**2:.3f})', color='#ccc')
ax.set_xlabel('X', color='#888')
ax.set_ylabel('Y', color='#888')
ax.tick_params(colors='#888')
ax.grid(True, alpha=0.2)
```

---

## 7. Prediction

**Mean response** at $x_h$ vs **new observation** at $x_h$: the prediction interval is wider because it adds future noise. In multiple regression, $\hat Y_h=x_h^T\hat\beta$. Rule: **PI width > CI width for the mean**.

### Visualization: confidence vs prediction bands

```python {run}
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
n = 35
x = np.sort(np.random.uniform(0, 10, n))
X = np.column_stack([np.ones(n), x])
beta = np.array([1.5, 0.9])
y = X @ beta + np.random.randn(n) * 1.3

b, *_ = np.linalg.lstsq(X, y, rcond=None)
yhat = X @ b
e = y - yhat
MSE = np.sum(e ** 2) / (n - 2)
xh = np.linspace(0, 10, 120)
Xh = np.column_stack([np.ones_like(xh), xh])
yhat_h = Xh @ b

xbar = x.mean()
Sxx = np.sum((x - xbar) ** 2)
se_mean = np.sqrt(MSE * (1 / n + (xh - xbar) ** 2 / Sxx))
se_pred = np.sqrt(MSE * (1 + 1 / n + (xh - xbar) ** 2 / Sxx))
t = 2.042  # approx t_{0.975, 30} for illustration
ci_lo, ci_hi = yhat_h - t * se_mean, yhat_h + t * se_mean
pi_lo, pi_hi = yhat_h - t * se_pred, yhat_h + t * se_pred

fig, ax = plt.subplots(figsize=(8, 5), facecolor='#1a1a1a')
ax.set_facecolor('#1a1a1a')
ax.scatter(x, y, c='#6b9ac4', s=36, zorder=4, label='Data')
ax.plot(xh, yhat_h, color='#c9a961', lw=2, label='Fitted mean')
ax.fill_between(xh, ci_lo, ci_hi, color='#c9a961', alpha=0.2, label='Mean response (approx 95% CI)')
ax.fill_between(xh, pi_lo, pi_hi, color='#6b9ac4', alpha=0.12, label='Prediction (approx 95% PI)')
ax.set_title('Narrower band = uncertainty in mean; wider = new Y', color='#ccc')
ax.legend(facecolor='#222', edgecolor='#444', labelcolor='#ccc', loc='upper left', fontsize=8)
ax.tick_params(colors='#888')
ax.grid(True, alpha=0.2)
plt.tight_layout()
```

---

## 8. Simultaneous inference

Many separate 95% intervals do not give 95% joint coverage. Common fixes: **Bonferroni**, **Working–Hotelling**. Stronger joint control ⇒ wider intervals.

---

## 9. Model adequacy and diagnostics

Check **linearity**, **constant variance**, **normality**, **independence**, and **influence**. Residuals $e_i=Y_i-\hat Y_i$; residual plots should look **unstructured**. Curvature ⇒ mean misspec; funnel ⇒ heteroscedasticity; patterns over order ⇒ dependence.

### Visualization: residual patterns

```python {run}
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
n = 40
x = np.linspace(0, 10, n)

# OK
y1 = 1 + 0.5 * x + np.random.randn(n) * 0.8
r1 = y1 - (np.poly1d(np.polyfit(x, y1, 1))(x))

# Curvature misspec
y2 = 1 + 0.5 * x + 0.08 * (x - 5) ** 2 + np.random.randn(n) * 0.5
r2 = y2 - (np.poly1d(np.polyfit(x, y2, 1))(x))

# Funnel
y3 = 1 + 0.5 * x + np.random.randn(n) * (0.3 + 0.15 * x)
r3 = y3 - (np.poly1d(np.polyfit(x, y3, 1))(x))

# Mild autocorrelation-like trend in residuals (synthetic)
eps = np.zeros(n)
for i in range(1, n):
    eps[i] = 0.6 * eps[i - 1] + np.random.randn()
y4 = 1 + 0.5 * x + eps
r4 = y4 - (np.poly1d(np.polyfit(x, y4, 1))(x))

fig, axes = plt.subplots(2, 2, figsize=(9, 7), facecolor='#1a1a1a')
titles = ['OK: random scatter', 'Curvature (nonlinear mean)', 'Funnel (heteroscedasticity)', 'Structured residuals (dependence suspect)']
series = [r1, r2, r3, r4]
for ax, r, t in zip(axes.flat, series, titles):
    ax.set_facecolor('#1a1a1a')
    ax.scatter(x, r, c='#6b9ac4', s=28)
    ax.axhline(0, color='#666', lw=1)
    ax.set_title(t, color='#ccc', fontsize=10)
    ax.tick_params(colors='#888')
    ax.set_xlabel('X', color='#888')
    ax.set_ylabel('Residual', color='#888')
plt.tight_layout()
```

---

## 10. When the model is not good enough

**Lack-of-fit:** $\mathrm{SSE}=\mathrm{SSPE}+\mathrm{SSLF}$ separates pure error from systematic lack of fit.

**Transformations:** e.g. $\log X$, $\sqrt X$, $1/X$, Box–Cox. Rule of thumb: **mean-structure issues** → transform $X$ or add polynomials; **variance/normality** → often transform $Y$.

---

## 11. Multiple linear regression

$$
Y_i=\beta_0+\beta_1X_{i1}+\cdots+\beta_{p-1}X_{i,p-1}+\varepsilon_i
$$

Matrix form: $Y=X\beta+\varepsilon$, $\hat\beta=(X^TX)^{-1}X^TY$, $\hat Y=HY$ with $H=X(X^TX)^{-1}X^T$, $e=(I-H)Y$. Each slope is interpreted **holding other predictors fixed**.

---

## 12. Inference in multiple regression

Global test $H_0:\beta_1=\cdots=\beta_{p-1}=0$ uses $F=\mathrm{MSM}/\mathrm{MSE}$. Individual $t=(b_k-\beta_k^\ast)/s\{b_k\}\sim t_{n-p}$. Generally $t_k^2\neq F_{\text{global}}$: the overall $F$ can be significant while some $t$’s are not.

---

## 13. Variable contribution

Extra sum of squares $\mathrm{SSR}(A\mid B)=\mathrm{SSE}(B)-\mathrm{SSE}(A,B)$. Partial $F$ compares nested models; partial $R^2_{Y A\mid B}=\mathrm{SSR}(A\mid B)/\mathrm{SSE}(B)$. Overall $R^2$ is total fit; partial $R^2$ is **incremental** contribution.

---

## 14. Type I / II SS and multicollinearity

**Type I SS** depends on **entry order**; **Type II SS** adjusts for the rest. Strong correlation ⇒ **multicollinearity**: unstable $\hat\beta$, large SEs, weak $t$’s, harder interpretation than in simple regression.

---

## 15. Extensions of the linear model

**Polynomial:** $Y=\beta_0+\beta_1X+\beta_2X^2+\varepsilon$.

**Interaction:** $Y=\beta_0+\beta_1X_1+\beta_2X_2+\beta_3X_1X_2+\varepsilon$.

**Qualitative predictors:** $c$ levels need $c-1$ dummies; interactions allow **different slopes** across groups.

### Visualization: linear vs quadratic fit

```python {run}
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19)
x = np.linspace(-2, 2, 45)
y = 0.5 - 0.3 * x + 0.9 * x ** 2 + np.random.randn(len(x)) * 0.35

X_lin = np.column_stack([np.ones_like(x), x])
X_quad = np.column_stack([np.ones_like(x), x, x ** 2])
b_lin, *_ = np.linalg.lstsq(X_lin, y, rcond=None)
b_quad, *_ = np.linalg.lstsq(X_quad, y, rcond=None)
xs = np.linspace(x.min(), x.max(), 120)

fig, ax = plt.subplots(figsize=(7, 5), facecolor='#1a1a1a')
ax.set_facecolor('#1a1a1a')
ax.scatter(x, y, c='#6b9ac4', s=36, zorder=3, label='Data')
ax.plot(xs, b_lin[0] + b_lin[1] * xs, color='#d4a5a5', lw=2, ls='--', label='Linear (misspecified)')
ax.plot(xs, b_quad[0] + b_quad[1] * xs + b_quad[2] * xs ** 2, color='#c9a961', lw=2, label='Quadratic (same linear framework)')
ax.set_title('Curvature: polynomial terms stay inside linear model', color='#ccc')
ax.legend(facecolor='#222', edgecolor='#444', labelcolor='#ccc')
ax.tick_params(colors='#888')
ax.grid(True, alpha=0.2)
plt.tight_layout()
```

---

## 16. The big picture

1. **Specify** mean structure and predictors  
2. **Fit** with least squares  
3. **Quantify uncertainty** (intervals, $t$, $F$, simultaneous methods)  
4. **Assess fit** ($R^2$, adjusted $R^2$, extra SS, partial $R^2$)  
5. **Diagnose** residuals, variance, influence  
6. **Refine** (transformations, polynomials, interactions, dummies)

---

## 17. What to remember

- Regression = **mean structure + error**; LS minimizes RSS.  
- Always report **uncertainty** with estimates; **mean vs new-$Y$** prediction tasks differ.  
- High $R^2$ $\not\Rightarrow$ valid model—**diagnostics matter**.  
- Multiple regression: slopes are **ceteris paribus**; **extra SS / partial $F$ / partial $R^2$** measure added value.  
- **Multicollinearity** hurts stability and interpretation.  
- Polynomials, interactions, and dummies remain **linear in parameters**.
