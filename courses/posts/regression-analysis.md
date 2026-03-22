---
title: Regression Analysis
category: STATISTICS
semester: 2023 S
---

# 1. Regression as a Model for Mean Response

## Core regression model

$$
Y_i=\beta_0+\beta_1X_{i1}+\cdots+\beta_{p-1}X_{i,p-1}+\varepsilon_i,\qquad \varepsilon_i\overset{iid}{\sim}N(0,\sigma^2)
$$

$$
E(Y_i\mid X_i)=\beta_0+\beta_1X_{i1}+\cdots+\beta_{p-1}X_{i,p-1},\qquad \mathrm{Var}(Y_i\mid X_i)=\sigma^2
$$

- Regression describes how the **conditional mean** of $Y$ moves with predictors.
- The whole course repeats one template:

$$\text{response}=\text{systematic mean structure}+\text{random error}$$

## What regression tries to do

- Explain how predictors associate with the response.
- Test whether effects are statistically meaningful.
- Predict future outcomes.
- Judge whether the model is adequate.

---

# 2. Fitting the Model: Least Squares

## Simple linear regression

$$Y_i=\beta_0+\beta_1X_i+\varepsilon_i$$

## Fitted line and residuals

$$\hat Y_i=b_0+b_1X_i,\qquad e_i=Y_i-\hat Y_i$$

- The fitted line estimates the **mean response**; residuals are what the line does not explain.

## Least squares criterion

$$\min_{b_0,b_1}\sum_{i=1}^n (Y_i-\hat Y_i)^2$$

## Closed-form estimators

$$b_1=\frac{\sum (X_i-\bar X)(Y_i-\bar Y)}{\sum (X_i-\bar X)^2},\qquad b_0=\bar Y-b_1\bar X$$

## Error variance estimate

$$\mathrm{SSE}=\sum e_i^2,\qquad \mathrm{MSE}=\frac{\mathrm{SSE}}{n-2},\qquad s=\sqrt{\mathrm{MSE}}$$

- Least squares is the computational hub: tests, intervals, prediction, and diagnostics all sit on $(\hat Y_i,e_i)$.

### Figure — fit, residuals, and residual vs.\ $X$

```python {run}
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(11)
n = 70
x = np.linspace(0.3, 9.7, n)
y = 1.25 + 0.95 * x + np.random.normal(0, 1.15, n)

xbar, ybar = x.mean(), y.mean()
Sxx = np.sum((x - xbar) ** 2)
Sxy = np.sum((x - xbar) * (y - ybar))
b1 = Sxy / Sxx
b0 = ybar - b1 * xbar
y_hat = b0 + b1 * x
res = y - y_hat
sse = np.sum(res ** 2)
mse = sse / (n - 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor="#1a1a1a")
for ax in (ax1, ax2):
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="#b0b0b0")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.grid(True, alpha=0.22)

ax1.scatter(x, y, s=22, c="#6b9ac4", alpha=0.85, edgecolors="none", label="Data", zorder=3)
xs = np.array([x.min(), x.max()])
ax1.plot(xs, b0 + b1 * xs, color="#c9a961", lw=2.4, label="OLS fit", zorder=2)
step = max(1, n // 12)
for i in range(0, n, step):
    ax1.plot([x[i], x[i]], [y[i], y_hat[i]], color="#555", lw=1.0, zorder=1)
ax1.set_title("Mean structure + residuals", color="#e0e0e0", fontsize=11)
ax1.set_xlabel("$X$", color="#aaa")
ax1.set_ylabel("$Y$", color="#aaa")
ax1.legend(frameon=True, facecolor="#252525", edgecolor="#444", labelcolor="#ccc")

ax2.axhline(0, color="#666", lw=1)
ax2.scatter(x, res, s=22, c="#97c4a0", alpha=0.85, edgecolors="none")
ax2.set_title("Residual vs $X$ (ideal: shapeless cloud)", color="#e0e0e0", fontsize=11)
ax2.set_xlabel("$X$", color="#aaa")
ax2.set_ylabel(r"$e = Y - \hat Y$", color="#aaa")

fig.suptitle(
    rf"$\hat\beta_0={b0:.2f},\ \hat\beta_1={b1:.2f}$  |  SSE={sse:.1f}, MSE={mse:.3f}",
    color="#b0b0b0",
    fontsize=10,
    y=1.02,
)
plt.tight_layout()
plt.show()
```

### Animation — slope walks toward OLS (SSE falls)

```python {run}
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import base64
import os

np.random.seed(7)
n = 32
x = np.linspace(0.5, 9.5, n)
y = 1.1 + 0.88 * x + np.random.randn(n) * 0.85
xbar, ybar = x.mean(), y.mean()
Sxx = np.sum((x - xbar) ** 2)
Sxy = np.sum((x - xbar) * (y - ybar))
b1_ols = Sxy / Sxx
b0_ols = ybar - b1_ols * xbar

b1_seq = np.linspace(-0.05, b1_ols, 20)

fig, ax = plt.subplots(figsize=(6.8, 4.2), facecolor="#1a1a1a")
ax.set_facecolor("#1a1a1a")


def frame(k):
    ax.clear()
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="#888")
    ax.grid(True, alpha=0.2)
    b1 = b1_seq[k]
    b0 = ybar - b1 * xbar
    yhat = b0 + b1 * x
    sse = np.sum((y - yhat) ** 2)
    ax.scatter(x, y, c="#6b9ac4", s=36, zorder=3)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, b0 + b1 * xs, color="#c9a961", lw=2.2)
    for i in range(n):
        ax.plot([x[i], x[i]], [y[i], yhat[i]], color="#444", lw=0.7)
    ax.set_title(f"Slope → OLS    SSE = {sse:.2f}", color="#ddd", fontsize=11)
    ax.set_xlabel("X", color="#888")
    ax.set_ylabel("Y", color="#888")


anim = animation.FuncAnimation(fig, frame, frames=len(b1_seq), interval=110, repeat=True)
path = "_reg_slopes.gif"
anim.save(path, writer="pillow", fps=9)
plt.close("all")
with open(path, "rb") as f:
    _ANIM_GIF = base64.b64encode(f.read()).decode()
try:
    os.remove(path)
except OSError:
    pass
print("Final OLS:", round(b1_ols, 3))
```

---

# 3. Inference: Quantifying Uncertainty

## Slope inference (SLR)

$$t=\frac{b_1-\beta_{1,0}}{s\{b_1\}}\sim t_{n-2}$$

$$b_1\pm t_{1-\alpha/2,n-2}\,s\{b_1\}$$

- Estimation is not enough—you need whether the slope departs from a reference (often $0$) and how wide uncertainty is.

## Overall model significance

$$F=\frac{\mathrm{MSR}}{\mathrm{MSE}}\sim F_{1,n-2}$$

In simple linear regression, $F=t^2$ for testing the slope.

## Explaining variation

$$\mathrm{SST}=\mathrm{SSR}+\mathrm{SSE}$$

$$R^2=\frac{\mathrm{SSR}}{\mathrm{SST}}=1-\frac{\mathrm{SSE}}{\mathrm{SST}}$$

- $R^2$ is explained fraction of variation—not a substitute for diagnostics or causal claims.

---

# 4. Prediction: Mean Response vs New Observation

## Two different targets

At a fixed $x_h$:

- **Mean response** — average $Y$ at $x_h$.
- **New observation** — one future draw at $x_h$.

## Interval estimation

Mean response CI:

$$\hat Y_h \pm t_{1-\alpha/2,n-2}\,s\{\hat Y_h\}$$

Prediction interval:

$$\hat Y_h \pm t_{1-\alpha/2,n-2}\,s_{\mathrm{pred}}$$

- Prediction intervals are **wider**: they add irreducible noise on top of estimation uncertainty.

### What to remember

- Estimating the mean is easier than predicting a single future point—that distinction drives applied work.

---

# 5. Model Checking and Diagnostics

## Residual-based diagnostics

$$e_i=Y_i-\hat Y_i$$

- Significance does not imply correct specification; check the assumptions inference leans on.

## Main assumptions

- Linearity of the mean.
- Constant variance.
- Approximate normality (especially for small $n$).
- Independence.
- Outliers / influence.

## Residual plot logic

- Random cloud $\to$ plausible fit.
- Curvature $\to$ wrong mean shape.
- Funnel $\to$ heteroscedasticity.
- Pattern over time/order $\to$ dependence.

## Lack-of-fit and transformation

$$\mathrm{SSE}=\mathrm{SSPE}+\mathrm{SSLF}$$

- Lack-of-fit separates “line is wrong” from pure noise.
- Transformations or richer terms follow when plots or tests demand them.

---

# 6. Multiple Linear Regression

## General model

$$Y_i=\beta_0+\beta_1X_{i1}+\cdots+\beta_{p-1}X_{i,p-1}+\varepsilon_i$$

## Matrix form

$$Y=X\beta+\varepsilon,\qquad \hat\beta=(X^TX)^{-1}X^TY$$

- Same logic as SLR, more predictors. Interpretation rule:

$$\beta_k \text{ is the change in mean } Y \text{ per unit } X_k \textbf{ holding other } X \text{ fixed.}$$

## Inference in MLR

Global test $H_0:\beta_1=\cdots=\beta_{p-1}=0$:

$$F=\frac{\mathrm{MSM}}{\mathrm{MSE}}$$

Per coefficient:

$$t=\frac{b_k-\beta_k^*}{s\{b_k\}}\sim t_{n-p}$$

- The overall $F$ can be significant while some individual $t$’s are not.

---

# 7. Variable Contribution and Model Comparison

## Extra sum of squares

$$\mathrm{SSR}(A\mid B)=\mathrm{SSE}(B)-\mathrm{SSE}(A,B)$$

## Partial $F$-test

$$F=\frac{(\mathrm{SSE}_R-\mathrm{SSE}_F)/(df_R-df_F)}{\mathrm{SSE}_F/df_F}$$

## Partial $R^2$

$$R^2_{Y,A\mid B}=\frac{\mathrm{SSR}(A\mid B)}{\mathrm{SSE}(B)}$$

- Importance in MLR is often **incremental**: what $A$ adds on top of $B$.

## Multicollinearity

- Strong linear relationships among predictors inflate SEs and destabilize $\hat\beta$.
- Hurts **interpretation** more often than raw predictive skill.

---

# 8. Extensions of the Linear Model

## Polynomial regression

$$Y=\beta_0+\beta_1X+\beta_2X^2+\varepsilon$$

## Interaction

$$Y=\beta_0+\beta_1X_1+\beta_2X_2+\beta_3X_1X_2+\varepsilon$$

## Categorical predictors

- Factors enter via dummies; interactions allow different **levels** and **slopes** by group.
- Still the **same linear model**—linear in $\beta$, possibly nonlinear in $X$.

---

# 9. Big Picture

## Regression workflow

1. Specify a mean structure.
2. Fit by least squares.
3. Quantify uncertainty (tests, intervals).
4. Assess explanatory power ($R^2$, partial tools).
5. Diagnose adequacy.
6. Refine (transformations, polynomials, interactions, dummies).

## What to remember

- **Mean + error**; LS gives $\hat Y$ and $e$.
- Good practice pairs **estimation** with **diagnostics**.
- $R^2$ helps, never completes the story.
- **New-observation prediction** is harder than **mean estimation**.
- In MLR, coefficients are **ceteris paribus**; extra SS / partial $F$ / partial $R^2$ measure **added** contribution.
- Polynomials, interactions, and dummies stay inside the linear-model framework.
