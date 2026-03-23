---
title: Statistical Machine Learning
category: AI/ML
semester: 2025 S
---


# 1. Learning, Risk, and Generalization

## Learning as Search in Hypothesis Space
Statistical machine learning starts from the view that learning is a search problem.  
Given an input space $\mathcal X$ and output space $\mathcal Y$, we seek a function
$$
h:\mathcal X \to \mathcal Y
$$
from a hypothesis space
$$
H=\{h_1,h_2,h_3,\dots\}.
$$
The role of a learning algorithm is to search this space and return a hypothesis that performs well under a chosen loss. This perspective unifies many models: linear regression, logistic regression, k-NN, SVM, neural networks, and kernel methods all differ mainly in the hypothesis space and optimization principle they impose.

## Expected Risk and Empirical Risk
The true statistical objective is population risk:
$$
R(h)=\mathbb E_{(x,y)\sim P}[\ell(h(x),y)].
$$
For binary classification with 0-1 loss,
$$
R(h)=\int \mathbf{1}(h(x)\neq y)\,P(x,y)\,dx\,dy.
$$
In practice, the distribution $P(x,y)$ is unknown, so training proceeds by minimizing empirical risk:
$$
\hat R(h)=\frac{1}{N}\sum_{i=1}^N \ell(h(x_i),y_i),
$$
and under 0-1 loss,
$$
\hat R(h)=\frac{1}{N}\sum_{i=1}^N \mathbf{1}(h(x_i)\neq y_i).
$$
This is the core idea of empirical risk minimization:
$$
\hat h=\arg\min_{h\in H}\hat R(h).
$$

---

# 2. Bayes Optimal Classification and k-Nearest Neighbors

## Bayes Decision Rule
In classification, the ideal predictor under the true distribution is the Bayes classifier:
$$
f^*(x)=\arg\max_{c\in[C]} \Pr(c\mid x).
$$
This classifier minimizes expected classification error, and its risk is always optimal:
$$
R(f^*)\le R(f)
$$
for any classifier $f$. It serves as a theoretical lower bound on achievable performance under the data-generating distribution.

An important implication is that even the optimal classifier may still make mistakes.  
If class-conditional distributions overlap, then irreducible error remains:
$$
R(f^*)>0.
$$

## k-Nearest Neighbors as a Local Nonparametric Rule
k-NN is a nonparametric method that makes predictions directly from the training sample.  
For a query point $x$, define the nearest neighbor:
$$
nn_1(x)=\arg\min_{n\in[N]}\|x-x_n\|_2.
$$
More generally, the set of the $k$ nearest neighbors is
$$
kNN(x)=\{nn_1(x),\dots,nn_k(x)\}.
$$
Classification is then based on majority vote:
$$
V_c=\sum_{n\in kNN(x)}\mathbf{1}(y_n=c),
$$
$$
h_k(x)=\arg\max_{c\in[C]}V_c.
$$
The prediction rule is therefore local: labels are inferred from nearby observations in feature space.

## Statistical Interpretation of k-NN
Unlike parametric methods, k-NN does not estimate a global finite-dimensional parameter.  
Instead, it approximates the local conditional label structure. This makes it flexible, but also heavily dependent on distance, feature scaling, and sample density.

For 1-NN, the Cover–Hart result gives a classical asymptotic bound:
$$
R(f^*)\le \lim_{N\to\infty}\mathbb E[R(f_N)]\le 2R(f^*).
$$
So even an extremely simple local rule remains theoretically competitive.

## Distance and Normalization
Because k-NN depends directly on metric geometry, normalization is essential:
$$
\tilde x_j=\frac{x_j-\mu_j}{\sigma_j}.
$$
Without scaling, one large-magnitude feature can dominate Euclidean distance and distort neighborhood structure. This issue extends beyond k-NN to clustering and many other distance-based methods.

---

# 3. Linear Regression as Statistical Estimation

## Linear Model
A central supervised learning model is linear regression:
$$
y_i=w^\top x_i+b+\varepsilon_i.
$$
Using augmented features, this is often written more compactly as
$$
y_i=w^\top x_i+\varepsilon_i.
$$
The statistical interpretation comes from the noise model. A standard assumption is Gaussian noise:
$$
\varepsilon_i\sim \mathcal N(0,\sigma^2).
$$

## Likelihood and Least Squares
Under the Gaussian noise assumption, the conditional distribution of $y_i$ given $x_i$ is Gaussian, and the likelihood becomes
$$
L(w)=\prod_{i=1}^N p(y_i\mid x_i,w).
$$
Ignoring constants,
$$
L(w)\propto \exp\left(
-\frac{1}{2\sigma^2}\sum_{i=1}^N (y_i-w^\top x_i)^2
\right).
$$
Thus maximum likelihood estimation is equivalent to minimizing the residual sum of squares:
$$
RSS(w)=\sum_{i=1}^N (y_i-w^\top x_i)^2.
$$
In matrix form,
$$
RSS(w)=\|Xw-y\|_2^2
=(Xw-y)^\top(Xw-y).
$$

## Normal Equation
To minimize $RSS(w)$, differentiate and set the gradient to zero:
$$
\nabla RSS(w)=2(X^\top Xw-X^\top y).
$$
Hence the stationary condition is
$$
X^\top Xw=X^\top y,
$$
which yields the normal equation solution
$$
w^*=(X^\top X)^{-1}X^\top y.
$$
This is the classical least-squares estimator. It exists in closed form when $X^\top X$ is invertible.

## Interpretation
Linear regression is therefore both:
- a geometric least-squares projection problem, and
- a probabilistic maximum likelihood estimator under Gaussian noise.

This dual interpretation is fundamental in statistical machine learning: optimization objectives often arise naturally from probabilistic assumptions.

## Stabilization and Ill-Conditioning
When $X^\top X$ is singular or poorly conditioned, a common stabilized inverse is
$$
\tilde w=(X^\top X+\varepsilon I)^{-1}X^\top y.
$$
This leads naturally to regularization.

---

# 4. Feature Maps, Model Complexity, and Regularization

## Linear Models in Transformed Spaces
Many nonlinear relationships can be modeled by applying a feature map
$$
\phi:\mathbb R^d\to\mathbb R^m
$$
and then fitting a linear model in feature space:
$$
f(x)=w^\top \phi(x).
$$
The corresponding least-squares objective is
$$
RSS(w)=\sum_{n=1}^N (w^\top \phi(x_n)-y_n)^2.
$$
If
$$
\Phi=
\begin{bmatrix}
\phi(x_1)^\top\\
\phi(x_2)^\top\\
\vdots\\
\phi(x_N)^\top
\end{bmatrix},
$$
then the closed-form solution is
$$
w^*=(\Phi^\top\Phi)^{-1}\Phi^\top y.
$$

## Polynomial Regression
A standard example is polynomial expansion:
$$
\phi(x)=[1,x,x^2,\dots,x^m]^\top.
$$
In multiple dimensions, interaction terms such as
$$
[x_1,x_2,x_1x_2]^\top
$$
capture nonlinear interactions while preserving linearity in parameters.

## Model Complexity
As model flexibility increases, training error generally decreases, but generalization may worsen.  
This gives the classic underfitting-overfitting tradeoff:
- low complexity $\to$ high bias,
- high complexity $\to$ high variance.

The bias–variance decomposition summarizes this:
$$
\mathbb E_D[(f(x)-\hat f_D(x))^2]
=
\underbrace{(f(x)-\mathbb E_D[\hat f_D(x)])^2}_{\text{Bias}^2}
+
\underbrace{\mathbb E_D[(\hat f_D(x)-\mathbb E_D[\hat f_D(x)])^2]}_{\text{Variance}}.
$$

## Regularization
To control complexity, statistical learning augments empirical loss with a penalty term:
$$
w^*=\arg\min_w \big[\mathcal L(w)+\lambda R(w)\big].
$$

### L2 Regularization / Ridge
$$
w^*=\arg\min_w \|\Phi w-y\|_2^2+\lambda \|w\|_2^2
$$
with closed form
$$
w^*=(\Phi^\top\Phi+\lambda I)^{-1}\Phi^\top y.
$$
Ridge shrinks coefficients, improves conditioning, and smooths estimates.

### L1 Regularization
$$
w^*=\arg\min_w \|\Phi w-y\|_2^2+\lambda \|w\|_1.
$$
L1 regularization promotes sparsity and can be viewed as embedded feature selection.

---

# 5. Kernel Methods

## Representer Perspective
In regularized linear models over feature space, the solution often lies in the span of transformed training points:
$$
w^*=\Phi^\top \alpha
=\sum_{i=1}^n \alpha_i \phi(x_i).
$$
This leads directly to the kernel trick.

## Kernel Function
A kernel is an inner product in feature space:
$$
K(x,x')=\phi(x)^\top\phi(x').
$$
The kernel matrix over training points is
$$
K_{ij}=K(x_i,x_j),
\qquad
K=\Phi\Phi^\top.
$$
Predictions take the form
$$
f(x)=\sum_{i=1}^n \alpha_i K(x_i,x)+b.
$$

## Dual Ridge Solution
In kernel ridge regression,
$$
\alpha=(K+\lambda I)^{-1}y.
$$
This avoids explicit construction of $\phi(x)$ and enables very high-dimensional implicit feature spaces.

## Common Kernels
Polynomial kernel:
$$
K(x,x')=(x^\top x'+c)^d.
$$

Gaussian / RBF kernel:
$$
K(x,x')=\exp\bigl(-\gamma\|x-x'\|_2^2\bigr).
$$

These kernels implement nonlinear similarity while preserving linear optimization structure in the dual.

## Valid Kernels and Mercer’s Condition
A valid kernel must be symmetric and positive semidefinite.  
Mercer’s theorem gives the canonical expansion
$$
K(x,y)=\sum_{i=1}^\infty \lambda_i \phi_i(x)\phi_i(y),\qquad \lambda_i\ge 0.
$$
Equivalently, for any coefficient vector $z$,
$$
z^\top K z\ge 0.
$$
This PSD condition is what makes kernelized learning well-defined.

---

# 6. Linear Classification, Logistic Regression, and Support Vector Machines

## Linear Classification
A linear classifier predicts by thresholding an affine score:
$$
\hat y=\operatorname{sign}(w^\top x+b).
$$
Its decision boundary is the hyperplane
$$
w^\top x+b=0.
$$
The signed margin of a point is
$$
z=y(w^\top x+b).
$$
Many loss functions depend only on this margin.

## Margin-Based Losses
Perceptron loss:
$$
\ell_P(z)=\max(0,-z).
$$

Hinge loss:
$$
\ell_H(z)=\max(0,1-z).
$$

Logistic loss:
$$
\ell_L(z)=\log(1+e^{-z}).
$$

These define different linear classification methods through different geometries of penalization.

## Logistic Regression
Logistic regression gives a probabilistic model for binary classification:
$$
\Pr(y=+1\mid x;w)=\sigma(w^\top x),
$$
$$
\Pr(y=-1\mid x;w)=\sigma(-w^\top x),
$$
where
$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$
This can be written compactly as
$$
\Pr(y\mid x;w)=\sigma(yw^\top x).
$$
Maximum likelihood gives the optimization problem
$$
w^*=\arg\min_w \sum_{i=1}^N \log(1+e^{-y_i w^\top x_i}).
$$
So logistic regression is both a smooth margin method and a likelihood-based probabilistic classifier.

## Geometric View of Hyperplanes
For a point $x_0$, distance to the hyperplane is
$$
d=\frac{|w^\top x_0+b|}{\|w\|_2}.
$$
With label-aware sign,
$$
d=\frac{y(w^\top x+b)}{\|w\|_2}.
$$
This geometric quantity becomes central in SVM.

## Hard-Margin SVM
Support vector machines choose the separating hyperplane with maximum margin:
$$
\min_{w,b}\frac12\|w\|_2^2
\quad\text{s.t.}\quad
y_i(w^\top x_i+b)\ge 1.
$$
This formulation is equivalent to maximizing the minimum distance of training points to the boundary.

## Soft-Margin SVM
When classes are not perfectly separable, slack variables are introduced:
$$
\min_{w,b,\xi}\frac12\|w\|_2^2+C\sum_{i=1}^N \xi_i
$$
subject to
$$
y_i(w^\top x_i+b)\ge 1-\xi_i,\qquad \xi_i\ge 0.
$$
This is equivalent to regularized hinge-loss minimization:
$$
\min_{w,b}\lambda\|w\|_2^2+\sum_{i=1}^N \max(0,1-y_i(w^\top x_i+b)).
$$

## Dual Form and Kernel SVM
The SVM Lagrangian leads to the representation
$$
w=\sum_{i=1}^N \alpha_i y_i\phi(x_i),
$$
with constraint
$$
\sum_{i=1}^N \alpha_i y_i=0.
$$
The dual objective becomes
$$
\max_\alpha \sum_{i=1}^N \alpha_i
-\frac12\sum_{i=1}^N\sum_{j=1}^N \alpha_i\alpha_j y_i y_j K(x_i,x_j),
$$
subject to
$$
0\le \alpha_i\le C,\qquad \sum_{i=1}^N \alpha_i y_i=0.
$$
Prediction is then
$$
f(x)=\operatorname{sign}\left(\sum_{i=1}^N \alpha_i^* y_i K(x_i,x)+b^*\right).
$$
Thus kernels make nonlinear large-margin classification possible without explicit feature expansion.

---

# 7. Optimization Duality and KKT Structure

## Constrained Optimization
Many learning problems can be written as
$$
\min_w F(w)
\qquad \text{s.t.} \qquad h_j(w)\le 0.
$$
The Lagrangian is
$$
L(w,\lambda)=F(w)+\sum_j \lambda_j h_j(w),\qquad \lambda_j\ge 0.
$$

## Dual Function and Dual Problem
Define
$$
g(\lambda)=\min_w L(w,\lambda).
$$
Then the dual problem is
$$
\max_{\lambda\ge 0} g(\lambda).
$$
Weak duality always holds:
$$
\max_{\lambda\ge 0}g(\lambda)\le \min_w H(w),
$$
while under convexity and suitable regularity, strong duality gives
$$
\min_w\max_{\lambda\ge 0}L(w,\lambda)
=
\max_{\lambda\ge 0}\min_w L(w,\lambda).
$$

## KKT Conditions
Optimality in constrained convex optimization is characterized by the KKT conditions.

Stationarity:
$$
\nabla F(w^*)+\sum_j \lambda_j^* \nabla h_j(w^*)=0.
$$

Primal feasibility:
$$
h_j(w^*)\le 0.
$$

Dual feasibility:
$$
\lambda_j^*\ge 0.
$$

Complementary slackness:
$$
\lambda_j^* h_j(w^*)=0.
$$

These conditions are fundamental for understanding SVM duality and support-vector behavior.

---

# 8. Neural Networks and Multiclass Learning

## Neural Networks
A neural network repeatedly alternates linear transformations and nonlinear activations:
$$
o=h(Wx).
$$
For a multilayer perceptron,
$$
f(x;W_1,\dots,W_L)=h_L\bigl(W_L h_{L-1}(\cdots h_1(W_1x)\cdots)\bigr).
$$
This extends linear models into highly expressive nonlinear function classes.

## Hidden Representation and Backpropagation
For a one-hidden-layer network,
$$
a_n=Wx_n,\qquad h_n=\sigma(a_n),\qquad f_n=u^\top h_n.
$$
With binary logistic loss,
$$
\ell_n=\log(1+\exp(-y_n f_n)).
$$
The output derivative is
$$
\frac{\partial \ell_n}{\partial f_n}
=
-\frac{y_n}{1+\exp(y_n f_n)}.
$$
Then
$$
\frac{\partial \ell_n}{\partial u}
=
\left(\frac{\partial \ell_n}{\partial f_n}\right)h_n,
$$
$$
\frac{\partial \ell_n}{\partial h_n}
=
\left(\frac{\partial \ell_n}{\partial f_n}\right)u.
$$
With sigmoid activation,
$$
\sigma'(a)=\sigma(a)(1-\sigma(a)),
$$
so
$$
\frac{\partial h_n}{\partial a_n}
=
\operatorname{diag}(h_n\circ(1-h_n)),
$$
and finally
$$
\frac{\partial \ell_n}{\partial W}
=
\left(\frac{\partial \ell_n}{\partial a_n}\right)x_n^\top.
$$
This recursive chain-rule structure is the essence of backpropagation.

## Activation Functions
Sigmoid:
$$
\sigma(x)=\frac{1}{1+e^{-x}}.
$$

ReLU:
$$
\mathrm{ReLU}(x)=\max(0,x).
$$

Tanh:
$$
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}.
$$

Nonlinearity is what allows neural networks to overcome the limitations of pure linear models.

## Multiclass Linear Classification
For multiclass prediction, the linear rule becomes
$$
f(x)=\arg\max_{i\in[C]} w_i^\top x.
$$
The probabilistic generalization is softmax:
$$
P(y=i\mid x)=\frac{e^{w_i^\top x}}{\sum_{j=1}^C e^{w_j^\top x}},
$$
with loss
$$
\ell_{\mathrm{softmax}}(x,y)
=
-\log \frac{e^{w_y^\top x}}{\sum_{i=1}^C e^{w_i^\top x}}.
$$
This is the standard multiclass extension of logistic regression.

---

# 9. Ensemble Methods, Clustering, and Latent Variable Models

## Ensemble Learning
A central idea in statistical machine learning is that many weak predictors, if combined properly, can produce a strong predictor.  
The simplest aggregation is majority vote:
$$
H(x)=\operatorname{sign}\left(\sum_{\ell=1}^L h_\ell(x)\right).
$$

## Bagging
Bagging creates bootstrap datasets
$$
D_1,D_2,\dots,D_L
$$
by sampling with replacement from the original dataset. A base learner is trained on each bootstrap sample, and predictions are aggregated. The statistical goal is variance reduction.

## AdaBoost
AdaBoost sequentially reweights samples toward hard cases.  
Initialize
$$
D_1(i)=\frac{1}{N}.
$$
At round $t$, compute weighted error
$$
\varepsilon_t=\sum_{i=1}^N D_t(i)\mathbf{1}(h_t(x_i)\ne y_i),
$$
then assign weak learner weight
$$
\beta_t=\frac12\ln\left(\frac{1-\varepsilon_t}{\varepsilon_t}\right).
$$
Update the sample distribution by
$$
D_{t+1}(i)=\frac{D_t(i)\exp(-\beta_t y_i h_t(x_i))}{Z_t}.
$$
The final classifier is
$$
H(x)=\operatorname{sign}\left(\sum_{t=1}^T \beta_t h_t(x)\right).
$$
AdaBoost can be interpreted as stagewise minimization of exponential loss.

## K-Means Clustering
In unsupervised learning, labels are absent and structure must be inferred from the data itself.  
K-means minimizes within-cluster squared distance:
$$
F(\{\gamma_{n,k}\},\{\mu_k\})
=
\sum_{n=1}^N\sum_{k=1}^K \gamma_{n,k}\|x_n-\mu_k\|_2^2,
$$
where
$$
\gamma_{n,k}\in\{0,1\},\qquad \sum_{k=1}^K \gamma_{n,k}=1.
$$
Centroids are updated by
$$
\mu_k=
\frac{\sum_{n=1}^N \gamma_{n,k}x_n}{\sum_{n=1}^N \gamma_{n,k}}.
$$
K-means alternates between assignment and centroid update.

## Gaussian Mixture Models
A probabilistic generalization of clustering is the Gaussian mixture model:
$$
p(x)=\sum_{k=1}^K w_k\,\mathcal N(x\mid \mu_k,\Sigma_k),
$$
with
$$
w_k\ge 0,\qquad \sum_{k=1}^K w_k=1.
$$
Each point is associated with a latent cluster indicator $z_n$, and the soft assignment / responsibility is
$$
\gamma_{n,k}=P(z_n=k\mid x_n).
$$

## Expectation-Maximization
The GMM likelihood is
$$
\sum_{n=1}^N \log\left(\sum_{k=1}^K w_k\mathcal N(x_n\mid \mu_k,\Sigma_k)\right),
$$
which is hard to optimize directly because of the log of a sum.  
EM solves this by alternating:

E-step:
$$
\gamma_{n,k}\propto w_k \mathcal N(x_n\mid \mu_k,\Sigma_k).
$$

M-step:
$$
w_k=\frac{\sum_n \gamma_{n,k}}{N},
$$
$$
\mu_k=\frac{\sum_n \gamma_{n,k}x_n}{\sum_n \gamma_{n,k}},
$$
$$
\Sigma_k=
\frac{\sum_n \gamma_{n,k}(x_n-\mu_k)(x_n-\mu_k)^\top}{\sum_n \gamma_{n,k}}.
$$