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