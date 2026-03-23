---
title: Randomized Algorithms
category: CS
semester: 2025 S
---
# Randomized Algorithms

Probabilistic algorithms and analysis.

## Course Mindset
Randomization is not “guessing”; it is a controlled computational resource. The recurring workflow in this course is:

- design a randomized object (pivot, sampled edge, hash, random bits)
- prove a clean guarantee (in expectation or with high probability)
- show how repetition/amplification turns weak guarantees into strong ones

The goal of this note is to record the core algorithms/ideas and the analysis patterns you reuse across topics.

---

## 1. Randomness: Las Vegas vs Monte Carlo

### Las Vegas
The output is always correct, but runtime is random.

### Monte Carlo
Runtime is bounded, but the output may be wrong with small probability.

Most algorithmic questions reduce to: how small can the failure probability be, and how do we amplify it?

---

## 2. Probabilistic Tools (the analysis toolkit)

### Linearity of Expectation
For any random variables $X_1,\dots,X_n$,
$$
\mathbb{E}\left[\sum_i X_i\right] = \sum_i \mathbb{E}[X_i].
$$
This is the fastest route to expected-cost analyses (sorting, hashing collisions, random cuts surviving).

### Markov and Chebyshev (quick tail control)
For nonnegative $X$ and $a>0$,
$$
\Pr[X \ge a] \le \frac{\mathbb{E}[X]}{a}.
$$
If $\mathrm{Var}(X)=\sigma^2$,
$$
\Pr[|X-\mathbb{E}X| \ge t] \le \frac{\sigma^2}{t^2}.
$$

### Chernoff Bounds (Bernoulli sums; concentration)
Let $X=\sum_{i=1}^n X_i$ where $X_i\in\{0,1\}$ are independent and $\mu=\mathbb{E}[X]$.
For $\varepsilon\in(0,1]$,
$$
\Pr[X \ge (1+\varepsilon)\mu] \le
\exp\left(-\frac{\varepsilon^2\mu}{3}\right),
$$
$$
\Pr[X \le (1-\varepsilon)\mu] \le
\exp\left(-\frac{\varepsilon^2\mu}{2}\right).
$$

### Amplification (failure probability to high probability)
If one run fails with probability at most $p$, then $k$ independent runs fail together with probability at most
$$
p^k.
$$
Equivalently, to get failure $\le \delta$, choose $k=O(\log(1/\delta)/\log(1/p))$.

---

## 3. Randomized Quicksort
Randomized Quicksort removes worst-case dependence on pivot quality by choosing the pivot uniformly at random.

### Guarantee
The expected running time is
$$
\mathbb{E}[T(n)] = O(n\log n).
$$
More precisely, expected comparisons satisfy
$$
\mathbb{E}[C_n] = 2(n+1)H_n - 4n = O(n\log n),
$$
where $H_n$ is the $n$-th harmonic number.

### Pseudocode
```text
RANDOMIZED-QUICKSORT(A, l, r):
    if l >= r:
        return
    p = UniformRandom(l, r)
    swap A[p], A[r]
    q = PARTITION(A, l, r)
    RANDOMIZED-QUICKSORT(A, l, q - 1)
    RANDOMIZED-QUICKSORT(A, q + 1, r)
```

### C++ Sketch
```cpp
int partition(vector<int>& a, int l, int r) {
    int pivot = a[r];
    int i = l - 1;
    for (int j = l; j < r; ++j) {
        if (a[j] <= pivot) {
            ++i;
            swap(a[i], a[j]);
        }
    }
    swap(a[i + 1], a[r]);
    return i + 1;
}

void randomizedQuickSort(vector<int>& a, int l, int r) {
    if (l >= r) return;
    int p = l + rand() % (r - l + 1);
    swap(a[p], a[r]);
    int q = partition(a, l, r);
    randomizedQuickSort(a, l, q - 1);
    randomizedQuickSort(a, q + 1, r);
}
```

---

## 4. Random Graphs and Cuts

## 4.1 Randomized Minimum Cut (Karger)
Karger's algorithm repeatedly contracts a uniformly random edge until only two supernodes remain.

### Core Idea
If the algorithm never contracts an edge inside a fixed minimum cut, then that cut survives to the end.
Since a minimum cut is small relative to the graph, the probability of avoiding it across contractions is nontrivial.

### Guarantee (one run)
For a graph with $n$ vertices, the probability that one run finds a minimum cut is at least
$$
\frac{2}{n(n-1)}.
$$

### Repetition
Repeating the algorithm $O(n^2\log n)$ times boosts success probability to high probability.

### Pseudocode
```text
KARGER-MINCUT(G):
    while |V(G)| > 2:
        choose a random edge e uniformly
        contract e
        delete self-loops
    return the cut between the two remaining supernodes
```

---

## 4.2 Randomized Rounding (LP to integral)
Randomized rounding converts a fractional solution $x_i\in[0,1]$ into an integral solution by independent randomized choices.

### Construction
Given $x_i\in[0,1]$, define a random variable $X_i$:
$$
X_i =
\begin{cases}
1 & \text{with probability } x_i \\
0 & \text{with probability } 1-x_i
\end{cases}
$$
so that
$$
\mathbb{E}[X_i]=x_i.
$$

### Guarantee Pattern
Once expectations match the LP, concentration bounds (often Chernoff) control constraint violations.

### Pseudocode (generic independent rounding)
```text
RANDOMIZED-ROUNDING(x[1..n]):
    for i in 1..n:
        X[i] = 1 with probability x[i]
        X[i] = 0 with probability 1 - x[i]
    return X
```

In concrete problems, you then (i) scale the fractional solution to satisfy constraints in expectation, and (ii) apply concentration to show the randomized solution violates constraints with small probability.

---

## 5. Hashing and Frequency (Randomized data structures)

## 5.1 Universal Hashing
A family $\mathcal{H}$ is universal if for distinct keys $x\neq y$,
$$
\Pr_{h\sim\mathcal{H}}[h(x)=h(y)] \le \frac{1}{m},
$$
where $m$ is the number of buckets.

This property is what makes collision analyses go through cleanly.

---

## 5.2 Count-Min Sketch (frequency estimation)
Count-Min Sketch stores $d$ hash tables of width $w$.

### Setup
Pick $d$ pairwise-independent (or 2-universal) hash functions $h_1,\dots,h_d$.
Maintain counts $C[k, b]$ where $b=h_k(i)$.

### Query (estimate)
The estimate for item $i$ is
$$
\hat{f}_i = \min_{k=1}^d C[k, h_k(i)].
$$

### Pseudocode (sketch maintenance)
```text
COUNT-MIN-UPDATE(i, +1):
    for k = 1..d:
        b = h_k(i)
        C[k, b] += 1

COUNT-MIN-QUERY(i):
    return min_{k=1..d} C[k, h_k(i)]
```

### Guarantee (representative form)
With appropriate parameters $w=\lceil e/\varepsilon\rceil$, $d=\lceil \ln(1/\delta)\rceil$,
you get bounds of the form:
$$
\Pr\left[\hat{f}_i \le f_i + \varepsilon\|f\|_1\right] \ge 1-\delta.
$$
(The key shape is “no large underestimation; additive error controlled by $\varepsilon$ and total mass.”)

---

## 5.3 Heavy Hitters (why randomized summaries work)
If an item is “heavy,” it stands out even after hashing noise.

Define heavy hitter threshold: an item with frequency
$$
f_i \ge \phi\,\|f\|_1
$$
for some $\phi$.

The randomized viewpoint is: you do not need exact counts for everything, only enough compressed evidence to recover unusually frequent items.

---

## 6. Sampling and Sketching

## 6.1 Reservoir Sampling (uniform sample without knowing length)
Given a stream and unknown length, reservoir sampling returns a uniformly random element among all seen items.

### Algorithm
```text
RESERVOIR-SAMPLING(x1, x2, ...):
    sample = x1
    for t = 2 to n:
        pick j uniformly from {1,...,t}
        if j == t:
            sample = xt
    return sample
```

### Guarantee
For each position $i$ in the stream,
$$
\Pr[\text{sample}=x_i] = \frac{1}{n}.
$$

---

## 6.2 Johnson-Lindenstrauss (JL) Lemma: random projections
Random projections approximately preserve pairwise distances in a lower dimension.

### Guarantee
For a suitable random map $f$,
$$
(1-\varepsilon)\|x-y\|_2^2 \le \|f(x)-f(y)\|_2^2 \le (1+\varepsilon)\|x-y\|_2^2
$$
for all points in a finite set, as long as the target dimension satisfies
$$
k = O\left(\frac{\log n}{\varepsilon^2}\right).
$$

---

## 7. Randomized Verification (when you can test correctness fast)
This is where randomness becomes a “cheap consistency checker.”

## 7.1 Schwartz-Zippel Lemma (polynomial identity test)
If a polynomial is nonzero, it cannot have too many roots.

### Lemma
Let $p(x_1,\dots,x_n)\neq 0$ be of total degree $d$ over a field $\mathbb{F}$.
Pick $a_1,\dots,a_n$ uniformly from $\mathbb{F}$.
Then
$$
\Pr[p(a_1,\dots,a_n)=0] \le \frac{d}{|\mathbb{F}|}.
$$

### Pseudocode
```text
RANDOMIZED-POLY-IDENTITY-TEST(p, d, F):
    sample a = (a1,...,an) uniformly from F^n
    if p(a) == 0:
        return "possibly zero"
    else:
        return "definitely nonzero"
```

---

## 7.2 Freivalds' Algorithm (verify matrix multiplication)
Given $A,B,C$ you want to verify $C=AB$ quickly.

### Core Idea
Pick a random vector $r$ and compare:
$$
ABr \stackrel{?}{=} Cr.
$$
If $AB\neq C$, then the random choice makes the mismatch detectable with high probability.

### Pseudocode
```text
FREIVALDS(A, B, C, iterations T):
    for t in 1..T:
        pick random vector r (uniform over {0,1}^k or field)
        if A*(B*r) != C*r:
            return "not equal"
    return "equal with probability 1-2^{-T} (representative)"
```

---

## 8. Random Walks and Mixing Analysis

### Spectral Gap
For an ergodic Markov chain with transition matrix $P$,
the second-largest eigenvalue (in magnitude) controls convergence speed.

Define the spectral gap
$$
\gamma = 1 - \lambda_2.
$$
Larger gap typically implies faster mixing.

### Mixing Time (total variation)
Mixing time is the minimal $t$ such that the distribution is close to stationarity:
$$
t_{\mathrm{mix}}(\varepsilon)
=
\min\left\{
t :
|P^t(x,\cdot) - \pi|_{\mathrm{TV}} \le \varepsilon
\right\}.
$$

### Conductance (bottleneck view)
Conductance measures how easily the chain escapes a subset $S$:
$$
\Phi(S) = \frac{Q(S,S^c)}{\pi(S)}.
$$

Low conductance corresponds to bottlenecks and slow mixing.

---

## 9. PageRank as a Random Process
PageRank models a random surfer who follows links most of the time, but teleports occasionally.

### Stationary Equation
$$
\pi = \alpha P^\top \pi + (1-\alpha)v,
$$
where $\alpha$ is the damping factor and $v$ is the teleportation distribution.

Teleportation ensures ergodicity and stabilizes the ranking.

---

## 10. Derandomization (brief record)
A central theme is whether randomized algorithms can be replaced with deterministic ones using less or structured randomness.

Random walks and limited independence are typical tools for building pseudorandom objects that mimic true randomness sufficiently for algorithmic guarantees.

---

## 11. Probabilistically Checkable Proofs (PCP) (high-level)
PCP theory connects randomness to hardness of approximation.

Informally:
every NP proof can be reformulated so a verifier reads only $O(1)$ bits of the proof, using $O(\log n)$ random bits.

### Informal Statement
$$
\mathrm{NP} = \mathrm{PCP}(O(\log n), O(1)).
$$

---

## Course Summary
The most important takeaway is not a single theorem, but the analysis habit:

- use randomness to avoid brittle worst cases
- compress large structures into probabilistic summaries
- replace exact optimization with approximate guarantees
- analyze through expectation, concentration, and geometry
- then ask whether randomness was truly necessary (derandomization / PCP)