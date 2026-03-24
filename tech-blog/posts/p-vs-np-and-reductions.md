---
title: P vs NP, Reductions, and Beyond
category: COMPLEXITY THEORY
date: 2026.03.24
readtime: 22 min
---

# P vs NP

## Definition of the Problem

A computational problem can be defined as a relation between an input set and a solution set.

- For any input `I` and candidate solution `S`, if `S` is correct, then `(I, S)` belongs to the relation.
- Since the object is a relation, one input may have multiple valid solutions.

When the goal is to decide whether an input is true or false, we call it a **decision problem**.

## Deterministic Algorithms

At each choice point, a deterministic algorithm follows exactly one next step and proceeds.

## Nondeterminism

- Nondeterminism is an abstract model that can explore multiple choices simultaneously.
- Intuitively: "What if all choices were explored at once?"
- Its runtime is defined by the worst branch among those parallel branches.

## P Problems

- `P` is the class of decision problems solvable in polynomial time by deterministic algorithms.
- These are "efficiently solvable" under the standard model.

## NP Problems

- `NP` is the class of decision problems solvable in polynomial time by nondeterministic algorithms.
- Equivalent verifier view: for YES instances, there exists a short proof that can be checked in polynomial time.

Two key notions:

- **Certificate (witness)**: a proof object for a YES instance.
- **Certifier (verifier)**: a polynomial-time algorithm that checks a certificate.

## Visualizing P and NP

The known relation is:

$$
P \subseteq NP.
$$

Whether equality holds remains open:

$$
P \stackrel{?}{=} NP.
$$

## Example 1: SAT (Certifier + Certificate)

Certificate: a truth assignment to variables.  
Certifier: checks all clauses in polynomial time.

Example formula:

$$
(x_1 \lor x_2 \lor x_3) \land (x_1 \lor \neg x_2 \lor x_3)
\land (x_1 \lor x_2 \lor x_4) \land (\neg x_1 \lor x_3 \lor \neg x_4).
$$

## Example 2: Hamiltonian Cycle (Certifier + Certificate)

Problem: given a graph `G=(V,E)`, determine whether a simple cycle visits each vertex exactly once.

- Certificate: a permutation of vertices.
- Certifier checks:
  1. each vertex appears exactly once;
  2. each consecutive pair is an edge in `E`;
  3. last returns to first.

---

## Verifier-Based NP Interpretation

Nondeterminism can be replaced by a deterministic verifier plus certificate:

- If input is YES, there exists a certificate accepted by the verifier.
- If input is NO, no certificate should be accepted.

### Core Properties

1. Certificate length must be polynomially bounded in input size.
2. Verification must run in polynomial time in `|I|` and `|h|`.

Therefore:

$$
P \subseteq NP.
$$

---

## Polynomial-Time Reducibility

If solving `X` can be transformed into solving `Y` with polynomial overhead, then:

$$
X \le_p Y.
$$

Formally, there exists polynomial-time computable `f` such that:

$$
x \in X \iff f(x) \in Y.
$$

This means `X` is no harder than `Y`.

### Intuition

Reduction compares difficulty, not implementation details of `Y`.
You only need correctness of the mapping and polynomial-time transformability.

---

## Hall's Theorem and Bipartite Matching

For bipartite graph `G=(U,V,E)`, a matching that covers all vertices in `U` exists iff:

$$
|N(S)| \ge |S| \quad \text{for all } S \subseteq U.
$$

This gives a complete criterion for perfect matching on the `U` side.

### Bipartite Matching (DFS Augmentation)

```cpp
bool dfs(int a) {
    if (visited[a]) return false;
    visited[a] = true;
    for (int b : adj[a]) {
        if (bMatch[b] == -1 || dfs(bMatch[b])) {
            aMatch[a] = b;
            bMatch[b] = a;
            return true;
        }
    }
    return false;
}

int bipartiteMatch() {
    aMatch.assign(n + 1, -1);
    bMatch.assign(n + 1, -1);
    int size = 0;
    for (int a = 1; a <= n; ++a) {
        visited.assign(n + 1, false);
        if (dfs(a)) ++size;
    }
    return size;
}
```

---

## Kőnig's Theorem via Max-Flow Min-Cut

In bipartite graphs:

$$
\text{Maximum Matching Size}=\text{Minimum Vertex Cover Size}.
$$

Reduction idea:

1. Add source to all `U` vertices (capacity 1).
2. Keep `U -> V` edges (capacity 1).
3. Add all `V` vertices to sink (capacity 1).

Then:

- max flow value = maximum matching size
- min cut value = minimum vertex cover size

By MFMC theorem, they are equal.

---

## NP-Hard and NP-Complete

### NP-Hard

Problem `Y` is NP-Hard if:

$$
\forall X \in NP,\; X \le_p Y.
$$

### NP-Complete

`Y` is NP-Complete if:

1. `Y in NP`
2. `Y` is NP-Hard.

### Key Consequence

If any NP-Hard problem is solved in polynomial time, then:

$$
P=NP.
$$

---

## 2-SAT

2-SAT is SAT where each clause has at most two literals.

Given clause `(a or b)`, implication graph edges are:

- `(not a) -> b`
- `(not b) -> a`

Satisfiability criterion:

- Formula is satisfiable iff no variable `x_i` and `not x_i` belong to the same SCC.

So 2-SAT is solvable in polynomial time via SCC decomposition.

---

## Vertex Cover, Independent Set, and Set Cover

### Vertex Cover <-> Independent Set

For graph `G=(V,E)`:

- `S` is an independent set iff `V-S` is a vertex cover.

Hence the two are polynomial-time equivalent.

### Vertex Cover -> Set Cover

Construct:

- Universe `U = E`
- For each vertex `v`, subset `S_v = {e in E : e incident to v}`

Then vertex cover of size `k` exists iff set cover of size `k` exists.

---

## Travelling Salesman Problem (TSP)

TSP asks for a minimum-cost tour visiting all cities exactly once and returning to start.
It is NP-hard.

### Dynamic Programming + Bitmasking

```cpp
const int INF = 0x3f3f3f3f;
int cache[16][1 << 16], n, cost[16][16], start = 0;

int tsp(int cur, int state) {
    if (state == (1 << n) - 1) return cost[cur][start] ? cost[cur][start] : INF;
    int &ret = cache[cur][state];
    if (ret != -1) return ret;
    ret = INF;
    for (int nxt = 0; nxt < n; ++nxt) {
        if (((state >> nxt) & 1) == 0 && cost[cur][nxt] != 0) {
            ret = min(ret, cost[cur][nxt] + tsp(nxt, state | (1 << nxt)));
        }
    }
    return ret;
}
```

### Reduction: HAM-CYCLE -> TSP

Use edge weights:

$$
d(u,v)=
\begin{cases}
1,&(u,v)\in E\\
2,&\text{otherwise}
\end{cases}
$$

Set threshold `D=n`.

- Hamiltonian cycle exists iff TSP tour of cost `<= n` exists.

---

## 3-SAT -> Independent Set

For each clause, create 3 literal vertices (triangle inside clause).  
Add edges between contradictory literals across clauses.

Then:

- satisfiable 3-SAT iff graph has independent set of size `m` (number of clauses).

Therefore:

$$
3\text{-SAT} \le_p \text{Independent Set}.
$$

---

## Circuit-SAT and 3-SAT

### Circuit-SAT is NP-Complete

1. Membership: assignment is polynomial-time verifiable by circuit evaluation.
2. Hardness: every NP verifier can be encoded as a polynomial-size circuit.

### 3-SAT is NP-Complete

3-SAT is in NP, and NP-hardness follows from polynomial reductions from known NP-complete problems.

---

## Subset Sum -> Partition

Claim:

$$
\text{Subset Sum} \le_p \text{Partition}.
$$

Given `w_1,...,w_n` and target `W`, construct:

$$
v_{n+1}=2\sum_{i=1}^n w_i-W,\qquad
v_{n+2}=\sum_{i=1}^n w_i+W.
$$

Then the partition instance has a balanced split iff original subset sum instance has a subset summing to `W`.

---

# PSPACE

PSPACE is the class of decision problems solvable in polynomial space by a deterministic Turing machine.

- Time may be exponential.
- Space is polynomially bounded.

Known containments:

$$
P \subseteq NP \subseteq PSPACE.
$$

## QSAT

Quantified SAT (alternating `exists` / `forall`) is PSPACE-complete.
It generalizes SAT by introducing quantifier alternation over variables.

---

## Probability Inequalities (Quick Notes)

### Markov's Inequality

For nonnegative random variable `X` and `a>0`:

$$
\Pr(X\ge a)\le \frac{\mathbb E[X]}{a}.
$$

### Chebyshev's Inequality

For mean `mu`, variance `sigma^2`, and `k>0`:

$$
\Pr(|X-\mu|\ge k\sigma)\le \frac{1}{k^2}.
$$

### Chernoff Bounds (Independent Bernoulli-style sums)

For sum `S` with expectation `mu`:

$$
\Pr(S\ge (1+\epsilon)\mu)\le
\exp\!\left(-\frac{\epsilon^2\mu}{2+\epsilon}\right),
$$
$$
\Pr(S\le (1-\epsilon)\mu)\le
\exp\!\left(-\frac{\epsilon^2\mu}{2}\right).
$$

---

## Final Takeaway

The central language of complexity theory is reduction:

- show `X <=p Y` to compare hardness,
- use NP-hardness / completeness to classify intractability,
- use structural classes (`P`, `NP`, `PSPACE`) to understand what resources are fundamentally required.

The open question

$$
P\stackrel{?}{=}NP
$$

remains the pivotal boundary between efficiently solvable and efficiently verifiable computation.
