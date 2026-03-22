---
title: Abstract Algebra
category: MATH
semester: 2023 F
---

# 1. Groups as Algebraic Structure

## Groups, Examples, and Basic Consequences

A group is a set $G$ with a binary operation $\cdot$ such that:

$$
\forall a,b \in G,\quad ab \in G
$$

$$
(ab)c = a(bc)
$$

$$
\exists e \in G \text{ such that } ea = ae = a
$$

$$
\forall a \in G,\ \exists a^{-1} \in G \text{ such that } aa^{-1} = a^{-1}a = e
$$

- A group is the basic algebraic model of symmetry and reversible transformation.
- Typical examples:
  - $(\mathbb{Z}, +)$
  - $(\mathbb{R}^\times, \cdot)$
  - permutation groups $S_n$
  - matrix groups such as $GL_n(\mathbb{R})$

Basic consequences:
$$
ae = ea = a,\qquad (a^{-1})^{-1} = a,\qquad (ab)^{-1} = b^{-1}a^{-1}
$$

## Commutativity and Symmetry

A group is abelian if
$$
ab = ba \qquad \forall a,b \in G
$$

- Abelian groups model commutative structure.
- Nonabelian groups capture richer symmetry, where the order of operations matters.

## Subgroups and Generators

A subgroup $H \le G$ is a subset that is itself a group under the same operation.

Subgroup test:
$$
a,b \in H \implies ab^{-1} \in H
$$

A subset $S \subseteq G$ generates $G$ if
$$
G = \langle S \rangle
$$

- Generators compress global structure into a small set of basic operations.
- Much of group theory asks how complicated a group is from the viewpoint of its generators and relations.

---

# 2. Group Actions and Symmetry

## Group Actions

A group action of $G$ on a set $X$ is a map
$$
G \times X \to X,\qquad (g,x)\mapsto g\cdot x
$$
such that
$$
e\cdot x = x,\qquad (gh)\cdot x = g\cdot(h\cdot x)
$$

- A group action turns an abstract group into an actual symmetry acting on objects.

## Orbits and Stabilizers

Orbit of $x$:
$$
\mathrm{Orb}(x) = \{g\cdot x : g \in G\}
$$

Stabilizer of $x$:
$$
\mathrm{Stab}(x) = \{g \in G : g\cdot x = x\}
$$

Orbit--stabilizer theorem:
$$
|G| = |\mathrm{Orb}(x)|\, |\mathrm{Stab}(x)|
$$

- Orbits measure how far an element can move under symmetry.
- Stabilizers measure what symmetry remains fixed at that element.

## Plane and Permutation Groups

- Plane symmetry groups describe rotations, reflections, and translations of geometric figures.
- Permutation groups are subgroups of $S_n$, acting by rearranging finite sets.

For $\sigma \in S_n$,
$$
\sigma : \{1,\dots,n\} \to \{1,\dots,n\}
$$

- These two viewpoints connect algebra directly to geometry and combinatorics.

---

# 3. Decomposition and Classification of Groups

## Cosets and Lagrange's Theorem

For a subgroup $H \le G$, a left coset is
$$
gH = \{gh : h \in H\}
$$

Lagrange's theorem:
$$
|G| = [G:H]\cdot |H|
$$

- The order of a subgroup divides the order of the group.
- Cosets partition the group into equally sized pieces.

## Normal Subgroups and Quotient Groups

A subgroup $N \trianglelefteq G$ is normal if
$$
gNg^{-1} = N \qquad \forall g \in G
$$

Then the quotient group
$$
G/N
$$
is well-defined.

- Normal subgroups are exactly the subgroups that support quotient structure.
- Quotients represent symmetry after identifying elements modulo a normal subgroup.

## Isomorphism Theorems

A group homomorphism
$$
\varphi : G \to H
$$
satisfies
$$
\varphi(ab)=\varphi(a)\varphi(b)
$$

First isomorphism theorem:
$$
G/\ker(\varphi) \cong \mathrm{im}(\varphi)
$$

- The isomorphism theorems formalize the relationship between homomorphisms, kernels, images, and quotient structure.

## Finite Abelian Group Classification

Every finite abelian group is isomorphic to a direct product of cyclic groups:
$$
G \cong \mathbb{Z}_{n_1}\times \cdots \times \mathbb{Z}_{n_r}
$$

more precisely in invariant factor or elementary divisor form.

- This theorem gives a complete structural description of finite abelian groups.
- It is one of the first major classification results in algebra.

---

# 4. Algebraic Applications

## RSA and Cryptographic Groups

RSA is built on arithmetic in modular rings and multiplicative groups.

For modulus $n=pq$,
$$
\mathbb{Z}_n^\times = \{a \in \mathbb{Z}_n : \gcd(a,n)=1\}
$$

Encryption and decryption use exponents:
$$
c \equiv m^e \pmod n,\qquad m \equiv c^d \pmod n
$$

with
$$
ed \equiv 1 \pmod{\varphi(n)}
$$

- The security comes from the difficulty of factoring large integers.
- The correctness comes from group-theoretic and number-theoretic structure.

## Euler $\varphi$ Function

$$
\varphi(n)=|\mathbb{Z}_n^\times|
$$

If $n=pq$ with $p,q$ distinct primes:
$$
\varphi(n)=(p-1)(q-1)
$$

Euler's theorem:
$$
a^{\varphi(n)} \equiv 1 \pmod n \qquad (\gcd(a,n)=1)
$$

- The $\varphi$ function measures the size of the multiplicative group modulo $n$.

## Sandpile Groups

- Sandpile groups arise from chip-firing dynamics on graphs.
- They form finite abelian groups associated to the graph Laplacian.

A typical construction is
$$
K(G) \cong \mathbb{Z}^{n-1}/\mathrm{im}(\widetilde{L})
$$
where $\widetilde{L}$ is a reduced Laplacian matrix.

- This is a striking example of algebra emerging from a combinatorial process.

---

# 5. Rings and Ideals

## Rings and Homomorphisms

A ring $R$ has two operations, usually written $+$ and $\cdot$, such that:
- $(R,+)$ is an abelian group
- multiplication is associative
- multiplication distributes over addition

Ring homomorphism:
$$
\varphi : R \to S
$$
satisfies
$$
\varphi(a+b)=\varphi(a)+\varphi(b),\qquad
\varphi(ab)=\varphi(a)\varphi(b)
$$

- Rings generalize arithmetic beyond groups by including both addition and multiplication.

## Ideals and Quotient Rings

An ideal $I \triangleleft R$ satisfies
$$
a,b \in I \implies a-b \in I
$$
and
$$
r \in R,\ a \in I \implies ra, ar \in I
$$

The quotient ring is
$$
R/I
$$

- Ideals play the ring-theoretic role that normal subgroups play in group theory.
- Quotient rings encode arithmetic modulo an ideal.

## Prime and Maximal Ideals

A prime ideal $P$ satisfies:
$$
ab \in P \implies a \in P \text{ or } b \in P
$$

A maximal ideal $M$ is a proper ideal such that there is no proper ideal strictly between $M$ and $R$.

Key characterizations:
$$
P \text{ prime } \iff R/P \text{ is an integral domain}
$$

$$
M \text{ maximal } \iff R/M \text{ is a field}
$$

- These conditions connect internal ideal structure to the algebra of quotients.

---

# 6. Factorization and Polynomial Rings

## Integral Domains and UFDs

An integral domain is a commutative ring with identity and no zero divisors:
$$
ab=0 \implies a=0 \text{ or } b=0
$$

A unique factorization domain (UFD) is an integral domain in which every nonzero nonunit factors uniquely into irreducibles up to order and units.

- The main question is when arithmetic behaves like the integers.

## Polynomial Rings

For a ring $R$,
$$
R[x]
$$
is the ring of polynomials in one variable over $R$.

Degree rule:
$$
\deg(fg)=\deg(f)+\deg(g)
$$
in an integral domain.

- Polynomial rings extend arithmetic into algebraic expressions while preserving structural control.

## Irreducibility

A polynomial $f(x)\in R[x]$ is irreducible if it cannot be written as
$$
f(x)=g(x)h(x)
$$
with both $g,h$ nonunits of smaller degree.

Typical tools:
- rational root test
- Eisenstein criterion
- reduction modulo $p$

- Irreducibility is the algebraic analogue of primality.
- It is central for constructing field extensions and understanding factorization.

---

# 7. Modules and Structural Generalization

## Modules as Generalized Vector Spaces

An $R$-module $M$ is an abelian group with scalar multiplication
$$
R \times M \to M
$$
satisfying the usual compatibility laws.

- Modules generalize vector spaces by replacing a field with a ring.
- Over a general ring, structure becomes richer and more subtle than in linear algebra.

## Finitely Generated Modules over PIDs

If $R$ is a PID, then finitely generated $R$-modules admit a decomposition theorem of the form
$$
M \cong R^r \oplus R/(d_1) \oplus \cdots \oplus R/(d_t)
$$
with
$$
d_1 \mid d_2 \mid \cdots \mid d_t
$$

- This theorem generalizes both:
  - finite abelian group classification
  - canonical forms from linear algebra

## Chinese Remainder Theorem

For pairwise comaximal ideals,
$$
R/(I_1\cdots I_n) \cong R/I_1 \times \cdots \times R/I_n
$$

A familiar integer version:
$$
\mathbb{Z}/mn\mathbb{Z} \cong \mathbb{Z}/m\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z}
\qquad \text{if } \gcd(m,n)=1
$$

- The CRT decomposes arithmetic into independent local pieces.

---

# 8. Fields and Extensions

## Fields

A field $F$ is a commutative ring in which every nonzero element is invertible:
$$
a \neq 0 \implies a^{-1} \in F
$$

- Fields are the natural setting for linear algebra, polynomial division, and classical algebraic computation.

## Field Extensions and Splitting Fields

An extension field is written
$$
E/F
$$
where $F \subseteq E$.

If $\alpha \in E$, then
$$
F(\alpha)
$$
is the smallest subfield containing both $F$ and $\alpha$.

A splitting field of a polynomial $f(x)\in F[x]$ is the smallest field over which $f$ factors completely into linear terms.

- Field extensions enlarge the number system just enough to solve polynomial equations.

## Finite Fields

A finite field exists exactly when its size is
$$
q = p^n
$$
for some prime $p$ and integer $n \ge 1$.

Such a field is denoted
$$
\mathbb{F}_q
$$

- Finite fields are fundamental in:
  - coding theory
  - cryptography
  - algebraic geometry
  - computational algebra

A key structural fact:
$$
\mathbb{F}_q^\times
$$
is a cyclic group of order $q-1$.

- This links field theory back to group structure and multiplicative symmetry.