---
title: Discrete Mathematics
category: "MATHEMATICS / COMPUTER SCIENCE"
semester: 2023 S
---

# 1. Mathematical Foundations

## Logic and Formal Reasoning

### Propositional logic

$$p,\ q,\ \neg p,\ p\land q,\ p\lor q,\ p\to q,\ p\leftrightarrow q$$

- Propositional logic studies statements that are either true or false.
- Complex statements are built from simpler ones using logical connectives.

### Predicate logic

$$P(x),\ \forall x\, P(x),\ \exists x\, P(x)$$

- Predicate logic extends propositional logic by allowing variables and quantified statements.
- It is the language used to express mathematical definitions and general claims.

### Quantifiers and inference

$$\forall x\in A,\ P(x) \qquad \exists x\in A,\ P(x)$$

Typical inference rules:

$$p,\ p\to q \ \Rightarrow\ q$$

$$\forall x\, P(x)\ \Rightarrow\ P(a)$$

- Much of mathematical reasoning is about converting assumptions into valid conclusions using formal rules.

## The Language of Mathematics

### Functions

$$f:A\to B,\qquad f(a)=b$$

- A function assigns each input in a domain to exactly one output in a codomain.
- Functions formalize transformation, dependence, and structure.

### Set theory

$$A\subseteq B,\qquad A\cup B,\qquad A\cap B,\qquad A\setminus B,\qquad A\times B$$

- Sets are the basic objects used to define number systems, relations, graphs, and languages.
- Much of discrete mathematics is written in set-theoretic language.

### Relations

$$R\subseteq A\times A$$

Common properties:

- reflexive  
- symmetric  
- antisymmetric  
- transitive  

- Relations describe structured connections between elements.
- Equivalence relations and partial orders are especially important recurring examples.

## Proof Techniques

### Direct proof

$$P \Rightarrow Q$$

- Start from the hypothesis and derive the conclusion directly.

### Proof by contrapositive

$$(P\Rightarrow Q)\equiv (\neg Q\Rightarrow \neg P)$$

- Often easier than direct proof when the conclusion is difficult to handle directly.

### Proof by contradiction

- Assume the statement is false and derive a contradiction.

### Mathematical induction

Weak induction:

$$P(1),\quad P(k)\Rightarrow P(k+1)$$

Strong induction:

$$P(1),\dots,P(k)\Rightarrow P(k+1)$$

- Induction is the standard method for proving statements indexed by the positive integers.
- It is central in recursive definitions, algorithms, and combinatorial formulas.

---

# 2. Discrete Structures

## Basic Number Theory

### Divisibility and primes

$$a\mid b \iff \exists k\in\mathbb{Z}\text{ such that }b=ak$$

- Divisibility is one of the most basic discrete relations on integers.
- Prime numbers are the indivisible building blocks of the integers.

### Modular arithmetic

$$a\equiv b \pmod n \iff n\mid (a-b)$$

- Modular arithmetic studies integers up to remainder classes.
- It is fundamental in cryptography, number theory, and algorithm design.

### Number representation

$$n=(a_k a_{k-1}\dots a_0)_b=\sum_{i=0}^k a_i b^i$$

- Binary and base-$b$ representation connect number theory to computation.
- Digital systems are built on these finite symbolic representations.

## Counting

### Permutations and combinations

$$P(n,r)=\frac{n!}{(n-r)!},\qquad \binom{n}{r}=\frac{n!}{r!(n-r)!}$$

- Counting provides exact formulas for finite arrangements and selections.
- It is the combinatorial backbone of discrete probability.

### Pigeonhole principle

- If more than $n$ objects are placed into $n$ boxes, then some box contains at least two objects.

- This principle is simple but powerful: existence can often be proved without explicitly constructing an example.

### Inclusion–exclusion

$$|A\cup B|=|A|+|B|-|A\cap B|$$

More generally,

$$\left|\bigcup_{i=1}^n A_i\right|=\sum |A_i|-\sum |A_i\cap A_j|+\cdots$$

- Inclusion–exclusion corrects for overcounting when sets overlap.

## Discrete Probability

### Finite probability spaces

$$P(A)=\frac{|A|}{|S|}\qquad \text{(equally likely outcomes)}$$

- Discrete probability treats uncertainty on finite or countable sample spaces.

### Conditional probability

$$P(A\mid B)=\frac{P(A\cap B)}{P(B)}$$

- Conditional probability updates probability after partial information is known.

### Independence

$$P(A\cap B)=P(A)P(B)$$

- Independence expresses the idea that one event does not change the probability of another.
- Many probabilistic arguments depend on distinguishing independence from mere non-overlap.

---

# 3. Graphs and Trees

## Graphs

### Basic graph model

$$G=(V,E)$$

- A graph is a set of vertices together with edges connecting them.
- Graphs model networks, communication, dependencies, and discrete structure.

### Directed and undirected graphs

Undirected:

$$E\subseteq \{\{u,v\}:u,v\in V\}$$

Directed:

$$E\subseteq V\times V$$

- Undirected graphs model symmetric relationships.
- Directed graphs model one-way relationships such as flow, precedence, or transition.

### Connectivity, paths, and cycles

A path:

$$v_0,v_1,\dots,v_k$$

- Connectivity asks whether vertices can be reached from one another.
- Cycles detect feedback, recurrence, and non-tree structure.

## Trees

### Tree structure

A tree is a connected graph with no cycles.

Equivalent characterization:

$$|E|=|V|-1$$

- Trees are the minimal connected graphs.
- They appear naturally in recursion, hierarchies, parsing, and search.

### Tree traversals

- preorder  
- inorder  
- postorder  
- breadth-first traversal  

- Traversal rules determine how tree structure is explored algorithmically.

### Spanning trees

A spanning tree of a graph is a subgraph that:

- contains all vertices  
- is connected  
- has no cycles  

- Spanning trees extract essential connectivity while removing redundancy.

---

# 4. Automata and Formal Languages

## Finite Automata

### DFA

A deterministic finite automaton is typically written as

$$M=(Q,\Sigma,\delta,q_0,F)$$

- $Q$: set of states  
- $\Sigma$: input alphabet  
- $\delta$: transition function  
- $q_0$: start state  
- $F$: accepting states  

- A DFA reads a string symbol by symbol and changes state deterministically.

### NFA

An NFA has a transition rule that allows multiple possible next states.

- NFAs and DFAs define the same class of languages, but NFAs are often easier to design.

### Language recognition

$$L(M)=\{w\in\Sigma^* : M \text{ accepts } w\}$$

- Automata are machines for recognizing formal languages.
- This connects syntax, computation, and system behavior.

## Regular Expressions

### Regular languages

Regular expressions describe patterns such as:

$$a^*,\quad (a\mid b),\quad ab,\quad (ab)^*$$

- Regular expressions provide an algebraic language for describing simple formal patterns.

### Equivalence with finite automata

- regular expressions  
- DFAs  
- NFAs  

all describe the same class of languages.

- This equivalence is one of the key unifying results of early formal language theory.

## Finite State Machines

- Finite state machines generalize automata as abstract models of reactive systems.
- They are used to represent controllers, protocols, and digital logic systems.
- The main idea is that a system with finitely many states can be analyzed through state transitions.

---

# 5. Computation and Complexity

## Turing Machines

### Formal model of computation

A Turing machine is an abstract machine with:

- a finite control  
- an infinite tape  
- a read/write head  
- a transition rule  

- Turing machines formalize the intuitive notion of algorithmic computation.
- They are more expressive than finite automata because they have unbounded memory.

### Church–Turing thesis

- Any effectively computable procedure can be modeled by a Turing machine.

- This is not a theorem in the usual mathematical sense, but a foundational claim about the nature of computation.

## Computability

### Decidable problems

A problem is decidable if there exists a Turing machine that halts on every input and returns the correct yes/no answer.

### Undecidable problems

- Some well-defined problems cannot be solved algorithmically for all inputs.

- This distinction shows that there are fundamental limits to computation, even before efficiency is considered.

## Complexity Theory

### Time and space complexity

Time complexity measures how running time grows with input size:

$$T(n)$$

Space complexity measures how memory usage grows with input size:

$$S(n)$$

- Complexity theory studies not just whether a problem is solvable, but how resource demands scale.

### Complexity classes

$$P=\{\text{problems solvable in polynomial time}\}$$

$$NP=\{\text{problems whose solutions can be verified in polynomial time}\}$$

- $P$ represents efficiently solvable problems.
- $NP$ represents efficiently verifiable problems.
- The relationship between $P$ and $NP$ is one of the most important open questions in theoretical computer science.

## Course perspective

- The course moves from logical reasoning and proof  
- to discrete mathematical structures  
- to formal models of language and computation  
- and finally to the limits and cost of computation itself  

- In that sense, discrete mathematics provides the conceptual language of theoretical computer science.
