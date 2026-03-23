---
title: Large-Scale Data Analytics
category: CS
semester: 2025 S
---
# 1. Data Models and Query Processing

## Relational Model
The relational model represents data as relations under a fixed schema.

$$
R(A_1,\dots,A_n)
$$

$$
r(R)\subseteq \mathrm{Dom}(A_1)\times\cdots\times \mathrm{Dom}(A_n)
$$

- A relation is a finite set of tuples.
- Logical structure is separated from physical storage.

## Relational Query Model
Relational query processing is built from a small set of core operators.

Selection:
$$
\sigma_{\theta}(R)
$$

Projection:
$$
\pi_{A_1,\dots,A_k}(R)
$$

Join:
$$
R \bowtie_{\theta} S
$$

Aggregation:
$$
\gamma_{G,\;f(A)}(R)
$$

- SQL is declarative; relational algebra is the logical model underneath it.

## SQL Querying
A user specifies the desired result, and the system determines the execution plan.

```sql
SELECT c.customer_id, SUM(o.amount) AS total_spend
FROM customers c
JOIN orders o
  ON c.customer_id = o.customer_id
WHERE o.order_date >= DATE '2025-01-01'
GROUP BY c.customer_id;
```

Indexes

Indexes accelerate lookup by organizing search keys separately from the base table.

$$
k \mapsto \text{record pointers}
$$

$$
O(N);\to;O(\log N)
$$

- Indexes improve search at the cost of extra storage and update overhead.

B+-Tree

Balanced tree indexes support point and range queries.

$$
\text{cost} \approx O(\log_B N)
$$

- B+-trees are the standard ordered index structure.

Hash Index

Hash indexes are optimized for equality predicates.

$$
h(k)\to \text{bucket}
$$

- Fast for exact-match lookup, but not for range scans.

Query Optimization

A query can be executed by many equivalent plans, so the system searches for a low-cost one.

$$
\text{BestPlan} = \arg\min_{P\in\mathcal P(Q)} \mathrm{Cost}(P)
$$

- Join order, access path, and filtering strategy strongly affect runtime.

---

# 2. Storage and Distributed Processing

## Large-Scale Storage
Large datasets are split across machines for parallel processing.

$$
D = \bigcup_{i=1}^m D_i
$$

- Partitioning enables scalability.
- Replication improves fault tolerance.

## File Systems and File Formats
Distributed file systems store large files as blocks across many nodes.

$$
\text{file} = { \text{blocks on many nodes} }
$$

Row-oriented storage:
$$
(r_1, r_2, r_3, \dots)
$$

Column-oriented storage:
$$
(A_1^{(1)},A_1^{(2)},\dots),;
(A_2^{(1)},A_2^{(2)},\dots)
$$

- Row formats are better for point access.
- Column formats are better for analytical scans and compression.

## MapReduce
MapReduce expresses distributed computation through map, shuffle, and reduce.

Map:
$$
\mathrm{map}(k,v)\to [(k_1,v_1),\dots,(k_r,v_r)]
$$

Reduce:
$$
\mathrm{reduce}(k,[v_1,\dots,v_t])\to \text{output}
$$

- Map emits key-value pairs.
- Shuffle groups values by key.
- Reduce aggregates grouped data.

Example aggregation pattern:

SELECT token, COUNT(*) AS cnt
FROM tokens
GROUP BY token
ORDER BY cnt DESC;

Distributed Query Processing

Distributed execution must reason about both operators and data movement.

$$
\text{Total Cost}

\text{CPU}+\text{Disk I/O}+\text{Network Shuffle}
$$

- Network shuffle is often the main bottleneck.

## Partitioning and Distributed Joins
Hash partitioning places tuples on workers by key.

$$
R_i = { t\in R : h(t.K)=i }
$$

A distributed join is
$$
R \bowtie_{R.K=S.K} S
$$

SELECT l.order_id, l.product_id, r.category
FROM lineitem l
JOIN products r
  ON l.product_id = r.product_id;

- Co-partitioning reduces join cost.
- Small tables may be broadcast instead of shuffled.

---

# 3. High-Level Data Systems

## Hive
Hive provides a SQL-like interface over distributed storage and batch execution.

- It allows analysts to query large datasets declaratively.
- Queries are compiled into distributed execution plans.

SELECT region,
       COUNT(*) AS num_orders,
       AVG(amount) AS avg_amount,
       SUM(amount) AS total_amount
FROM orders
WHERE order_date >= DATE '2025-01-01'
GROUP BY region
ORDER BY total_amount DESC;

## Schema-on-Read
Hive-style systems often interpret raw data at query time rather than enforcing schema at write time.

$$
\text{raw data} + \text{external schema} \to \text{queryable relation}
$$

- This is useful for heterogeneous analytical data.

## Spark
Spark supports general distributed data processing beyond classic MapReduce.

$$
D = T_n(T_{n-1}(\cdots T_1(D_0)\cdots))
$$

- Computation is expressed as a lineage of transformations.
- Lazy evaluation allows global optimization before execution.

## Spark SQL / DataFrames
Spark SQL reintroduces declarative processing on top of a distributed runtime.

SELECT user_id,
       COUNT(*) AS sessions,
       MAX(event_time) AS last_seen
FROM events
WHERE event_type = 'session_start'
GROUP BY user_id;

- Spark is especially effective for multi-stage and iterative workloads.

---

# 4. Cloud and Modern Data Workloads

## Cloud Computing
Cloud systems separate compute, storage, and services.

$$
\text{Analytics System} = \text{Elastic Compute} + \text{Shared Storage} + \text{Managed Services}
$$

- Resources scale with workload.
- Managed services reduce operational complexity.

## Data Integration and Warehouses
Modern analytics combines many data sources into unified warehouse tables.

$$
D_{\text{warehouse}} = T(D_1,D_2,\dots,D_k)
$$

A typical warehouse pattern is

$$
\text{Fact Table} \bowtie \text{Dimension Tables}
$$

SELECT d.year,
       p.category,
       SUM(f.sales_amount) AS revenue
FROM sales_fact f
JOIN date_dim d
  ON f.date_key = d.date_key
JOIN product_dim p
  ON f.product_key = p.product_key
GROUP BY d.year, p.category
ORDER BY d.year, revenue DESC;

- Fact tables store measurable events.
- Dimension tables store descriptive context.

## ETL / ELT
Data pipelines move data from sources into analytical systems.

$$
D_{\text{sources}} \to D_{\text{staging}} \to D_{\text{warehouse}}
$$

- ETL transforms before loading.
- ELT loads first and transforms inside the warehouse.

## Data Streams
Streaming systems process continuously arriving records.

$$
S={(x_1,t_1),(x_2,t_2),\dots}
$$

Windowed aggregation:
$$
A_t = \mathrm{Agg}\big(S[t,t+\Delta)\big)
$$

SELECT window_start,
       window_end,
       COUNT(*) AS events,
       AVG(latency_ms) AS avg_latency
FROM stream_events
GROUP BY TUMBLE(event_time, INTERVAL '5' MINUTE);

- Stream processing introduces event time, windowing, and late-data handling.

---

# 5. Graph and Specialized Processing

## Graph Model
Some large-scale workloads are better expressed as graphs than tables.

$$
G=(V,E)
$$

- Vertices represent entities.
- Edges represent relationships.

## Vertex-Centric Processing
Pregel and Giraph organize graph computation around per-vertex updates.

$$
\mathrm{state}_v^{(t+1)}

F\bigl(\mathrm{state}v^{(t)},{\mathrm{msg}{u\to v}^{(t)}}_{u\in N(v)}\bigr)
$$

- Computation proceeds in synchronized supersteps.
- Each vertex updates state and sends messages to neighbors.

## Pregel / Giraph
These systems are designed for iterative graph algorithms such as PageRank and shortest paths.

PageRank update:
$$
PR(v)=\frac{1-d}{|V|} + d\sum_{u\to v}\frac{PR(u)}{\mathrm{outdeg}(u)}
$$

- Graph processing differs from standard SQL because it is iterative and relationship-driven.

---

# 6. Core System Themes

## Abstraction and Execution
A recurring theme in large-scale analytics is the separation between user intent and distributed execution.

$$
\text{User Query / Program}
;\to;
\text{Logical Plan}
;\to;
\text{Physical Plan}
;\to;
\text{Distributed Execution}
$$

- Users express computation at a high level.
- Engines handle optimization, scheduling, and fault tolerance.

## Data Locality and System Design
Performance improves when computation is moved closer to data.

$$
\text{minimize network movement}
$$

- This principle appears across databases, MapReduce, Spark, warehouses, and graph systems.

## Unifying View
Large-scale data analytics repeatedly asks three questions:
1. How is data modeled?
2. How is computation expressed?
3. How is execution scaled?

$$
\text{abstraction} \leftrightarrow \text{performance} \leftrightarrow \text{scalability}
$$