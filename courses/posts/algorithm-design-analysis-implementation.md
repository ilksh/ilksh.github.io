---
title: Algorithm Design, Analysis, and Implementation
category: CS
semester: 2024 F
---

# 1. Asymptotic Order of Growth


## Big-O, $\Theta$, and $\Omega$

- **Big-O Notation ($O$):**  
  This notation provides an upper bound on the growth of a function. A function $f(n)$ is said to be $O(g(n))$ if there exist positive constants $c$ and $n_0$ such that
  $$
  f(n) \leq c \cdot g(n)
  $$
  for all $n \geq n_0$.

- **Theta Notation ($\Theta$):**  
  This notation represents a tight bound on the growth of a function. A function $f(n)$ is $\Theta(g(n))$ if it is both $O(g(n))$ and $\Omega(g(n))$, meaning
  $$
  c_1 \cdot g(n) \leq f(n) \leq c_2 \cdot g(n)
  $$
  for some positive constants $c_1, c_2$, and sufficiently large $n$.

- **Omega Notation ($\Omega$):**  
  This notation provides a lower bound on the growth of a function. A function $f(n)$ is $\Omega(g(n))$ if there exist positive constants $c$ and $n_0$ such that
  $$
  f(n) \geq c \cdot g(n)
  $$
  for all $n \geq n_0$.

---

# 2. Greedy Algorithms

Greedy algorithms are a class of algorithms that solve optimization problems by making a sequence of choices, each of which is locally optimal at the current stage. They commit to each choice, focusing **only on the immediate benefits** without considering future consequences.

## 2.1 Matching Problem

### Algorithm: Propose-and-Reject (Gale–Shapley style)

1. Initialize all elements in one set (e.g., applicants) as unmatched.
2. While there exist unmatched elements in the proposing set:
   - Each unmatched element proposes to its most preferred option that has not yet rejected it.
   - Each element in the receiving set evaluates its current proposals and accepts the best one (if not already matched).
   - Reject all other proposals.
3. Repeat until no unmatched elements remain in the proposing set.

### Time Complexity

For $n$ elements with $m$ preferences, the complexity is
$$
O(m)
$$

## 2.2 Job Scheduling

### Algorithm: Earliest Deadline First (EDF)

1. Sort all jobs by their deadlines in ascending order.
2. Assign each job sequentially to the earliest available time slot.
3. If a job cannot be completed before its deadline, compute its lateness.

### Time Complexity

Overall complexity is
$$
O(n \log n)
$$

## 2.3 Interval Partitioning

### Algorithm: Earliest Start Time First

1. Sort all intervals by their start times in ascending order.
2. Assign each interval to the earliest available resource that does not conflict with it.
3. If no resource is available, allocate a new resource.

### Time Complexity

The total complexity is
$$
O(n \log n)
$$

## 2.4 Minimize Lateness

### Algorithm: Earliest Deadline First (EDF)

1. Sort all jobs by their deadlines in ascending order.
2. Assign each job sequentially to the earliest available time slot.
3. Record the lateness for each job as the difference between its completion time and deadline, if positive.

### Time Complexity

Overall complexity is
$$
O(n \log n)
$$

---

# 3. Graph Algorithms

Graphs are fundamental data structures consisting of a set of nodes (vertices) and edges connecting pairs of nodes.

## 3.1. DFS & BFS

### Depth-First Search (DFS)

DFS explores as far as possible along each branch before backtracking. It uses a stack, either explicitly or through recursion, to track visited nodes.

```cpp
void dfs(int num_1) {
    visited[num_1][0] = 1;
    for (int i = 1; i < n + 1; ++i) {
        if (arr[num_1][i] && !visited[i][0]) {
            dfs(i);
        }
    }
}
```

### Time Complexity

$$
O(V + E)
$$

### Breadth-First Search (BFS)

BFS explores all neighbors at the current depth before moving to the next level. It uses a queue to track nodes to be visited.

```cpp
void bfs(int num_2) {
    q.push(num_2);
    while (!q.empty()) {
        auto cur = q.front(); q.pop();
        visited[cur][1] = 1;

        for (int i = 1; i <= n; ++i) {
            if (arr[cur][i] && !visited[i][1]) {
                q.push(i);
                visited[i][1] = 1;
            }
        }
    }
}
```

### Time Complexity

$$
O(V + E)
$$

## 3.2. Dijkstra

Dijkstra is a greedy algorithm used to find the shortest paths from a single source node to all other nodes in a graph with **non-negative edge weights**. It maintains a priority queue so that the next node processed always has the smallest known distance.

```cpp
while (!pq.empty()) {
    pii cur = pq.top(); pq.pop();

    if (disp[cur.second] != -1 * cur.first) continue;

    for (pii e : adj[cur.second]) {
        int next = e.second;
        int next_disp = (-1 * cur.first) + e.first;

        if (next_disp < disp[next]) {
            disp[next] = next_disp;
            pq.push({ -1 * disp[next], next });
        }
    }
}
```
### Time Complexity

- In a graph with $V$ vertices and $E$ edges:
  - Each vertex is inserted into the priority queue at most once, and each insertion or extraction takes
    $$
    O(\log V)
    $$
  - Each edge is relaxed at most once, and relaxing an edge may require updating the priority queue, which also takes
    $$
    O(\log V)
    $$
- Total complexity:
  $$
  O((V + E)\log V)
  $$

## 3.3. Knapsack Problem

The Knapsack Problem is a classic optimization problem where each item has a weight and a value, and the knapsack has a maximum capacity. The goal is to maximize the total value of selected items without exceeding the capacity.

```cpp
int solve(int idx, int capacity) { // return the maximum value
    if (capacity < 0) return -0x3f3f3f3f; 
    if (idx > n) return 0; // Base case: no more items

    int& ret = dp[idx][capacity]; // Use memoized result
    if (ret != -1) return ret;
    // Option 1: Do not choose the current item
    int notChoose = solve(idx + 1, capacity);
    // Option 2: Choose the current item, reduce capacity
    int choose = solve(idx + 1, capacity - w[idx]) + v[idx];

    return ret = max(notChoose, choose); 
}
```

### Time Complexity

- There are $n$ items and a maximum capacity $C$.
- Each state $dp[i][c]$ is computed once, leading to
  $$
  O(n \times C)
  $$
  time complexity.
- Space complexity is also
  $$
  O(n \times C)
  $$
  for the $dp$ array. This can be optimized to
  $$
  O(C)
  $$
  using a 1D array.


## 3.4. Bellman-Ford

Bellman-Ford is a shortest path algorithm that can handle graphs with negative edge weights and can also detect negative-weight cycles.

```cpp
bool bellmanFord(int start) {
    bool isCycle = false;
    memset(dist, 0x3f, sizeof(dist)); // initialize the distance
    dist[start] = 0;
    
    // repeat multiple rounds to determine negative cycle existence
    for(int round = 0; round < m; ++round){
        bool updated = false;
        
        for(auto& e: edge) {
            int newDist = dist[e.u] + e.w ;
            if (dist[e.u] == INF) continue;
            if (newDist >= dist[e.v]) continue;
            
            dist[e.v] = dist[e.u] + e.w;
            updated = true;
            
            // m - 1 repeat means continues to be update
            // there is a negative cycle
            if(round == m-1) isCycle = true;
        }
        if (!updated) break;
    }
    
    return isCycle;
}
```

### Time Complexity

$$
O(V \cdot E)
$$

## 3.5. Topological Sorting

Topological sorting orders the vertices of a **directed acyclic graph (DAG)** such that for every directed edge $(u, v)$, vertex $u$ appears before $v$ in the ordering.

```cpp
for(int i=1; i<=n; i++) {
    if(indegree[i] == 0) {
        visited[i] = true;
        pq.push(-i); // Push negative for min-heap behavior
    }
}

while(!pq.empty()) {
    int cur = -1 * pq.top(); pq.pop();
    printf("%d ", cur);
    for(int i=0; i<v[cur].size(); i++) {
        int next = v[cur][i];
        if(--indegree[next] == 0 && !visited[next]) {
            visited[next] = true;
            pq.push(-next);
        }
    }
}
```

### Time Complexity

- Computing indegrees for all vertices takes
  $$
  O(V + E)
  $$
- Each vertex is pushed and popped from the priority queue exactly once, taking
  $$
  O(V \log V)
  $$
- Overall complexity:
  $$
  O(V \log V + E)
  $$

---

# 4. Tree

A tree is a **hierarchical data structure** consisting of nodes connected by edges. It is an acyclic, connected graph where one node is designated as the root, and every other node has exactly one parent.

## 4.1. Union-Find

Union-Find is a data structure used to efficiently handle dynamic connectivity problems. It supports two main operations:

- **Find:** Determine the representative parent of the set containing a particular element.
- **Union:** Merge two sets into one.

```cpp
void unite(int a, int b) {
    a = find(a); // find the parent of a
    b = find(b); // find the parent of b
    par[a] = b;
}

int find(int x) {
    if (par[x] == x) return x;  // Base case: root node
    par[x] = find(par[x]);      // Path compression
    return par[x];
}
```

### Time Complexity

- **Find:**
  $$
  O(\alpha(n))
  $$
  where $\alpha(n)$ is the inverse Ackermann function.

- **Union:**
  $$
  O(\alpha(n))
  $$

- **Overall:**
  $$
  O(m \cdot \alpha(n))
  $$

## 4.2. Kruskal's Algorithm

Kruskal's algorithm is a **greedy algorithm** used to find the Minimum Spanning Tree (MST) of a graph. The MST is a subset of edges that connects all vertices with minimum total edge weight, without forming cycles.

```cpp
typedef vector<int> vi;
typedef pair<int, int> pii;
typedef pair<int, pii> ipii;

vi par;
vector<ipii> edges;

int V, E; // number of vertices and edges

int find(int x) {
    if (x == par[x]) return x;
    return par[x] = find(par[x]);
}

void unite(int x, int y) {
    x = find(x);
    y = find(y);
    par[x] = y;
}

int kruskal() {
    sort(edges.begin(), edges.end()); // sort edges by weight

    int ans = 0;
    int cnt = 0;
    for (int i = 0; i < edges.size(); ++i) {
        auto curEdge = edges[i];
        int curWeight = curEdge.first;
        int from = curEdge.second.first;
        int to = curEdge.second.second;

        // cycle check
        if (find(from) == find(to)) continue;

        unite(from, to);
        ans += curWeight;
        if (++cnt == V - 1) break;
    }

    return ans;
}
```

### Time Complexity

- Sorting the edges takes
  $$
  O(E \log E)
  $$
  where $E$ is the number of edges.

- Each `find` and `union` operation in Union-Find takes
  $$
  O(\alpha(n))
  $$

- For $E$ edges, the total Union-Find cost is
  $$
  O(E \cdot \alpha(n))
  $$

- Overall complexity:
  $$
  O(E \log E + E \cdot \alpha(n)) \approx O(E \log E)
  $$

## 4.3. Prim's Algorithm

Prim's algorithm is a greedy algorithm for finding the Minimum Spanning Tree (MST) of a weighted, connected, and undirected graph. Unlike Kruskal's algorithm, which operates on edges, Prim's algorithm grows the MST one vertex at a time by adding the smallest edge that connects a vertex in the MST to a vertex outside the MST.

```cpp
int prim() {
    priority_queue<pii> pq;
    int ans = 0;

    visited[1] = true;
    for (int i = 0; i < graph[1].size(); ++i) {
        int to = graph[1][i].first;
        int curW = graph[1][i].second;
        pq.push({-curW, to});
    }

    while(!pq.empty()) {
        auto cur = pq.top(); pq.pop();

        int curNode = cur.second;
        int curWeight = -cur.first;

        if (visited[curNode]) continue;

        ans += curWeight;
        visited[curNode] = true;

        for (int i = 0; i < graph[curNode].size(); ++i) {
            int nextWeight = -graph[curNode][i].second;
            int nextNode = graph[curNode][i].first;
            pq.push({nextWeight, nextNode});
        }
    }
    return ans;
}
```

### Time Complexity

- Let $V$ be the number of vertices and $E$ the number of edges.
- Inserting edges into the priority queue and extracting the smallest edge take
  $$
  O(\log E)
  $$
- In the worst case, all edges are processed, leading to
  $$
  O(E \log E)
  $$
- For dense graphs where $E \approx V^2$, the complexity becomes
  $$
  O(V^2 \log V)
  $$
- For sparse graphs where $E \approx V$, the complexity simplifies to
  $$
  O(V \log V)
  $$

---

# 5. Divide and Conquer

Divide and conquer algorithms solve a problem by dividing it into smaller subproblems, recursively solving each subproblem, and then combining the subproblem solutions to solve the original problem.

## 5.1. Square Matrix Power Calculation

```cpp
class SquareMatrix {
private:
    vector<vector<int>> matrix;
    int n;

public:
    // Constructor to initialize an n x n matrix
    SquareMatrix(int size) : n(size) {
        matrix.resize(n, vector<int>(n, 0));
    }

    int size() const {return n;}

    // Access element (i, j)
    int& operator()(int i, int j) {return matrix[i][j];}

    SquareMatrix operator*(const SquareMatrix& other) const {
        SquareMatrix result(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    result(i, j) += matrix[i][k] * other.matrix[k][j];
                }
            }
        }
        return result;
    }
};

// Return the identity matrix whose size is n x n
SquareMatrix identityMatrix(int n) {
    SquareMatrix I(n);
    for (int i = 0; i < n; ++i) I(i, i) = 1;
    return I;
}

// calculate A^m
SquareMatrix pow(const SquareMatrix& A, int m) {
    // base case: A^0 = identity matrix
    if (m == 0) return identityMatrix(A.size());

    if (m % 2 > 0) return pow(A, m - 1) * A;
    SquareMatrix half = pow(A, m / 2);

    // A^m = (A^{m/2}) * (A^{m/2})
    return half * half;
}
```

For two $n \times n$ matrices, matrix multiplication takes
$$
O(n^3)
$$
Using exponentiation by squaring reduces the number of multiplications to
$$
O(\log m)
$$
so the overall time complexity of matrix exponentiation is
$$
O(n^3 \log m)
$$


## 5.2. Karatsuba Algorithm

Given two $n$-digit numbers $X$ and $Y$, the goal is to compute
$$
X \times Y
$$

Write them as
$$
X = X_1 \times 10^{n/2} + X_0
$$
$$
Y = Y_1 \times 10^{n/2} + Y_0
$$

where $X_1, Y_1$ are the higher-order halves and $X_0, Y_0$ are the lower-order halves.

Introduce the intermediate value
$$
Z_1 = (X_1 + X_0)(Y_1 + Y_0)
$$

Expanding,
$$
Z_1 = X_1Y_1 + X_1Y_0 + X_0Y_1 + X_0Y_0
$$

Using this, the original product becomes
$$
XY = X_1Y_1 \times 10^n + \left(Z_1 - X_1Y_1 - X_0Y_0\right) \times 10^{n/2} + X_0Y_0
$$

This requires only three multiplications:

1. $A = X_1Y_1$
2. $B = X_0Y_0$
3. $C = (X_1 + X_0)(Y_1 + Y_0)$

Final formula:
$$
XY = A \times 10^n + (C - A - B)\times 10^{n/2} + B
$$

## 5.3. Fast Fourier Transform (FFT)

The goal is to compute, for a sequence of length $n$,
$$
a_0, a_1, \dots, a_{n-1}
$$

the values
$$
A_k = \sum_{j=0}^{n-1} a_j \omega_n^{jk}
$$

where $\omega_n$ is an $n$-th root of unity.

The key idea is to split terms into even-indexed and odd-indexed parts.

Given
$$
A(x) = a_0 + a_1x + a_2x^2 + \cdots + a_{n-1}x^{n-1}
$$

we rewrite it as
$$
A(x) = A_{\text{even}}(x^2) + xA_{\text{odd}}(x^2)
$$

This reduces the problem recursively and leads to the classic FFT runtime:
$$
O(n \log n)
$$

```cpp
#include<bits/stdc++.h>

using namespace std;

typedef vector<int> vi;
typedef long long ll;

typedef complex<double> base;
typedef vector<base> vb;

const double PI = acos(-1.0);

void fft(vb &a, bool invert) {
    int n = a.size();
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        while (j >= bit) {
            j -= bit;
            bit >>=1;
        }
        j += bit;
        if (i < j) swap(a[i], a[j]);
    }

    for (int len = 2; len < n + 1; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        base wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            base w(1);
            for (int j = 0; j < len / 2; ++j) {
                base u = a[i + j];
                base v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (base &x : a) x /= n;
    }
}

vi multiply(const vi& a, const vi& b) {
    vb fa(a.begin(), a.end());
    vb fb(b.begin(), b.end());

    int n = 1;
    int mxSize = a.size() + b.size();
    while (n < mxSize) n <<= 1;

    fa.resize(n);
    fb.resize(n);

    fft(fa, false);
    fft(fb, false);

    for (int i = 0; i < n; ++i) fa[i] *= fb[i];

    fft(fa, true);

    vi result(n);
    ll carry = 0;

    for (int i = 0; i < n; ++i) {
        ll t = (ll) (fa[i].real() + 0.5) + carry;
        carry = t / 10;
        result[i] = t % 10;
    }

    while (result.size() > 1 && result.back() == 0) result.pop_back();

    return result;
}
```

## 5.4. Merge Sort

Merge Sort **divides** the input array into smaller subarrays, recursively sorts them, and then **merges** the sorted subarrays back together.

```cpp
void merge(int *arr, int left, int right) {
    int mid = (left + right) / 2;
    
    int i, j, k = 0;
    int temp[MAX_N];
    
    i = left;
    j = mid + 1;
    
    while(i <= mid && j <= right) {        
        if(arr[i] < arr[j]) temp[k] = arr[i++];
        else temp[k] = arr[j++];
        k++;
    }
    
    while(i <= mid) temp[k++] = arr[i++];
    
    while(j <= right) temp[k++] = arr[j++];
    
    // arr[left...right] = temp[0..k-1]
    for(int i = 0; i < k; ++i) arr[left + i] = temp[i];
}

// arr[left..right] sorting
void mergeSort(int *arr, int left, int right) {
    if(left < right) {
        int mid = (left + right) / 2;
        mergeSort(arr, left, mid);      // arr[left..mid] sorting
        mergeSort(arr, mid + 1, right);   // arr[mid+1..right] sorting
        merge(arr, left, right);        // arr[left..right] sorting
    }
}
```

### Time Complexity

Merge Sort operates in
$$
O(n \log n)
$$
time, where $n$ is the size of the array.

- **Divide Step:** Splitting the array into two halves takes
  $$
  O(1)
  $$
- **Conquer Step:** Sorting each half gives the recurrence
  $$
  T(n) = 2T(n/2) + O(n)
  $$
- **Combine Step:** Merging the two sorted halves takes
  $$
  O(n)
  $$
- Solving the recurrence yields
  $$
  O(n \log n)
  $$

## 5.5. Closest Pair Algorithm

The Closest Pair Algorithm finds the minimum distance between any two points in a set of 2D points.

### Algorithm

1. **Divide:** Split the set of points into two halves based on their $x$-coordinates.
2. **Conquer:** Recursively find the closest pair in each half.
3. **Combine:** Check whether a closer pair exists across the dividing line by examining points inside a strip whose width is determined by the best distance found so far.

```cpp
// get the distance between two points
int distance(int ptOne, int ptTwo) {
    int lowX = arr[ptOne].first;
    int lowY = arr[ptOne].second;

    int highX = arr[ptTwo].first;
    int highY = arr[ptTwo].second;

    int distX = highX - lowX;
    int distY = highY - lowY;
    
    return distX * distX + distY * distY;
}

// using binary search to find the closest pair
int bst(int low, int high) {
    if (low == high) return MAX;
    if (low + 1 == high) return distance(low, high);

    int mid = (low + high) / 2;
    int distMin = min(bst(low, mid), bst(mid + 1, high)); // recursive call

    // middle area's minimum distance
    vector<pii> inner;
    int lineX = arr[mid].first;

    for (int i = mid; i >= low; --i) { // left area
        int x = arr[i].first;
        int dist = lineX - x;
        if (dist * dist >= distMin) break;
        inner.emplace_back(arr[i].second, arr[i].first); // {y, x}
    }

    for (int i = mid + 1; i <= high; ++i) { // right area
        int x = arr[i].first;
        int dist = lineX - x;
        if (dist * dist >= distMin) break;
        inner.emplace_back(arr[i].second, arr[i].first); // {y, x}
    }

    // if no valid points found, return the current minimum distance
    if (inner.empty()) return distMin;

    // sort by y coordinate
    sort(inner.begin(), inner.end());

    for (int i = 0; i < inner.size(); ++i) {
        for (int j = i + 1; j < inner.size(); ++j) {
            int dy = inner[j].first - inner[i].first;
            if (dy * dy >= distMin) break;

            int dx = inner[j].second - inner[i].second;
            int dist = dx * dx + dy * dy;
            distMin = min(distMin, dist);
        }
    }
    return distMin;
}
```

### Time Complexity

The time complexity is
$$
O(n \log n)
$$

---

# 6. Dynamic Programming

Dynamic Programming (DP) is an algorithmic technique for solving complex problems by **breaking them down** into simpler subproblems, solving each subproblem once, and storing their solutions. This **avoids redundant** computations and is especially useful for optimization problems.

## 6.1. Memoization

**Memoization** is a key concept in DP where the results of solved subproblems are stored for future reference. This technique is typically used with recursion in the **top-down approach** to avoid recalculating the same subproblem.

```cpp
// Top-Down Dynamic Programming with Memoization
int topDown(int n) {
    // base case
    if (n == 1) return stairs[1];
    if (n == 2) {
        return stairs[1] + stairs[2];
    }

    int& ret = cache[n];  
    if (ret != -1) {
        return ret;
    }

    // (n - 3) -> (n - 1) -> n
    int routeOne = topDown(n - 3) + stairs[n - 1] + stairs[n];
    // (n - 2) -> n
    int routeTwo = topDown(n - 2) + stairs[n];

    ret = max(routeOne, routeTwo);
    return ret;
}

// Bottom-Up Dynamic Programming
int bottomUp(int n) {
    dp[1] = stairs[1];
    dp[2] = stairs[1] + stairs[2];

    for (int i = 3; i < n + 1; ++i) {
        int routeOne = dp[i - 3] + stairs[i - 1];
        int routeTwo = dp[i - 2];
        dp[i] = max(routeOne, routeTwo) + stairs[i];
    }
    return dp[n];
}
```

## 6.2. Weighted Interval Scheduling

The Weighted Interval Scheduling problem is a classic optimization problem where we are given a set of $n$ intervals, each with a start time, an end time, and a weight (value).

- Each interval $i$ is represented as
  $$
  (s_i, e_i, w_i)
  $$
  where $s_i$ is the start time, $e_i$ is the end time, and $w_i$ is the weight.

- Two intervals $i$ and $j$ overlap if
  $$
  e_i > s_j \quad \text{and} \quad e_j > s_i
  $$

- The goal is to select a subset of intervals
  $$
  S \subseteq \{1,2,\dots,n\}
  $$
  such that:
  1. Intervals in $S$ do not overlap.
  2. The total weight
     $$
     \sum_{i \in S} w_i
     $$
     is maximized.

## 6.3. Longest Common Subsequence (LCS)

The **LCS** problem asks for the longest subsequence common to two strings. Unlike substrings, subsequences do not need to occupy consecutive positions.

The recurrence is:
$$
dp[i][j] =
\begin{cases}
dp[i-1][j-1] + 1 & \text{if } X[i-1] = Y[j-1], \\
\max(dp[i-1][j], dp[i][j-1]) & \text{otherwise.}
\end{cases}
$$

### Base Case

$$
dp[i][0] = 0, \qquad dp[0][j] = 0
$$

```cpp
str1 = 'a' + str1;
str2 = 'a' + str2;
int sz1 = str1.size() - 1;
int sz2 = str2.size() - 1;

for (int i = 1; i < sz1 + 1; ++i) {
    for (int j = 1; j < sz2 + 1; ++j) {
        if (str1[i] == str2[j]) 
            dp[i][j] = dp[i - 1][j - 1] + 1;
        else {
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
        }
    }
}
```

### Time and Space Complexity Analysis

- **Time.** Filling the $dp$ table requires
  $$
  O(m \times n)
  $$
  operations, where $m$ and $n$ are the lengths of $X$ and $Y$.

- **Space.** The algorithm requires
  $$
  O(m \times n)
  $$
  space to store the $dp$ table.

### Space Optimization

The $2D$ table in the standard algorithm can be reduced to a $1D$ array because $dp[i][j]$ only depends on:
- $dp[i-1][j-1]$
- $dp[i-1][j]$
- $dp[i][j-1]$

Using a rolling array, the space complexity can be reduced to
$$
O(n)
$$

```cpp
int longestCommonSubsequence(const string& X, const string& Y) {
    int m = X.size(), n = Y.size();
    vector<int> dp(n + 1, 0); 

    for (int i = 1; i <= m; ++i) {
        int prev = 0; 
        for (int j = 1; j <= n; ++j) {
            int temp = dp[j]; // dp[j]
            if (X[i - 1] == Y[j - 1]) {
                dp[j] = prev + 1; // dp[j] = dp[i-1][j-1] + 1
            } else {
                // max(dp[i-1][j], dp[i][j-1])
                dp[j] = max(dp[j], dp[j - 1]); 
            }
            prev = temp; // dp[i-1][j-1]
        }
    }

    return dp[n];
}
```

---

# 7. Network Flow

The Network Flow Algorithm is used to solve problems related to finding the maximum flow from a source node to a sink node in a graph. This algorithm is widely used in applications such as transportation systems, electrical circuits, data transmission networks, and more.

## 7.1. Ford-Fulkerson Algorithm

The Ford-Fulkerson algorithm works by repeatedly finding augmenting paths, that is, paths with available residual capacity, and increasing the flow along these paths.

```cpp
int dfs(int here, int sink, int minFlow) {
    if (here == sink) return minFlow;
    visited[here] = true;

    for (int there = 0; there < V; ++there) {
        if (visited[there]) continue;
        int r = capacity[here][there] - flow[here][there];
        if (r <= 0) continue;

        int possibleFlow = min(minFlow, r);
        int resultFlow = dfs(there, sink, possibleFlow);
        if (resultFlow > 0) {
            flow[here][there] += resultFlow;
            flow[there][here] -= resultFlow;
            return resultFlow;
        }
    
    }
    return 0;
}

int fordFulkerson(int source, int sink) {
    memset(flow, 0, sizeof(flow));
    int totalFlow = 0;

    while (true) {
        memset(visited, false, sizeof(visited));

        int pathFlow = dfs(source, sink, INF);
        if (pathFlow == 0) break; 

        totalFlow += pathFlow; 
    }
    return totalFlow;
}
```

### Time Complexity

The time complexity is
$$
O(E \cdot F)
$$
where $E$ is the number of edges in the graph and $F$ is the maximum flow.

## 7.2. Edmonds-Karp Algorithm

The Edmonds-Karp algorithm is an optimized version of the Ford-Fulkerson algorithm that uses BFS to find augmenting paths. It always selects the shortest augmenting path in terms of the number of edges from the source to the sink.

```cpp
int edmondsKarp(int source, int sink) {
    memset(flow, 0, sizeof(flow));
    int totalFlow = 0;

    while (true) { // BFS for finding an augmenting path 
        vi parent(MAX_V, -1);
        qi q;
        parent[source] = source;
        q.push(source);

        while (!q.empty() && parent[sink] == -1) {
            int here = q.front(); q.pop();
            for (int there = 0; there < V; ++there) {
                // Check remaining capacity and if 'there' has not been visited
                if (capacity[here][there] - flow[here][there] > 0 && parent[there] == -1) {
                    parent[there] = here;
                    q.push(there);
                    if (there == sink) break; // Stop early if we reached the sink
                }
            }
        }

        if (parent[sink] == -1) break; // No augmenting path found

        // Find the bottleneck capacity in the augmenting path
        int amount = INF;
        
        // Apply the flow along the path
        for (int p = sink; p != source; p = parent[p]) {
            flow[parent[p]][p] += amount;
            flow[p][parent[p]] -= amount;
        }

        totalFlow += amount;
    }
    return totalFlow;
}
```

## 7.3. Max-Flow Min-Cut Theorem

A key mathematical background for network flow problems is the Max-Flow Min-Cut Theorem.

- **Maximum Flow:** The maximum amount of flow that can be sent from the source to the sink.
- **Minimum Cut:** The minimum total capacity of edges that, when removed, disconnect the source from the sink. The cut separates the graph into two parts, with the source on one side and the sink on the other.

According to the theorem,
$$
\text{max flow from } s \text{ to } t = \text{min cut capacity of the network}
$$

## 7.4. Minimum Cost Maximum Flow

The Minimum Cost Maximum Flow (MCMF) algorithm finds the maximum flow in a flow network such that the total cost of sending the flow is minimized. It combines maximum flow and shortest path ideas to achieve optimal cost.

```cpp
pii mcmf(int source, int sink) {
    flow.assign(V, vi(V, 0));
    int totalFlow = 0, totalCost = 0;
    int mx = -1;

    while (true) {
        vi dist(V, INF), parent(V, -1);
        vector<bool> inQueue(V, false);
        queue<int> q;

        dist[source] = 0;
        parent[source] = source;
        q.push(source);
        inQueue[source] = true;

        while (!q.empty()) {
            auto from = q.front(); q.pop();
            inQueue[from] = false;

            for (int to : adj[from]) {
                int residual = capacity[from][to] - flow[from][to];
                int newCost = dist[from] + cost[from][to];
                if (residual <= 0 || dist[to] <= newCost) continue;

                dist[to] = newCost;
                parent[to] = from;
                if (!inQueue[to]) {
                    q.push(to);
                    inQueue[to] = true;
                }
            }
        }

        if (parent[sink] == -1) break;

        int curFlow = INF;
        for (int p = sink; p != source; p = parent[p]) {
            int par = parent[p];
            curFlow = min(curFlow, capacity[par][p] - flow[par][p]);
        }

        for (int p = sink; p != source; p = parent[p]) {
            int par = parent[p];
            flow[par][p] += curFlow;
            flow[p][par] -= curFlow;
            totalCost += cost[par][p];
        }

        mx = max(curFlow, mx);
        totalFlow += curFlow;
    }
    return {totalFlow, totalCost};
}
```

### Time Complexity Analysis

The time complexity depends on the shortest path algorithm used.

- **Using SPFA (Shortest Path Faster Algorithm):**
  - Each shortest path computation takes
    $$
    O(VE)
    $$
    in the worst case.
  - There can be at most
    $$
    O(F)
    $$
    augmenting paths, where $F$ is the total flow.
  - Total time complexity:
    $$
    O(F \cdot V \cdot E)
    $$

- **Using Dijkstra with a priority queue:**
  - Each shortest path computation takes
    $$
    O(E \log V)
    $$
  - Total time complexity:
    $$
    O(F \cdot E \log V)
    $$

## 7.5. Bipartite Matching

Bipartite matching is a fundamental algorithmic problem that aims to find the maximum matching in a bipartite graph, where a matching is a set of edges such that no two edges share a common vertex. This is widely used in applications such as job assignment, scheduling, and network design.

```cpp
bool dfs(int a) {
    if (visited[a]) return false;
    visited[a] = true;

    for (int b : adj[a]) { 
        // If vertex b is not currently matched
        if (bMatch[b] == -1) {
            aMatch[a] = b;
            bMatch[b] = a;
            return true;
        } 
        else {
            // If the current match of b can be re-matched
            int curPerson = bMatch[b];
            if (dfs(curPerson)) {
                aMatch[a] = b;
                bMatch[b] = a;
                return true;
            }
        }
    }
    return false;
}

int bipartiteMatch() {
    aMatch = vi(n + 1, -1); 
    bMatch = vi(n + 1, -1);  
    int size = 0;

    for (int start = 1; start < n + 1; ++start) {
        visited = vb(n + 1, false);  
        if (dfs(start)) ++size;
    }
    return size;
}
```

### Time Complexity Analysis

$$
O(n \cdot E)
$$

---

# 8. Linear Programming

Linear Programming (LP) is a mathematical optimization technique used to maximize or minimize a linear objective function subject to linear constraints.

## 8.1. Duality Theorem

The Duality Theorem is a fundamental concept in linear programming:

> Every linear programming problem (the **primal**) has a corresponding **dual** problem, and the optimal solutions to these two problems are closely related.

Consider the primal LP problem:
$$
\begin{aligned}
\text{Maximize } & c^T x \\
\text{subject to } & Ax \leq b, \\
& x \geq 0
\end{aligned}
$$

The corresponding dual problem is:
$$
\begin{aligned}
\text{Minimize } & b^T y \\
\text{subject to } & A^T y \geq c, \\
& y \geq 0
\end{aligned}
$$

## 8.2. LP for Max Flow Formulation

Linear programming can also be used to formulate the maximum flow problem. The objective is to maximize the flow from a source $s$ to a sink $t$ in a flow network.

$$
\begin{aligned}
\text{Maximize } & \sum_{(s,j)\in E} f_{sj} \\
\text{subject to } & \sum_{(i,j)\in E} f_{ij} - \sum_{(j,k)\in E} f_{jk} = 0, \quad \forall j \neq s,t, \\
& 0 \leq f_{ij} \leq c_{ij}, \quad \forall (i,j)\in E
\end{aligned}
$$

where:
- $f_{ij}$ is the flow on edge $(i,j)$
- $c_{ij}$ is the capacity of edge $(i,j)$