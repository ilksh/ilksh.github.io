---
title: Competitive Programming
category: CS
semester: 2023 S
---

# 1. Ad Hoc

### Algorithm

* Ad hoc algorithms are problem-specific solutions.
* They usually do not generalize well.
* They are often useful when constraints are small or when direct implementation is enough.
* They are common when no standard pattern is obvious.

### When to Use

* The problem does not match a standard algorithmic pattern.
* The constraints are small enough for direct case analysis or brute force.
* The solution depends heavily on special observations.

---

# 2. Dynamic Programming

## 2.1 Matrix Chain Multiplication

```cpp {fold}
using pii = pair<int, int>;

int n;
int dp[501][501];
vector<pii> matrices;

int solve(int left, int right) {
    if (left == right) return 0;

    int& ret = dp[left][right];
    if (ret != 0x7f7f7f7f) return ret;

    for (int mid = left; mid < right; mid++) {
        auto left = matrices[left].first;
        auto mid = matrices[mid].second;
        auto right = matrices[right].second;
        ret = min(
            ret,
            solve(left, mid) + solve(mid + 1, right) + left * mid * right
        );
    }
    return ret;
}
```

### Code Notes

* `dp[left][right]` stores the minimum multiplication cost for matrices in the interval `[left, right]`.
* The recursion splits the chain at every possible `mid`.
* Memoization prevents repeated computation.
* The multiplication cost of combining two subchains is computed from matrix dimensions.

### Algorithm Notes

* This is a classic interval DP.
* The recurrence tries every possible final split point.
* Time complexity is typically $O(n^3)$.
* Space complexity is $O(n^2)$.

---

## 2.2 Coin Change Counting

```cpp {fold}
int coins[MAX_N];
int dp[MAX_N][MAX_M];
int n, target;

int solve(int idx, int amount) {
    if (amount == 0) return 1;
    if (idx > n || amount < 0) return 0;

    int& ret = dp[idx][amount];
    if (ret != -1) return ret;

    ret = solve(idx + 1, amount);
    if (amount - coins[idx] >= 0) {
        ret += solve(idx, amount - coins[idx]);
    }
    return ret;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> n >> target;
    for (int i = 1; i <= n; i++) cin >> coins[i];

    memset(dp, -1, sizeof(dp));
    cout << solve(1, target) << '\n';
    return 0;
}
```

### Code Notes

* `solve(idx, amount)` counts the number of ways to make `amount` using coins from index `idx`.
* The transition branches into skipping the current coin or using it.
* Memoization stores repeated states.
* The base case `amount == 0` means one valid construction has been found.

### Algorithm Notes

* This is top-down DP with memoization.
* It counts combinations, not permutations.
* Time complexity is about $O(n \cdot target)$.
* Space complexity is $O(n \cdot target)$.

---
# 3. Tree and Graph Algorithms

## 3.1 DFS and BFS

```cpp {fold}
#include <bits/stdc++.h>
using namespace std;

int n, m;
bool graph[1001][1001];
bool visitedDfs[1001];
bool visitedBfs[1001];
queue<int> q;

void dfs(int node) {
    visitedDfs[node] = true;
    cout << node << ' ';
    for (int next = 1; next <= n; next++) {
        if (graph[node][next] && !visitedDfs[next]) {
            dfs(next);
        }
    }
}

void bfs(int start) {
    q.push(start);
    visitedBfs[start] = true;

    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        cout << cur << ' ';

        for (int next = 1; next <= n; next++) {
            if (graph[cur][next] && !visitedBfs[next]) {
                visitedBfs[next] = true;
                q.push(next);
            }
        }
    }
}
```

### Code Notes

* `graph[u][v]` is an adjacency matrix representation.
* DFS explores recursively as deep as possible before backtracking.
* BFS explores level by level using a queue.
* Separate visited arrays are used for DFS and BFS.

### Algorithm Notes

* DFS is useful for connectivity, components, recursion-based traversal, and tree DP.
* BFS is useful for shortest paths in unweighted graphs.
* With an adjacency matrix, both traversals take $O(n^2)$.
* With an adjacency list, both become $O(V + E)$.

---

## 3.2 Lowest Common Ancestor (Binary Lifting)

```cpp {fold}
#include <bits/stdc++.h>
using namespace std;

int nodeCount;
int parent[100001][18];
int depthArr[100001];
int maxHeight;
vector<int> adj[100001];

void buildTree(int par, int now, int depth) {
    parent[now][0] = par;
    depthArr[now] = depth;

    for (int next : adj[now]) {
        if (next == par) continue;
        buildTree(now, next, depth + 1);
    }
}

int lca(int a, int b) {
    if (depthArr[a] < depthArr[b]) swap(a, b);

    int diff = depthArr[a] - depthArr[b];
    for (int i = 0; diff > 0; i++) {
        if (diff & 1) a = parent[a][i];
        diff >>= 1;
    }

    if (a == b) return a;

    for (int k = maxHeight; k >= 0; k--) {
        if (parent[a][k] != 0 && parent[a][k] != parent[b][k]) {
            a = parent[a][k];
            b = parent[b][k];
        }
    }

    return parent[a][0];
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> nodeCount;
    for (int i = 0; i < nodeCount - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    buildTree(0, 1, 0);

    int temp = nodeCount;
    while (temp > 1) {
        temp >>= 1;
        maxHeight++;
    }

    for (int k = 1; k <= maxHeight; k++) {
        for (int i = 1; i <= nodeCount; i++) {
            if (parent[i][k - 1] != 0) {
                parent[i][k] = parent[parent[i][k - 1]][k - 1];
            }
        }
    }

    int q;
    cin >> q;
    while (q--) {
        int a, b;
        cin >> a >> b;
        cout << lca(a, b) << '\n';
    }

    return 0;
}
```

### Code Notes

* `parent[node][k]` stores the $2^k$-th ancestor of `node`.
* `depthArr[node]` stores the node depth.
* The first phase builds direct parents and depths.
* The second phase fills the binary lifting table.

### Algorithm Notes

* LCA answers ancestor queries on trees efficiently.
* Each query first aligns depths, then lifts both nodes upward.
* Preprocessing takes $O(N \log N)$.
* Each query takes $O(\log N)$.

---

## 3.3 Fenwick Tree

```cpp {fold}
#include <bits/stdc++.h>
using namespace std;

struct FenwickTree {
    vector<int> tree;

    FenwickTree(int n) : tree(n + 1, 0) {}

    int sum(int pos) {
        pos++;
        int ret = 0;
        while (pos > 0) {
            ret += tree[pos];
            pos &= (pos - 1);
        }
        return ret;
    }

    void add(int pos, int val) {
        pos++;
        while (pos < (int)tree.size()) {
            tree[pos] += val;
            pos += (pos & -pos);
        }
    }
};

int main() {
    FenwickTree ft(10);
    ft.add(2, 5);
    ft.add(4, 3);
    cout << ft.sum(4) << '\n';
    return 0;
}
```

### Code Notes

* The structure supports prefix sums and point updates.
* `sum(pos)` computes the prefix sum from index `0` to `pos`.
* `add(pos, val)` updates a single position.
* Internal indexing is 1-based.

### Algorithm Notes

* Fenwick trees are efficient for dynamic prefix sums.
* Update and query are both $O(\log N)$.
* Space complexity is $O(N)$.
* They are simpler than segment trees for sum-based problems.

---
## 3.4 Segment Tree

```cpp {fold}
#include <bits/stdc++.h>
using namespace std;

using ll = long long;

const int TREE_SIZE = 1 << 17;
const int INF = 0x3f3f3f3f;

ll maxTree[TREE_SIZE << 1];
ll minTree[TREE_SIZE << 1];
int n, m;

ll queryMax(int left, int right) {
    ll ret = 0;
    for (left += TREE_SIZE, right += TREE_SIZE; left <= right; left >>= 1, right >>= 1) {
        if (left & 1) ret = max(ret, maxTree[left++]);
        if (!(right & 1)) ret = max(ret, maxTree[right--]);
    }
    return ret;
}

ll queryMin(int left, int right) {
    ll ret = INF;
    for (left += TREE_SIZE, right += TREE_SIZE; left <= right; left >>= 1, right >>= 1) {
        if (left & 1) ret = min(ret, minTree[left++]);
        if (!(right & 1)) ret = min(ret, minTree[right--]);
    }
    return ret;
}

void update(int idx, ll value) {
    idx += TREE_SIZE;
    maxTree[idx] = value;
    minTree[idx] = value;

    while (idx >>= 1) {
        maxTree[idx] = max(maxTree[idx << 1], maxTree[idx << 1 | 1]);
        minTree[idx] = min(minTree[idx << 1], minTree[idx << 1 | 1]);
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    memset(maxTree, 0, sizeof(maxTree));
    memset(minTree, 0x3f, sizeof(minTree));

    cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        cin >> maxTree[TREE_SIZE + i];
        minTree[TREE_SIZE + i] = maxTree[TREE_SIZE + i];
    }

    for (int i = TREE_SIZE - 1; i > 0; i--) {
        maxTree[i] = max(maxTree[i << 1], maxTree[i << 1 | 1]);
        minTree[i] = min(minTree[i << 1], minTree[i << 1 | 1]);
    }

    while (m--) {
        int l, r;
        cin >> l >> r;
        cout << queryMin(l, r) << ' ' << queryMax(l, r) << '\n';
    }

    return 0;
}
```

### Code Notes

* Two segment trees are maintained in parallel: one for minimum and one for maximum.
* Queries are processed iteratively.
* Updates propagate values upward to the root.
* Leaves start at `TREE_SIZE`.

### Algorithm Notes

* Segment trees support fast range queries and updates.
* Query time is $O(\log N)$.
* Update time is $O(\log N)$.
* Space complexity is $O(N)$, usually with a constant-factor expansion.

---

# 4. Convex Hull

```cpp {fold}
#include <bits/stdc++.h>
using namespace std;

using ll = long long;

struct Point {
    ll x, y;

    bool operator<(const Point& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }
};

int ccw(const Point& a, const Point& b, const Point& c) {
    ll value = a.x * b.y + b.x * c.y + c.x * a.y
             - (a.x * c.y + b.x * a.y + c.x * b.y);

    if (value > 0) return 1;
    if (value < 0) return -1;
    return 0;
}

ll dist2(const Point& a, const Point& b) {
    ll dx = a.x - b.x;
    ll dy = a.y - b.y;
    return dx * dx + dy * dy;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;
    vector<Point> points(n);

    for (int i = 0; i < n; i++) {
        cin >> points[i].x >> points[i].y;
    }

    sort(points.begin(), points.end());

    sort(points.begin() + 1, points.end(), [&](const Point& p1, const Point& p2) {
        int turn = ccw(points[0], p1, p2);
        if (turn != 0) return turn > 0;
        return dist2(points[0], p1) < dist2(points[0], p2);
    });

    vector<Point> hull;
    for (const Point& p : points) {
        while (hull.size() >= 2 &&
               ccw(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0) {
            hull.pop_back();
        }
        hull.push_back(p);
    }

    cout << hull.size() << '\n';
    return 0;
}
```

### Code Notes

* Points are first sorted lexicographically.
* The first point becomes the pivot.
* The remaining points are sorted by polar angle relative to the pivot.
* The hull is maintained with a stack-like vector.

### Algorithm Notes

* This is Graham scan.
* The `ccw` test determines whether a turn is left, right, or collinear.
* Time complexity is dominated by sorting: $O(N \log N)$.
* Space complexity is $O(N)$.

---

# 5. Number Theory

## 5.1 Binomial Coefficient Modulo Prime

```cpp {fold}
#include <bits/stdc++.h>
using namespace std;

using ll = long long;

const ll MOD = 1000000007;
ll factorial[4000001];

ll modPow(ll base, ll exp) {
    if (exp == 0) return 1;
    if (exp == 1) return base % MOD;

    if (exp % 2 == 0) {
        ll half = modPow(base, exp / 2);
        return (half * half) % MOD;
    } else {
        ll part = modPow(base, exp - 1);
        return (part * base) % MOD;
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int q;
    cin >> q;

    factorial[0] = 1;
    for (int i = 1; i < 4000001; i++) {
        factorial[i] = factorial[i - 1] * i % MOD;
    }

    while (q--) {
        int n, k;
        cin >> n >> k;

        ll numerator = factorial[n];
        ll denominator = factorial[k] * factorial[n - k] % MOD;
        ll inverseDenominator = modPow(denominator, MOD - 2);

        cout << numerator * inverseDenominator % MOD << '\n';
    }

    return 0;
}
```

### Code Notes

* `factorial[i]` precomputes factorial values modulo `MOD`.
* Fermat’s little theorem is used to compute the modular inverse.
* Each query evaluates $\binom{n}{k}$ efficiently.
* `modPow` performs fast modular exponentiation.

### Algorithm Notes

* This method works when `MOD` is prime.
* Preprocessing takes $O(N)$.
* Each query takes $O(\log \mathrm{MOD})$.
* It is standard for repeated combination queries under a prime modulus.

---
## 5.2 Miller–Rabin + Pollard’s Rho

```cpp {fold}
#include <bits/stdc++.h>
using namespace std;

using ll = long long;
using ull = unsigned long long;
using i128 = __int128_t;

vector<ull> factors;

ull modPow(ull a, ull b, ull mod) {
    ull result = 1;
    a %= mod;
    while (b > 0) {
        if (b & 1) result = (ull)((i128)result * a % mod);
        a = (ull)((i128)a * a % mod);
        b >>= 1;
    }
    return result;
}

bool millerRabin(ull n, ull a) {
    if (a % n == 0) return true;

    ull d = n - 1;
    while ((d & 1) == 0) d >>= 1;

    ull x = modPow(a, d, n);
    if (x == 1 || x == n - 1) return true;

    while (d != n - 1) {
        x = (ull)((i128)x * x % n);
        d <<= 1;
        if (x == n - 1) return true;
        if (x == 1) return false;
    }
    return false;
}

bool isPrime(ull n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;

    vector<ull> bases = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (ull a : bases) {
        if (n <= a) break;
        if (!millerRabin(n, a)) return false;
    }
    return true;
}

ull gcdValue(ull a, ull b) {
    return b ? gcdValue(b, a % b) : a;
}

ull pollardRho(ull n) {
    if (n % 2 == 0) return 2;

    ull x = 2, y = 2, c = 1, d = 1;
    auto f = [&](ull v) {
        return (ull)((i128)v * v % n + c) % n;
    };

    while (d == 1) {
        x = f(x);
        y = f(f(y));
        d = gcdValue(x > y ? x - y : y - x, n);

        if (d == n) {
            x = rand() % (n - 2) + 2;
            y = x;
            c = rand() % 10 + 1;
            d = 1;
        }
    }
    return d;
}

void factorize(ull n) {
    if (n == 1) return;
    if (isPrime(n)) {
        factors.push_back(n);
        return;
    }

    ull divisor = pollardRho(n);
    factorize(divisor);
    factorize(n / divisor);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    ull n;
    cin >> n;

    factorize(n);
    sort(factors.begin(), factors.end());

    for (ull x : factors) {
        cout << x << '\n';
    }
    return 0;
}
```

### Code Notes

* `modPow` uses `__int128` to avoid overflow during modular multiplication.
* `isPrime` uses deterministic Miller–Rabin bases for 64-bit integers.
* `pollardRho` finds a nontrivial divisor probabilistically.
* `factorize` recursively splits the number until only primes remain.

### Algorithm Notes

* Miller–Rabin is a fast primality test.
* Pollard’s Rho is effective for integer factorization in practice.
* Together they are standard for large integer factorization.
* This approach is much faster than trial division for large inputs.

---

# 6. Network Flow

## 6.1 Maximum Flow (Edmonds–Karp Style)

```cpp {fold}
#include <bits/stdc++.h>
using namespace std;

const int MAX_V = 53;
const int INF = 1e9;

int capacity[MAX_V][MAX_V];
int flowArr[MAX_V][MAX_V];
int V = MAX_V;

int maxFlow(int source, int sink) {
    memset(flowArr, 0, sizeof(flowArr));
    int totalFlow = 0;

    while (true) {
        vector<int> parent(V, -1);
        queue<int> q;

        parent[source] = source;
        q.push(source);

        while (!q.empty() && parent[sink] == -1) {
            int cur = q.front();
            q.pop();

            for (int next = 0; next < V; next++) {
                int residual = capacity[cur][next] - flowArr[cur][next];
                if (residual <= 0 || parent[next] != -1) continue;
                parent[next] = cur;
                q.push(next);
            }
        }

        if (parent[sink] == -1) break;

        int amount = INF;
        for (int p = sink; p != source; p = parent[p]) {
            int par = parent[p];
            amount = min(amount, capacity[par][p] - flowArr[par][p]);
        }

        for (int p = sink; p != source; p = parent[p]) {
            int par = parent[p];
            flowArr[par][p] += amount;
            flowArr[p][par] -= amount;
        }

        totalFlow += amount;
    }

    return totalFlow;
}

int convert(char c) {
    if ('A' <= c && c <= 'Z') return c - 'A' + 1;
    if ('a' <= c && c <= 'z') return c - 'a' + 27;
    return 0;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int e;
    cin >> e;

    memset(capacity, 0, sizeof(capacity));

    for (int i = 0; i < e; i++) {
        char a, b;
        int w;
        cin >> a >> b >> w;

        int u = convert(a);
        int v = convert(b);

        capacity[u][v] += w;
        capacity[v][u] += w;
    }

    int source = convert('A');
    int sink = convert('Z');

    cout << maxFlow(source, sink) << '\n';
    return 0;
}
```

### Code Notes

* `capacity[u][v]` stores edge capacity.
* `flowArr[u][v]` stores current flow.
* BFS finds an augmenting path in the residual graph.
* Each augmentation increases total flow until no augmenting path exists.

### Algorithm Notes

* This is the Edmonds–Karp pattern.
* The key concept is the residual graph.
* BFS ensures the shortest augmenting path in terms of edges.
* Time complexity is $O(VE^2)$ in the standard adjacency-list analysis.

---

## 6.2 Min-Cost Max-Flow

Function-only snippet (wire `adj`, `capacity`, `cost`, and zero `flowArr` in your `main`, then call `minCostMaxFlow(source, sink)`):

```cpp {fold}
#include <bits/stdc++.h>
using namespace std;

const int MAX = 505;
const int INF = 1e9;

vector<int> adj[MAX];
int capacity[MAX][MAX];
int flowArr[MAX][MAX];
int cost[MAX][MAX];

int minCostMaxFlow(int source, int sink) {
    int totalCost = 0;

    while (true) {
        vector<int> parent(MAX, -1);
        vector<int> dist(MAX, INF);
        vector<bool> inQueue(MAX, false);
        queue<int> q;

        dist[source] = 0;
        q.push(source);
        inQueue[source] = true;

        while (!q.empty()) {
            int cur = q.front();
            q.pop();
            inQueue[cur] = false;

            for (int next : adj[cur]) {
                int residual = capacity[cur][next] - flowArr[cur][next];
                int nextCost = dist[cur] + cost[cur][next];

                if (residual <= 0 || nextCost >= dist[next]) continue;

                dist[next] = nextCost;
                parent[next] = cur;

                if (!inQueue[next]) {
                    q.push(next);
                    inQueue[next] = true;
                }
            }
        }

        if (parent[sink] == -1) break;

        int currentFlow = INF;
        for (int p = sink; p != source; p = parent[p]) {
            int par = parent[p];
            currentFlow = min(currentFlow, capacity[par][p] - flowArr[par][p]);
        }

        for (int p = sink; p != source; p = parent[p]) {
            int par = parent[p];
            flowArr[par][p] += currentFlow;
            flowArr[p][par] -= currentFlow;
            totalCost += currentFlow * cost[par][p];
        }
    }

    return totalCost;
}
```

### Code Notes

* The shortest augmenting path is chosen with respect to edge cost.
* Residual capacity determines whether more flow can pass.
* `dist` tracks the current best cost to each node.
* Reverse edges are required for flow cancellation.

### Algorithm Notes

* This solves flow optimization with both capacity and cost.
* SPFA-style shortest path is commonly used in competitive programming.
* The algorithm augments until no more source-to-sink path exists.
* It is especially useful for assignment, transportation, and matching with weights.

---

## 6.3 Bipartite Matching via Flow Reduction

```cpp {fold}
#include <bits/stdc++.h>
using namespace std;

using vi = vector<int>;
using vvi = vector<vi>;

const int INF = 0x3f3f3f3f;

int V;
vvi capacity;
vvi flowArr;
vvi adj;

int solve(int source, int sink) {
    flowArr.assign(V, vi(V, 0));
    int totalFlow = 0;

    while (true) {
        vi parent(V, -1);
        parent[source] = source;

        queue<int> q;
        q.push(source);

        while (!q.empty() && parent[sink] == -1) {
            int from = q.front();
            q.pop();

            for (int to : adj[from]) {
                int residual = capacity[from][to] - flowArr[from][to];
                if (residual <= 0) continue;
                if (parent[to] != -1) continue;
                parent[to] = from;
                q.push(to);
            }
        }

        if (parent[sink] == -1) break;

        int currentFlow = INF;
        for (int p = sink; p != source; p = parent[p]) {
            int par = parent[p];
            currentFlow = min(currentFlow, capacity[par][p] - flowArr[par][p]);
        }

        for (int p = sink; p != source; p = parent[p]) {
            int par = parent[p];
            flowArr[par][p] += currentFlow;
            flowArr[p][par] -= currentFlow;
        }

        totalFlow += currentFlow;
    }

    return totalFlow;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N, M;
    cin >> N >> M;

    V = N + M + 2;
    int source = 0;
    int sink = N + M + 1;

    capacity.assign(V, vi(V, 0));
    adj.assign(V, vi());

    for (int i = 1; i <= N; i++) {
        adj[source].push_back(i);
        adj[i].push_back(source);
        capacity[source][i] = 1;

        int z;
        cin >> z;
        while (z--) {
            int stall;
            cin >> stall;
            adj[i].push_back(N + stall);
            adj[N + stall].push_back(i);
            capacity[i][N + stall] = 1;
        }
    }

    for (int i = 1; i <= M; i++) {
        adj[N + i].push_back(sink);
        adj[sink].push_back(N + i);
        capacity[N + i][sink] = 1;
    }

    cout << solve(source, sink) << '\n';
    return 0;
}
```

### Code Notes

* The bipartite graph is converted into a flow network.
* Source connects to left-side nodes with capacity `1`.
* Right-side nodes connect to sink with capacity `1`.
* A maximum flow equals a maximum matching.

### Algorithm Notes

* This is a standard reduction from bipartite matching to max flow.
* Unit capacities enforce one-to-one matches.
* The solution is correct because every feasible matching corresponds to an integral flow.
* It is a clean modeling technique for assignment-like problems.

---
## Python animations

Press **Run** to build a looping GIF of the traversal or flow buildup. First run may install packages (matplotlib, pillow, networkx).

### DFS / BFS step-by-step

```python {run}
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import base64
import os

graph = {
    1: [2, 3],
    2: [1, 4, 5],
    3: [1, 6],
    4: [2],
    5: [2, 6],
    6: [3, 5, 7],
    7: [6],
}

pos = {
    1: (0, 1),
    2: (-1.1, 0.2),
    3: (1.1, 0.2),
    4: (-1.8, -0.9),
    5: (-0.4, -0.9),
    6: (1.1, -0.9),
    7: (1.8, -1.8),
}

edges = []
for u, nbrs in graph.items():
    for v in sorted(nbrs):
        if u < v:
            edges.append((u, v))


def dfs_order(g, start):
    visited = set()
    order = []

    def dfs(u):
        visited.add(u)
        order.append(u)
        for v in sorted(g[u]):
            if v not in visited:
                dfs(v)

    dfs(start)
    return order


def bfs_order(g, start):
    visited = {start}
    q = deque([start])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in sorted(g[u]):
            if v not in visited:
                visited.add(v)
                q.append(v)
    return order


dfs_o = dfs_order(graph, 1)
bfs_o = bfs_order(graph, 1)
max_f = max(len(dfs_o), len(bfs_o))


def draw_ax(ax, active, title, subtitle):
    ax.clear()
    ax.set_facecolor('#1a1a1a')
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color='#444', lw=2, zorder=1)
    for n, (x, y) in pos.items():
        if n in active:
            face, edge = '#c9a961', '#e8d5a3'
        else:
            face, edge = '#2a2a2a', '#444444'
        ax.scatter([x], [y], s=1100, c=face, edgecolors=edge, linewidths=2, zorder=3)
        ax.text(x, y, str(n), ha='center', va='center', color='white', fontsize=11, fontweight='bold', zorder=4)
    ax.set_title(title + '\n' + subtitle, color='#cccccc', fontsize=11)
    ax.axis('off')
    lim = 2.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-2.2, 1.4)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#1a1a1a')


def update(fid):
    i_dfs = min(fid + 1, len(dfs_o))
    i_bfs = min(fid + 1, len(bfs_o))
    active_d = set(dfs_o[:i_dfs])
    active_b = set(bfs_o[:i_bfs])
    draw_ax(
        ax1,
        active_d,
        'DFS (depth-first)',
        'visited: ' + ' → '.join(map(str, dfs_o[:i_dfs])),
    )
    draw_ax(
        ax2,
        active_b,
        'BFS (breadth-first)',
        'visited: ' + ' → '.join(map(str, bfs_o[:i_bfs])),
    )


anim = animation.FuncAnimation(fig, update, frames=max_f, interval=650, repeat=True)
_gif_path = '_cp_anim_dfs_bfs.gif'
anim.save(_gif_path, writer='pillow', fps=1.4)
plt.close('all')
with open(_gif_path, 'rb') as f:
    _ANIM_GIF = base64.b64encode(f.read()).decode()
try:
    os.remove(_gif_path)
except OSError:
    pass
print('Frames:', max_f)
```

### Simulation Notes

* Left panel grows the DFS visit set in discovery order.
* Right panel grows the BFS frontier layer by layer.
* The GIF loops so you can compare how the two frontiers expand.

---

### Maximum flow buildup

```python {run}
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import base64
import os

edges = [
    ("S", "A", 10),
    ("S", "B", 8),
    ("A", "B", 5),
    ("A", "C", 5),
    ("B", "C", 10),
    ("B", "D", 7),
    ("C", "T", 10),
    ("D", "T", 10),
]

G = nx.DiGraph()
for u, v, c in edges:
    G.add_edge(u, v, capacity=c)

flow_val, flow_dict = nx.maximum_flow(G, "S", "T")
pos = {
    "S": (0, 1),
    "A": (1, 2),
    "B": (1, 0),
    "C": (2, 2),
    "D": (2, 0),
    "T": (3, 1),
}

nframes = 14
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a1a')


def update(t):
    ax.clear()
    ax.set_facecolor('#1a1a1a')
    frac = min(1.0, (t + 1) / nframes)
    for u, v, d in G.edges(data=True):
        cap = d['capacity']
        used = flow_dict[u][v] * frac
        width = 1.4 + 5.0 * (used / cap if cap else 0)
        color = '#555555' if used < 1e-6 else '#c9a961'
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            ax=ax,
            edge_color=color,
            width=width,
            arrows=True,
            arrowsize=16,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.05',
        )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color='#2a2a2a',
        edgecolors='#c9a961',
        linewidths=2,
        node_size=2000,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_color='white', font_size=11)
    for u, v, d in G.edges(data=True):
        cap = d['capacity']
        used = flow_dict[u][v] * frac
        xm = (pos[u][0] + pos[v][0]) / 2
        ym = (pos[u][1] + pos[v][1]) / 2
        ax.text(
            xm,
            ym + 0.07,
            f'{used:.1f}/{cap}',
            ha='center',
            fontsize=9,
            color='#aaaaaa',
            zorder=5,
        )
    ax.set_title(
        f'Maximum flow animation (target {flow_val})  step {t + 1}/{nframes}',
        color='#cccccc',
        fontsize=12,
    )
    ax.axis('off')


anim = animation.FuncAnimation(fig, update, frames=nframes, interval=480, repeat=True)
_gif_path = '_cp_anim_maxflow.gif'
anim.save(_gif_path, writer='pillow', fps=2)
plt.close('all')
with open(_gif_path, 'rb') as f:
    _ANIM_GIF = base64.b64encode(f.read()).decode()
try:
    os.remove(_gif_path)
except OSError:
    pass
print('Max flow value:', flow_val)
```

### Simulation Notes

* Edge labels show `flow/capacity` as the animation approaches the max-flow assignment from NetworkX.
* Thicker gold edges carry more flow; grey edges are still at zero in that frame.
* This mirrors the augmenting-path intuition without re-implementing the solver in the browser.

---

## Clean summary format you can reuse

Each algorithm can follow this shape going forward.

### Template

* **Code**
* **Code Notes**

  * what each state / array / function means
  * what the transition or update does
  * what data structure is being used
* **Algorithm Notes**

  * core idea
  * time complexity
  * space complexity
  * when to use it

---
