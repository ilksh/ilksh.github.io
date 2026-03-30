---
title: Basic Note I
category: PROBLEM SOLVING
date: 2026.03.29
readtime: 35 min
---

# 1. Usage of Stack & Queue in Python

## 1.1 Stack

- **Last In, First Out (LIFO)** — the most recently pushed element is popped first.

**In Python for PS**

- Use `list`.
- `append()` = push.
- `pop()` = pop from the end.
- Top element = `stack[-1]`.

**Important patterns**

- `stack = []`
- `stack.append(x)`
- `x = stack.pop()`
- `if stack:` — non-empty check
- `stack[-1]` — read top without popping

```python {run}
def demo_basic_stack() -> None:
    stack: list[int] = []

    stack.append(10)
    stack.append(20)
    stack.append(30)

    print(stack)  # [10, 20, 30]
    print(stack[-1])  # 30 (top)

    x = stack.pop()
    print(x)  # 30
    print(stack)  # [10, 20]

    if stack:
        print("not empty")
    else:
        print("empty")


def demo_stack_operations() -> None:
    stack: list[int] = []
    operations = [
        ("push", 5),
        ("push", 9),
        ("push", 2),
        ("pop", None),
        ("push", 7),
        ("top", None),
        ("pop", None),
    ]

    for op, value in operations:
        if op == "push":
            assert value is not None
            stack.append(value)
            print(f"push({value:>2}) -> {stack}")
        elif op == "pop":
            removed = stack.pop()
            print(f"pop() -> removed {removed}, stack = {stack}")
        elif op == "top":
            print(f"top() -> {stack[-1]}, stack = {stack}")


def is_valid_parentheses(s: str) -> bool:
    stack: list[str] = []
    match = {")": "(", "]": "[", "}": "{"}

    for ch in s:
        if ch in "([{":
            stack.append(ch)
        else:
            if not stack or stack[-1] != match[ch]:
                return False
            stack.pop()

    return len(stack) == 0


def main() -> None:
    print("=== Basic stack ===")
    demo_basic_stack()

    print("\n=== Stack simulation ===")
    demo_stack_operations()

    print("\n=== Valid parentheses ===")
    print(is_valid_parentheses("()[]{}"))  # True
    print(is_valid_parentheses("([)]"))  # False
    print(is_valid_parentheses("((()))"))  # True


if __name__ == "__main__":
    main()
```

## 1.2 Queue

- **First In, First Out (FIFO)** — the earliest enqueued element leaves first.

**In Python for PS**

- Use `collections.deque`.
- `append()` = enqueue at the right.
- `popleft()` = dequeue from the left.

**Important patterns**

- `from collections import deque`
- `q = deque()`
- `q.append(x)`
- `x = q.popleft()`
- `q[0]` — front, `q[-1]` — back

**Warning**

- Do not use `list.pop(0)` as a queue in PS; `deque.popleft()` is the standard choice.

```python {run}
from collections import deque


def demo_basic_queue() -> None:
    q: deque[int] = deque()

    q.append(10)
    q.append(20)
    q.append(30)

    print(q)  # deque([10, 20, 30])
    print(q[0])  # 10 (front)
    print(q[-1])  # 30 (back)

    x = q.popleft()
    print(x)  # 10
    print(q)  # deque([20, 30])


def demo_queue_operations() -> None:
    q: deque[int] = deque()
    operations = [
        ("push", 5),
        ("push", 9),
        ("push", 2),
        ("pop", None),
        ("push", 7),
        ("front", None),
        ("pop", None),
    ]

    for op, value in operations:
        if op == "push":
            assert value is not None
            q.append(value)
            print(f"append({value:>2}) -> {list(q)}")
        elif op == "pop":
            removed = q.popleft()
            print(f"popleft() -> removed {removed}, queue = {list(q)}")
        elif op == "front":
            print(f"front() -> {q[0]}, queue = {list(q)}")


def main() -> None:
    print("=== Basic queue ===")
    demo_basic_queue()

    print("\n=== Queue simulation ===")
    demo_queue_operations()


if __name__ == "__main__":
    main()
```

## 1.3 Hash table (dictionary)

- **Key → value** mapping with average-case fast lookup, insertion, and update.

**In Python for PS**

- Use `dict` for frequency counting, index memoization, and grouping.

**Important patterns**

- `d = {}`, `d[key] = value`
- `if key in d:`, `d.get(key, default)`, `del d[key]`
- `for k, v in d.items():`

**PS-heavy idioms**

- Counting: `freq[x] = freq.get(x, 0) + 1`
- First-index map, complement lookup (e.g. Two Sum)

```python {run}
def demo_basic_dict() -> None:
    d: dict[str, int] = {}

    d["apple"] = 3
    d["banana"] = 5

    print(d)
    print(d["apple"])
    print("banana" in d)
    print(d.get("orange", 0))

    for key, value in d.items():
        print(key, value)


def demo_frequency_count() -> None:
    arr = [3, 1, 3, 2, 1, 3, 4]
    freq: dict[int, int] = {}

    for x in arr:
        freq[x] = freq.get(x, 0) + 1
        print(f"read {x} -> freq = {freq}")


def two_sum(nums: list[int], target: int) -> list[int]:
    pos: dict[int, int] = {}

    for i, x in enumerate(nums):
        need = target - x
        if need in pos:
            return [pos[need], i]
        pos[x] = i

    return []


def simulate_two_sum(nums: list[int], target: int) -> None:
    pos: dict[int, int] = {}
    print(f"nums = {nums}, target = {target}\n")

    for i, x in enumerate(nums):
        need = target - x
        print(f"i={i}, x={x}, need={need}")
        print(f"  current map = {pos}")

        if need in pos:
            print(f"  found pair: indices [{pos[need]}, {i}]")
            return

        pos[x] = i
        print(f"  store {x}:{i} -> map = {pos}\n")

    print("  no pair found")


def demo_grouping() -> None:
    words = ["eat", "tea", "tan", "ate", "nat", "bat"]
    groups: dict[str, list[str]] = {}

    for word in words:
        key = "".join(sorted(word))
        groups.setdefault(key, []).append(word)

    print(groups)
    print(list(groups.values()))


def main() -> None:
    print("=== Basic dictionary ===")
    demo_basic_dict()

    print("\n=== Frequency count ===")
    demo_frequency_count()

    print("\n=== Two Sum ===")
    print(two_sum([2, 7, 11, 15], 9))
    print(two_sum([3, 2, 4], 6))

    print("\n=== Two Sum simulation ===")
    simulate_two_sum([2, 7, 11, 15], 9)

    print("\n=== Grouping (anagrams) ===")
    demo_grouping()


if __name__ == "__main__":
    main()
```

## 1.4 Recommended mental checklist for PS

- **Last inserted should come out first?** → stack
- **First inserted should come out first?** → queue
- **Fast lookup by key / value / frequency?** → dictionary
- **Repeated work at the front of a sequence?** → prefer `deque` over `list`
- **Only top push/pop?** → `list` is enough and stays simple

# 2. DP

## 2.1 Caching

**Main idea**

- If the exact same computation appears again later, do not recompute it.
- Compute once, store the result, and reuse it.

**In one line:** caching saves the answer to a previous computation, so the next identical call can load it instantly.

**Example**

- If `factorial(1000)` is called only once, caching does not change much.
- If the same factorial values are requested many times, repeated multiplication becomes unnecessary.

**When caching helps**

- Repeated queries, expensive computation, identical inputs appearing multiple times.

**When caching helps less**

- Almost every input is new.

```python {run}
import time


def factorial_plain(n: int) -> int:
    result = 1
    for x in range(2, n + 1):
        result *= x
    return result


def factorial_cached(n: int, cache: dict[int, int]) -> int:
    if n in cache:
        return cache[n]

    result = 1
    for x in range(2, n + 1):
        result *= x
    cache[n] = result
    return result


def cache_simulation() -> None:
    cache: dict[int, int] = {}
    queries = [5, 4, 7, 3, 10, 9]

    print("[Caching Simulation]")
    for n in queries:
        already_saved = n in cache
        value = factorial_cached(n, cache)
        action = "load from cache" if already_saved else "compute and store"
        print("query=factorial({}), action={}, result={}".format(n, action, value))
        print("cache_keys={}".format(sorted(cache.keys())))
    print()


def cache_timing_demo() -> None:
    queries = [3000, 2800, 3000, 2500, 2800, 3000, 2600, 2500, 3000] * 200

    start = time.perf_counter()
    for n in queries:
        factorial_plain(n)
    plain_time = time.perf_counter() - start

    cache: dict[int, int] = {}
    start = time.perf_counter()
    for n in queries:
        factorial_cached(n, cache)
    cached_time = time.perf_counter() - start

    print("[Caching Timing]")
    print("num_queries={}".format(len(queries)))
    print("num_unique_inputs={}".format(len(cache)))
    print("plain_time={:.6f} sec".format(plain_time))
    print("cached_time={:.6f} sec".format(cached_time))
    if cached_time > 0:
        print("speedup={:.2f}x".format(plain_time / cached_time))
    print()


def main() -> None:
    cache_simulation()
    cache_timing_demo()


if __name__ == "__main__":
    main()
```

## 2.2 Memoization

**Main idea**

- Memoization is caching inside recursion.
- If a recursive function revisits the same state, return the stored answer instead of solving it again.

**Why it matters**

- Recursive branching often creates the same subproblem many times.
- Memoization makes each state get solved once.

**Classic example: climbing stairs**

- To reach stair `n`, the last move came from `n - 1` or `n - 2`.
- Recurrence: `solve(n) = solve(n - 1) + solve(n - 2)`.
- **State:** `dp[x]` = number of ways to reach stair `x`; use `-1` for “not computed yet”.

**Practical recursive pattern**

1. Define `solve(state)`.
2. Write the base case.
3. Check whether the state is already stored.
4. Compute only when necessary.

```python {run}
import time


def climb_stairs_plain(n: int) -> int:
    if n <= 2:
        return n
    return climb_stairs_plain(n - 1) + climb_stairs_plain(n - 2)


def climb_stairs_memoized(n: int) -> int:
    dp = [-1] * (n + 1)

    def solve(x: int) -> int:
        if x <= 2:
            return x

        if dp[x] != -1:
            return dp[x]

        dp[x] = solve(x - 1) + solve(x - 2)
        return dp[x]

    return solve(n)


def memoization_simulation(n: int) -> int:
    dp = [-1] * (n + 1)

    def solve(x: int) -> int:
        print("enter solve({})".format(x))

        if x <= 2:
            print("base_case: solve({})={}".format(x, x))
            return x

        if dp[x] != -1:
            print("memo_hit: dp[{}]={}".format(x, dp[x]))
            return dp[x]

        print("compute: solve({}) + solve({})".format(x - 1, x - 2))
        dp[x] = solve(x - 1) + solve(x - 2)
        print("store: dp[{}]={}".format(x, dp[x]))
        return dp[x]

    answer = solve(n)
    print("final_dp={}".format(dp))
    print("answer={}".format(answer))
    print()
    return answer


def memoization_timing_demo() -> None:
    n = 35

    start = time.perf_counter()
    plain_answer = climb_stairs_plain(n)
    plain_time = time.perf_counter() - start

    start = time.perf_counter()
    memo_answer = climb_stairs_memoized(n)
    memo_time = time.perf_counter() - start

    print("[Memoization Timing]")
    print("n={}".format(n))
    print("plain_answer={}, plain_time={:.6f} sec".format(plain_answer, plain_time))
    print("memo_answer={}, memo_time={:.6f} sec".format(memo_answer, memo_time))
    if memo_time > 0:
        print("speedup={:.2f}x".format(plain_time / memo_time))
    print()


def main() -> None:
    memoization_simulation(6)
    memoization_timing_demo()


if __name__ == "__main__":
    main()
```

## 2.3 DP implementation guideline

**One-line goal:** DP solves a problem by building the answer from smaller subproblems.

**Most important question:** what is the state?

**Practical PS checklist**

1. **Define the state** — what minimum information uniquely describes one subproblem?
   - Examples: `dp[i]` (answer up to index `i`), `dp[r][c]` (cell `(r, c)`), `dp[i][k]` (index `i` with `k` operations left).
2. **Define the meaning in one sentence** — e.g. `dp[i]` = minimum cost to reach index `i`, or number of ways to reach stair `i`.
3. **Write the transition** — from which smaller states can the current state be built?
4. **Set the base case** — where does the process start?

**Common DP mistakes**

- Vague state definition, wrong base case, incorrect transition, wrong iteration order, mixing “exactly” and “at most”.

**Habit:** if you cannot define `dp[...]` clearly in one sentence, the DP is probably not ready yet.

```python {run}
from functools import lru_cache


def fibonacci_top_down(n: int) -> int:
    @lru_cache(maxsize=None)
    def solve(x: int) -> int:
        if x <= 1:
            return x
        return solve(x - 1) + solve(x - 2)

    return solve(n)


def fibonacci_bottom_up(n: int) -> int:
    if n <= 1:
        return n

    dp = [0] * (n + 1)
    dp[1] = 1

    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]


def fibonacci_optimized(n: int) -> int:
    if n <= 1:
        return n

    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    return prev1


def min_cost_climbing_stairs(cost: list[int]) -> int:
    n = len(cost)
    dp = [0] * (n + 1)

    for i in range(2, n + 1):
        one_step = dp[i - 1] + cost[i - 1]
        two_steps = dp[i - 2] + cost[i - 2]
        dp[i] = min(one_step, two_steps)

    return dp[n]


def guideline_simulation() -> None:
    n = 10
    cost = [10, 15, 20, 5, 8]

    print("[DP Guideline Simulation]")
    print("fibonacci_top_down({})={}".format(n, fibonacci_top_down(n)))
    print("fibonacci_bottom_up({})={}".format(n, fibonacci_bottom_up(n)))
    print("fibonacci_optimized({})={}".format(n, fibonacci_optimized(n)))
    print()

    print("cost={}".format(cost))
    print("min_cost_climbing_stairs={}".format(min_cost_climbing_stairs(cost)))
    print()

    dp = [0] * 8
    dp[1] = 1
    print("[Bottom-Up Build Demo]")
    print("i=1, dp={}".format(dp))
    for i in range(2, 8):
        dp[i] = dp[i - 1] + dp[i - 2]
        print("i={}, dp={}".format(i, dp))


def main() -> None:
    guideline_simulation()


if __name__ == "__main__":
    main()
```

# 3. Recursion

## 3.1 Backtracking template

**One-line idea:** backtracking is recursion that tries a choice, goes deeper, and undoes the choice if that path fails.

**Core flow:** choose → recurse → undo.

**Why it works**

- It systematically explores all valid possibilities.
- When one path becomes impossible, it returns to the previous state and tries the next option.

**Typical problems:** permutations, combinations, subsets, N-Queens, Sudoku, word search.

**Standard template**

1. Define the current state.
2. Define the stopping condition.
3. Iterate over all possible choices.
4. Check whether a choice is valid.
5. Apply the choice, recurse, then undo the choice.

**Easy way to recognize backtracking**

- “Find all”, “generate all”, “try every possible arrangement”, “fill the board”, “search until a valid configuration is found”.

```python {run}
def backtracking_dfs(
    nums: list[int],
    result: list[list[int]],
    path: list[int],
    used: list[bool],
) -> None:
    if len(path) == len(nums):
        result.append(path[:])
        return

    for i in range(len(nums)):
        if used[i]:
            continue

        path.append(nums[i])
        used[i] = True

        backtracking_dfs(nums, result, path, used)

        used[i] = False
        path.pop()


def backtracking_template(nums: list[int]) -> list[list[int]]:
    result: list[list[int]] = []
    path: list[int] = []
    used = [False] * len(nums)

    backtracking_dfs(nums, result, path, used)
    return result


def backtracking_simulation_dfs(
    nums: list[int],
    path: list[int],
    used: list[bool],
    depth: int,
) -> None:
    print("enter depth={}, path={}, used={}".format(depth, path, used))

    if len(path) == len(nums):
        print("complete path found: {}".format(path))
        print()
        return

    for i in range(len(nums)):
        if used[i]:
            continue

        print("choose nums[{}]={}".format(i, nums[i]))
        path.append(nums[i])
        used[i] = True

        backtracking_simulation_dfs(nums, path, used, depth + 1)

        used[i] = False
        removed = path.pop()
        print("undo choice {}, path becomes {}".format(removed, path))

    print("return from depth={}, path={}".format(depth, path))
    print()


def backtracking_simulation(nums: list[int]) -> None:
    path: list[int] = []
    used = [False] * len(nums)
    backtracking_simulation_dfs(nums, path, used, 0)


def main() -> None:
    nums = [1, 2, 3]
    print("all permutations={}".format(backtracking_template(nums)))
    print()
    backtracking_simulation(nums)


if __name__ == "__main__":
    main()
```

## 3.2 Sudoku as a backtracking example

- In each empty cell there are multiple candidate digits; some choices are valid now but may fail later, so we try, recurse, and undo if needed.

**State:** the current Sudoku board.

**Choice:** which digit to place in the current empty cell.

**Validity check:** the digit must not already appear in the same row, column, or 3×3 box.

**Success:** no empty cells left → solved.

**Failure:** no digit works for the current empty cell → return `False`.

**Structure:** find an empty cell → try digits 1–9 → if valid, place and recurse → if recursion fails, reset the cell to `.`.

```python {run}
def print_board(board: list[list[str]]) -> None:
    for row in board:
        print(" ".join(row))
    print()


def is_valid(board: list[list[str]], row: int, col: int, ch: str) -> bool:
    for i in range(9):
        if board[i][col] == ch:
            return False
        if board[row][i] == ch:
            return False

        box_row = 3 * (row // 3) + i // 3
        box_col = 3 * (col // 3) + i % 3
        if board[box_row][box_col] == ch:
            return False

    return True


def solve_sudoku(board: list[list[str]]) -> bool:
    for i in range(9):
        for j in range(9):
            if board[i][j] == ".":
                for ch in "123456789":
                    if is_valid(board, i, j, ch):
                        board[i][j] = ch

                        if solve_sudoku(board):
                            return True

                        board[i][j] = "."

                return False

    return True


def solve_sudoku_verbose(board: list[list[str]]) -> bool:
    for i in range(9):
        for j in range(9):
            if board[i][j] == ".":
                print("first empty cell found at ({}, {})".format(i, j))

                for ch in "123456789":
                    if is_valid(board, i, j, ch):
                        print("place {} at ({}, {})".format(ch, i, j))
                        board[i][j] = ch

                        if solve_sudoku(board):
                            print("choice {} at ({}, {}) leads to a solution".format(ch, i, j))
                            return True

                        board[i][j] = "."
                        print("undo {} at ({}, {})".format(ch, i, j))

                print("no valid digit works at ({}, {})".format(i, j))
                return False

    return True


def main() -> None:
    board = [
        ["5", "3", ".", ".", "7", ".", ".", ".", "."],
        ["6", ".", ".", "1", "9", "5", ".", ".", "."],
        [".", "9", "8", ".", ".", ".", ".", "6", "."],
        ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
        ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
        ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
        [".", "6", ".", ".", ".", ".", "2", "8", "."],
        [".", ".", ".", "4", "1", "9", ".", ".", "5"],
        [".", ".", ".", ".", "8", ".", ".", "7", "9"],
    ]

    print("[Initial Board]")
    print_board(board)

    solved = solve_sudoku_verbose(board)
    print("solved={}".format(solved))
    print()

    print("[Solved Board]")
    print_board(board)


if __name__ == "__main__":
    main()
```

## 3.3 Relationship between backtracking and stack

**One-line idea:** backtracking is usually implemented with recursion, and recursion runs on the call stack.

**Interpretation**

- Every recursive call corresponds to pushing a frame onto the stack.
- When that call finishes or fails, the frame is popped.

**Simple picture**

- Choose a value → go deeper → that deeper call is a new stack frame.
- If the deeper path fails, control returns to the previous frame and tries another choice.

**Why it “feels like a stack”**

- The latest decision is the first one to be undone — that is LIFO (last in, first out).

**Comparison**

- **Recursion:** often cleaner for tree search and backtracking templates.
- **Manual stack:** more control; can help when recursion depth is a concern.

# 4. Graph

## 4.1 BFS & DFS

### Basic template

**One-line idea:** BFS explores level by level; DFS explores one path deeply before backtracking.

**When to use BFS**

- Shortest path in an **unweighted** graph, level-order traversal, minimum number of moves, processes that spread layer by layer.

**When to use DFS**

- Full exploration, connected components, cycle detection, backtracking-style graph traversal.

**Core difference**

- BFS uses a **queue**; DFS uses **recursion** or an explicit **stack**.

**Representation (PS)**

- Adjacency list: `graph[u] = [v1, v2, ...]`.

**Visited**

- Track visited unless the problem allows revisiting; otherwise cycles can loop forever.

```python {run}
from collections import deque


def bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    visited = {start}
    queue = deque([start])
    order: list[int] = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for nxt in graph[node]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)

    return order


def dfs_recursive(graph: dict[int, list[int]], start: int) -> list[int]:
    visited: set[int] = set()
    order: list[int] = []

    def dfs(node: int) -> None:
        visited.add(node)
        order.append(node)

        for nxt in graph[node]:
            if nxt not in visited:
                dfs(nxt)

    dfs(start)
    return order


def dfs_iterative(graph: dict[int, list[int]], start: int) -> list[int]:
    visited: set[int] = set()
    stack = [start]
    order: list[int] = []

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        order.append(node)

        for nxt in reversed(graph[node]):
            if nxt not in visited:
                stack.append(nxt)

    return order


def traversal_simulation(graph: dict[int, list[int]], start: int) -> None:
    print("[BFS Simulation]")
    visited = {start}
    queue = deque([start])

    while queue:
        print("queue={}".format(list(queue)))
        node = queue.popleft()
        print("pop={}".format(node))

        for nxt in graph[node]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
                print("visit={}, queue_after_push={}".format(nxt, list(queue)))
        print()

    print("[DFS Iterative Simulation]")
    visited = set()
    stack = [start]

    while stack:
        print("stack={}".format(stack))
        node = stack.pop()
        print("pop={}".format(node))

        if node in visited:
            print("{} already visited".format(node))
            print()
            continue

        visited.add(node)
        print("visit={}".format(node))

        for nxt in reversed(graph[node]):
            if nxt not in visited:
                stack.append(nxt)
                print("push={}".format(nxt))
        print()


def main() -> None:
    graph = {
        1: [2, 3],
        2: [4, 5],
        3: [6],
        4: [5],
        5: [6],
        6: [],
    }

    print("bfs_order={}".format(bfs(graph, 1)))
    print("dfs_recursive_order={}".format(dfs_recursive(graph, 1)))
    print("dfs_iterative_order={}".format(dfs_iterative(graph, 1)))
    print()

    traversal_simulation(graph, 1)


if __name__ == "__main__":
    main()
```

### Implementation guideline

1. **Build the graph** — usually an adjacency list; directed: `graph[u].append(v)`; undirected: also `graph[v].append(u)`.
2. **Choose traversal** — BFS for unweighted shortest path / levels; DFS for exhaustive search / recursive structure.
3. **Visited** — `set()` for general labels, or a boolean list for indices `0..n-1` or `1..n`.
4. **Start** — single source if connected or given; **iterate all nodes** if you need components or the graph may be disconnected.
5. **When to mark visited**
   - **BFS:** often when **enqueueing** (avoids duplicate queue entries).
   - **DFS:** often when **entering** the node.

**Common mistakes:** ignoring disconnected components, marking visited too late, mixing directed vs undirected edges.

## 4.2 Topological sorting

### Basic template

**One-line idea:** a topological order lists nodes so every directed edge goes from earlier to later.

**When it applies:** DAG (directed acyclic graph).

**Typical signals:** course schedule, prerequisites, task dependencies, build order.

**Kahn’s idea (in-degree)**

- Compute in-degree for each node.
- Enqueue all nodes with in-degree 0.
- Pop a node, append to answer, decrease in-degree of neighbors; enqueue any neighbor that reaches 0.

**Cycle check:** if the result does not contain all nodes, a cycle exists.

**Why in-degree:** it counts unfinished prerequisites; in-degree 0 means “ready now”.

```python {run}
from collections import deque


def topological_sort(n: int, edges: list[list[int]]) -> list[int]:
    graph = [[] for _ in range(n + 1)]
    indegree = [0] * (n + 1)

    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1

    queue: deque[int] = deque()
    for node in range(1, n + 1):
        if indegree[node] == 0:
            queue.append(node)

    order: list[int] = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for nxt in graph[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(order) != n:
        return []

    return order


def topological_sort_simulation(n: int, edges: list[list[int]]) -> None:
    graph = [[] for _ in range(n + 1)]
    indegree = [0] * (n + 1)

    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1

    queue: deque[int] = deque()
    for node in range(1, n + 1):
        if indegree[node] == 0:
            queue.append(node)

    order: list[int] = []

    print("[Topological Sort Simulation]")
    print("initial_indegree={}".format(indegree[1:]))
    print("initial_queue={}".format(list(queue)))
    print()

    while queue:
        node = queue.popleft()
        order.append(node)

        print("pop={}, order={}".format(node, order))

        for nxt in graph[node]:
            indegree[nxt] -= 1
            print("decrease indegree[{}] to {}".format(nxt, indegree[nxt]))
            if indegree[nxt] == 0:
                queue.append(nxt)
                print("push={} because indegree is now 0".format(nxt))

        print("queue={}".format(list(queue)))
        print()

    if len(order) == n:
        print("topological_order={}".format(order))
    else:
        print("cycle_detected, no valid topological ordering")


def main() -> None:
    n = 6
    edges = [[1, 3], [2, 3], [3, 4], [3, 5], [4, 6], [5, 6]]

    print("topological_order={}".format(topological_sort(n, edges)))
    print()

    topological_sort_simulation(n, edges)


if __name__ == "__main__":
    main()
```

### Implementation guideline

1. **Confirm directed** dependency structure (not for arbitrary undirected graphs).
2. **Build** adjacency list and in-degree array.
3. **Enqueue** every node with in-degree 0.
4. **Process queue:** pop → append to answer → relax outgoing edges → enqueue new zeros.
5. **Validate:** `len(order) == n` else cycle.

**Common mistakes:** using topo sort on undirected graphs, forgetting isolated nodes, wrong node range `1..n`, skipping cycle check.

**In-degree vs out-degree**

- **In-degree:** incoming edges / remaining prerequisites pointing at this node.
- **Out-degree:** outgoing edges — how many nodes this node points to.
