---
title: Competitive Programming
category: CS
semester: 2023 S
---

# 1. Ad Hoc
특정 상황에서만 정답이 되고 일반화될 수 없는 해답을 찾는 알고리즘.  재사용되는 것이 거의 불가능. 

언제 사용하는가? 
1. 문제 조건이 정해진 알고리즘으로는 해결하기 애매할 때
2. 문제의 제약이 작고 구현 위주로 구성된 문제
3. 패턴이 눈에 보이지 않고 브루트포스도 가능해 보일 때

# 2. Dynamic Programming
memoization
```cpp
int fib(int x) {	
	int& result = cache[x];

	if (x == 0) return 0;

	if (x == 1) return 1;
		
	if (result != -1) return result;
	
	result = fib(x - 1) + fib(x - 2);
	return result;
}
```

```cpp
int cache[MAX_N]; // for topDown
int dp[MAX_N]; // for bottomUp

int topDown(int n) {
    // base case
    if (n == 1) return stairs[1];
    if (n == 2) return stairs[1] + stairs[2];
    
    int& ret = cache[n];  
    if (ret != -1) return ret;

    // (n - 3) -> (n - 1) -> n
    int routeOne = topDown(n - 3) + stairs[n - 1] + stairs[n];
    // (n - 2) -> n
    int routeTwo = topDown(n - 2) + stairs[n];

    ret = max(routeOne, routeTwo);
    return ret;
}

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

# 3. Tree
## 3.1. DFS & BFS

## 3.2. LCA

## 3.3. Fenwick Tree

## 3.4. Segment Tree

# 4. Convex Hull

# 5. Number Theory
- Prime Numbers
- GCD 
- Counting

# 6. Network Flow

# 7. Bitmasking