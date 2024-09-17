#include<bits/stdc++.h>
using namespace std;

const int MAX_N = 5e4 + 4;
const int MAX_B = 16;       // 2^x + 2^(x-1) + 2^(x-2) >= 50000

vector<int> adj[MAX_N];
int n, m, ST[MAX_N][MAX_B], dep[MAX_N];

void makeTree(int cur, int par) {
    
    for(int child : adj[cur]) {
        // don't search node that is adjacent but parent-related
        if(child == par) continue;
        
        ST[child][0] = cur;
        dep[child] = dep[cur] + 1;
        
        // Configure trees with adjacent nodes in dfs manner
        makeTree(child, cur);
    }
}

int lca(int u, int v) {
    // desired situation: dep[u] > dep[v]
    // control of situation as desired
    if(dep[u] < dep[v]) return lca(v, u);

    int diff = dep[u] - dep[v];

    // move node as high as possible
    for(int i=MAX_B-1; diff && i>=0; --i) {
        if((1 << i) <= diff) {
            u = ST[u][i];
            diff -= (1 << i);
        }
    }
    
    if(u == v) return u;

    // the height of vertext u and that of vertex v are same
    for(int i = MAX_B - 1; i >= 0; --i) {
        // two nodes belong to different component
        if(ST[u][i] != ST[v][i]) {
            u = ST[u][i];
            v = ST[v][i];
        }
    }

    return ST[u][0];
}

void adj_check(int node1, int node2)
{
    adj[node1].push_back(node2);
    adj[node2].push_back(node1);
}

void fill_adj()
{
    adj_check(1, 2); adj_check(1 , 3);
    
    adj_check(2, 4); adj_check(2, 5);
    
    adj_check(3, 6); adj_check(3, 7);
    
    adj_check(4, 8); adj_check(4, 9);
    
    adj_check(5 ,10); adj_check(5, 11); adj_check(5, 12);
    
    adj_check(7, 13); adj_check(7, 14);
    
    adj_check(8, 15);
    
    adj_check(9, 16); adj_check(9, 17);
    
    adj_check(12, 18);
    
    adj_check(18, 19);
    
    adj_check(19, 20);
    
    adj_check(14, 21);
}

int main()
{
    ios_base::sync_with_stdio(0);
    cin.tie(0);

    fill_adj();
    dep[1] = 0;
    makeTree(1, -1);

    for(int idx = 1; idx < MAX_B; ++idx) {
        for(int cur = 1; cur <= 21; ++cur)
        {
            // (2^idx)th parent of node 'cur'
            // = (2^idx-1)th parent of ( node cur's (2^idx-1)th parent)
            ST[cur][idx] = ST[ST[cur][idx-1]][idx-1];
        }
    }
    
    // lca(node1, node2) = find the lowest common ancestor of node1 and node2
    
    printf("Lowest Common Ancestor of %d and %d is %d\n", 4, 19, lca(4,19));
    printf("Lowest Common Ancestor of %d and %d is %d\n", 15, 7, lca(15,7));
    printf("Lowest Common Ancestor of %d and %d is %d\n", 10, 5, lca(10,5));
    printf("Lowest Common Ancestor of %d and %d is %d\n", 19, 21,lca(19,21));
    printf("Lowest Common Ancestor of %d and %d is %d\n", 15, 17,lca(15,17));
    
    return 0;
}

/*
 Lowest Common Ancestor of 4 and 19 is 2
 Lowest Common Ancestor of 15 and 7 is 1
 Lowest Common Ancestor of 10 and 5 is 5
 Lowest Common Ancestor of 19 and 21 is 1
 Lowest Common Ancestor of 15 and 17 is 4
 */