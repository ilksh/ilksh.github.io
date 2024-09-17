#include<bits/stdc++.h>

using namespace std;

// djb2 hash
size_t djb2(const char* str) {
   // str = "blue"
   size_t hash = 5381;

   for(; *str; ++str) {
      hash = ((hash << 5) + hash) + *str;
   }

   return hash;
}

const int MAX_N = 1e4 + 4;
const int MAX_LEN = 5e2 + 5;

typedef struct Node{
   char str[MAX_LEN]; // key
   Node* next;
} Node;

// Memory Pool Technique
int node_count;
Node nodes[MAX_N];

Node* new_node(const char str[MAX_LEN]) {
   strcpy(nodes[node_count].str, str);
   nodes[node_count].next = NULL;

   return &nodes[node_count++];
}

class HashMap {
   static constexpr size_t TABLE_SIZE = 1 << 12;
   static constexpr size_t DIV = TABLE_SIZE - 1;
   // hash % DIV
   // hash & (TABLE_SIZE -1)

   Node hash_table[TABLE_SIZE];

public:
   HashMap() {}

   void init() {
      memset(hash_table, 0, sizeof(hash_table));
      node_count = 0;
   }

   void insert(const char str[MAX_LEN]) {
      Node* prev_node = get_prev_node(str);

      if(prev_node->next == NULL) {
         prev_node->next = new_node(str);
      }
   }

   void remove(const char str[MAX_LEN]) {
      Node* prev_node = get_prev_node(str);

      if(prev_node->next != NULL) {
         prev_node->next = prev_node->next->next;
      }
   }

   Node* get(const char str[MAX_LEN]) {
      return get_prev_node(str)->next;
   }

private:
   Node* get_prev_node(const char str[MAX_LEN]) {
      // Node* prev_ptr = &hash_table[djb2(str) % TABLE_SIZE];
      Node* prev_ptr = &hash_table[djb2(str) & DIV];

      while(prev_ptr->next != NULL && strcmp(prev_ptr->next->str, str) != 0) {
         prev_ptr = prev_ptr->next;
      }

      return prev_ptr;
   }
};

int main()
{
   ios_base::sync_with_stdio(0), cin.tie(0); 

   HashMap hash_map;
   int n, m, cnt = 0;
   char s[MAX_LEN];

   hash_map.init();

   cin >> n >> m;
   for(int i=0; i<n; ++i) {
      cin >> s;
      hash_map.insert(s);
   }

   for(int i=0; i<m; ++i) {
      cin >> s;
      if(hash_map.get(s) != NULL) {
         cnt++;
      }
   }

   cout << cnt << '\n';
   return 0;
}