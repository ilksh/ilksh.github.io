template <typename T>

class priority_queue{
private:
    int cnt;    // number of elements in priority_queue
    T heap[MAX_N];
    
public:
    int size(){ return cnt; }
    
    priority_queue(){ cnt = 0;} // init
    
    void swap(int aIdx, int bIdx){
        int temp = heap[aIdx];
        heap[aIdx] = heap[bIdx];
        heap[bIdx] = temp;
    }
    
    void push(T item){
        heap[++cnt] = item;
        int cur = cnt;
    
        while(cur > 1){
            int par = cur >> 1;
        
            if(heap[cur] < heap[par]) break;
            
            swap(cur, par);
            cur = par;
        }
    }
    
    void pop(){ // move the lowest data and delete
        heap[1] = heap[cnt--];
        
        int cur = 1;
        
        while(cur <= cnt){
            int left =(cur << 1) <= cnt ? (cur << 1) : -1;
            int right = (cur << 1 | 1) <= cnt ? (cur << 1 | 1): -1;
            int child = cur; // the index of the largest value is child
            
            // cur node is leaf node
            if(left == -1) break;
            
            if(heap[cur] < heap[left]) child = left;
            
            if(right != -1 && heap[child] < heap[right]) child = right;
            
            if(child == left){
                swap(cur, left);
                cur = left;
            }
            
            else if(child == right){
                swap(cur, right);
                cur = right;
            }
            
            // cur == child
            else break;
        }
    }
    
    T top(){ return heap[1];}
    
    bool empty(){return cnt == 0;}
};

