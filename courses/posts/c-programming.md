---
title: C Programming
category: CS
semester: 2023 S
---

# C Programming

This course uses C as a vehicle to understand how programs interact with
memory, the operating system, and the underlying machine.
Rather than focusing on syntax, the emphasis is on reasoning about
program behavior in a low-level, resource-constrained environment.

## Core Themes Overview

1. [C as a model & Compilation](#section-1)
2. [File I/O](#section-2)
3. [Variables & Struct](#section-3)
4. [Pointers](#section-4)
5. [Basic Data Structures](#section-5)
6. [Recursion](#section-6)
7. [Trees](#section-7)
8. [Rust & Ownership](#section-8)

---

<a id="section-1"></a>
## 1. C as a model & Compilation

C is often described as a low-level language, but what it actually provides is a low-level model of program execution. By avoiding built-in safety mechanisms, C exposes memory, control flow, and resource management directly, making costs and responsibilities explicit rather than implicit.

### Compilation
In C, compilation is a multi-stage process. Each source file is translated into an object file independently, and executables are produced later during linking, explaining the distinction between compilation and linking errors, the role of headers, and why object files form the basic unit of incremental builds.

### Object Files as an Explicit Boundary
Object files decouple translation from assembly. Each `.c` file can be compiled in isolation, producing machine code without requiring knowledge of the entire program.

``` bash
gcc -c file1.c
gcc -c file2.c
gcc file1.o file2.o -o program
```

### Making the Compilation Pipeline Visible
``` bash
gcc -E file.c    # preprocessing
gcc -S file.c    # generate assembly
gcc -c file.c    # generate object file
```

### Debugging and Optimization
``` bash
gcc -g -O0 file.c
gcc -O2 file.c
```
<a id="section-1"></a>

---
<a id="section-2"></a>

## 2. File I/O

- Files are accessed through `FILE*` streams, not filenames
- `fopen` can fail and must always be checked
- Closing a file does **not** invalidate the pointer automatically
- File streams maintain an internal **cursor**
- `fseek` / `rewind` manipulate the cursor explicitly
- EOF is detected **only after a read fails** (`feof` is not predictive)
- `fgets` is the safe default for reading text input
- File I/O errors are expected and must be handled explicitly

### Usage
```c
FILE *fp = fopen("data.txt", "r");
if (!fp) return 1;

char buf[128];
while (fgets(buf, sizeof(buf), fp)) {
    /* process input */
}

fclose(fp);
```

### Common Mistake
``` c
while (!feof(fp)) {  // incorrect
    fgets(buf, sizeof(buf), fp);
}
```
- `feof` becomes true only after an attempted read reaches EOF
- Always check the result of the read operation instead
<a id="section-2"></a>

---
<a id="section-3"></a>
## 3. Variables & Structs

### Variables: What Actually Matters

- A declaration introduces a name; a definition allocates storage
- Signed vs. unsigned affects **representation and overflow behavior**
- Storage class determines **lifetime**, **scope**, and **memory segment**

### Storage Classes (Mental Model)

- `auto`   : stack-allocated, block scope, short-lived
- `static` : data segment, lifetime of the program
- `extern` : shared global state across translation units
- `register` : optimization hint (may be ignored)

```c
static int counter;   // persists across function calls
extern int global_x;  // defined in another file
```

### Struct

``` c
struct TreeNode {
    int value;
    bool isInvalid;
    struct TreeNode* left;
    struct TreeNode* right;
};
```

On a 64-bit system, the memory layout of this struct is:

- Offset 0 : `value` : 4 bytes (`int`)
- Offset 4 : `isInvalid` : 1 byte (`bool`)
- Offset 5–7 : padding : 3 bytes (align pointer to 8 bytes)
- Offset 8 : `left` : 8 bytes (pointer)
- Offset 16 : `right` : 8 bytes (pointer)
- Total size : **24 bytes**

### Endianness

- Endianness defines **byte order in memory**
- `Big-endian`:  most significant byte stored at the lowest address
- `Little-endian`:  least significant byte stored at the lowest address
- Modern systems (e.g. x86) use **little-endian**

Example value: `0x12345678`

- Big-endian memory order: `12 34 56 78`
- Little-endian memory order: `78 56 34 12`
<a id="section-3"></a>

---

<a id="section-4"></a>
## 4. Pointers
A pointer stores the **address** of another variable

``` c
int x = 0;
int y = 0;
int *p = NULL;
int *q = NULL;

p = &y; // p -> address of y
q = &x; // q -> address of x  
*p = 2; // pointer value = 2  

printf("%d %d\n", x, y);  // 0, 2
```

### Double Pointer
``` c
int main() {
    int x = 0;
    int *p = NULL;
    int **pp = NULL;

    p = &x;   // p points to x
    pp = &p;  // pp points to p

    **pp = 42;  
    printf("x = %d\n", x); 

    return 0;
}
```

| Variable | Address  | Value    | Points To |
|----------|----------|----------|-----------|
| `x`      | `0x1000` | `42`     | –         |
| `p`      | `0x2000` | `0x1000` | `x`       |
| `pp`     | `0x3000` | `0x2000` | `p`       |

### Array Pointer

``` c
int main() {
    int arr[3] = {10, 20, 30};  
    int *p = arr;             

    for (int i = 0; i < 3; i++) {
        printf("arr[%d] = %d (via pointer: %d)\n", i, arr[i], *(p + i));
    }

    return 0;
}
```

### Function Pointer
A pointer that stores the address of a function. It allows functions to be passed as arguments to other functions, enabling callback mechanisms and dynamic function calls.

``` c
int add(int a, int b) {
    return a + b;
}

int calculate(int x, int y, int (*operation)(int, int)) {
    return operation(x, y);
}

int main() {
    int result = calculate(5, 3, add);  
    printf("Result: %d\n", result); // 8
    return 0;
}
```

### Dynamic Allocation
1. Malloc
``` c
// Allocate memory for 5 integers (The memory is uninitialized)
int *ptr = (int *)malloc(sizeof(int) * 5); 
if (ptr == NULL) {
    printf("Memory allocation failed!\n");
}
```
2. calloc
``` c
// Allocate memory for 5 integers (all initialized to 0)
int *ptr = (int *)calloc(5, sizeof(int));
if (ptr == NULL) printf("Memory allocation failed!\n");
```
3. realloc
``` c
// Resize memory to hold 10 integers
ptr = (int *)realloc(ptr, sizeof(int) * 10); 
if (ptr == NULL) {
    printf("Reallocation failed!\n");
}
```
4. free
``` c
free(ptr);
ptr = NULL; // Prevent dangling pointer issues
```
<a id="section-4"></a>

---

<a id="section-5"></a>
## 5. Basic Data Structures

### 1. Array

```c
#include <stdio.h>

int main() {
    int arr[3] = {1, 2, 3}; // Fixed-size, contiguous memory.
    printf("%d %d %d\n", arr[0], arr[1], arr[2]);
    return 0;
}
``` 
### 2. Linked List
``` c
struct Node {
    int value;
    struct Node *next;
};

int main() {
    struct Node a = {1, NULL};
    struct Node b = {2, NULL};

    a.next = &b;

    printf("%d %d\n", a.value, a.next->value); // 1, 2
    return 0;
}
```
### 3. Stack

```cpp
const int MAX_SIZE = 10001;
typedef int element;
typedef struct {
    element array[MAX_SIZE];
    int top;
} Stack;

void init(Stack *s) {s->top = -1;}

bool empty(Stack *s) {return s->top == -1;}

bool full(Stack *s) {return s->top == MAX_SIZE - 1;}

void push(Stack *s, element item) {
    if (full(s)) {
        cout << "Stack is saturated\n";
        return;
    }
    s->array[++s->top] = item;
}

void pop(Stack *s) {
    if (empty(s)) {
        cout << "Stack is empty\n";
        return;
    }
    s->top--;
}
```
### 4. Queue
```cpp
const int MAX_SIZE = 10001;
typedef int element;
typedef struct {
    int front;
    int rear;
    element array[MAX_SIZE];
} Queue;

void init(Queue *q) {
    q->front = -1;
    q->rear = -1;
}

bool empty(Queue *q) {
    return q->front == q->rear;
}

bool full(Queue *q) {return q->rear == MAX_SIZE - 1;}

void enqueue(Queue *q, element item) {
    if (full(q)) return;
    
    q->array[++q->rear] = item;
}

void dequeue(Queue *q) {
    if (empty(q)) return;
    cout << q->array[++q->front] << " was removed\n";
}
```
<a id="section-5"></a>

---

<a id="section-6"></a>
## 6. Recursion
A function calls itself

Must have a **base case**

Each call creates a new stack frame

### Example 1: Factorial
```cpp
#include <stdio.h>

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    printf("%d\n", factorial(5));
    return 0;
}
```
### Example 2: Binary Search

```cpp
int binarySearch(int arr[], int low, int high, int target) {
    if (low > high) return -1;

    int mid = low + (high - low) / 2;

    if (arr[mid] == target) return mid;
    if (arr[mid] > target) return binarySearch(arr, low, mid - 1, target);
    return binarySearch(arr, mid + 1, high, target);
}

int main() {
    int arr[] = {1, 3, 5, 7, 9};
    printf("%d\n", binarySearch(arr, 0, 4, 7)); // 3
    return 0;
}
```
<a id="section-6"></a>

---

<a id="section-7"></a>
## 7. Trees

```c
struct TreeNode {
    int value;
    struct TreeNode *left;
    struct TreeNode *right;
};
```

### Tree Traversal

``` c
void preorder(struct TreeNode *root) { // Root → Left → Right
    if (!root) return;
    printf("%d ", root->value);
    preorder(root->left);
    preorder(root->right);
}

void inorder(struct TreeNode *root) { // Left → Root → Right
    if (!root) return;
    inorder(root->left);
    printf("%d ", root->value);
    inorder(root->right);
}

void postorder(struct TreeNode *root) { // Left → Right → Root
    if (!root) return;
    postorder(root->left);
    postorder(root->right);
    printf("%d ", root->value);
}
```

### Tree Height
```c
int height(struct TreeNode *root) {
    if (!root) return 0;
    int l = height(root->left);
    int r = height(root->right);
    return 1 + (l > r ? l : r);
}
```

### Count Leaf Nodes
``` c
int countLeaves(struct TreeNode *root) {
    if (!root) return 0;
    if (!root->left && !root->right) return 1;
    return countLeaves(root->left) + countLeaves(root->right);
}
```


<a id="section-7"></a>

---

<a id="section-8"></a>
## 8. Rust & Ownership

### Core Idea
- Each value has **one owner**
- Ownership is **moved**, not copied
- Memory is freed automatically when the owner goes out of scope

---

### Ownership Move

```rust
fn main() {
    let x = String::from("hello");
    let y = x;
    println!("{}", y);
}
```

### Borrowing

```rust
fn print_len(s: &String) {
    println!("{}", s.len()); // 4
}

fn main() {
    let s = String::from("rust");
    print_len(&s);
    println!("{}", s); // rust
}
```

### Rust Compile (Terminal)
```bash
rustc main.rs
./main
```
or
```bash
cargo new project
cd project
cargo run
```