---
title: Computer Architecture
category: "COMPUTER SYSTEMS"
semester: 2023 S
---

# 1. Program Execution and Debugging Perspective

## Stack frames and function execution

A running program is not just “code being executed.” It is a sequence of function calls, each creating its own local execution context.

### Typical stack frame contents

- return address  
- saved base/frame pointer  
- local variables  
- spilled registers  
- function arguments not kept in registers  

A simplified view:

$$
\text{Stack Frame}_k \rightarrow
\begin{cases}
\text{return address} \\
\text{saved frame pointer} \\
\text{locals} \\
\text{temporary values}
\end{cases}
$$

Each call pushes a frame; each return pops one.

### Function calls

For a call such as

```cpp
int add(int a, int b) {
    int c = a + b;
    return c;
}
```

the machine-level story is about:

- passing arguments  
- saving the return location  
- creating local storage  
- restoring state on return  

### Register state

At any instant, execution depends on registers:

- general-purpose registers hold temporaries  
- special-purpose registers track control flow  
- stack/base pointers define frame layout  

Debugging low-level code often means answering:

- where is the program now?  
- what function is active?  
- what lives in registers?  
- what memory does each pointer refer to?  

---

# 2. Memory as an address space

## Virtual memory view

Programs operate in a **virtual** address space, not raw physical RAM cells.

A common layout:

$$
\text{Address Space} =
\begin{cases}
\text{Code / Text} \\
\text{Global / Static Data} \\
\text{Heap } \uparrow \\
\text{Unused Region} \\
\text{Stack } \downarrow
\end{cases}
$$

- **Code** — instructions  
- **Global/static** — long-lived data  
- **Heap** — dynamic allocation  
- **Stack** — active frames  

This ties together pointers, recursion, `malloc`, and segfaults.

## Byte-level inspection

Memory is byte-addressed. If

$$x = 0x12345678$$

then layout depends on **endianness**. On a little-endian machine you might see:

$$[\texttt{78}\ \texttt{56}\ \texttt{34}\ \texttt{12}]$$

So debuggers often show bytes differently from source-level `int` prints.

## Pointer arithmetic

A pointer is an address plus a **type**. If `int* p = arr`, then `p+1` is **not** the next byte:

$$\text{addr}(p+1) = \text{addr}(p) + \text{sizeof(int)}$$

**Example**

```cpp
int arr[4] = {10, 20, 30, 40};
int* p = arr;
```

- `p` → `arr[0]`  
- `p + 1` → `arr[1]`  
- `*(p + 2)` reads `30`  

This is where the C/C++ abstraction meets raw memory.

## Memory dump (byte grid)

```python {run}
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

base_addr = 0x1000
bytes_data = [
    0x78, 0x56, 0x34, 0x12,
    0x2A, 0x00, 0x00, 0x00,
    0x41, 0x42, 0x43, 0x00,
    0x10, 0x10, 0x00, 0x00,
]

fig, ax = plt.subplots(figsize=(12, 3.6), facecolor="#1a1a1a")
ax.set_facecolor("#1a1a1a")
cell_w, cell_h = 1.0, 1.0

for i, b in enumerate(bytes_data):
    rect = Rectangle(
        (i, 0), cell_w, cell_h,
        fill=True,
        facecolor="#252525",
        edgecolor="#c9a961",
        linewidth=1.2,
    )
    ax.add_patch(rect)
    ax.text(
        i + 0.5, 0.62, f"{b:02X}",
        ha="center", va="center",
        fontsize=12, color="#e8e8e8", fontfamily="monospace",
    )
    ax.text(
        i + 0.5, -0.28, f"+{i:02X}",
        ha="center", va="center",
        fontsize=8, color="#888",
    )

ax.text(
    len(bytes_data) / 2, -0.62,
    f"base ≈ 0x{base_addr:04X} (offsets shown)",
    ha="center", fontsize=9, color="#777",
)

groups = [
    (0, 4, "int x = 0x12345678\n(little-endian bytes)"),
    (4, 4, "int y = 42"),
    (8, 4, "char[4] = \"ABC\\0\""),
    (12, 4, "pointer-like word"),
]
for start, length, label in groups:
    ax.plot([start, start + length], [1.12, 1.12], color="#6b9ac4", linewidth=2)
    ax.text(start + length / 2, 1.32, label, ha="center", va="bottom", fontsize=9, color="#ccc")

ax.set_xlim(-0.5, len(bytes_data) + 0.5)
ax.set_ylim(-0.85, 1.85)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Byte-level memory dump (conceptual)", color="#ddd", fontsize=12)
ax.set_frame_on(False)
plt.tight_layout()
plt.show()
```

This highlights:

- addresses as indices  
- raw bytes  
- **interpretation** (grouping + type) layered on top  

---

# 3. Instruction set as an execution contract

## ISA: software ↔ hardware

The **ISA** specifies what the processor understands:

- instruction set  
- register names and roles  
- memory model  
- call/return behavior  

So the ISA is a contract among **compilers**, **assembly code**, and **hardware**.

## Register model (x86-64 sketch)

Typical roles:

- `rax` — return / accumulator  
- `rbx` — callee-saved GPR  
- `rcx`, `rdx`, `rsi`, `rdi` — args / temps (by convention)  
- `rsp` — stack pointer  
- `rbp` — frame pointer  
- `rip` — instruction pointer  

Abstract state:

$$
\text{Machine State} = (\text{Registers},\ \text{Memory},\ \text{Instruction Pointer})
$$

Each instruction updates that state.

## Calling conventions

They fix:

- argument passing  
- return value location  
- caller-saved vs callee-saved registers  
- stack alignment  

Without them, C/C++ and assembly could not link reliably.

## C ↔ assembly correspondence

```cpp
int add(int a, int b) {
    return a + b;
}
```

Typical x86-64 style:

```text
add:
    mov eax, edi
    add eax, esi
    ret
```

- `edi`, `esi` — first and second `int` args (SysV AMD64)  
- `eax` — `int` return  

Same function, ISA primitives.

### Detailed example

**C++**

```cpp
int f(int x, int y) {
    int t = x * 4;
    int z = t + y;
    return z - 3;
}
```

**Representative assembly**

```text
f:
    mov     eax, edi
    shl     eax, 2
    add     eax, esi
    sub     eax, 3
    ret
```

- `mov eax, edi` — `x` into working register  
- `shl eax, 2` — multiply by 4  
- `add eax, esi` — add `y`  
- `sub eax, 3` — subtract constant  
- `ret` — return to caller  

High-level ops **are** register/instruction sequences.

### C++ calling assembly

**C++**

```cpp
#include <iostream>
using namespace std;

extern "C" int asm_sum(int a, int b);

int main() {
    int result = asm_sum(7, 5);
    cout << result << endl;
    return 0;
}
```

**NASM-style callee**

```nasm
global asm_sum
section .text
asm_sum:
    mov eax, edi
    add eax, esi
    ret
```

**Why `extern "C"`** — C++ name mangling is disabled so the linker sees a stable symbol matching assembly.

Interoperability needs: **symbol names**, **calling convention**, **return register**.

---

# 4. Control flow and performance

## Branching

Control flow uses **jumps** and **conditional branches**.

High-level:

```cpp
if (x > y) {
    return x;
} else {
    return y;
}
```

Might look like:

```text
cmp edi, esi
jg  .L1
mov eax, esi
ret
.L1:
mov eax, edi
ret
```

- `cmp` sets flags  
- branches read flags  
- **misprediction** can cost a lot  

## Memory access patterns

CPU time is often dominated by **memory**, not ALU.

- **Sequential** access — cache-friendly, fast  
- **Irregular** (`index[i]`) — misses, slower  

Same asymptotics, very different wall-clock.

## Instruction cost

Costs differ by:

- ALU vs load/store  
- branch prediction hit vs miss  
- cache hit vs miss  
- scalar vs SIMD  
- pipeline stalls / dependencies  

## Example: same logic, different locality

```cpp
for (int i = 0; i < n; i++) {
    sum += arr[i];          // sequential
}

for (int i = 0; i < n; i++) {
    sum += arr[index[i]];   // irregular
}
```

The second loop often loses to the first on real hardware.

---

# 5. Measuring and reasoning about performance

## Profiling mindset

1. Observe  
2. Find the bottleneck  
3. Change one thing  
4. Re-measure  

## Bottlenecks

Often one dominates:

- hot loop  
- memory bandwidth  
- allocation churn  
- branchy code  
- redundant work  

## Evidence-based model

$$
\text{Total Runtime} \approx \sum (\text{frequency}) \times (\text{cost per op})
$$

Ask: **what runs most?** and **what is most expensive?**

## Dynamic programming in C++

Algorithm is high-level; **layout and access** decide speed.

**Table DP**

```cpp
#include <iostream>
#include <vector>
using namespace std;

long long fib_dp(int n) {
    if (n <= 1) return n;
    vector<long long> dp(n + 1);
    dp[0] = 0;
    dp[1] = 1;
    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}

int main() {
    cout << fib_dp(10) << endl;
    return 0;
}
```

- contiguous `dp`  
- sequential indices  
- **locality** friendly  

**O(1) space**

```cpp
long long fib_dp_optimized(int n) {
    if (n <= 1) return n;
    long long a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        long long c = a + b;
        a = b;
        b = c;
    }
    return b;
}
```

Less memory traffic, more register reuse — same math, different machine story.

### Loop ↔ assembly shape

**C++**

```cpp
for (int i = 2; i <= n; i++) {
    dp[i] = dp[i - 1] + dp[i - 2];
}
```

**Illustrative structure**

```text
mov     ecx, 2
.Lloop:
cmp     ecx, edi
jg      .Ldone
mov     rax, [rsi + 8*rcx - 8]
add     rax, [rsi + 8*rcx - 16]
mov     [rsi + 8*rcx], rax
add     ecx, 1
jmp     .Lloop
.Ldone:
```

Assume `edi = n`, `rsi =` base of `dp`, 8-byte `long long`: indexing becomes **explicit address arithmetic**; the recurrence is **load–add–store** in a loop.

## DP table as contiguous memory

```python {run}
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

dp = [0, 1, 1, 2, 3, 5, 8, 13]
base_addr = 0x2000
cell_bytes = 8

fig, ax = plt.subplots(figsize=(11, 3.8), facecolor="#1a1a1a")
ax.set_facecolor("#1a1a1a")

for i, val in enumerate(dp):
    ax.add_patch(
        Rectangle(
            (i, 0), 1, 1,
            fill=True,
            facecolor="#2a2a2a",
            edgecolor="#97c4a0",
            linewidth=1.2,
        )
    )
    ax.text(i + 0.5, 0.62, str(val), ha="center", va="center", fontsize=12, color="#eee")
    ax.text(i + 0.5, -0.2, f"dp[{i}]", ha="center", va="center", fontsize=9, color="#888")
    ax.text(
        i + 0.5, -0.45,
        f"0x{base_addr + i * cell_bytes:04X}",
        ha="center", va="center", fontsize=7, color="#666",
    )

for i in range(2, len(dp)):
    ax.annotate(
        "",
        xy=(i + 0.42, 1.02),
        xytext=(i - 1 + 0.58, 1.02),
        arrowprops=dict(arrowstyle="->", color="#6b9ac4", lw=1.4),
    )
    ax.annotate(
        "",
        xy=(i + 0.42, 1.18),
        xytext=(i - 2 + 0.58, 1.18),
        arrowprops=dict(arrowstyle="->", color="#c9a961", lw=1.1),
    )

ax.text(
    (len(dp) - 1) / 2, 1.45,
    "Each cell depends on two neighbors → sequential memory traffic",
    ha="center",
    fontsize=10,
    color="#bbb",
)

ax.set_xlim(-0.5, len(dp) + 0.5)
ax.set_ylim(-0.75, 1.75)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("DP table as contiguous layout (Fibonacci)", color="#ddd", fontsize=12)
ax.set_frame_on(False)
plt.tight_layout()
plt.show()
```

Links **recurrence** ↔ **array** ↔ **addresses** ↔ **locality**.

---

# 6. From source code to machine behavior

Architecture here means reading software **through** the machine.

The course connects:

- source  
- stack frames  
- registers  
- address arithmetic  
- assembly / ISA  
- memory patterns  
- measured performance  

Layered view:

$$
\text{Algorithm} \rightarrow \text{Source} \rightarrow \text{Assembly / ISA} \rightarrow \text{Execution} \rightarrow \text{Observed Performance}
$$

That stack explains not only **what** is computed, but **how the hardware is driven** to compute it.
