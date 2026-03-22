---
title: Applied Quantum Computing
category: QUANTUM
semester: 2024 S
---

# Applied Quantum Computing

Quantum algorithms are built on **discrete state spaces** and **linear algebra**. The same habit of encoding configurations in bits shows up classically—below is a standard **bitmask** exercise: counting non-overlapping four-seat **families** per airplane row when some middle seats are reserved.

---

## Cinema row model

- **10 seats** per row (labels `1 … 10`). Only seats **`2–9`** matter for families; **`1` and `10`** are aisles and do not block the three four-seat patterns.
- A **family** needs one of these contiguous blocks (LeetCode-style “cinema seat allocation”):

| Block   | Seats   |
|---------|---------|
| Left    | 2–5     |
| Middle  | 4–7     |
| Right   | 6–9     |

- Each row can host **at most two** families if the middle is empty; reservations in `2–9` shrink that count. We store occupied seats in `2–9` as a **bitmask** per row: bit `col` set means seat `col` is taken.

### Seat map and three family regions

```python {run}
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(11, 2.6), facecolor="#1a1a1a")
ax.set_facecolor("#1a1a1a")

left_seats = {2, 3, 4, 5}
mid_seats = {4, 5, 6, 7}
right_seats = {6, 7, 8, 9}
region_style = [
    ("left", left_seats, "#6b9ac4", 0.42),
    ("mid", mid_seats, "#c9a961", 0.35),
    ("right", right_seats, "#97c4a0", 0.42),
]

for name, seats_set, hex_c, a in region_style:
    for s in seats_set:
        ax.add_patch(
            Rectangle(
                (s - 1 - 0.45, 0.15), 0.9, 0.7,
                facecolor=hex_c, alpha=a, edgecolor="none", zorder=1,
            )
        )

for s in range(1, 11):
    ax.add_patch(
        Rectangle(
            (s - 1 - 0.48, 0.12),
            0.96,
            0.76,
            fill=False,
            edgecolor="#666",
            linewidth=1.2,
            zorder=2,
        )
    )
    col = "#888" if s in (1, 10) else "#ddd"
    ax.text(s - 1, 0.48, str(s), ha="center", va="center", fontsize=12, color=col, fontweight="bold", zorder=3)

ax.text(0, -0.35, "1", ha="center", fontsize=9, color="#666")
ax.text(9, -0.35, "10", ha="center", fontsize=9, color="#666")
ax.text(4.5, -0.55, "aisle seats (ignored for family masks)", ha="center", fontsize=9, color="#777")

legend_el = [
    mpatches.Patch(facecolor="#6b9ac4", alpha=0.5, edgecolor="#6b9ac4", label="Left 2–5"),
    mpatches.Patch(facecolor="#c9a961", alpha=0.45, edgecolor="#c9a961", label="Middle 4–7"),
    mpatches.Patch(facecolor="#97c4a0", alpha=0.5, edgecolor="#97c4a0", label="Right 6–9"),
]
ax.legend(handles=legend_el, loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=3, frameon=False, labelcolor="#ccc")

ax.set_xlim(-0.8, 9.8)
ax.set_ylim(-0.7, 1.05)
ax.axis("off")
ax.set_title("One row: three overlapping four-seat family regions (seats 2–9)", color="#e0e0e0", fontsize=11, pad=28)
plt.tight_layout()
plt.show()
```

---

## Bitmasks per row

For each row, build `mask` with `mask |= (1 << col)` for every reserved `col` in `2…9`. Test blocks:

- `leftBlock   = (1<<2)|(1<<3)|(1<<4)|(1<<5)`
- `middleBlock = (1<<4)|(1<<5)|(1<<6)|(1<<7)`
- `rightBlock  = (1<<6)|(1<<7)|(1<<8)|(1<<9)`

Start from **`2n`** families (two per row). For each row that has at least one reservation in `2–9`:

- If **both** left and right regions are free → still **2** families (`continue`).
- Else if **any one** of left / middle / right fits → **1** family (`maxGroups -= 1`).
- Else → **0** families (`maxGroups -= 2`).

Rows with **no** reserved seats in `2–9` are absent from the map and keep the default **2** families.

### Example: reserved seats and block feasibility

```python {run}
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def blocks_free(mask):
    left = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
    mid = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
    right = (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9)
    return (
        (mask & left) == 0,
        (mask & mid) == 0,
        (mask & right) == 0,
    )

reserved_cols = [2, 3, 8]
mask = 0
for c in reserved_cols:
    mask |= 1 << c
cl, cm, cr = blocks_free(mask)

fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), facecolor="#1a1a1a", height_ratios=[1.1, 0.9])
for ax in axes:
    ax.set_facecolor("#1a1a1a")

ax = axes[0]
for s in range(1, 11):
    taken = s in reserved_cols
    face = "#5c3030" if taken else "#2a2a2a"
    edge = "#c9a961" if taken else "#555"
    ax.add_patch(Rectangle((s - 1 - 0.45, 0.2), 0.9, 0.65, facecolor=face, edgecolor=edge, linewidth=1.5))
    ax.text(s - 1, 0.52, "R" if taken else str(s), ha="center", va="center", fontsize=11, color="#eee")
ax.set_xlim(-0.7, 9.7)
ax.set_ylim(0, 1.05)
ax.axis("off")
bitstr = "".join("1" if mask & (1 << c) else "0" for c in range(2, 10))
ax.set_title(
    f"Sample row — reserved at {reserved_cols}  |  bits for seats 2→9: {bitstr}",
    color="#ddd",
    fontsize=10,
)

ax = axes[1]
ax.axis("off")
status = [
    ("Left 2–5", cl, "#6b9ac4"),
    ("Middle 4–7", cm, "#c9a961"),
    ("Right 6–9", cr, "#97c4a0"),
]
y = 0.75
for name, ok, color in status:
    ax.text(0.05, y, f"{name}:  {'FITS' if ok else 'blocked'}", fontsize=11, color=color if ok else "#888", transform=ax.transAxes, family="monospace")
    y -= 0.35
ax.text(
    0.05,
    0.05,
    "Here: left & right blocks hit reserved seats; only middle 4–7 is free → one family.",
    fontsize=10,
    color="#aaa",
    transform=ax.transAxes,
)

plt.tight_layout()
plt.show()
```

---

## C++ reference implementation

```cpp
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    int maxNumberOfFamilies(int n, vector<vector<int>>& reservedSeats) {
        unordered_map<int, int> rowMasks;

        for (const auto& seat : reservedSeats) {
            int row = seat[0];
            int col = seat[1];
            if (col >= 2 && col <= 9) {
                rowMasks[row] |= (1 << col);
            }
        }

        int maxGroups = 2 * n;

        int leftBlock = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5);
        int middleBlock = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7);
        int rightBlock = (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9);

        for (const auto& [row, mask] : rowMasks) {
            bool canFitLeft = (mask & leftBlock) == 0;
            bool canFitRight = (mask & rightBlock) == 0;
            bool canFitMiddle = (mask & middleBlock) == 0;

            if (canFitLeft && canFitRight) {
                continue;
            } else if (canFitLeft || canFitRight || canFitMiddle) {
                maxGroups -= 1;
            } else {
                maxGroups -= 2;
            }
        }

        return maxGroups;
    }
};
```

---

## Topics (course outline)

- Quantum circuits and gate notation  
- Frameworks (e.g. Qiskit-style programming patterns)  
- Variational and optimization-style quantum algorithms  
- NISQ hardware constraints and noise  

The seating example is **classical**; it rehearses **bit-encoded state** and **combinatorial constraints** that reappear when you label basis states and feasible configurations in quantum information.
