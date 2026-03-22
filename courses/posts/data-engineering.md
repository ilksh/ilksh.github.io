---
title: Data Engineering
category: "DATA / PROGRAMMING"
semester: 2023 S
---

# 1. Python Foundations for Data Work

## Core Python Building Blocks

### Data types

```python
x = 3               # int
y = 2.5             # float
name = "Alice"      # str
flag = True         # bool
values = [1, 2, 3]  # list
point = (4, 5)      # tuple
```

- Basic types underpin cleaning, transformation, and analysis.
- Lists, strings, numbers, and booleans show up everywhere.

### Loops and conditionals

```python
scores = [72, 91, 64, 88]

for s in scores:
    if s >= 70:
        print("pass")
    else:
        print("fail")
```

- Loops repeat the same operation over data.
- Conditionals branch on values.

### Functions

```python
def normalize(x, mean, std):
    return (x - mean) / std
```

- Functions bundle logic for reuse—essential when preprocessing repeats.

### File I/O

```python
with open("data.txt", "r") as f:
    text = f.read()
```

- File I/O links code to real sources; workflows usually start with read and end with write.

---

# 2. Classes and Object-Oriented Programming

## Organizing logic with classes

```python
class Dataset:
    def __init__(self, name, rows):
        self.name = name
        self.rows = rows

    def summary(self):
        return f"{self.name}: {len(self.rows)} rows"
```

- OOP groups **data + behavior** in one place.
- Useful when a dataset or pipeline has several related operations.

## Why it matters

- Clearer organization and reusable pipeline pieces.
- Easier maintenance as projects grow.
- In practice, OOP is less about theory and more about **structuring repeated workflows**.

---

# 3. Data Engineering

## The goal

- Turn **raw, messy** data into an **analysis-ready** dataset.
- Typical stages: **clean** → **combine** → **inspect quality** (often with quick visuals).

## Cleaning data

Typical issues:

- Missing values  
- Duplicated rows  
- Inconsistent categories  
- Wrong dtypes  
- Outliers or impossible values  

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Bob", None],
    "age": ["24", "31", "31", "19"],
    "city": ["Seoul", "seoul", "Seoul ", "Busan"],
})

df = df.drop_duplicates()
df["age"] = pd.to_numeric(df["age"])
df["city"] = df["city"].str.strip().str.title()
df = df.dropna(subset=["name"])
```

- Cleaning is not cosmetic—small inconsistencies break grouping, merges, and stats later.

## Combining data

**Vertical (stack):**

```python
df_all = pd.concat([df_2022, df_2023], axis=0)
```

**Horizontal (join on keys):**

```python
merged = sales.merge(customers, on="customer_id", how="left")
```

- `concat` stacks; `merge` aligns rows on keys—getting alignment right is most of the job.

## Visualizing data quality

Worth a quick look before heavy analysis:

- Missingness  
- Category imbalance  
- Numeric distributions  
- Odd patterns after merges  

```python {run}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)

df = pd.DataFrame({
    "sales": np.random.normal(120, 25, 200),
    "region": np.random.choice(["North", "South", "East", "West"], 200),
    "channel": np.random.choice(["Online", "Offline"], 200),
})

missing_idx = np.random.choice(df.index, 20, replace=False)
df.loc[missing_idx[:10], "sales"] = np.nan
df.loc[missing_idx[10:], "region"] = None

missing = df.isna().sum().sort_values(ascending=False)

plt.figure(figsize=(8, 4.5))
plt.bar(missing.index.astype(str), missing.values, color="#6b9ac4", edgecolor="#333")
plt.title("Missing values by column")
plt.xlabel("Column")
plt.ylabel("Count")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
```

- A missing-value bar chart often shows where to clean first.

## What to remember

- Good analysis needs **good tables**.
- Cleaning and combining often matter more than modeling.
- Engineering is where raw rows become **trustworthy structure**.

---

# 4. Data Analysis

## From tables to structure

- Analysis starts when data are clean enough to inspect systematically.
- A central skill is **reshaping** tables so patterns are easy to summarize and plot.

## MultiIndex

A MultiIndex gives rows or columns **multiple label levels**.

```python
import pandas as pd

df = pd.DataFrame({
    "region": ["East", "East", "West", "West"],
    "year": [2022, 2023, 2022, 2023],
    "sales": [100, 120, 90, 140],
})

multi = df.set_index(["region", "year"])
print(multi)
```

$$\text{index} = (\text{region}, \text{year})$$

- Natural for hierarchical data; grouped summaries stay compact.

## Stack and unstack

Reshape between wide and long-ish layouts.

```python
stacked = multi.unstack()
restored = stacked.stack()
```

- `stack` pushes columns into the index; `unstack` pulls index levels out to columns—key for summaries and plots.

## Relational databases

Real data are often split across tables (customers, orders, products, …) and linked by **keys** instead of one giant sheet.

$$\text{final table} = \text{table A} \bowtie \text{table B}$$

```python
customers = pd.DataFrame({
    "customer_id": [1, 2, 3],
    "region": ["East", "West", "East"],
})

orders = pd.DataFrame({
    "customer_id": [1, 1, 2, 3, 3],
    "amount": [100, 140, 90, 160, 110],
})

joined = orders.merge(customers, on="customer_id", how="left")
```

- Analysis depends on **linking entities correctly** across tables.

## Example: grouped sales and a line plot

```python {run}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1)

df = pd.DataFrame({
    "region": np.random.choice(["East", "West", "South"], 180),
    "year": np.random.choice([2022, 2023, 2024], 180),
    "sales": np.random.normal(100, 20, 180),
})

summary = (
    df.groupby(["region", "year"])["sales"]
    .mean()
    .unstack()
)

plt.figure(figsize=(9, 4.8))
for region in summary.index:
    plt.plot(
        summary.columns,
        summary.loc[region],
        marker="o",
        linewidth=2,
        label=region,
    )
plt.title("Average sales by region and year")
plt.xlabel("Year")
plt.ylabel("Mean sales")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
```

- Group → `unstack` gives a **region × year** matrix; the plot shows **cross-section** and **time** together.
