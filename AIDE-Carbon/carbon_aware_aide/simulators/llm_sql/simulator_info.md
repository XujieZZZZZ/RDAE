# Simulator Interface Guide for DataFrame Column Reordering Algorithms

## Overview

This guide provides essential constraints and best practices for generating column reordering algorithms that work correctly with the simulator. The simulator tests DataFrame optimization algorithms for maximizing LLM prompt caching efficiency through intelligent column reordering.

**Important**: The baseline algorithm is the `Evolved` class in `initial_program.py`. Your task is to create an improved `Evolved` class in `new_program.py` that outperforms the baseline. There is also a `QuickGreedy` class in `quick_greedy.py` that serves as a simpler reference implementation.

## 1. Algorithm Interface Requirements

### 1.1 Class Structure

Your algorithm MUST follow this exact structure:

```python
import pandas as pd
from solver import Algorithm
from typing import List, Tuple

class Evolved(Algorithm):             # MUST be named Evolved, MUST inherit Algorithm
    def __init__(self, df: pd.DataFrame = None):
        self.df = df               # MUST include
        # Add your custom state variables here
    
    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: int = None,
        col_stop: int = None,
        col_merge: List[List[str]] = [],
        one_way_dep: List[Tuple[str, str]] = [],
        distinct_value_threshold: float = 0.8,
        parallel: bool = True,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:     # MUST implement
        # Your reordering logic here
        return reordered_df, column_orderings
```

**Critical Requirements:**
- Class name MUST be `Evolved`
- MUST inherit from `Algorithm`
- MUST implement `reorder` method with exact signature
- Constructor MUST accept optional `df` parameter

### 1.2 Method Parameters Explained

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | pd.DataFrame | Input DataFrame to reorder. Contains mixed data types (str, int, float, bool, NaN). |
| `early_stop` | int | Early stopping threshold for recursive algorithms. If weighted count falls below this, stop recursion. Default: 0 (no early stop). |
| `row_stop` | int or None | Maximum depth for row-wise recursion. Limits how many times the algorithm recursively splits rows. Default: None (no limit). |
| `col_stop` | int or None | Maximum depth for column-wise recursion. Limits how many times the algorithm recursively reorders columns. Default: None (no limit). |
| `col_merge` | List[List[str]] | List of column groups to merge before reordering. Each inner list contains column names to merge. Example: `[['col1', 'col2'], ['col3', 'col4']]`. |
| `one_way_dep` | List[Tuple[str, str]] | List of (parent, child) column dependencies. If col1 depends on col2, place col2 before col1. Example: `[('city', 'country')]`. |
| `distinct_value_threshold` | float | Threshold (0-1) for discarding high-cardinality columns. Columns with `nunique() > len(df) * threshold` are moved to the end. Default: 0.8. |
| `parallel` | bool | Whether to use parallel processing (ThreadPoolExecutor). Set to False for debugging. Default: True. |

**Constraint:** Your algorithm MUST return a DataFrame with the same shape as the input (same number of rows and columns, unless col_merge is used).

## 2. Input Constraints (DataFrame Properties)

### 2.1 DataFrame Characteristics

The input DataFrame `df` has the following properties:

| Property | Description |
|----------|-------------|
| **Shape** | Typically 1,000 to 100,000 rows, 5 to 50 columns |
| **Data Types** | Mixed: `str`, `int`, `float`, `bool`, `NaN` (missing values) |
| **Column Names** | String column names, may contain spaces or special characters |
| **Index** | Default integer index (0, 1, 2, ...) |
| **Missing Values** | May contain NaN values that must be handled gracefully |

### 2.2 Safe Data Access Patterns

**Safe access patterns:**
```python
# Column statistics
num_rows, column_stats = self.calculate_col_stats(df, enable_index=True)
# column_stats: List[(col_name, num_groups, avg_length, score)]

# Value counts
value_counts = Counter(df.stack())  # All values across all columns

# Handle missing values
df.fillna("")  # Replace NaN with empty string
df.dropna()    # Drop rows with NaN

# Column access
for col in df.columns:
    unique_count = df[col].nunique()
    values = df[col].values
```

**AVOID these common errors:**
```python
# ❌ WRONG: Not handling NaN values
row_string = "".join(row.astype(str).values)  # NaN becomes "nan"

# ✅ CORRECT: Handle NaN explicitly
row_string = "".join(row.fillna("").astype(str).values)

# ❌ WRONG: Assuming column names are valid Python identifiers
value = row.column_name  # Fails if column has spaces

# ✅ CORRECT: Use bracket notation
value = row['column name']
```

## 3. Output Constraints (Return Value)

The `reorder` method MUST return a 2-tuple:

```python
return (reordered_df, column_orderings)
```

### 3.1 Return Value Specification

| Element | Type | Constraints |
|---------|------|-------------|
| `reordered_df` | pd.DataFrame | Reordered DataFrame with same shape as input (unless col_merge used) |
| `column_orderings` | List[List[str]] | List of column orderings per row (can be empty list `[]`) |

### 3.2 Critical Output Rules

**RULE 1: Shape Preservation**
```python
# ❌ WRONG: Changing DataFrame shape
def reorder(self, df, ...):
    reordered = df.drop(columns=['col1'])  # Lost a column!
    return reordered, []

# ✅ CORRECT: Preserve shape
def reorder(self, df, ...):
    reordered = df[new_column_order]  # Same columns, different order
    assert reordered.shape == df.shape
    return reordered, []
```

**RULE 2: Handle Column Merging**
```python
# When col_merge is provided, output may have fewer columns
if col_merge:
    # Merge specified columns
    for cols_to_merge in col_merge:
        df = self.merging_columns(df, cols_to_merge)
    # Output shape: (same rows, fewer columns)
    assert reordered_df.shape[0] == original_df.shape[0]
else:
    # No merging: exact shape match
    assert reordered_df.shape == original_df.shape
```

**RULE 3: Index Preservation**
```python
# ❌ WRONG: Losing original row order information
reordered = df.sort_values(by=df.columns.tolist())
# Can't map back to original rows!

# ✅ CORRECT: Track original index
df['original_index'] = range(len(df))
# ... perform reordering ...
final_df = final_df.drop(columns=['original_index'])
return final_df, []
```

**RULE 4: Column Orderings (Optional)**
```python
# Option 1: Return empty list (most common)
return reordered_df, []

# Option 2: Return per-row column orderings
column_orderings = [
    ['col1', 'col2', 'col3'],  # Row 0 column order
    ['col2', 'col1', 'col3'],  # Row 1 column order
    # ... one list per row
]
return reordered_df, column_orderings
```

## 4. Evaluation Metrics

### 4.1 Prefix Hit Count (PHC)

The primary metric is **prefix hit count**, computed as follows:

```python
def evaluate_df_prefix_hit_cnt(df: pd.DataFrame) -> Tuple[int, float]:
    """
    Compute prefix hit count using Trie data structure.
    
    Returns:
        (total_prefix_hit_count, hit_rate_percentage)
    """
    trie = Trie()
    total_prefix_hit_count = 0
    total_string_length = 0
    
    for _, row in df.iterrows():
        row_string = "".join(row.fillna("").astype(str).values)
        total_string_length += len(row_string)
        
        # Count matching prefix characters
        prefix_hit = trie.longest_common_prefix(row_string)
        total_prefix_hit_count += prefix_hit
        
        # Insert current row into trie
        trie.insert(row_string)
    
    hit_rate = (total_prefix_hit_count / total_string_length) * 100
    return total_prefix_hit_count, hit_rate
```

**Key Points:**
- Rows are processed sequentially (order matters!)
- Prefix matching is character-level, not token-level
- Each row is compared only to the Trie (all previous rows)
- Higher hit rate = better caching efficiency

### 4.2 Runtime Measurement

Runtime is measured as wall-clock time:

```python
import time

start_time = time.time()
reordered_df, _ = Evolved().reorder(df, ...)
runtime = time.time() - start_time
```

### 4.3 Combined Score

The final score balances hit rate and runtime:

```python
# Test script scoring (test.py)
score = 0.5 * average_hit_rate + 0.5 * (1.0 / (1.0 + average_runtime))

# Evaluator scoring (evaluator.py)
score = 0.95 * average_hit_rate + 0.05 * (12 - min(12, average_runtime)) / 12
```

## 5. Testing and Validation

### 5.1 Test Command

```bash
cd /Users/dingch/Desktop/Projections/AI4Research/aideml-main-FIFO/llm_sql_simulator
python3 test.py
```

This will:
1. Load 5 test datasets (movies, beer, BIRD, PDMX, products)
2. Run your `Evolved` algorithm on each dataset
3. Compute hit rates and runtimes
4. Print combined score and metrics JSON

### 5.2 Success Criteria

**Your algorithm succeeds if:**
1. No Python exceptions (KeyError, AttributeError, ValueError, AssertionError)
2. Completes within reasonable time (< 60 seconds per dataset)
3. Returns valid DataFrames with correct shapes
4. Achieves hit rate > 0 (non-zero prefix reuse)
5. Combined score > baseline (QuickGreedy)

### 5.3 Common Failure Modes

| Error | Cause | Fix |
|-------|-------|-----|
| `KeyError: 'column_name'` | Column doesn't exist or was dropped | Check column exists before accessing |
| `AssertionError: shape mismatch` | Lost or added columns/rows | Verify shape preservation |
| `AttributeError: 'NoneType'` | Returned None instead of DataFrame | Always return valid DataFrame |
| `ValueError: cannot reindex` | Index mismatch during merge | Use `reset_index(drop=True)` |
| Timeout (> 60s) | Algorithm too slow | Optimize recursion depth or use parallel processing |

## 6. Performance Optimization Tips

### 6.1 Use Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
    results = [f.result() for f in futures]
```

### 6.2 Cache Expensive Computations

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def calculate_length(self, value):
    # Cached computation
    return len(str(value)) ** 2
```

### 6.3 Limit Recursion Depth

```python
if row_stop is not None and current_depth >= row_stop:
    # Fall back to simple fixed reordering
    return self.fixed_reorder(df)
```

### 6.4 Early Stopping

```python
max_value = self.find_max_group_value(df, value_counts, early_stop=early_stop)
if max_value is None:
    # No significant groups found, stop recursion
    return self.fixed_reorder(df)
```

## 7. Quick Checklist

Before submitting your algorithm, verify:

- [ ] Class named `Evolved`, inherits `Algorithm`
- [ ] Constructor signature: `__init__(self, df: pd.DataFrame = None)`
- [ ] `reorder` method signature matches exactly (8 parameters)
- [ ] Returns 2-tuple: `(pd.DataFrame, List[List[str]])`
- [ ] Output DataFrame has same shape as input (or correct shape if col_merge used)
- [ ] Handles NaN values without errors (use `.fillna("")`)
- [ ] No undefined methods or attributes
- [ ] Uses `.get()` or `try/except` for safe dictionary access
- [ ] Respects `row_stop` and `col_stop` limits
- [ ] Completes within 60 seconds per dataset
- [ ] No hardcoded dataset-specific logic

## 8. Example Minimal Implementation

```python
import pandas as pd
from solver import Algorithm
from typing import List, Tuple

class Evolved(Algorithm):
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
    
    def reorder(
        self,
        df: pd.DataFrame,
        early_stop: int = 0,
        row_stop: int = None,
        col_stop: int = None,
        col_merge: List[List[str]] = [],
        one_way_dep: List[Tuple[str, str]] = [],
        distinct_value_threshold: float = 0.8,
        parallel: bool = True,
    ) -> Tuple[pd.DataFrame, List[List[str]]]:
        # Simple strategy: sort columns by number of unique values
        num_rows, column_stats = self.calculate_col_stats(df)
        sorted_cols = [col for col, _, _, _ in column_stats]
        
        # Reorder columns
        reordered_df = df[sorted_cols]
        
        # Sort rows by all columns
        reordered_df = reordered_df.sort_values(by=sorted_cols)
        
        return reordered_df, []
```

This minimal implementation satisfies all interface requirements and will run successfully, though it may not achieve high scores.
