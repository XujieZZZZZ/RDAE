Problem Context:
- You are given a pandas DataFrame `df` with text data in rows and columns
- The goal is to reorder columns to maximize prefix reuse when processing rows sequentially
- Prefix reuse occurs when consecutive rows have matching values in the same column positions
- Prefix reuse is defined as consecutive field values (starting from the first column) that are **exact matches** with the corresponding fields of the previous row.
- The **hit score** of a row is defined as the **sum of squares of the string lengths** of the matching prefix fields.
- The algorithm will be evaluated on a combined metric that balances accuracy (prefix reuse) and speed (runtime).

Formally:
- For a given column ordering `C`, PHC(C) = sum over all rows `r` of `hit(C, r)`
- `hit(C, r)` = sum of `len(df[r][C[f]])^2` for all f in prefix where `df[r][C[f]] == df[r-1][C[f]]`; zero if mismatch starts at the first field.
- Runtime is measured as wall-clock seconds to compute the reordered DataFrame from the input DataFrame.
- Combined score used for selection: `combined_score = 0.95 * average_hit_rate + 0.05 * (12 - min(12, average_runtime)) / 12`.

Required API (DO NOT CHANGE):
- You must keep the existing NewEvolved class structure and the reorder method signature:
    ```python
    class NewEvolved(Algorithm):
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
    ```
- You can modify the internal implementation of methods but must preserve the class structure and method signatures
- The reorder method must return a tuple of (reordered_dataframe, column_orderings)

Constraints:
- Only reorder columns, do not change row order or add/remove rows or columns
- You must have different column orderings for different rows to maximize prefit hit rate
- Return a DataFrame with the same shape as input
- Use exact string matching for prefix calculations
- Keep memory usage reasonable for large datasets
- Preserve all existing method signatures and class structure
- The algorithm will be called with the same parameters as the original NewEvolved

Simply return the optimized NewEvolved class, do not provide explanations.

