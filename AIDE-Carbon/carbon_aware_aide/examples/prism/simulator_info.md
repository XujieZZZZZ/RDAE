# Simulator Interface Guide for GPU Model Placement Algorithms

## Overview

This guide provides essential constraints and best practices for generating GPU model placement algorithms that work correctly with the simulator. The simulator tests algorithms that optimize the placement of multiple LLM models across GPUs to minimize KVCache pressure while respecting memory constraints.

## 1. Algorithm Interface Requirements

### 1.1 Function Structure

Your algorithm MUST follow this exact structure:

```python
GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Args:
        gpu_num: Number of available GPUs
        models: List of Model objects to place
    
    Returns:
        dict: Placement mapping {gpu_id: [Model, Model, ...]}
    """
    # Your placement logic here
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    # ... algorithm implementation ...
    return placement
```

**Critical Requirements:**
- Function name MUST be `compute_model_placement`
- MUST accept exactly two parameters: `gpu_num` (int) and `models` (list)
- MUST return a dictionary mapping GPU IDs to lists of Model objects
- The constant `GPU_MEM_SIZE = 80` should be defined at module level

### 1.2 Input Parameters Explained

| Parameter | Type | Description |
|-----------|------|-------------|
| `gpu_num` | int | Number of available GPUs in the cluster (typically 5-10 in tests) |
| `models` | list[Model] | List of Model objects to be placed across GPUs |

### 1.3 Model Object Structure

Each Model object has the following attributes:

```python
@dataclass
class Model:
    model_name: str      # Unique identifier for the model
    model_size: int      # Memory footprint in GB (10-30 GB range)
    req_rate: int        # Request rate (1-10 requests/sec)
    slo: int            # Service Level Objective latency in ms (5-10 ms)
    cur_gpu_id: int     # Current GPU assignment (for reference)
```

## 2. Output Constraints (Return Value)

The `compute_model_placement` function MUST return a dictionary:

```python
{
    0: [Model1, Model2, ...],  # Models assigned to GPU 0
    1: [Model3, Model4, ...],  # Models assigned to GPU 1
    ...
    gpu_num-1: [...]           # Models assigned to last GPU
}
```

### 2.1 Critical Output Rules

**RULE 1: Memory Constraint**
```python
# ❌ WRONG: Exceeding GPU memory causes validation failure
for gpu_id in range(gpu_num):
    placement[gpu_id] = all_models  # Total size > 80 GB!

# ✅ CORRECT: Ensure total model size per GPU ≤ GPU_MEM_SIZE
for gpu_id, models_on_gpu in placement.items():
    total_size = sum(model.model_size for model in models_on_gpu)
    assert total_size <= GPU_MEM_SIZE, f"GPU {gpu_id} overcommitted"
```

**RULE 2: All Models Must Be Placed**
```python
# ❌ WRONG: Leaving models unplaced
placement = {0: [models[0]], 1: [models[1]]}  # Where are the rest?

# ✅ CORRECT: Every model must appear exactly once
all_placed = [m for models_list in placement.values() for m in models_list]
assert len(all_placed) == len(models), "All models must be placed"
```

**RULE 3: Valid GPU IDs**
```python
# ❌ WRONG: Using invalid GPU IDs
placement = {gpu_num: [model1]}  # gpu_num is out of range!

# ✅ CORRECT: GPU IDs must be in range [0, gpu_num-1]
placement = {gpu_id: [] for gpu_id in range(gpu_num)}
```

**RULE 4: Handle Infeasible Cases**
```python
# ❌ WRONG: Silently returning invalid placement
if model.model_size > GPU_MEM_SIZE:
    return {}  # Invalid!

# ✅ CORRECT: Raise an error for infeasible cases
if model.model_size > GPU_MEM_SIZE:
    raise ValueError(f"Model size {model.model_size} exceeds GPU capacity")
```

## 3. Testing and Validation

### 3.1 Test Command

```bash
python3 simulators/prism/test.py
```

### 3.2 Success Criteria

**Your algorithm succeeds if:**
1. No Python exceptions (ValueError, KeyError, AssertionError)
2. All memory constraints satisfied
3. Returns valid placement dictionary
4. Metrics are comparable to baseline (not INF or NaN)

## 4. Quick Checklist

Before submitting your algorithm, verify:

- [ ] Function named `compute_model_placement` with correct signature
- [ ] Returns `dict[int, list[Model]]` format
- [ ] All models placed exactly once
- [ ] Memory constraint satisfied: `Σ model_size ≤ GPU_MEM_SIZE` per GPU
- [ ] Valid GPU IDs: `0 ≤ gpu_id < gpu_num`
- [ ] Handles edge cases (large models, tight memory)
- [ ] Raises `ValueError` for infeasible placements
- [ ] No division by zero in KVPR calculations
- [ ] Completes within 10 seconds per test case
- [ ] Code is within `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` markers
