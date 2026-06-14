# Simulator Interface Guide for Carbon-Aware Scheduling Algorithms

## Overview

This guide provides essential constraints and best practices for generating scheduling algorithms that work correctly with the simulator. The simulator tests carbon-aware task scheduling algorithms in a Spark-like environment.

## 1. Algorithm Interface Requirements

### 1.1 Class Structure

Your algorithm MUST follow this exact structure:

```python
import numpy as np
from agents.agent import Agent

class NewAgent(Agent):             # MUST be named NewAgent, MUST inherit Agent
    def __init__(self, exec_cap, carbon_schedule, exec_lower_bound=20, **):
        Agent.__init__(self)       # MUST call parent __init__
        self.exec_map = {}         # MUST include
        self.carbon_schedule = carbon_schedule
        self.exec_cap_MAX = exec_cap   # MUST include
        self.exec_cap = exec_cap
        self.B = exec_lower_bound      # MUST include
        # Add your custom state variables here
    
    def get_carbon_intensity(self, current_time):  # recommended
        # returns (current_carbon, L, U) with L < U
        ...

    def get_action(self, obs):     # MUST implement
        # Your scheduling logic here
        return node, use_exec, carbon_aware_flag
```

**Critical Requirements:**
- Class name MUST be `NewAgent`
- MUST inherit from `Agent`
- MUST call `Agent.__init__(self)` in constructor
- MUST initialize `self.exec_map = {}`
- Constructor signature MUST match exactly

### 1.2 Constructor Parameters Explained

| Parameter | Type | Description |
|-----------|------|-------------|
| `exec_cap` | int | Maximum number of executors (workers) available in the cluster. This is the upper bound for resource allocation. |
| `carbon_schedule` | dict | Dictionary mapping timestamps (in milliseconds) to carbon intensity values. Format: `{timestamp: carbon_intensity}`. Used to query carbon intensity at any given time. |
| `exec_lower_bound` | int | Minimum number of executors that must remain active (default: 20). This is the lower bound for carbon-aware throttling - even during high carbon periods, at least `B` executors should stay active to ensure progress. |

**Constraint:** Your algorithm MUST maintain: `self.B <= self.exec_cap <= self.exec_cap_MAX` at all times.

## 2. Input Constraints (Observation Tuple)

The `get_action(self, obs)` method receives a 9-element tuple:

```python
job_dags, source_job, num_source_exec, \
frontier_nodes, executor_limits, \
exec_commit, moving_executors, action_map, current_time = obs
```

### 2.1 Input Elements

| Element | Type | Description |
|---------|------|-------------|
| `job_dags` | OrderedSet | Active job DAGs in the system |
| `source_job` | JobDAG or None | Job that just freed executors |
| `num_source_exec` | int | Number of available executors to allocate |
| `frontier_nodes` | OrderedSet | Currently executable nodes (all dependencies met) |
| `executor_limits` | dict | Executor limits per job: `{job_dag: int}` |
| `exec_commit` | ExecutorCommit | Committed executor assignments |
| `moving_executors` | MovingExecutors | Executors currently transitioning |
| `action_map` | dict | Action mapping structure |
| `current_time` | int | Current simulation time (milliseconds) |

### 2.2 Key Data Access Patterns

**Safe access patterns:**
```python
# Node properties
task_remaining = node.num_tasks - node.next_task_idx
node.job_dag  # Parent job DAG
node.frontier_nodes  # Frontier nodes for this job

# Commitment tracking (ALWAYS use .get() to avoid KeyError)
committed = exec_commit.node_commit.get(node, 0)
moving_count = moving_executors.count(node)

# Carbon schedule access
carbon_intensity = self.carbon_schedule[timestamp_key]
```

**AVOID these common errors:**
```python
# ❌ WRONG: Direct indexing can cause KeyError
committed = exec_commit.node_commit[node]

# ✅ CORRECT: Use .get() with default value
committed = exec_commit.node_commit.get(node, 0)
```

## 3. Output Constraints (Return Value)

The `get_action` method MUST return a 3-tuple:

```python
return (node, use_exec, carbon_aware_flag)
```

### 3.1 Return Value Specification

| Element | Type | Constraints |
|---------|------|-------------|
| `node` | Node or None | Selected node to schedule (None = defer execution) |
| `use_exec` | int | Number of executors to allocate |
| `carbon_aware_flag` | bool | True if deferring for carbon reasons |

### 3.2 Critical Output Rules

**RULE 1: Always Schedule When Possible**
```python
# ❌ WRONG: Returning None when tasks are available causes infinite loop
if frontier_nodes and num_source_exec > 0:
    return None, num_source_exec, False  # TIMEOUT!

# ✅ CORRECT: Always schedule at least one task
if frontier_nodes and num_source_exec > 0:
    selected_node = max(frontier_nodes, key=lambda n: score(n))
    use_exec = compute_executors(selected_node, num_source_exec)
    return selected_node, use_exec, False
```

**RULE 2: use_exec Must Be Positive**
```python
# ❌ WRONG: Returning 0 executors causes assertion failure
return node, 0, False

# ✅ CORRECT: Ensure use_exec >= 1
task_remaining = node.num_tasks - node.next_task_idx
committed = exec_commit.node_commit.get(node, 0) + moving_executors.count(node)
use_exec = max(1, min(num_source_exec, task_remaining - committed))
return node, use_exec, False
```

**RULE 3: Executor Cap Constraints**
```python
# MUST maintain: self.B <= self.exec_cap <= self.exec_cap_MAX
# When reducing exec_cap, ensure scheduler doesn't get stuck

# ❌ WRONG: Can cause infinite loop
if num_exec >= self.exec_cap:
    return None, num_source_exec, True  # Stuck forever if tasks never finish

# ✅ CORRECT: Still schedule when frontier_nodes available
if num_exec >= self.exec_cap and frontier_nodes:
    # Pick one node, use minimal executors
    node = next(iter(frontier_nodes))
    return node, 1, False
```

## 4. Testing and Validation

### 4.1 Test Command

```bash
python3 simulator/test.py \
    --num_exp 1 \
    --exec_cap 50 \
    --num_init_dags 1 \
    --num_stream_dags 50 \
    --canvs_visualization 0 \
    --test_schemes fifo newalgo
```

### 4.2 Success Criteria

**Your algorithm succeeds if:**
1. No timeout (completes within 180 seconds)
2. No Python exceptions (KeyError, AttributeError, AssertionError)
3. Returns valid metrics (carbon usage, execution time, job completion time)
4. Metrics are comparable to baseline (not INF or NaN)

## 5. Quick Checklist

Before submitting your algorithm, verify:

- [ ] Class named `NewAgent`, inherits `Agent`
- [ ] Constructor signature matches: `__init__(self, exec_cap, carbon_schedule, exec_lower_bound=20)`
- [ ] Initializes `self.exec_map = {}`
- [ ] `get_action` returns 3-tuple: `(node, use_exec, carbon_aware_flag)`
- [ ] `use_exec > 0` when returning a node
- [ ] Always schedules when `frontier_nodes` is non-empty and `num_source_exec > 0`
- [ ] Uses `.get(node, 0)` for dictionary access
- [ ] Maintains `self.B <= self.exec_cap <= self.exec_cap_MAX`
- [ ] No undefined methods (e.g., `self.invoke_model()`)
- [ ] Handles empty `frontier_nodes` gracefully
