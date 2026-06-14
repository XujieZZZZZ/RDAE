# generate_enhanced.py - Enhanced code generation with domain knowledge
import logging
from llm import call_llm
from typing import List, Dict, Optional
from utils import *
from logger import log_llm_output

def generate_pseudocode(code: str) -> str:
    """
    Extract high-level algorithm pseudocode from implementation code.
    """

    prompt = f"""
You are an expert in scheduling and systems optimization.

Your task is to ABSTRACT the core algorithmic idea of the given system
into a HIGH-LEVEL algorithm description, similar in style to
"Algorithm 1" in academic papers.

Input:
- A baseline scheduling algorithm implementation
- The algorithm contains low-level control logic and implementation details

IMPORTANT:
This is NOT a code translation task.

DO NOT:
- Translate code line by line
- Include specific variable names, constants, or formulas
- Mirror the original control flow structure
- Describe helper functions or low-level computations

INSTEAD:
- Identify the algorithm's key STATE variables
- Identify the DECISION points
- Identify the OVERALL control loop
- Describe the algorithm in terms of conceptual steps

Abstraction requirements:
1. Focus on WHAT decisions are made, not HOW they are computed
2. Merge low-level logic into meaningful algorithmic phases
3. Express thresholds, scores, or models conceptually (e.g., "carbon intensity bounds")
4. Use concise, structured pseudocode suitable for an academic paper

Output format (strictly follow):

Algorithm X: <Concise Algorithm Name>

Inputs:
- Describe high-level inputs (e.g., workload state, system state)

State:
- List core maintained state variables

Procedure:
1. Initialization phase
2. Decision phase
3. Adjustment / adaptation phase
4. Output / action selection

Notes:
- Briefly explain how this algorithm differs from the baseline at a conceptual level

Code:
{code}

Do NOT include equations, code, or citations.
The output should read like a textbook or paper algorithm, not implementation pseudocode.

"""

    response = call_llm(prompt)
    log_llm_output("pseudocode", response)

    return response.strip()


def generate_algorithm_with_constraints(
    baseline_algo: str,
    baseline_metrics: Optional[Dict] = None,
    parent_metrics: Optional[Dict] = None,
    failed_attempts: Optional[List[str]] = None,
    summary_content: str = ""
) -> str:
    """
    Generate an improved algorithm description with domain knowledge constraints.
    
    Args:
        baseline_algo: Baseline algorithm description
        baseline_metrics: Performance metrics of the fifo baseline
        parent_metrics: Performance metrics of the parent solution
        failed_attempts: List of previously failed modification attempts
        summary_content: Content of Summary.md for RAG context (injected at design stage)
    """
    
    # Build baseline performance context
    baseline_context = ""
    if baseline_metrics:
        baseline_context = f"""
【FIFO BASELINE PERFORMANCE】
- Total Carbon Usage: {baseline_metrics.get('carbon', 'N/A')}
- Total Execution Time: {baseline_metrics.get('time', 'N/A')}
- Average Job Completion Time: {baseline_metrics.get('jct', 'N/A')}

NOTE: Due to test data variability, focus on RELATIVE improvements rather than absolute values.

YOUR GOAL: Add carbon-awareness to this pure FIFO baseline to reduce carbon usage while maintaining acceptable job completion time.
"""
    
    # Build parent performance context
    parent_context = ""
    if parent_metrics:
        parent_context = f"""
【PARENT SOLUTION PERFORMANCE】
- Total Carbon Usage: {parent_metrics.get('carbon', 'N/A')}
- Total Execution Time: {parent_metrics.get('time', 'N/A')}
- Average Job Completion Time: {parent_metrics.get('jct', 'N/A')}

YOUR GOAL: Generate an algorithm that IMPROVES upon the parent solution.
"""
    
    # Build failed attempts context
    failed_context = ""
    if failed_attempts and len(failed_attempts) > 0:
        failed_context = f"""
【PREVIOUSLY FAILED APPROACHES】
The following modification strategies have been tried and FAILED (do NOT repeat them):
{chr(10).join(f"- {attempt}" for attempt in failed_attempts[-3:])}

You MUST propose a DIFFERENT approach.
"""

    # RAG: inject Summary.md at algorithm DESIGN stage (most impactful moment -
    # LLM is constructing the approach, not just writing code)
    rag_context = ""
    if summary_content:
        rag_context = (
            "\n《REFERENCE: Proven Carbon-Aware Scheduling Techniques (from research)》\n"
            + summary_content[:3000]
            + "\n\nUse the above techniques (CAP thresholding, PCAPS importance scoring, "
            + "valley forecasting, DAG-aware deferral, etc.) as concrete starting points. "
            + "You may adapt, combine, or extend them.\n"
        )

    time_out_prevention = f"""
    **TIMEOUT PREVENTION - RULE 1: ALWAYS SCHEDULE WHEN POSSIBLE**
   get_action MUST schedule at least one task whenever frontier_nodes is non-empty AND
   num_source_exec > 0. Carbon-aware filtering may REDUCE which nodes are preferred, but must
   NEVER cause all nodes to be skipped. Always include an unconditional fallback:
       selected_node = max(frontier_nodes, key=lambda n: <some_score>)
       use_exec = max(1, min(num_source_exec, <task_remaining>))
       return selected_node, use_exec, False
   Returning (None, num_source_exec, False) when tasks are available = INFINITE LOOP = timeout.

   **TIMEOUT PREVENTION - RULE 2: use_exec MUST BE > 0 BEFORE RETURNING A NODE**
   Every code path that returns (node, use_exec, flag) MUST guarantee use_exec >= 1.
   NEVER return (node, 0, flag) or (node, negative, flag).
   The safe pattern for computing use_exec:
       task_remaining = node.num_tasks - node.next_task_idx
       committed = exec_commit.node_commit.get(node, 0) + moving_executors.count(node)
       use_exec = max(1, min(num_source_exec, max(0, task_remaining - committed)))
   If task_remaining - committed <= 0, pick a DIFFERENT node or fall back to use_exec=1.
   Do NOT use exec_commit.node_commit[node] (KeyError risk); use .get(node, 0) instead.

   **TIMEOUT PREVENTION - RULE 3: exec_cap FLOOR MUST NOT TRAP THE SCHEDULER**
   When exec_cap is reduced by carbon logic (e.g. to self.B), and num_exec >= exec_cap,
   the scheduler returns None forever if tasks never finish. This causes infinite loops.
   SAFE PATTERN: if num_exec >= exec_cap AND frontier_nodes is non-empty, still schedule
   one task (pick one node, use_exec=1) rather than returning None.
   Alternatively, only return None when frontier_nodes is empty.

   **SCORE/THRESHOLD DIMENSION MISMATCH (CAUSES TIMEOUT)**:
   Never compare a task score computed from task counts or DAG depth (values like 0, 1, 2, 3...)
   directly against a carbon intensity threshold (values like 200, 350, 450...).
   Keep task scoring and carbon thresholding as SEPARATE steps:
     - Step 1: carbon gate -> decide HOW MANY executors to use based on carbon level
     - Step 2: task ranking -> pick WHICH node to run based on DAG/urgency score (0.0-1.0 range)
   Normalize task scores to [0, 1] before applying any threshold."""

    prompt = f"""
You are an expert in carbon-aware scheduling and systems optimization.

Your task is to DESIGN an improved algorithm by IMPROVING the given baseline algorithm.

{baseline_context}

{parent_context}

{failed_context}

【BASELINE ALGORITHM】
{baseline_algo}

【REFRENCE】
{rag_context}

【DOMAIN KNOWLEDGE - Carbon-Aware Scheduling】

Core Principles:
1. **Carbon Intensity Variability**: 
   - Carbon intensity changes over time based on energy sources
   - Future carbon intensity can be predicted using historical data
   - Scheduling decisions should consider both current and future carbon levels
   
2. **Task Scheduling Constraints**:
   - Tasks have precedence constraints (DAG structure)
   - Tasks require computational resources (executors)
   - Delaying tasks affects overall job completion time
   
3. **Trade-offs**:
   - Lower carbon usage vs. Faster completion time
   - Immediate execution vs. Delayed execution
   - Resource utilization vs. Carbon efficiency

【CRITICAL INVARIANTS - DO NOT VIOLATE】
1. self.B <= self.exec_cap <= self.exec_cap_MAX (executor cap must be within bounds)
2. L < U must always hold (lower bound < upper bound)
3. Must respect task precedence constraints (DAG dependencies)
4. Do NOT modify __init__ signature, class name (NewAgent), or self.exec_map structure


【AVAILABLE SYSTEM INFORMATION】
The system provides rich information that you can leverage:

1. **Carbon Intensity Data**:
   - self.carbon_schedule: Historical and future carbon intensity values
   - current_time: Current timestamp
   - You can analyze patterns, trends, and predictions

2. **Task Information** (from FIFO scheduler state):
   - job_dags: Active job dependency graphs (DAG structure)
   - frontier_nodes: Currently executable tasks
   - source_job: Job that just freed executors
   - exec_map: Current executor allocation per job
   - exec_commit: Committed executor assignments
   - moving_executors: Executors currently transitioning

3. **Resource Information**:
   - num_source_exec: Available executors
   - executor_levels: Possible executor allocation levels
   - exec_commit: Committed executors
   - moving_executors: Executors in transition

【EXPLORATION DIRECTIONS】

Think broadly about how to improve carbon-aware scheduling. Consider:

1. **Temporal Strategies**:
   - When should tasks be executed vs. delayed?
   - How to predict optimal execution windows?
   - How to balance immediate vs. future carbon costs?

2. **Resource Management**:
   - How to allocate executors efficiently?
   - Should all tasks receive equal resources?
   - How to adapt resource allocation to carbon levels?

3. **Decision Granularity**:
   - Should decisions be made globally or per-task?
   - Should decisions be binary or continuous?
   - How to handle heterogeneous workloads?

4. **Information Utilization**:
   - What information is currently unused?
   - How can Decima's outputs inform carbon decisions?
   - Are there patterns in carbon intensity or task characteristics?

5. **Adaptive Mechanisms**:
   - Should thresholds/parameters be static or dynamic?
   - How to adapt to different carbon intensity patterns?
   - How to learn from execution history?

【FORBIDDEN MODIFICATIONS】
1. DO NOT rename the class (must remain NewAgent)
2. DO NOT change the __init__ signature
3. DO NOT violate the critical invariants listed above
4. DO NOT remove core FIFO functionality without replacement

【DESIGN PHILOSOPHY】
- Be creative and think outside the box
- Consider unconventional approaches
- Don't be constrained by the baseline's design choices
- Focus on the fundamental problem: reducing carbon while maintaining performance
- Use all available information effectively

【DESIGN REQUIREMENTS】
1. The improved algorithm MUST remain compatible with the baseline structure
2. Each modification should have a clear rationale based on:
   - Carbon-aware scheduling principles
   - Observed limitations of the baseline
   - Theoretical or empirical evidence
3. The algorithm should be implementable with minimal code changes
4. Modifications should be conservative and incremental

【OUTPUT FORMAT】

Provide your response in the following structure:

1. **Observation**
   - What patterns or opportunities do you observe in the baseline?
   - What information is available but underutilized?
   - What assumptions does the baseline make that could be challenged?

2. **Novel Approach**
   - Describe your proposed approach at a high level
   - Explain the core idea and intuition
   - What makes this approach different from the baseline?
   
3. **Mechanism**
   - How does your approach work?
   - What decisions does it make and when?
   - How does it use available information?

4. **Algorithm Description**
   - Step-by-step decision logic
   - Key differences from the baseline
   - Use pseudocode-like language (not actual code)

5. **Expected Behavior**
   - How should this behave in high carbon periods?
   - How should this behave in low carbon periods?
   - What trade-offs does this approach make?

Do NOT include citations, equations, or actual code.
Be creative and explore unconventional ideas.
Don't be afraid to propose significantly different approaches.
"""
    
    response = call_llm(prompt)
    log_llm_output("algorithm_enhanced", response)
    
    return response


def modify_code_block_with_constraints(
    code_block: str,
    change_instruction: str,
    baseline_metrics: Optional[Dict] = None,
    validation_rules: Optional[List[str]] = None,
    summary_content: str = ""
) -> str:
    """
    Modify the marked code region with domain knowledge constraints and validation rules.
    
    Args:
        code_block: Full code with markers
        change_instruction: High-level algorithm description
        baseline_metrics: Baseline performance metrics (fifo baseline)
        validation_rules: List of validation rules to enforce
        summary_content: Content of Summary.md for RAG context
    """
    
    # Build baseline context
    baseline_context = ""
    if baseline_metrics:
        baseline_context = f"""
【BASELINE PERFORMANCE METRICS】
- Total Carbon Usage: {baseline_metrics.get('carbon', 'N/A')}
- Total Execution Time: {baseline_metrics.get('time', 'N/A')}
- Average Job Completion Time: {baseline_metrics.get('jct', 'N/A')}

YOUR IMPLEMENTATION MUST REDUCE CARBON USAGE vs THIS BASELINE.
"""
    
    # Build validation rules
    validation_context = ""
    if validation_rules:
        validation_context = f"""
【VALIDATION RULES - YOUR CODE MUST SATISFY】
{chr(10).join(f"{i+1}. {rule}" for i, rule in enumerate(validation_rules))}
"""
    

    # RAG: inject Summary.md content
    rag_context = ""
    # if summary_content:
    #     rag_context = (
    #         "\n《REFERENCE: Carbon-Aware Scheduling Techniques (from research summary)》\n"
    #         + summary_content[:3000]
    #         + "\n\nUse these techniques as inspiration for your implementation.\n"
    #     )

    prompt = f"""
You are rewriting a specific region of an existing Python source file for a carbon-aware scheduling algorithm.

This is NOT a patching or incremental editing task.
This is a FULL REWRITE of the code inside a marked region.

Your task is to regenerate the ENTIRE code block between the markers as a COMPLETE and SELF-CONTAINED Python implementation.

━━━━━━━━━━━━━━━━━━━━
STRICT EDIT BOUNDARY
━━━━━━━━━━━━━━━━━━━━

The file contains two markers:

# === BEGIN CARBON-AWARE LOGIC ===
# === END CARBON-AWARE LOGIC ===

You are ONLY allowed to output code that belongs BETWEEN these two markers.

Do NOT output the markers themselves.
Do NOT output any code before or after the marked region.

IMPORTANT: The marked region includes import statements AND the full class definition.
Your output must start with import lines and include the complete NewAgent class.

MANDATORY OUTPUT STRUCTURE (follow exactly):

    import numpy as np
    from agents.agent import Agent
    # add other imports as needed


    class NewAgent(Agent):             # MUST be named NewAgent, MUST inherit Agent
        def __init__(self, exec_cap, carbon_schedule, exec_lower_bound=20):
            Agent.__init__(self)       # MUST call parent __init__
            self.exec_map = {{}}         # MUST include
            self.carbon_schedule = carbon_schedule
            self.exec_cap_MAX = exec_cap   # MUST include
            self.exec_cap = exec_cap
            self.B = exec_lower_bound      # MUST include
            # your additional state here

        def get_carbon_intensity(self, current_time):  # recommended
            # returns (current_carbon, L, U) with L < U
            ...

        def get_action(self, obs):     # MUST implement
            # returns (node, use_exec, carbon_aware_flag)
            ...

HARD RULES (any violation = immediate test failure):
1. First line of output MUST be an import (e.g. import numpy as np)
2. Class MUST be named exactly `NewAgent`
3. MUST inherit Agent: `class NewAgent(Agent):`
4. MUST include `from agents.agent import Agent`
5. MUST call `Agent.__init__(self)` in __init__
6. MUST set self.exec_map = {{}} in __init__
7. MUST set self.exec_cap_MAX = exec_cap in __init__
8. MUST set self.B = exec_lower_bound in __init__
9. get_action MUST return exactly a 3-tuple: (node, use_exec, carbon_aware_flag)
10. self.B <= self.exec_cap <= self.exec_cap_MAX at all times
11. Do NOT call self.invoke_model() -- it does not exist in this codebase
12. frontier_nodes comes from obs tuple directly (not job_dag.frontier_nodes)
13. Methods indented 4 spaces inside class body
14. No undefined variables, no TODOs, no placeholders


━━━━━━━━━━━━━━━━━━━━
IMPLEMENTATION CONTEXT
━━━━━━━━━━━━━━━━━━━━

{baseline_context}


【AVAILABLE SYSTEM STATE】
You have access to:
- self.carbon_schedule: Dict mapping timestamps to carbon intensity values
- current_time: Current timestamp
- obs: Observation tuple containing job_dags, frontier_nodes, exec_map, etc.
In get_action(self, obs), the obs tuple contains:
    job_dags, source_job, num_source_exec,
    frontier_nodes, executor_limits,
    exec_commit, moving_executors, action_map, current_time

Class attributes available:
- self.carbon_schedule: dict mapping timestamp_ms -> carbon_intensity -- USE THIS for carbon decisions
- self.exec_cap_MAX, self.exec_cap, self.B, self.exec_map

You may import any standard Python library (numpy, math, scipy, etc.).

【REQUIRED FUNCTIONS】
You MUST implement these two functions with the exact signatures:

1. get_carbon_intensity(self, current_time) -> (current_carbon, L, U)
   - Returns current carbon intensity and bounds
   - L and U are used for decision making
   - Must satisfy: L < U

2. get_action(self, obs) -> (node, use_exec, carbon_aware_flag)
   - Makes scheduling decisions
   - node: Selected task to execute (or None to defer)
   - use_exec: Number of executors to allocate
   - carbon_aware_flag: Boolean indicating if decision is carbon-aware

{validation_context}

━━━━━━━━━━━━━━━━━━━━
IMPLEMENTATION REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━

1. COMPLETE FUNCTION IMPLEMENTATION
   - Both functions must be fully implemented
   - Include all necessary logic and return statements
   - Handle edge cases appropriately

2. SELF-CONTAINED CODE
   - Use only class attributes (self.*) and function parameters
   - Do not reference undefined variables
   - Do not leave TODOs or placeholders

3. CORRECT PYTHON SYNTAX
   - Proper indentation (4 spaces per level)
   - Valid Python syntax
   - No syntax errors

4. PRESERVE INTERFACE
   - Function signatures must match exactly
   - Return types must match expected format
   - Must integrate with existing class structure

5. USE AVAILABLE INFORMATION
   - Leverage self.carbon_schedule for carbon data
   - Access obs tuple for system state
   - Utilize any other available class attributes

━━━━━━━━━━━━━━━━━━━━
CREATIVE FREEDOM
━━━━━━━━━━━━━━━━━━━━

You have significant freedom in HOW you implement these functions:
- Choose your own approach to carbon prediction
- Design your own decision-making logic
- Use any mathematical or algorithmic techniques
- Leverage any available information creatively

The only constraints are:
- Must satisfy validation rules (if provided)
- Must return correct types
- Must be valid Python code

Think creatively about how to reduce carbon usage while maintaining performance!

━━━━━━━━━━━━━━━━━━━━
IMPLEMENTATION INSTRUCTION
━━━━━━━━━━━━━━━━━━━━

{change_instruction}

━━━━━━━━━━━━━━━━━━━━
FULL FILE FOR CONTEXT
━━━━━━━━━━━━━━━━━━━━

{code_block}

━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT)
━━━━━━━━━━━━━━━━━━━━

Output ONLY valid Python code. No markdown fences. No explanation.
Your output MUST start with: import numpy as np
Your output MUST include: from agents.agent import Agent
Your output MUST define: class NewAgent(Agent):
Your output ends with the last line of the NewAgent class.
"""

    response = call_llm(prompt)
    log_llm_output("code_enhanced", response)

    return response


def modify_code_block_with_debug_constraints(
        code_block: str,
        change_instruction: str,
        baseline_metrics: Optional[Dict] = None,
        validation_rules: Optional[List[str]] = None
) -> str:
    """
    Modify the marked code region with domain knowledge constraints and validation rules.

    Args:
        code_block: Full code with markers
        change_instruction: High-level algorithm description
        baseline_metrics: Baseline performance metrics
        validation_rules: List of validation rules to enforce
    """

    # Build baseline context
    baseline_context = ""
    if baseline_metrics:
        baseline_context = f"""
【BASELINE PERFORMANCE METRICS】
- Total Carbon Usage: {baseline_metrics.get('carbon', 'N/A')}
- Total Execution Time: {baseline_metrics.get('time', 'N/A')}
- Average Job Completion Time: {baseline_metrics.get('jct', 'N/A')}

YOUR IMPLEMENTATION MUST AIM TO REDUCE CARBON USAGE.
"""

    # Build validation rules
    validation_context = ""
    if validation_rules:
        validation_context = f"""
【VALIDATION RULES - YOUR CODE MUST SATISFY】
{chr(10).join(f"{i + 1}. {rule}" for i, rule in enumerate(validation_rules))}
"""

    prompt = f"""
You are rewriting a specific region of an existing Python source file for a carbon-aware scheduling algorithm.

This is NOT a patching or incremental editing task.
This is a FULL REWRITE of the code inside a marked region.

Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. 
Your task is to regenerate the ENTIRE code block between the markers as a COMPLETE and SELF-CONTAINED Python implementation.

━━━━━━━━━━━━━━━━━━━━
STRICT EDIT BOUNDARY
━━━━━━━━━━━━━━━━━━━━

The file contains two markers:

# === BEGIN CARBON-AWARE LOGIC ===
# === END CARBON-AWARE LOGIC ===

You are ONLY allowed to output code that belongs BETWEEN these two markers.

Do NOT output the markers themselves.
Do NOT output any code before or after the marked region.

━━━━━━━━━━━━━━━━━━━━
IMPLEMENTATION CONTEXT
━━━━━━━━━━━━━━━━━━━━

{baseline_context}

【AVAILABLE SYSTEM STATE】
You have access to:
- self.carbon_schedule: Dict mapping timestamps to carbon intensity values
- current_time: Current timestamp
- obs: Observation tuple containing job_dags, frontier_nodes, exec_map, etc.
In get_action(self, obs), the obs tuple unpacks as:
    job_dags, source_job, num_source_exec,
    frontier_nodes, executor_limits,
    exec_commit, moving_executors, action_map, current_time

Valid attributes and accesses:
- self.carbon_schedule[key]  (key is timestamp_ms int)
- self.exec_cap_MAX, self.exec_cap, self.B, self.exec_map
- exec_commit.node_commit[node]  (int)
- moving_executors.count(node)  (int)
- node.job_dag, node.num_tasks, node.next_task_idx
- job_dag.frontier_nodes  (list of ready nodes for that job)
- frontier_nodes  (all ready nodes across jobs, from obs)
Do NOT use self.invoke_model() -- it does not exist.

【REQUIRED FUNCTIONS】
You MUST implement these two functions with the exact signatures:

1. get_carbon_intensity(self, current_time) -> (current_carbon, L, U)
   - Returns current carbon intensity and bounds
   - L and U are used for decision making
   - Must satisfy: L < U

2. get_action(self, obs) -> (node, use_exec, carbon_aware_flag)
   - Makes scheduling decisions
   - node: Selected task to execute (or None to defer)
   - use_exec: Number of executors to allocate
   - carbon_aware_flag: Boolean indicating if decision is carbon-aware

{validation_context}

━━━━━━━━━━━━━━━━━━━━
IMPLEMENTATION REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━

1. COMPLETE FUNCTION IMPLEMENTATION
   - Both functions must be fully implemented
   - Include all necessary logic and return statements
   - Handle edge cases appropriately

2. SELF-CONTAINED CODE
   - Use only class attributes (self.*) and function parameters
   - Do not reference undefined variables
   - Do not leave TODOs or placeholders

3. CORRECT PYTHON SYNTAX
   - Proper indentation (4 spaces per level)
   - Valid Python syntax
   - No syntax errors

4. PRESERVE INTERFACE
   - Function signatures must match exactly
   - Return types must match expected format
   - Must integrate with existing class structure

5. USE AVAILABLE INFORMATION
   - Leverage self.carbon_schedule for carbon data
   - Access obs tuple for system state
   - Do NOT use self.invoke_model()
   - Utilize any other available class attributes

━━━━━━━━━━━━━━━━━━━━
CREATIVE FREEDOM
━━━━━━━━━━━━━━━━━━━━

You have significant freedom in HOW you implement these functions:
- Choose your own approach to carbon prediction
- Design your own decision-making logic
- Use any mathematical or algorithmic techniques
- Leverage any available information creatively

The only constraints are:
- Must satisfy validation rules (if provided)
- Must return correct types
- Must be valid Python code

Think creatively about how to reduce carbon usage while maintaining performance!

━━━━━━━━━━━━━━━━━━━━
ERROR MESSAGE FROM THE PREVIOUS SOLUTION
━━━━━━━━━━━━━━━━━━━━

{change_instruction}

━━━━━━━━━━━━━━━━━━━━
FULL FILE FOR CONTEXT
━━━━━━━━━━━━━━━━━━━━

{code_block}

━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT)
━━━━━━━━━━━━━━━━━━━━

Output ONLY valid Python code. No markdown fences. No explanation.
Your output MUST start with: import numpy as np
Your output MUST include: from agents.agent import Agent
Your output MUST define: class NewAgent(Agent):
Your output ends with the last line of the NewAgent class.
"""

    response = call_llm(prompt)
    log_llm_output("code_enhanced", response)

    return response


def generate_incremental_modifications(
    current_code: str,
    improvement_goal: str,
    baseline_metrics: Optional[Dict] = None
) -> List[Dict]:
    """
    Generate a sequence of small, incremental modifications.
    
    Returns:
        List of modification steps, each containing:
        - step: step number
        - description: what to modify
        - target: which part to modify
        - rationale: why this modification helps
        - validation: how to validate this step
    """
    
    baseline_context = ""
    if baseline_metrics:
        baseline_context = f"""
Current baseline performance:
- Carbon: {baseline_metrics.get('carbon', 'N/A')}
- Time: {baseline_metrics.get('time', 'N/A')}
- JCT: {baseline_metrics.get('jct', 'N/A')}
"""
    
    prompt = f"""
You are a carbon-aware scheduling expert. Your task is to break down an improvement goal into small, incremental, testable modification steps.

{baseline_context}

【IMPROVEMENT GOAL】
{improvement_goal}

【REQUIREMENTS】
1. Generate 2-4 small modification steps
2. Each step should modify only ONE aspect of the algorithm
3. Each step should be independently testable
4. Steps should be ordered from safest to most impactful
5. Each step should have clear expected effects

【OUTPUT FORMAT】
Return a JSON array with the following structure:

[
    {{
        "step": 1,
        "description": "Brief description of what to modify",
        "target": "Which function or logic block to modify (e.g., 'get_carbon_intensity', 'threshold_calculation')",
        "modification": "Specific change to make (e.g., 'Change L coefficient from 1.1 to 1.05')",
        "rationale": "Why this modification should reduce carbon usage",
        "expected_effect": "Predicted impact on carbon/time/jct",
        "validation": "How to verify this step works correctly",
        "risk_level": "low/medium/high"
    }},
    ...
]

【EXAMPLE】
[
    {{
        "step": 1,
        "description": "Reduce pessimism in lower bound estimation",
        "target": "get_carbon_intensity",
        "modification": "Change L calculation from min(future_carbon) * 1.1 to min(future_carbon) * 1.05",
        "rationale": "Current L is too pessimistic, causing algorithm to be overly conservative. Reducing the multiplier allows more aggressive scheduling during low-carbon periods.",
        "expected_effect": "Carbon: -5%, Time: +2%, JCT: +1%",
        "validation": "Check that L < current_carbon < U holds for >80% of time slots",
        "risk_level": "low"
    }}
]

Generate the modification steps now. Return ONLY the JSON array, no other text.
"""
    
    response = call_llm(prompt)
    log_llm_output("incremental_modifications", response)
    
    try:
        import json
        # Try to extract JSON from response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        steps = json.loads(response)
        return steps
    except Exception as e:
        logging.error(f"Failed to parse incremental modifications: {e}")
        return []

