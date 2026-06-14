import os
import logging
from llm import call_llm
from typing import List, Dict, Optional, Any, Tuple
import humanize
from utils import *
from logger import log_llm_output
from aide.utils.response import extract_code, extract_text_up_to_code, wrap_code

class CarbonCodeAgentCodeAgent:
    def __init__(self, metric_keys):
        self.metric_keys = metric_keys

    def generate_pseudocode(self, code: str) -> str:
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
            self,
            baseline_algo: str,
            baseline_metrics: Optional[Dict] = None,
            parent_metrics: Optional[Dict] = None,
            failed_attempts: Optional[List[str]] = None,
            summary_content: str = "",
            unified_guide: str = "",
            simulator_info: str = "",
    ) -> str:
        """
        Generate an improved algorithm description with domain knowledge constraints.

        Args:
            baseline_algo: Baseline algorithm description
            baseline_metrics: Performance metrics of the fifo baseline
            parent_metrics: Performance metrics of the parent solution
            failed_attempts: List of previously failed modification attempts
            summary_content: Content of Summary.md for RAG context (injected at design stage)
            unified_guide: Unified implementation guide from task decomposition (NEW)
        """

        # Build baseline performance context
        baseline_context = ""
        if baseline_metrics:
            metric_info = ''.join(
                [f"- {metric}: {baseline_metrics.get(metric, 'N/A')}\n" for metric in self.metric_keys])
            baseline_context = f"""
【FIFO BASELINE PERFORMANCE】
{metric_info}

NOTE: Due to test data variability, focus on RELATIVE improvements rather than absolute values.

YOUR GOAL: Add carbon-awareness to this pure FIFO baseline to reduce carbon usage while maintaining acceptable job completion time.
"""

        # Build parent performance context
        parent_context = ""
        if parent_metrics:
            metric_info = ''.join([f"- {metric}: {parent_metrics.get(metric, 'N/A')}\n" for metric in self.metric_keys])
            parent_context = f"""
【PARENT SOLUTION PERFORMANCE】
{metric_info}

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

        # NEW: Inject unified implementation guide from task decomposition
        unified_guide_context = ""
        if unified_guide:
            unified_guide_context = f"""
【TASK DECOMPOSITION & IMPLEMENTATION GUIDE】

The following guide provides a detailed, step-by-step implementation strategy
synthesized from similar task decomposition patterns. This guide integrates:
- High-level strategy and approach
- Detailed implementation sequence with transitions
- Key implementation details (state management, decision logic, edge cases)
- Complete implementation checklist

IMPLEMENTATION GUIDE:
{unified_guide}

IMPORTANT: Follow this guide's implementation sequence and design decisions.
The guide provides proven decomposition patterns adapted to this specific task.
Use it as your primary reference for structuring the algorithm.
"""

        simulator_info_context = ""
        if simulator_info:
            simulator_info_context = f"""
【SIMULATOR INFO】

The simulator is designed to test the algorithm's effectiveness and robustness.
It simulates the execution of tasks and provides insights into the algorithm's performance. The simulator is essential for validating your implementation and identifying potential issues or areas for improvement.

SIMULATOR INFO:
{simulator_info}
"""

        prompt = f"""
You are an expert in scheduling and systems optimization.

Your task is to DESIGN an improved algorithm by IMPROVING the given baseline algorithm.

{simulator_info_context}

{baseline_context}

{parent_context}

{failed_context}

【BASELINE ALGORITHM】
{baseline_algo}

{unified_guide_context}

【FORBIDDEN MODIFICATIONS】
1. DO NOT rename the class (must remain NewAgent)
2. DO NOT change the __init__ signature
3. DO NOT violate the critical invariants listed above
4. DO NOT remove core FIFO functionality without replacement

【DESIGN PHILOSOPHY】
- Be creative and think outside the box
- Consider unconventional approaches
- Don't be constrained by the baseline's design choices
- Focus on the fundamental problem
- Use all available information effectively

【DESIGN REQUIREMENTS】
1. The improved algorithm MUST remain compatible with the baseline structure
2. Each modification should have a clear rationale based on

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
"""

        response = call_llm(prompt)
        log_llm_output("algorithm_enhanced", response)

        return response

    def modify_code_block_with_constraints(
            self,
            code_block: str,
            change_instruction: str,
            baseline_metrics: Optional[Dict] = None,
            validation_rules: Optional[List[str]] = None,
            summary_content: str = "",
            unified_guide: str = "",
            simulator_info: str = "",
    ) -> str:
        """
        Modify the marked code region with domain knowledge constraints and validation rules.

        Args:
            code_block: Full code with markers
            change_instruction: High-level algorithm description
            baseline_metrics: Baseline performance metrics (fifo baseline)
            validation_rules: List of validation rules to enforce
            summary_content: Content of Summary.md for RAG context
            unified_guide: Unified implementation guide from task decomposition (NEW)
        """

        # Build baseline context
        baseline_context = ""
        if baseline_metrics:
            metric_info = ''.join(
                [f"- {metric}: {baseline_metrics.get(metric, 'N/A')}\n" for metric in self.metric_keys])
            baseline_context = f"""
【BASELINE PERFORMANCE METRICS】
{metric_info}

YOUR IMPLEMENTATION MUST REDUCE CARBON USAGE vs THIS BASELINE.
"""

        # Build validation rules
        validation_context = ""
        if validation_rules:
            validation_context = f"""
【VALIDATION RULES - YOUR CODE MUST SATISFY】
{chr(10).join(f"{i + 1}. {rule}" for i, rule in enumerate(validation_rules))}
"""

        # NEW: Inject unified implementation guide
        unified_guide_context = ""
        if unified_guide:
            unified_guide_context = f"""
【IMPLEMENTATION GUIDE】

Follow this detailed implementation guide which provides:
- Step-by-step implementation sequence
- Key implementation details and decision logic
- State management and error handling
- Complete implementation checklist

GUIDE:
{unified_guide}

IMPORTANT: Use this guide as your primary reference for implementation structure and logic.
"""
        simulator_info_context = ""
        if simulator_info:
            simulator_info_context = f"""
【SIMULATOR INFO】

The simulator is designed to test the algorithm's effectiveness and robustness.
It simulates the execution of tasks and provides insights into the algorithm's performance. The simulator is essential for validating your implementation and identifying potential issues or areas for improvement.

SIMULATOR INFO:
{simulator_info}
"""

        prompt = f"""
You are rewriting a specific region of an existing Python source file for a scheduling algorithm.

This is NOT a patching or incremental editing task.
This is a FULL REWRITE of the code inside a marked region.

Your task is to regenerate the ENTIRE code block between the markers as a COMPLETE and SELF-CONTAINED Python implementation.

【STRICT EDIT BOUNDARY】

The file contains two markers:

# === BEGIN CARBON-AWARE LOGIC ===
# === END CARBON-AWARE LOGIC ===

You are ONLY allowed to output code that belongs BETWEEN these two markers.

Do NOT output the markers themselves.
Do NOT output any code before or after the marked region.

{simulator_info_context}

{baseline_context}

{validation_context}

【IMPLEMENTATION INSTRUCTION】
{change_instruction}

【PREVIOUS SOLUTION】
{code_block}

{unified_guide_context}

【IMPLEMENTATION REQUIREMENTS】
- Do not reference undefined variables
- Do not leave TODOs or placeholders
- Valid Python syntax
- No syntax errors
- Function signatures must match exactly
- Return types must match expected format
- Leverage self.carbon_schedule for carbon data
- Access obs tuple for system state
- Do NOT use self.invoke_model()
- Utilize any other available class attributes
- Output ONLY valid Python code. No markdown fences. No explanation.
- Your output MUST include: from agents.agent import Agent
- Your output MUST define: class NewAgent(Agent):
- Your output ends with the last line of the NewAgent class.

Think creatively about how to reduce carbon usage while maintaining performance!
"""

        response = call_llm(prompt)
        log_llm_output("code_enhanced", response)

        return response

    def modify_code_block_with_debug_constraints(
            self,
            code_block: str,
            change_instruction: str,
            baseline_metrics: Optional[Dict] = None,
            validation_rules: Optional[List[str]] = None,
            unified_guide: str = "",
            simulator_info: str = "",
    ) -> str:
        """
        Modify the marked code region with domain knowledge constraints and validation rules.
        Used for debugging/fixing failed implementations.

        Args:
            code_block: Full code with markers
            change_instruction: Error message or debug instruction
            baseline_metrics: Baseline performance metrics
            validation_rules: List of validation rules
            unified_guide: Unified implementation guide from task decomposition (NEW)
        """

        # Build unified guide context
        unified_guide_context = ""
        if unified_guide:
            unified_guide_context = f"""
【TASK DECOMPOSITION & IMPLEMENTATION GUIDE】

The following guide provides a detailed, step-by-step implementation strategy based on similar tasks:

IMPLEMENTATION GUIDE:
{unified_guide}

IMPORTANT: Follow this guide's implementation sequence and design decisions when fixing the bug.
"""

        simulator_info_context = ""
        if simulator_info:
            simulator_info_context = f"""
【SIMULATOR INFO】

The simulator is designed to test the algorithm's effectiveness and robustness.
It simulates the execution of tasks and provides insights into the algorithm's performance. The simulator is essential for validating your implementation and identifying potential issues or areas for improvement.
We provide a detailed interaction interface between the algorithm and the simulator, including information such as observable states, executable actions, constraint rules, and so on.

SIMULATOR INFO:
{simulator_info}
"""

        # Build baseline context
        baseline_context = ""
        if baseline_metrics:
            metric_info = ''.join(
                [f"- {metric}: {baseline_metrics.get(metric, 'N/A')}\n" for metric in self.metric_keys])
            baseline_context = f"""
【BASELINE PERFORMANCE METRICS】
{metric_info}

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
You are rewriting a specific region of an existing Python source file for a scheduling algorithm.

This is NOT a patching or incremental editing task.
This is a FULL REWRITE of the code inside a marked region.

Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. 
Your task is to regenerate the ENTIRE code block between the markers as a COMPLETE and SELF-CONTAINED Python implementation.

【STRICT EDIT BOUNDARY】

The file contains two markers:

# === BEGIN CARBON-AWARE LOGIC ===
# === END CARBON-AWARE LOGIC ===

You are ONLY allowed to output code that belongs BETWEEN these two markers.

Do NOT output the markers themselves.
Do NOT output any code before or after the marked region.

{simulator_info_context}

{baseline_context}

{validation_context}

【ERROR MESSAGE FROM THE PREVIOUS SOLUTION】
{change_instruction}

【PREVIOUS SOLUTION】
{code_block}

{unified_guide_context}

【IMPLEMENTATION REQUIREMENTS】
- Do not reference undefined variables
- Do not leave TODOs or placeholders
- Valid Python syntax
- No syntax errors
- Function signatures must match exactly
- Return types must match expected format
- Leverage self.carbon_schedule for carbon data
- Access obs tuple for system state
- Do NOT use self.invoke_model()
- Utilize any other available class attributes
- Output ONLY valid Python code. No markdown fences. No explanation.
- Your output MUST include: from agents.agent import Agent
- Your output MUST define: class NewAgent(Agent):
- Your output ends with the last line of the NewAgent class.
"""

        response = call_llm(prompt)
        log_llm_output("code_enhanced", response)

        return response


class BaseCodeAgent:
    def __init__(self, cfg, metric_keys):
        self.cfg = cfg
        self.acfg = cfg.agent
        self.metric_keys = metric_keys

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response MUST follow this EXACT structure:\n\n"
                "1. First line: PLAN: <SPECIFIC and CONCRETE description of what you're optimizing>\n"
                "2. Second line: REASON: <why this optimization is needed>\n"
                "3. Third line: (blank line)\n"
                "4. Finally: A single markdown code block (wrapped in ```) which implements this solution\n\n"
                "CRITICAL REQUIREMENTS FOR PLAN:\n"
                "- MUST be SPECIFIC and CONCRETE, NOT vague or high-level\n"
                "- MUST describe EXACTLY what was changed (which function, which parameter, which module)\n"
                "- GOOD examples:\n"
                "  * 'Modified process_data() function to use binary search (O(log n)) instead of linear search (O(n))'\n"
                "  * 'Added new CacheManager module with LRU eviction policy (max_size=1000)'\n"
                "  * 'Changed batch_size parameter from fixed 32 to dynamic calculation based on available_memory/item_size'\n"
                "  * 'Introduced priority queue in __init__() that ranks items by (importance_score * urgency_factor)'\n"
                "- BAD examples (DO NOT USE):\n"
                "  * 'Optimize the algorithm' (too vague)\n"
                "  * 'Improve performance' (not specific)\n"
                "  * 'Design better logic' (no concrete details)\n"
                "  * 'Implement caching mechanism' (missing implementation details)\n\n"
                "Example:\n"
                "PLAN: Modified select_action() to use epsilon-greedy strategy (epsilon=0.1) with exponential decay (decay_rate=0.995), added reward prediction using moving average (window_size=100)\n"
                "REASON: Pure greedy selection leads to suboptimal solutions due to lack of exploration\n\n"
                "```python\n"
                "# code here\n"
                "```\n"
            )
        }

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str, str]:
        """
        Generate plan, reason, and code in the same LLM call.
        Returns: (plan, reason, code)
        """
        completion_text = None
        for _ in range(retries):
            completion_text = call_llm(
                prompt=prompt,
            )

            # Extract plan and reason
            plan = ""
            reason = ""
            remaining_text = completion_text

            lines = completion_text.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith("PLAN:"):
                    plan = line.split("PLAN:", 1)[1].strip()
                elif line.strip().startswith("REASON:"):
                    reason = line.split("REASON:", 1)[1].strip()
                    # Remaining text starts after this line
                    remaining_text = '\n'.join(lines[i + 1:])
                    break

            # Extract code from remaining text
            code = extract_code(remaining_text)

            if code and plan and reason:
                return plan, reason, code

            print("Plan + reason + code extraction failed, retrying...")

        print("Final extraction attempt failed, using fallback...")
        # Fallback: try to extract at least code
        code = extract_code(completion_text)
        return "Plan not specified", "Reason not specified", code or completion_text


class CarbonCodeAgent(BaseCodeAgent):

    def _prompt_impl_guideline(
            self,
            unified_guide
    ):
        impl_guideline = [
            "The code should **implement the proposed solution**.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                f"""
【TASK DECOMPOSITION & IMPLEMENTATION GUIDE】

The following guide provides a detailed, step-by-step implementation strategy
synthesized from similar task decomposition patterns. This guide integrates:
- High-level strategy and approach
- Detailed implementation sequence with transitions
- Key implementation details (state management, decision logic, edge cases)
- Complete implementation checklist

IMPLEMENTATION GUIDE:
{unified_guide}

IMPORTANT: Follow this guide's implementation sequence and design decisions.
The guide provides proven decomposition patterns adapted to this specific task.
Use it as your primary reference for structuring the algorithm.
""")

        return {"Implementation guideline": impl_guideline}

    def generate_draft_code(
            self,
            task_desc,
            success_memory,
            unified_guide,
            baseline_code,
            bug_history: str = "",
            experience_summary: str = "",
            data_preview=None,
    ) -> Tuple[str, str, str]:
        # Build bug context
        bug_context = ""
        if bug_history:
            bug_context = f"""
【KNOWN BUGS TO AVOID】
The following bugs have occurred in previous attempts. DO NOT repeat them:

{bug_history}

Make sure your implementation avoids these common pitfalls.
"""

        # Build experience context
        experience_context = ""
        if experience_summary:
            experience_context = f"""
{experience_summary}

Apply these proven solutions when encountering similar patterns.
"""

        prompt: Any = {
            "Introduction": (
                "You are an expert in systems optimization. Your task is to DESIGN an improved algorithm by IMPROVING the given baseline algorithm."
                "In order to win this competition, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            ),
            "Task description": task_desc,
            "Memory": success_memory,
            "Bug History": bug_context,
            "Learned Experiences": experience_context,
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline(unified_guide)
        prompt["Instructions"] |= baseline_code

        if self.acfg.data_preview:
            prompt["Data Overview"] = data_preview

        plan, reason, code = self.plan_and_code_query(prompt)

        return plan, reason, code

    def generate_improve_code(
            self,
            parent_node,
            task_desc,
            success_memory,
            unified_guide,
            bug_history: str = "",
            optimization_path: str = "",
            experience_summary: str = "",
    ) -> Tuple[str, str, str]:
        # Build bug context
        bug_context = ""
        if bug_history:
            bug_context = f"""
【KNOWN BUGS TO AVOID】
{bug_history}
"""

        # Build optimization path context
        path_context = ""
        if optimization_path:
            path_context = f"""
【OPTIMIZATION PATH】
{optimization_path}

Build upon these previous optimizations and propose a new improvement.
"""

        # Build experience context
        experience_context = ""
        if experience_summary:
            experience_context = f"""
{experience_summary}

Apply these proven solutions when encountering similar patterns.
"""

        # Build parent metrics context
        parent_metrics_context = ""
        if hasattr(parent_node, 'metrics') and parent_node.metrics:
            metrics_str = "\n".join([f"  - {key}: {value}" for key, value in parent_node.metrics.items()])
            parent_metrics_context = f"""
【PARENT NODE METRICS】
{metrics_str}

IMPORTANT: Analyze these metrics to identify weaknesses and improvement opportunities.
"""

        prompt: Any = {
            "Introduction": (
                "You are an expert in algorithm optimization. Your task is to IMPROVE the given solution."
                "You should analyze the current solution's weaknesses based on its metrics, "
                "then propose and implement a specific improvement."
            ),
            "Task description": task_desc,
            "Memory": success_memory,
            "Bug History": bug_context,
            "Optimization Path": path_context,
            "Learned Experiences": experience_context,
            "Parent Metrics": parent_metrics_context,
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "REASON Requirements": [
                "REASON must analyze the WEAKNESSES of the current solution based on metrics",
                "Point out SPECIFIC issues: which metrics are suboptimal and why",
                "Explain what aspects of the algorithm cause these issues",
            ],
            "PLAN Requirements": [
                "PLAN must be based on the weaknesses identified in REASON",
                "Provide SPECIFIC modifications to address those weaknesses",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline(unified_guide)

        plan, reason, code = self.plan_and_code_query(prompt)
        return plan, reason, code

    def generate_debug_code(
            self,
            parent_node,
            task_desc,
            unified_guide,
            bug_history: str = "",
            optimization_path: str = "",
            experience_summary: str = "",
            data_preview=None,
    ) -> Tuple[str, str, str]:
        # Build bug context
        bug_context = ""
        if bug_history:
            bug_context = f"""
{bug_history}
"""

        # Build optimization path context
        path_context = ""
        if optimization_path:
            path_context = f"""
【OPTIMIZATION PATH】
{optimization_path}

This context helps you understand what the code was trying to achieve.
"""

        # Build experience context
        experience_context = ""
        if experience_summary:
            experience_context = f"""
{experience_summary}

Apply these proven solutions when encountering similar error patterns.
"""

        # Extract detailed error information
        error_type = parent_node.exc_type or "UnknownError"
        error_message = str(parent_node.exc_info) if parent_node.exc_info else "No error message"

        # Get the specific error location from terminal output
        error_location = ""
        if parent_node.term_out:
            term_lines = parent_node.term_out.split('\n')
            # Find traceback lines
            for i, line in enumerate(term_lines):
                if 'Traceback' in line or 'File "' in line or 'Error' in line or 'line' in line:
                    error_location += line + '\n'

        # Extract the problematic code snippet from the error location
        problematic_code_context = ""
        if parent_node.code and error_location:
            # Try to find line number from error location
            import re
            line_match = re.search(r'line (\d+)', error_location)
            if line_match:
                error_line_num = int(line_match.group(1))
                code_lines = parent_node.code.split('\n')
                # Show context: 5 lines before and after the error line
                start_line = max(0, error_line_num - 6)  # -6 because line numbers start at 1
                end_line = min(len(code_lines), error_line_num + 4)

                problematic_code_context = "\n【PROBLEMATIC CODE CONTEXT】\n"
                problematic_code_context += f"Lines {start_line + 1} to {end_line} (error at line {error_line_num}):\n\n"
                for i in range(start_line, end_line):
                    marker = ">>> " if i == error_line_num - 1 else "    "
                    problematic_code_context += f"{marker}{i + 1:4d} | {code_lines[i]}\n"

        error_analysis = f"""
【DETAILED ERROR ANALYSIS】

Error Type: {error_type}
Error Message: {error_message}

Error Location and Traceback:
{error_location if error_location else "See execution output below"}
{problematic_code_context}

CRITICAL DEBUGGING STEPS:
1. Identify the ROOT CAUSE (not just the symptom)
   - If AttributeError: The object doesn't have this attribute. Don't guess other names.
   - If AssertionError: The computation produces invalid values. Don't force values to be valid.
   - If KeyError/TypeError: Wrong data structure type. Don't assume the type.

2. Check FORBIDDEN approaches in bug history
   - If similar fixes have failed before, DO NOT try variations

3. Add DIAGNOSTIC code first (if needed)
   - For API errors: Add print(dir(object)) to see actual attributes
   - For value errors: Add print statements to trace computation
   - For type errors: Add print(type(variable)) to check actual type

4. Only then write the fix based on UNDERSTANDING, not guessing

IMPORTANT: The "Previous (buggy) implementation" section below contains the FULL CODE that produced this error.
Read it carefully to understand the context and logic flow.
"""

        prompt: Any = {
            "Introduction": (
                "You are an expert in debugging and root cause analysis. "
                "Your previous solution had a bug. You must identify the ROOT CAUSE and fix it properly.\n\n"
                "CRITICAL PRINCIPLES:\n"
                "1. Understand the ROOT CAUSE before fixing (don't guess)\n"
                "2. Don't repeat failed approaches from bug history\n"
                "3. Don't force values to be valid without understanding why they're invalid\n"
                "4. Don't guess API names - if attributes don't exist, the API is different than assumed\n"
                "5. Add diagnostic code if needed to understand the problem\n\n"
                "IMPORTANT: If errors indicate missing attributes or methods, it means the framework "
                "does NOT have them. Do NOT try variations. "
                "Instead, you must find the ACTUAL API or use a completely different approach."
            ),
            "Task description": task_desc,
            "Error Analysis": error_analysis,
            "Bug History": bug_context,
            "Optimization Path": path_context,
            "Learned Experiences": experience_context,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Full execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "REASON Requirements": [
                "REASON must identify the ROOT CAUSE (not just describe the symptom)",
                "Explain WHY the error occurred (what assumption was wrong)",
                "Reference the bug history if this is a repeated error pattern",
            ],
            "PLAN Requirements": [
                "PLAN must be based on ROOT CAUSE understanding, not symptom treatment",
                "If the ROOT CAUSE is unknown API: Add diagnostic code (print statements) OR use alternative approach",
                "If the ROOT CAUSE is logic error: Trace the computation, don't force values",
                "DO NOT repeat approaches from FORBIDDEN list in bug history",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline(unified_guide)

        if self.acfg.data_preview:
            prompt["Data Overview"] = data_preview

        plan, reason, code = self.plan_and_code_query(prompt)
        return plan, reason, code
