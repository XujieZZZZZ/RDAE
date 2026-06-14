"""
Carbon-Aware Agent for AIDE framework adaptation.
This agent extends the base Agent to support:
1. Code improvement based on existing baseline algorithm
2. RAG retrieval for relevant papers
3. Multi-path exploration (priority constraints, preference constraints, etc.)
"""
import logging
import sys
from typing import Any, Callable, cast
import json
import os
import humanize
from pathlib import Path

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from aide.agent import Agent, ExecCallbackType
from aide.backend import query, FunctionSpec
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.utils.config import Config
from aide.utils.response import extract_code, extract_text_up_to_code, wrap_code

from utils import load_code_file
from adaptive_search import AdaptiveSearchPolicy, classify_modification_type
from multi_objective_metric import MultiObjectiveMetric
from code_agents.code_agent import CarbonCodeAgent
from llm import call_llm

logger = logging.getLogger("aide")


review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "if there is a bug, propose a fix. Otherwise, write a short summary describing the empirical findings.",
            },
            "total_time": {
                "type": "number",
                "description": "Total execution time of newcap scheduler from test.py output. Null if buggy.",
            },
            "total_carbon": {
                "type": "number",
                "description": "Total carbon usage of newcap scheduler from test.py output. Null if buggy.",
            },
            "avg_job_completion": {
                "type": "number",
                "description": "Average job completion time of newcap scheduler from test.py output. Null if buggy.",
            },
            "cap_decima_total_time": {
                "type": "number",
                "description": "Total execution time of cap_decima (baseline) from test.py output. Null if not available.",
            },
            "cap_decima_total_carbon": {
                "type": "number",
                "description": "Total carbon usage of cap_decima (baseline) from test.py output. Null if not available.",
            },
            "cap_decima_avg_job_completion": {
                "type": "number",
                "description": "Average job completion time of cap_decima (baseline) from test.py output. Null if not available.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if lower metric values are better (we want to minimize), false if higher is better.",
            },
        },
        "required": ["is_bug", "summary", "total_time", "total_carbon", "avg_job_completion", "lower_is_better"],
    },
    description="Submit a review evaluating the output of the carbon-aware scheduler test.",
)


class CarbonAwareAgent(Agent):
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        baseline_code_path: str,
        task_solution_agent
    ):
        """
        Initialize Carbon-Aware Agent.

        Args:
            task_desc: Task description
            cfg: Configuration object
            journal: Journal instance
            baseline_code_path: Path to baseline FIFO algorithm code (fifo_agent.py)
        """
        # generate solutions
        self.task_solution_agent = task_solution_agent
        self.solutions = self.task_solution_agent.run(cfg.workspace_dir)

        super().__init__(task_desc, cfg, journal)
        self.metric_keys = cfg.metrics.metric_keys
        self.metric_weights = {key: weight for key, weight in zip(self.metric_keys, cfg.metrics.metric_weights)}
        self.baseline_name = cfg.metrics.baseline_name

        self.baseline_code_path = Path(baseline_code_path)

        self.code_agent = CarbonCodeAgent(cfg, self.metric_keys)

        # Load baseline code
        self.baseline_code = load_code_file(str(self.baseline_code_path))
        self.baseline_pseudocode = None

        # Initialize adaptive search policy
        self.search_policy_adaptive = AdaptiveSearchPolicy(exploration_weight=1.0)

        # Store baseline metrics for reference
        self.baseline_metrics = None

        # Initialize data_preview to empty string if disabled to avoid encoding issues
        if not self.acfg.data_preview:
            self.data_preview = ""

        # Initialize BugTracker and OptimizationPathTracker
        from bug_tracker import BugTracker
        from optimization_path_tracker import OptimizationPathTracker
        from experience_tracker import ExperienceTracker
        self.bug_tracker = BugTracker(workspace_dir=cfg.workspace_dir)
        self.path_tracker = OptimizationPathTracker(workspace_dir=cfg.workspace_dir, task_desc=task_desc)
        self.experience_tracker = ExperienceTracker(workspace_dir=cfg.workspace_dir)

        logger.info("Initialized BugTracker, OptimizationPathTracker, and ExperienceTracker")

    def update_data_preview(self):
        """
        Override to skip data preview generation for carbon-aware optimization.
        Data preview is not applicable since we're improving code, not working with datasets.
        This prevents encoding errors when reading files in the workspace directory.
        """
        # If data_preview is disabled, keep it as empty string
        if not self.acfg.data_preview:
            self.data_preview = ""
            return
        # If data_preview is enabled, call parent method
        super().update_data_preview()

    def step(self, exec_callback: ExecCallbackType):
        """
        Override step() to add high-signal logs showing whether we are drafting/debugging/improving,
        and which parent node a new node evolves from.
        """
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")

        if parent_node is None:
            logger.info("[search] stage=draft parent=None")
            result_node = self._draft()
        elif parent_node.is_buggy:
            logger.info(f"[search] stage=debug parent_step={parent_node.step} parent_id={parent_node.id}")
            result_node = self._debug(parent_node)
        else:
            logger.info(f"[search] stage=improve parent_step={parent_node.step} parent_id={parent_node.id}")
            result_node = self._improve(parent_node)

        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        self.journal.append(result_node)

    def search_policy(self) -> Node | None:
        """
        Enhanced search policy with UCB-based exploration.
        Selects nodes to work on, prioritizing:
        1. Initial drafts for each optimization path
        2. UCB-based node selection (balances exploration/exploitation, includes both good and buggy nodes)
        """
        import random
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.debug("[search policy] drafting new node (not enough drafts)")
            return None

        # Check if we have any nodes to work with
        all_candidate_nodes = self.journal.good_nodes + [
            n for n in self.journal.buggy_nodes
            if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
        ]

        if not all_candidate_nodes:
            logger.debug("[search policy] drafting new node (no candidate nodes)")
            return None

        # Use UCB-based selection for all nodes (good + debuggable buggy)
        selected_node = self.search_policy_adaptive.select_node_ucb(
            self.journal,
            max_debug_depth=search_cfg.max_debug_depth
        )
        if selected_node:
            logger.debug(f"[search policy] UCB selected node {selected_node.id} (buggy={selected_node.is_buggy})")
            return selected_node

        # Fallback: greedy best node (prefer good nodes, but allow buggy if no good nodes)
        if self.journal.good_nodes:
            greedy_node = self.journal.get_best_node()
            logger.debug("[search policy] greedy good node selected (fallback)")
        else:
            debuggable_nodes = [
                n for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            greedy_node = random.choice(debuggable_nodes) if debuggable_nodes else None
            logger.debug("[search policy] random buggy node selected (fallback)")

        return greedy_node

    def _prompt_base_code(self):
        basecodes_prompt = {
            "Base Code":f"The basic code for implementing this problem is as follows: \n\n{self.baseline_code}"
        }
        
        return basecodes_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "The code should **implement the proposed solution**.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "IMPORTANT: The marked region includes import statements AND the full class definition."
                "Your output must start with import lines and include the complete NewAgent class."
                "MANDATORY OUTPUT STRUCTURE (follow exactly):"
                "    import numpy as np"
                "    from agents.agent import Agent"
                "    # add other imports as needed"
                "    class NewAgent(Agent):             # MUST be named NewAgent, MUST inherit Agent"
                "        def __init__(self, exec_cap, carbon_schedule, exec_lower_bound=20):"
                "            Agent.__init__(self)       # MUST call parent __init__"
                "            self.exec_map = {{}}         # MUST include"
                "            self.carbon_schedule = carbon_schedule"
                "            self.exec_cap_MAX = exec_cap   # MUST include"
                "            self.exec_cap = exec_cap"
                "            self.B = exec_lower_bound      # MUST include"
                "            # your additional state here"
                "        def get_carbon_intensity(self, current_time):  # recommended"
                "            # returns (current_carbon, L, U) with L < U"
                "            ..."
                "        def get_action(self, obs):     # MUST implement"
                "            # returns (node, use_exec, carbon_aware_flag)"
                "            ..."
                "HARD RULES (any violation = immediate test failure):"
                "1. First line of output MUST be an import (e.g. import numpy as np)"
                "2. Class MUST be named exactly `NewAgent`"
                "3. MUST inherit Agent: `class NewAgent(Agent):`"
                "4. MUST include `from agents.agent import Agent`"
                "5. get_action MUST return exactly a 3-tuple: (node, use_exec, carbon_aware_flag)"
                "6. No undefined variables, no TODOs, no placeholders"
                "The prediction function should be well-documented, especially the function signature."
            )

        return {"Implementation guideline": impl_guideline}

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
            # completion_text = query(
            #     system_message=prompt,
            #     user_message=None,
            #     model=self.acfg.code.model,
            #     temperature=self.acfg.code.temp,
            # )

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
                    remaining_text = '\n'.join(lines[i+1:])
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

    def _draft(self) -> Node:
        success_memory = self.journal.generate_summary()
        unified_guide=self.solutions
        baseline_code = self._prompt_base_code()

        # Get bug history and experiences
        bug_history = self.bug_tracker.get_bug_summary(max_bugs=10)
        experience_summary = self.experience_tracker.get_experience_summary(max_experiences=10)

        plan, reason, code = self.code_agent.generate_draft_code(
            self.task_desc,
            success_memory=success_memory,
            unified_guide=unified_guide,
            baseline_code=baseline_code,
            bug_history=bug_history,
            experience_summary=experience_summary,
        )

        # Create draft node
        node = Node(plan=plan, code=code)

        # Create draft edge (parent is None for root)
        # Use plan as optimization_point and reason as optimization_reason
        self.path_tracker.create_edge(
            parent_node=None,
            child_node=node,
            edge_type="draft",
            optimization_point=plan,
            optimization_reason=reason,
        )

        return node

    def _improve(self, parent_node: Node) -> Node:
        success_memory = self.journal.generate_summary()
        unified_guide=self.solutions

        # Get bug history, optimization path, and experiences
        bug_history = self.bug_tracker.get_bug_summary(max_bugs=10)
        optimization_path = self.path_tracker.get_path_summary_for_llm(parent_node, max_edges=10)
        experience_summary = self.experience_tracker.get_experience_summary(max_experiences=10)

        plan, reason, code = self.code_agent.generate_improve_code(
            parent_node=parent_node,
            task_desc=self.task_desc,
            success_memory=success_memory,
            unified_guide=unified_guide,
            bug_history=bug_history,
            optimization_path=optimization_path,
            experience_summary=experience_summary,
        )

        node = Node(plan=plan, code=code, parent=parent_node)

        # Create improve edge
        # Use plan as optimization_point and reason as optimization_reason
        self.path_tracker.create_edge(
            parent_node=parent_node,
            child_node=node,
            edge_type="improve",
            optimization_point=plan,
            optimization_reason=reason,
        )

        return node

    def _debug(self, parent_node: Node) -> Node:
        # Get bug history, optimization path, and experiences
        bug_history = self.bug_tracker.get_bug_summary(max_bugs=10)
        optimization_path = self.path_tracker.get_path_summary_for_llm(parent_node, max_edges=10)
        experience_summary = self.experience_tracker.get_experience_summary(max_experiences=10)

        # Use code_agent to generate debug code
        plan, reason, code = self.code_agent.generate_debug_code(
            parent_node=parent_node,
            task_desc=self.task_desc,
            unified_guide=self.solutions,
            bug_history=bug_history,
            optimization_path=optimization_path,
            experience_summary=experience_summary,
        )

        node = Node(
            plan=plan,
            code=code,
            parent=parent_node,
        )

        # Create debug edge
        # Use plan as optimization_point and reason as optimization_reason
        self.path_tracker.create_edge(
            parent_node=parent_node,
            child_node=node,
            edge_type="debug",
            optimization_point=plan,
            optimization_reason=reason,
        )

        return node

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult):
        """Parse execution results with multi-objective metrics and learning."""
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        # Preferred: parse machine-readable metrics emitted by simulator/test.py
        try:
            raw_lines = node._term_out or []
            for line in reversed(raw_lines):
                if isinstance(line, str) and line.startswith("AIDE_METRICS_JSON="):
                    payload = json.loads(line.split("AIDE_METRICS_JSON=", 1)[1])
                    # Persist raw payload for HTML rendering
                    node.metrics_payload = payload

                    delta = (payload or {}).get("delta_vs_baseline") or {}
                    node.delta_vs_baseline = delta

                    node.is_buggy = (node.exc_type is not None or any(delta.get(f"{key}_pct", None) is None for key in self.metric_keys))

                    if node.is_buggy:
                        _newalgo_error_types = (
                            "NewAgentRuntimeError",
                            "NewAgentLogicDeadlock",
                            "TestExecutionError",
                        )
                        if node.exc_type in _newalgo_error_types:
                            raise ValueError(
                                f"Routing to LLM fallback: exc_type={node.exc_type}. "
                                "Error details already in node.term_out."
                            )
                        from aide.utils.metric import WorstMetricValue
                        node.metric = WorstMetricValue()
                        node.analysis = (
                            f"exc_type={node.exc_type}: "
                            "AIDE_METRICS_JSON parsed but metrics are missing. "
                            "Check NewAgent.get_action() for runtime errors or infinite loops."
                        )
                    else:
                            # Extract actual metrics for multi-objective tracking
                        scheme_results = payload.get('scheme_results', [])
                        newalgo_result = next((r for r in scheme_results if r['scheme'] == 'newalgo'), None)
                        baseline_result = next((r for r in scheme_results if r['scheme'] == self.baseline_name), None)

                        if newalgo_result:
                            # Store actual metrics
                            node.metrics = {
                                key: newalgo_result.get(key) for key in self.metric_keys
                            }

                            # Create multi-objective metric
                            node.multi_objective_metric = MultiObjectiveMetric(
                                objective_keys=self.metric_keys,
                                objectives=node.metrics,
                                weights=self.metric_weights,
                                maximize=False
                            )
                        
                        # Store baseline metrics for future reference (first time only, using fifo)
                        if self.baseline_metrics is None and baseline_result:
                            self.baseline_metrics = {
                                key: newalgo_result.get(key) for key in self.metric_keys
                            }
                            logger.info(f"Stored baseline metrics: {self.baseline_metrics}")

                        # Use weighted % improvement as primary metric (for compatibility)
                        from aide.utils.metric import MetricValue
                        score = sum([self.metric_weights.get(key, 0) * delta.get(f"{key}_pct", 0) for key in self.metric_keys])
                        
                        node.metric = MetricValue(score, maximize=True)
                        node.analysis = (
                            f"Metric = weighted % improvement of newalgo vs fifo. " + " ".join([f"{key}_pct={delta.get(f'{key}_pct', 0):.3f}%" for key in self.metric_keys]) + f"score={score:.3f} metric_weights=metric_weights."
                        )

                        # Learn from this modification
                        if node.parent:
                            mod_type = classify_modification_type(node.parent.code, node.code)
                            success = score > 0  # Positive score = improvement
                            self.search_policy_adaptive.record_modification_result(
                                parent=node.parent,
                                child=node,
                                modification_type=mod_type,
                                success=success
                            )
                            logger.info(f"Recorded modification type '{mod_type}' with success={success}")

                    # Record bug or analyze optimization result
                    if node.is_buggy:
                        # Check if this is a debug attempt
                        if node.parent and node.parent.is_buggy:
                            # This is a failed debug attempt
                            self.bug_tracker.record_fix_attempt(
                                node=node,
                                parent_bug_node=node.parent,
                                success=False
                            )
                        # Record the new bug
                        self.bug_tracker.record_bug(node)
                    else:
                        # Check if this is a successful debug
                        if node.parent and node.parent.is_buggy:
                            # This is a successful debug attempt
                            self.bug_tracker.record_fix_attempt(
                                node=node,
                                parent_bug_node=node.parent,
                                success=True
                            )
                        # Record edge result and analyze optimization
                        self.path_tracker.record_edge_result(node)
                        self.path_tracker.analyze_optimization(node)

                    return
        except Exception as e:
            logger.warning(f"Failed to parse AIDE_METRICS_JSON: {e}")
            # Fall back to LLM-based parsing below
            pass

        prompt = {
            "Introduction": (
                "You are an expert in carbon-aware scheduling algorithms. "
                "You have written code to test a carbon-aware scheduler and now need to evaluate the output. "
                "You should determine if there were any bugs as well as report the empirical findings, "
                "including total execution time, total carbon usage, and average job completion time."
            ),
            "Task description": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        logger.info(f"Node.term_out: {node.term_out}")

        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=review_func_spec,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
            ),
        )

        node.analysis = response["summary"]
        node.is_buggy = (
            response["is_bug"]
            or node.exc_type is not None
            or response.get(self.metric_keys[0]) is None
        )

        # Store metrics
        node.metrics = {
            key: response.get(key) for key in self.metric_keys
        }

        if node.is_buggy:
            from aide.utils.metric import WorstMetricValue
            node.metric = WorstMetricValue()
        else:
            # Calculate percentage improvement vs baseline (consistent with AIDE_METRICS_JSON branch)
            from aide.utils.metric import MetricValue
            
            if self.baseline_metrics is None:
                # If no baseline yet, use a neutral score
                logger.warning("No baseline metrics available for LLM fallback, using neutral score")
                composite_score = 0.0
            else:
                # Calculate percentage improvement for each metric
                # Negative percentage = improvement (lower is better)
                metrics_pct = {
                    "{key}_pct": ((-node.metrics[key] + self.baseline_metrics[key]) / self.baseline_metrics[key] * 100) if node.metrics.get(key) else 0 for key in self.metric_keys
                }

                # Use weighted % improvement (same as AIDE_METRICS_JSON branch)
                composite_score = sum([self.metric_weights.get(key, 0) * metrics_pct.get(f"{key}_pct", 0) for key in self.metric_keys])
                
                logger.info(f"LLM fallback metrics: " + " ".join([f"{key}_pct={delta.get(f'{key}_pct', 0):.3f}%" for key in self.metric_keys]) + f"score={composite_score:.3f}")
                
            # Use maximize=True (higher % improvement = better)
            # Note: negative percentages mean improvement, so we want to maximize negative values
            node.metric = MetricValue(composite_score, maximize=True)

        # Record bug or analyze optimization result (for LLM fallback branch)
        if node.is_buggy:
            # Check if this is a debug attempt
            if node.parent and node.parent.is_buggy:
                # This is a failed debug attempt
                self.bug_tracker.record_fix_attempt(
                    node=node,
                    parent_bug_node=node.parent,
                    success=False
                )
            # Record the new bug
            self.bug_tracker.record_bug(node)
        else:
            # Check if this is a successful debug
            if node.parent and node.parent.is_buggy:
                # This is a successful debug attempt
                self.bug_tracker.record_fix_attempt(
                    node=node,
                    parent_bug_node=node.parent,
                    success=True
                )
            # Record edge result and analyze optimization
            self.path_tracker.record_edge_result(node)
            self.path_tracker.analyze_optimization(node)
