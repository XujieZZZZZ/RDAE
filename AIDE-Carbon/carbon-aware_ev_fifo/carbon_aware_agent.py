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
from pathlib import Path

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from aide.agent import Agent, ExecCallbackType
from aide.backend import query, FunctionSpec
from aide.interpreter import ExecutionResult
from aide.journal import Node
from aide.utils.response import extract_code, extract_text_up_to_code, wrap_code

from generate_enhanced import (
    generate_pseudocode,
    generate_algorithm_with_constraints,
    modify_code_block_with_constraints,
    modify_code_block_with_debug_constraints,
)

from utils import load_code_file, extract_edit_block, apply_modified_block
from adaptive_search import AdaptiveSearchPolicy, classify_modification_type
from multi_objective_metric import MultiObjectiveMetric, compute_pareto_front, select_from_pareto_front

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
        cfg,
        journal,
        baseline_code_path: str,
        optimization_paths: list[str] | None = None,
    ):
        """
        Initialize Carbon-Aware Agent.

        Args:
            task_desc: Task description
            cfg: Configuration object
            journal: Journal instance
            baseline_code_path: Path to baseline FIFO algorithm code (fifo_agent.py)
            optimization_paths: List of optimization paths to explore
        """

        super().__init__(task_desc, cfg, journal)
        self.baseline_code_path = Path(baseline_code_path)
        self.optimization_paths = optimization_paths or ["carbon_threshold_scheduling", "valley_forecasting", "dag_aware_carbon_scheduling"]
        
        # Load baseline code
        self.baseline_code = load_code_file(str(self.baseline_code_path))
        self.baseline_pseudocode = None

        # Track current optimization path
        self.current_path = None
        self.path_history = {}  # Track nodes by path

        # Initialize adaptive search policy
        self.search_policy_adaptive = AdaptiveSearchPolicy(exploration_weight=0.2)

        # Store baseline metrics for reference
        self.baseline_metrics = None

        # Initialize data_preview to empty string if disabled to avoid encoding issues
        if not self.acfg.data_preview:
            self.data_preview = ""

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
        2. Debugging buggy nodes
        3. UCB-based node selection (balances exploration/exploitation)
        """
        import random
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.debug("[search policy] drafting new node (not enough drafts)")
            return None
        
        # Debugging: prioritize buggy nodes
        if random.random() < search_cfg.debug_prob:
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                logger.debug("[search policy] debugging")
                return random.choice(debuggable_nodes)
            logger.debug("[search policy] not debugging by chance")

        # Check if we should explore (draft new) or exploit (improve existing)
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            # If everything is buggy, force debugging
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                logger.debug("[search policy] forcing debugging (no good nodes)")
                return random.choice(debuggable_nodes)
            logger.debug("[search policy] drafting new node (no good nodes and nothing to debug)")
            return None
        
        # Use UCB-based selection for exploitation
        selected_node = self.search_policy_adaptive.select_node_ucb(self.journal)
        if selected_node:
            logger.debug(f"[search policy] UCB selected node {selected_node.id}")
            return selected_node
        
        # Fallback: greedy best node
        greedy_node = self.journal.get_best_node()
        logger.debug("[search policy] greedy node selected (fallback)")
        return greedy_node
    
    def _draft(self) -> Node:
        """Generate initial solution with enhanced domain knowledge constraints."""
        # Generate pseudocode from baseline if not done
        block, span = extract_edit_block(self.baseline_code)
        self.baseline_pseudocode = generate_pseudocode(block)

        # Get failed attempts for context
        failed_attempts = self.search_policy_adaptive.get_failed_attempts(max_recent=3)

        # Generate improved algorithm with constraints
        improved_algo = generate_algorithm_with_constraints(
            baseline_algo=self.baseline_pseudocode,
            baseline_metrics=self.baseline_metrics,
            parent_metrics=None,
            failed_attempts=failed_attempts,
            # summary_content=self.summary_content,
            # task_desc_sections=self.task_desc_sections,
        )

        # Validation rules for code generation
        validation_rules = [
            "class must be named NewAgent (do NOT rename)",
            "self.B <= self.exec_cap <= self.exec_cap_MAX must hold at all times",
            "get_action(self, obs) must return a 3-tuple: (node, use_exec, carbon_aware_flag)",
            "do NOT modify __init__ signature or self.exec_map logic outside the marked region",
            "if get_carbon_intensity is implemented: must return (current_carbon, L, U) with L < U",
        ]

        # Modify code block with constraints
        modified_block = modify_code_block_with_constraints(
            code_block=self.baseline_code,
            change_instruction=improved_algo,
            baseline_metrics=self.baseline_metrics,
            validation_rules=validation_rules,
            # task_desc_sections=self.task_desc_sections,
            # summary_content=summary_context,
        )
        new_code = apply_modified_block(self.baseline_code, span, modified_block)
        new_code = new_code.replace("```python", "").replace("```", "")

        # Create plan description
        plan = f"Improved algorithm:\n{improved_algo}"

        node = Node(plan=plan, code=new_code)

        return node

    def _improve(self, parent_node: Node) -> Node:
        """Improve existing solution with enhanced constraints and parent metrics."""
        block, span = extract_edit_block(parent_node.code)
        current_pseudocode = generate_pseudocode(block)

        # Extract parent metrics if available
        parent_metrics = None
        if hasattr(parent_node, 'delta_vs_cap_decima') and parent_node.delta_vs_cap_decima:
            # Try to get actual metrics from scheme_results
            if hasattr(parent_node, 'metrics_payload') and parent_node.metrics_payload:
                scheme_results = parent_node.metrics_payload.get('scheme_results', [])
                newcap_result = next((r for r in scheme_results if r['scheme'] == 'newcap'), None)
                if newcap_result:
                    parent_metrics = {
                        'carbon': newcap_result.get('total_carbon'),
                        'time': newcap_result.get('total_time'),
                        'jct': newcap_result.get('avg_job_completion')
                    }

        # Get failed attempts for context
        failed_attempts = self.search_policy_adaptive.get_failed_attempts(max_recent=3)

        # Generate improved algorithm with constraints
        improved_algo = generate_algorithm_with_constraints(
            baseline_algo=current_pseudocode,
            baseline_metrics=self.baseline_metrics,
            parent_metrics=parent_metrics,
            failed_attempts=failed_attempts,
            # summary_content=self.summary_content,
            # task_desc_sections=self.task_desc_sections,
        )

        # Validation rules for code generation
        validation_rules = [
            "class must be named NewAgent (do NOT rename)",
            "self.B <= self.exec_cap <= self.exec_cap_MAX must hold at all times",
            "get_action(self, obs) must return a 3-tuple: (node, use_exec, carbon_aware_flag)",
            "do NOT modify __init__ signature or self.exec_map logic outside the marked region",
            "if get_carbon_intensity is implemented: must return (current_carbon, L, U) with L < U",
        ]

        # Modify code block with constraints
        modified_block = modify_code_block_with_constraints(
            code_block=parent_node.code,
            change_instruction=improved_algo,
            baseline_metrics=self.baseline_metrics,
            validation_rules=validation_rules,
            # task_desc_sections=self.task_desc_sections,
            # summary_content=summary_context,
        )
        new_code = apply_modified_block(parent_node.code, span, modified_block)
        new_code = new_code.replace("```python", "").replace("```", "")

        # Create plan description
        plan = f"Improved algorithm:\n{improved_algo}"

        node = Node(plan=plan, code=new_code, parent=parent_node)

        return node

    def _debug(self, parent_node: Node) -> Node:
        """Debug a buggy solution."""
        change=parent_node.analysis

        validation_rules = [
            "class must be named NewAgent (do NOT rename)",
            "self.B <= self.exec_cap <= self.exec_cap_MAX must hold at all times",
            "get_action(self, obs) must return a 3-tuple: (node, use_exec, carbon_aware_flag)",
            "do NOT modify __init__ signature or self.exec_map logic outside the marked region",
            "if get_carbon_intensity is implemented: must return (current_carbon, L, U) with L < U",
        ]

        # Apply modifications similar to _improve
        block, span = extract_edit_block(parent_node.code)
        # modified_block = modify_code_block(parent_node.code, plan)
        modified_block = modify_code_block_with_debug_constraints(
            code_block=parent_node.code,
            change_instruction=change,
            baseline_metrics=self.baseline_metrics,
            validation_rules=validation_rules,
            # task_desc_sections=self.task_desc_sections,
        )
        final_code = apply_modified_block(parent_node.code, span, modified_block)
        final_code = final_code.replace("```python", "").replace("```", "")

        plan = f"revision suggestion:\n{change}"

        node = Node(plan=plan, code=final_code, parent=parent_node)

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

                    delta = (payload or {}).get("delta_vs_fifo") or {}
                    node.delta_vs_fifo = delta

                    pct_time = delta.get("total_time_pct")
                    pct_carbon = delta.get("total_carbon_pct")
                    pct_jct = delta.get("avg_job_completion_pct")

                    node.is_buggy = (
                        node.exc_type is not None
                        or pct_time is None
                        or pct_carbon is None
                        or pct_jct is None
                    )

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
                        fifo_result = next((r for r in scheme_results if r['scheme'] == 'fifo'), None)

                        if newalgo_result:
                            # Store actual metrics
                            node.actual_time = newalgo_result.get('total_time')
                            node.actual_carbon = newalgo_result.get('total_carbon')
                            node.actual_jct = newalgo_result.get('avg_job_completion')

                            # Create multi-objective metric
                            node.multi_objective_metric = MultiObjectiveMetric(
                                time=node.actual_time,
                                carbon=node.actual_carbon,
                                jct=node.actual_jct,
                                maximize=False
                            )
                        
                        # Store baseline metrics for future reference (first time only, using fifo)
                        if self.baseline_metrics is None and fifo_result:
                            self.baseline_metrics = {
                                'carbon': fifo_result.get('total_carbon'),
                                'time': fifo_result.get('total_time'),
                                'jct': fifo_result.get('avg_job_completion')
                            }
                            logger.info(f"Stored baseline metrics: {self.baseline_metrics}")

                        # Use weighted % improvement as primary metric (for compatibility)
                        from aide.utils.metric import MetricValue
                        w_time, w_carbon, w_jct = 0.4, 0.6, 0.0
                        score = (
                            w_time * float(pct_time)
                            + w_carbon * float(pct_carbon)
                            + w_jct * float(pct_jct)
                        )
                        
                        node.metric = MetricValue(score, maximize=True)
                        node.analysis = (
                            f"Metric = weighted % improvement of newalgo vs fifo. "
                            f"time={pct_time:.3f}%, carbon={pct_carbon:.3f}%, jct={pct_jct:.3f}%, "
                            f"score={score:.3f} (w_time={w_time}, w_carbon={w_carbon}, w_jct={w_jct})."
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
            or response.get("total_time") is None
        )

        # Store metrics
        node.total_time = response.get("total_time")
        node.total_carbon = response.get("total_carbon")
        node.avg_job_completion = response.get("avg_job_completion")

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
                pct_time = ((-node.total_time + self.baseline_metrics['time']) /
                           self.baseline_metrics['time'] * 100) if node.total_time else 0
                pct_carbon = ((-node.total_carbon + self.baseline_metrics['carbon']) /
                             self.baseline_metrics['carbon'] * 100) if node.total_carbon else 0
                pct_jct = ((-node.avg_job_completion + self.baseline_metrics['jct']) /
                          self.baseline_metrics['jct'] * 100) if node.avg_job_completion else 0
                
                # Use weighted % improvement (same as AIDE_METRICS_JSON branch)
                w_time, w_carbon, w_jct = 0.4, 0.6, 0.0
                composite_score = (
                    w_time * pct_time +
                    w_carbon * pct_carbon +
                    w_jct * pct_jct
                )
                
                logger.info(f"LLM fallback metrics: time={pct_time:.3f}%, carbon={pct_carbon:.3f}%, "
                           f"jct={pct_jct:.3f}%, score={composite_score:.3f}")
                
            # Use maximize=True (higher % improvement = better)
            # Note: negative percentages mean improvement, so we want to maximize negative values
            node.metric = MetricValue(composite_score, maximize=True)