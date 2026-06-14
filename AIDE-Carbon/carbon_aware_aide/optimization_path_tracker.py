"""
Optimization Path Tracker for AIDE-Carbon
Tracks optimization paths to help LLM understand the evolution of solutions.
Enhanced version: tracks ALL edges (draft, improve, debug) with detailed analysis.
"""
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional
from aide.journal import Node, Edge
from llm import call_llm

logger = logging.getLogger("aide")


class OptimizationPathTracker:
    """Enhanced optimization path tracking system"""

    def __init__(self, workspace_dir: Path, task_desc: str = ""):
        self.workspace_dir = Path(workspace_dir)
        self.path_file = self.workspace_dir / "optimization_path.json"
        self.edges: Dict[str, Edge] = {}  # edge_id -> Edge
        self.task_desc = task_desc  # Store task_desc once at initialization
        self._load_history()

    def create_edge(self, parent_node: Node, child_node: Node,
                   edge_type: str, optimization_point: str,
                   optimization_reason: str) -> Edge:
        """
        Create an edge between parent and child nodes
        Also generates merged memory from root to this node

        Args:
            parent_node: Parent node (can be None for root/draft)
            child_node: Child node
            edge_type: "draft", "improve", or "debug"
            optimization_point: What was optimized (PLAN)
            optimization_reason: Why this optimization was made (REASON)
        """
        edge = Edge(
            parent_node_id=parent_node.id if parent_node else "root",
            child_node_id=child_node.id,
            edge_type=edge_type,
            optimization_point=optimization_point,
            optimization_reason=optimization_reason,
            task_desc="",  # No longer store task_desc in each edge
        )

        # Record before state (parent node results)
        if parent_node:
            edge.before_result = {
                "metric": parent_node.metric.value if parent_node.metric else None,
                "metrics": parent_node.metrics if hasattr(parent_node, 'metrics') else {},
                "is_buggy": parent_node.is_buggy,
                "has_error": parent_node.exc_type is not None,
                "error_type": parent_node.exc_type,
            }

        # Generate merged memory (from root to this node)
        edge.merged_memory = self._generate_merged_memory(parent_node, edge)

        self.edges[edge.edge_id] = edge
        child_node.edge_to_parent = edge
        self._save_history()
        logger.info(f"Created {edge_type} edge: {edge.edge_id} with merged memory")
        return edge

    def _generate_merged_memory(self, parent_node: Node, current_edge: Edge) -> str:
        """
        Generate merged memory from root to current node
        This is a knowledge evolution summary based on all edges in the path

        Args:
            parent_node: Parent node (None for root)
            current_edge: Current edge being created

        Returns:
            Merged memory string summarizing the evolution
        """
        if parent_node is None:
            # Root node - no parent, just return current edge info
            return f"Initial solution: {current_edge.optimization_point}"

        # Collect all edges from root to parent
        path_edges = []
        current = parent_node
        while current is not None:
            if hasattr(current, 'edge_to_parent') and current.edge_to_parent:
                path_edges.insert(0, current.edge_to_parent)
            current = current.parent if hasattr(current, 'parent') else None

        # If no path edges, parent is root
        if not path_edges:
            return f"From baseline to current: {current_edge.optimization_point}"

        # Call LLM to merge the knowledge
        return self._merge_knowledge_with_llm(path_edges, current_edge)

    def _merge_knowledge_with_llm(self, path_edges: list, current_edge: Edge) -> str:
        """
        Use LLM to merge knowledge from all edges in the path
        This creates an evolving knowledge summary

        Args:
            path_edges: List of edges from root to parent
            current_edge: Current edge being created

        Returns:
            Merged memory string
        """

        # Prepare edge history
        edge_history = []
        for i, edge in enumerate(path_edges, 1):
            edge_info = {
                "step": i,
                "type": edge.edge_type,
                "plan": edge.optimization_point,
                "reason": edge.optimization_reason,
                "metrics_before": edge.before_result.get("metrics", {}),
                "metrics_after": edge.after_result.get("metrics", {}),
            }
            edge_history.append(edge_info)

        # Add current edge
        edge_history.append({
            "step": len(path_edges) + 1,
            "type": current_edge.edge_type,
            "plan": current_edge.optimization_point,
            "reason": current_edge.optimization_reason,
            "metrics_before": current_edge.before_result.get("metrics", {}),
        })

        import json
        history_json = json.dumps(edge_history, indent=2)

        prompt = f"""
You are an expert in analyzing algorithm evolution and knowledge accumulation.

【TASK】
Summarize the knowledge evolution from the initial solution to the current state.
This is NOT a simple concatenation - it's an EVOLVING KNOWLEDGE SUMMARY.

【EDGE HISTORY (from root to current)】
{history_json}

【REQUIREMENTS】
Your summary should capture:
1. What problems were identified in the initial solution
2. How each modification addressed (or failed to address) those problems
3. What NEW problems emerged after each modification
4. The accumulated knowledge about what works and what doesn't

Example of good merged memory:
"Initial FIFO solution had high carbon usage (100). First attempt added carbon threshold (0.5)
which reduced carbon to 80 but increased time to 150s due to queue buildup. Second attempt
adjusted threshold to 0.7 and added adaptive mechanism, balancing carbon (85) and time (130s).
Current understanding: threshold must be dynamic based on queue state, not fixed."

Example of bad merged memory (simple concatenation):
"Step 1: added threshold. Step 2: adjusted threshold. Step 3: added adaptive mechanism."

【OUTPUT FORMAT】
Provide a 2-4 sentence summary that captures the EVOLUTION of understanding, not just a list of changes.
Focus on: initial problems → attempted fixes → new problems → current understanding.
"""

        try:
            response = call_llm(prompt)
            merged_memory = response.strip()
            logger.info(f"Generated merged memory: {merged_memory[:100]}...")
            return merged_memory
        except Exception as e:
            logger.error(f"Failed to generate merged memory with LLM: {e}")
            # Fallback: simple concatenation
            return " → ".join([f"{e['type']}: {e['plan'][:50]}" for e in edge_history])

    def record_edge_result(self, node: Node) -> None:
        """
        Record the result of an edge after child node execution
        This should be called after the child node has been executed and evaluated
        """
        if node.edge_to_parent is None:
            return

        try:
            edge = node.edge_to_parent

            # Record after state (child node results)
            edge.after_result = {
                "metric": node.metric.value if node.metric else None,
                "is_buggy": node.is_buggy,
                "has_error": node.exc_type is not None,
                "error_type": node.exc_type,
            }

            # Calculate metric change
            if node.parent and node.metric and node.parent.metric:
                edge.metric_change = {
                    "score_change": node.metric.value - node.parent.metric.value,
                    "parent_score": node.parent.metric.value,
                    "child_score": node.metric.value,
                    "improvement_percentage": ((node.metric.value - node.parent.metric.value) /
                                              abs(node.parent.metric.value) * 100) if node.parent.metric.value != 0 else 0
                }

            self._save_history()
            logger.info(f"Recorded result for edge: {edge.edge_id}")

        except Exception as e:
            logger.error(f"Failed to record edge result: {e}")

    def analyze_optimization(self, node: Node) -> None:
        """
        Analyze optimization effectiveness with LLM
        This performs a comprehensive analysis including:
        1. Effectiveness analysis (was it successful and why)
        2. Detailed summary (algorithm-level changes)
        3. Optimization score (0-5)
        """
        if node.edge_to_parent is None:
            return

        try:
            edge = node.edge_to_parent

            # Call LLM for comprehensive analysis
            analysis_result = self._analyze_with_llm(node, edge)

            # Update edge with analysis results
            edge.effectiveness_analysis = analysis_result.get("effectiveness_analysis", "")
            edge.detailed_summary = analysis_result.get("detailed_summary", "")
            edge.optimization_score = analysis_result.get("optimization_score", 0.0)

            self._save_history()
            logger.info(f"Analyzed optimization for edge: {edge.edge_id}, score: {edge.optimization_score}/5.0")

        except Exception as e:
            logger.error(f"Failed to analyze optimization: {e}")

    def get_path_to_root(self, node: Node) -> List[Edge]:
        """Get complete path from root to current node"""
        path = []
        current = node
        while current.parent is not None:
            if hasattr(current, 'edge_to_parent') and current.edge_to_parent:
                path.append(current.edge_to_parent)
            current = current.parent
        return list(reversed(path))  # From root to leaf

    def get_path_summary_for_llm(self, node: Node, max_edges: int = 10) -> str:
        """
        Get path summary for LLM context
        Returns the merged memory from the node's edge (knowledge evolution summary)
        """
        if node is None:
            return ""

        # If node has edge_to_parent with merged_memory, return it
        if hasattr(node, 'edge_to_parent') and node.edge_to_parent and node.edge_to_parent.merged_memory:
            return f"""
【KNOWLEDGE EVOLUTION SUMMARY】
{node.edge_to_parent.merged_memory}

This summary captures the accumulated knowledge from all previous optimizations.
Use this to understand what has been tried and what problems remain.
"""

        # Fallback: no merged memory available
        return ""

    def get_path_to_root(self, node: Node) -> List[Edge]:
        """Get all edges from root to node"""
        path = []
        current = node
        while current is not None:
            if hasattr(current, 'edge_to_parent') and current.edge_to_parent:
                path.insert(0, current.edge_to_parent)
            current = current.parent if hasattr(current, 'parent') else None
        return path

    def _analyze_with_llm(self, node: Node, edge: Edge) -> Dict[str, any]:
        """
        Comprehensive LLM analysis of optimization
        Returns: dict with effectiveness_analysis, detailed_summary, optimization_score
        """

        parent = node.parent

        # Prepare context
        parent_info = "N/A (root node)"
        if parent:
            parent_info = f"""
Metric: {parent.metric.value if parent.metric else "N/A"}
Is Buggy: {parent.is_buggy}
Has Error: {parent.exc_type is not None}
Error Type: {parent.exc_type if parent.exc_type else "None"}
"""

        current_info = f"""
Metric: {node.metric.value if node.metric else "N/A"}
Is Buggy: {node.is_buggy}
Has Error: {node.exc_type is not None}
Error Type: {node.exc_type if node.exc_type else "None"}
"""

        # Get previous successful node for comparison (for debug edges)
        comparison_context = ""
        if edge.edge_type == "debug" and node.exc_type is None and node.metric:
            # Find the last successful node in the path
            prev_successful = self._find_previous_successful_node(node)
            if prev_successful and prev_successful.metric:
                score_diff = node.metric.value - prev_successful.metric.value
                comparison_context = f"""
【COMPARISON WITH PREVIOUS SUCCESSFUL NODE】
Previous Successful Score: {prev_successful.metric.value}
Current Score: {node.metric.value}
Score Difference: {score_diff}
Note: For debug success, compare with the last successful node, not the buggy parent.
"""

        # Build prompt
        prompt = f"""
You are an expert in analyzing algorithm optimizations.

【TASK DESCRIPTION】
{self.task_desc if self.task_desc else "Algorithm optimization task"}

【OPTIMIZATION INFORMATION】
Edge Type: {edge.edge_type}
Optimization Point: {edge.optimization_point}
Optimization Reason: {edge.optimization_reason}

【PARENT NODE STATE】
{parent_info}

【CURRENT NODE STATE】
{current_info}

【METRIC CHANGE】
{json.dumps(edge.metric_change, indent=2) if edge.metric_change else "N/A"}

{comparison_context}

【YOUR TASKS】
Analyze this optimization and provide THREE outputs:

1. **Effectiveness Analysis** (2-3 sentences):
   - Was the optimization successful? Why or why not?
   - What was the actual impact on performance?
   - Any unexpected side effects or insights?

2. **Detailed Summary** (3-5 sentences, ALGORITHM-LEVEL):
   - Describe SPECIFIC changes made to the algorithm
   - Mention CONCRETE strategies, parameters, or modules modified
   - Explain HOW the new modules/strategies work
   - MUST include specific function names, parameter values, and implementation details
   - DO NOT use vague terms like "performed optimization", "improved algorithm", "enhanced decision logic"
   - GOOD examples:
     * "Modified the scheduling strategy from FIFO to priority-based queuing. Introduced a carbon intensity threshold parameter (CI_threshold=0.5) that triggers job deferral. Implemented a prediction module that forecasts carbon intensity using a 24-hour rolling window and exponential smoothing (alpha=0.3)."
     * "Changed get_action() to calculate dynamic exec_cap using formula: exec_cap = exec_cap_MAX * (1 - current_carbon/max_carbon). Added new helper function predict_next_carbon() that uses linear regression on last 10 timesteps."
   - BAD examples (DO NOT USE):
     * "Optimized the scheduling algorithm" (too vague)
     * "Improved carbon awareness mechanism" (no concrete details)
     * "Enhanced decision-making logic" (missing implementation specifics)

3. **Optimization Score** (0-4 scale):
   Score based on parent-child relationship:

   **If parent is ROOT (no parent):**
   - Child is correct (no error): 3
   - Child has error: 2 (good idea, implementation error)

   **If parent is BUGGY (has error):**
   - Child is correct (fixed bug): 3
   - Child still has error: 0 (not recommended for exploration)

   **If parent is CORRECT (no error):**
   - Child performance decreased: 1
   - Child has bug: 2
   - Child slightly improved (< 10%): 3
   - Child greatly improved (≥ 10%): 4

   Note: 2 is the neutral center score. Use metric comparison to determine improvement.

【STRICT OUTPUT FORMAT】
Respond with ONLY valid JSON (no markdown, no code blocks):

{{
  "effectiveness_analysis": "2-3 sentence analysis here",
  "detailed_summary": "3-5 sentence algorithm-level detailed summary here",
  "optimization_score": 0.0 to 4.0
}}

CRITICAL RULES:
1. Output ONLY the JSON object
2. No markdown code blocks (no ```)
3. All string values use double quotes
4. optimization_score must be a number (float) between 0.0 and 4.0
5. All three fields are required
"""

        try:
            response = call_llm(prompt)

            # Clean up response
            response = response.strip()
            if response.startswith("```"):
                lines = response.split('\n')
                start_idx = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith("```"):
                        start_idx = i + 1
                    elif line.strip().startswith("{"):
                        start_idx = i
                        break

                end_idx = len(lines)
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip().startswith("```"):
                        end_idx = i
                    elif lines[i].strip().endswith("}"):
                        end_idx = i + 1
                        break

                response = '\n'.join(lines[start_idx:end_idx])

            # Parse JSON
            result = json.loads(response)

            # Validate required fields
            required_fields = ["effectiveness_analysis", "detailed_summary", "optimization_score"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")

            # Ensure score is float
            result["optimization_score"] = float(result["optimization_score"])

            # Clamp score to 0-4 range
            result["optimization_score"] = max(0.0, min(4.0, result["optimization_score"]))

            logger.info(f"LLM analysis complete: score={result['optimization_score']}/4.0")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response: {response[:500]}")
            return self._fallback_analysis(edge)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._fallback_analysis(edge)

    def _find_previous_successful_node(self, node: Node) -> Optional[Node]:
        """Find the previous successful node (no error, has metric) in the path"""
        current = node.parent
        while current:
            if current.exc_type is None and current.metric is not None:
                return current
            current = current.parent
        return None

    def _fallback_analysis(self, edge: Edge) -> Dict[str, any]:
        """Fallback analysis when LLM call fails"""
        return {
            "effectiveness_analysis": "Failed to analyze with LLM. Manual review needed.",
            "detailed_summary": f"Edge type: {edge.edge_type}. Optimization: {edge.optimization_point}. Reason: {edge.optimization_reason}.",
            "optimization_score": 0.0
        }

    def _load_history(self) -> None:
        """Load optimization path from file"""
        if not self.path_file.exists():
            logger.info("No existing optimization path file")
            return

        try:
            with open(self.path_file, 'r') as f:
                data = json.load(f)
                # Reconstruct edges from saved data
                for edge_data in data.get("edges", []):
                    edge = Edge.from_dict(edge_data)
                    self.edges[edge.edge_id] = edge
            logger.info(f"Loaded {len(self.edges)} edges from history")
        except Exception as e:
            logger.error(f"Failed to load optimization path: {e}")
            self.edges = {}

    def _save_history(self) -> None:
        """Save optimization path to file"""
        try:
            # Ensure directory exists
            self.workspace_dir.mkdir(parents=True, exist_ok=True)

            data = {
                "edges": [edge.to_dict() for edge in self.edges.values()],
                "metadata": {
                    "total_edges": len(self.edges),
                    "last_updated": time.time()
                }
            }

            with open(self.path_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.edges)} edges to {self.path_file}")
        except Exception as e:
            logger.error(f"Failed to save optimization path: {e}")
