"""
The journal is the core datastructure in AIDE that contains:
- the generated code samples
- information how code samples relate to each other (the tree structure)
- code execution results
- evaluation information such as metrics
...
"""

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

from dataclasses_json import DataClassJsonMixin
from .interpreter import ExecutionResult
from .utils.metric import MetricValue
from .utils.response import trim_long_string


@dataclass
class Edge(DataClassJsonMixin):
    """Represents optimization relationship between parent and child nodes"""
    edge_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_node_id: str = field(default="")
    child_node_id: str = field(default="")
    edge_type: str = field(default="")  # "draft", "improve", "debug"
    optimization_point: str = field(default="")
    optimization_reason: str = field(default="")
    optimization_result: str = field(default="")
    metric_change: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())

    # New fields for enhanced tracking
    task_desc: str = field(default="")  # Task description for LLM context
    before_result: dict = field(default_factory=dict)  # Parent node results
    after_result: dict = field(default_factory=dict)  # Child node results
    effectiveness_analysis: str = field(default="")  # LLM analysis of effectiveness
    detailed_summary: str = field(default="")  # Algorithm-level detailed summary
    optimization_score: float = field(default=0.0)  # 0-5 score
    merged_memory: str = field(default="")  # Merged memory from root to this node

    def to_summary(self) -> str:
        """Generate summary for LLM"""
        summary = f"""
Edge Type: {self.edge_type}
Optimization: {self.optimization_point}
Reason: {self.optimization_reason}
"""
        if self.detailed_summary:
            summary += f"Detailed Changes: {self.detailed_summary}\n"
        if self.effectiveness_analysis:
            summary += f"Effectiveness: {self.effectiveness_analysis}\n"
        if self.optimization_score > 0:
            summary += f"Score: {self.optimization_score}/5.0\n"
        if self.metric_change:
            summary += f"Metric Change: {self.metric_change}\n"
        return summary

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "edge_id": self.edge_id,
            "parent_node_id": self.parent_node_id,
            "child_node_id": self.child_node_id,
            "edge_type": self.edge_type,
            "optimization_point": self.optimization_point,
            "optimization_reason": self.optimization_reason,
            "optimization_result": self.optimization_result,
            "metric_change": self.metric_change,
            "timestamp": self.timestamp,
            "task_desc": self.task_desc,
            "before_result": self.before_result,
            "after_result": self.after_result,
            "effectiveness_analysis": self.effectiveness_analysis,
            "detailed_summary": self.detailed_summary,
            "optimization_score": self.optimization_score,
            "merged_memory": self.merged_memory,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Edge':
        """Create from dictionary"""
        # Handle old edges without new fields
        for field_name in ["edge_type", "task_desc", "before_result", "after_result",
                          "effectiveness_analysis", "detailed_summary", "optimization_score", "merged_memory"]:
            if field_name not in data:
                if field_name in ["before_result", "after_result"]:
                    data[field_name] = {}
                elif field_name == "optimization_score":
                    data[field_name] = 0.0
                else:
                    data[field_name] = ""
        return cls(**data)


@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    code: str
    plan: str = field(default=None, kw_only=True)  # type: ignore

    # ---- general attrs ----
    step: int = field(default=None, kw_only=True)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    children: set["Node"] = field(default_factory=set, kw_only=True)

    # ---- execution info ----
    _term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    exec_time: float = field(default=None, kw_only=True)  # type: ignore
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None, kw_only=True)  # type: ignore
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore
    metrics_payload: dict = field(default=None, kw_only=True)
    metrics: dict = field(default=None, kw_only=True)
    # whether the agent decided that the code is buggy
    # -> always True if exc_type is not None or no valid metric
    is_buggy: bool = field(default=None, kw_only=True)  # type: ignore

    # ---- optimization info ----
    optimization_point: str = field(default=None, kw_only=True)  # type: ignore
    optimization_reason: str = field(default=None, kw_only=True)  # type: ignore
    edge_to_parent: Optional["Edge"] = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.parent is not None:
            self.parent.children.add(self)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "stage" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution (after truncating it)."""
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path
        - 0 if the node is not a debug node (parent is not buggy)
        - 1 if the parent is buggy but the skip parent isn't
        - n if there were n consecutive debugging steps
        """
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore


@dataclass
class InteractiveSession(DataClassJsonMixin):
    """
    A collection of nodes for an interaction session
    (when the agent interacts with a Jupyter notebook-like interface).
    """

    nodes: list[Node] = field(default_factory=list)
    completed: bool = False

    def append(self, node: Node) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    def generate_nb_trace(self, include_prompt, comment_headers=True) -> str:
        """Generate a trace of the interactive session in IPython format."""
        trace = []
        header_prefix = "## " if comment_headers else ""
        for n in self.nodes:
            trace.append(f"\n{header_prefix}In [{n.step + 1}]:\n")
            trace.append(n.code)
            trace.append(f"\n{header_prefix}Out [{n.step + 1}]:\n")
            trace.append(n.term_out)

        if include_prompt and self.nodes:
            trace.append(f"\n{header_prefix}In [{self.nodes[-1].step + 2}]:\n")

        return "\n".join(trace).strip()


@dataclass
class Journal(DataClassJsonMixin):
    """A collection of nodes representing the solution tree."""

    nodes: list[Node] = field(default_factory=list)
    # eda: InteractiveSession = field(default_factory=lambda: InteractiveSession())

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal."""
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[Node]:
        """Return a list of nodes representing intial coding drafts"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        """Return a list of nodes that are considered buggy by the agent."""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        """Return a list of nodes that are not considered buggy by the agent."""
        return [n for n in self.nodes if not n.is_buggy]

    def get_metric_history(self) -> list[MetricValue]:
        """Return a list of all metric values in the journal."""
        return [n.metric for n in self.nodes]

    def get_best_node(self, only_good=True) -> None | Node:
        """Return the best solution found so far (node with the highest validation metric)."""
        if only_good:
            nodes = self.good_nodes
            if not nodes:
                return None
        else:
            nodes = self.nodes
        return max(nodes, key=lambda n: n.metric)

    def generate_summary(self, include_code: bool = False) -> str:
        """Generate a summary of the journal for the agent."""
        summary = []
        for n in self.good_nodes:
            summary_part = f"Design: {n.plan}\n"
            if include_code:
                summary_part += f"Code: {n.code}\n"
            summary_part += f"Results: {n.analysis}\n"
            summary_part += f"Validation Metric: {n.metric.value}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)


