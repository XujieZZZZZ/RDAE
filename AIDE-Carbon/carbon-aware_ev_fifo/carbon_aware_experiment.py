"""
Carbon-Aware Experiment class for AIDE framework adaptation.
"""
import logging
from pathlib import Path
from dataclasses import dataclass
import sys
from rich.status import Status

sys.path.insert(0, str(Path(__file__).parent))

from aide.journal import Journal
from aide.utils.config import _load_cfg, prep_cfg, load_task_desc, prep_agent_workspace, save_run
from carbon_aware_agent import CarbonAwareAgent
from carbon_aware_interpreter import CarbonAwareInterpreter

logger = logging.getLogger("aide")

@dataclass
class CarbonAwareSolution:
    """Solution result from carbon-aware experiment."""
    code: str
    score: float | None
    delta_vs_fifo: dict | None
    scheme_results: list[dict] | None


class CarbonAwareExperiment:
    def __init__(
        self,
        baseline_code_path: str,
        simulator_dir: str,
        goal: str | None = None,
        eval: str | None = None,
        optimization_paths: list[str] | None = None,
        config_path: str | None = None,
        runtime_config: dict | None = None,
    ):
        self.baseline_code_path = Path(baseline_code_path)
        self.simulator_dir = Path(simulator_dir)
        self.runtime_config = runtime_config or {}

        cfg_path = Path(config_path) if config_path else Path(__file__).parent.parent / "aide" / "utils" / "config.yaml"
        _cfg = _load_cfg(path=cfg_path, use_cli_args=False)

        # Set up task description
        if goal is None:
            goal = (
                "Optimize the carbon-aware scheduling algorithm to minimize "
                "total execution time, total carbon usage, and average job completion time."
            )
        if eval is None:
            eval = (
                "Evaluate using three metrics: total execution time (avg executors), "
                "total carbon usage, and average job completion time. "
                "Lower values are better for all metrics."
            )

        _cfg.goal = goal
        _cfg.eval = eval
        _cfg.data_dir = str(self.simulator_dir)

        self.cfg = prep_cfg(_cfg)

        # Prepare workspace
        with Status("Preparing agent workspace..."):
            prep_agent_workspace(self.cfg)

        self.task_desc = load_task_desc(self.cfg)
        self.journal = Journal()
        self.agent = CarbonAwareAgent(
            task_desc=self.task_desc,
            cfg=self.cfg,
            journal=self.journal,
            baseline_code_path=str(self.baseline_code_path),
            optimization_paths=optimization_paths,
        )
        self.interpreter = CarbonAwareInterpreter(
            working_dir=self.cfg.workspace_dir,
            baseline_code_path=str(self.baseline_code_path),
            simulator_dir=str(self.simulator_dir),
            timeout=self.cfg.exec.timeout,
            **self.runtime_config,
        )

    def run(self, steps: int) -> CarbonAwareSolution:
        """
        Run the experiment for a specified number of steps.
        
        Args:
            steps: Number of improvement iterations
            
        Returns:
            CarbonAwareSolution with best found solution
        """
        for i in range(steps):
            logger.info(f"Step {i+1}/{steps}")
            self.agent.step(exec_callback=self.interpreter.run)
            save_run(self.cfg, self.journal)
        
        self.interpreter.cleanup_session()

        # Get best node
        best_node = self.journal.get_best_node(only_good=True)

        return CarbonAwareSolution(
            code=best_node.code,
            score=best_node.metric.value if best_node.metric else None,
            delta_vs_fifo=getattr(best_node, "delta_vs_fifo", None),
            scheme_results=getattr(best_node, "metrics_payload", {}).get("scheme_results") if getattr(best_node, "metrics_payload", None) else None,
        )