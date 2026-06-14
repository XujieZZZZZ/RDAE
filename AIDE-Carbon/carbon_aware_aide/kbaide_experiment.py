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
from aide.utils.config import _load_cfg, prep_cfg, load_task_desc, prep_agent_workspace, save_run, load_journal
from kbaide_agent import CarbonAwareAgent
from kbaide_interpreter import CarbonAwareInterpreter
from adaptive_interpreters.base_interpreter import BaseInterpreter
from knowledge_agents.retrival_from_summarization import RetrievalQwenWithSummarizationAgent
from task_agents.solution_agent import SolutionAgent

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
        simulator_info_file: str = None,
        user_requirement_file: str = None,
        goal: str | None = None,
        eval: str | None = None,
        config_path: str | None = None,
        resume_path: str | None = None,
    ):
        self.baseline_code_path = Path(baseline_code_path)
        self.simulator_dir = Path(simulator_dir)

        # resume_path can be used to load a previous journal and continue the experiment
        self.resume_path = Path(resume_path) if resume_path is not None else None
        if self.resume_path is not None:
            logger.info(f"Resuming experiment from {self.resume_path}")
            self.resume = True
        else:
            self.resume = False

        cfg_path = Path(config_path) if config_path else Path(__file__).parent.parent / "aide" / "utils" / "config.yaml"
        _cfg = _load_cfg(path=cfg_path, use_cli_args=False)

        # Set up task description
        if goal is None:
            goal = (
                "Optimize the algorithm to maximize score."
            )
        if eval is None:
            eval = (
                "Evaluate using one metric: score, higher value is better."
            )

        _cfg.goal = goal
        _cfg.eval = eval
        _cfg.data_dir = str(self.simulator_dir)

        self.cfg = prep_cfg(_cfg, resume_path=self.resume_path)

        # Prepare workspace
        if not self.resume:
            with Status("Preparing agent workspace..."):
                prep_agent_workspace(self.cfg)

        self.task_desc = load_task_desc(self.cfg)
        if self.resume:
            self.journal = load_journal(self.resume_path)
        else:
            self.journal = Journal()

        retreival_agent = RetrievalQwenWithSummarizationAgent()
        task_solution_agent = SolutionAgent(
            simulator_info_file=simulator_info_file,
            user_requirement_file=user_requirement_file,
            retrieval_agent=retreival_agent
        )

        self.agent = CarbonAwareAgent(
            task_desc=self.task_desc,
            cfg=self.cfg,
            journal=self.journal,
            baseline_code_path=str(self.baseline_code_path),
            task_solution_agent=task_solution_agent,
        )
        # self.interpreter = CarbonAwareInterpreter(
        #     working_dir=self.cfg.workspace_dir,
        #     baseline_code_path=str(self.baseline_code_path),
        #     simulator_dir=str(self.simulator_dir),
        #     timeout=self.cfg.exec.timeout,
        #     **self.runtime_config,
        # )

        self.interpreter = BaseInterpreter(
            cfg=self.cfg.interpret,
            working_dir=self.cfg.workspace_dir,
            metric_keys=self.cfg.metrics.metric_keys
        )

    def run(self, steps: int) -> CarbonAwareSolution:
        """
        Run the experiment for a specified number of steps.
        
        Args:
            steps: Number of improvement iterations
            
        Returns:
            CarbonAwareSolution with best found solution
        """
        if self.resume:
            start = self.journal.__len__()
        else:
            start = 0

        # node = self.journal[16]
        # raise 0
        # self.agent.parse_exec_result(node, self.interpreter.run(node.code, True))
        # raise 0
        
        for i in range(start,steps):
            logger.info(f"Step {i+1}/{steps}")
            self.agent.step(exec_callback=self.interpreter.run)
            save_run(self.cfg, self.journal)
        
        self.interpreter.cleanup_session()

        # Extract and consolidate experiences after all steps complete
        logger.info("Extracting experiences from this run...")
        from experience_tracker import ExperienceTracker
        experience_tracker = ExperienceTracker(workspace_dir=self.cfg.workspace_dir)
        bug_history_file = Path(self.cfg.workspace_dir) / "bug_history.json"
        experience_tracker.extract_and_consolidate_experiences(bug_history_file)

        # Get best node
        best_node = self.journal.get_best_node(only_good=True)

        return CarbonAwareSolution(
            code=best_node.code,
            score=best_node.metric.value if best_node.metric else None,
            delta_vs_fifo=getattr(best_node, "delta_vs_fifo", None),
            scheme_results=getattr(best_node, "metrics_payload", {}).get("scheme_results") if getattr(best_node, "metrics_payload", None) else None,
        )