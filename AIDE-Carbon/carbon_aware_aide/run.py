"""
Main script to run Carbon-Aware AIDE experiment.
Supports external task configuration so the workflow can be reused for other domains.
"""
import logging
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from kbaide_experiment import CarbonAwareExperiment
from task_config import resolve_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("aide")
logger.setLevel(logging.INFO)

def main(ex_resource_path):
    base_dir = Path(__file__).parent
    cfg_base = base_dir
    with open(ex_resource_path / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    task = config.get("task", {})
    paths = config.get("paths", {})

    simulator_dir = resolve_path(cfg_base, paths.get("simulator_dir"))
    baseline_code_path = resolve_path(simulator_dir, paths.get("baseline_code"))
    cfg_path = resolve_path(ex_resource_path, paths.get("aide_config"))

    if not baseline_code_path or not Path(baseline_code_path).exists():
        logger.error(f"Baseline code not found: {baseline_code_path}")
        return
    if simulator_dir is None or not Path(simulator_dir).exists():
        logger.error(f"Simulator directory not found: {simulator_dir}")
        return

    experiment = CarbonAwareExperiment(
        baseline_code_path=baseline_code_path,
        simulator_dir=simulator_dir,
        simulator_info_file=resolve_path(ex_resource_path, paths.get("simulator_info_file")),
        user_requirement_file=resolve_path(ex_resource_path, paths.get("user_requirement_file")),
        goal=task.get("goal"),
        eval=task.get("eval"),
        config_path=cfg_path,
        resume_path=config.get("resume_path", None),
    )

    logger.info("Starting configurable AIDE experiment...")
    logger.info(f"Baseline: {baseline_code_path}")
    logger.info(f"Simulator: {simulator_dir}")
    logger.info(f"Logs directory: {experiment.cfg.log_dir}")
    logger.info(f"Tree visualization (will update each step): {experiment.cfg.log_dir / 'tree_plot.html'}")

    solution = experiment.run(steps=int(task.get("steps", 30)))

    logger.info("\n=== Experiment Results ===")
    logger.info(f"Best score (weighted % improvement vs baseline): {solution.score}")
    if solution.delta_vs_fifo:
        d = solution.delta_vs_fifo
        logger.info(
            "Delta newalgo vs fifo (%%): time=%.3f, carbon=%.3f, jct=%.3f",
            d.get("total_time_pct"),
            d.get("total_carbon_pct"),
            d.get("avg_job_completion_pct"),
        )
    else:
        logger.warning("No delta_vs_fifo available for best solution")
    logger.info("\nExperiment completed!")


if __name__ == "__main__":
    ex_resource_path = Path(sys.argv[1])
    main(ex_resource_path)