"""
Main script to run Carbon-Aware AIDE experiment.
Supports external task configuration so the workflow can be reused for other domains.
"""
import argparse
import logging
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from carbon_aware_experiment import CarbonAwareExperiment
from task_config import load_task_config, resolve_path, extract_summary_content, load_markdown_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("aide")
logger.setLevel(logging.INFO)

def main():
    base_dir = Path(__file__).parent
    cfg_base = base_dir
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    task = config.get("task", {})
    runtime = config.get("runtime", {})
    paths = config.get("paths", {})

    baseline_code_path = resolve_path(cfg_base, paths.get("baseline_code"))
    simulator_dir = resolve_path(cfg_base, paths.get("simulator_dir"))
    cfg_path = resolve_path(cfg_base, paths.get("aide_config"))

    if not baseline_code_path or not Path(baseline_code_path).exists():
        logger.error(f"Baseline code not found: {baseline_code_path}")
        return
    if simulator_dir is None or not Path(simulator_dir).exists():
        logger.error(f"Simulator directory not found: {simulator_dir}")
        return

    experiment = CarbonAwareExperiment(
        baseline_code_path=baseline_code_path,
        simulator_dir=simulator_dir,
        goal=task.get("goal"),
        eval=task.get("eval"),
        optimization_paths=task.get("optimization_paths"),
        config_path=cfg_path,
        runtime_config=runtime,
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
    main()