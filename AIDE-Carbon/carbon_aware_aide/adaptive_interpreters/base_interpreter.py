"""
Generic interpreter for executing candidate code against configurable test environments.

This interpreter provides a flexible framework that can be adapted to different simulation
environments without modifying the core logic. It supports:
- Configurable test scripts and environments
- Flexible code validation (class names, methods, etc.)
- Customizable metrics parsing
- Environment-specific error handling
"""
import logging
import sys
import subprocess
import shutil
import json
import ast
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

from aide.utils.config import Config, InterpretConfig
# Try to import from aide, fall back to local definition
try:
    from aide.interpreter import ExecutionResult
except ImportError:
    @dataclass
    class ExecutionResult:
        """Result of executing code in the interpreter."""
        term_out: list
        exec_time: float
        exc_type: Optional[str]
        exc_info: Optional[dict] = None
        exc_stack: Optional[list] = None

logger = logging.getLogger("aide")

class BaseInterpreter:
    """
    Generic interpreter that can work with different simulation environments.

    Usage:
        config = SparkSchedulerConfig(...)
        interpreter = GenericInterpreter(
            working_dir="./workspace",
            simulator_dir="./simulator",
            test_script_path="simulator/test.py",
            env_config=config
        )
        result = interpreter.run(candidate_code)
    """

    def __init__(
        self,
        cfg: InterpretConfig,
        working_dir: Path | str,
        metric_keys: List[str] = None,
    ):  
        self.cfg = cfg
        self.working_dir = Path(working_dir).resolve()
        self.simulator_dir = Path(cfg.simulator_dir).resolve()
        self.test_script_path = Path(cfg.test_script_path)
        self.timeout = cfg.timeout
        self.candidate_filename = cfg.candidate_filename
        self.prepare_baseline_metric = cfg.prepare_baseline_metric
        self.target_agent_relpath = self.working_dir / cfg.target_agent_relpath
        self.candidate_class_name = cfg.candidate_class_name
        self.required_methods = cfg.required_methods
        self.metrics_json_prefix = cfg.metrics_json_prefix
        self.algo_name = cfg.algo_name
        self.baseline_algo_name = cfg.baseline_algo_name
        self.metric_keys = metric_keys
        self.test_args = cfg.test_args
        self.project_root = self.simulator_dir.parent.parent

        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.working_dir / "test_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_args = [] if self.test_args is None else self.test_args
        # self.test_args = self.test_args + ["--result_folder", str(self.output_dir), "--log_file", str(self.output_dir / "test_log.txt")]

        if self.prepare_baseline_metric:
            self.baseline_results = self.run_baseline_metric()
        else:
            self.baseline_results = {}

    def run_baseline_metric(self):
        """
        Run baseline algorithm to collect baseline metrics for comparison.
        Similar to evaluator.py's approach of running baseline and collecting results.
        """
        logger.info(f"Running baseline metric with algorithm: {self.baseline_algo_name}")

        return json.dumps({
            "baseline": {
                "scheme": "baseline",
                "average_hit_rate": sum([0.7941135692665768, 0.7073511457809135, 0.7927047623051182, 0.39661573972344916, 0.8423868210104711])/5,
                "average_runtime": sum([20.97733783721924, 72.69543290138245, 2.992535352706909, 296.8255195617676, 21.05297541618347])/5, 
                "score": 0.35927605719566147,
            },
            "quick_greedy": {
                "scheme": "quick_greedy",
                "average_hit_rate": sum([0.7941136775829335, 0.707345422269654, 0.7934478092983207, 0.3969344051255755, 0.8426762660369552])/5,
                "average_runtime": sum([3.000903606414795, 5.149852752685547, 1.7654576301574707, 10.806805849075317, 4.089353799819946])/5,
                "score": 0.4373095558150322,
            }
        })

        # Read the baseline code
        baseline_code_path = self.working_dir / self.candidate_filename
        if not baseline_code_path.exists():
            logger.warning(f"Baseline code not found at {baseline_code_path}")
            return {}

        with open(baseline_code_path, "r") as f:
            baseline_code = f.read()

        # Write baseline code to target location
        target_path = self.simulator_dir / self.target_agent_relpath
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline_code_path, target_path)

        # Verify test script exists
        test_script_full_path = self.project_root / self.test_script_path
        if not test_script_full_path.exists():
            logger.error(f"Test script not found: {test_script_full_path}")
            return {}

        # Prepare output directory
        baseline_output_dir = self.working_dir / "baseline_output"
        baseline_output_dir.mkdir(parents=True, exist_ok=True)

        # Build test command for baseline
        baseline_test_args = ["--result_folder", str(baseline_output_dir),
                             "--log_file", str(baseline_output_dir / "baseline_log.txt")]
        test_cmd = [sys.executable, str(test_script_full_path), *baseline_test_args]

        # Execute baseline test
        try:
            logger.info(f"Executing baseline test: {' '.join(test_cmd)}")
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.project_root)
            )

            output_lines = []
            if result.stdout:
                output_lines.extend(result.stdout.splitlines())
            if result.stderr:
                output_lines.extend(result.stderr.splitlines())

            if result.returncode != 0:
                logger.warning(f"Baseline test failed with return code: {result.returncode}")
                return {}

            # Parse baseline metrics
            baseline_metrics = self._parse_metrics(output_lines)
            if baseline_metrics:
                logger.info(f"Baseline metrics collected: {baseline_metrics}")
                return baseline_metrics
            else:
                logger.warning("No baseline metrics found in output")
                return {}

        except subprocess.TimeoutExpired:
            logger.error(f"Baseline test execution exceeded timeout of {self.timeout} seconds")
            return {}
        except Exception as e:
            logger.error(f"Error running baseline test: {str(e)}")
            return {}
        
    def validate_candidate_structure(self, code: str, start_time):
        try:
            tree = ast.parse(code)
            if self.candidate_class_name:
                class_node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == self.candidate_class_name), None)
                if class_node is None:
                    return False, ExecutionResult(term_out=[f"PreflightError: class `{self.candidate_class_name}` not found in generated code."], exec_time=time.time() - start_time, exc_type="PreflightError")
                method_names = {n.name for n in class_node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
                missing = sorted(set(self.required_methods) - method_names)
            else:
                func_names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,), )}
                missing = sorted(set(self.required_methods) - func_names)
            if missing:
                return False, ExecutionResult(term_out=[f"PreflightError: missing required method(s): {', '.join(missing)}"], exec_time=time.time() - start_time, exc_type="PreflightError")
            return True, None
        except Exception as e:
            return False, ExecutionResult(term_out=["PreflightError: failed to validate generated code structure.", f"{e.__class__.__name__}: {e}"], exec_time=time.time() - start_time, exc_type="PreflightError")


    def _parse_metrics(self, output_lines: list[str]) -> Optional[dict]:
        for line in output_lines:
            if line.startswith(self.metrics_json_prefix):
                try:
                    payload = json.loads(line.split(self.metrics_json_prefix, 1)[1])
                    return {
                        "scheme_results": payload.get("scheme_results"),
                        f"normalized_vs_{self.baseline_algo_name}": payload.get(f"normalized_vs_{self.baseline_algo_name}") or payload.get("normalized_vs_fifo"),
                        f"delta_vs_{self.baseline_algo_name}": payload.get(f"delta_vs_{self.baseline_algo_name}") or payload.get("delta_vs_fifo"),
                        "raw": payload,
                    }
                except Exception:
                    continue
        return None

    def detect_runtime_errors(self,
        output_lines: List[str],
        log_content: str,
        metrics: Optional[Dict]
    ) -> Optional[ExecutionResult]:
        newalgo_is_inf = False
        if metrics:
            scheme_results_list = (metrics.get("raw") or {}).get("scheme_results") or []
            for sr in scheme_results_list:
                if sr.get("scheme") in [self.algo_name, "newalgo"]:
                    vals = [sr.get(key, 0) for key in self.metric_keys]
                    if any(v >= 1e17 for v in vals):
                        newalgo_is_inf = True
                    break

        if newalgo_is_inf:
            log_file_path = self.output_dir / "test_log.txt"
            log_content = ""
            if log_file_path.exists():
                try:
                    log_content = log_file_path.read_text(encoding="utf-8", errors="replace")
                except Exception as _le:
                    log_content = f"(could not read log: {_le})"

            # Check for explicit runtime error keywords in log
            error_keywords = [
                "ERROR", "Traceback (most recent call last)",
                "AttributeError", "TypeError", "ValueError",
                "KeyError", "IndexError", "NameError", "RuntimeError"
            ]
            found_error = any(kw in log_content for kw in error_keywords)

            if found_error:
                # Case A: buggy code - extract the error detail
                error_detail = ""
                error_idx = log_content.find("ERROR in")
                tb_idx = log_content.find("Traceback (most recent call last)")
                start_idx = error_idx if error_idx != -1 else tb_idx
                if start_idx != -1:
                    error_detail = log_content[start_idx:start_idx + 1000]
                else:
                    for kw in error_keywords[2:]:
                        idx = log_content.find(kw)
                        if idx != -1:
                            error_detail = log_content[max(0, idx - 100):idx + 600]
                            break
                output_lines.append(f"=== {self.algo_name} RUNTIME ERROR (metrics=INF, bug detected in log) ===")
                output_lines.extend(error_detail.splitlines())
                output_lines.append(f"Fix the error above in {self.candidate_class_name}.{self.required_methods[0]}() before re-running.")
                return ExecutionResult(
                    term_out=output_lines,
                    exec_time=0,
                    exc_type=f"{self.candidate_class_name}Error",
                )
            else:
                output_lines.append(f"=== {self.algo_name} LOGIC ERROR (metrics=INF, no exception in log) ===")
                output_lines.append(f"{self.algo_name} metrics are 1e18 (INF sentinel), but no exception was found in the log.")
                return ExecutionResult(
                    term_out=output_lines,
                        exec_time=0,
                        exc_type=f"{self.candidate_class_name}LogicDeadlock",
                    )

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        """Execute candidate code against the test environment."""
        import time
        start_time = time.time()

        # Step 1: Syntax check
        try:
            compile(code, self.candidate_filename, "exec")
        except SyntaxError as e:
            return ExecutionResult(
                term_out=[
                    "PreflightError: SyntaxError while compiling generated code.",
                    f"{e.__class__.__name__}: {e}"
                ],
                exec_time=time.time() - start_time,
                exc_type="SyntaxError"
            )

        # Step 2: Structure validation
        is_valid, error_msg = self.validate_candidate_structure(code, start_time)
        if not is_valid:
            return error_msg

        # Step 3: Write candidate code to working directory
        candidate_file_path = self.working_dir / self.candidate_filename
        candidate_file_path.write_text(code, encoding="utf-8")

        # # Step 4: Copy to simulator target location
        # target_path = self.simulator_dir / self.target_agent_relpath
        # target_path.parent.mkdir(parents=True, exist_ok=True)
        # shutil.copy2(candidate_file_path, target_path)

        # Step 5: Verify test script exists
        #test_script_full_path = self.project_root / self.test_script_path
        test_script_full_path = (self.working_dir/ "input"/ Path(self.test_script_path).name)
        if not test_script_full_path.exists():
            return ExecutionResult(
                term_out=[f"Test script not found: {test_script_full_path}"],
                exec_time=time.time() - start_time,
                exc_type="FileNotFoundError"
            )

        # Step 6: Prepare log file
        log_file_path = self.output_dir / "test_log.txt"
        log_file_path.write_text("", encoding="utf-8")

        # Step 7: Build test command
        if self.prepare_baseline_metric:
            self.test_args = self.test_args + [self.baseline_results]
            test_cmd = [sys.executable, str(test_script_full_path), *self.test_args]
        else:
            test_cmd = [sys.executable, str(test_script_full_path), *self.test_args]


        # Step 8: Execute test
        try:
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                #cwd=str(self.project_root)
                cwd = str(self.working_dir)
            )
            exec_time = time.time() - start_time

            output_lines = []
            if result.stdout:
                output_lines.extend(result.stdout.splitlines())
            if result.stderr:
                output_lines.extend(result.stderr.splitlines())

            if result.returncode != 0:
                output_lines.append(f"\nTest failed with return code: {result.returncode}")
                return ExecutionResult(
                    term_out=output_lines,
                    exec_time=exec_time,
                    exc_type="TestExecutionError"
                )

            # Step 9: Parse metrics
            metrics = self._parse_metrics(output_lines)
            if metrics:
                payload = metrics.get("raw") or {
                    "scheme_results": metrics.get("scheme_results"),
                    f"normalized_vs_{self.baseline_algo_name}": metrics.get(f"normalized_vs_{self.baseline_algo_name}"),
                    f"delta_vs_{self.baseline_algo_name}": metrics.get(f"delta_vs_{self.baseline_algo_name}"),
                }
                output_lines.append(
                    self.metrics_json_prefix + json.dumps(payload)
                )

            # Step 10: Check for environment-specific runtime errors
            log_content = ""
            if log_file_path.exists():
                try:
                    log_content = log_file_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass

            error_result = self.detect_runtime_errors(output_lines, log_content, metrics)
            if error_result:
                error_result.exec_time = exec_time
                return error_result

            return ExecutionResult(
                term_out=output_lines,
                exec_time=exec_time,
                exc_type=None,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                term_out=[f"Test execution exceeded timeout of {self.timeout} seconds"],
                exec_time=time.time() - start_time,
                exc_type="TimeoutError"
            )
        except Exception as e:
            return ExecutionResult(
                term_out=[f"Error running test: {str(e)}"],
                exec_time=time.time() - start_time,
                exc_type=e.__class__.__name__
            )

    def cleanup_session(self):
        """Cleanup any session-specific resources."""
        pass
