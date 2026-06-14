"""
Genericizable interpreter for executing candidate scheduler code against a task-specific test script.
"""
import logging
import sys
import subprocess
import shutil
import json
import ast
from pathlib import Path
from typing import Optional

from aide.interpreter import ExecutionResult

logger = logging.getLogger("aide")


class CarbonAwareInterpreter:
    def __init__(
        self,
        working_dir: Path | str,
        baseline_code_path: str,
        simulator_dir: str,
        test_script_path: str = "simulator/test.py",
        timeout: int = 3600,
        candidate_filename: str = "fifo_agent.py",
        target_agent_relpath: str = "agents/new_agent.py",
        candidate_class_name: str = "NewAgent",
        required_methods: tuple[str, ...] = ("get_action",),
        metrics_json_prefix: str = "AIDE_METRICS_JSON=",
        scheme_name: str = "newalgo",
        baseline_scheme_name: str = "fifo",
        test_args: list[str] | None = None,
    ):
        self.working_dir = Path(working_dir).resolve()
        self.baseline_code_path = Path(baseline_code_path)
        self.simulator_dir = Path(simulator_dir)
        self.test_script_path = Path(test_script_path)
        self.timeout = timeout
        self.candidate_filename = candidate_filename
        self.target_agent_relpath = Path(target_agent_relpath)
        self.candidate_class_name = candidate_class_name
        self.required_methods = required_methods
        self.metrics_json_prefix = metrics_json_prefix
        self.scheme_name = scheme_name
        self.baseline_scheme_name = baseline_scheme_name
        self.project_root = self.simulator_dir.parent
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.working_dir / "test_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_args = test_args or [
            "--num_exp", "1",
            "--exec_cap", "50",
            "--exec_cap", "100",
            "--num_init_dags", "1",
            "--num_stream_dags", "50",
            "--canvs_visualization", "0",
            "--test_schemes", "decima", "cap_decima", "pcaps", "fifo", "cap_fifo", "newalgo",
            "--result_folder", str(self.output_dir),
            "--log_file", str(self.output_dir / "test_log.txt"),
        ]

    def run(self, code: str, reset_session: bool = True) -> ExecutionResult:
        import time
        start_time = time.time()
        try:
            compile(code, self.candidate_filename, "exec")
        except SyntaxError as e:
            return ExecutionResult(term_out=["PreflightError: SyntaxError while compiling generated code.", f"{e.__class__.__name__}: {e}"], exec_time=time.time() - start_time, exc_type="SyntaxError")

        try:
            tree = ast.parse(code)
            # ------------------------
            # Class-based task
            # ------------------------
            if self.candidate_class_name:
                class_node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == self.candidate_class_name), None)
                if class_node is None:
                    return ExecutionResult(term_out=[f"PreflightError: class `{self.candidate_class_name}` not found."],exec_time=time.time() - start_time,exc_type="PreflightError",)
                method_names = {n.name for n in class_node.body if
                                isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
                missing = sorted(set(self.required_methods) - method_names)
            # ------------------------
            # Function-based task
            # ------------------------
            else:
                func_names = {n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,),)}
                missing = sorted(set(self.required_methods) - func_names)
            if missing:
                return ExecutionResult(term_out=[f"PreflightError: missing required method(s): {', '.join(missing)}"], exec_time=time.time() - start_time, exc_type="PreflightError")
        except Exception as e:
            return ExecutionResult(term_out=["PreflightError: failed to validate generated code structure.", f"{e.__class__.__name__}: {e}"], exec_time=time.time() - start_time, exc_type="PreflightError")

        agent_file_path = self.working_dir / self.candidate_filename
        agent_file_path.write_text(code, encoding="utf-8")
        target_agent_path = self.simulator_dir / self.target_agent_relpath
        target_agent_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(agent_file_path, target_agent_path)

        #test_script_full_path = self.project_root / self.test_script_path
        test_script_full_path = (self.working_dir/ "input"/ Path(self.test_script_path).name)
        if not test_script_full_path.exists():
            return ExecutionResult(term_out=[f"Test script not found: {test_script_full_path}"], exec_time=time.time() - start_time, exc_type="FileNotFoundError")

        log_file_path = self.output_dir / "test_log.txt"
        log_file_path.write_text("", encoding="utf-8")
        test_cmd = [sys.executable, str(self.test_script_path), *self.test_args]

        try:
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=self.timeout, cwd=str(self.project_root))
            exec_time = time.time() - start_time
            output_lines = []
            if result.stdout:
                output_lines.extend(result.stdout.splitlines())
            if result.stderr:
                output_lines.extend(result.stderr.splitlines())
            if result.returncode != 0:
                output_lines.append(f"\nTest failed with return code: {result.returncode}")
                return ExecutionResult(term_out=output_lines, exec_time=exec_time, exc_type="TestExecutionError")

            metrics = self._parse_test_results(output_lines)
            if metrics:
                payload = metrics.get("raw") or {
                    "scheme_results": metrics.get("scheme_results"),
                    f"normalized_vs_{self.baseline_scheme_name}": metrics.get(f"normalized_vs_{self.baseline_scheme_name}"),
                    f"delta_vs_{self.baseline_scheme_name}": metrics.get(f"delta_vs_{self.baseline_scheme_name}"),
                }
                output_lines.append(self.metrics_json_prefix + json.dumps(payload))

            newalgo_is_inf = False
            if metrics:
                scheme_results_list = (metrics.get("raw") or {}).get("scheme_results") or []
                for sr in scheme_results_list:
                    if sr.get("scheme") == "newalgo":
                        vals = [sr.get("total_time", 0), sr.get("total_carbon", 0), sr.get("avg_job_completion", 0)]
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
                error_keywords = ["ERROR", "Traceback (most recent call last)",
                                  "AttributeError", "TypeError", "ValueError",
                                  "KeyError", "IndexError", "NameError", "RuntimeError"]
                found_error = any(kw in log_content for kw in error_keywords)

                if found_error:
                    # Case A: buggy code - extract the error detail
                    error_detail = ""
                    error_idx = log_content.find("ERROR in get_action")
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
                    output_lines.append("=== newalgo RUNTIME ERROR (metrics=INF, bug detected in log) ===")
                    output_lines.extend(error_detail.splitlines())
                    output_lines.append("Fix the error above in NewAgent.get_action() before re-running.")
                    return ExecutionResult(
                        term_out=output_lines,
                        exec_time=exec_time,
                        exc_type="NewAgentRuntimeError",
                    )
                else:
                    # Case B: no explicit error, but metrics are INF -> logic deadlock / infinite loop
                    output_lines.append("=== newalgo LOGIC ERROR (metrics=INF, no exception in log) ===")
                    output_lines.append("newalgo metrics are 1e18 (INF sentinel), but no exception was found in the log.")
                    output_lines.append("This indicates a LOGIC DEADLOCK / INFINITE LOOP in get_action:")
                    output_lines.append("  - The scheduler timed out because it never scheduled a task (returned None repeatedly),")
                    output_lines.append("    OR exec_cap was reduced so aggressively that no tasks could ever run.")
                    output_lines.append("Common causes:")
                    output_lines.append("  1. use_exec computed as 0 or negative -> returned (node, 0, flag) -> env stalls")
                    output_lines.append("  2. exec_cap <= num_exec always true -> returned (None, ...) forever")
                    output_lines.append("  3. Carbon filter eliminated all candidates AND fallback also returned None")
                    output_lines.append("Fix: ensure get_action always schedules >=1 task when frontier_nodes is non-empty.")
                    return ExecutionResult(
                        term_out=output_lines,
                        exec_time=exec_time,
                        exc_type="NewAgentLogicDeadlock",
                    )

            return ExecutionResult(
                term_out=output_lines,
                exec_time=exec_time,
                exc_type=None,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(term_out=[f"Test execution exceeded timeout of {self.timeout} seconds"], exec_time=time.time() - start_time, exc_type="TimeoutError")
        except Exception as e:
            return ExecutionResult(term_out=[f"Error running test: {str(e)}"], exec_time=time.time() - start_time, exc_type=e.__class__.__name__)

    def _parse_test_results(self, output_lines: list[str]) -> Optional[dict]:
        for line in output_lines:
            if line.startswith(self.metrics_json_prefix):
                try:
                    payload = json.loads(line.split(self.metrics_json_prefix, 1)[1])
                    return {
                        "scheme_results": payload.get("scheme_results"),
                        f"normalized_vs_{self.baseline_scheme_name}": payload.get(f"normalized_vs_{self.baseline_scheme_name}") or payload.get("normalized_vs_fifo"),
                        f"delta_vs_{self.baseline_scheme_name}": payload.get(f"delta_vs_{self.baseline_scheme_name}") or payload.get("delta_vs_fifo"),
                        "raw": payload,
                    }
                except Exception:
                    continue
        return None

    def cleanup_session(self):
        pass
