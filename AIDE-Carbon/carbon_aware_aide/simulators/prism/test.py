import importlib.util
import numpy as np
import time
import json
import concurrent.futures
import traceback
import sys
from dataclasses import dataclass

GPU_MEM_SIZE = 80 # GB
MIN_INT = float('-inf')  # Define MIN_INT as negative infinity
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

@dataclass
class Model:
    model_name: str
    model_size: int
    req_rate: int
    slo: int
    cur_gpu_id: int


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """
    Run a function with a timeout using concurrent.futures
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")


def safe_float(value):
    """Convert a value to float safely"""
    try:
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def verify_gpu_mem_constraint(placement_data: dict[int, list[Model]]) -> bool:
    """
    Verify the whether models can fit into GPU memory
    """
    # Check if the placement data is valid
    if placement_data is None:
        return False

    # Check if the placement data is valid
    for gpu_id, models in placement_data.items():
        if sum(model.model_size for model in models) > GPU_MEM_SIZE:
            return False

    return True


def calculate_kvcache_pressure(placement_data: dict[int, list[Model]]) -> float:
    """
    Calculate the KVCache pressure
    """
    max_kvpr = MIN_INT
    for gpu_id, models in placement_data.items():
        total_model_size = sum(model.model_size for model in models)
        total_weighted_req_rate = sum(model.req_rate / model.slo for model in models)
        if GPU_MEM_SIZE - total_model_size > 0:
            kvpr = total_weighted_req_rate / (GPU_MEM_SIZE - total_model_size)
        else:
            kvpr = 1000000
        max_kvpr = max(max_kvpr, kvpr)

    return max_kvpr


def generate_test_gpu_models(num_tests=50):
    """
    Generate multiple test signals with different characteristics
    """
    test_cases = []
    np.random.seed(42)

    for i in range(num_tests):
        gpu_num = np.random.randint(5, 10)
        gpu_models = []
        for j in range(gpu_num*2):
            model_size = np.random.randint(10, 30)
            req_rate = np.random.randint(1, 10)
            slo = np.random.randint(5, 10)
            gpu_models.append(Model(model_name=f"model_{j}", model_size=model_size, req_rate=req_rate, slo=slo, cur_gpu_id=j))

        test_cases.append((gpu_num, gpu_models))

    return test_cases

def evaluate(program_path):
    """
    Main evaluation function that tests the signal processing algorithm
    on multiple test signals and calculates the composite performance metric.
    """

    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if required function exists
        if not hasattr(program, "compute_model_placement"):
            return {
                "max_kvpr": 0.0,
                "success_rate": 0.0,
                "combined_score": 0.0,
                "error": "Missing compute_model_placement function",
                }
        
        # Generate test gpu and models
        test_gpu_models = generate_test_gpu_models()

        # Collect metrics across all tests
        all_kvpr = []
        all_metrics = []
        successful_runs = 0

        for i, (gpu_num, gpu_models) in enumerate(test_gpu_models):
            try:
                # Run the algorithm with timeout
                start_time = time.time()

                # Call the program's main function
                result = run_with_timeout(
                    program.compute_model_placement,
                    kwargs={
                        'gpu_num': gpu_num,
                        'models': gpu_models
                    },
                    timeout_seconds=10
                )

                execution_time = time.time() - start_time

                # Validate result format
                if not isinstance(result, dict):
                    print(f"Placement {i}: Invalid result format")
                    continue

                # Calculate metrics using the generated test signal
                max_kvpr = calculate_kvcache_pressure(result)

                # Store metrics
                metrics = {
                    'max_kvpr': safe_float(max_kvpr),
                    'execution_time': safe_float(execution_time),
                }

                all_kvpr.append(safe_float(max_kvpr))
                all_metrics.append(metrics)
                successful_runs += 1

            except TimeoutError:
                print(f"Placement {i}: Timeout")
                continue
            except Exception as e:
                print(f"Placement {i}: Error - {str(e)}")
                continue

        # If no successful runs, return minimal scores
        if successful_runs == 0:
            return {
                    "max_kvpr": 0.0,
                    "success_rate": 0.0,
                    "combined_score": 0.0,
                    "error": "All test signals failed"
                }

        # print(all_metrics)
        # Calculate aggregate metrics
        avg_kvpr = np.mean(all_kvpr)
        if avg_kvpr != 0:
            avg_kvpr = 1.0 / avg_kvpr
        avg_execution_time = np.mean([m['execution_time'] for m in all_metrics])
        success_rate = successful_runs / len(test_gpu_models)

        return {
                "max_kvpr": safe_float(avg_kvpr),
                "execution_time": safe_float(avg_execution_time),
                "success_rate": safe_float(success_rate),
                "combined_score": safe_float(avg_kvpr) + safe_float(success_rate),
            }

    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        print(traceback.format_exc())
        return {
                "max_kvpr": 0.0,
                "success_rate": 0.0,
                "combined_score": 0.0,
                "error": str(e)
            }

def run(new_algo_path):
    baseline_path = BASE_DIR / "initial_program.py"
    best_path = BASE_DIR / "best_program.py"
    baseline_results = evaluate(str(baseline_path))
    best_results = evaluate(str(best_path))
    new_algo_results = evaluate(new_algo_path)

    return baseline_results, best_results, new_algo_results

def improve_base(new_val, base_val):
    return None if base_val in (0, 0.0, None) else (base_val - new_val) / base_val * 100.0

def safe_div(a, b):
    return None if b in (0, 0.0, None) else a / b

if __name__ == "__main__":
    try:
        WORKSPACE_DIR = BASE_DIR.parent
        new_algo_path = WORKSPACE_DIR / "new_program.py"
        # new_algo_path = BASE_DIR / "new_program.py"
        baseline_results, best_results, new_algo_results = run(new_algo_path)


        scheme_results_dict = []
        scheme_results_dict.append({
            "scheme": "baseline",
            "max_kvpr": baseline_results["max_kvpr"],
            "execution_time": baseline_results["execution_time"],
            "success_rate": baseline_results["success_rate"],
            "combined_score":baseline_results["combined_score"]
        })
        scheme_results_dict.append({
            "scheme": "best_algo",
            "max_kvpr": best_results["max_kvpr"],
            "execution_time": best_results["execution_time"],
            "success_rate": best_results["success_rate"],
            "combined_score": best_results["combined_score"]
        })
        scheme_results_dict.append({
            "scheme": "newalgo",
            "max_kvpr": new_algo_results["max_kvpr"],
            "execution_time": new_algo_results["execution_time"],
            "success_rate": new_algo_results["success_rate"],
            "combined_score": new_algo_results["combined_score"]
        })

        normalized_vs_baseline = {}
        normalized_vs_baseline["baseline"] = {
            "baseline": "baseline",
            "target": "baseline",
            "max_kvpr": 1.0,
            "execution_time": 1.0,
            "success_rate": 1.0,
            "combined_score": 1.0
        }
        normalized_vs_baseline["newalgo"] = {
            "baseline": "baseline",
            "target": "newalgo",
            "max_kvpr": safe_div(new_algo_results["max_kvpr"], baseline_results["max_kvpr"]),
            "execution_time": safe_div(new_algo_results["execution_time"], baseline_results["execution_time"]),
            "success_rate": safe_div(new_algo_results["success_rate"], baseline_results["success_rate"]),
            "combined_score": safe_div(new_algo_results["combined_score"], baseline_results["combined_score"])
        }

        delta_vs_baseline = {
            "baseline": "baseline",
            "target": "newalgo",
            "max_kvpr_pct": improve_base(new_algo_results["max_kvpr"], baseline_results["max_kvpr"]),
            "execution_time_pct": improve_base(new_algo_results["execution_time"], baseline_results["execution_time"]),
            "success_rate_pct": improve_base(new_algo_results["success_rate"], baseline_results["success_rate"]),
            "combined_score_pct": improve_base(new_algo_results["combined_score"], baseline_results["combined_score"])
        }
        payload = {
            "scheme_results": scheme_results_dict,
            "normalized_vs_baseline": normalized_vs_baseline,
            "delta_vs_baseline": delta_vs_baseline,
        }
        print("AIDE_METRICS_JSON=" + json.dumps(payload), flush=True)
    except Exception as e:
        print(f"AIDE_METRICS_JSON_ERROR={e}", flush=True)