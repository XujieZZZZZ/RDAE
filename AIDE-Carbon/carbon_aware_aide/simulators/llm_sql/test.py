import sys
import os
import traceback
import time
import json

import pandas as pd
from pandas.io.sql import com
import importlib.util

from utils import evaluate_df_prefix_hit_cnt
from initial_program import Evolved
from new_program import NewEvolved
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

def run_new_evolved(
        master_df,
        col_merge,
):
    st = time.time()
    reordered, _ = NewEvolved().reorder(
        master_df,
        early_stop=100000,
        distinct_value_threshold=0.7,
        row_stop=4,
        col_stop=2,
        col_merge=col_merge,
    )
    end = time.time() - st

    results = evaluate_df_prefix_hit_cnt(reordered)
    return results, end

def run_evolved(
        master_df,
        col_merge,
):
    st = time.time()
    reordered, _ = Evolved().reorder(
        master_df,
        early_stop=100000,
        distinct_value_threshold=0.7,
        row_stop=4,
        col_stop=2,
        col_merge=col_merge,
    )
    end = time.time() - st

    results = evaluate_df_prefix_hit_cnt(reordered)
    return results, end

def run(filename, alg="", col_merge=[]):
    master_df = pd.read_csv(filename)

    print(f"Evaluate master df shape: {master_df.shape}")
    print(f"Nunique: {master_df.nunique().sort_values()}")

    if alg == "baseline":
        return run_evolved(master_df, col_merge)

    return run_new_evolved(master_df, col_merge)

def improve_base(new_val, base_val):
    return None if base_val in (0, 0.0, None) else (base_val - new_val) / base_val * 100.0

def safe_div(a, b):
    return None if b in (0, 0.0, None) else a / b

if __name__ == "__main__":

    WORKSPACE_DIR = BASE_DIR.parent
    new_algo_path = WORKSPACE_DIR / "new_program.py"
    test_files = [
        WORKSPACE_DIR / "input/datasets/movies.csv",
        WORKSPACE_DIR / "input/datasets/beer.csv",
        WORKSPACE_DIR / "input/datasets/BIRD.csv",
        WORKSPACE_DIR / "input/datasets/PDMX.csv",
        WORKSPACE_DIR / "input/datasets/products.csv"

        # "simulators/llm_sql/datasets/movies.csv",
        # "simulators/llm_sql/datasets/beer.csv",
        # "simulators/llm_sql/datasets/BIRD.csv",
        # "simulators/llm_sql/datasets/PDMX.csv",
        # "simulators/llm_sql/datasets/products.csv"
    ]
    col_merges = [
        [['movieinfo', 'movietitle', 'rottentomatoeslink']],
        [['beer/beerId', 'beer/name']],
        [['PostId', 'Body']],
        [['path', 'metadata'], ['hasmetadata', 'isofficial', 'isuserpublisher', 'isdraft', 'hasannotations', 'subsetall']],
        [['product_title', 'parent_asin']],
    ]

    try:
        baseline = sys.argv[1]
        baseline = json.loads(baseline)
        new_algo_hit_rates = []
        new_algo_runtimes = []
        for filename, col_merge in zip(test_files, col_merges):
            results, runtime = run(filename, "", col_merge)
            hit_rate = results[1] / 100
            new_algo_hit_rates.append(hit_rate)
            new_algo_runtimes.append(runtime)

        new_algo_average_hit_rate = sum(new_algo_hit_rates) / len(new_algo_hit_rates)
        new_algo_average_runtime = sum(new_algo_runtimes) / len(test_files)
        new_algo_score = 0.5 * new_algo_average_hit_rate + 0.5 * (
            1.0 / (1.0 + new_algo_average_runtime) if new_algo_average_runtime > 0 else 1.0
        )


        scheme_results_dict = []
        for name in ["baseline", "quick_greedy"]:
            scheme_results_dict.append({
                "scheme": baseline[name]["scheme"],
                "average_hit_rate": baseline[name]["average_hit_rate"],
                "average_runtime": baseline[name]["average_runtime"],
                "score": baseline[name]["score"]
            })
        scheme_results_dict.append({
            "scheme": "newalgo",
            "average_hit_rate": new_algo_average_hit_rate,
            "average_runtime": new_algo_average_runtime,
            "score": new_algo_score
        })

        normalized_vs_baseline = {}
        normalized_vs_baseline["baseline"] = {
            "baseline": "baseline",
            "target": "baseline",
            "average_hit_rate": 1.0,
            "average_runtime": 1.0,
            "score": 1.0
        }
        normalized_vs_baseline["newalgo"] = {
            "baseline": "baseline",
            "target": "newalgo",
            "average_hit_rate": safe_div(new_algo_average_hit_rate, baseline["baseline"]["average_hit_rate"]),
            "average_runtime": safe_div(new_algo_average_runtime, baseline["baseline"]["average_runtime"]),
            "score": safe_div(new_algo_score, baseline["baseline"]["score"])
        }

        delta_vs_baseline = {
            "baseline": "baseline",
            "target": "newalgo",
            "average_hit_rate_pct": improve_base(new_algo_average_hit_rate, baseline["baseline"]["average_hit_rate"]),
            "average_runtime_pct": improve_base(new_algo_average_runtime, baseline["baseline"]["average_runtime"]),
            "score_pct": improve_base(new_algo_score, baseline["baseline"]["score"])
        }

        payload = {
            "scheme_results": scheme_results_dict,
            "normalized_vs_baseline": normalized_vs_baseline,
            "delta_vs_baseline": delta_vs_baseline,
        }
        print("AIDE_METRICS_JSON=" + json.dumps(payload), flush=True)
    except Exception as e:
        print(f"AIDE_METRICS_JSON_ERROR={e}", flush=True)