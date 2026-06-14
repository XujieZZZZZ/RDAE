import os

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from spark_env.env import Environment
from agents.spark_agent import SparkAgent
from agents.heuristic_agent import DynamicPartitionAgent
from agents.carbon_aware_heuristic_agent import CarbonPartitionAgent
from agents.actor_agent import ActorAgent
from agents.carbon_aware_actor_agent import CarbonActorAgent
from agents.pcaps_actor_agent import PCAPSAgent
from agents.carbon_aware_fifo_agent import CarbonAgent
from agents.green_hadoop_agent import GreenHadoopThetaAgent
from agents.new_agent import NewAgent
from agents.fifo_agent import FifoAgent
from spark_env.canvas import *
from param import *
from utils import *
import pandas as pd
import json

# create result folder
if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

# tensorflow seeding
tf.compat.v1.set_random_seed(args.seed)

df = pd.read_csv(args.carbon_trace)
c = df["carbon_intensity_avg"]
r = df['power_production_percent_renewable_avg']

# pick a random start time in the trace
start_time = np.random.randint(0, len(c) - 100)
c = c[start_time:start_time + 100].to_list()
r = r[start_time:start_time + 100].to_list()

carbon_schedule = [(60000 * i, c[i]) for i in range(len(c))]

carbon_dict = {}
for i in range(len(c)):
    carbon_dict[60000*i] = c[i]

renewable_dict = {}
for i in range(len(r)):
    renewable_dict[60000*i] = r[i]

# set up environment
env = Environment(carbon_schedule=carbon_dict)

# set up agents
agents = {}
carbon_time_list = []

for scheme in args.test_schemes:
    if scheme == 'decima':
        sess = tf.compat.v1.Session()
        agents[scheme] = ActorAgent(
            sess, args.node_input_dim, args.job_input_dim,
            args.hid_dims, args.output_dim, args.max_depth,
            range(1, args.exec_cap + 1))
    elif scheme == 'pcaps' or scheme == 'cap_decima':
        agents[scheme] = None
    elif scheme == 'dynamic_partition':
        agents[scheme] = DynamicPartitionAgent()
    elif scheme == 'spark_fifo':
        agents[scheme] = SparkAgent(exec_cap=args.exec_cap)
    elif scheme == 'cap_fifo':
        agents[scheme] = CarbonAgent(exec_cap=args.exec_cap, carbon_schedule=carbon_dict)
    elif scheme == 'fifo':
        agents[scheme] = FifoAgent(exec_cap=args.exec_cap, carbon_schedule=carbon_dict)
    elif scheme == 'newalgo':
        try:
            agents[scheme] = NewAgent(exec_cap=args.exec_cap, carbon_schedule=carbon_dict)
        except Exception as _newalgo_err:
            import traceback as _tb
            print(f"ERROR: Failed to instantiate NewAgent: {_newalgo_err}", flush=True)
            print("Traceback (most recent call last):", flush=True)
            print(_tb.format_exc(), flush=True)
            exit(1)
    elif scheme == 'cap_partition':
        agents[scheme] = CarbonPartitionAgent(exec_cap=args.exec_cap, carbon_schedule=carbon_dict)
    elif scheme == 'green_hadoop':
        agents[scheme] = GreenHadoopThetaAgent(exec_cap=args.exec_cap, renewable_dict=renewable_dict)
    else:
        print('scheme ' + str(scheme) + ' not recognized')
        exit(1)

# store info for all schemes
all_total_reward = {}
for scheme in args.test_schemes:
    all_total_reward[scheme] = []


scheme_results = []

import sys
import os

sys.path.append("/home/student2025/dch/workspace/aideml-main/carbon_aware_aide")
# from logger import Logger, redirect_stdout_stderr
# logger = Logger(args.log_file)

# 捕获 main.py 内所有 print 输出
# with redirect_stdout_stderr(logger.log):

for exp in range(args.num_exp):
    print('Experiment ' + str(exp + 1) + ' of ' + str(args.num_exp))

    for scheme in args.test_schemes:
        print('Scheme ' + scheme)
        # reset environment with seed
        env.seed(args.num_ep + exp)
        env.reset()

        # load an agent
        agent = agents[scheme]

        # start experiment
        obs = env.observe()

        total_reward = 0
        done = False
        i = 0
        if scheme != 'pcaps' and scheme != 'newalgo' and scheme != 'cap_fifo' and scheme != 'fifo' and scheme != 'cap_partition' and scheme != 'cap_decima' and scheme != 'green_hadoop':
            while not done:
                # print a single dot every 10 steps to indicate progress (all on same line)
                if i % 10 == 0:
                    print('.', end='', flush=True)
                i += 1
                node, use_exec = agent.get_action(obs)
                obs, reward, done = env.step(node, use_exec)
                total_reward += reward

        elif scheme == 'pcaps':
            # refresh tensorflow completely
            tf.compat.v1.reset_default_graph()
            tf.compat.v1.set_random_seed(args.seed)
            sess = tf.compat.v1.Session()
            # initialize scheduler
            agent = PCAPSAgent(
                sess, args.node_input_dim, args.job_input_dim,
                args.hid_dims, args.output_dim, args.max_depth,
                range(1, args.exec_cap + 1), carbon_dict)
            while not done:
                node, use_exec, cw = agent.get_action(obs)
                if i % 10 == 0:
                    print('.', end='', flush=True)
                i += 1
                obs, reward, done = env.step(node, use_exec, carbon_aware=cw)
                total_reward += reward

        elif scheme == 'cap_decima':
            # refresh tensorflow completely
            tf.compat.v1.reset_default_graph()
            tf.compat.v1.set_random_seed(args.seed)
            sess = tf.compat.v1.Session()
            agent = CarbonActorAgent(
                sess, args.node_input_dim, args.job_input_dim,
                args.hid_dims, args.output_dim, args.max_depth,
                range(1, args.exec_cap + 1), carbon_dict)
            while not done:
                node, use_exec, cw = agent.get_action(obs)
                if i % 10 == 0:
                    print('.', end='', flush=True)
                i += 1
                obs, reward, done = env.step(node, use_exec, carbon_aware=cw)
                total_reward += reward

        elif scheme == 'cap_fifo' or scheme == 'fifo' or scheme == 'newalgo' or scheme == 'cap_partition' or scheme == 'green_hadoop':
            import time as _time

            _loop_start = _time.time()
            _timeout_sec = 180  # 3 minutes per experiment run
            _timed_out = False
            while not done:
                # Timeout guard: skip if stuck for more than _timeout_sec
                if _time.time() - _loop_start > _timeout_sec:
                    print(f'\n[TIMEOUT] {scheme} exp={exp} exceeded {_timeout_sec}s, skipping.', flush=True)
                    _timed_out = True
                    break
                if i % 10 == 0:
                    print('.', end='', flush=True)
                i += 1
                try:
                    node, use_exec, cw = agent.get_action(obs)
                except Exception as _act_err:
                    import traceback as _tb

                    print(f'\nERROR in get_action: {_act_err}', flush=True)
                    print(_tb.format_exc(), flush=True)
                    _timed_out = True
                    break
                obs, reward, done = env.step(node, use_exec, carbon_aware=cw)
                total_reward += reward
            if _timed_out:
                all_total_reward[scheme].append(float('nan'))
                print(f'[SKIP] {scheme} timed out, all metrics set to INF.')
                scheme_results.append((
                    scheme,
                    1e18,  # total_time = INF
                    1e18,  # total_carbon = INF
                    1e18,  # avg_job_completion = INF
                ))
                continue
            else:
                all_total_reward[scheme].append(total_reward)

        executor_occupation = np.zeros(int(env.wall_time.curr_time))
        total_carbon_usage = 0
        default_carbon_value = carbon_dict[next(iter(carbon_dict))]

        print("")
        for job_dag in env.finished_job_dags:
            for node in job_dag.nodes:
                for task in node.tasks:
                    start_time = int(task.start_time)
                    finish_time = int(task.finish_time)
                    executor_occupation[
                        start_time:finish_time] += 1

                    start_key = start_time - (start_time % 60000)
                    end_key = finish_time - (finish_time % 60000)
                    for key in range(start_key, end_key + 1, 60000):
                        carbon_value = carbon_dict.get(key, default_carbon_value)
                        # Calculate overlap of this 100-second bucket with the task's time window
                        bucket_start = max(start_time, key)
                        bucket_end = min(finish_time, key + 60000)
                        duration_in_bucket = bucket_end - bucket_start
                        total_carbon_usage += duration_in_bucket * carbon_value

        # Add scheme data to results
        job_dags = env.finished_job_dags
        job_durations = [job_dag.completion_time - job_dag.start_time for job_dag in job_dags]

        scheme_results.append((
            scheme,  # Scheme name
            env.wall_time.curr_time,  # Total time (avg executors)
            total_carbon_usage,  # Total carbon usage
            np.mean(job_durations)  # Average job completion time
        ))
    print("Creating graphs\n")
    if args.canvs_visualization == 0:
        visualize_carbon_usage_aggregated(
            scheme_results,
            os.path.join(args.result_folder, 'aggregated_carbon_usage.png')
        )

    elif args.canvs_visualization == 1:
        visualize_dag_time_save_pdf(
            env.finished_job_dags, env.executors,
            os.path.join(args.result_folder, 'visualization_exp_' + \
                         str(exp) + '_scheme_' + scheme + \
                         '.png'), plot_type='app')
    elif args.canvs_visualization == 2:
        visualize_executor_usage(env.finished_job_dags,
                                 os.path.join(args.result_folder, 'visualization_exp_' + \
                                              str(exp) + '_scheme_' + scheme + '.png'), carbon_dict)

        # # plot CDF of performance
        # if args.canvs_visualization == 0:
        #     visualize_carbon(carbon_time_list)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)

        # for scheme in args.test_schemes:
        #     x, y = compute_CDF(all_total_reward[scheme])
        #     ax.plot(x, y)

        # plt.xlabel('Total reward')
        # plt.ylabel('CDF')
        # plt.legend(args.test_schemes)
        # fig.savefig(args.result_folder + 'total_reward.png')

        # plt.close(fig)

        # ---- AIDE-friendly machine readable summary ----
        # scheme_results: List[(scheme, total_time, total_carbon_usage, avg_job_completion)]
        # We emit a JSON line so the outer agent can parse metrics without another LLM call.
        #
        # Metric logic expected by the outer agent:
        # - we provide raw metrics for all schemes
        # - we provide normalization vs decima for cap_decima and newcap
        # - we provide % improvement of newcap vs cap_decima (can be negative)

try:
    # Use only the last occurrence of each scheme (last experiment)
    # to avoid stale data when num_exp > 1
    scheme_results_last = {}
    for (s, t, carb, jct) in scheme_results:
        scheme_results_last[s] = (s, t, carb, jct)

    scheme_results_dict = [
        {
            "scheme": s,
            "total_time": float(t),
            "total_carbon": float(carb),
            "avg_job_completion": float(jct),
        }
        for (s, t, carb, jct) in scheme_results_last.values()
    ]


    def safe_div(a, b):
        return None if b in (0, 0.0, None) else a / b


    def pct_improve(new_val, base_val):
        # Positive means improvement (reduction). Negative means worse.
        # (base - new) / base * 100
        return None if base_val in (0, 0.0, None) else (base_val - new_val) / base_val * 100.0


    baseline_name = "fifo"
    baseline = next((x for x in scheme_results_dict if x["scheme"] == baseline_name), None)

    # normalization vs fifo for multiple targets
    normalized_vs_fifo = {}
    for map_name, target_name in zip(["baseline", "newalgo"],["fifo", "newalgo"]):
        target = next((x for x in scheme_results_dict if x["scheme"] == target_name), None)
        if baseline and target:
            normalized_vs_fifo[map_name] = {
                "baseline": baseline_name,
                "target": target_name,
                "total_time": safe_div(target["total_time"], baseline["total_time"]),
                "total_carbon": safe_div(target["total_carbon"], baseline["total_carbon"]),
                "avg_job_completion": safe_div(
                    target["avg_job_completion"], baseline["avg_job_completion"]
                ),
            }
        else:
            normalized_vs_fifo[map_name] = None

    # delta: newalgo vs fifo (% improvement, can be negative)
    fifo_result = next((x for x in scheme_results_dict if x["scheme"] == "fifo"), None)
    new_result = next((x for x in scheme_results_dict if x["scheme"] == "newalgo"), None)
    delta_vs_fifo = None
    if fifo_result and new_result:
        delta_vs_fifo = {
            "baseline": "fifo",
            "target": "newalgo",
            "total_time_pct": pct_improve(new_result["total_time"], fifo_result["total_time"]),
            "total_carbon_pct": pct_improve(new_result["total_carbon"], fifo_result["total_carbon"]),
            "avg_job_completion_pct": pct_improve(new_result["avg_job_completion"], fifo_result["avg_job_completion"]),
        }

    payload = {
        "scheme_results": scheme_results_dict,
        "normalized_vs_baseline": normalized_vs_fifo,
        "delta_vs_baseline": delta_vs_fifo,
    }
    print("AIDE_METRICS_JSON=" + json.dumps(payload), flush=True)
except Exception as e:
    print(f"AIDE_METRICS_JSON_ERROR={e}", flush=True)
 