# green_hadoop_agent.py implements an adapted version of GreenHadoop for Spark scheduling:
# original GreenHadoop paper:
# Íñigo Goiri, Kien Le, Thu D. Nguyen, Jordi Guitart, Jordi Torres, and Ricardo Bianchini. 2012. 
# GreenHadoop: leveraging green energy in data-processing frameworks. 
# In Proceedings of the 7th ACM european conference on Computer Systems (EuroSys '12). 
# Association for Computing Machinery, New York, NY, USA, 57–70. https://doi.org/10.1145/2168836.2168843


import numpy as np
import pandas as pd
from param import *
import math
from agents.agent import Agent
from spark_env.job_dag import JobDAG
from spark_env.node import Node


class GreenHadoopThetaAgent(Agent):
    
    def __init__(self, exec_cap, renewable_dict, theta=0.5):
        Agent.__init__(self)
        
        self.exec_cap_MAX = exec_cap
        self.exec_cap = exec_cap
        self.exec_map = {}

        # Parameter for convex combination
        self.theta = theta

        # Sort and store dictionary in ascending order of time
        all_times_sorted = sorted(renewable_dict.keys())
        self.min_time = all_times_sorted[0]
        self.max_time = all_times_sorted[-1]

        # Convert to minute indices
        self.min_minute = self.min_time // 60000  # earliest minute
        self.max_minute = self.max_time // 60000  # latest minute
        size = (self.max_minute - self.min_minute) + 1

        # Initialize array of percent_renewable
        # We'll store float in [0,1]
        self.renewable_dict = renewable_dict
        self.final_window = 1
        self.remaining_work = 0

    
    def set_renewable_schedule(self, renewable_dict):
        self.renewable_dict = renewable_dict

    def get_current_percent_renewable(self, current_time):
        # round the current time to the nearest (previous) time in the keys
        keys = [key for key in self.renewable_dict.keys() if key < current_time]
        if len(keys) > 0:
            current_CI_time = max(keys)
            current_renewable = self.renewable_dict[current_CI_time]
            # get the keys for the 48 slots 
            # if current_renewable is NaN, set it to 0
            if np.isnan(current_renewable):
                current_renewable = 0
            return current_renewable
        # else just return the first value
        else:
            current_renewable = self.renewable_dict[list(self.renewable_dict.keys())[0]]
            if np.isnan(current_renewable):
                current_renewable = 0
            return current_renewable


    def compute_job_workload(self, job_dags):
        """
        Compute the total remaining workload across all job DAGs,
        measured by the sum of *remaining task durations* (not just task counts).
        """
        total_duration = 0.0
        for job_dag in job_dags:
            if job_dag.completed:
                continue
            for node in job_dag.nodes:
                # For each node, consider only tasks that haven't finished yet
                for task_idx in range(node.next_task_idx, node.num_tasks):
                    task = node.tasks[task_idx]
                    # 'get_duration()' returns how many units of time are still needed
                    total_duration += task.get_duration()
        return total_duration//60000


    def compute_lookahead_window(self, current_time, total_work):
        # 1) Start with partial leftover in the current minute
        # time_minute = current_time // 60000
        
        # fraction of the current minute that remains
        percent_remaining_time = (60000 - (current_time % 60000)) / 60000.0
        accumulated_green_capacity = percent_remaining_time * self.get_current_percent_renewable(current_time) * self.exec_cap_MAX
        accumulated_capacity = percent_remaining_time * self.exec_cap_MAX

        # 2) Move minute by minute
        green_window_count = 0
        brown_window_count = 0
        future_keys = [keys for keys in self.renewable_dict.keys() if keys > current_time]

        for k in future_keys:
            if accumulated_capacity >= total_work and accumulated_green_capacity >= total_work:
                break
            if accumulated_green_capacity < total_work:
                # each minute has some green capacity
                p_renew = self.renewable_dict[k]
                green_capacity = int(np.floor(p_renew * self.exec_cap_MAX))
                accumulated_green_capacity += green_capacity
                green_window_count += 1
            if accumulated_capacity < total_work:
                # also add brown capacity
                accumulated_capacity += self.exec_cap_MAX
                brown_window_count += 1

        return green_window_count, brown_window_count


    def get_lookahead_window(self, current_time, job_dags):
        """
        1) Compute total_work in queue
        2) Find green_lookahead_window & normal_lookahead_window
        3) Return a convex combination:
               lookahead_window = theta * green_window + (1 - theta) * normal_window
        """
        total_work = self.compute_job_workload(job_dags)
        if total_work <= 0:
            return 0  # no work => no lookahead needed

        # green_window = self.compute_green_lookahead_window(current_time, total_work)
        # normal_window = self.compute_normal_lookahead_window(current_time, total_work)
        green_window, normal_window = self.compute_lookahead_window(current_time, total_work)
        # Convex combination
        combined = (1 - self.theta) *  green_window + self.theta * normal_window
        return int(math.ceil(combined))


    def get_lookahead_keys(self, current_time, lookahead_window):
        """
        Return the next lookahead_window keys (timesteps) starting from current_time.
        We'll simply take the first `lookahead_window` items from future_keys.
        """
        future_keys = [key for key in self.renewable_dict.keys() if key > current_time]
        # Slice up to lookahead_window elements
        return future_keys[:lookahead_window]
    

    def get_action(self, obs):
        """
        1) Determine effective lookahead window via the convex combination (theta).
        2) Gather the next X keys from the dictionary.
        3) Summarize total green vs brown across them; pick an executor_cap for this step.
        4) Use FIFO scheduling (source_job first).
        """

        job_dags, source_job, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, action_map, current_time = obs

        if not job_dags:
            # No jobs => do nothing
            return None, num_source_exec, False
        
        exec_map_changed = False
        # Bookkeeping: maintain exec_map
        for job_dag in job_dags:
            if job_dag not in self.exec_map:
                self.exec_map[job_dag] = 0
                exec_map_changed = True
        for job_dag in list(self.exec_map):
            if job_dag not in job_dags:
                del self.exec_map[job_dag]
                exec_map_changed = True

        if exec_map_changed:
            # If exec_map changed, we need to recompute the lookahead window
            # to reflect the new set of job_dags
            self.final_window = self.get_lookahead_window(current_time, job_dags)
            self.remaining_work = self.compute_job_workload(job_dags)

        if self.final_window <= 0:
            # fallback to a naive approach
            current_p = self.get_current_percent_renewable(current_time)
            self.exec_cap = int(current_p * self.exec_cap_MAX)
        else:
            # 2) Gather next final_window keys
            keys_slice = self.get_lookahead_keys(current_time, self.final_window)
            if not keys_slice:
                # If empty, fallback
                current_p = self.get_current_percent_renewable(current_time)
                self.exec_cap = int(current_p * self.exec_cap_MAX)
            else:
                # Sum up green vs. brown
                total_green = 0.0
                total_brown = 0.0
                for k in keys_slice:
                    p_renew = self.renewable_dict[k]
                    total_green += p_renew * self.exec_cap_MAX
                    total_brown += (1.0 - p_renew) * self.exec_cap_MAX
                # Simple heuristic: 
                current_p = self.get_current_percent_renewable(current_time)
                brown_workload = (self.remaining_work - total_green)/self.final_window
                
                # Example: combine current step's renewables with ratio_green
                # This sets how many executors we spin up now.
                candidate = (current_p + (brown_workload * (1-current_p))/((1-current_p) * self.exec_cap_MAX)) * self.exec_cap_MAX
                # if candidate is nan, skip
                if np.isnan(candidate):
                    self.exec_cap = 1
                else:
                    self.exec_cap = max( int(candidate), 1 )

        num_exec = 0
        for job_dag in list(self.exec_map):
            if job_dag.name != 'dummy':
                num_exec += self.exec_map[job_dag]    

        if num_exec >= self.exec_cap:
            if source_job is not None and source_job in list(self.exec_map):
                self.exec_map[source_job] = max(self.exec_map[source_job] - num_source_exec, 0)
            return None, num_source_exec, True

        # 3) FIFO scheduling (source_job first)
        # Try the frontier node of source_job
        if source_job is not None:
            for node in source_job.frontier_nodes:
                if node in frontier_nodes:
                    diff = self.exec_cap - num_exec - num_source_exec
                    if diff < 1:
                       
                        return None, num_source_exec, True
                    return node, min(diff, num_source_exec), False
            # or any node in source_job that is schedulable
            for node in frontier_nodes:
                if node.job_dag == source_job:
                    diff = self.exec_cap - num_exec - num_source_exec
                    if diff < 1:
                        return None, num_source_exec, True
                    return node, min(diff, num_source_exec), False

        # If no source_job or it's done, pick next job in FIFO order
        for job_dag in job_dags:
            if self.exec_map[job_dag] < self.exec_cap:
                next_node = None
                # pick a frontier node first
                for node in job_dag.frontier_nodes:
                    if node in frontier_nodes:
                        next_node = node
                        break
                # fallback: any schedulable node
                if next_node is None:
                    for node in frontier_nodes:
                        if node in job_dag.nodes:
                            next_node = node
                            break
                if next_node is not None:
                    use_exec_init = min(
                        next_node.num_tasks - next_node.next_task_idx
                        - exec_commit.node_commit[next_node]
                        - moving_executors.count(next_node),
                        self.exec_cap - self.exec_map[job_dag],
                        num_source_exec
                    )
                    if use_exec_init < 1:
                        # return dummy task
                        return None, num_source_exec, True
                    use_exec = use_exec_init
                    self.exec_map[job_dag] += use_exec
                    return next_node, use_exec, False

        # No node schedulable
        return None, num_source_exec, False
