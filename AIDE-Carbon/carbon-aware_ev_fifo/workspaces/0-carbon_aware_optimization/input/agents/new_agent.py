# fifo_agent.py implements pure FIFO scheduling (no carbon awareness).
# === BEGIN CARBON-AWARE LOGIC ===
import numpy as np
from agents.agent import Agent
import heapq
import math
from collections import defaultdict, deque


class NewAgent(Agent):
    def __init__(self, exec_cap, carbon_schedule, exec_lower_bound=20):
        Agent.__init__(self)
        self.exec_map = {}
        self.carbon_schedule = carbon_schedule
        self.exec_cap_MAX = exec_cap
        self.exec_cap = exec_cap
        self.B = exec_lower_bound

        self.carbon_history = []
        self.valley_demand_map = defaultdict(list)
        self.reserved_deferral_queue = defaultdict(deque)
        self.task_criticality = {}
        self.avg_task_duration = 1000.0
        self.planning_horizon = 48
        self.valley_threshold_percentile = 30
        self.valley_min_duration = 3
        self.throttle_alpha = 0.7
        self.throttle_beta = 150.0
        self.carbon_forecast_cache = {}
        self.last_valley_scan_time = -float('inf')
        self.valley_scan_interval = 5000

    def get_carbon_intensity(self, current_time):
        current_carbon = self.carbon_schedule.get(current_time, 0.0)
        if len(self.carbon_history) > 0:
            mean_carbon = np.mean(self.carbon_history)
            std_carbon = np.std(self.carbon_history) if len(self.carbon_history) > 1 else 1.0
            L = mean_carbon - 0.5 * std_carbon
            U = mean_carbon + 0.5 * std_carbon
        else:
            L, U = current_carbon - 15, current_carbon + 15
        if L >= U:
            L, U = U - 1, L + 1
        self.carbon_history.append(current_carbon)
        if len(self.carbon_history) > 500:
            self.carbon_history.pop(0)
        return current_carbon, L, U

    def _get_carbon_forecast(self, current_time, horizon):
        forecast = []
        for i in range(horizon):
            t = current_time + i * 1000
            if t in self.carbon_forecast_cache:
                forecast.append(self.carbon_forecast_cache[t])
            else:
                val = self.carbon_schedule.get(t, 0.0)
                self.carbon_forecast_cache[t] = val
                forecast.append(val)
        return forecast

    def _identify_valleys(self, forecast):
        if len(forecast) < 3:
            return []
        threshold = np.percentile(forecast, self.valley_threshold_percentile)
        valleys = []
        i = 0
        while i < len(forecast):
            if forecast[i] <= threshold:
                start = i
                while i < len(forecast) and forecast[i] <= threshold:
                    i += 1
                end = i - 1
                duration = end - start + 1
                if duration >= self.valley_min_duration:
                    avg_intensity = np.mean(forecast[start:end+1])
                    valleys.append((start, end, avg_intensity, duration))
            else:
                i += 1
        valleys.sort(key=lambda v: v[2])
        return valleys[:5]

    def _compute_criticality(self, node, frontier_nodes, job_dags):
        if node in self.task_criticality:
            return self.task_criticality[node]
        job_frontier = [n for n in frontier_nodes if n.job_dag == node.job_dag]
        if not job_frontier:
            self.task_criticality[node] = 1.0
            return 1.0
        total_remaining = sum(n.num_tasks - n.next_task_idx for n in job_frontier)
        if total_remaining == 0:
            self.task_criticality[node] = 1.0
            return 1.0
        node_remaining = node.num_tasks - node.next_task_idx
        criticality = node_remaining / total_remaining
        self.task_criticality[node] = criticality
        return criticality

    def _update_valley_demand_map(self, current_time, frontier_nodes, job_dags):
        if current_time - self.last_valley_scan_time < self.valley_scan_interval:
            return
        self.last_valley_scan_time = current_time
        forecast = self._get_carbon_forecast(current_time, self.planning_horizon)
        valleys = self._identify_valleys(forecast)
        self.valley_demand_map.clear()
        for valley_start, valley_end, avg_intensity, duration in valleys:
            valley_start_time = current_time + valley_start * 1000
            valley_end_time = current_time + valley_end * 1000
            eligible_tasks = []
            for node in frontier_nodes:
                latest_start = self._estimate_latest_start(node, current_time, job_dags)
                if latest_start >= valley_start_time:
                    exec_needed = node.num_tasks - node.next_task_idx
                    if exec_needed > 0:
                        eligible_tasks.append((node, exec_needed))
            self.valley_demand_map[(valley_start_time, valley_end_time)] = eligible_tasks

    def _estimate_latest_start(self, node, current_time, job_dags):
        job = node.job_dag
        job_nodes = [n for n in job.nodes if n.num_tasks - n.next_task_idx > 0]
        if not job_nodes:
            return current_time
        total_remaining = sum(n.num_tasks - n.next_task_idx for n in job_nodes)
        est_completion = current_time + total_remaining * self.avg_task_duration
        node_remaining = node.num_tasks - node.next_task_idx
        latest_start = est_completion - node_remaining * self.avg_task_duration
        return latest_start

    def _calculate_carbon_advantage_score(self, current_time):
        forecast = self._get_carbon_forecast(current_time, self.planning_horizon)
        if not forecast:
            return 0.0
        background_avg = np.mean(forecast)
        valleys = self._identify_valleys(forecast)
        if not valleys:
            return 0.0
        best_valley = valleys[0]
        valley_start, valley_end, valley_avg, duration = best_valley
        advantage = (background_avg - valley_avg) * duration
        return max(0.0, advantage)

    def _dynamic_capacity_scaling(self, current_time):
        cas = self._calculate_carbon_advantage_score(current_time)
        throttle_factor = self.throttle_alpha * np.tanh(cas / self.throttle_beta)
        effective_cap = self.B + (self.exec_cap_MAX - self.B) * (1.0 - throttle_factor)
        self.exec_cap = int(max(self.B, min(effective_cap, self.exec_cap_MAX)))
        return throttle_factor > 0.1

    def _is_in_valley(self, current_time):
        forecast = self._get_carbon_forecast(current_time, 1)
        if not forecast:
            return False
        current_carbon = forecast[0]
        threshold = np.percentile(self._get_carbon_forecast(current_time, self.planning_horizon), self.valley_threshold_percentile)
        return current_carbon <= threshold

    def get_action(self, obs):
        job_dags, source_job, num_source_exec, frontier_nodes, executor_limits, exec_commit, moving_executors, action_map, current_time = obs

        for job_dag in job_dags:
            if job_dag not in self.exec_map:
                self.exec_map[job_dag] = 0
        for job_dag in list(self.exec_map):
            if job_dag not in job_dags:
                del self.exec_map[job_dag]

        num_exec = sum(self.exec_map.values())
        current_carbon, L, U = self.get_carbon_intensity(current_time)

        self._update_valley_demand_map(current_time, frontier_nodes, job_dags)
        in_valley = self._is_in_valley(current_time)
        valley_reservation_mode = self._dynamic_capacity_scaling(current_time) and not in_valley

        if in_valley:
            self.exec_cap = self.exec_cap_MAX

        if num_exec >= self.exec_cap:
            if source_job is not None and source_job in self.exec_map:
                self.exec_map[source_job] = max(self.exec_map[source_job] - num_source_exec, 0)
            return None, num_source_exec, True

        if in_valley:
            for valley_key, deferred_tasks in list(self.reserved_deferral_queue.items()):
                valley_start, valley_end = valley_key
                if valley_start <= current_time <= valley_end:
                    while deferred_tasks:
                        node, job_dag = deferred_tasks.popleft()
                        if self.exec_map[job_dag] < self.exec_cap:
                            remaining = node.num_tasks - node.next_task_idx
                            exec_needed = min(remaining, self.exec_cap - self.exec_map[job_dag])
                            if exec_needed > 0:
                                self.exec_map[job_dag] += exec_needed
                                return node, exec_needed, True
                    del self.reserved_deferral_queue[valley_key]

        candidate_tasks = []
        for node in frontier_nodes:
            job_dag = node.job_dag
            if self.exec_map[job_dag] >= self.exec_cap:
                continue
            remaining = node.num_tasks - node.next_task_idx
            if remaining <= 0:
                continue
            exec_needed = min(remaining, num_source_exec)
            criticality = self._compute_criticality(node, frontier_nodes, job_dags)

            if valley_reservation_mode and criticality < 0.4:
                deferral_priority = (exec_needed * current_carbon) / (criticality + 0.1)
                best_valley = None
                best_valley_time = float('inf')
                for valley_start, valley_end in self.valley_demand_map.keys():
                    if valley_start < best_valley_time and valley_start > current_time:
                        best_valley_time = valley_start
                        best_valley = (valley_start, valley_end)
                if best_valley:
                    self.reserved_deferral_queue[best_valley].append((node, job_dag))
                    continue

            priority = -criticality
            if not in_valley:
                priority -= current_carbon * 0.01
            else:
                priority += 2.0

            candidate_tasks.append((priority, exec_needed, node, job_dag, criticality))

        candidate_tasks.sort(key=lambda x: x[0], reverse=True)

        for priority, exec_needed, node, job_dag, criticality in candidate_tasks:
            if self.exec_map[job_dag] < self.exec_cap:
                actual_exec = min(exec_needed, self.exec_cap - self.exec_map[job_dag])
                if actual_exec > 0:
                    self.exec_map[job_dag] += actual_exec
                    carbon_aware = valley_reservation_mode or in_valley or (criticality < 0.5)
                    return node, actual_exec, carbon_aware

        if source_job is not None:
            for node in frontier_nodes:
                if node.job_dag == source_job:
                    if self.exec_map[source_job] < self.exec_cap:
                        remaining = node.num_tasks - node.next_task_idx
                        fallback_exec = min(remaining, self.exec_cap - self.exec_map[source_job])
                        if fallback_exec > 0:
                            self.exec_map[source_job] += fallback_exec
                            return node, fallback_exec, False

        for job_dag in job_dags:
            if self.exec_map[job_dag] < self.exec_cap:
                for node in frontier_nodes:
                    if node.job_dag == job_dag:
                        remaining = node.num_tasks - node.next_task_idx
                        fallback_exec = min(remaining, self.exec_cap - self.exec_map[job_dag])
                        if fallback_exec > 0:
                            self.exec_map[job_dag] += fallback_exec
                            return node, fallback_exec, False

        return None, num_source_exec, False
# === END CARBON-AWARE LOGIC ===