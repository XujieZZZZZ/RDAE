import numpy as np
from agents.agent import Agent
from collections import deque


class NewAgent(Agent):
    def __init__(self, exec_cap, carbon_schedule, exec_lower_bound=20):
        Agent.__init__(self)
        self.exec_map = {}
        self.B = exec_lower_bound
        self.exec_cap_MAX = exec_cap
        self.exec_cap = self.exec_cap_MAX
        self.carbon_schedule = carbon_schedule
        self.base_time_per_unit = 1000  # ms per task unit

    def get_carbon_intensity(self, current_time):
        values = list(self.carbon_schedule.values())
        L = min(values)
        U = max(values)
        if L >= U:
            U = L + 1
        current_carbon = self.carbon_schedule.get(current_time, (L + U) / 2)
        return current_carbon, L, U

    def _compute_slack_and_downstream(self, job_dag):
        """Compute slack and downstream count for all nodes in a job DAG."""
        nodes = list(job_dag.nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        n_nodes = len(nodes)

        forward_edges = {n: [] for n in nodes}
        reverse_edges = {n: [] for n in nodes}

        for node in nodes:
            if hasattr(node, "dependencies") and node.dependencies:
                for dep in node.dependencies:
                    if dep in node_to_idx:
                        forward_edges[dep].append(node)
                        reverse_edges[node].append(dep)

        in_degree = {n: len(reverse_edges[n]) for n in nodes}
        queue = deque([n for n in nodes if in_degree[n] == 0])
        topo_order = []
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for succ in forward_edges[node]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        est = {n: 0 for n in nodes}
        ect = {n: 0 for n in nodes}
        for node in topo_order:
            remaining = node.num_tasks - node.next_task_idx
            duration = remaining * self.base_time_per_unit
            if reverse_edges[node]:
                est[node] = max(ect[dep] for dep in reverse_edges[node])
            ect[node] = est[node] + duration

        lst = {n: float("inf") for n in nodes}
        lct = {n: float("inf") for n in nodes}

        for node in nodes:
            if not forward_edges[node]:
                lct[node] = ect[node]
                lst[node] = (
                    lct[node]
                    - (node.num_tasks - node.next_task_idx) * self.base_time_per_unit
                )

        for node in reversed(topo_order):
            if forward_edges[node]:
                lct[node] = min(lst[succ] for succ in forward_edges[node])
                lst[node] = (
                    lct[node]
                    - (node.num_tasks - node.next_task_idx) * self.base_time_per_unit
                )

        slack = {}
        downstream_count = {}
        for node in nodes:
            slack[node] = lst[node] - est[node] if lst[node] != float("inf") else 0
            visited = set()
            queue = deque([node])
            while queue:
                curr = queue.popleft()
                for succ in forward_edges[curr]:
                    if succ not in visited:
                        visited.add(succ)
                        queue.append(succ)
            downstream_count[node] = len(visited)

        return slack, downstream_count

    def get_action(self, obs):
        (
            job_dags,
            source_job,
            num_source_exec,
            frontier_nodes,
            executor_limits,
            exec_commit,
            moving_executors,
            action_map,
            current_time,
        ) = obs

        for job_dag in job_dags:
            if job_dag not in self.exec_map:
                self.exec_map[job_dag] = 0
        for job_dag in list(self.exec_map):
            if job_dag not in job_dags:
                del self.exec_map[job_dag]

        num_exec = sum(self.exec_map.values())
        if num_exec >= self.exec_cap:
            if source_job is not None and source_job in self.exec_map:
                self.exec_map[source_job] = max(
                    self.exec_map[source_job] - num_source_exec, 0
                )
            return None, num_source_exec, True

        if not frontier_nodes or num_source_exec == 0:
            return None, num_source_exec, False

        c_current, L, U = self.get_carbon_intensity(current_time)
        carbon_ratio = (c_current - L) / (U - L) if U > L else 0.5
        carbon_ratio = max(0.0, min(1.0, carbon_ratio))

        job_slack = {}
        job_downstream = {}
        for job_dag in job_dags:
            slack, downstream = self._compute_slack_and_downstream(job_dag)
            job_slack[job_dag] = slack
            job_downstream[job_dag] = downstream

        penalties = {}
        is_critical = {}
        for node in frontier_nodes:
            job_dag = node.job_dag
            if job_dag in job_slack:
                sl = job_slack[job_dag].get(node, 0)
                ds = job_downstream[job_dag].get(node, 0)
                if sl <= 0:
                    penalty = 1.0 * (ds + 1)
                    is_critical[node] = True
                else:
                    penalty = (1.0 / (1.0 + sl)) * (ds + 1)
                    is_critical[node] = False
            else:
                penalty = 1.0
                is_critical[node] = False
            penalties[node] = penalty

        if penalties:
            max_penalty = max(penalties.values())
            if max_penalty > 0:
                for node in penalties:
                    penalties[node] /= max_penalty

        critical_exists = any(is_critical.get(node, False) for node in frontier_nodes)

        # Only defer when carbon is very high and no critical tasks exist
        if carbon_ratio > 0.85 and not critical_exists:
            min_penalty = min(penalties.values()) if penalties else 1.0
            if min_penalty < 0.2:  # Only defer very low-impact tasks
                if source_job is not None and source_job in self.exec_map:
                    self.exec_map[source_job] = max(
                        self.exec_map[source_job] - num_source_exec, 0
                    )
                return None, num_source_exec, True

        best_node = None
        best_score = float("inf")
        best_exec = 1

        for node in frontier_nodes:
            job_dag = node.job_dag
            remaining = node.num_tasks - node.next_task_idx
            base_duration = remaining * self.base_time_per_unit

            committed = exec_commit.node_commit.get(node, 0)
            moving = moving_executors.count(node)
            job_cap = executor_limits.get(job_dag, self.exec_cap_MAX)
            max_exec = min(num_source_exec, job_cap - committed - moving)
            max_exec = max(1, max_exec)

            # For critical tasks, use maximum executors to minimize delay
            if is_critical.get(node, False):
                a = max_exec
                duration = base_duration / a
                carbon_cost = c_current * duration
                time_penalty = penalties[node] * duration
                total_score = time_penalty + 0.1 * carbon_cost  # Low carbon weight
                if total_score < best_score:
                    best_score = total_score
                    best_node = node
                    best_exec = a
            else:
                # For non-critical tasks, try minimal executors to save carbon
                for a in range(1, max_exec + 1):
                    duration = base_duration / a
                    carbon_cost = c_current * duration
                    time_penalty = penalties[node] * duration
                    # Weight carbon heavily for non-critical tasks
                    total_score = time_penalty + 0.9 * carbon_cost
                    if total_score < best_score:
                        best_score = total_score
                        best_node = node
                        best_exec = a

        if best_node is not None:
            job_dag = best_node.job_dag
            committed = exec_commit.node_commit.get(best_node, 0)
            moving = moving_executors.count(best_node)
            job_cap = executor_limits.get(job_dag, self.exec_cap_MAX)
            max_exec = min(num_source_exec, job_cap - committed - moving)
            max_exec = max(1, max_exec)
            use_exec = min(best_exec, max_exec)
            use_exec = max(1, use_exec)

            self.exec_map[job_dag] = self.exec_map.get(job_dag, 0) + use_exec
            carbon_aware_flag = carbon_ratio > 0.5 and not is_critical.get(
                best_node, False
            )
            return best_node, use_exec, carbon_aware_flag

        return None, num_source_exec, False
