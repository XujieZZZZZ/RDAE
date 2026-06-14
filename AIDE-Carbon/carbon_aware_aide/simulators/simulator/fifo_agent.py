# fifo_agent.py implements pure FIFO scheduling (no carbon awareness).
# === BEGIN CARBON-AWARE LOGIC ===
import numpy as np
from agents.agent import Agent


class FifoAgent(Agent):
    # statically partition the cluster resource
    # scheduling complexity: O(num_nodes * num_executors)
    def __init__(self, exec_cap, carbon_schedule, exec_lower_bound=20):
        Agent.__init__(self)

        # map for executor assignment
        self.exec_map = {}

        self.B = exec_lower_bound

        # executor limit set to each job
        self.exec_cap_MAX = exec_cap
        self.exec_cap = self.exec_cap_MAX

    def get_action(self, obs):

        # parse observation
        job_dags, source_job, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, action_map, current_time = obs

        # sort out the new exec_map
        for job_dag in job_dags:
            if job_dag not in self.exec_map:
                self.exec_map[job_dag] = 0
        for job_dag in list(self.exec_map):
            if job_dag not in job_dags:
                del self.exec_map[job_dag]

        # look at how many executors are currently up and running
        num_exec = 0
        for job_dag in list(self.exec_map):
            num_exec += self.exec_map[job_dag]

        # the source job is finished or does not exist
        if num_exec >= self.exec_cap:
            # we need to pause execution, so just return a null action
            if source_job is not None and source_job in list(self.exec_map):
                self.exec_map[source_job] = max(self.exec_map[source_job] - num_source_exec, 0)
            return None, num_source_exec, True

        scheduled = False
        # first assign executor to the same job
        if source_job is not None:
            # immediately scheduable nodes
            for node in source_job.frontier_nodes:
                if node in frontier_nodes:
                    return node, num_source_exec, False

            # schedulable node in the job
            for node in frontier_nodes:
                if node.job_dag == source_job:
                    return node, num_source_exec, False

        for job_dag in job_dags:
            if self.exec_map[job_dag] < self.exec_cap:
                next_node = None
                # immediately scheduable node first
                for node in job_dag.frontier_nodes:
                    if node in frontier_nodes:
                        next_node = node
                        break
                # then schedulable node in the job
                if next_node is None:
                    for node in frontier_nodes:
                        if node in job_dag.nodes:
                            next_node = node
                            break
                # node is selected, compute limit
                if next_node is not None:
                    use_exec_init = min(
                        node.num_tasks - node.next_task_idx - \
                        exec_commit.node_commit[node] - \
                        moving_executors.count(node),
                        num_source_exec)
                    use_exec = use_exec_init
                    self.exec_map[job_dag] += use_exec
                    return node, use_exec, False

        return None, num_source_exec, False
# === END CARBON-AWARE LOGIC ===