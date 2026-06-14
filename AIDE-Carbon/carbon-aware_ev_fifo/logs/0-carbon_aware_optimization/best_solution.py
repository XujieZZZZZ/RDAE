# fifo_agent.py implements pure FIFO scheduling (no carbon awareness).
# === BEGIN CARBON-AWARE LOGIC ===
import numpy as np
from agents.agent import Agent
import math
from collections import defaultdict
import heapq
from typing import List, Tuple, Optional

class NewAgent(Agent):
    def __init__(self, exec_cap, carbon_schedule, exec_lower_bound=20):
        Agent.__init__(self)
        self.exec_map = {}
        self.carbon_schedule = carbon_schedule
        self.exec_cap_MAX = exec_cap
        self.exec_cap = exec_cap
        self.B = exec_lower_bound
        
        # Carbon budget tracking
        self.job_carbon_budgets = {}  # job_dag -> remaining carbon budget
        self.job_carbon_debt = {}     # job_dag -> accumulated carbon debt
        self.job_remaining_tasks = {} # job_dag -> number of remaining frontier tasks
        self.job_last_allocation = {} # job_dag -> last executor allocation count
        
        # Rolling carbon statistics
        self.carbon_history = []      # list of (timestamp, carbon_intensity)
        self.max_history_size = 1440  # 24 hours at 1-minute resolution
        self.rolling_avg_window = 720 # 12 hours
        
        # Starvation prevention
        self.job_wait_times = {}      # job_dag -> time since last executor allocation
        self.starvation_threshold_ms = 300000  # 5 minutes
        
        # Budget parameters
        self.budget_scale_factor = 1.5
        self.budget_min_ratio = 0.2
        self.budget_max_ratio = 1.5
        self.debt_repay_rate = 0.8
        
    def get_carbon_intensity(self, current_time):
        current_carbon = self.carbon_schedule.get(current_time, 300)
        
        # Update rolling carbon history
        self.carbon_history.append((current_time, current_carbon))
        cutoff_time = current_time - self.max_history_size * 60000  # approximate minute
        self.carbon_history = [(t, c) for t, c in self.carbon_history if t >= cutoff_time]
        if len(self.carbon_history) > self.max_history_size:
            self.carbon_history = self.carbon_history[-self.max_history_size:]
        
        # Compute rolling average carbon
        if len(self.carbon_history) >= 60:
            recent_history = self.carbon_history[-self.rolling_avg_window:] if len(self.carbon_history) > self.rolling_avg_window else self.carbon_history
            rolling_avg = np.mean([c for _, c in recent_history])
        else:
            rolling_avg = current_carbon
        
        # Compute bounds based on recent variability
        if len(self.carbon_history) >= 60:
            recent_carbons = [c for _, c in self.carbon_history[-60:]]
            std = np.std(recent_carbons)
            L = max(0, rolling_avg - 2 * std)
            U = rolling_avg + 2 * std
            if L >= U:
                L = U - 10
        else:
            L = current_carbon * 0.7
            U = current_carbon * 1.3
            if L >= U:
                L = U - 10
        
        return current_carbon, L, U
    
    def _compute_carbon_ratio(self, current_time):
        """Compute ratio of current carbon to rolling average."""
        current_carbon = self.carbon_schedule.get(current_time, 300)
        
        if len(self.carbon_history) >= 60:
            recent_history = self.carbon_history[-self.rolling_avg_window:] if len(self.carbon_history) > self.rolling_avg_window else self.carbon_history
            rolling_avg = np.mean([c for _, c in recent_history])
            if rolling_avg > 0:
                return current_carbon / rolling_avg
        return 1.0
    
    def _compute_job_remaining_work(self, job_dag):
        """Estimate remaining work for a job based on frontier nodes."""
        if not hasattr(job_dag, 'frontier_nodes') or not job_dag.frontier_nodes:
            return 0
        
        total_tasks = 0
        for node in job_dag.frontier_nodes:
            if hasattr(node, 'num_tasks') and hasattr(node, 'next_task_idx'):
                remaining = node.num_tasks - node.next_task_idx
                total_tasks += remaining
            else:
                total_tasks += 1
        
        return total_tasks
    
    def _update_carbon_budgets(self, current_time, job_dags):
        """Update carbon budgets for all active jobs based on current conditions."""
        carbon_ratio = self._compute_carbon_ratio(current_time)
        
        # Determine budget adjustment factor
        if carbon_ratio > 1.2:
            budget_factor = self.budget_min_ratio
        elif carbon_ratio < 0.8:
            budget_factor = self.budget_max_ratio
        else:
            # Linear interpolation between 0.2 and 1.5
            budget_factor = self.budget_min_ratio + (self.budget_max_ratio - self.budget_min_ratio) * (1.2 - carbon_ratio) / (1.2 - 0.8)
            budget_factor = max(self.budget_min_ratio, min(self.budget_max_ratio, budget_factor))
        
        # Update budgets for each job
        for job_dag in job_dags:
            remaining_tasks = self._compute_job_remaining_work(job_dag)
            self.job_remaining_tasks[job_dag] = remaining_tasks
            
            if remaining_tasks > 0:
                # Base budget proportional to remaining work
                base_budget = remaining_tasks * 2  # Assume 2 executors per task as baseline
                
                # Apply carbon ratio adjustment
                adjusted_budget = base_budget * budget_factor
                
                # Apply debt reduction
                debt = self.job_carbon_debt.get(job_dag, 0)
                if debt > 0:
                    adjusted_budget = max(adjusted_budget * self.debt_repay_rate, adjusted_budget - debt)
                
                # Ensure minimum budget
                adjusted_budget = max(1, adjusted_budget)
                
                self.job_carbon_budgets[job_dag] = adjusted_budget
            else:
                self.job_carbon_budgets[job_dag] = 0
    
    def _allocate_executors_from_budget(self, job_dag, available_exec, current_time):
        """Allocate executors to a job based on its carbon budget."""
        budget = self.job_carbon_budgets.get(job_dag, 0)
        remaining_tasks = self.job_remaining_tasks.get(job_dag, 0)
        
        if remaining_tasks <= 0 or budget <= 0:
            return 0
        
        # Compute executor allocation proportional to budget per remaining task
        alloc = min(available_exec, max(1, int(budget / remaining_tasks)))
        
        # Cap allocation to prevent over-allocation
        alloc = min(alloc, remaining_tasks * 2)
        
        return alloc
    
    def _check_starvation(self, job_dag, current_time):
        """Check if a job is starving for executors."""
        last_alloc_time = self.job_wait_times.get(job_dag, current_time)
        wait_time = current_time - last_alloc_time
        
        if wait_time >= self.starvation_threshold_ms:
            return True
        return False
    
    def _update_debt(self, job_dag, allocated_exec, current_time):
        """Update carbon debt based on executor usage."""
        carbon_ratio = self._compute_carbon_ratio(current_time)
        
        # If carbon ratio is high, allocate more debt for each executor used
        debt_increase = allocated_exec * max(0, carbon_ratio - 1.0) * 10
        
        current_debt = self.job_carbon_debt.get(job_dag, 0)
        if debt_increase > 0:
            self.job_carbon_debt[job_dag] = current_debt + debt_increase
        else:
            # Repay debt during low carbon periods
            repayment = abs(debt_increase) * 0.5
            self.job_carbon_debt[job_dag] = max(0, current_debt - repayment)
    
    def get_action(self, obs):
        job_dags, source_job, num_source_exec, frontier_nodes, executor_limits, exec_commit, moving_executors, action_map, current_time = obs
        
        # Initialize exec_map for new jobs
        for job_dag in job_dags:
            if job_dag not in self.exec_map:
                self.exec_map[job_dag] = 0
                self.job_carbon_budgets[job_dag] = 0
                self.job_carbon_debt[job_dag] = 0
                self.job_remaining_tasks[job_dag] = 0
                self.job_last_allocation[job_dag] = 0
                self.job_wait_times[job_dag] = current_time
        
        # Remove stale jobs
        for job_dag in list(self.exec_map):
            if job_dag not in job_dags:
                del self.exec_map[job_dag]
                del self.job_carbon_budgets[job_dag]
                del self.job_carbon_debt[job_dag]
                del self.job_remaining_tasks[job_dag]
                del self.job_last_allocation[job_dag]
                del self.job_wait_times[job_dag]
        
        # Update exec_map for source job
        if source_job is not None and source_job in self.exec_map:
            self.exec_map[source_job] = max(self.exec_map[source_job] - num_source_exec, 0)
        
        # Get carbon state
        current_carbon, L, U = self.get_carbon_intensity(current_time)
        
        # Update carbon budgets for all active jobs
        self._update_carbon_budgets(current_time, job_dags)
        
        # Update wait times for jobs that haven't received executors
        for job_dag in job_dags:
            if self.exec_map.get(job_dag, 0) == 0:
                if job_dag not in self.job_wait_times:
                    self.job_wait_times[job_dag] = current_time
            else:
                self.job_wait_times[job_dag] = current_time
        
        # Collect all candidate nodes from frontier
        all_candidates = list(frontier_nodes)
        
        if not all_candidates or num_source_exec <= 0:
            return None, num_source_exec, False
        
        # Compute available executors globally
        total_used_exec = sum(self.exec_map.values())
        available_global = self.exec_cap - total_used_exec
        
        if available_global <= 0:
            return None, num_source_exec, False
        
        # Build candidate list with job information
        candidates_with_jobs = []
        for node in all_candidates:
            job_dag = node.job_dag
            candidates_with_jobs.append((node, job_dag))
        
        # Phase 1: Check for starving jobs first
        starving_jobs = []
        for node, job_dag in candidates_with_jobs:
            if self._check_starvation(job_dag, current_time):
                starving_jobs.append((node, job_dag))
        
        if starving_jobs:
            # Allocate 1 executor to a starving job
            node, job_dag = starving_jobs[0]
            job_used = self.exec_map.get(job_dag, 0)
            available_for_job = min(available_global, self.exec_cap - job_used)
            
            if available_for_job > 0:
                alloc = min(1, num_source_exec, available_for_job)
                if alloc > 0:
                    self.exec_map[job_dag] = job_used + alloc
                    self.job_wait_times[job_dag] = current_time
                    self._update_debt(job_dag, alloc, current_time)
                    return node, alloc, True
        
        # Phase 2: Sort candidates by remaining work (most work first) for fairness
        candidates_with_work = []
        for node, job_dag in candidates_with_jobs:
            remaining_tasks = self.job_remaining_tasks.get(job_dag, 0)
            if remaining_tasks > 0:
                candidates_with_work.append((remaining_tasks, node, job_dag))
        
        # Sort by remaining work descending to prioritize jobs with more to do
        candidates_with_work.sort(key=lambda x: (-x[0]))
        
        # Phase 3: Allocate executors based on carbon budgets
        allocated_this_round = 0
        selected_node = None
        use_exec = 0
        carbon_aware_flag = False
        
        for remaining_tasks, node, job_dag in candidates_with_work:
            if allocated_this_round >= num_source_exec:
                break
            
            job_used = self.exec_map.get(job_dag, 0)
            available_for_job = min(available_global, self.exec_cap - job_used)
            
            if available_for_job <= 0:
                continue
            
            # Allocate executors from budget
            alloc = self._allocate_executors_from_budget(job_dag, min(num_source_exec - allocated_this_round, available_for_job), current_time)
            
            if alloc <= 0:
                continue
            
            selected_node = node
            use_exec = alloc
            self.exec_map[job_dag] = job_used + alloc
            allocated_this_round += alloc
            carbon_aware_flag = True
            self.job_wait_times[job_dag] = current_time
            self._update_debt(job_dag, alloc, current_time)
            break
        
        # Fallback: if no task selected, pick any available with minimal allocation
        if selected_node is None and all_candidates:
            for node in all_candidates:
                job_dag = node.job_dag
                job_used = self.exec_map.get(job_dag, 0)
                available_for_job = min(available_global, self.exec_cap - job_used)
                if available_for_job > 0:
                    alloc = min(1, num_source_exec - allocated_this_round, available_for_job)
                    if alloc > 0:
                        selected_node = node
                        use_exec = alloc
                        self.exec_map[job_dag] = job_used + alloc
                        carbon_aware_flag = True
                        self.job_wait_times[job_dag] = current_time
                        self._update_debt(job_dag, alloc, current_time)
                        break
        
        if selected_node is not None:
            return selected_node, use_exec, carbon_aware_flag
        
        return None, num_source_exec, False
# === END CARBON-AWARE LOGIC ===