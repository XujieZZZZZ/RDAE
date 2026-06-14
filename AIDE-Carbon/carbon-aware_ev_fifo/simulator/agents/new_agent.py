# fifo_agent.py implements pure FIFO scheduling (no carbon awareness).
# === BEGIN CARBON-AWARE LOGIC ===
import numpy as np
from agents.agent import Agent
import math
from collections import defaultdict, deque
from typing import List, Tuple, Optional

class NewAgent(Agent):
    def __init__(self, exec_cap, carbon_schedule, exec_lower_bound=20):
        Agent.__init__(self)
        self.exec_map = {}
        self.carbon_schedule = carbon_schedule
        self.exec_cap_MAX = exec_cap
        self.exec_cap = exec_cap
        self.B = exec_lower_bound
        
        # Carbon history and prediction
        self.carbon_history = deque(maxlen=1000)
        self.carbon_buffer = deque(maxlen=100)
        self.clean_window_cache = {}
        self.last_cache_update = 0
        
        # Task tracking
        self.task_flexibility = {}
        self.job_remaining_tasks = defaultdict(int)
        self.job_last_execution = {}
        self.job_starvation_counter = defaultdict(int)
        self.job_carbon_debt = defaultdict(float)
        
        # Performance tracking
        self.total_allocated = 0
        self.avg_inter_execution = 60000  # 1 minute initial estimate
        
    def get_carbon_intensity(self, current_time):
        current_carbon = self.carbon_schedule.get(current_time, 300)
        
        # Update history
        self.carbon_history.append((current_time, current_carbon))
        self.carbon_buffer.append(current_carbon)
        
        # Compute rolling statistics
        if len(self.carbon_buffer) >= 10:
            recent = list(self.carbon_buffer)
            mean = np.mean(recent)
            std = np.std(recent)
            L = max(0, mean - 1.5 * std)
            U = mean + 1.5 * std
            if L >= U:
                L = U - 10
        else:
            L = current_carbon * 0.8
            U = current_carbon * 1.2
            if L >= U:
                L = U - 10
        
        return current_carbon, L, U
    
    def _predict_clean_window(self, current_time):
        """Predict next clean window by analyzing carbon schedule."""
        if current_time - self.last_cache_update < 60000 and self.clean_window_cache:
            return self.clean_window_cache
        
        # Get recent carbon statistics
        if len(self.carbon_buffer) < 20:
            self.clean_window_cache = (current_time + 300000, 120000)  # Default: 5 min start, 2 min duration
            self.last_cache_update = current_time
            return self.clean_window_cache
        
        recent_carbons = list(self.carbon_buffer)
        median = np.median(recent_carbons)
        p25 = np.percentile(recent_carbons, 25)
        
        # Look ahead in schedule
        look_ahead = 3600000  # 1 hour in ms
        step = 60000  # 1 minute steps
        best_start = None
        best_duration = 0
        current_start = None
        current_duration = 0
        
        for t in range(current_time, current_time + look_ahead, step):
            carbon = self.carbon_schedule.get(t, median)
            
            if carbon <= p25:
                if current_start is None:
                    current_start = t
                current_duration += step
            else:
                if current_duration > best_duration and current_duration >= 300000:  # At least 5 min
                    best_start = current_start
                    best_duration = current_duration
                current_start = None
                current_duration = 0
        
        # Check last window
        if current_duration > best_duration and current_duration >= 300000:
            best_start = current_start
            best_duration = current_duration
        
        if best_start is None:
            # No clean window found, use default
            best_start = current_time + 600000  # 10 min from now
            best_duration = 120000  # 2 min
        
        self.clean_window_cache = (best_start, best_duration)
        self.last_cache_update = current_time
        return self.clean_window_cache
    
    def _compute_flexibility(self, job_dag, node, current_time):
        """Compute how flexible a task is (0 = critical, 1 = very flexible)."""
        key = (job_dag, node)
        if key in self.task_flexibility:
            return self.task_flexibility[key]
        
        # Estimate remaining work and criticality
        if hasattr(job_dag, 'num_tasks') and hasattr(node, 'next_task_idx'):
            total_tasks = job_dag.num_tasks
            completed = node.next_task_idx
            remaining = total_tasks - completed
        else:
            remaining = 1
        
        # Estimate slack based on job progress
        if hasattr(job_dag, 'start_time') and hasattr(job_dag, 'expected_duration'):
            elapsed = current_time - job_dag.start_time
            expected = job_dag.expected_duration
            progress_ratio = elapsed / max(expected, 1)
            
            if progress_ratio < 0.3:
                # Early in job - more flexibility
                flexibility = min(1.0, remaining / 10.0)
            elif progress_ratio < 0.7:
                # Middle of job - moderate flexibility
                flexibility = min(0.7, remaining / 15.0)
            else:
                # Late in job - less flexibility
                flexibility = min(0.3, remaining / 20.0)
        else:
            # Default heuristic
            flexibility = min(1.0, 1.0 / max(remaining, 1))
        
        # Adjust for starvation
        last_time = self.job_last_execution.get(job_dag, current_time)
        wait_time = current_time - last_time
        if wait_time > self.avg_inter_execution * 3:
            flexibility = max(0.1, flexibility - 0.3)
        
        self.task_flexibility[key] = flexibility
        return flexibility
    
    def _compute_allocation_fraction(self, current_carbon, max_carbon, flexibility):
        """Compute allocation fraction based on carbon and flexibility."""
        if max_carbon == 0:
            max_carbon = 1
        
        carbon_ratio = current_carbon / max_carbon
        
        # High carbon + high flexibility = very low allocation
        # Low carbon + low flexibility = high allocation
        urgency = 1 - flexibility
        carbon_penalty = carbon_ratio
        
        fraction = urgency * (1 - carbon_penalty * 0.7)
        return max(0.1, min(1.0, fraction))
    
    def get_action(self, obs):
        job_dags, source_job, num_source_exec, frontier_nodes, executor_limits, exec_commit, moving_executors, action_map, current_time = obs
        
        # Initialize tracking for new jobs
        for job_dag in job_dags:
            if job_dag not in self.exec_map:
                self.exec_map[job_dag] = 0
                self.job_last_execution[job_dag] = current_time
                self.job_starvation_counter[job_dag] = 0
            
            # Update remaining tasks
            if hasattr(job_dag, 'num_tasks') and hasattr(job_dag, 'completed_tasks'):
                self.job_remaining_tasks[job_dag] = job_dag.num_tasks - job_dag.completed_tasks
        
        # Clean up stale jobs
        for job_dag in list(self.exec_map):
            if job_dag not in job_dags:
                del self.exec_map[job_dag]
                del self.job_last_execution[job_dag]
                del self.job_starvation_counter[job_dag]
                del self.job_carbon_debt[job_dag]
                if job_dag in self.job_remaining_tasks:
                    del self.job_remaining_tasks[job_dag]
        
        # Update exec_map for source job
        if source_job is not None and source_job in self.exec_map:
            self.exec_map[source_job] = max(self.exec_map[source_job] - num_source_exec, 0)
        
        # Get carbon state
        current_carbon, L, U = self.get_carbon_intensity(current_time)
        
        # Compute available executors
        total_used = sum(self.exec_map.values())
        available_global = self.exec_cap - total_used
        
        if available_global <= 0 or num_source_exec <= 0 or not frontier_nodes:
            return None, num_source_exec, False
        
        # Build candidate list with flexibility scores
        candidates = []
        for node in frontier_nodes:
            job_dag = node.job_dag
            if job_dag not in job_dags:
                continue
            
            flexibility = self._compute_flexibility(job_dag, node, current_time)
            job_used = self.exec_map.get(job_dag, 0)
            
            if job_used >= self.exec_cap:
                continue
            
            candidates.append((node, job_dag, flexibility))
        
        if not candidates:
            return None, num_source_exec, False
        
        # Get carbon statistics for scaling
        if len(self.carbon_buffer) > 0:
            max_carbon = max(self.carbon_buffer)
            min_carbon = min(self.carbon_buffer)
        else:
            max_carbon = current_carbon + 100
            min_carbon = max(0, current_carbon - 100)
        
        carbon_range = max_carbon - min_carbon if max_carbon > min_carbon else 1
        carbon_normalized = (current_carbon - min_carbon) / carbon_range
        
        # Predict clean window
        clean_start, clean_duration = self._predict_clean_window(current_time)
        time_to_clean = clean_start - current_time
        
        # Phase 1: Starvation prevention
        starving_candidates = []
        for node, job_dag, flexibility in candidates:
            last_time = self.job_last_execution.get(job_dag, current_time)
            wait_time = current_time - last_time
            starvation_threshold = self.avg_inter_execution * (3 + flexibility * 2)
            
            if wait_time > starvation_threshold:
                starving_candidates.append((wait_time, node, job_dag, flexibility))
        
        if starving_candidates:
            # Sort by wait time (longest first)
            starving_candidates.sort(key=lambda x: -x[0])
            
            for wait_time, node, job_dag, flexibility in starving_candidates:
                job_used = self.exec_map.get(job_dag, 0)
                available_for_job = min(available_global, self.exec_cap - job_used)
                
                if available_for_job > 0:
                    alloc = min(1, num_source_exec, available_for_job)
                    if alloc > 0:
                        self.exec_map[job_dag] = job_used + alloc
                        self.job_last_execution[job_dag] = current_time
                        self.job_starvation_counter[job_dag] = 0
                        
                        # Update average inter-execution time
                        self.avg_inter_execution = 0.9 * self.avg_inter_execution + 0.1 * wait_time
                        
                        return node, alloc, True
        
        # Phase 2: Carbon-aware proportional allocation
        # Sort candidates by flexibility (least flexible first)
        candidates.sort(key=lambda x: x[2])
        
        allocated_this_round = 0
        selected_node = None
        use_exec = 0
        carbon_aware_flag = False
        
        for node, job_dag, flexibility in candidates:
            if allocated_this_round >= num_source_exec:
                break
            
            job_used = self.exec_map.get(job_dag, 0)
            available_for_job = min(available_global - allocated_this_round, self.exec_cap - job_used)
            
            if available_for_job <= 0:
                continue
            
            # Compute allocation fraction
            fraction = self._compute_allocation_fraction(carbon_normalized, 1.0, flexibility)
            
            # Adjust for clean window proximity
            if time_to_clean > 0 and time_to_clean < 600000:  # Within 10 min of clean window
                if carbon_normalized > 0.5:  # Currently dirty
                    fraction *= 0.3  # Reduce allocation to wait for clean
                else:
                    fraction *= 1.2  # Slightly increase if already clean
            
            # Base allocation on remaining tasks
            remaining = self.job_remaining_tasks.get(job_dag, 1)
            base_alloc = max(1, int(remaining * fraction))
            
            # Apply carbon debt adjustment
            debt = self.job_carbon_debt.get(job_dag, 0)
            if debt > 0 and carbon_normalized < 0.3:  # Clean period with debt
                debt_multiplier = min(2.0, 1.0 + debt / 100.0)
                base_alloc = int(base_alloc * debt_multiplier)
            
            # Final allocation
            alloc = min(base_alloc, available_for_job, num_source_exec - allocated_this_round)
            alloc = max(1, alloc)  # At least 1 executor
            
            if alloc <= 0:
                continue
            
            selected_node = node
            use_exec = alloc
            self.exec_map[job_dag] = job_used + alloc
            self.job_last_execution[job_dag] = current_time
            self.job_starvation_counter[job_dag] = 0
            
            # Update carbon debt
            if carbon_normalized > 0.7:  # Dirty period
                self.job_carbon_debt[job_dag] += alloc * carbon_normalized * 2
            elif carbon_normalized < 0.3:  # Clean period
                self.job_carbon_debt[job_dag] = max(0, self.job_carbon_debt[job_dag] - alloc * 0.5)
            
            allocated_this_round += alloc
            carbon_aware_flag = True
            break
        
        # Phase 3: Fallback - allocate to job with highest carbon debt
        if selected_node is None:
            best_debt = -1
            best_node = None
            best_job_dag = None
            
            for node, job_dag, flexibility in candidates:
                job_used = self.exec_map.get(job_dag, 0)
                available_for_job = min(available_global, self.exec_cap - job_used)
                
                if available_for_job <= 0:
                    continue
                
                debt = self.job_carbon_debt.get(job_dag, 0)
                if debt > best_debt:
                    best_debt = debt
                    best_node = node
                    best_job_dag = job_dag
            
            if best_node is not None and best_job_dag is not None:
                job_used = self.exec_map.get(best_job_dag, 0)
                alloc = min(1, num_source_exec, available_global)
                if alloc > 0:
                    selected_node = best_node
                    use_exec = alloc
                    self.exec_map[best_job_dag] = job_used + alloc
                    self.job_last_execution[best_job_dag] = current_time
                    carbon_aware_flag = True
        
        if selected_node is not None:
            return selected_node, use_exec, carbon_aware_flag
        
        return None, num_source_exec, False
# === END CARBON-AWARE LOGIC ===