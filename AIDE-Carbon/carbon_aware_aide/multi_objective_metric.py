# multi_objective_metric.py - Multi-objective optimization support
import numpy as np
from typing import List, Optional, Dict
from aide.utils.metric import MetricValue

class MultiObjectiveMetric:
    def __init__(
        self, 
        objective_keys: List[str],
        objectives: Dict[str, float],
        weights: Dict[str, float] = None,
        maximize: bool = False
    ): 
        self.maximize = maximize
        # 用于补全未计算的目标值，确保比较时不会出错
        if self.maximize:
            self.null_value = 0.0
        else:
            self.null_value = 1e8

        self.objective_keys = objective_keys
        self.unuse_objectives = {k: v for k, v in objectives.items() if k not in objective_keys}
        self.objectives = {key: objectives.get(key, self.null_value) for key in objective_keys}
        self.weights = weights if weights else {k: 1.0 for k in objective_keys}

        if len(self.unuse_objectives) >= 0:
            print(f"Warning: Unused objectives in MultiObjectiveMetric: {self.unuse_objectives}")

        # Compute weighted score
        self.value = self._compute_weighted_score()

    def _compute_weighted_score(self):
        score = sum(self.weights[k] * self.objectives[k] for k in self.objective_keys)
        
        if self.maximize:
            return -score  # Negate for maximization
        else:
            return score
        
    def dominates(self, other: 'MultiObjectiveMetric') -> bool:
        better_in_one = False
        not_worse_in_all = True

        for k in self.objective_keys:
            self_val = self.objectives[k]
            other_val = other.objectives[k]

            if self.maximize:
                if self_val > other_val:
                    better_in_one = True
                if self_val < other_val:
                    not_worse_in_all = False
            else:
                if self_val < other_val:
                    better_in_one = True
                if self_val > other_val:
                    not_worse_in_all = False

        return better_in_one and not_worse_in_all
    
    def __lt__(self, other):
        """For sorting - use weighted score."""
        if isinstance(other, MultiObjectiveMetric):
            return self.value < other.value
        return self.value < other
    
    def __le__(self, other):
        if isinstance(other, MultiObjectiveMetric):
            return self.value <= other.value
        return self.value <= other
    
    def __gt__(self, other):
        if isinstance(other, MultiObjectiveMetric):
            return self.value > other.value
        return self.value > other
    
    def __ge__(self, other):
        if isinstance(other, MultiObjectiveMetric):
            return self.value >= other.value
        return self.value >= other
    
    def __eq__(self, other):
        if isinstance(other, MultiObjectiveMetric):
            return all(self.objectives[key] == other.objectives[key] for key in self.objectives.keys())
        return self.value == other

    def __repr__(self):
        return f"MultiObjectiveMetric({self.objectives})"

