# multi_objective_metric.py - Multi-objective optimization support
import numpy as np
from typing import List, Optional, Dict
from aide.utils.metric import MetricValue


class MultiObjectiveMetric:
    """
    Multi-objective metric for carbon-aware scheduling.
    Tracks three objectives: time, carbon, and job completion time.
    """
    
    def __init__(self, time: float, carbon: float, jct: float, maximize: bool = False):
        """
        Args:
            time: Total execution time (lower is better)
            carbon: Total carbon usage (lower is better)
            jct: Average job completion time (lower is better)
            maximize: If True, higher values are better (for compatibility)
        """
        self.time = time
        self.carbon = carbon
        self.jct = jct
        self.maximize = maximize
        
        # For compatibility with AIDE's MetricValue
        # Use weighted sum as default scalar value
        self.value = self._compute_weighted_score()
    
    def _compute_weighted_score(self, w_time: float = 0.3, 
                                w_carbon: float = 0.5, 
                                w_jct: float = 0.2) -> float:
        """
        Compute weighted score for comparison.
        Lower is better (we want to minimize all three objectives).
        """
        if self.maximize:
            # If maximizing, negate the score
            return -(w_time * self.time + w_carbon * self.carbon + w_jct * self.jct)
        else:
            return w_time * self.time + w_carbon * self.carbon + w_jct * self.jct
    
    def dominates(self, other: 'MultiObjectiveMetric') -> bool:
        """
        Check if this solution Pareto-dominates another solution.
        
        Returns:
            True if this solution is better in at least one objective
            and not worse in any objective.
        """
        if self.maximize:
            better_in_one = (
                self.time > other.time or 
                self.carbon > other.carbon or 
                self.jct > other.jct
            )
            not_worse_in_all = (
                self.time >= other.time and 
                self.carbon >= other.carbon and 
                self.jct >= other.jct
            )
        else:
            better_in_one = (
                self.time < other.time or 
                self.carbon < other.carbon or 
                self.jct < other.jct
            )
            not_worse_in_all = (
                self.time <= other.time and 
                self.carbon <= other.carbon and 
                self.jct <= other.jct
            )
        
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
            return (self.time == other.time and 
                   self.carbon == other.carbon and 
                   self.jct == other.jct)
        return self.value == other
    
    def __repr__(self):
        return f"MultiObjectiveMetric(time={self.time:.2f}, carbon={self.carbon:.2f}, jct={self.jct:.2f})"


def compute_pareto_front(solutions: List) -> List:
    """
    Compute the Pareto front from a list of solutions.
    
    Args:
        solutions: List of objects with 'metric' attribute (MultiObjectiveMetric)
    
    Returns:
        List of non-dominated solutions
    """
    pareto_front = []
    
    for solution in solutions:
        if not hasattr(solution, 'metric') or not isinstance(solution.metric, MultiObjectiveMetric):
            continue
        
        is_dominated = False
        for other in solutions:
            if not hasattr(other, 'metric') or not isinstance(other.metric, MultiObjectiveMetric):
                continue
            if other is solution:
                continue
            
            if other.metric.dominates(solution.metric):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front.append(solution)
    
    return pareto_front


def select_from_pareto_front(pareto_front: List, 
                             preference: Optional[Dict[str, float]] = None,
                             strategy: str = 'min_carbon') -> Optional[object]:
    """
    Select a solution from the Pareto front based on preference or strategy.
    
    Args:
        pareto_front: List of Pareto-optimal solutions
        preference: Dictionary with weights {'time': w1, 'carbon': w2, 'jct': w3}
        strategy: Selection strategy if no preference given
                 - 'min_carbon': minimize carbon usage
                 - 'min_time': minimize execution time
                 - 'min_jct': minimize job completion time
                 - 'balanced': use equal weights
    
    Returns:
        Selected solution or None if pareto_front is empty
    """
    if not pareto_front:
        return None
    
    if len(pareto_front) == 1:
        return pareto_front[0]
    
    if preference:
        # Use weighted sum with user preferences
        w_time = preference.get('time', 0.33)
        w_carbon = preference.get('carbon', 0.33)
        w_jct = preference.get('jct', 0.34)
        
        scores = [
            w_time * sol.metric.time + 
            w_carbon * sol.metric.carbon + 
            w_jct * sol.metric.jct
            for sol in pareto_front
        ]
        return pareto_front[np.argmin(scores)]
    
    else:
        # Use predefined strategy
        if strategy == 'min_carbon':
            return min(pareto_front, key=lambda s: s.metric.carbon)
        elif strategy == 'min_time':
            return min(pareto_front, key=lambda s: s.metric.time)
        elif strategy == 'min_jct':
            return min(pareto_front, key=lambda s: s.metric.jct)
        elif strategy == 'balanced':
            scores = [
                0.33 * sol.metric.time + 
                0.33 * sol.metric.carbon + 
                0.34 * sol.metric.jct
                for sol in pareto_front
            ]
            return pareto_front[np.argmin(scores)]
        else:
            # Default: min carbon
            return min(pareto_front, key=lambda s: s.metric.carbon)


def visualize_pareto_front(pareto_front: List, 
                           baseline_metric: Optional[MultiObjectiveMetric] = None,
                           save_path: Optional[str] = None):
    """
    Visualize the Pareto front in 3D space.
    
    Args:
        pareto_front: List of Pareto-optimal solutions
        baseline_metric: Baseline metric for comparison
        save_path: Path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        if not pareto_front:
            return
        
        fig = plt.figure(figsize=(12, 5))
        
        # 3D scatter plot
        ax1 = fig.add_subplot(121, projection='3d')
        
        times = [sol.metric.time for sol in pareto_front]
        carbons = [sol.metric.carbon for sol in pareto_front]
        jcts = [sol.metric.jct for sol in pareto_front]
        
        ax1.scatter(times, carbons, jcts, c='blue', marker='o', s=50, label='Pareto Front')
        
        if baseline_metric:
            ax1.scatter([baseline_metric.time], [baseline_metric.carbon], [baseline_metric.jct],
                       c='red', marker='*', s=200, label='Baseline')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Carbon')
        ax1.set_zlabel('JCT')
        ax1.set_title('Pareto Front (3D)')
        ax1.legend()
        
        # 2D projections
        ax2 = fig.add_subplot(122)
        ax2.scatter(carbons, times, c='blue', marker='o', s=50, label='Pareto Front')
        
        if baseline_metric:
            ax2.scatter([baseline_metric.carbon], [baseline_metric.time],
                       c='red', marker='*', s=200, label='Baseline')
        
        ax2.set_xlabel('Carbon Usage')
        ax2.set_ylabel('Execution Time')
        ax2.set_title('Pareto Front (Carbon vs Time)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Pareto front visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"Error visualizing Pareto front: {e}")