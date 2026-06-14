# adaptive_search.py - Adaptive search policy with UCB
import numpy as np
import logging
from typing import Optional, Dict, List
from aide.journal import Node, Journal

logger = logging.getLogger("aide")


class AdaptiveSearchPolicy:
    """
    Adaptive search policy using Upper Confidence Bound (UCB) algorithm.
    Balances exploration and exploitation in the search space.
    """
    
    def __init__(self, exploration_weight: float = 5):
        """
        Args:
            exploration_weight: Weight for exploration term in UCB (higher = more exploration)
        """
        self.exploration_weight = exploration_weight
        self.modification_history: Dict[str, Dict] = {}  # Track success rate of modification types
        self.failed_modifications: List[str] = []  # Track failed modification descriptions
        
    def select_node_ucb(self, journal: Journal) -> Optional[Node]:
        """
        Select a node to improve using UCB algorithm.
        
        Args:
            journal: Journal containing all nodes
            
        Returns:
            Selected node or None if no good nodes available
        """
        good_nodes = journal.good_nodes
        if not good_nodes:
            return None
        
        # Initialize visit counts if not present
        for node in good_nodes:
            if not hasattr(node, 'visit_count'):
                node.visit_count = 0
        
        # Calculate total visits
        total_visits = sum(getattr(n, 'visit_count', 0) for n in good_nodes)
        if total_visits == 0:
            total_visits = 1
        
        # Calculate UCB scores
        ucb_scores = []
        for node in good_nodes:
            visit_count = getattr(node, 'visit_count', 0)
            
            # Exploitation: node's performance (normalized)
            if hasattr(node.metric, 'value'):
                exploitation = node.metric.value  
            else:
                exploitation = 0
            
            # Exploration: uncertainty based on visit count
            if visit_count == 0:
                exploration = float('inf')  # Prioritize unvisited nodes
            else:
                exploration = np.sqrt(2 * np.log(total_visits) / visit_count)
            
            # UCB score
            ucb = exploitation + self.exploration_weight * exploration
            ucb_scores.append(ucb)
        
        # Select node with highest UCB score
        selected_idx = np.argmax(ucb_scores)
        selected_node = good_nodes[selected_idx]
        
        # Increment visit count
        if not hasattr(selected_node, 'visit_count'):
            selected_node.visit_count = 0
        selected_node.visit_count += 1
        
        logger.debug(f"[UCB] Selected node {selected_node.id} with UCB score {ucb_scores[selected_idx]:.4f}")
        
        return selected_node
    
    def record_modification_result(self, parent: Node, child: Node, 
                                   modification_type: str, success: bool):
        """
        Record the result of a modification for learning.
        
        Args:
            parent: Parent node
            child: Child node (result of modification)
            modification_type: Type of modification (e.g., 'L_coefficient', 'threshold_formula')
            success: Whether the modification improved performance
        """
        if modification_type not in self.modification_history:
            self.modification_history[modification_type] = {
                'success': 0,
                'total': 0,
                'avg_improvement': 0.0
            }
        
        self.modification_history[modification_type]['total'] += 1
        
        if success:
            self.modification_history[modification_type]['success'] += 1

            # Calculate improvement (score is maximize=True, so child > parent = improvement)
            parent_val = getattr(parent.metric, 'value', None) if hasattr(parent, 'metric') else None
            child_val = getattr(child.metric, 'value', None) if hasattr(child, 'metric') else None
            if parent_val is not None and child_val is not None:
                improvement = child_val - parent_val  # Positive = child is better
                current_avg = self.modification_history[modification_type]['avg_improvement']
                total = self.modification_history[modification_type]['total']
                # Update running average
                new_avg = (current_avg * (total - 1) + improvement) / total
                self.modification_history[modification_type]['avg_improvement'] = new_avg
        else:
            # Record failed modification description
            if hasattr(child, 'plan'):
                self.failed_modifications.append(child.plan[:200])  # First 200 chars
    
    def get_promising_modification_types(self, top_k: int = 3) -> List[str]:
        """
        Get modification types with highest success rate.
        
        Args:
            top_k: Number of top modification types to return
            
        Returns:
            List of modification type names
        """
        if not self.modification_history:
            return []
        
        # Sort by success rate, then by average improvement
        sorted_types = sorted(
            self.modification_history.items(),
            key=lambda x: (
                x[1]['success'] / max(x[1]['total'], 1),
                x[1]['avg_improvement']
            ),
            reverse=True
        )
        
        return [t[0] for t in sorted_types[:top_k]]
    
    def get_failed_attempts(self, max_recent: int = 5) -> List[str]:
        """
        Get recent failed modification attempts.
        
        Args:
            max_recent: Maximum number of recent failures to return
            
        Returns:
            List of failed modification descriptions
        """
        return self.failed_modifications[-max_recent:]
    
    def get_modification_stats(self) -> Dict:
        """
        Get statistics about modification history.
        
        Returns:
            Dictionary with statistics
        """
        if not self.modification_history:
            return {
                'total_modifications': 0,
                'total_successes': 0,
                'success_rate': 0.0,
                'by_type': {}
            }
        
        total_mods = sum(h['total'] for h in self.modification_history.values())
        total_success = sum(h['success'] for h in self.modification_history.values())
        
        return {
            'total_modifications': total_mods,
            'total_successes': total_success,
            'success_rate': total_success / max(total_mods, 1),
            'by_type': {
                mod_type: {
                    'success_rate': h['success'] / max(h['total'], 1),
                    'avg_improvement': h['avg_improvement'],
                    'count': h['total']
                }
                for mod_type, h in self.modification_history.items()
            }
        }
    
    def should_explore(self, journal: Journal, explore_prob: float = 0.3) -> bool:
        """
        Decide whether to explore (draft new solution) or exploit (improve existing).
        
        Args:
            journal: Journal containing all nodes
            explore_prob: Base probability of exploration
            
        Returns:
            True if should explore, False if should exploit
        """
        # Increase exploration if success rate is low
        stats = self.get_modification_stats()
        success_rate = stats['success_rate']
        
        # Adaptive exploration: explore more if not finding good solutions
        if success_rate < 0.3:
            adjusted_prob = min(explore_prob * 1.5, 0.5)
        elif success_rate > 0.7:
            adjusted_prob = explore_prob * 0.5
        else:
            adjusted_prob = explore_prob
        
        return np.random.random() < adjusted_prob


def classify_modification_type(parent_code: str, child_code: str) -> str:
    """
    Classify the type of modification made between parent and child code.
    
    Args:
        parent_code: Parent node's code
        child_code: Child node's code
        
    Returns:
        Modification type string
    """
    # Simple heuristic-based classification
    if 'min(future_carbon)' in parent_code and 'min(future_carbon)' in child_code:
        # Check if L coefficient changed
        parent_l = parent_code.count('* 1.1')
        child_l = child_code.count('* 1.1')
        if parent_l != child_l:
            return 'L_coefficient'
    
    if 'max(future_carbon)' in parent_code and 'max(future_carbon)' in child_code:
        # Check if U coefficient changed
        parent_u = parent_code.count('* 1.0')
        child_u = child_code.count('* 1.0')
        if parent_u != child_u:
            return 'U_coefficient'
    
    if 'future_keys[:48]' in parent_code:
        if 'future_keys[:48]' not in child_code:
            return 'prediction_window'
    
    if 'lambertw' in parent_code and 'lambertw' in child_code:
        # Check if threshold formula changed
        parent_thresh = parent_code.count('(U)*(1 - (1 - (1/alpha))')
        child_thresh = child_code.count('(U)*(1 - (1 - (1/alpha))')
        if parent_thresh != child_thresh:
            return 'threshold_formula'
    
    if 'if threshold < current_carbon:' in parent_code:
        if 'if threshold < current_carbon:' not in child_code:
            return 'threshold_comparison'
    
    # Default
    return 'other'
