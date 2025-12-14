import random
import structlog
from dataclasses import dataclass

logger = structlog.get_logger()

@dataclass
class SystemState:
    prompt_len: int
    cache_hit_rate: float
    gpu_utilization: float

class RLRouter:
    """
    Contextual Bandit implementation for Routing.
    Learns to map State -> Action (Worker ID).
    """
    def __init__(self):
        # Q-Table simulation: State (High/Low Load) -> Action Rewards
        self.q_table = {
            "high_load": {"local": -1.0, "remote": 1.0},
            "low_load":  {"local": 1.0,  "remote": 0.5}
        }
        self.epsilon = 0.1 # Exploration rate

    def route(self, state: SystemState) -> str:
        """
        Decides:
        1. 'local_worker': Compute on this node (Fast for small/cached)
        2. 'remote_worker': Offload to cluster (Fast for heavy prefill)
        """
        # Discretize state
        state_key = "high_load" if state.gpu_utilization > 0.8 or state.prompt_len > 1000 else "low_load"
        
        # Epsilon-Greedy Policy
        if random.random() < self.epsilon:
            action = random.choice(["local", "remote"])
            logger.info("router_exploration", action=action)
        else:
            # Exploitation: Pick best Q-value
            rewards = self.q_table[state_key]
            action = max(rewards, key=rewards.get)
        
        # Override for cache hits (The "Radix" advantage)
        if state.cache_hit_rate > 0.8:
            action = "local" # Data Locality
            
        return action

    def update(self, state, action, reward):
        # Mock Q-Learning update
        pass