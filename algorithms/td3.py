"""
TD3 (Twin Delayed DDPG) 算法
[TODO] 预留接口，后续实现
"""

from .base_agent import BaseAgent


class TD3Agent(BaseAgent):
    """
    TD3算法 - 待实现
    """
    
    def __init__(self, state_dim, action_dim, config=None, device=None):
        raise NotImplementedError("TD3算法尚未实现，请使用SAC")
        
    def select_action(self, state, evaluate=False):
        raise NotImplementedError
        
    def update(self, replay_buffer, batch_size):
        raise NotImplementedError
        
    def save(self, path):
        raise NotImplementedError
        
    def load(self, path):
        raise NotImplementedError
