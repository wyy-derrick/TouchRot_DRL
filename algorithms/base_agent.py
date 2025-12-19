"""
算法基类
定义所有RL算法的统一接口
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    强化学习算法基类
    
    所有算法（SAC, PPO, TD3等）都需要继承此类并实现以下方法
    """
    
    @abstractmethod
    def select_action(self, state, evaluate=False):
        """
        选择动作
        
        Args:
            state: 当前状态
            evaluate: 是否为评估模式（如果True，使用确定性策略）
            
        Returns:
            action: 选择的动作
        """
        raise NotImplementedError
        
    @abstractmethod
    def update(self, replay_buffer, batch_size):
        """
        更新网络参数
        
        Args:
            replay_buffer: 经验回放池
            batch_size: 批次大小
            
        Returns:
            info: 更新信息字典（包含loss等）
        """
        raise NotImplementedError
        
    @abstractmethod
    def save(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        raise NotImplementedError
        
    @abstractmethod
    def load(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        raise NotImplementedError
        
    def train_mode(self):
        """切换到训练模式"""
        pass
        
    def eval_mode(self):
        """切换到评估模式"""
        pass
