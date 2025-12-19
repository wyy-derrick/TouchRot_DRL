"""
经验回放池
用于存储和采样训练经验
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    经验回放池
    
    存储 (state, action, reward, next_state, done) 元组
    支持随机采样用于off-policy训练
    """
    
    def __init__(self, capacity, state_dim, action_dim):
        """
        初始化回放池
        
        Args:
            capacity: 最大容量
            state_dim: 状态维度
            action_dim: 动作维度
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 使用numpy数组存储以提高效率
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.ptr = 0  # 当前写入位置
        self.size = 0  # 当前存储数量
        
    def push(self, state, action, reward, next_state, done):
        """
        添加一条经验
        
        Args:
            state: 状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否终止
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        """
        随机采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            batch: (states, actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )
        
    def __len__(self):
        return self.size
        
    def is_ready(self, batch_size):
        """检查是否有足够的样本进行采样"""
        return self.size >= batch_size


class PrioritizedReplayBuffer:
    """
    优先级经验回放池 (可选扩展)
    
    基于TD误差的优先级采样
    """
    
    def __init__(self, capacity, state_dim, action_dim, alpha=0.6, beta=0.4):
        """
        初始化优先级回放池
        
        Args:
            capacity: 最大容量
            state_dim: 状态维度
            action_dim: 动作维度
            alpha: 优先级指数
            beta: 重要性采样指数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.buffer = ReplayBuffer(capacity, state_dim, action_dim)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
    def push(self, state, action, reward, next_state, done):
        """添加经验，使用最大优先级"""
        self.priorities[self.buffer.ptr] = self.max_priority
        self.buffer.push(state, action, reward, next_state, done)
        
    def sample(self, batch_size):
        """基于优先级采样"""
        if self.buffer.size < batch_size:
            return None
            
        # 计算采样概率
        priorities = self.priorities[:self.buffer.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样
        indices = np.random.choice(self.buffer.size, batch_size, p=probs, replace=False)
        
        # 计算重要性采样权重
        weights = (self.buffer.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = weights.astype(np.float32)
        
        # 逐渐增加beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = (
            self.buffer.states[indices],
            self.buffer.actions[indices],
            self.buffer.rewards[indices],
            self.buffer.next_states[indices],
            self.buffer.dones[indices]
        )
        
        return batch, indices, weights
        
    def update_priorities(self, indices, td_errors):
        """更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
            self.max_priority = max(self.max_priority, self.priorities[idx])
            
    def __len__(self):
        return len(self.buffer)
        
    def is_ready(self, batch_size):
        return self.buffer.is_ready(batch_size)
