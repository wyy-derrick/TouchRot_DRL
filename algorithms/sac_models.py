"""
SAC (Soft Actor-Critic) 网络架构
包含Actor网络和Critic网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# 初始化权重
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    """
    SAC策略网络 (Actor)
    
    输出动作的均值和对数标准差，使用重参数化技巧采样动作
    """
    
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        初始化Actor网络
        
        Args:
            state_dim: 状态维度 (e.g., 52)
            action_dim: 动作维度 (e.g., 16)
            hidden_dim: 隐藏层维度
        """
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        self.apply(weights_init_)
        
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            
        Returns:
            mean: 动作均值 [batch_size, action_dim]
            log_std: 动作对数标准差 [batch_size, action_dim]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return mean, log_std
        
    def sample(self, state):
        """
        采样动作（使用重参数化技巧）
        
        Args:
            state: 状态张量
            
        Returns:
            action: 采样的动作（经过tanh压缩到[-1, 1]）
            log_prob: 动作的对数概率
            mean: 动作均值（用于确定性策略）
        """
        mean, log_std = self.forward(state)
        # 限制标准差范围并避免极端值造成数值溢出
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp().clamp(min=1e-6)

        # 若出现NaN/inf输入，使用安全默认值防止CUDA launch failure
        if (not torch.isfinite(mean).all()) or (not torch.isfinite(std).all()):
            mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
            std = torch.nan_to_num(std, nan=0.1, posinf=1.0, neginf=1.0)
        
        # 创建正态分布
        normal = Normal(mean, std)
        
        # 重参数化采样: z = mean + std * epsilon
        x_t = normal.rsample()
        
        # 应用tanh将动作压缩到[-1, 1]
        action = torch.tanh(x_t)
        
        # 计算对数概率（考虑tanh变换的雅可比行列式）
        # log_prob = log_prob_normal - log(1 - tanh^2(x) + epsilon)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # 数值安全: 避免NaN传递到后续计算
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 确定性动作（评估时使用）
        mean_action = torch.tanh(mean)
        mean_action = torch.nan_to_num(mean_action, nan=0.0, posinf=0.0, neginf=0.0)
        
        return action, log_prob, mean_action
        
    def get_action(self, state, evaluate=False):
        """
        获取动作（用于与环境交互）
        
        Args:
            state: 状态（numpy数组或张量）
            evaluate: 是否使用确定性策略
            
        Returns:
            action: numpy数组
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0)
            
        with torch.no_grad():
            if evaluate:
                mean, _ = self.forward(state)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.sample(state)
                
        return action.cpu().numpy().flatten()


class Critic(nn.Module):
    """
    SAC价值网络 (Critic)
    
    使用Double Q-Network结构缓解过高估计
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        初始化Critic网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(Critic, self).__init__()
        
        # Q1网络
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)
        
        # Q2网络
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
        
    def forward(self, state, action):
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            action: 动作张量 [batch_size, action_dim]
            
        Returns:
            q1, q2: 两个Q值 [batch_size, 1]
        """
        x = torch.cat([state, action], dim=-1)
        
        # Q1
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = F.relu(self.q1_fc3(q1))
        q1 = self.q1_out(q1)
        
        # Q2
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = F.relu(self.q2_fc3(q2))
        q2 = self.q2_out(q2)
        
        return q1, q2
        
    def q1_forward(self, state, action):
        """只计算Q1（用于策略更新）"""
        x = torch.cat([state, action], dim=-1)
        
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = F.relu(self.q1_fc3(q1))
        q1 = self.q1_out(q1)
        
        return q1


# 添加numpy导入（Actor类的get_action方法需要）
import numpy as np
