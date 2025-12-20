"""
SAC (Soft Actor-Critic) 算法实现
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .base_agent import BaseAgent
from .sac_models import Actor, Critic


class SACAgent(BaseAgent):
    """
    SAC (Soft Actor-Critic) 算法
    
    核心特点:
    1. 最大熵强化学习框架
    2. 自动调整熵系数alpha
    3. Double Q-Network缓解过估计
    4. 软更新目标网络
    """
    
    def __init__(self, state_dim, action_dim, config=None, device=None):
        """
        初始化SAC Agent
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            config: 配置字典
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # 设备
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 超参数
        self.gamma = self.config.get('gamma', 0.99)  # 折扣因子
        self.tau = self.config.get('tau', 0.005)      # 软更新系数
        self.lr_actor = self.config.get('lr_actor', 3e-4)
        self.lr_critic = self.config.get('lr_critic', 3e-4)
        self.lr_alpha = self.config.get('lr_alpha', 3e-4)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        
        # 自动调整熵系数
        self.auto_entropy_tuning = self.config.get('auto_entropy_tuning', True)
        
        # 初始化网络
        self.actor = Actor(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, self.hidden_dim).to(self.device)
        
        # 复制参数到目标网络
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # 熵系数
        if self.auto_entropy_tuning:
            # 目标熵 = -dim(A) (启发式)
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_alpha)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = self.config.get('alpha', 0.2)
            
        # 训练计数器
        self.update_count = 0
        
    def select_action(self, state, evaluate=False):
        """
        选择动作
        
        Args:
            state: 状态（numpy数组）
            evaluate: 是否为评估模式
            
        Returns:
            action: 动作（numpy数组）
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                # 确定性策略
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                # 随机策略
                action, _, _ = self.actor.sample(state)
                
        return action.cpu().numpy().flatten()
        
    def update(self, replay_buffer, batch_size):
        """
        更新网络参数
        
        Args:
            replay_buffer: 经验回放池
            batch_size: 批次大小
            
        Returns:
            info: 更新信息
        """
        # 采样
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # ==================== Critic Update ====================
        with torch.no_grad():
            # 从当前策略采样下一动作
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            
            # 计算目标Q值
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
            
        # 计算当前Q值
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # ==================== Actor Update ====================
        # 重新采样动作
        new_actions, log_probs, _ = self.actor.sample(states)
        
        # 计算Q值
        q1, q2 = self.critic(states, new_actions)
        min_q = torch.min(q1, q2)
        
        # Actor损失: 最大化 Q - alpha * log_prob
        actor_loss = (self.alpha * log_probs - min_q).mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # ==================== Alpha Update ====================
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.0)
            
        # ==================== Soft Update Target Network ====================
        self._soft_update(self.critic, self.critic_target)
        
        self.update_count += 1
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item() if self.auto_entropy_tuning else 0.0,
            'alpha': self.alpha,
            'q_mean': min_q.mean().item()
        }
        
    def _soft_update(self, source, target):
        """软更新目标网络"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
            
    def save(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'update_count': self.update_count,
            'config': self.config
        }
        
        if self.auto_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha.item()
            checkpoint['alpha_optimizer'] = self.alpha_optimizer.state_dict()
            
        torch.save(checkpoint, path)
        print(f"模型已保存至: {path}")
        
    def load(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.update_count = checkpoint.get('update_count', 0)
        
        if self.auto_entropy_tuning and 'log_alpha' in checkpoint:
            self.log_alpha.data.fill_(checkpoint['log_alpha'])
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp().item()
            
        print(f"模型已从 {path} 加载")
        
    def train_mode(self):
        """切换到训练模式"""
        self.actor.train()
        self.critic.train()
        
    def eval_mode(self):
        """切换到评估模式"""
        self.actor.eval()
        self.critic.eval()
