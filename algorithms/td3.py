"""TD3 (Twin Delayed Deep Deterministic Policy Gradient) 实现."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .base_agent import BaseAgent


def _init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)


class TD3Actor(nn.Module):
    """TD3策略网络."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(_init_weights)

    def forward(self, state):
        return torch.tanh(self.net(state))


class TD3Critic(nn.Module):
    """TD3双Q网络."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        input_dim = state_dim + action_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(_init_weights)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_value(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class TD3Agent(BaseAgent):
    """TD3智能体."""

    def __init__(self, state_dim, action_dim, config=None, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 超参数
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)
        self.lr_actor = self.config.get('lr_actor', 3e-4)
        self.lr_critic = self.config.get('lr_critic', 3e-4)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.policy_noise = self.config.get('policy_noise', 0.2)
        self.noise_clip = self.config.get('noise_clip', 0.5)
        self.policy_freq = self.config.get('policy_freq', 2)
        self.exploration_noise = self.config.get('exploration_noise', 0.1)

        # 网络
        self.actor = TD3Actor(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.actor_target = TD3Actor(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic = TD3Critic(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic_target = TD3Critic(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr_critic)

        self.update_count = 0
        self.max_action = 1.0
        self._last_actor_loss = 0.0  # 记录最近一次非零的actor_loss

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        if not evaluate:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        return action

    def update(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = self.actor_target(next_states) + noise
            next_actions = next_actions.clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        actor_updated = False
        if self.update_count % self.policy_freq == 0:
            actor_loss = -self.critic.q1_value(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            self._last_actor_loss = actor_loss.item()  # 保存真正的actor_loss
            actor_updated = True

        self.update_count += 1

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': self._last_actor_loss,  # 返回最近一次真正的actor_loss
            'actor_updated': actor_updated,       # 标记本次是否更新了actor
            'q_mean': current_q1.mean().item()
        }

    def _soft_update(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'config': self.config
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        if 'actor_target' in checkpoint:
            self.actor_target.load_state_dict(checkpoint['actor_target'])
        if 'critic_target' in checkpoint:
            self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.config = checkpoint.get('config', self.config)

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()
