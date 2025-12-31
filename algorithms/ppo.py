"""PPO (Proximal Policy Optimization) 实现."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from .base_agent import BaseAgent


def _init_linear(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)


class GaussianPolicy(nn.Module):
    """高斯策略网络，输出tanh后的动作."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.apply(_init_linear)

    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        return action, log_prob, entropy, mean

    def evaluate_actions(self, states, actions):
        clipped_actions = torch.clamp(actions, -0.999999, 0.999999)
        pre_tanh = 0.5 * (torch.log1p(clipped_actions) - torch.log1p(-clipped_actions))
        mean, log_std = self.forward(states)
        std = log_std.exp()
        normal = Normal(mean, std)
        log_prob = normal.log_prob(pre_tanh) - torch.log(1 - clipped_actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy, mean


class ValueFunction(nn.Module):
    """状态价值网络."""

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(_init_linear)

    def forward(self, state):
        return self.net(state)


class RolloutBuffer:
    """PPO用On-Policy数据缓冲."""

    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = None
        self.returns = None

    def add(self, state, action, reward, done, log_prob, value):
        if len(self.states) >= self.buffer_size:
            raise RuntimeError("RolloutBuffer已满，请先执行update")
        self.states.append(np.array(state, copy=True))
        self.actions.append(np.array(action, copy=True))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def is_full(self):
        return len(self.states) >= self.buffer_size

    def __len__(self):
        return len(self.states)

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        size = len(self.rewards)
        values = np.array(self.values + [last_value], dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        advantages = np.zeros(size, dtype=np.float32)
        gae = 0.0
        for step in reversed(range(size)):
            next_value = values[step + 1]
            delta = rewards[step] + gamma * next_value * (1.0 - dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1.0 - dones[step]) * gae
            advantages[step] = gae
        returns = advantages + values[:-1]
        self.advantages = advantages
        self.returns = returns

    def as_tensors(self, device):
        if self.advantages is None or self.returns is None:
            raise RuntimeError("在取数据前需要先计算优势和回报")
        data = {
            'states': torch.tensor(np.array(self.states), dtype=torch.float32, device=device),
            'actions': torch.tensor(np.array(self.actions), dtype=torch.float32, device=device),
            'log_probs': torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device).unsqueeze(-1),
            'returns': torch.tensor(self.returns, dtype=torch.float32, device=device).unsqueeze(-1),
            'advantages': torch.tensor(self.advantages, dtype=torch.float32, device=device).unsqueeze(-1)
        }
        return data


class PPOAgent(BaseAgent):
    """PPO智能体."""

    def __init__(self, state_dim, action_dim, config=None, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 超参数
        self.gamma = self.config.get('gamma', 0.99)
        self.gae_lambda = self.config.get('gae_lambda', 0.95)
        self.clip_range = self.config.get('clip_range', 0.2)
        self.policy_lr = self.config.get('lr_actor', 3e-4)
        self.value_lr = self.config.get('lr_critic', 3e-4)
        self.entropy_coef = self.config.get('entropy_coef', 0.0)
        self.value_coef = self.config.get('value_coef', 0.5)
        self.ppo_epochs = self.config.get('ppo_epochs', 10)
        self.hidden_dim = self.config.get('hidden_dim', 256)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
        self.rollout_length = self.config.get('rollout_length', 2048)

        self.policy = GaussianPolicy(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.value_function = ValueFunction(state_dim, self.hidden_dim).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.policy_lr)
        self.value_optimizer = Adam(self.value_function.parameters(), lr=self.value_lr)

        self.rollout_buffer = RolloutBuffer(self.rollout_length, state_dim, action_dim)

    def act(self, state, evaluate=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.policy.forward(state_tensor)
                action = torch.tanh(mean)
                log_prob = torch.zeros((1, 1), device=self.device)
                entropy = torch.zeros((1, 1), device=self.device)
            else:
                action, log_prob, entropy, _ = self.policy.sample(state_tensor)
            value = self.value_function(state_tensor)
        return (
            action.cpu().numpy().flatten(),
            log_prob.cpu().numpy().item(),
            value.cpu().numpy().item(),
            entropy.cpu().numpy().item()
        )

    def select_action(self, state, evaluate=False):
        action, _, _, _ = self.act(state, evaluate=evaluate)
        return action

    def get_value(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.value_function(state_tensor)
        return value.cpu().item()

    def update(self, rollout_buffer, batch_size):
        data = rollout_buffer.as_tensors(self.device)
        advantages = data['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_actor_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        update_steps = 0

        num_samples = len(rollout_buffer)
        indices = np.arange(num_samples)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                batch_idx = torch.LongTensor(batch_idx).to(self.device)

                states = data['states'][batch_idx]
                actions = data['actions'][batch_idx]
                old_log_probs = data['log_probs'][batch_idx]
                returns = data['returns'][batch_idx]
                adv = advantages[batch_idx]

                log_probs, entropy, _ = self.policy.evaluate_actions(states, actions)
                ratios = torch.exp(log_probs - old_log_probs)
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                values = self.value_function(states)
                value_loss = F.mse_loss(values, returns)

                entropy_loss = entropy.mean()

                self.policy_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_function.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                update_steps += 1

        mean_actor_loss = total_actor_loss / max(update_steps, 1)
        mean_value_loss = total_value_loss / max(update_steps, 1)
        mean_entropy = total_entropy / max(update_steps, 1)

        return {
            'actor_loss': mean_actor_loss,
            'value_loss': mean_value_loss,
            'entropy': mean_entropy
        }

    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'value_function': self.value_function.state_dict(),
            'config': self.config
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value_function.load_state_dict(checkpoint['value_function'])
        self.config = checkpoint.get('config', self.config)

    def train_mode(self):
        self.policy.train()
        self.value_function.train()

    def eval_mode(self):
        self.policy.eval()
        self.value_function.eval()
