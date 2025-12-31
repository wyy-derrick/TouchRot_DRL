"""规则基准控制器,用于与RL算法对比."""

import json
import numpy as np

from .base_agent import BaseAgent


class BaselineController(BaseAgent):
    """基于触觉启发的简单控制器."""

    def __init__(self, state_dim, action_dim, config=None, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        self.num_tactile = self.config.get('num_tactile', 17)
        self.contact_threshold = self.config.get('contact_threshold', 0.2)
        self.twist_gain = self.config.get('twist_gain', 0.15)
        self.closure_gain = self.config.get('closure_gain', 0.3)
        self.release_gain = self.config.get('release_gain', 0.1)
        self.damping_gain = self.config.get('damping_gain', 0.05)
        self.bias_action = np.array(
            self.config.get('bias_action', [0.0] * action_dim),
            dtype=np.float32
        )

    def select_action(self, state, evaluate=False):
        state = np.asarray(state, dtype=np.float32)
        qvel = state[self.action_dim: self.action_dim * 2]
        tactile_start = self.action_dim * 2
        tactile_end = tactile_start + self.num_tactile
        tactile = state[tactile_start:tactile_end]
        target_axis = state[tactile_end:tactile_end + 3]

        contact_level = float(np.mean(tactile))
        twist_direction = np.sign(target_axis[2]) or 1.0

        action = self.bias_action.copy()
        if contact_level > self.contact_threshold:
            action += twist_direction * self.twist_gain
            action[: self.action_dim // 2] -= self.closure_gain * (contact_level - self.contact_threshold)
        else:
            action[: self.action_dim // 2] += self.release_gain

        action -= self.damping_gain * qvel
        return np.clip(action, -1.0, 1.0)

    def update(self, replay_buffer, batch_size):
        return {'actor_loss': 0.0, 'critic_loss': 0.0}

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'config': self.config}, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        self.config = payload.get('config', self.config)

    def train_mode(self):
        return None

    def eval_mode(self):
        return None
