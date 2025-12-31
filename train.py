"""
训练脚本
使用SAC算法训练灵巧手手内旋转任务

用法:
    python train.py
    python train.py --config configs/config.yaml
    
注意: 此脚本为无头模式（不渲染），适合服务器训练
"""

import os
import sys
import argparse
import time
from collections import defaultdict

import numpy as np
import torch
import yaml
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs import LeapHandEnv
from algorithms import (
    SACAgent,
    TD3Agent,
    PPOAgent,
    ReplayBuffer,
    BaselineController
)
from utils import Logger


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device(device_config):
    """获取计算设备"""
    if device_config == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_config)


def evaluate(env, agent, num_episodes=5):
    """评估Agent性能"""
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            state = next_state
            
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths)
    }


SUPPORTED_ALGOS = {'sac', 'td3', 'ppo', 'baseline'}


def resolve_checkpoint_dir(base_dir, algo_name):
    """根据算法名称解析checkpoint目录."""
    normalized = base_dir.replace('\\', '/').rstrip('/')
    if '{algo}' in normalized:
        return normalized.format(algo=algo_name)
    if normalized.endswith(algo_name):
        return normalized
    return os.path.join(base_dir, algo_name)


def build_agent(algo_name, state_dim, action_dim, algo_config, device):
    """根据算法名称构建Agent."""
    if algo_name == 'sac':
        return SACAgent(state_dim=state_dim, action_dim=action_dim, config=algo_config, device=device)
    if algo_name == 'td3':
        return TD3Agent(state_dim=state_dim, action_dim=action_dim, config=algo_config, device=device)
    if algo_name == 'ppo':
        return PPOAgent(state_dim=state_dim, action_dim=action_dim, config=algo_config, device=device)
    if algo_name == 'baseline':
        return BaselineController(state_dim=state_dim, action_dim=action_dim, config=algo_config, device=device)
    raise ValueError(f"不支持的算法: {algo_name}")


def train_off_policy_loop(env, eval_env, agent, replay_buffer, logger, train_config,
                          checkpoint_dir, algo_name, timestamp_str):
    total_timesteps = train_config['total_timesteps']
    start_steps = train_config['start_steps']
    batch_size = train_config['batch_size']
    update_every = train_config['update_every']
    updates_per_step = train_config.get('updates_per_step', 1)
    eval_freq = train_config['eval_freq']
    save_freq = train_config['save_freq']
    log_freq = train_config['log_freq']

    best_eval_reward = -float('inf')
    episode_count = 0
    episode_reward = 0.0
    episode_length = 0
    episode_reward_components_raw = defaultdict(float)
    episode_reward_components_weighted = defaultdict(float)

    state, _ = env.reset()
    start_time = time.time()

    for timestep in range(1, total_timesteps + 1):
        action = env.action_space.sample() if timestep < start_steps else agent.select_action(state, evaluate=False)
        next_state, reward, done, truncated, info = env.step(action)
        replay_buffer.push(state, action, reward, next_state, float(done or truncated))

        state = next_state
        episode_reward += reward
        episode_length += 1

        reward_info = info.get('reward_info', {}) if info is not None else {}
        if reward_info:
            for key, value in reward_info.items():
                if key == 'total':
                    episode_reward_components_weighted['total'] += value
                    continue
                episode_reward_components_raw[key] += value
                weight = env.reward_weights.get(key, 0.0)
                episode_reward_components_weighted[key] += weight * value

        if done or truncated:
            state, _ = env.reset()
            episode_count += 1
            logger.log_scalar('Episode/Reward', episode_reward, episode_count)
            logger.log_scalar('Episode/Length', episode_length, episode_count)
            # 额外按训练步数记录，用于不同算法统一以Training Iterations为横轴对比
            logger.log_scalar('Train/EpisodeReward', episode_reward, timestep)
            logger.log_scalar('Train/EpisodeLength', episode_length, timestep)
            if episode_reward_components_weighted:
                logger.log_reward_components(episode_count, dict(episode_reward_components_weighted), prefix='Reward')
            if episode_reward_components_raw:
                logger.log_reward_components(episode_count, dict(episode_reward_components_raw), prefix='RewardRaw')
            if episode_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Episode {episode_count} | Step {timestep}/{total_timesteps} | Reward: {episode_reward:.2f} | Length: {episode_length} | Time: {elapsed:.1f}s")
            episode_reward = 0.0
            episode_length = 0
            episode_reward_components_raw = defaultdict(float)
            episode_reward_components_weighted = defaultdict(float)

        if timestep >= start_steps and timestep % update_every == 0 and replay_buffer.is_ready(batch_size):
            steps_to_update = int(updates_per_step * update_every)
            update_info = None
            for _ in range(steps_to_update):
                update_info = agent.update(replay_buffer, batch_size)
            if update_info and timestep % log_freq == 0:
                if 'actor_loss' in update_info:
                    logger.log_scalar('Loss/Actor', update_info['actor_loss'], timestep)
                if 'critic_loss' in update_info:
                    logger.log_scalar('Loss/Critic', update_info['critic_loss'], timestep)
                if 'alpha' in update_info:
                    logger.log_scalar('Train/Alpha', update_info['alpha'], timestep)
                if 'q_mean' in update_info:
                    logger.log_scalar('Train/Q_mean', update_info['q_mean'], timestep)

        if timestep % eval_freq == 0:
            eval_result = evaluate(eval_env, agent, num_episodes=train_config.get('eval_episodes', 5))
            logger.log_scalar('Eval/Mean_Reward', eval_result['mean_reward'], timestep)
            logger.log_scalar('Eval/Mean_Length', eval_result['mean_length'], timestep)
            print(f"\n[评估] Step {timestep} | 平均奖励: {eval_result['mean_reward']:.2f} ± {eval_result['std_reward']:.2f} | 平均长度: {eval_result['mean_length']:.1f}\n")
            if eval_result['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_result['mean_reward']
                agent.save(os.path.join(checkpoint_dir, f'best_{algo_name}_{timestamp_str}.pth'))
                print(f"[保存] 新的最佳模型! 奖励: {best_eval_reward:.2f}")

        if timestep % save_freq == 0:
            agent.save(os.path.join(checkpoint_dir, f'{algo_name}_{timestamp_str}_step_{timestep}.pth'))

    total_time = time.time() - start_time
    return best_eval_reward, episode_count, total_time


def train_ppo_loop(env, eval_env, agent, logger, train_config, checkpoint_dir, timestamp_str):
    total_timesteps = train_config['total_timesteps']
    eval_freq = train_config['eval_freq']
    save_freq = train_config['save_freq']
    log_freq = train_config['log_freq']
    batch_size = agent.config.get('mini_batch_size', 512)

    best_eval_reward = -float('inf')
    episode_count = 0
    episode_reward = 0.0
    episode_length = 0
    state, _ = env.reset()
    start_time = time.time()

    for timestep in range(1, total_timesteps + 1):
        action, log_prob, value, _ = agent.act(state, evaluate=False)
        next_state, reward, done, truncated, info = env.step(action)
        agent.rollout_buffer.add(state, action, reward, float(done or truncated), log_prob, value)

        state = next_state
        episode_reward += reward
        episode_length += 1

        if done or truncated:
            logger.log_scalar('Episode/Reward', episode_reward, episode_count + 1)
            logger.log_scalar('Episode/Length', episode_length, episode_count + 1)
            # 额外按训练步数记录，方便跨算法以步数为横轴对比
            logger.log_scalar('Train/EpisodeReward', episode_reward, timestep)
            logger.log_scalar('Train/EpisodeLength', episode_length, timestep)
            episode_count += 1
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0

        if agent.rollout_buffer.is_full():
            last_value = 0.0 if (done or truncated) else agent.get_value(state)
            agent.rollout_buffer.compute_returns_and_advantages(last_value, agent.gamma, agent.gae_lambda)
            update_info = agent.update(agent.rollout_buffer, batch_size)
            agent.rollout_buffer.reset()
            # 每次PPO update都记录loss，不再受log_freq限制
            logger.log_scalar('Loss/Actor', update_info['actor_loss'], timestep)
            logger.log_scalar('Loss/Critic', update_info['value_loss'], timestep)
            logger.log_scalar('Loss/Entropy', update_info['entropy'], timestep)

        if timestep % eval_freq == 0:
            eval_result = evaluate(eval_env, agent, num_episodes=train_config.get('eval_episodes', 5))
            logger.log_scalar('Eval/Mean_Reward', eval_result['mean_reward'], timestep)
            logger.log_scalar('Eval/Mean_Length', eval_result['mean_length'], timestep)
            print(f"\n[评估] Step {timestep} | 平均奖励: {eval_result['mean_reward']:.2f} ± {eval_result['std_reward']:.2f} | 平均长度: {eval_result['mean_length']:.1f}\n")
            if eval_result['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_result['mean_reward']
                agent.save(os.path.join(checkpoint_dir, f'best_ppo_{timestamp_str}.pth'))
                print(f"[保存] 新的最佳模型! 奖励: {best_eval_reward:.2f}")

        if timestep % save_freq == 0:
            agent.save(os.path.join(checkpoint_dir, f'ppo_{timestamp_str}_step_{timestep}.pth'))

    total_time = time.time() - start_time
    return best_eval_reward, episode_count, total_time


def run_baseline_loop(env, agent, logger, train_config):
    total_timesteps = train_config['total_timesteps']
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    state, _ = env.reset()
    start_time = time.time()

    for timestep in range(1, total_timesteps + 1):
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, truncated, _ = env.step(action)
        state = next_state
        episode_reward += reward
        episode_length += 1

        if done or truncated:
            episode_count += 1
            logger.log_scalar('Episode/Reward', episode_reward, episode_count)
            logger.log_scalar('Episode/Length', episode_length, episode_count)
            # Baseline 同样按训练步数记录，便于和RL算法统一对比
            logger.log_scalar('Train/EpisodeReward', episode_reward, timestep)
            logger.log_scalar('Train/EpisodeLength', episode_length, timestep)
            print(f"Baseline Episode {episode_count} | Reward: {episode_reward:.2f} | Length: {episode_length}")
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0

    total_time = time.time() - start_time
    return episode_count, total_time


def train(config_path='configs/config.yaml', algo=None):
    """主训练函数，支持SAC/TD3/PPO/Baseline."""

    config = load_config(config_path)
    env_config = config['env']
    train_config = config['training']

    algo_from_cfg = train_config.get('algo', 'sac')
    algo_name = (algo or algo_from_cfg).lower()
    if algo_name not in SUPPORTED_ALGOS:
        raise ValueError(f"不支持的算法: {algo_name}")

    algo_config = config.get(algo_name)
    if algo_config is None:
        raise KeyError(f"配置文件中缺少 {algo_name} 配置段")

    set_seed(train_config['seed'])
    device = get_device(train_config['device'])
    print(f"使用设备: {device}")

    print("创建环境...")
    env = LeapHandEnv(config=env_config, render_mode=None)
    eval_env = None if algo_name == 'baseline' else LeapHandEnv(config=env_config, render_mode=None)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")

    agent = build_agent(algo_name, state_dim, action_dim, algo_config, device)

    replay_buffer = None
    if algo_name in {'sac', 'td3'}:
        replay_buffer = ReplayBuffer(
            capacity=train_config['buffer_size'],
            state_dim=state_dim,
            action_dim=action_dim
        )

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{algo_name}_{timestamp_str}"
    logger = Logger(log_dir=train_config['log_dir'], experiment_name=experiment_name)

    checkpoint_dir = resolve_checkpoint_dir(train_config['checkpoint_dir'], algo_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print(f"开始训练 - 算法: {algo_name.upper()}")
    print(f"总步数: {train_config['total_timesteps']}")
    if algo_name in {'sac', 'td3'}:
        print(f"开始学习步数: {train_config['start_steps']}")
        print(f"批次大小: {train_config['batch_size']}")
    print("=" * 60)

    if algo_name in {'sac', 'td3'}:
        best_eval_reward, episode_count, total_time = train_off_policy_loop(
            env, eval_env, agent, replay_buffer, logger, train_config,
            checkpoint_dir, algo_name, timestamp_str
        )
    elif algo_name == 'ppo':
        best_eval_reward, episode_count, total_time = train_ppo_loop(
            env, eval_env, agent, logger, train_config, checkpoint_dir, timestamp_str
        )
    else:
        episode_count, total_time = run_baseline_loop(env, agent, logger, train_config)
        best_eval_reward = float('nan')

    print("=" * 60)
    print("训练完成!")
    print(f"总时间: {total_time/3600:.2f} 小时")
    print(f"总Episode: {episode_count}")
    if algo_name != 'baseline':
        print(f"最佳评估奖励: {best_eval_reward:.2f}")
    print("=" * 60)

    if algo_name != 'baseline':
        agent.save(os.path.join(checkpoint_dir, f'final_{algo_name}_{timestamp_str}.pth'))

    logger.close()
    env.close()
    if eval_env is not None:
        eval_env.close()


if __name__ == '__main__':
    # 创建一个 ArgumentParser 对象 也就是parser
    # ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
    parser = argparse.ArgumentParser(description='LeapHand SAC训练')
    # 可以通过--路径指定参数
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--algo', type=str, default=None, choices=sorted(SUPPORTED_ALGOS),
                        help='选择训练算法 (优先级高于配置文件)')
    # 调用 parse_args() 方法来实际解析命令行参数
    # 返回一个对象 args，可以用 args.config 访问配置文件路
    args = parser.parse_args()
    
    # 进入tarin主函数 带着命令行加载的参数
    train(args.config, args.algo)
