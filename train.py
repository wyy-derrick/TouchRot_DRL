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
import numpy as np
import torch
import yaml
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs import LeapHandEnv
from algorithms import SACAgent, ReplayBuffer
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


def train(config_path='configs/config.yaml'):
    """主训练函数"""
    
    # 加载配置
    config = load_config(config_path)
    env_config = config['env']
    sac_config = config['sac']
    train_config = config['training']
    
    # 设置随机种子
    set_seed(train_config['seed'])
    
    # 获取设备
    device = get_device(train_config['device'])
    print(f"使用设备: {device}")
    
    # 创建环境
    print("创建环境...")
    env = LeapHandEnv(config=env_config, render_mode=None)
    eval_env = LeapHandEnv(config=env_config, render_mode=None)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 创建Agent
    print("创建SAC Agent...")
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=sac_config,
        device=device
    )
    
    # 创建经验回放池
    replay_buffer = ReplayBuffer(
        capacity=train_config['buffer_size'],
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    # 创建Logger
    experiment_name = f"sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = Logger(
        log_dir=train_config['log_dir'],
        experiment_name=experiment_name
    )
    
    # 创建checkpoint目录
    checkpoint_dir = train_config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练参数
    total_timesteps = train_config['total_timesteps']
    start_steps = train_config['start_steps']
    batch_size = train_config['batch_size']
    update_every = train_config['update_every']
    eval_freq = train_config['eval_freq']
    save_freq = train_config['save_freq']
    log_freq = train_config['log_freq']
    
    # 训练统计
    best_eval_reward = -float('inf')
    episode_count = 0
    episode_reward = 0
    episode_length = 0
    
    # 初始化环境
    state, _ = env.reset()
    
    print("=" * 60)
    print("开始训练")
    print(f"总步数: {total_timesteps}")
    print(f"开始学习步数: {start_steps}")
    print(f"批次大小: {batch_size}")
    print("=" * 60)
    
    start_time = time.time()
    
    for timestep in range(1, total_timesteps + 1):
        # 选择动作
        if timestep < start_steps:
            # 随机探索
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, evaluate=False)
            
        # 执行动作
        next_state, reward, done, truncated, info = env.step(action)
        
        # 存储经验
        replay_buffer.push(
            state, action, reward, next_state, 
            float(done or truncated)
        )
        
        state = next_state
        episode_reward += reward
        episode_length += 1
        
        # Episode结束
        if done or truncated:
            state, _ = env.reset()
            episode_count += 1
            
            # 记录Episode信息
            logger.log_scalar('Episode/Reward', episode_reward, episode_count)
            logger.log_scalar('Episode/Length', episode_length, episode_count)
            
            if episode_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Episode {episode_count} | "
                      f"Step {timestep}/{total_timesteps} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Length: {episode_length} | "
                      f"Time: {elapsed:.1f}s")
                
            episode_reward = 0
            episode_length = 0
            
        # 更新网络
        if timestep >= start_steps and timestep % update_every == 0:
            if replay_buffer.is_ready(batch_size):
                update_info = agent.update(replay_buffer, batch_size)
                
                if timestep % log_freq == 0:
                    logger.log_scalar('Loss/Actor', update_info['actor_loss'], timestep)
                    logger.log_scalar('Loss/Critic', update_info['critic_loss'], timestep)
                    logger.log_scalar('Train/Alpha', update_info['alpha'], timestep)
                    logger.log_scalar('Train/Q_mean', update_info['q_mean'], timestep)
                    
        # 评估
        if timestep % eval_freq == 0:
            eval_result = evaluate(eval_env, agent, num_episodes=train_config.get('eval_episodes', 5))
            
            logger.log_scalar('Eval/Mean_Reward', eval_result['mean_reward'], timestep)
            logger.log_scalar('Eval/Mean_Length', eval_result['mean_length'], timestep)
            
            print(f"\n[评估] Step {timestep} | "
                  f"平均奖励: {eval_result['mean_reward']:.2f} ± {eval_result['std_reward']:.2f} | "
                  f"平均长度: {eval_result['mean_length']:.1f}\n")
            
            # 保存最佳模型
            if eval_result['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_result['mean_reward']
                agent.save(os.path.join(checkpoint_dir, 'best_model.pth'))
                print(f"[保存] 新的最佳模型! 奖励: {best_eval_reward:.2f}")
                
        # 定期保存
        if timestep % save_freq == 0:
            agent.save(os.path.join(checkpoint_dir, f'model_step_{timestep}.pth'))
            
    # 训练结束
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"训练完成!")
    print(f"总时间: {total_time/3600:.2f} 小时")
    print(f"总Episode: {episode_count}")
    print(f"最佳评估奖励: {best_eval_reward:.2f}")
    print("=" * 60)
    
    # 保存最终模型
    agent.save(os.path.join(checkpoint_dir, 'final_model.pth'))
    
    # 关闭
    logger.close()
    env.close()
    eval_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LeapHand SAC训练')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    args = parser.parse_args()
    
    train(args.config)
