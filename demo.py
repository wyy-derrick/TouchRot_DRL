"""
演示脚本
加载训练好的模型，可视化展示效果

用法:
    python demo.py
    python demo.py --model checkpoints/sac/best_model.pth
    python demo.py --episodes 5
"""

import os
import sys
import argparse
import time
import numpy as np
import yaml
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms import SACAgent, TD3Agent, PPOAgent, BaselineController
from envs import LeapHandEnv
from utils.tactile_utils import SENSOR_NAMES


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def print_tactile_state(tactile_info):
    """打印触觉传感器状态"""
    readings = tactile_info['readings']
    binary = tactile_info['binary']
    
    print("\n--- 触觉传感器状态 ---")
    activated = []
    for i, name in enumerate(SENSOR_NAMES):
        force = readings[name]
        is_active = binary[i] > 0.5
        
        if is_active:
            activated.append(name)
            print(f"  [✓] {name}: {force:.4f} N")
            
    print(f"\n激活: {len(activated)}/{len(SENSOR_NAMES)} 个传感器")
    print("-" * 40)


def build_demo_agent(algo_name, state_dim, action_dim, algo_config):
    algo_name = algo_name.lower()
    if algo_name == 'sac':
        return SACAgent(state_dim=state_dim, action_dim=action_dim, config=algo_config)
    if algo_name == 'td3':
        return TD3Agent(state_dim=state_dim, action_dim=action_dim, config=algo_config)
    if algo_name == 'ppo':
        return PPOAgent(state_dim=state_dim, action_dim=action_dim, config=algo_config)
    if algo_name == 'baseline':
        return BaselineController(state_dim=state_dim, action_dim=action_dim, config=algo_config)
    raise ValueError(f"不支持的算法: {algo_name}")


def demo(model_path, config_path='configs/config.yaml', num_episodes=50, print_tactile=True, algo='sac'):
    """演示函数"""
    
    # 创建日志目录和文件
    log_dir = 'demo_logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'demo_log_{algo}_{timestamp}.txt')
    
    # 重定向stdout到文件和终端
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", buffering=1)
   
 # 行缓冲

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
    # 设置日志输出
    sys.stdout = Logger(log_file)
    
    # 加载配置
    config = load_config(config_path)
    env_config = config['env']
    algo_name = algo.lower()
    if algo_name not in {'sac', 'td3', 'ppo', 'baseline'}:
        raise ValueError(f"不支持的算法: {algo_name}")
    algo_config = config.get(algo_name)
    if algo_config is None:
        raise KeyError(f"配置中缺少 {algo_name} 段")
    demo_config = config.get('demo', {})
    
    render_fps = demo_config.get('render_fps', 30)
    
    # 创建环境 (带渲染)
    print("创建环境...")
    env = LeapHandEnv(config=env_config, render_mode='human')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"状态维度: {state_dim}, 动作维度: {action_dim}")
    
    # 创建Agent并加载模型
    print(f"创建Agent ({algo_name.upper()})...")
    agent = build_demo_agent(algo_name, state_dim, action_dim, algo_config)
    if algo_name != 'baseline':
        print(f"加载模型: {model_path}")
        agent.load(model_path)
    agent.eval_mode()
    
    print("=" * 60)
    print("开始演示")
    print(f"Episode数: {num_episodes}")
    print("按 Ctrl+C 退出")
    print("=" * 60)
    print(f"日志文件: {log_file}")
    
    total_rewards = []
    
    try:
        for ep in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            step = 0
            
            print(f"\n=== Episode {ep + 1}/{num_episodes} ===")
            
            while not (done or truncated):
                # 选择动作 (确定性策略)
                action = agent.select_action(state, evaluate=True)
                
                # 执行动作
                next_state, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                step += 1
                
                # 渲染
                env.render()
                
                # 打印更多信息用于调试
                if step % 10 == 0:  # 更频繁地打印调试信息
                    print(f"Step {step}:")
                    print(f"  Action mean: {np.mean(action):.4f}, std: {np.std(action):.4f}")
                    print(f"  Action min: {np.min(action):.4f}, max: {np.max(action):.4f}")
                    print(f"  Reward: {reward:.4f}")
                    if 'reward_info' in info:
                        print(f"  Reward components: {info['reward_info']}")
                    print(f"  State mean: {np.mean(state):.4f}, std: {np.std(state):.4f}")
                    print(f"  State min: {np.min(state):.4f}, max: {np.max(state):.4f}")
                    print("-" * 40)
                
                # 打印触觉信息
                if print_tactile and step % 50 == 0:
                    tactile_info = env.get_tactile_info()
                    print_tactile_state(tactile_info)
                    
                    # 打印物体位置
                    box_pos = info.get('box_pos', [0, 0, 0])
                    print(f"物体位置: x={box_pos[0]:.3f}, y={box_pos[1]:.3f}, z={box_pos[2]:.3f}")
                    
                state = next_state
                
                # 控制帧率
                time.sleep(1.0 / render_fps)
                
            total_rewards.append(episode_reward)
            
            print(f"\nEpisode {ep + 1} 结束")
            print(f"总奖励: {episode_reward:.2f}")
            print(f"步数: {step}")
            
            if done:
                print("终止原因: 物体掉落")
            elif truncated:
                print("终止原因: 达到最大步数")
                
    except KeyboardInterrupt:
        print("\n用户中断演示")
        
    finally:
        # 统计
        if total_rewards:
            print("\n" + "=" * 60)
            print("演示统计")
            print(f"完成Episode数: {len(total_rewards)}")
            print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
            print(f"最高奖励: {np.max(total_rewards):.2f}")
            print(f"最低奖励: {np.min(total_rewards):.2f}")
            print("=" * 60)
            
        env.close()
        
        # 恢复stdout
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal
        print(f"日志已保存到: {log_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LeapHand 模型演示')
    parser.add_argument('--model', type=str, default='checkpoints/sac/sac_20251231_115018_step_250000.pth',
                        help='模型路径')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--episodes', type=int, default=50,
                        help='演示episode数')
    parser.add_argument('--no-tactile', action='store_true',
                        help='不打印触觉信息')
    parser.add_argument('--algo', type=str, default='sac', choices=['sac', 'td3', 'ppo', 'baseline'],
                        help='选择演示算法')
    args = parser.parse_args()
    
    demo(
        model_path=args.model,
        config_path=args.config,
        num_episodes=args.episodes,
        print_tactile=not args.no_tactile,
        algo=args.algo
    )