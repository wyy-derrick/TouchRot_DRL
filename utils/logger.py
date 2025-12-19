"""
日志工具模块
封装TensorBoard记录功能
"""

import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    TensorBoard日志记录器
    """
    
    def __init__(self, log_dir="logs", experiment_name=None):
        """
        初始化Logger
        
        Args:
            log_dir: 日志根目录
            experiment_name: 实验名称，如果为None则使用时间戳
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        self.log_path = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_path, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_path)
        self.step = 0
        
        print(f"TensorBoard日志保存至: {self.log_path}")
        print(f"运行 'tensorboard --logdir={log_dir}' 查看训练曲线")
        
    def log_scalar(self, tag, value, step=None):
        """
        记录标量数据
        
        Args:
            tag: 数据标签
            value: 数据值
            step: 步数，如果为None则使用内部计数器
        """
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
        
    def log_scalars(self, main_tag, tag_scalar_dict, step=None):
        """
        同时记录多个标量
        
        Args:
            main_tag: 主标签
            tag_scalar_dict: 标签-值字典
            step: 步数
        """
        if step is None:
            step = self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        
    def log_histogram(self, tag, values, step=None):
        """
        记录直方图
        
        Args:
            tag: 数据标签
            values: 数据数组
            step: 步数
        """
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)
        
    def log_episode(self, episode, episode_reward, episode_length, 
                    actor_loss=None, critic_loss=None, alpha=None, extra_info=None):
        """
        记录一个episode的完整信息
        
        Args:
            episode: episode编号
            episode_reward: episode总奖励
            episode_length: episode长度
            actor_loss: Actor损失
            critic_loss: Critic损失
            alpha: 熵系数
            extra_info: 额外信息字典
        """
        self.writer.add_scalar('Episode/Reward', episode_reward, episode)
        self.writer.add_scalar('Episode/Length', episode_length, episode)
        
        if actor_loss is not None:
            self.writer.add_scalar('Loss/Actor', actor_loss, episode)
            
        if critic_loss is not None:
            self.writer.add_scalar('Loss/Critic', critic_loss, episode)
            
        if alpha is not None:
            self.writer.add_scalar('Train/Alpha', alpha, episode)
            
        if extra_info is not None:
            for key, value in extra_info.items():
                self.writer.add_scalar(f'Extra/{key}', value, episode)
                
    def log_training_step(self, step, actor_loss, critic_loss, alpha=None):
        """
        记录训练步信息
        
        Args:
            step: 训练步数
            actor_loss: Actor损失
            critic_loss: Critic损失
            alpha: 熵系数
        """
        self.writer.add_scalar('Loss/Actor', actor_loss, step)
        self.writer.add_scalar('Loss/Critic', critic_loss, step)
        
        if alpha is not None:
            self.writer.add_scalar('Train/Alpha', alpha, step)
            
    def log_reward_components(self, step, reward_dict):
        """
        记录奖励各分量
        
        Args:
            step: 步数
            reward_dict: 奖励分量字典 {'rotation': ..., 'contact': ..., ...}
        """
        for key, value in reward_dict.items():
            self.writer.add_scalar(f'Reward/{key}', value, step)
            
    def log_network_gradients(self, step, model, name_prefix=''):
        """
        记录网络梯度信息（用于调试）
        
        Args:
            step: 步数
            model: 网络模型
            name_prefix: 名称前缀
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'{name_prefix}/grad/{name}', param.grad, step)
                grad_norm = param.grad.norm().item()
                self.writer.add_scalar(f'{name_prefix}/grad_norm/{name}', grad_norm, step)
            
    def increment_step(self):
        """增加内部步数计数器"""
        self.step += 1
        
    def set_step(self, step):
        """设置内部步数计数器"""
        self.step = step
        
    def flush(self):
        """刷新写入缓冲区"""
        self.writer.flush()
        
    def close(self):
        """关闭Logger"""
        self.writer.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
