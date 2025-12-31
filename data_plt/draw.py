"""
训练曲线绘图工具
从 TensorBoard 导出的 CSV 文件中读取数据，绘制各算法的对比图

用法:
    cd data_plt
    python draw.py
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==================== 样式设置 ====================
sns.set(style="darkgrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def smooth_data(data, window_size=40):
    """
    使用滑动平均平滑数据
    
    参数:
        data: 原始数据序列
        window_size: 滑动窗口大小
        
    返回:
        平滑后的数据
    """
    return data.rolling(window=window_size, min_periods=1).mean()


def find_file(algo, metric_tag, file_list):
    """
    根据算法名称和指标标签查找对应的 CSV 文件
    
    参数:
        algo: 算法名称 (如 'SAC', 'TD3', 'PPO', 'Baseline')
        metric_tag: 指标标签 (如 'Train_EpisodeReward')
        file_list: 当前目录下的文件列表
        
    返回:
        匹配的文件名，未找到返回 None
    """
    algo_lower = algo.lower()
    
    for f in file_list:
        # 检查文件是否以 run-{algo} 开头且包含指定的 tag
        if f.startswith(f"run-{algo_lower}") and metric_tag in f:
            return f
    return None


def compute_smart_ylim(all_values, margin_ratio=0.1, percentile_low=2, percentile_high=98):
    """
    智能计算 Y 轴范围，基于数据的百分位数（不删除数据，只调整显示范围）
    
    参数:
        all_values: 所有数据值的列表
        margin_ratio: 上下边距比例
        percentile_low: 下界百分位数
        percentile_high: 上界百分位数
        
    返回:
        (y_min, y_max) 元组
    """
    if len(all_values) == 0:
        return None, None
    
    # 使用百分位数来确定显示范围，避免极端值影响可视化
    y_low = np.percentile(all_values, percentile_low)
    y_high = np.percentile(all_values, percentile_high)
    
    # 计算范围和边距
    y_range = y_high - y_low
    if y_range == 0:
        y_range = abs(y_high) * 0.1 if y_high != 0 else 1.0
    
    margin = y_range * margin_ratio
    
    return y_low - margin, y_high + margin


def plot_single_metric(metric_name, metric_tag, algorithms, file_list, 
                       window_size=40, save_path=None):
    """
    绘制单个指标的对比图
    
    参数:
        metric_name: 指标显示名称
        metric_tag: CSV 文件中的标签名
        algorithms: 算法字典 {算法名: 颜色}
        file_list: CSV 文件列表
        window_size: 平滑窗口大小
        save_path: 保存路径，None 则不保存
    """
    # 创建新图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_title(metric_name, fontsize=16, fontweight='bold')
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    
    has_data = False
    all_values = []  # 收集所有数据用于智能 Y 轴
    
    for algo_name, color in algorithms.items():
        file_path = find_file(algo_name, metric_tag, file_list)
        
        if file_path and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df = df.sort_values('Step')  # 按步数排序
                
                steps = df['Step']
                values = df['Value']
                
                # 收集数据用于计算 Y 轴范围（不删除任何数据）
                all_values.extend(values.tolist())
                
                # 平滑处理
                smoothed_values = smooth_data(values, window_size)
                
                # 绘制原始数据（半透明背景，保留所有数据点）
                ax.plot(steps, values, color=color, alpha=0.2, linewidth=0.5)
                
                # 绘制平滑曲线（主要显示）
                ax.plot(steps, smoothed_values, label=algo_name, 
                       color=color, linewidth=2)
                
                has_data = True
                print(f"  ✓ 加载 {algo_name}: {len(values)} 个数据点")
                
            except Exception as e:
                print(f"  ✗ 读取 {file_path} 失败: {e}")
        else:
            print(f"  - {algo_name}: 未找到数据文件")
    
    if has_data:
        # 智能设置 Y 轴范围（根据数据大小自动选择合适范围）
        y_min, y_max = compute_smart_ylim(all_values)
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No Data Available", 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → 已保存: {save_path}")
    
    # 显示图片
    plt.show()
    plt.close()


def main():
    """主函数：绘制所有训练曲线"""
    
    print("=" * 60)
    print("训练曲线绘图工具")
    print("=" * 60)
    
    # ==================== 配置 ====================
    # 算法配色方案
    algorithms = {
        'SAC': '#d62728',       # 红色
        'TD3': '#2ca02c',       # 绿色
        'PPO': '#1f77b4',       # 蓝色
        'Baseline': '#ff7f0e'   # 橙色
    }
    
    # Metrics config: (display name, CSV tag, save filename)
    metrics = [
        ("Episode Reward", "Train_EpisodeReward", "episode_reward.png"),
        ("Episode Length", "Train_EpisodeLength", "episode_length.png"),
        ("Critic Loss", "Loss_Critic", "loss_critic.png"),
        ("Actor Loss", "Loss_Actor", "loss_actor.png"),
    ]
    
    # ==================== 获取文件列表 ====================
    file_list = [f for f in os.listdir('.') if f.endswith('.csv')]
    print(f"\n找到 {len(file_list)} 个 CSV 文件")
    
    if len(file_list) == 0:
        print("\n错误: 当前目录下没有 CSV 文件!")
        print("请先从 TensorBoard 导出数据到 data_plt 目录")
        return
    
    # ==================== 依次绘制每个指标 ====================
    for metric_name, metric_tag, save_name in metrics:
        print(f"\n绘制: {metric_name}")
        print("-" * 40)
        plot_single_metric(
            metric_name=metric_name,
            metric_tag=metric_tag,
            algorithms=algorithms,
            file_list=file_list,
            window_size=40,
            save_path=save_name
        )
    
    # ==================== 绘制汇总图 ====================
    print(f"\n绘制: 汇总图 (2x2)")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (metric_name, metric_tag, _) in enumerate(metrics):
        ax = axes[i]
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_xlabel("Training Steps", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        
        all_values = []
        has_data = False
        
        for algo_name, color in algorithms.items():
            file_path = find_file(algo_name, metric_tag, file_list)
            
            if file_path and os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df = df.sort_values('Step')
                    
                    steps = df['Step']
                    values = df['Value']
                    all_values.extend(values.tolist())
                    
                    smoothed_values = smooth_data(values, 40)
                    
                    # 绘制所有数据（不删除）
                    ax.plot(steps, values, color=color, alpha=0.15, linewidth=0.5)
                    ax.plot(steps, smoothed_values, label=algo_name, 
                           color=color, linewidth=1.5)
                    has_data = True
                except:
                    pass
        
        if has_data:
            # 智能 Y 轴范围
            y_min, y_max = compute_smart_ylim(all_values)
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', 
                   transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('training_curves_all.png', dpi=300, bbox_inches='tight')
    print(f"  → 已保存: training_curves_all.png")
    plt.show()
    plt.close()
    
    print("\n" + "=" * 60)
    print("绘图完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()