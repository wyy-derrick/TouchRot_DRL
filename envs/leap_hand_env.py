"""
Leap Hand 强化学习环境
基于MuJoCo的灵巧手手内旋转任务环境
"""

import os
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium import Env
from gymnasium.spaces import Box

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tactile_utils import (
    SENSOR_NAMES, NUM_SENSORS, 
    get_binary_tactile_state, 
    get_sensor_data_with_tolerance
)
from utils.math_utils import (
    quat_to_euler, 
    get_rotation_angle_z, 
    compute_rotation_delta_z
)


class LeapHandEnv(Env):
    """
    Leap Hand 灵巧手手内旋转任务环境
    
    观测空间:
        - qpos: 16维 (16个关节位置)
        - qvel: 16维 (16个关节速度)
        - tactile: 17维 (17个触觉传感器二值状态)
        - target_axis: 3维 (目标旋转轴)
        总维度: 52
        
    动作空间:
        - 16维连续动作 [-1, 1]
        
    奖励函数:
        - r_rot: 旋转奖励 (绕目标轴旋转的增量)
        - r_contact: 接触奖励 (鼓励指尖接触物体)
        - r_velocity: 速度惩罚 (防止物体飞出)
        - r_action: 动作惩罚 (平滑动作)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, config=None, render_mode=None):
        """
        初始化环境
        
        Args:
            config: 配置字典
            render_mode: 渲染模式 ('human' 或 'rgb_array' 或 None)
        """
        super().__init__()
        
        # 默认配置
        self.config = config or {}
        
        # 路径配置
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = self.config.get('model_path', 
                                           os.path.join(self.base_path, 'scene_left(cubic).xml'))
        
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        
        # 仿真参数
        self.dt = self.model.opt.timestep
        self.frame_skip = self.config.get('frame_skip', 5)  # 每个RL步执行的物理步数
        
        # 环境参数
        self.max_episode_steps = self.config.get('max_episode_steps', 1000)
        self.current_step = 0
        
        # 动作缩放因子
        self.action_scale = self.config.get('action_scale', 0.1)
        
        # 奖励权重
        self.reward_weights = self.config.get('reward_weights', {
            'rotation': 10.0,      # 旋转奖励权重
            'contact': 1.0,        # 接触奖励权重
            'velocity': -0.1,      # 速度惩罚权重
            'action': -0.01,       # 动作惩罚权重
            'drop': -50.0          # 掉落惩罚
        })
        
        # 触觉参数
        self.tactile_threshold = self.config.get('tactile_threshold', 0.01)
        self.tactile_margin = self.config.get('tactile_margin', 0.005)
        
        # 获取物体和地面ID
        self.box_body_name = 'palm_box'
        self.floor_geom_name = 'floor'
        self.box_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.box_body_name)
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.floor_geom_name)
        self.box_joint_id = self.model.body_jntadr[self.box_body_id]
        
        # 目标旋转轴 (默认Z轴)
        self.target_axis = np.array([0.0, 0.0, 1.0])
        
        # 定义空间
        self.n_joints = 16  # 手部关节数
        
        # 动作空间: 16个关节的控制信号
        self.action_space = Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.n_joints,), 
            dtype=np.float32
        )
        
        # 观测空间: qpos(16) + qvel(16) + tactile(17) + target_axis(3) = 52
        obs_dim = self.n_joints * 2 + NUM_SENSORS + 3
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # 渲染设置
        self.render_mode = render_mode
        self.viewer = None
        
        # 上一步的物体姿态（用于计算旋转增量）
        self.prev_box_quat = None
        
        # 掉落检测阈值
        self.drop_height = self.config.get('drop_height', 0.05)
        
    def reset(self, seed=None, options=None):
        """
        重置环境
        
        Returns:
            observation: 初始观测
            info: 额外信息
        """
        super().reset(seed=seed)
        
        # 重置MuJoCo数据
        mujoco.mj_resetData(self.model, self.data)
        
        # 重置物体位置
        self._reset_box_position()
        
        # 前向运动学计算
        mujoco.mj_forward(self.model, self.data)
        
        # 记录初始姿态
        self.prev_box_quat = self._get_box_quaternion().copy()
        
        # 重置计数器
        self.current_step = 0
        
        # 获取观测
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """
        执行一步动作
        
        Args:
            action: 动作向量 (16,)
            
        Returns:
            observation: 新观测
            reward: 奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 额外信息
        """
        # 应用动作 (相对位置控制)
        action = np.clip(action, -1.0, 1.0)
        current_ctrl = self.data.ctrl.copy()
        target_ctrl = current_ctrl + action * self.action_scale
        
        # 限制在执行器范围内
        for i in range(self.n_joints):
            target_ctrl[i] = np.clip(
                target_ctrl[i],
                self.model.actuator_ctrlrange[i, 0],
                self.model.actuator_ctrlrange[i, 1]
            )
        
        self.data.ctrl[:self.n_joints] = target_ctrl
        
        # 执行物理仿真
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            
        # 更新步数
        self.current_step += 1
        
        # 获取新的物体姿态
        current_box_quat = self._get_box_quaternion()
        
        # 计算奖励
        reward, reward_info = self._compute_reward(action, current_box_quat)
        
        # 更新上一步姿态
        self.prev_box_quat = current_box_quat.copy()
        
        # 检查终止条件
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        # 获取观测
        obs = self._get_observation()
        
        info = {
            'reward_info': reward_info,
            'box_pos': self._get_box_position().copy(),
            'box_quat': current_box_quat.copy(),
            'step': self.current_step
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self):
        """获取当前观测"""
        # 手部关节位置和速度 (只取前16个，排除物体的freejoint)
        qpos = self.data.qpos[:self.n_joints].copy()
        qvel = self.data.qvel[:self.n_joints].copy()
        
        # 归一化关节位置到[-1, 1] (可选优化)
        # qpos_normalized = self._normalize_qpos(qpos)
        
        # 触觉传感器二值状态
        tactile = get_binary_tactile_state(
            self.model, self.data, 
            threshold=self.tactile_threshold,
            margin=self.tactile_margin
        )
        
        # 目标旋转轴
        target = self.target_axis.copy()
        
        # 拼接观测 (保持52维: 16+16+17+3=52)
        obs = np.concatenate([qpos, qvel, tactile, target]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self, action, current_quat):
        """
        计算奖励
        
        奖励公式: r = w_rot * r_rot + w_contact * r_contact + w_vel * r_vel + w_action * r_action
        """
        reward_info = {}
        
        # 1. 旋转奖励 (绕Z轴的旋转增量)
        delta_angle = compute_rotation_delta_z(self.prev_box_quat, current_quat)
        r_rot = delta_angle  # 正向旋转为正奖励
        reward_info['rotation'] = r_rot
        
        # 2. 接触奖励 (鼓励指尖接触物体)
        tactile_state = get_binary_tactile_state(
            self.model, self.data,
            threshold=self.tactile_threshold,
            margin=self.tactile_margin
        )
        # 指尖传感器索引 (索引13-16对应if_tip, mf_tip, rf_tip, th_tip)
        # SENSOR_NAMES中指尖位于索引13,14,15,16
        tip_indices = list(range(13, 17))  # [13, 14, 15, 16]
        tip_contacts = sum(tactile_state[i] for i in tip_indices)
        r_contact = tip_contacts / len(tip_indices)  # 归一化到[0, 1]
        reward_info['contact'] = r_contact
        
        # 3. 速度惩罚 (物体线速度)
        box_vel = self._get_box_velocity()
        r_velocity = np.linalg.norm(box_vel[:3])  # 只考虑线速度
        reward_info['velocity'] = r_velocity
        
        # 4. 动作惩罚 (动作幅度)
        r_action = np.sum(np.square(action))
        reward_info['action'] = r_action
        
        # 5. 掉落惩罚
        box_pos = self._get_box_position()
        r_drop = 0.0
        if box_pos[2] < self.drop_height:
            r_drop = 1.0
        reward_info['drop'] = r_drop
        
        # 计算总奖励
        total_reward = (
            self.reward_weights['rotation'] * r_rot +
            self.reward_weights['contact'] * r_contact +
            self.reward_weights['velocity'] * r_velocity +
            self.reward_weights['action'] * r_action +
            self.reward_weights['drop'] * r_drop
        )
        
        reward_info['total'] = total_reward
        
        return total_reward, reward_info
    
    def _check_termination(self):
        """检查是否终止"""
        # 物体掉落
        box_pos = self._get_box_position()
        if box_pos[2] < self.drop_height:
            return True
            
        # 检查是否与地面接触
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_body = self.model.geom_bodyid[contact.geom1]
            geom2_body = self.model.geom_bodyid[contact.geom2]
            
            is_box_floor = (
                (geom1_body == self.box_body_id and geom2_body == 0) or
                (geom2_body == self.box_body_id and geom1_body == 0)
            )
            if is_box_floor:
                return True
                
        return False
    
    def _reset_box_position(self):
        """
        重置物体位置
        按照1.md要求：方向固定，确保旋转轴（Z轴）相对手掌明确
        """
        # 随机位置（在手掌上方）
        new_x = np.random.uniform(-0.1, 0.02)
        new_y = np.random.uniform(0.015, 0.085)
        new_z = np.random.uniform(0.2, 0.25)
        
        # 获取qpos地址
        qpos_adr = self.model.jnt_qposadr[self.box_joint_id]
        
        # 设置位置
        self.data.qpos[qpos_adr:qpos_adr+3] = [new_x, new_y, new_z]
        
        # 设置四元数：方向固定，仅添加少量随机噪声
        # 确保Z轴旋转方向明确
        noise_scale = 0.05  # 小噪声保持方向基本固定
        noise = np.random.uniform(-noise_scale, noise_scale, size=3)
        quat = np.array([1.0, noise[0], noise[1], noise[2]])
        quat = quat / np.linalg.norm(quat)
        self.data.qpos[qpos_adr+3:qpos_adr+7] = quat
        
        # 重置速度
        qvel_adr = self.model.jnt_dofadr[self.box_joint_id]
        self.data.qvel[qvel_adr:qvel_adr+6] = 0.0
        
    def _reset_hand_position(self):
        """重置手部关节到默认位置（可选）"""
        # 设置手部关节到中间位置
        for i in range(self.n_joints):
            low = self.model.actuator_ctrlrange[i, 0]
            high = self.model.actuator_ctrlrange[i, 1]
            self.data.qpos[i] = (low + high) / 2
            self.data.ctrl[i] = (low + high) / 2
        
    def _get_box_position(self):
        """获取物体位置"""
        qpos_adr = self.model.jnt_qposadr[self.box_joint_id]
        return self.data.qpos[qpos_adr:qpos_adr+3].copy()
    
    def _get_box_quaternion(self):
        """获取物体四元数"""
        qpos_adr = self.model.jnt_qposadr[self.box_joint_id]
        return self.data.qpos[qpos_adr+3:qpos_adr+7].copy()
    
    def _get_box_velocity(self):
        """获取物体速度 [线速度, 角速度]"""
        qvel_adr = self.model.jnt_dofadr[self.box_joint_id]
        return self.data.qvel[qvel_adr:qvel_adr+6].copy()
    
    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            self.viewer.sync()
            
        elif self.render_mode == 'rgb_array':
            # TODO: 实现RGB数组渲染
            pass
            
    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            
    def get_tactile_info(self):
        """获取触觉传感器详细信息（用于调试）"""
        readings = get_sensor_data_with_tolerance(
            self.model, self.data, SENSOR_NAMES, self.tactile_margin
        )
        binary = get_binary_tactile_state(
            self.model, self.data, self.tactile_threshold, self.tactile_margin
        )
        return {
            'readings': readings,
            'binary': binary,
            'sensor_names': SENSOR_NAMES
        }
