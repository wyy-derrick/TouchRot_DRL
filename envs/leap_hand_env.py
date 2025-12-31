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
    compute_rotation_delta_z,
    quat_to_rotation_matrix
)


class LeapHandEnv(Env):
    """
    Leap Hand 灵巧手手内旋转任务环境
    
    观测空间:
        - qpos: 16维 (16个关节位置)
        - qvel: 16维 (16个关节速度)
        - tactile: 19维 (19个触觉传感器二值状态)
        - target_axis: 3维 (目标旋转轴)
        - last_action: 16维 (上一帧平滑动作)
        总维度: 70
        
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
        
        # 观测空间: qpos(16) + qvel(16) + tactile(19) + target_axis(3) + last_action(16) = 70
        obs_dim = self.n_joints * 2 + NUM_SENSORS + 3 + self.n_joints
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

        # 动作平滑 (EMA) 设置
        self.act_smoothing = self.config.get('act_smoothing', 0.8)
        self.last_smoothed_action = np.zeros(self.n_joints, dtype=np.float32)

        # 指尖Site列表（用于距离奖励）
        default_tip_sites = ['if_tip_site', 'mf_tip_site', 'rf_tip_site', 'th_tip_site']
        tip_site_names = self.config.get('tip_site_names', default_tip_sites)
        config_tip_weights = self.config.get('tip_site_weights')
        if config_tip_weights is not None:
            base_tip_weights = [float(w) for w in config_tip_weights]
        else:
            base_tip_weights = [0.55 if name == 'th_tip_site' else 0.15 for name in tip_site_names]

        self.tip_site_ids = []
        self.tip_site_weights = []
        for idx, site_name in enumerate(tip_site_names):
            try:
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            except mujoco.Error:
                site_id = -1
            if site_id != -1:
                weight = base_tip_weights[idx] if idx < len(base_tip_weights) else 0.0
                self.tip_site_ids.append(site_id)
                self.tip_site_weights.append(weight)
        
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
        
        # 重置平滑动作缓存
        self.last_smoothed_action = np.zeros(self.n_joints, dtype=np.float32)

        # 重置物体位置
        self._reset_box_position()
        
        # 设置th_cmc_act初始值为-0.68
        th_cmc_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'th_cmc_act')
        if th_cmc_actuator_id != -1:
            actuator_joint_id = self.model.actuator_trnid[th_cmc_actuator_id, 0]
            self.data.qpos[actuator_joint_id] = -0.68
            # 同时设置控制信号，使电机保持在该位置
            self.data.ctrl[th_cmc_actuator_id] = -0.68
        
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
        # 应用动作 (相对位置控制 + EMA平滑)
        raw_action = np.clip(action, -1.0, 1.0)
        smoothed_action = (
            self.act_smoothing * raw_action +
            (1.0 - self.act_smoothing) * self.last_smoothed_action
        )
        self.last_smoothed_action = smoothed_action.copy()

        current_ctrl = self.data.ctrl.copy()
        target_ctrl = current_ctrl + smoothed_action * self.action_scale
        
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
        reward, reward_info = self._compute_reward(smoothed_action, current_box_quat)
        
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
        
        # 上一帧平滑动作
        last_act = self.last_smoothed_action.copy()

        # 拼接观测 (68维)
        obs = np.concatenate([qpos, qvel, tactile, target, last_act]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self, action, current_quat):
        """
        计算奖励
        新增: torque, work, distance 奖励项
        """
        reward_info = {}

        # 0. 物体与指尖距离奖励 (有界)
        box_pos = self._get_box_position()
        fingertip_accum = 0.0
        fingertip_weight_sum = 0.0
        for site_id, tip_weight in zip(self.tip_site_ids, getattr(self, 'tip_site_weights', [])):
            tip_pos = self.data.site_xpos[site_id]
            dist = np.linalg.norm(tip_pos - box_pos)
            tip_reward = np.clip(0.1 / (0.02 + 4.0 * dist), 0.0, 1.0)
            fingertip_accum += tip_weight * tip_reward
            fingertip_weight_sum += tip_weight
        if fingertip_weight_sum > 0.0:
            r_fingertip_dist = float(fingertip_accum / fingertip_weight_sum)
        else:
            r_fingertip_dist = 0.0
        reward_info['fingertip_dist'] = r_fingertip_dist
        
        # 1. 旋转奖励 (spin_coef)
        delta_angle = compute_rotation_delta_z(self.prev_box_quat, current_quat)
        r_rot = np.clip(delta_angle, -0.157, 0.157)
        reward_info['rotation'] = r_rot
        
        # 2. 速度惩罚 (vel_coef)
        box_vel = self._get_box_velocity()
        r_velocity = np.linalg.norm(box_vel[:3])
        reward_info['velocity'] = r_velocity

        # 3. 力矩惩罚 (torque_coef)
        # 获取执行器输出的力/力矩
        # 注意：只取前 n_joints 个，对应手部关节
        actuator_forces = self.data.qfrc_actuator[:self.n_joints]
        r_torque = np.sum(np.square(actuator_forces))
        reward_info['torque'] = r_torque

        # 4. 功/功率惩罚 (work_coef)
        # 功率 P = Force * Velocity
        joint_vel = self.data.qvel[:self.n_joints]
        r_work = np.sum(np.abs(actuator_forces * joint_vel))
        reward_info['work'] = r_work

        # 5. 距离惩罚 (distRewardScale)
        # 计算物体偏离手掌中心的距离
        # 假设手掌中心安全区域大概在 x=-0.05, y=0.05
        target_pos_xy = np.array([-0.05, 0.05]) # 目标XY中心
        dist = np.linalg.norm(box_pos[:2] - target_pos_xy)
        r_dist = dist
        reward_info['distance'] = r_dist
        
        # 6. 掉落惩罚 (fallPenalty)
        # 已经在 _check_termination 中处理重置，这里作为单步惩罚
        r_drop = 0.0
        unsafe_x = box_pos[0] < -0.145 or box_pos[0] > 0.045
        unsafe_y = box_pos[1] < 0.01 or box_pos[1] > 0.10
        if box_pos[2] < self.drop_height or unsafe_x or unsafe_y:
            r_drop = 1.0
        reward_info['drop'] = r_drop

        # 7. 动作惩罚 (action_coef)
        r_action = np.sum(np.square(action))
        reward_info['action'] = r_action

        # 8. 非物体接触惩罚 (collision_coef)
        r_collision = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 < 0 or contact.geom2 < 0:
                continue

            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if body1 == self.box_body_id or body2 == self.box_body_id:
                continue  # 允许物体相关接触

            contact_force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, contact_force)
            r_collision += np.linalg.norm(contact_force[:3])

        reward_info['collision'] = r_collision
        
        # 获取权重 (兼容旧配置，如果没有定义的键则默认为0)
        weights = self.reward_weights

        # 计算总奖励
        total_reward = (
            weights.get('rotation', 0) * r_rot +
            weights.get('velocity', 0) * r_velocity +
            weights.get('torque', 0) * r_torque +
            weights.get('work', 0) * r_work +
            weights.get('fingertip_dist', 0) * r_fingertip_dist +
            weights.get('distance', 0) * r_dist +
            weights.get('drop', 0) * r_drop +
            weights.get('collision', 0) * r_collision +
            weights.get('action', 0) * r_action
        )
        
        reward_info['total'] = total_reward
        
        return total_reward, reward_info
    
    def _check_termination(self):
        """检查是否终止 (重置条件)"""
        box_pos = self._get_box_position()
        
        # 1. 掉落检测 (Z轴高度)
        if box_pos[2] < self.drop_height:
            return True
            
        # 2. 检查物体 XY 是否超出安全区域
        # X: [-0.145, 0.045] 对应左右范围
        # Y: [0.01, 0.10] 对应手掌前后安全区
        if (
            box_pos[0] < -0.145 or box_pos[0] > 0.045 or
            box_pos[1] < 0.01 or box_pos[1] > 0.10
        ):
            return True

        # 3. 检查物体旋转轴倾角 (Z轴偏差)
        current_quat = self._get_box_quaternion()
        # 将四元数转为旋转矩阵
        R = quat_to_rotation_matrix(current_quat)
        # 旋转矩阵的第三列 R[:, 2] 是物体局部 Z 轴在世界坐标系下的方向向量
        obj_z_axis = R[:, 2]
        # 目标 Z 轴 (世界坐标系 Z 轴)
        target_axis = np.array([0.0, 0.0, 1.0])
        
        # 计算夹角: dot = cos(theta)
        dot_product = np.clip(np.dot(obj_z_axis, target_axis), -1.0, 1.0)
        angle_diff = np.arccos(dot_product)
        
        # 阈值 0.4 * pi
        if angle_diff > 0.4 * np.pi:
            return True

        # 4. 检查是否与地面接触 (原有逻辑)
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
        # 限制在手心中心的小圆区域
        center_x, center_y = -0.11, 0.05
        radius = 0.01
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, radius)
        new_x = center_x + r * np.cos(angle)
        new_y = center_y + r * np.sin(angle)
        new_z = 0.15
        
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
