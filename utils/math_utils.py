"""
数学工具模块
包含四元数、欧拉角、旋转计算等功能
"""

import numpy as np


def quat_to_euler(quat):
    """
    四元数转欧拉角 (ZYX顺序)
    
    Args:
        quat: 四元数 [w, x, y, z]
        
    Returns:
        euler: 欧拉角 [roll, pitch, yaw] (弧度)
    """
    w, x, y, z = quat
    
    # Roll (x轴旋转)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y轴旋转)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # 使用90度如果超出范围
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z轴旋转)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


def euler_to_quat(euler):
    """
    欧拉角转四元数 (ZYX顺序)
    
    Args:
        euler: 欧拉角 [roll, pitch, yaw] (弧度)
        
    Returns:
        quat: 四元数 [w, x, y, z]
    """
    roll, pitch, yaw = euler
    
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def quat_multiply(q1, q2):
    """
    四元数乘法 q1 * q2
    
    Args:
        q1, q2: 四元数 [w, x, y, z]
        
    Returns:
        result: 四元数乘积 [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return np.array([w, x, y, z])


def quat_inverse(q):
    """
    四元数求逆
    
    Args:
        q: 四元数 [w, x, y, z]
        
    Returns:
        q_inv: 四元数的逆 [w, -x, -y, -z] (假设单位四元数)
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q):
    """
    四元数归一化
    
    Args:
        q: 四元数 [w, x, y, z]
        
    Returns:
        q_normalized: 单位四元数
    """
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def get_rotation_angle_z(quat):
    """
    从四元数获取绕Z轴的旋转角度
    
    Args:
        quat: 四元数 [w, x, y, z]
        
    Returns:
        angle: 绕Z轴的旋转角度 (弧度)
    """
    euler = quat_to_euler(quat)
    return euler[2]  # yaw角即为绕Z轴旋转


def compute_rotation_delta_z(quat_prev, quat_curr):
    """
    计算两个姿态之间绕Z轴的旋转增量
    
    Args:
        quat_prev: 上一时刻四元数 [w, x, y, z]
        quat_curr: 当前时刻四元数 [w, x, y, z]
        
    Returns:
        delta_angle: 绕Z轴的旋转增量 (弧度)
    """
    angle_prev = get_rotation_angle_z(quat_prev)
    angle_curr = get_rotation_angle_z(quat_curr)
    
    delta = angle_curr - angle_prev
    
    # 处理角度跨越 -pi/pi 边界的情况
    while delta > np.pi:
        delta -= 2 * np.pi
    while delta < -np.pi:
        delta += 2 * np.pi
        
    return delta


def rotation_matrix_to_quat(R):
    """
    旋转矩阵转四元数
    
    Args:
        R: 3x3旋转矩阵
        
    Returns:
        quat: 四元数 [w, x, y, z]
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
        
    return quat_normalize(np.array([w, x, y, z]))


def quat_to_rotation_matrix(quat):
    """
    四元数转旋转矩阵
    
    Args:
        quat: 四元数 [w, x, y, z]
        
    Returns:
        R: 3x3旋转矩阵
    """
    w, x, y, z = quat_normalize(quat)
    
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    
    return R


def angle_axis_to_quat(axis, angle):
    """
    轴角表示转四元数
    
    Args:
        axis: 旋转轴 (单位向量)
        angle: 旋转角度 (弧度)
        
    Returns:
        quat: 四元数 [w, x, y, z]
    """
    axis = np.array(axis) / np.linalg.norm(axis)
    half_angle = angle / 2
    
    w = np.cos(half_angle)
    xyz = np.sin(half_angle) * axis
    
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def compute_angular_velocity(quat_prev, quat_curr, dt):
    """
    从两个姿态计算角速度
    
    Args:
        quat_prev: 上一时刻四元数
        quat_curr: 当前时刻四元数
        dt: 时间间隔
        
    Returns:
        omega: 角速度向量 [wx, wy, wz]
    """
    if dt < 1e-8:
        return np.zeros(3)
    
    # q_diff = q_curr * q_prev^-1
    q_diff = quat_multiply(quat_curr, quat_inverse(quat_prev))
    q_diff = quat_normalize(q_diff)
    
    # 从q_diff提取角速度
    # 对于小角度: omega ≈ 2 * [x, y, z] / dt
    omega = 2 * q_diff[1:4] / dt
    
    # 如果w为负，取共轭
    if q_diff[0] < 0:
        omega = -omega
        
    return omega
