"""
触觉传感器工具模块
包含传感器数据处理、带容忍度的接触检测、二值化等功能
"""

import numpy as np
import mujoco

# 触觉传感器名称列表 (已核对XML)
SENSOR_NAMES = [
    # 手掌 (7个)
    "palm_1_sensor", "palm_2_sensor", "palm_3_sensor",
    "palm_7a_sensor", "palm_7b_sensor", "palm_7c_sensor",
    "palm_8_sensor",
    # 近节 (4个)
    "if_px_sensor", "mf_px_sensor", "rf_px_sensor", "th_px_sensor",
    # 中节/拇指远节 (4个)
    "if_md_sensor", "mf_md_sensor", "rf_md_sensor", "th_ds_sensor",
    # 指尖 (4个)
    "if_tip_sensor", "mf_tip_sensor", "rf_tip_sensor", "th_tip_sensor"
]

# 传感器数量
NUM_SENSORS = len(SENSOR_NAMES)


def get_sensor_ids(model, sensor_names=None):
    """
    获取传感器ID列表
    
    Args:
        model: MuJoCo模型
        sensor_names: 传感器名称列表，默认使用SENSOR_NAMES
        
    Returns:
        sensor_ids: 传感器ID列表
        sensor_adrs: 传感器在sensordata中的地址列表
    """
    if sensor_names is None:
        sensor_names = SENSOR_NAMES
        
    sensor_ids = []
    sensor_adrs = []
    
    for name in sensor_names:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid == -1:
            print(f"警告: 传感器 '{name}' 未找到")
            continue
        sensor_ids.append(sid)
        sensor_adrs.append(model.sensor_adr[sid])
        
    return sensor_ids, sensor_adrs


def get_sensor_data_with_tolerance(model, data, sensor_names=None, margin=0.005):
    """
    手动计算传感器读数，给 Site 区域增加额外的容忍空间 (margin)。
    
    Args:
        model: MuJoCo模型 (MjModel)
        data: MuJoCo数据 (MjData)
        sensor_names: 传感器名称列表，默认使用SENSOR_NAMES
        margin: 容忍距离 (单位: 米), 比如 0.005 代表向外扩大 5mm
        
    Returns:
        sensor_readings: 字典，键为传感器名称，值为力的大小
    """
    if sensor_names is None:
        sensor_names = SENSOR_NAMES
        
    # 初始化读数
    sensor_readings = {name: 0.0 for name in sensor_names}
    
    # 1. 预先缓存所有传感器对应的 Site 信息 (位置、旋转、大小)
    sites_info = []
    for name in sensor_names:
        # 找到传感器 ID
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id == -1:
            continue
            
        # 找到该传感器绑定的 Site ID
        # model.sensor_objid 存储了传感器绑定的对象 ID (在这里就是 Site 的 ID)
        site_id = model.sensor_objid[sensor_id]
        
        # 获取 Site 的全局信息
        # site_xpos: Site 在世界坐标系的位置
        # site_xmat: Site 在世界坐标系的旋转矩阵 (3x3)
        # site_size: Site 的半长宽高 [x, y, z]
        sites_info.append({
            'name': name,
            'pos': data.site_xpos[site_id].copy(),
            'mat': data.site_xmat[site_id].reshape(3, 3).copy(),
            'size': model.site_size[site_id].copy()
        })

    # 2. 遍历物理引擎产生的所有接触点
    for i in range(data.ncon):
        contact = data.contact[i]
        
        # 获取接触点的位置 (世界坐标)
        contact_pos = contact.pos
        
        # 获取接触力 (我们只关心法向力 normal force)
        c_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, c_force)
        normal_force = c_force[0]  # 法向力通常在索引 0
        
        # 如果力太小，忽略
        if normal_force < 1e-4:
            continue

        # 3. 检查这个接触点属于哪个传感器 (包含容忍度)
        for s in sites_info:
            # --- 核心数学：将世界坐标转换为 Site 局部坐标 ---
            # 向量 V = 接触点 - Site中心
            vec_global = contact_pos - s['pos']
            # 将 V 投影到 Site 的三个轴上 (相当于乘以旋转矩阵的转置)
            vec_local = s['mat'].T @ vec_global
            
            # --- 范围判定 ---
            # 检查是否在 (原始尺寸 + 容忍值) 的盒子内
            # abs(local_x) <= size_x + margin
            bounds = s['size'] + margin
            
            is_inside = (abs(vec_local[0]) <= bounds[0] and 
                         abs(vec_local[1]) <= bounds[1] and 
                         abs(vec_local[2]) <= bounds[2])
            
            if is_inside:
                # 如果在范围内，将力累加到该传感器
                sensor_readings[s['name']] += normal_force

    return sensor_readings


def get_binary_tactile_state(model, data, threshold=0.01, margin=0.005, sensor_names=None):
    """
    获取二值化的触觉状态
    
    Args:
        model: MuJoCo模型
        data: MuJoCo数据
        threshold: 力的阈值，大于此值视为有接触
        margin: 容忍距离
        sensor_names: 传感器名称列表，默认使用SENSOR_NAMES
        
    Returns:
        binary_state: numpy数组，长度等于sensor_names，值为0或1
    """
    if sensor_names is None:
        sensor_names = SENSOR_NAMES
        
    # 获取带容忍度的传感器读数
    sensor_readings = get_sensor_data_with_tolerance(model, data, sensor_names, margin)
    
    # 二值化
    binary_state = np.zeros(len(sensor_names), dtype=np.float32)
    for i, name in enumerate(sensor_names):
        if sensor_readings[name] > threshold:
            binary_state[i] = 1.0
            
    return binary_state


def get_raw_tactile_data(model, data, sensor_names=None):
    """
    直接从MuJoCo获取原始传感器数据（不带容忍度）
    
    Args:
        model: MuJoCo模型
        data: MuJoCo数据
        sensor_names: 传感器名称列表
        
    Returns:
        raw_data: numpy数组，包含原始传感器读数
    """
    if sensor_names is None:
        sensor_names = SENSOR_NAMES
        
    raw_data = np.zeros(len(sensor_names), dtype=np.float32)
    
    for i, name in enumerate(sensor_names):
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id == -1:
            continue
        sensor_adr = model.sensor_adr[sensor_id]
        raw_data[i] = data.sensordata[sensor_adr]
        
    return raw_data


def print_tactile_state(model, data, threshold=0.01, margin=0.005):
    """
    打印触觉传感器状态（用于调试）
    """
    sensor_readings = get_sensor_data_with_tolerance(model, data, SENSOR_NAMES, margin)
    
    print("--- 触觉传感器状态 ---")
    activated = []
    for name in SENSOR_NAMES:
        force = sensor_readings[name]
        if force > threshold:
            activated.append(f"{name}: {force:.4f}N")
            print(f"  [!] {name}: 检测到接触! 力: {force:.5f} N")
        else:
            print(f"  [ ] {name}: 无接触")
    
    print(f"激活的传感器数量: {len(activated)}/{NUM_SENSORS}")
    print("-" * 40)
    
    return activated
