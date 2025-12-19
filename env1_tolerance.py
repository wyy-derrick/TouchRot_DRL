import time
import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = 'scene_left(cubic).xml'
FLOOR_GEOM_NAME = 'floor'  # 地面几何体的名称
BOX_BODY_NAME = 'palm_box' # 立方体Body的名称
# 加载MuJoCo模型 (MjModel)
# MjModel 包含模型的静态描述（几何形状、惯性属性、关节定义等），这些在仿真过程中是不变的。
staic_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
# 初始化MuJoCo数据 (MjData)
# MjData 包含仿真的动态状态（位置 qpos、速度 qvel、受力、接触信息等），这些在每一步仿真中都会更新。
dynamic_data = mujoco.MjData(staic_model)


# MuJoCo 使用整数 ID 来引用模型中的对象。mj_name2id 函数用于根据名称查找 ID。
floor_geom_id = mujoco.mj_name2id(staic_model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM_NAME)
box_body_id = mujoco.mj_name2id(staic_model, mujoco.mjtObj.mjOBJ_BODY, BOX_BODY_NAME)
   


# 获取立方体的关节的ID，用于修改位置
# m.body_jntadr[body_id] 返回该 Body 关联的第一个关节的地址。
# 对于 freejoint (自由关节)，它允许物体在空间中自由移动和旋转。
box_joint_id = staic_model.body_jntadr[box_body_id]

    

# 立方体重置函数 
def reset_box_position(m, d, box_joint_id):
    """将立方体随机放置到一个新的、足够高的位置并重置速度。"""
   
    new_x = np.random.uniform(-0.1, 0.02)
    new_y = np.random.uniform(0.015, 0.085)
    new_z = np.random.uniform(0.2, 0.25) 
    
    # Body在 d.qpos 中的起始地址 (freejoint 占据 7 个 qpos 变量)
    # m.jnt_qposadr[joint_id] 返回该关节在 qpos 数组中的起始索引。
    # freejoint 的 qpos 包含 7 个值: [x, y, z, qw, qx, qy, qz] (位置 + 四元数)
    qpos_adr = m.jnt_qposadr[box_joint_id]
    
    # 设置新的位置 (d.qpos[adr:adr+3])
    # 直接修改 d.qpos 数组来改变物体的位置
    d.qpos[qpos_adr:qpos_adr+3] = [new_x, new_y, new_z]
    
    # 保持四元数不变或设置为单位四元数 [1, 0, 0, 0] (可选：重置方向)
    d.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0] 
    
    # 重置 Body 的速度 (d.qvel[adr:adr+6])
    # m.jnt_dofadr[joint_id] 返回该关节在 qvel 数组中的起始索引。
    # freejoint 的 qvel 包含 6 个值: [vx, vy, vz, wx, wy, wz] (线速度 + 角速度)
    qvel_adr = m.jnt_dofadr[box_joint_id]
    d.qvel[qvel_adr:qvel_adr+6] = 0.0
    
    print(f"立方体重置到: x={new_x:.3f}, y={new_y:.3f}, z={new_z:.3f}")
    
# 初始重置立方体位置
reset_box_position(staic_model, dynamic_data, box_joint_id)

# --- 新增辅助函数：带容忍度的接触检测 ---
def get_sensor_data_with_tolerance(model, data, sensor_names, margin=0.005):
    """
    手动计算传感器读数，给 Site 区域增加额外的容忍空间 (margin)。
    :param margin: 容忍距离 (单位: 米), 比如 0.005 代表向外扩大 5mm
    """
    # 初始化读数
    sensor_readings = {name: 0.0 for name in sensor_names}
    
    # 1. 预先缓存所有传感器对应的 Site 信息 (位置、旋转、大小)
    sites_info = []
    for name in sensor_names:
        # 找到传感器 ID
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        # 找到该传感器绑定的 Site ID
        # model.sensor_objid 存储了传感器绑定的对象 ID (在这里就是 Site 的 ID)
        site_id = model.sensor_objid[sensor_id]
        
        # 获取 Site 的全局信息
        # site_xpos: Site 在世界坐标系的位置
        # site_xmat: Site 在世界坐标系的旋转矩阵 (3x3)
        # site_size: Site 的半长宽高 [x, y, z]
        sites_info.append({
            'name': name,
            'pos': data.site_xpos[site_id],
            'mat': data.site_xmat[site_id].reshape(3, 3),
            'size': model.site_size[site_id]
        })

    # 2. 遍历物理引擎产生的所有接触点
    for i in range(data.ncon):
        contact = data.contact[i]
        
        # 获取接触点的位置 (世界坐标)
        contact_pos = contact.pos
        
        # 获取接触力 (我们只关心法向力 normal force)
        c_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, c_force)
        normal_force = c_force[0] # 法向力通常在索引 0
        
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

# --- 触觉传感器设置  ---
TACTILE_SENSOR_NAMES = [
    "palm_1_sensor", "palm_2_sensor", "palm_3_sensor", "palm_7_sensor", "palm_8_sensor",
    "if_px_sensor", "mf_px_sensor", "rf_px_sensor",
    "if_md_sensor", "mf_md_sensor", "rf_md_sensor",
    "if_tip_sensor", "mf_tip_sensor", "rf_tip_sensor", "th_tip_sensor"
]

sensor_ids = []
sensor_adrs = []
for name in TACTILE_SENSOR_NAMES:
    sid = mujoco.mj_name2id(staic_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    sensor_ids.append(sid)
    # m.sensor_adr[sid] 存储了该传感器数据在 d.sensordata 数组中的起始索引
    sensor_adrs.append(staic_model.sensor_adr[sid])
    

# 1e. 初始化打印计时器 
last_print_time = -1.0
print_interval = 0.5


# 简单的初始控制设置：将所有执行器控制信号设置为0
dynamic_data.ctrl[:] = 0.0

print("模型加载成功，仿真开始...")
print(f"仿真步长: {staic_model.opt.timestep:.5f} 秒")
print(f"执行器数量: {staic_model.nu}")


# 启动 MuJoCo Viewer
with mujoco.viewer.launch_passive(staic_model, dynamic_data) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    
    start_time = time.time()
    
    # 仿真运行的主循环，限制运行时间为 300 秒
    while viewer.is_running() and time.time() - start_time < 300:

        
        step_start = time.time()
        
        # --- 2. 物理步进 ---
        # mujoco.mj_step 是核心仿真函数。它根据当前的 qpos, qvel 和 ctrl 计算下一时刻的状态。
        # 这一步会更新 d.qpos, d.qvel, d.sensordata 以及 d.contact (接触信息) 等。
        mujoco.mj_step(staic_model, dynamic_data)
        
        # --- 3. 新增：检查立方体是否接触地面并重置 ---
        
        # d.ncon 是当前检测到的接触点总数
        # d.contact 是一个结构体数组，包含每个接触点的详细信息
        for i in range(dynamic_data.ncon):
            contact = dynamic_data.contact[i]
            
            # 检查接触点是否发生在 立方体几何体 (geom1) 和 地面几何体 (geom2) 之间
            # 注意: contact.geom1 是较小的 geom ID，contact.geom2 是较大的 geom ID
            # 地面 'floor' (geom_id = floor_geom_id) 的 Body 是世界 Body (ID=0)
            
            # m.geom_bodyid[geom_id] 用于查找几何体所属的 Body ID
            geom1_body_id = staic_model.geom_bodyid[contact.geom1]
            geom2_body_id = staic_model.geom_bodyid[contact.geom2]

            
            # 检查是否有接触发生在 box_body_id (立方体) 和 world body (0, 地面) 之间
            # (确保 box_body_id 确实是立方体的Body ID)
            is_box_floor_contact = (
                (geom1_body_id == box_body_id and geom2_body_id == 0) or
                (geom2_body_id == box_body_id and geom1_body_id == 0)
            )

            if is_box_floor_contact:
                # 确定接触力是否足够大
                # MuJoCo 接触力的法线分量是 force[0] (z轴)
                # 接触力大小可以从 contact.efc_force[adr] (法线接触力) 间接判断,
                # 但直接检查接触数量更简单。
                
                # 简单地：只要检测到接触，就重置
                print("\n--- 接触检测 ---")
                print("!!! 检测到立方体与地面接触。重置立方体位置...")
                reset_box_position(staic_model, dynamic_data, box_joint_id)
                # 一旦重置，退出循环并进行下一步仿真
                break
                
        
        # --- 4. 打印传感器数据 (保持原样，只是修改了编号) ---
        
        if dynamic_data.time - last_print_time >= print_interval:
            print(f"\n--- 触觉传感器状态 (仿真时间: {dynamic_data.time:.3f} s) ---")
            tolerant_readings = get_sensor_data_with_tolerance(
                staic_model, dynamic_data, TACTILE_SENSOR_NAMES, margin=0.01
            )
            for name in TACTILE_SENSOR_NAMES:
                # 直接从我们计算的字典里取值，不再查 d.sensordata
                force_magnitude = tolerant_readings[name]
                
                # 打印状态
                threshold = 1e-4
                if force_magnitude > threshold:
                    print(f"  [!] {name}: 检测到接触! (含容忍度) 力: {force_magnitude:.5f} N")
                else:
                    print(f"  [ ] {name}: 无接触")
            
            
            print("--------------------------------------------------")
            
            last_print_time = dynamic_data.time
            
        
        # --- 5. 可视化同步 (保持原样) ---
        # 将当前的物理状态 (d) 同步到 viewer 进行渲染
        viewer.sync()
        
        # 保持实时仿真速度 (保持原样)
        time_until_next_step = staic_model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

print("仿真结束。")