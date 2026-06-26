# Sim2Real ROS2 部署开发指南

> 目标平台: 地瓜 RDK5 (4核 4GB, aarch64) | 中间件: ROS2 Humble  
> 参考实现: UniLab/Sim2real (C++14/ROS Noetic)

---

## 1. 整体架构

```
┌──────────────────────────────────────────────────────────┐
│  物理硬件                                                  │
│  ├─ LPMS-SI2 IMU  ──USB/UART──→ /dev/imu                  │
│  └─ 电机 (CAN/RS485)  ──USB转CANFD──→ /dev/motor             │
└──────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ LPMS-SI2_imu │ │ drive_motor  │ │   sim2real   │
│              │ │              │ │              │
│  发布话题:    │ │  发布话题:    │ │  核心节点:    │
│  /imu/data   │ │  /motor_state│ │  订阅以上话题  │
│              │ │              │ │  RL推理       │
│              │ │  订阅话题:    │ │  发布:        │
│              │ │  /motor_cmd  │ │  /motor_cmd   │
└──────────────┘ └──────────────┘ └──────────────┘
                          ▲               │
                          │               │
                     motor_state      motor_cmd
                   (位置/速度/力矩)   (目标位置/Kp/Kd)
```

### 数据流

```
LPMS-SI2_imu → /imu/data (sensor_msgs/Imu)
    │
    ▼
┌──────────────────────────────────────────────┐
│  sim2real 节点                                │
│                                                │
│  输入:                                          │
│    /imu/data     → 姿态/角速度/加速度            │
│    /motor_state  → 电机位置/速度/力矩            │
│    /joy          → 手柄指令 (可选)               │
│                                                │
│  RL推理线程 (50Hz):                              │
│    MotorState → RobotState 坐标变换              │
│    构造观测 → 策略推理 → 动作输出                 │
│    RobotOutput → MotorCmd 逆变换                │
│    写入 motor_cmd_ (共享)                       │
│                                                │
│  PD控制线程 (500Hz):                             │
│    读取 motor_cmd_ (ZOH)                        │
│    发送 /motor_cmd                             │
│    坠落检测                                      │
└──────────────────────────────────────────────┘
    │
    ▼
drive_motor ← /motor_cmd → CANFD → 电机
(伺服PD闭环 1-10kHz)
```

---

## 2. 包分工与接口定义

### 2.1 drive_motor — 电机驱动包

**职责**: 硬件抽象层，负责与电机通信、收发指令。

#### 发布话题

| 话题 | 消息类型 | 频率 | 说明 |
|------|---------|------|------|
| `/motor_state` | `MotorState.msg` | 500Hz | 电机当前状态反馈 |

```msg
# MotorState.msg (自定义消息)
float32[] position       # N个关节当前位置 (rad)
float32[] velocity       # N个关节当前速度 (rad/s)
float32[] torque         # N个关节当前力矩 (N·m)
```

**位置数据含义**: 每个电机返回的绝对编码器值，未经任何坐标变换。sim2real会通过 `urdf_offset` 和 `direction` 转换成机器人关节角度。

#### 订阅话题

| 话题 | 消息类型 | 频率 | 说明 |
|------|---------|------|------|
| `/motor_cmd` | `MotorCmd.msg` | 500Hz | sim2real下发的电机指令 |

```msg
# MotorCmd.msg (自定义消息)
float32[] target_q       # N个关节的目标位置 (电机坐标系, rad)
float32[] target_dq      # N个关节的目标速度 (rad/s), 可为0
float32[] target_tau     # N个关节的目标力矩 (N·m), 可为0
float32[] target_kp      # N个关节的PD比例增益
float32[] target_kd      # N个关节的PD微分增益
```

#### 必须实现的核心功能

```
drive_motor 节点内部:
  ┌─────────────────────────────────────────────┐
  │ main_loop() @ 500Hz                         │
  │                                              │
  │ 1. 读取电机编码器                             │
  │    → 每个电机: position, velocity, torque     │
  │                                              │
  │ 2. 发布 /motor_state                         │
  │    → MotorState.position[] = raw_encoder      │
  │    → MotorState.velocity[] = raw_velocity     │
  │    → MotorState.torque[]   = raw_torque       │
  │                                              │
  │ 3. 检查是否有新的 /motor_cmd                  │
  │    if motor_cmd 为空:                         │
  │      → keepLastCmd()  保持上一帧指令           │
  │                                              │
  │ 4. 发送指令到硬件                             │
  │    for each motor i:                         │
  │      cmd[i] = {                               │
  │        position: motor_cmd.target_q[i],       │
  │        velocity: motor_cmd.target_dq[i],      │
  │        torque:   motor_cmd.target_tau[i],     │
  │        kp:       motor_cmd.target_kp[i],      │
  │        kd:       motor_cmd.target_kd[i]       │
  │      }                                        │
  │    serial_send(cmd)                           │
  └─────────────────────────────────────────────┘
```

#### 硬件通信实现要点

```python
# drive_motor 伪代码
class DriveMotorNode(Node):
    def __init__(self):
        # 参数
        self.declare_parameter('num_motors', 12)
        self.declare_parameter('serial_port', '/dev/motor')
        self.declare_parameter('baudrate', 4000000)
        self.declare_parameter('motor_ids', [1,2,3,4,5,6,7,8,9,10,11,12])
        self.declare_parameter('direction', [1,1,-1,-1,1,1, -1,1,-1,1,-1,1])
        
        self.num_motors = self.get_parameter('num_motors').value
        self.last_cmd = MotorCmd()  # 保持上一次指令
        
        # ROS2
        self.state_pub = self.create_publisher(MotorState, '/motor_state', 10)
        self.cmd_sub = self.create_subscription(
            MotorCmd, '/motor_cmd', self.cmd_callback, 10)
        
        # 串口通信
        self.serial = self.open_serial()
        
        # 500Hz 主循环
        self.timer = self.create_timer(0.002, self.loop)  # 2ms = 500Hz
    
    def cmd_callback(self, msg):
        self.last_cmd = msg  # 更新最新指令
    
    def loop(self):
        # 1. 读取电机状态
        raw_positions, raw_velocities, raw_torques = self.read_motors()
        
        # 2. 发布状态
        state = MotorState()
        state.position = raw_positions
        state.velocity = raw_velocities
        state.torque = raw_torques
        self.state_pub.publish(state)
        
        # 3. 发送指令 (使用最新收到的指令)
        self.send_motor_cmd(self.last_cmd)
    
    def read_motors(self):
        """通过CAN/RS485协议读取电机编码器"""
        # 发送查询指令 → 等待回复 → 解析position/velocity/torque
        pass
    
    def send_motor_cmd(self, cmd):
        """发送位置+速度+力矩+PD增益到电机"""
        for i in range(self.num_motors):
            packet = build_packet(
                motor_id = self.motor_ids[i],
                position = cmd.target_q[i],
                velocity = cmd.target_dq[i],
                torque   = cmd.target_tau[i],
                kp       = cmd.target_kp[i],
                kd       = cmd.target_kd[i]
            )
            self.serial.write(packet)
```

---

### 2.2 LPMS-SI2_imu — IMU驱动包

**职责**: 驱动LPMS-SI2传感器，发布标准ROS2 IMU消息。

#### 发布话题

| 话题 | 消息类型 | 频率 | 说明 |
|------|---------|------|------|
| `/imu/data` | `sensor_msgs/Imu` | 200-500Hz | 姿态+角速度+加速度 |

```msg
# sensor_msgs/Imu (ROS2 标准消息，无需自定义)
std_msgs/Header header
    builtin_interfaces/Time stamp
    string frame_id                    # "imu_link"

geometry_msgs/Quaternion orientation   # 四元数 w,x,y,z
float64[9] orientation_covariance

geometry_msgs/Vector3 angular_velocity # 角速度 (rad/s)
float64[9] angular_velocity_covariance

geometry_msgs/Vector3 linear_acceleration  # 线加速度 (m/s²)
float64[9] linear_acceleration_covariance
```

#### 必须实现的核心功能

```python
class LPMSIMUNode(Node):
    def __init__(self):
        self.declare_parameter('port', '/dev/imu')
        self.declare_parameter('baudrate', 921600)
        self.declare_parameter('frequency', 200)  # Hz
        
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.serial = self.open_serial()
        
        # 200Hz 主循环
        dt = 1.0 / self.get_parameter('frequency').value
        self.timer = self.create_timer(dt, self.loop)
    
    def loop(self):
        # 1. 读取LPMS-SI2数据包
        raw = self.serial.read_until(...)
        
        # 2. 解析四元数 (LPMS-SI2输出标准四元数)
        qx, qy, qz, qw = parse_quaternion(raw)
        
        # 3. 解析角速度 (或者从四元数差分计算)
        gx, gy, gz = parse_gyro(raw)  # rad/s
        
        # 4. 解析加速度
        ax, ay, az = parse_accel(raw)  # m/s²
        
        # 5. 构造并发布消息
        msg = Imu()
        msg.header.stamp      = self.get_clock().now().to_msg()
        msg.header.frame_id   = "imu_link"
        msg.orientation.x     = qx
        msg.orientation.y     = qy
        msg.orientation.z     = qz
        msg.orientation.w     = qw
        msg.angular_velocity.x = gx
        msg.angular_velocity.y = gy
        msg.angular_velocity.z = gz
        msg.linear_acceleration.x = ax
        msg.linear_acceleration.y = ay
        msg.linear_acceleration.z = az
        
        self.imu_pub.publish(msg)
```

#### sim2real 需要从 IMU 获得的维度

| 字段 | 维度 | 用途 |
|------|------|------|
| `orientation` (四元数) | 4 | → Euler角 → 策略观测: 欧拉角或重力投影 |
| `angular_velocity` | 3 | → 策略观测前3维: `base_ang_vel × scale` |
| `linear_acceleration` | 3 | → 坠落检测 (加速度 > 阈值) |

---

## 3. sim2real — 核心推理部署包

### 3.1 节点结构

```
sim2real 节点 (ROS2)
├─ 订阅:
│   ├─ /imu/data       (sensor_msgs/Imu)         @ 200Hz
│   ├─ /motor_state    (MotorState)               @ 500Hz
│   └─ /cmd_vel        (geometry_msgs/Twist)     @ 50Hz
│
├─ RL推理线程: @ 50Hz
│   ├─ motor_state → RobotState 变换
│   ├─ 构造观测向量
│   ├─ ONNX Runtime 推理
│   ├─ 动作裁剪 + 缩放
│   ├─ RobotOutput → MotorCmd 逆变换
│   └─ 写入 motor_cmd_ (outputMutex_)
│
├─ PD控制线程: @ 500Hz
│   ├─ 读取 motor_cmd_ (ZOH)
│   ├─ 发送 /motor_cmd
│   └─ 坠落检测
│
└─ 发布:
    └─ /motor_cmd (MotorCmd) @ 500Hz
```

### 3.2 观测输入构造 (策略需要什么)

基于参考实现 AMPPolicy (69维) 和 HumanoidgymPolicy (47维)，你的策略训练时必须匹配以下之一：

#### 方案A: HumanoidgymPolicy (47维, 更简单)

```python
# 单帧观测: sin/cos(2) + cmd(3) + q(N) + dq(N) + last(N) + ang_vel(3) + euler(3) = 47
# 其中 N = 机器人关节数

idx = 0
obs[idx:idx+2]   = [sin(2π·t/f), cos(2π·t/f)]     # 步态相位
idx += 2
obs[idx:idx+3]   = cmd_vel × scale                 # 手柄速度指令
idx += 3
obs[idx:idx+N]   = q[0:N] × pose_scale              # N个关节位置 (机器人坐标)
idx += N
obs[idx:idx+N]   = dq[0:N] × vel_scale              # N个关节速度
idx += N
obs[idx:idx+N]   = last_action[0:N]                  # N个上一帧动作
idx += N
obs[idx:idx+3]   = base_ang_vel × ang_vel_scale     # IMU角速度
idx += 3
obs[idx:idx+3]   = eu_ang                           # IMU欧拉角 [roll, pitch, yaw]

# 帧堆叠 × 4~6 → 模型输入 = 47 × frame_stack
```

#### 方案B: AMPPolicy (69维 for 20DOF)

```python
# 单帧: ang_vel(3) + projected_gravity(3) + cmd_vel(3) + q(N) + dq(N) + last(N)

idx = 0
obs[idx:idx+3]   = base_ang_vel × scale             # IMU角速度
idx += 3
obs[idx:idx+3]   = quat.rotate([0,0,-1])             # 重力方向 (用IMU四元数)
idx += 3
obs[idx:idx+3]   = cmd_vel × scale                  # 手柄速度指令
idx += 3
obs[idx:idx+N]   = q[0:N] × pose_scale              # 关节位置
idx += N
obs[idx:idx+N]   = dq[0:N] × vel_scale              # 关节速度
idx += N
obs[idx:idx+N]   = last_action[0:N]                  # 上一帧动作
```

### 3.3 IMU → sim2real 输入映射

```python
# imu_callback: 从 sensor_msgs/Imu 提取所需维度

def imu_callback(self, msg):
    # 1. 四元数: msg.orientation (x,y,z,w)
    self.base_quat = [msg.orientation.w,    # w
                       msg.orientation.x,    # x
                       msg.orientation.y,    # y
                       msg.orientation.z]    # z
    
    # 2. 四元数 → 欧拉角 ZYX (用于Humanoidgym策略)
    self.euler = quat_to_euler_zyx(self.base_quat)  # [roll, pitch, yaw]
    
    # 3. 四元数 → 重力投影 (用于AMP策略)
    gravity_world = [0, 0, -1]
    self.projected_gravity = quat_rotate(self.base_quat, gravity_world)
    
    # 4. 角速度: msg.angular_velocity (x,y,z)
    self.base_ang_vel = [msg.angular_velocity.x,
                          msg.angular_velocity.y,
                          msg.angular_velocity.z]
    
    # 5. 加速度: msg.linear_acceleration (x,y,z) — 仅用于坠落检测
    self.accel = [msg.linear_acceleration.x,
                   msg.linear_acceleration.y,
                   msg.linear_acceleration.z]
```

| IMU提供的字段 | 类型 | 维度 | 用于策略的哪个部分 |
|--------------|------|------|-------------------|
| `euler` [roll,pitch,yaw] | float[3] | 3 | Humanoidgym obs[44:47] |
| `projected_gravity` [gx,gy,gz] | float[3] | 3 | AMP obs[3:6] |
| `base_ang_vel` [wx,wy,wz] | float[3] | 3 | 两种策略 obs[0:3] 或 [41:44] |
| `accel` [ax,ay,az] | float[3] | 3 | 坠落检测 (|a| > 5.0 m/s²) |
| `base_quat` [w,x,y,z] | float[4] | 4 | 中间计算用 |

### 3.4 drive_motor → sim2real 输入映射

```python
# motor_state_callback: 将电机原始数据转为 RobotState

def motor_state_callback(self, msg):
    """
    电机坐标系 → 机器人坐标系变换:
    
    robot_q[i] = motor_q[i] × direction[i] - urdf_offset[i]
    robot_dq[i] = motor_dq[i] × direction[i]
    robot_tau[i] = motor_tau[i]
    """
    for i in range(self.num_joints):
        motor_q  = msg.position[i]     # 电机当前编码器值
        motor_dq = msg.velocity[i]
        
        # 应用方向符号 (direction = ±1)
        self.robot_q[i]  = motor_q  * self.direction[i]
        self.robot_dq[i] = motor_dq * self.direction[i]
        
        # 减去零点偏移 (策略默认姿态=0)
        self.robot_q[i] -= self.urdf_offset[i]  # ← 关键!

# 变换后得到 robot_q[N], robot_dq[N] → 填入观测向量
```

| drive_motor 提供 | 类型 | 维度 | 经过变换后 → | 用于策略的哪个部分 |
|-----------------|------|------|-------------|-------------------|
| `position[N]` | float[N] | N | `robot_q[N] = motor_q×dir - offset` | 观测关节位置段 |
| `velocity[N]` | float[N] | N | `robot_dq[N] = motor_dq×dir` | 观测关节速度段 |
| `torque[N]` | float[N] | N | 不参与观测, 可用于日志 | — |

### 3.5 动作输出 → drive_motor

```python
# RL推理输出 → /motor_cmd 完整链路

# Step 1: ONNX推理 → action[N] (N个float)
action = onnx_inference(obs_input)  # shape: [1, N]
action = np.clip(action, clip_lower, clip_upper)

# Step 2: action → robot_target_q (乘以action_scale模拟真机阻尼)
for i in range(N):
    robot_target_q[i] = action[i] * self.action_scale

# Step 3: Robot → Motor 逆变换
# motor_target = (robot_target + urdf_offset) × direction
for i in range(N):
    motor_target_q[i]  = (robot_target_q[i] + self.urdf_offset[i]) * self.direction[i]
    motor_target_dq[i] = 0.0        # 速度前馈(通常为0)
    motor_target_tau[i] = 0.0       # 力矩前馈(通常为0)
    motor_target_kp[i] = self.p_kp[i]   # 策略融合PD增益
    motor_target_kd[i] = self.p_kd[i]

# Step 4: 发布 /motor_cmd
pub_motor_cmd(motor_target_q, motor_target_dq, motor_target_tau, motor_target_kp, motor_target_kd)
```

| 输出阶段 | 值 | 坐标系 | 示例 (单个关节) |
|---------|-----|--------|----------------|
| ONNX推理 | `action[i]` | 策略空间 | `0.8` (无单位) |
| × action_scale | `robot_target_q[i]` | 机器人空间 | `0.8 × 0.25 = 0.2 rad` |
| +urdf_offset | 电机零点偏移 | — | `0.2 + 0 = 0.2` |
| × direction | `motor_target_q[i]` | 电机空间 | `0.2 × 1 = 0.2 rad` |

---

## 4. sim2real 核心代码框架

### 4.1 目录结构

```
sim2real/
├── sim2real/
│   ├── __init__.py
│   ├── sim2real_node.py        # 主节点 (三线程)
│   ├── policy.py                # 策略基类 + HumanoidgymPolicy
│   ├── inference.py             # ONNX Runtime 推理封装
│   ├── config_loader.py         # YAML配置加载
│   └── fsm.py                   # 状态机
├── config/
│   ├── pd_config.yaml           # 关节映射 + PD增益
│   └── rl_config.yaml           # 策略配置 + 观测参数
├── policy/
│   └── walk_policy.onnx         # ONNX策略模型
├── launch/
│   └── sim2real.launch.py       # ROS2 launch
├── package.xml
└── setup.py
```

### 4.2 主节点伪代码

```python
# sim2real_node.py
import rclpy
from rclpy.node import Node
from threading import Thread, Lock
import numpy as np
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist

class Sim2RealNode(Node):
    def __init__(self):
        super().__init__('sim2real')
        
        # 加载配置
        self.pd_config = load_yaml('pd_config.yaml')
        self.rl_config = load_yaml('rl_config.yaml')
        
        # 关节数量
        self.pd_dofs = self.pd_config['dofs']     # 实际电机数 (如22)
        self.rl_dofs = self.rl_config['dofs']     # 策略控制数 (如20)
        
        # 关节映射参数
        self.map_index = self.pd_config['map_index']
        self.direction = np.array(self.pd_config['direction'])
        self.urdf_offset = np.array(self.pd_config['urdf_offset'])
        self.p_kp = np.array(self.pd_config['p_kp'])      # 策略PD增益
        self.p_kd = np.array(self.pd_config['p_kd'])
        
        # 状态
        self.robot_q  = np.zeros(self.pd_dofs)
        self.robot_dq = np.zeros(self.pd_dofs)
        self.base_ang_vel = np.zeros(3)
        self.euler = np.zeros(3)
        self.base_quat = np.array([1.0, 0, 0, 0])  # w,x,y,z
        self.last_action = np.zeros(self.rl_dofs)
        self.cmd_vel = np.zeros(3)  # vx, vy, dyaw
        
        # 共享变量 (RL线程写入, PD线程读取)
        self.motor_cmd_lock = Lock()
        self.motor_cmd = None
        
        # ---------- 订阅 ----------
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10)
        self.motor_state_sub = self.create_subscription(
            MotorState, '/motor_state', self.motor_state_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        
        # ---------- 发布 ----------
        self.motor_cmd_pub = self.create_publisher(
            MotorCmd, '/motor_cmd', 10)
        
        # ---------- 初始化推理引擎 ----------
        self.engine = ONNXRuntimeInference(
            model_path = self.rl_config['policy_path'],
            input_dim  = self.rl_config['num_single_obs'] * self.rl_config['frame_stack'],
            output_dim = self.rl_dofs
        )
        
        # ---------- 启动线程 ----------
        self.rl_thread = Thread(target=self.exec_rl_loop, daemon=True)
        self.pd_thread = Thread(target=self.exec_pd_loop, daemon=True)
        self.rl_thread.start()
        self.pd_thread.start()
    
    # ============== 回调函数 ==============
    
    def imu_callback(self, msg):
        """IMU数据回调"""
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        self.base_quat = [qw, qx, qy, qz]
        self.euler = quat_to_euler_zyx(qw, qx, qy, qz)  # [roll, pitch, yaw]
        self.base_ang_vel = [msg.angular_velocity.x,
                              msg.angular_velocity.y,
                              msg.angular_velocity.z]
        self.accel = [msg.linear_acceleration.x,
                       msg.linear_acceleration.y,
                       msg.linear_acceleration.z]
    
    def motor_state_callback(self, msg):
        """电机状态 → 机器人坐标变换"""
        motor_q = np.array(msg.position)
        motor_dq = np.array(msg.velocity)
        
        # robot_q[i] = motor_q[i] × direction[i] - urdf_offset[i]
        self.robot_q  = motor_q  * self.direction - self.urdf_offset
        self.robot_dq = motor_dq * self.direction
    
    def cmd_vel_callback(self, msg):
        self.cmd_vel = [msg.linear.x, msg.linear.y, msg.angular.z]
    
    # ============== RL推理线程 (50Hz) ==============
    
    def exec_rl_loop(self):
        rate = self.create_rate(50)
        obs_history = []  # 帧堆叠
        
        while rclpy.ok():
            # 1. 构造单帧观测
            obs = self.build_observation()
            
            # 2. 帧堆叠
            obs_history.append(obs)
            if len(obs_history) > self.rl_config['frame_stack']:
                obs_history.pop(0)
            if len(obs_history) < self.rl_config['frame_stack']:
                rate.sleep(); continue  # 等帧数够
            
            # 3. 拼接帧 → 模型输入
            model_input = np.concatenate(obs_history)  # shape: (M × frame_stack,)
            
            # 4. ONNX推理 → 动作
            action = self.engine.infer(model_input)  # shape: (rl_dofs,)
            action = np.clip(action, 
                self.rl_config['clip_actions_lower'],
                self.rl_config['clip_actions_upper'])
            
            # 5. action → robot_target_q
            robot_target_q = action * self.rl_config['action_scale']
            
            # 6. Robot → Motor 逆变换
            motor_target_q  = (robot_target_q + self.urdf_offset[self.map_index[:self.rl_dofs]]) * self.direction[self.map_index[:self.rl_dofs]]
            motor_target_dq = np.zeros(self.rl_dofs)
            motor_target_tau = np.zeros(self.rl_dofs)
            motor_target_kp  = self.p_kp[self.map_index[:self.rl_dofs]]
            motor_target_kd  = self.p_kd[self.map_index[:self.rl_dofs]]
            
            # 7. 写入共享变量 (通知PD线程)
            with self.motor_cmd_lock:
                self.motor_cmd = (motor_target_q, motor_target_dq, motor_target_tau, motor_target_kp, motor_target_kd)
            
            # 8. 保存动作供下一帧观测
            self.last_action = action
            
            rate.sleep()
    
    def build_observation(self):
        """构造单帧观测 (HumanoidgymPolicy 风格)"""
        cfg = self.rl_config
        N = self.rl_dofs
        
        obs = np.zeros(cfg['num_single_obs'])
        
        # [0:2] 步态相位
        t = self.get_elapsed_time()
        obs[0] = np.sin(2 * np.pi * t / cfg['frequency'])
        obs[1] = np.cos(2 * np.pi * t / cfg['frequency'])
        
        # [2:5] 速度指令
        obs[2] = self.cmd_vel[0] * cfg['cmd_lin_vel_scale']
        obs[3] = self.cmd_vel[1] * cfg['cmd_lin_vel_scale']
        obs[4] = self.cmd_vel[2] * cfg['cmd_ang_vel_scale']
        
        # [5:5+N] 关节位置
        obs[5:5+N] = self.robot_q[self.map_index[:N]] * cfg['rbt_lin_pos_scale']
        
        # [5+N:5+2N] 关节速度
        obs[5+N:5+2*N] = self.robot_dq[self.map_index[:N]] * cfg['rbt_lin_vel_scale']
        
        # [5+2N:5+3N] 上一帧动作
        obs[5+2*N:5+3*N] = self.last_action
        
        # [5+3N:8+3N] 角速度
        obs[5+3*N:8+3*N] = self.base_ang_vel * cfg['rbt_ang_vel_scale']
        
        # [8+3N:11+3N] 欧拉角
        obs[8+3*N:11+3*N] = self.euler
        
        obs = np.clip(obs, -cfg['clip_obs'], cfg['clip_obs'])
        return obs
    
    # ============== PD控制线程 (500Hz) ==============
    
    def exec_pd_loop(self):
        rate = self.create_rate(500)
        while rclpy.ok():
            with self.motor_cmd_lock:
                cmd = self.motor_cmd  # ZOH: 读取RL线程输出的最新指令
            
            if cmd is not None:
                target_q, target_dq, target_tau, target_kp, target_kd = cmd
                
                msg = MotorCmd()
                msg.target_q   = target_q.tolist()
                msg.target_dq  = target_dq.tolist()
                msg.target_tau = target_tau.tolist()
                msg.target_kp  = target_kp.tolist()
                msg.target_kd  = target_kd.tolist()
                
                self.motor_cmd_pub.publish(msg)
            
            # 坠落检测
            if self.detect_fall():
                self.enter_safe_mode()
            
            rate.sleep()
    
    def detect_fall(self):
        """检查机器人是否坠落"""
        # 欧拉角阈值: pitch/roll > 45° (~0.78 rad)
        if abs(self.euler[0]) > 0.78 or abs(self.euler[1]) > 0.78:
            return True
        # 加速度阈值: > 5 m/s²
        if max(abs(np.array(self.accel))) > 5.0:
            return True
        return False
    
    def enter_safe_mode(self):
        """安全模式: 发送零力矩 (电机泄力)"""
        with self.motor_cmd_lock:
            self.motor_cmd = (
                np.zeros(self.rl_dofs),   # target_q = 0
                np.zeros(self.rl_dofs),   # target_dq = 0
                np.zeros(self.rl_dofs),   # target_tau = 0
                np.zeros(self.rl_dofs),   # kp = 0
                np.zeros(self.rl_dofs)    # kd = 0
            )
```

### 4.3 ONNX Runtime 推理封装

```python
# inference.py
import onnxruntime as ort
import numpy as np

class ONNXRuntimeInference:
    def __init__(self, model_path, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 使用CPU执行 (RDK5 无CUDA)
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        
        # 获取输入输出名称
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def infer(self, obs_array):
        """obs_array: numpy array, shape (input_dim,)"""
        inputs = {self.input_name: obs_array.reshape(1, -1).astype(np.float32)}
        outputs = self.session.run([self.output_name], inputs)
        return outputs[0].flatten()
```

### 4.4 配置文件

```yaml
# pd_config.yaml - 关节映射与硬件参数
dofs: 12                          # 实际关节总数 (电机数)

joint_names:
  - l_hip_pitch, l_hip_roll, l_thigh, l_calf, l_ankle_pitch, l_ankle_roll
  - r_hip_pitch, r_hip_roll, r_thigh, r_calf, r_ankle_pitch, r_ankle_roll

map_index: [5,4,3,2,1,0, 11,10,9,8,7,6]  # joint_names[i] → 电机CAN ID

direction: [1,1,-1,-1,1,1, -1,1,-1,1,-1,1]  # 电机旋向

urdf_offset: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 零点偏移

kp:   [110, 110, 110, 110, 210, 110, 110, 110, 110, 110, 210, 110]
kd:   [1.1, 1.4, 1.1, 1.1, 1.1, 1.1, 1.1, 1.4, 1.1, 1.1, 1.1, 1.1]
p_kp: [80,  80,  80,  80,  80,  60,  80,  80,  80,  80,  80,  60]
p_kd: [1.9, 1.3, 1.3, 1.3, 1.3, 2.3, 1.9, 1.3, 1.3, 1.3, 1.3, 2.3]

lower: [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14]
upper: [ 3.14,  3.14,  3.14,  3.14,  3.14,  3.14,  3.14,  3.14,  3.14,  3.14,  3.14,  3.14]
```

```yaml
# rl_config.yaml - 策略与观测配置
# --- 行走策略 ---
walk:
  - name: "walk_default"
    algorithm: "humanoidgym"
    policy_path: "policy/walk_policy.onnx"
    dofs: 12
    frame_stack: 5
    num_single_obs: 47          # 2 + 3 + 12 + 12 + 12 + 3 + 3
    pd_ctrl_f: 500.0
    rl_ctrl_f: 50.0
    frequency: 0.8
    
    action_scale: 0.25          # sim2real阻尼 (0.1~0.5)
    clip_obs: 18.0
    
    cmd_lin_vel_scale: 0.25
    cmd_ang_vel_scale: 0.25
    rbt_lin_pos_scale: 1.0
    rbt_lin_vel_scale: 1.0
    rbt_ang_vel_scale: 0.25
    
    clip_actions_lower: [-10, ...]   # N个
    clip_actions_upper: [ 10, ...]
    
    cmd_vel_x_min: -0.5
    cmd_vel_x_max:  0.5
    cmd_vel_y_min: -0.3
    cmd_vel_y_max:  0.3
    cmd_vel_yaw_min: -1.0
    cmd_vel_yaw_max:  1.0
```

---

## 5. 各包输入输出维度速查表

```
                    IMU 提供              电机提供
                    ────────              ────────
话题:               /imu/data             /motor_state
消息类型:           sensor_msgs/Imu       MotorState (自定义)
                    ├─ orientation x4     ├─ position[] × N
                    ├─ angular_vel x3     ├─ velocity[] × N
                    └─ linear_accel x3    └─ torque[]   × N
                           │                      │
                           ▼                      ▼
                    ┌──────────────────────────────────┐
                    │         sim2real 节点             │
                    │                                    │
                    │  IMU → 策略观测输入:               │
                    │    euler[3]        → obs[44:47]    │
                    │    或:                             │
                    │    proj_gravity[3] → obs[3:6]      │
                    │    ang_vel[3]      → obs[0:3]      │
                    │    accel[3]        → 坠落检测       │
                    │                                    │
                    │  电机 → 策略观测输入:               │
                    │    motor_q[N]  → robot_q[N]        │
                    │      robot_q ×1.0 → obs[5:5+N]     │
                    │    motor_dq[N] × direction         │
                    │      robot_dq ×1.0 → obs[5+N:5+2N] │
                    │                                    │
                    │  ONNX推理:                          │
                    │    input:  (1, M×frame_stack)      │
                    │    output: (1, N)                   │
                    │                                    │
                    │  动作 → 电机指令:                   │
                    │    action[N] × action_scale         │
                    │    → robot_target[N]                │
                    │    → motor_target[N]                │
                    │    → target_q[N] + target_kp[N]/kd[N]│
                    │                                    │
                    └──────────────────┬───────────────┘
                                       │
                                       ▼
                              /motor_cmd (MotorCmd)
                              ├─ target_q[] × N
                              ├─ target_dq[] × N
                              ├─ target_tau[] × N
                              ├─ target_kp[] × N
                              └─ target_kd[] × N
                                       │
                                       ▼
                              drive_motor → 硬件
```

---

## 6. 模型转换与推理后端

### 6.1 RDK5 推理后端选择

```
RDK5 (4核 A55, 4GB, 可能有NPU)
     │
     ├─ 无NPU → ONNX Runtime CPU (推荐)
     │             pip install onnxruntime
     │
     └─ 有NPU (如RK3588s同款) → RKNN
              pip install rknn-toolkit-lite2
```

### 6.2 模型转换

```bash
# 训练产出 .pt (TorchScript) → ONNX
python3 pt2onnx.py

# ONNX → RKNN (如果有NPU)
python3 onnx2rknn.py
```

### 6.3 ONNX Runtime 推理代码

```python
import onnxruntime as ort
import numpy as np

class ONNXRuntimeInference:
    def __init__(self, model_path, input_dim, output_dim):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # RDK5无GPU
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def infer(self, obs: np.ndarray) -> np.ndarray:
        """obs shape: (input_dim,) → action shape: (output_dim,)"""
        input_dict = {self.input_name: obs.reshape(1, -1).astype(np.float32)}
        outputs = self.session.run([self.output_name], input_dict)
        return outputs[0].flatten()
```

---

## 7. 实施顺序建议

```
Week 1: LPMS-SI2_imu
  ├─ 实现串口通信 (LPMS-SI2协议解析)
  ├─ 发布 /imu/data (sensor_msgs/Imu)
  └─ 用 rqt_plot 验证数据正确性

Week 2: drive_motor
  ├─ 实现CAN/RS485通信 (电机协议)
  ├─ 发布 /motor_state (MotorState)
  ├─ 订阅 /motor_cmd (MotorCmd)
  └─ 用 rqt 验证电机响应

Week 3: sim2real (推理部分)
  ├─ ONNX Runtime 推理封装
  ├─ IMU回调 → 姿态解算
  ├─ MotorState回调 → 坐标变换
  ├─ 观测构造 + 帧堆叠
  ├─ 动作输出 + 逆变换
  └─ 单线程测试推理正确性

Week 4: sim2real (控制部分)
  ├─ 双线程架构 (RL 50Hz + PD 500Hz)
  ├─ /motor_cmd 发布
  ├─ 坠落检测 + 安全模式
  └─ 手柄控制接口

Week 5: 联调 + 实测
  ├─ 方向校准 (每个电机 direction)
  ├─ 零点校准 (每个电机 urdf_offset)
  ├─ action_scale 调参
  └─ 行走验证
```

---

## 8. 关键配置调参

| 参数 | 起步值 | 说明 |
|------|--------|------|
| `action_scale` | 0.1 → 0.5 | 越小越稳但动作幅度小 |
| `kp` (腿) | 80 → 110 | 越大越硬但可能振荡 |
| `kd` (腿) | 1.0 → 2.0 | 阻尼，抑制振荡 |
| `direction` | ±1 | 必须逐个电机实测 |
| `urdf_offset` | 实测 | 机器人默认站立姿态的电机编码器值 |
| `clip_obs` | 18.0 | 观测截断，防止异常值 |
| `frame_stack` | 4~6 | 堆叠越多越平稳但延迟越大 |

**实测 direction**: 对单个电机发送 `target_q = 0.2`，观察关节旋转方向。若仿真中关节位置增大则 `direction = 1`，减小则 `direction = -1`。

**实测 urdf_offset**: 将机器人摆到训练默认姿态(站立)，记录每个电机当前编码器值即为 `urdf_offset`。
