# 电机零点标定与 ID 设置指导

> 适用平台: 人形机器人 (CAN 总线电机)  
> 配套文档: SIM2REAL_ROS2_GUIDE.md

---

## 1. 硬件拓扑与 ID 命名

### 1.1 CAN 总线物理拓扑

```
上位机 (RDK5 USB)
    │
    ▼
USB转CAN 适配器
    │
    ▼
CAN 总线 (CAN-FD / CAN 2.0)
    │
    ├── ID=1  → 左腿髋俯仰 (l_hip_pitch)
    ├── ID=2  → 左腿髋横滚 (l_hip_roll)
    ├── ID=3  → 左腿大腿   (l_thigh)
    ├── ID=4  → 左腿小腿   (l_calf)
    ├── ID=5  → 左腿脚踝俯仰(l_ankle_pitch)
    ├── ID=6  → 左腿脚踝横滚(l_ankle_roll)
    ├── ID=7  → 右腿髋俯仰 (r_hip_pitch)
    ├── ID=8  → 右腿髋横滚 (r_hip_roll)
    ├── ID=9  → 右腿大腿   (r_thigh)
    ├── ID=10 → 右腿小腿   (r_calf)
    ├── ID=11 → 右腿脚踝俯仰(r_ankle_pitch)
    └── ID=12 → 右腿脚踝横滚(r_ankle_roll)
```

> 多路 CAN 板: 每个 CAN 口独立编址，ID 从 1 开始。不同 CAN 口之间 ID 可重复。

### 1.2 命名约定

```
关节命名规则: {左/右}_{关节名}
  示例: l_hip_pitch   = 左腿髋俯仰
        r_ankle_roll  = 右腿脚踝横滚

电机 ID: 从 1 开始递增，与 map_index 配合使用

map_index: 将 joint_names[i] 映射到电机 ID = map_index[i]
  例: map_index[0]=5  →  joint_names[0] ("l_hip_pitch") 对应 CAN ID=5 的电机
```

---

## 2. 电机 ID 设置

### 2.1 逐个电机设置 ID

**前置条件**: 每次只给一个电机上电，避免 ID 冲突。

```
步骤:
  1. 只给第 1 个电机上电（其余断电/断开CAN）
  2. 使用电机调试工具或串口指令设置 ID
  3. 断电，换下一个电机
  4. 重复直到所有电机完成
```

#### 方式A: 电机调试工具 (图形界面)

```
连接上位机 → 打开电机调试软件 → 选择串口 → 连接
→ 读取当前ID → 修改为新ID → 写入 → 断电重启
```

#### 方式B: CAN 指令 (代码方式)

```python
# set_motor_id.py — 逐个设置电机ID
import serial
import time

# 协议示例 (根据实际电机协议修改)
def set_motor_id(ser, current_id, new_id):
    """设置电机ID: 发送到 current_id, 写新ID到寄存器"""
    # 命令格式: [帧头] [当前ID] [指令码] [新ID] [CRC]
    cmd = build_set_id_command(current_id, new_id)
    ser.write(cmd)
    time.sleep(0.1)
    
    # 验证
    response = read_motor_id(ser, new_id)
    assert response == new_id, f"设置失败: 期望 {new_id}, 读到 {response}"
    print(f"电机 {current_id} → {new_id} ✓")

# 依次设置 12 个电机
ser = serial.Serial('/dev/motor', 4000000, timeout=0.5)

motor_id_map = {
    'l_hip_pitch':  1,   # 左腿髋俯仰
    'l_hip_roll':   2,   # 左腿髋横滚
    'l_thigh':      3,   # 左腿大腿
    'l_calf':       4,   # 左腿小腿
    'l_ankle_pitch':5,   # 左腿脚踝俯仰
    'l_ankle_roll': 6,   # 左腿脚踝横滚
    'r_hip_pitch':  7,   # 右腿髋俯仰
    'r_hip_roll':   8,   # 右腿髋横滚
    'r_thigh':      9,   # 右腿大腿
    'r_calf':       10,  # 右腿小腿
    'r_ankle_pitch':11,  # 右腿脚踝俯仰
    'r_ankle_roll': 12,  # 右腿脚踝横滚
}

for joint_name, target_id in motor_id_map.items():
    set_motor_id(ser, current_id=1, new_id=target_id)  # 新电机默认ID通常为1
```

### 2.2 电机 ID 分配规则

```
规则1: 同一条 CAN 总线上 ID 必须唯一
规则2: ID 从 1 开始连续编号
规则3: 分组建议:
        左腿: ID 1-6
        右腿: ID 7-12
        手臂: ID 13-20 (如有)
        头部: ID 21-22 (如有)

规则4: 多个 CAN 口时:
        CAN1 → 左腿 (ID 1-6)  + 右腿 (ID 7-12)
        CAN2 → 左臂 (ID 1-4)  + 右臂 (ID 5-8)
        不同 CAN 口之间 ID 可重复
```

### 2.3 完成确认

```bash
# 全部上电后，用调试工具 ping 扫描
for id in range(1, 13):
    response = ping_motor_id(id)
    if response:
        print(f"ID={id} ✓ 在线")
    else:
        print(f"ID={id} ✗ 离线 - 检查接线!")

# 全部 12 个电机都应在线
```

---

## 3. 零点标定 (urdf_offset)

### 3.1 什么是零点偏移

```
电机编码器零点: 电机出厂时定义的机械零点
机器人关节零点: RL 策略训练时定义的默认站立姿态 (q=0)

两者之间存在偏移:
  robot_q = motor_q × direction - urdf_offset
          
其中:
  motor_q    = 电机编码器当前值 (硬件读出)
  robot_q    = 机器人关节位置 (策略使用的值)
  direction  = ±1 (旋转方向, 见第4节)
  urdf_offset= 零点偏移 (本节标定)
```

**目标**: 当机器人处于 RL 训练的默认站立姿态时，策略观测到的 `robot_q = 0`。

### 3.2 标定步骤

```
步骤1: 准备工作
  ├─ 确保电机 ID 已正确设置 (第2节)
  ├─ 搭建标定支架 (让机器人保持站立姿态)
  └─ 准备 spread-sheet 记录软件 (Excel/Google Sheets)

步骤2: 定义默认站立姿态
  在仿真中确定训练使用的默认关节角度。例如:
  l_hip_pitch  = 0.0         r_hip_pitch  = 0.0
  l_hip_roll   = 0.0         r_hip_roll   = 0.0
  l_thigh      = -0.25       r_thigh      = -0.25
  l_calf       = 0.65        r_calf       = 0.65
  l_ankle_pitch = -0.40      r_ankle_pitch = -0.40
  l_ankle_roll  = 0.0        r_ankle_roll  = 0.0

步骤3: 物理摆位
  ├─ 手动将每个关节摆到上述角度
  ├─ 使用水平尺/量角器辅助校准
  ├─ 用支架固定身体保持直立
  └─ 双脚底与地面平行

步骤4: 上电读取编码器值
  └─ 运行标定脚本 → 记录每个电机的当前位置

步骤5: 计算 urdf_offset
  ├─ urdf_offset[i] = motor_q[i] × direction[i]   (因为 robot_q=0)
  └─ 填入 pd_config.yaml

步骤6: 验证
  └─ 设置 offset 后重新上电 → 读取 robot_q → 应全为 0
```

### 3.3 标定脚本

```python
# calibrate_offset.py — 记录默认姿态下的电机位置
import yaml
import numpy as np
from drive_motor import DriveMotor  # 你的电机驱动类

# 1. 定义默认站立姿态 (从仿真训练配置获取)
DEFAULT_POSE = {
    'l_hip_pitch':   0.0,
    'l_hip_roll':    0.0,
    'l_thigh':      -0.25,
    'l_calf':        0.65,
    'l_ankle_pitch': -0.40,
    'l_ankle_roll':  0.0,
    'r_hip_pitch':   0.0,
    'r_hip_roll':    0.0,
    'r_thigh':      -0.25,
    'r_calf':        0.65,
    'r_ankle_pitch': -0.40,
    'r_ankle_roll':  0.0,
}

# 2. 读取 PD 配置中的 joint_names、direction
with open('pd_config.yaml') as f:
    pd = yaml.safe_load(f)

joint_names = pd['joint_names']
direction   = np.array(pd['direction'])

print("=" * 70)
print("零点标定: 请将机器人摆成默认站立姿态后按 Enter")
print("=" * 70)
input()

# 3. 连接电机
dm = DriveMotor(port='/dev/motor', motor_ids=list(range(1, len(joint_names)+1)))
motor_q_raw = dm.read_all_positions()  # 返回 len(joint_names) 个编码器值

# 4. 计算 urdf_offset
# robot_q = motor_q × direction - urdf_offset
# 让 robot_q = 0 (默认姿态) → urdf_offset = motor_q × direction
urdf_offset = motor_q_raw * direction

# 5. 输出结果
print("\n" + "=" * 70)
print("标定结果 - 请复制到 pd_config.yaml 的 urdf_offset 字段")
print("=" * 70)

for i, name in enumerate(joint_names):
    print(f"  {name:20s}: motor_raw={motor_q_raw[i]:8.4f}  →  urdf_offset={urdf_offset[i]:8.4f}")

print("\n# YAML 格式:")
print("urdf_offset: [" + ", ".join(f"{v:.4f}" for v in urdf_offset) + "]")

# 6. 验证: 读取 robot_q (应接近0)
robot_q = motor_q_raw * direction - urdf_offset
print("\n" + "=" * 70)
print("验证 robot_q (应全为 0):")
print("=" * 70)
for i, name in enumerate(joint_names):
    status = "✓" if abs(robot_q[i]) < 0.05 else "✗ 偏差较大"
    print(f"  {name:20s}: robot_q={robot_q[i]:8.4f}  {status}")
```

### 3.4 精细化校准

如果某些关节偏差 > 0.05 rad (约 3°)，手动微调后重复:

```python
# 微调单个关节
urdf_offset[3] += 0.02   # l_calf 偏了, 加 0.02 rad
# 重新验证
robot_q = motor_q_raw * direction - urdf_offset
# 直到所有 robot_q < 0.05
```

---

## 4. 电机旋转方向标定 (direction)

### 4.1 什么是方向符号

```
direction = +1: 电机正转 → 关节位置增大 (RL 策略的正方向)
direction = -1: 电机正转 → 关节位置减小 (需要翻转)
```

### 4.2 逐个电机标定

```python
# calibrate_direction.py — 逐个电机标定方向
import time
import numpy as np

def calibrate_direction(dm, motor_id, joint_name):
    """
    发送小角度正转指令，观察编码器变化方向。
    如果编码器值增大 → direction = +1
    如果编码器值减小 → direction = -1
    """
    TEST_ANGLE = 0.3  # 测试角度 (rad), 约 17°
    
    # 读取当前位置
    pos_before = dm.read_motor_position(motor_id)
    
    # 发送正转指令 (target_q = pos_before + TEST_ANGLE)
    dm.send_position_cmd(motor_id, pos_before + TEST_ANGLE, kp=20, kd=0.5)
    time.sleep(0.5)
    
    # 读取新位置
    pos_after = dm.read_motor_position(motor_id)
    
    # 判断方向
    delta = pos_after - pos_before
    if abs(delta) < 0.01:
        print(f"  ⚠ {joint_name}: 电机未转动, 检查连接或指令格式")
        return None
    
    if delta > 0:
        direction = +1
        print(f"  ┃ {joint_name}: delta=+{delta:.4f} → direction = +1")
    else:
        direction = -1
        print(f"  ┃ {joint_name}: delta={delta:.4f} → direction = -1")
    
    # 恢复原位
    dm.send_position_cmd(motor_id, pos_before, kp=20, kd=0.5)
    time.sleep(0.3)
    
    return direction

# 主流程
print("=" * 70)
print("方向标定: 电机将逐个小幅度转动, 观察运动方向")
print("=" * 70)

directions = []
for motor_id in range(1, len(joint_names)+1):
    dir_sign = calibrate_direction(dm, motor_id, joint_names[motor_id-1])
    if dir_sign is None:
        print(f"\n⚠ 电机 {motor_id} 异常, 修复后重新运行")
        break
    directions.append(dir_sign)

print("\n# YAML 格式:")
print("direction: [" + ", ".join(str(d) for d in directions) + "]")
```

### 4.3 人工快速判断法

```
规则: 人站在机器人后方观察

正方向 (+1): 关节往训练动作的正方向走时, 电机编码器值增大

关节         +1 方向 (编码器增大)
──────────────────────────────────
l_hip_pitch   髋往前踢 (俯仰正)
l_hip_roll    腿往外摆 (外展)
l_thigh       大腿往前弯
l_calf        小腿往后弯 (膝弯)
l_ankle_pitch 脚掌往下 (跖屈)
l_ankle_roll  脚掌往外翻 (外翻)
r_hip_pitch   髋往前踢
r_hip_roll    腿往外摆
r_thigh       大腿往前弯
r_calf        小腿往后弯
r_ankle_pitch 脚掌往下
r_ankle_roll  脚掌往外翻
```

---

## 5. 将标定结果写入配置文件

### 5.1 填入 pd_config.yaml

```yaml
# pd_config.yaml
dofs: 12

joint_names:
  - l_hip_pitch, l_hip_roll, l_thigh, l_calf, l_ankle_pitch, l_ankle_roll
  - r_hip_pitch, r_hip_roll, r_thigh, r_calf, r_ankle_pitch, r_ankle_roll

# ID 映射: joint_names[0] → CAN ID 5
map_index: [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6]

# ─── 以下为标定结果 ───

# 方向标定 (±1)
direction: [1, 1, -1, -1, 1, 1, 1, -1, 1, -1, 1, -1]

# 零点偏移 (默认站立姿态的电机编码器值)
urdf_offset: [0.52, -0.03, 1.85, 2.30, -0.15, 0.01,
              0.48, 0.02, 1.90, 2.28, -0.12, -0.01]

# PD 增益
kp: [80, 80, 80, 80, 60, 60, 80, 80, 80, 80, 60, 60]
kd: [1.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.0, 1.0]

# 策略融合 PD 增益 (行走时用)
p_kp: [60, 60, 60, 60, 40, 40, 60, 60, 60, 60, 40, 40]
p_kd: [1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8]

# 关节限位
lower: [-1.57, -0.79, -2.09, -0.87, -0.87, -0.52,
        -1.57, -0.79, -2.09, -0.87, -0.87, -0.52]
upper: [ 1.57,  0.79,  1.57,  2.09,  0.87,  0.52,
         1.57,  0.79,  1.57,  2.09,  0.87,  0.52]
```

### 5.2 填入 robot_param.yaml (电机拓扑)

```yaml
# robot_param.yaml - 驱动电机包用的硬件拓扑
SDK_version: 1
serial:
  port: "/dev/motor"
  baudrate: 4000000

can_bus:
  - can_id: 1                          # CAN口1
    motors:
      - name: "l_ankle_roll"           # 电机ID=1
        id: 1
        model: "your_motor_model"
      - name: "l_ankle_pitch"         # 电机ID=2
        id: 2
        model: "your_motor_model"
      - name: "l_calf"                # 电机ID=3
        id: 3
        model: "your_motor_model"
      - name: "l_thigh"               # 电机ID=4
        id: 4
        model: "your_motor_model"
      - name: "l_hip_roll"           # 电机ID=5
        id: 5
        model: "your_motor_model"
      - name: "l_hip_pitch"          # 电机ID=6
        id: 6
        model: "your_motor_model"
      - name: "r_ankle_roll"         # 电机ID=7
        id: 7
        model: "your_motor_model"
      - name: "r_ankle_pitch"        # 电机ID=8
        id: 8
        model: "your_motor_model"
      - name: "r_calf"               # 电机ID=9
        id: 9
        model: "your_motor_model"
      - name: "r_thigh"              # 电机ID=10
        id: 10
        model: "your_motor_model"
      - name: "r_hip_roll"           # 电机ID=11
        id: 11
        model: "your_motor_model"
      - name: "r_hip_pitch"          # 电机ID=12
        id: 12
        model: "your_motor_model"
```

---

## 6. 验证清单

```bash
□ 电机 ID 全部唯一且连续 (CAN 扫描通过)
□ direction 逐个电机测试 (小角度正转, 编码器增大=+1)
□ urdf_offset 标定 (摆默认姿态, 读取编码器, 填入配置)
□ robot_q 验证 (标定后读取 robot_q, 全部 < 0.05 rad)
□ PD 增益初始值 (kp=40~80, kd=0.5~1.5, 从低起步)
□ 关节限位设置 (机械限位×0.8 作为软件限位)
□ 全部上电后无异常发热/噪音
□ 零力矩模式测试 (kp=kd=0, 机器人松软可手推动)
□ 硬直模式测试 (kp=80, kd=1.5, 机器人在默认姿态站稳)
```

---

## 7. 故障排查

| 现象 | 可能原因 | 解决方法 |
|------|---------|---------|
| 电机不响应 | ID 冲突 / 接线错误 | CAN 扫描检查, 逐个电机 ping |
| 关节方向反了 | direction 标定错误 | 重新跑 4.2 标定脚本 |
| robot_q 不全为0 | urdf_offset 不准 | 重新摆位标定, 或手动微调 ±0.02 |
| 机器人站立不稳 | 质心不在脚底上方 | 调整默认姿态的 urdf_offset |
| 电机抖动/发热 | kp 过大 / kd 过小 | 降低 kp, 增大 kd |
| 策略输出后乱动 | action_scale 太大 / offset 有偏差 | action_scale 从 0.1 起步 |
