# 机器人配置系统

## 概述

这个配置系统替代了原本在 `livelybot_description` 包中的机器人参数配置功能。所有的机器人硬件配置现在都在 `sim2real` 包中管理。

## 配置文件

### robot_params_12dof.yaml
12自由度机器人的专用配置文件。

## 配置参数说明

### 基础参数
- `SDK_version`: SDK版本号
- `robot_name`: 机器人名称
- `arm_dof`: 手臂自由度数量
- `leg_dof`: 腿部自由度数量
- `Serial_Type`: 串口设备类型
- `Seial_baudrate`: 串口波特率
- `CAN_Type`: CAN总线类型
- `control_type`: 控制模式
- `imu_limt_flag`: IMU角度限制开关
- `imu_dir`: IMU安装方向
- `imu_limt_num`: IMU角度限制值

### 硬件配置
- `CANboard_num`: CAN板数量
- `CANboard`: CAN板详细配置
  - `CANport_num`: 每个CAN板的端口数量
  - `CANport`: 每个端口的配置
    - `serial_id`: 串口ID
    - `motor_num`: 电机数量
    - `motor`: 每个电机的配置
      - `type`: 电机类型
      - `id`: 电机ID
      - `name`: 电机名称
      - `num`: 电机编号
      - `pos_limit_enable`: 位置限制开关
      - `pos_upper`: 位置上限
      - `pos_lower`: 位置下限
      - `tor_limit_enable`: 扭矩限制开关
      - `tor_upper`: 扭矩上限
      - `tor_lower`: 扭矩下限

## 注意事项

- 确保电机ID在同一个CAN端口内不重复
- 位置和扭矩限制参数要根据实际硬件设置
- 电机类型要与实际使用的电机型号匹配
- 串口设备路径要根据实际系统设置 

