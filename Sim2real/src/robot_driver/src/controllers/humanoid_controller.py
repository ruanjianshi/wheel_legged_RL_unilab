#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人形机器人驱动控制器 (Humanoid Robot Driver Controller)

本模块是 sim2real 项目的核心控制器，负责处理人形机器人的运动控制。

主要功能：
1. 遥控手柄数据接收与解析
2. IMU传感器数据采集与姿态解算
3. 速度指令一阶惯性滤波（平滑处理）
4. 基于IMU的PD姿态补偿（保持机器人平衡）
5. 自旋自动前进功能（旋转时自动前进）
6. 处理后的Joy消息重发布

核心工作流程：
1. 订阅 /joy_input 话题接收遥控手柄指令
2. 订阅 IMU 话题获取机器人姿态数据
3. 使用一阶惯性滤波平滑速度指令
4. 根据IMU姿态数据（roll/pitch）进行PD补偿
5. 实现自旋自动前进逻辑
6. 将处理后的速度指令封装为Joy消息发布

支持的配置参数：
- ~alpha: 一阶惯性滤波系数（默认0.8）
- ~alpha_imu: IMU补偿混合系数（默认0.5）
- ~auto_forward/enable: 是否启用自旋自动前进（默认True）
- ~auto_forward/speed: 自动前进速度（默认0.18）
- ~auto_forward/rotation_threshold: 旋转阈值（默认0.05）
- ~pd_x/kp: X方向PD比例系数（对应pitch角度，默认1.0）
- ~pd_x/kd: X方向PD微分系数（默认0.05）
- ~pd_y/kp: Y方向PD比例系数（对应roll角度，默认1.0）
- ~pd_y/kd: Y方向PD微分系数（默认0.05）
- ~local_publish/enable: 是否本地发布（默认True）
- ~local_publish/topic: 本地发布话题（默认/joy）
- ~imu_topic: IMU话题名称（默认/imu/data）

Joy轴映射（Xbox手柄）：
- axes[0]: 左摇杆X → 左右速度
- axes[1]: 左摇杆Y → 前后速度
- axes[3]: 右摇杆X → 旋转速度

话题接口：
- 订阅: /joy_input (Joy) - 原始遥控手柄输入
- 订阅: /imu/data (Imu) - IMU姿态数据
- 发布: /joy (Joy) - 处理后的控制指令
- 发布: /imu_compensation (Vector3) - IMU补偿向量
"""

import numpy as np
import rospy
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Joy, Imu
import threading
import tf.transformations


class HumanoidDriver:
    """
    人形机器人驱动器类
    """
    
    def __init__(self):
        """
        初始化人形机器人驱动器
        """
        # 关键变量：输入速度和输出速度
        self.v_input = np.zeros(3)   # [x, y, rotation] 输入速度向量
        self.v_output = np.zeros(3)  # [x, y, rotation] 输出速度向量
        self.v_imu_compensation = np.zeros(3)  # [x, y, rotation] IMU补偿速度向量
        
        # IMU数据变量
        self.imu_angles = np.zeros(3)  # [roll, pitch, yaw] IMU姿态角度数据
        self.imu_data_lock = threading.Lock()
        
        # 保存原始Joy消息
        self.original_joy_msg = None
        self.joy_msg_lock = threading.Lock()
        
        # 初始化ROS节点
        rospy.init_node('humanoid_driver', anonymous=True)
        
        # 从参数服务器读取配置参数
        self.alpha = rospy.get_param('~alpha', 0.8)  # 一阶惯性滤波系数，默认值0.8
        
        # IMU补偿混合参数
        self.alpha_imu = rospy.get_param('~alpha_imu', 0.5)  # IMU补偿混合系数
        
        # 自旋自动前进参数
        self.auto_forward_enable = rospy.get_param('~auto_forward/enable', True)  # 是否启用自旋自动前进
        self.auto_forward_speed = rospy.get_param('~auto_forward/speed', 0.18)  # 自动前进速度
        self.auto_forward_rotation_threshold = rospy.get_param('~auto_forward/rotation_threshold', 0.05)  # 旋转阈值
        
        # PD控制器参数
        self.pd_x_kp = rospy.get_param('~pd_x/kp', 1.0)  # X方向比例系数 (对应pitch角度)
        self.pd_x_kd = rospy.get_param('~pd_x/kd', 0.05)  # X方向微分系数 (对应pitch角度)
        self.pd_y_kp = rospy.get_param('~pd_y/kp', 1.0)  # Y方向比例系数 (对应roll角度)
        self.pd_y_kd = rospy.get_param('~pd_y/kd', 0.05)  # Y方向微分系数 (对应roll角度)
        
        # PD控制器状态变量
        self.pitch_last_error = 0.0  # pitch角度上次误差
        self.roll_last_error = 0.0   # roll角度上次误差
        
        # 本地发布配置
        self.local_enable = rospy.get_param('~local_publish/enable', True)
        self.local_topic = rospy.get_param('~local_publish/topic', '/joy')
        
        # IMU话题配置
        self.imu_topic = rospy.get_param('~imu_topic', '/imu/data')
        
        # 订阅遥控手柄话题
        self.joy_subscriber = rospy.Subscriber(
            '/joy_input', 
            Joy, 
            self.joy_callback, 
            queue_size=1
        )
        
        # 订阅IMU话题
        self.imu_subscriber = rospy.Subscriber(
            self.imu_topic,
            Imu,
            self.imu_callback,
            queue_size=1
        )
        
        # 初始化本地发布者
        self.local_publisher = None
        if self.local_enable:
            self.local_publisher = rospy.Publisher(
                self.local_topic, 
                Joy, 
                queue_size=1
            )
        
        # IMU补偿向量发布者
        self.imu_compensation_publisher = rospy.Publisher(
            '/imu_compensation',
            Vector3,
            queue_size=1
        )
    
    def joy_callback(self, msg):
        """
        遥控手柄数据回调函数
        
        Args:
            msg (Joy): 来自遥控手柄的joy消息
        """
        with self.joy_msg_lock:
            # 保存完整的原始Joy消息
            self.original_joy_msg = msg
        
        # 从Joy消息中提取速度数据（根据手柄轴的映射）
        # Xbox手柄：轴0=左摇杆X（左右）, 轴1=左摇杆Y（前后）, 轴3=右摇杆X（旋转）
        # 注意：轴2是LT扳机，轴5是RT扳机，不要覆盖它们！
        if len(msg.axes) >= 4:
            self.v_input[0] = msg.axes[1]   # 前后速度
            self.v_input[1] = msg.axes[0]   # 左右速度  
            self.v_input[2] = msg.axes[3]   # 旋转速度
    
    def imu_callback(self, msg):
        """
        IMU数据回调函数
        
        Args:
            msg (Imu): 来自IMU的数据消息
        """
        # 从四元数中提取欧拉角
        orientation_q = [
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ]
        
        try:
            # 转换四元数到欧拉角 (roll, pitch, yaw)
            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_q)
            current_angles = np.array([roll, pitch, yaw])
        except:
            # 如果四元数转换失败，使用零值
            current_angles = np.zeros(3)
        
        with self.imu_data_lock:
            # 保存IMU角度数据
            self.imu_angles = current_angles.copy()

    def imu_compensation(self):
        """
        IMU补偿函数
        基于pitch和roll角度进行PD补偿
        """
        # 获取当前IMU角度数据
        with self.imu_data_lock:
            current_angles = self.imu_angles.copy()
        
        # 提取roll和pitch角度
        roll = current_angles[0]   # 绕X轴旋转角度
        pitch = current_angles[1]  # 绕Y轴旋转角度
        
        # PD控制器补偿计算
        # X方向补偿使用pitch角度
        self.v_imu_compensation[0] = pitch * self.pd_x_kp + (pitch - self.pitch_last_error) * self.pd_x_kd
        
        # Y方向补偿使用roll角度  
        self.v_imu_compensation[1] = roll * self.pd_y_kp + (roll - self.roll_last_error) * self.pd_y_kd
        
        # Z方向暂不补偿
        self.v_imu_compensation[2] = 0.0
        # if self.v_imu_compensation[0] > 1.0:
        #     self.v_imu_compensation[0] = 1.0
        # if self.v_imu_compensation[0] < -1.0:
        #     self.v_imu_compensation[0] = -1.0
        if self.v_imu_compensation[0] > 0.3:
            self.v_imu_compensation[0] = 0.3
        if self.v_imu_compensation[0] < -0.3:
            self.v_imu_compensation[0] = -0.3
        if self.v_imu_compensation[0] > -0.05 and self.v_imu_compensation[0] < 0.05:
            self.v_imu_compensation[0] = 0.0
        # if self.v_imu_compensation[1] > 1.0:
        #     self.v_imu_compensation[1] = 1.0
        # if self.v_imu_compensation[1] < -1.0:
        #     self.v_imu_compensation[1] = -1.0
        if self.v_imu_compensation[1] > 0.3:
            self.v_imu_compensation[1] = 0.3
        if self.v_imu_compensation[1] < -0.3:
            self.v_imu_compensation[1] = -0.3
        if self.v_imu_compensation[1] > -0.05 and self.v_imu_compensation[1] < 0.05:
            self.v_imu_compensation[1] = 0.0
        # 更新上次误差
        self.pitch_last_error = pitch
        self.roll_last_error = roll
        
        # 发布IMU补偿向量
        compensation_msg = Vector3()
        compensation_msg.x = self.v_imu_compensation[0]
        compensation_msg.y = self.v_imu_compensation[1]
        compensation_msg.z = self.v_imu_compensation[2]
        
        self.imu_compensation_publisher.publish(compensation_msg)
        
        # 只打印x方向的速度补偿
        # print(f"X方向速度补偿: {self.v_imu_compensation[0]:.4f}")
        
    def update(self):
        """
        更新机器人状态
        
        使用一阶惯性滤波平滑输入速度得到输出速度
        """
        # 对输入速度进行一阶惯性滤波
        self.imu_compensation()
        self.v_output = self.alpha * self.v_input + (1 - self.alpha) * self.v_output
        # IMU补偿叠加（而不是混合，避免稀释输出）
        # self.v_output = self.v_output + self.alpha_imu * self.v_imu_compensation

        # 自旋自动前进：如果只有旋转控制量，自动添加前进速度
        if self.auto_forward_enable:
            has_rotation = abs(self.v_output[2]) > self.auto_forward_rotation_threshold
            has_translation = abs(self.v_output[0]) > 0.01 or abs(self.v_output[1]) > 0.01
            if has_rotation and not has_translation:
                self.v_output[0] = self.auto_forward_speed

        # 获取原始Joy消息的副本
        with self.joy_msg_lock:
            if self.original_joy_msg is None:
                return  # 还没有接收到Joy消息
            original_msg = self.original_joy_msg

        # 构造Joy消息
        joy_msg = Joy()
        joy_msg.header.stamp = rospy.Time.now()
        joy_msg.header.frame_id = original_msg.header.frame_id  # 保持原始frame_id
        
        # 复制原始轴数据
        joy_msg.axes = list(original_msg.axes)  # 创建副本
        
        # 只对速度相关的轴进行滤波替换（如果轴数量足够）
        if len(joy_msg.axes) >= 4:
            joy_msg.axes[0] = float(self.v_output[1])   # 轴0: 左右速度（滤波后）
            joy_msg.axes[1] = float(self.v_output[0])   # 轴1: 前后速度（滤波后）
            joy_msg.axes[3] = float(self.v_output[2])   # 轴2: 旋转速度（滤波后）
            # 其他轴保持原始值
        
        # 复制原始按钮数据（保持不变）
        joy_msg.buttons = list(original_msg.buttons)
        
        # 发布到本地
        if self.local_enable and self.local_publisher:
            self.local_publisher.publish(joy_msg)

    def run(self):
        """
        运行主循环
        """
        rate = rospy.Rate(50)  # 50Hz
        while not rospy.is_shutdown():
            self.update()
            rate.sleep()

    def spin(self):
        """
        保持ROS节点运行
        """
        rospy.spin()


if __name__ == '__main__':
    try:
        driver = HumanoidDriver()
        driver.run()  # 使用主循环而不是spin
    except rospy.ROSInterruptException:
        pass