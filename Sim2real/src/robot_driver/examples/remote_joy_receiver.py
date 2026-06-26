#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
远端Joy消息接收器示例

本脚本是一个ROS节点，运行在远端机器人上，用于接收从本地发送的Joy控制指令，
并将其转换为Twist消息发布给机器人控制器。

功能说明：
1. 订阅Joy话题（默认：/joy），接收来自远端的游戏手柄控制消息
2. 解析Joy消息中的轴数据，提取前后、左右、旋转速度指令
3. 根据配置的最大速度限制进行缩放
4. 将速度指令转换为Twist消息并发布（默认：/cmd_vel）

消息映射：
- Joy.axes[0] → 左右移动速度
- Joy.axes[1] → 前后移动速度
- Joy.axes[2] → 旋转速度

配置参数：
- ~input_topic: 输入Joy话题名称（默认：/joy）
- ~output_topic: 输出Twist话题名称（默认：/cmd_vel）
- ~max_linear_vel: 最大线速度（默认：1.0 m/s）
- ~max_angular_vel: 最大角速度（默认：1.0 rad/s）

使用方式：
1. 在机器人端运行：rosrun robot_driver remote_joy_receiver.py
2. 通过参数覆盖配置：
   rosrun robot_driver remote_joy_receiver.py _input_topic:=/remote_joy _max_linear_vel:=0.5

配合使用：
需与 remote_joy_sender.py 配合使用，sender运行在本地发送Joy消息，
receiver运行在远端机器人接收并转换为运动指令。
"""

import rospy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist


class RemoteJoyReceiver:
    """
    远端Joy消息接收器类
    """
    
    def __init__(self):
        """
        初始化远端Joy接收器
        """
        # 初始化ROS节点
        rospy.init_node('remote_joy_receiver', anonymous=True)
        
        # 从参数服务器获取配置
        self.input_topic = rospy.get_param('~input_topic', '/joy')
        self.output_topic = rospy.get_param('~output_topic', '/cmd_vel')
        self.max_linear_vel = rospy.get_param('~max_linear_vel', 1.0)  # 最大线速度 m/s
        self.max_angular_vel = rospy.get_param('~max_angular_vel', 1.0)  # 最大角速度 rad/s
        
        # 订阅Joy话题
        self.joy_subscriber = rospy.Subscriber(
            self.input_topic,
            Joy,
            self.joy_callback,
            queue_size=1
        )
        
        # 发布cmd_vel话题
        self.cmd_vel_publisher = rospy.Publisher(
            self.output_topic,
            Twist,
            queue_size=1
        )
        
        print("远端Joy接收器已启动")
        print(f"订阅话题: {self.input_topic}")
        print(f"发布话题: {self.output_topic}")
        print(f"最大线速度: {self.max_linear_vel} m/s")
        print(f"最大角速度: {self.max_angular_vel} rad/s")
        print("等待控制指令...")
    
    def joy_callback(self, msg):
        """
        Joy消息回调函数
        
        Args:
            msg (Joy): 接收到的Joy消息
        """
        if len(msg.axes) < 3:
            rospy.logwarn("Joy消息轴数量不足，需要至少3个轴")
            return
        
        # 从Joy消息中提取速度指令
        # 按照发送端的格式：axes[0]=左右, axes[1]=前后, axes[2]=旋转
        raw_vx = msg.axes[1]  # 前后速度
        raw_vy = msg.axes[0]  # 左右速度
        raw_vrot = msg.axes[2]  # 旋转速度
        
        # 应用速度限制
        vx = raw_vx * self.max_linear_vel
        vy = raw_vy * self.max_linear_vel
        vrot = raw_vrot * self.max_angular_vel
        
        # 构造并发布Twist消息
        twist_msg = Twist()
        twist_msg.linear.x = vx
        twist_msg.linear.y = vy
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = vrot
        
        self.cmd_vel_publisher.publish(twist_msg)
        
        # 记录日志
        rospy.loginfo(f"收到远端控制指令: "
                     f"vx={vx:.3f}, vy={vy:.3f}, vrot={vrot:.3f} "
                     f"(原始: {raw_vx:.3f}, {raw_vy:.3f}, {raw_vrot:.3f})")
    
    def run(self):
        """
        运行接收器
        """
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("\n远端Joy接收器已停止")


if __name__ == '__main__':
    try:
        receiver = RemoteJoyReceiver()
        receiver.run()
    except rospy.ROSInterruptException:
        pass