#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robot Driver Package Test Script
机器人驱动包测试脚本
"""

import rospy
import sys
import time
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from controllers.humanoid_controller import HumanoidDriver


if __name__ == "__main__":
    driver = HumanoidDriver()
    # 设定机器人的控制频率为params.yaml中的control_frequency
    control_freq = rospy.get_param('~control_frequency', 10.0)  # 从参数服务器获取控制频率
    rate = rospy.Rate(control_freq)  # 设置ROS节点运行频率
    print(f"控制频率设置为: {control_freq} Hz")
    
    while not rospy.is_shutdown():
        driver.update()
        rate.sleep()
    rospy.spin()
