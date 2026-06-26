#!/usr/bin/env python3
# coding: utf-8

import rospy
import actionlib
from sim2real_msg.msg import lowlevel_controllerAction, lowlevel_controllerGoal
from sensor_msgs.msg import JointState

class LowlevelControllerClientRobotOutput:
    def __init__(self, namespace: str, group: str, joint_names: list):
        """
        :param namespace: action 服务器的命名空间前缀
        :param group:    要控制的电机组名称
        :param joint_names: 本组所有关节的名字列表，顺序对应 position 数组
        """
        self.group = group
        self.joint_names = joint_names
        self.client = actionlib.SimpleActionClient(
            f"{namespace}/lowlevel_controller",
            lowlevel_controllerAction
        )
        rospy.loginfo(f"[{self.group}] Waiting for action server {namespace}/lowlevel_controller...")
        self.client.wait_for_server()
        rospy.loginfo(f"[{self.group}] Connected to action server.")

    def sendTrajectory(self,
                       position_list: list,
                       transition_type: int = 4,
                       duration: float = 0.05,
                       isRobot = True):
        """
        直接传入多帧“关节位置”列表，每帧是一个 float 数组，长度 = len(self.joint_names)
        :param position_list: [[p1, p2, ...], [p1', p2', ...], ...]
        :param transition_type: TransitionType 枚举值
        :param duration:        持续时间 / 单位时间采样间隔
        """
        goal = lowlevel_controllerGoal()
        goal.group = self.group
        goal.type = transition_type
        goal.duration = duration

        for pos in position_list:
            js = JointState()
            js.name = self.joint_names
            js.position = pos
            if isRobot:
                goal.robotOutput.append(js)
            else:
                goal.motorOutput.append(js)

        rospy.loginfo(f"[{self.group}] Sending goal with {len(position_list)} points")
        self.client.send_goal(goal)

    def waitResult(self, timeout: float = 20.0):
        finished = self.client.wait_for_result(rospy.Duration(timeout))
        if not finished:
            rospy.logerr(f"[{self.group}] Action did not finish within {timeout}s")
            return None
        result = self.client.get_result()
        rospy.loginfo(f"[{self.group}] Result: success={result.success}"
                      + (f", errorMessage={result.errorMessage}" if hasattr(result, 'errorMessage') else ""))
        return result

if __name__ == "__main__":
    rospy.init_node('lowlevel_controller_multi_client_py')

    # 定义各组关节名称
    ankle_names = ["l_ankle_pitch_joint", "l_ankle_roll_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint"]
    hip_names   = ["l_hip_pitch_joint",   "l_hip_roll_joint",   "r_hip_pitch_joint",   "r_hip_roll_joint"]
    thigh_names = ["l_thigh_joint", "r_thigh_joint"]

    # 实例化三个 client
    ankle_client = LowlevelControllerClientRobotOutput("/sim2real_master_node", "ankle_joint", ankle_names)
    hip_client   = LowlevelControllerClientRobotOutput("/sim2real_master_node", "hip_joint",   hip_names)
    thigh_client = LowlevelControllerClientRobotOutput("/sim2real_master_node", "thigh_joint", thigh_names)

    # 直接传入多帧位置列表 RobotOutput

    ankle_positions = [
        [-0.000, 0.000,  -0.000,  -0.000],
        [-0.100, 0.000,  -0.100,  -0.000],
        [-0.200, 0.000,  -0.200,  -0.000],
        [-0.300, 0.000,  -0.300,  -0.000],
        [-0.400, 0.000,  -0.400,  -0.000],
        [-0.300, 0.000,  -0.300,  -0.000],
        [-0.200, 0.000,  -0.200,  -0.000],
        [-0.100, 0.000,  -0.100,  -0.000],
        [-0.000, 0.000,  -0.000,  -0.000]
    ]
    hip_positions = [
        [-0.0,0.0,-0.0,-0.0],
        [-0.1,0.0,-0.1,-0.0],
        [-0.2,0.0,-0.2,-0.0],
        [-0.3,0.0,-0.3,-0.0],
        [-0.4,0.0,-0.4,-0.0],
        [-0.3,0.0,-0.3,-0.0],
        [-0.2,0.0,-0.2,-0.0],
        [-0.1,0.0,-0.1,-0.0],
        [-0.0,0.0,-0.0,-0.0],
    ]
    thigh_positions = [
        [-0.000, 0.000],
        [-0.100, 0.100],
        [-0.200, 0.200],
        [-0.300, 0.300],
        [-0.400, 0.400],
        [-0.500, 0.500],
        [-0.400, 0.400],
        [-0.300, 0.300],
        [-0.200, 0.200],
        [-0.100, 0.100],
        [-0.000, 0.000],
    ]

    # 发送轨迹
    #  ankle_client.sendTrajectory(ankle_positions, transition_type=3, duration=1)
    hip_client.  sendTrajectory(hip_positions,   transition_type=4, duration=0.05)
    #  thigh_client.sendTrajectory(thigh_positions, transition_type=4, duration=0.05)

    # 等待结果

    #  ankle_client.waitResult()
    hip_client.  waitResult()
    #  thigh_client.waitResult()

