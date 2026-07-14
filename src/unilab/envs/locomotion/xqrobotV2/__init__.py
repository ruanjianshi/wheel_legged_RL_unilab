"""xqrobotV2 轮腿双足机器人环境包

四类任务, 均继承自平地行走基类:
- XqRobotV2WalkFlat   — 平地步态, 5D 速度命令 [vx, vy, vyaw, tsk, height]
- XqRobotV2WalkRough  — 崎岖地形, 继承 flat + 地形扫描 + 出界终止
- XqRobotV2JumpFlat   — 跳跃任务, 第 5 维为跳跃触发, 4 个专用奖励
- XqRobotV2ToeWalkFlat — 踮脚步态, 正弦参考轨迹 + 策略修正量输出

每个任务成对导出: Cfg(配置) + Env(环境实例)
"""
from .joystick import XqRobotV2WalkFlatCfg, XqRobotV2WalkFlatEnv
from .jump import XqRobotV2JumpFlatCfg, XqRobotV2JumpFlatEnv
from .rough import XqRobotV2WalkRoughCfg, XqRobotV2WalkRoughEnv
from .stairs import XqRobotV2StairsCfg, XqRobotV2StairsEnv
from .toe_walk import XqRobotV2ToeWalkFlatCfg, XqRobotV2ToeWalkFlatEnv
