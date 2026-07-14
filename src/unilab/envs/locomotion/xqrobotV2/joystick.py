"""xqrobotV2 平地行走环境 — 5D 速度命令 + 接触终止 + 历史堆叠 + 课程学习

这是整个 xqrobotV2 项目的核心文件, 所有其他任务 (rough/jump/toe_walk) 均继承自此。
策略以 joystick 方式接收速度命令 [vx, vy, vyaw, tsk, height], 
学习在各种速度/高度/髋差动指令下稳定行走。

关键设计决策:
- 轮子用速度控制 (kv=1),  腿用位置控制 (kp=30)
- 100Hz 控制频率 (ctrl_dt=0.01), 200Hz 物理仿真 (sim_dt=0.005)
- 9 帧历史堆叠 (297 维 actor obs, 324 维 critic obs)
- 14 个奖励函数: tracking, 稳定性, 动作平滑, 对称性, 脚距
- 对称课程学习: 跟踪误差驱动速度范围双向扩展
- 解耦命令采样: 训练时每次只激活 Vx 或 Vy 一个轴
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import DomainRandomizationCapabilities, ResetPlan
from unilab.dr.dr_utils import (
    build_common_reset_randomization,
    build_interval_push_plan,
    validate_interval_push_support,
    zero_actions,
)
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_mul, np_yaw_to_quat
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.xqrobotV2.base import (
    DEFAULT_ANGLES,
    NUM_ACTIONS,
    NUM_LEG_ACTIONS,
    NUM_WHEEL_ACTIONS,
    XqRobotBaseCfg,
    XqRobotBaseEnv,
    stack_joint_sensors,
    stack_joint_vel_sensors,
)

# ═══ 历史堆叠参数 ═══
# 9 帧 × 0.01s = 90ms 历史窗口, 给策略提供速度和姿态变化信息
_HISTORY_LEN = 9


# ═══ 域随机化配置 ═══


@dataclass
class XqRobotDomainRandConfig(DomainRandConfig):
    """域随机化参数 — 当前训练关闭, 后续收敛后可开启

    已实现的随机化维度:
    - 初始朝向 (randomize_init_yaw): 随机 yaw 角, 避免过拟合特定方向
    - 机身质量 (randomize_base_mass): ±1.5kg, 模拟负载变化
    - 地面摩擦 (randomize_ground_friction): 0.2~1.6×, 模拟不同路面
    - 执行器刚度/阻尼 (randomize_kp/kd): 0.8~1.2×, 模拟电机差异
    - 质心偏移 (random_com): ±3cm X/Y, 模拟装配误差

    未启用 (风险较高):
    - randomize_gravity: 重力扰动
    - push_robots: 外力推搡 (需 backend 支持)
    - randomize_leg_length: 腿长缩放 (需重编译 MjSpec, 太慢)
    """
    randomize_init_yaw: bool = True
    init_yaw_range: list[float] = field(default_factory=lambda: [-math.pi, math.pi])

    randomize_base_mass: bool = True
    added_mass_range: list[float] = field(default_factory=lambda: [-1.5, 1.5])
    randomize_ground_friction: bool = True
    ground_friction_multiplier_range: list[float] = field(default_factory=lambda: [0.2, 1.6])
    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.8, 1.2])
    random_com: bool = True
    com_offset_x: list[float] = field(default_factory=lambda: [-0.03, 0.03])
    com_offset_y: list[float] = field(default_factory=lambda: [-0.03, 0.03])

    randomize_gravity: bool = False
    push_robots: bool = False
    randomize_leg_length: bool = False
    leg_length_scale_range: list[float] = field(default_factory=lambda: [0.8, 1.2])


# ═══ 课程学习配置 ═══


@dataclass
class XqRobotCurriculumConfig:
    """课程学习 — 从简单到困难逐步扩展命令范围

    算法: 对称扩展
    - 每 update_interval(25) 步评估一次
    - 存活率 ≥ 50% 且跟踪误差 < err_threshold(0.35) → 扩速
    - 扩速量: vel_step(0.001)/ang_vel_step(0.002) 每步
    - 初始范围 = 满量程 × min_vel_range_frac(0.3)

    设计意图: 避免训练初期策略面对太快速度直接崩溃
    """
    enabled: bool = True
    vel_step: float = 0.001             # 线速度每次扩 0.001 m/s
    ang_vel_step: float = 0.002         # 角速度每次扩 0.002 rad/s
    min_vel_range_frac: float = 0.3     # 初始范围 = 满量程 × 30%
    min_ang_range_frac: float = 0.05    # 初始角速度 = 满量程 × 5% (先学走直线)
    update_interval: int = 25           # 每 25 步评估一次
    err_threshold: float = 0.35         # Vx 跟踪误差阈值, 低于此值才能扩速


# ═══ 奖励配置 ═══


@dataclass
class XqRobotRewardConfig:
    """奖励函数配置

    scales: 各 reward term 的权重 (由 Hydra YAML 注入)
    tracking_sigma: exp(-error^2 / sigma^2) 中的 sigma, 控制跟踪精度要求
    base_height_target: 目标 base 高度 (m), 默认 0.65
    only_positive_rewards: 是否裁剪负奖励为 0 (当前 false, 参考设计用 true)
    max_tilt_deg: 最大允许倾斜角 (度), 超过即终止
    min_base_height: 最低允许高度 (m), 低于即终止
    """
    scales: dict[str, float]
    tracking_sigma: float = 0.25
    base_height_target: float = 0.65
    only_positive_rewards: bool = True
    max_tilt_deg: float = 60.0
    min_base_height: float = 0.20


# ── 自定义奖励函数 (7 个) ─────────────────────────────────────────────────
# 注意: tracking_lin_vel, tracking_ang_vel, lin_vel_z, ang_vel_xy,
#       base_height, orientation, alive 是框架共享函数, 无需在此定义


def _reward_joint_action_rate(ctx: RewardContext) -> np.ndarray:
    """腿部动作平滑惩罚 — 相邻两帧腿关节动作差异的平方和
    权重: -0.1 (较小, 允许灵活, 但不能高频抖动)
    """
    current = ctx.info["current_actions"][:, :NUM_LEG_ACTIONS]
    last = ctx.info["last_actions"][:, :NUM_LEG_ACTIONS]
    return np.sum(np.square(current - last), axis=1)


def _reward_wheel_action_rate(ctx: RewardContext) -> np.ndarray:
    """轮子动作平滑惩罚 — 相邻两帧轮子动作差异的平方和
    权重: -0.005 (极小, 轮子本身就需要快速响应, 只防止极端突变)
    """
    current = ctx.info["current_actions"][:, NUM_LEG_ACTIONS:]
    last = ctx.info["last_actions"][:, NUM_LEG_ACTIONS:]
    return np.sum(np.square(current - last), axis=1)


def _reward_hip_roll(ctx: RewardContext) -> np.ndarray:
    """髋外展惩罚 — 前进/侧移时髋角必须接近 0 (不能靠大幅外展来产生推力)
    - 移动幅度越大, 惩罚越重 (moving/0.2 线性门控)
    - 0.3 缩放因子: 不压倒其他奖励
    权重: -2.0 (较重, 避免策略用髋来"划"着走)
    """
    moving = np.abs(ctx.info["commands"][:, 0]) + np.abs(ctx.info["commands"][:, 1])
    hip_mag = np.square(ctx.dof_pos[:, 0]) + np.square(ctx.dof_pos[:, 3])
    return hip_mag * np.clip(moving / 0.2, 0.0, 1.0) * 0.3


def _reward_similar_calf(ctx: RewardContext) -> np.ndarray:
    """腿部对称性惩罚 — 三项联合约束:
    - hip (镜像):  left + right ≈ 0  (双髋对称外展, L=-0.1 R=+0.1 时和为 0)
    - thigh (平行): left - right ≈ 0  (两大腿角度一致)
    - calf (平行):  left - right ≈ 0  (两小腿角度一致)
    权重: -1.0
    """
    hip = ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3]
    thigh = ctx.dof_pos[:, 1] - ctx.dof_pos[:, 4]
    calf = ctx.dof_pos[:, 2] - ctx.dof_pos[:, 5]
    return np.square(hip) + np.square(thigh) + np.square(calf)


def _reward_wheel_symmetry(ctx: RewardContext) -> np.ndarray:
    """轮子对称惩罚 — 直线行走时左右轮速必须相等, 否则会产生意外偏航
    - 转弯时 (|vyaw| > 0.1) 不惩罚, 允许差动转向
    - 0.5 因子: 稍放松, 允许微小不对称
    权重: -0.5
    """
    commands = ctx.info["commands"]
    turning = np.abs(commands[:, 2]) > 0.1  # vyaw > 0.1 = 转弯
    wheel_actions = ctx.info["current_actions"][:, -2:]
    diff = np.square(wheel_actions[:, 0] - wheel_actions[:, 1])
    return diff * (1.0 - turning.astype(np.float64)) * 0.5


def _reward_tsk(ctx: RewardContext) -> np.ndarray:
    """髋差动跟踪奖励 — 跟踪命令的第 4 维 tsk 目标值
    - tsk = hip_left - hip_right, 控制髋差动幅度
    - 策略通过调整双髋差值来实现转向辅助
    权重: -2.0
    """
    tsk_cmd = ctx.info["commands"][:, 3]
    hip_diff = ctx.dof_pos[:, 0] - ctx.dof_pos[:, 3]
    return np.square(hip_diff - tsk_cmd)


def _reward_feet_distance(ctx: RewardContext) -> np.ndarray:
    """脚距约束 — 两轮间距必须在 [0.3, 0.6]m 内
    - over:  间距 > 0.6m (腿分太开) 惩罚
    - under: 间距 < 0.3m (腿收太拢) 惩罚
    - 0.3 因子: 温和约束, 不强制精确值
    权重: -1.0
    """
    feet_dist = ctx.info.get("feet_distance")
    if feet_dist is None:
        return np.zeros((ctx.num_envs,), dtype=np.float64)
    dist = np.asarray(feet_dist, dtype=np.float64).reshape(-1)
    over = np.maximum(0.0, dist - 0.6)
    under = np.maximum(0.0, 0.3 - dist)
    return (over + under) * 0.3


def _reward_collision(ctx: RewardContext) -> np.ndarray:
    dof_pos = ctx.dof_pos
    thigh_margin = np.maximum(0.0, 0.12 - dof_pos[:, 1]) + np.maximum(0.0, 0.12 - dof_pos[:, 4])
    calf_margin = np.maximum(0.0, np.abs(dof_pos[:, 2]) - 0.75) + np.maximum(
        0.0, np.abs(dof_pos[:, 5]) - 0.75
    )
    collapse = thigh_margin + calf_margin
    moving = np.linalg.norm(ctx.info["commands"][:, :2], axis=1) > 0.1
    return np.asarray(collapse * moving, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════
# ★ 示例: 自定义奖励函数模板
# ══════════════════════════════════════════════════════════════════
# 
# 添加新奖励的 4 个步骤:
#   1. 在下面写函数 (签名必须为 (ctx: RewardContext) -> np.ndarray)
#   2. 在 _init_reward_functions 中注册
#   3. 在 YAML 的 reward.scales 中加权重
#   4. 跑训练验证
#
# RewardContext 可用字段:
#   ctx.num_envs      — 环境数量
#   ctx.dof_pos       — 腿关节角度 (num_envs, 6), 索引: L[0,1,2] R[3,4,5]
#   ctx.dof_vel       — 腿关节角速度 (num_envs, 6)
#   ctx.linvel        — 机体线速度 (num_envs, 3), [vx, vy, vz]
#   ctx.gyro          — 机体角速度 (num_envs, 3)
#   ctx.gravity       — 重力方向投影 (num_envs, 3), 直立时 ≈ [0, 0, -1]
#   ctx.base_height   — base_link 世界 Z 坐标 (num_envs,)
#   ctx.default_angles— 默认站姿 (6,)
#   ctx.info["commands"]       — 当前命令 (num_envs, 5): [vx, vy, vyaw, tsk, height]
#   ctx.info["current_actions"]— 当前动作 (num_envs, 8)
#   ctx.info["last_actions"]   — 上一帧动作 (num_envs, 8)
#   ctx.info["feet_distance"]  — 两轮横向间距 (num_envs,), 可能为 None
# ══════════════════════════════════════════════════════════════════


def _reward_example_smooth_height(ctx: RewardContext) -> np.ndarray:
    """【示例】平滑高度奖励 — 惩罚 base 高度剧烈抖动

    设计模式:
    - 用 base_height 获取传感器数据
    - 用 info["commands"][:, 4] 获取 height 命令
    - 返回值是"惩罚值", YAML 权重为负时它越大 → 最终 reward 越低
    - 形状必须是 (num_envs,), 每个 env 独立计算

    用法: 在 YAML 中将 example_smooth_height: 0.0 改为非零值即可激活
    """
    # ── 第 1 步: 获取数据 ──
    actual_height = ctx.base_height                              # (N,)  当前高度
    target_height = ctx.info["commands"][:, 4]                   # (N,)  命令中的目标高度

    # ── 第 2 步: 计算误差 ──
    height_error = actual_height - target_height                 # (N,)  有正有负

    # ── 第 3 步: 转为惩罚值 ──
    # 平方: 无论偏高还是偏低都惩罚; 二次增长: 偏差越大惩罚越重
    penalty = np.square(height_error)

    # ── 第 4 步: (可选) 获取上一次的 error 做平滑 ──
    # 从 info 中取自定义缓存的上一帧误差
    prev_error = ctx.info.get("_prev_height_error")
    if prev_error is not None:
        # |当前误差 - 上一帧误差| = error 变化量, 变化大 = 高度剧烈抖动
        jitter = np.abs(height_error - prev_error)
        penalty = penalty + jitter * 0.5  # 0.5 加权: 抖动惩罚不要太强

    # 缓存当前误差供下一帧用
    ctx.info["_prev_height_error"] = height_error

    return penalty.astype(np.float64)                            # ← 必须返回 (N,) 的 float64


# ═══ 任务配置 ═══


@registry.envcfg("XqRobotV2WalkFlat")
@dataclass
class XqRobotV2WalkFlatCfg(XqRobotBaseCfg):
    """平地行走任务配置

    注册为 Hydra 配置节点 "XqRobotV2WalkFlat",
    训练时通过 task=<name>/<backend> 选择 (例: task=xqrobotV2_walk_flat/mujoco)
    """
    # 场景: 平地 + xqrobotV2 机器人
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "xqrobotV2" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0  # 最大 20 秒 / 2000 步 (20/0.01)
    commands: Commands = field(default_factory=Commands)
    reward_config: XqRobotRewardConfig | None = None
    domain_rand: XqRobotDomainRandConfig = field(default_factory=XqRobotDomainRandConfig)
    curriculum: XqRobotCurriculumConfig = field(default_factory=XqRobotCurriculumConfig)

    # 触地终止: 检测大腿 (link_2) 或小腿 (link_3) 是否接触地面
    # 腿碰地 = 塌陷 → 终止 episode
    contact_body_names: list[str] = field(default_factory=lambda: [
        "left_link_2", "left_link_3", "right_link_2", "right_link_3",
    ])


# ═══ 域随机化提供者 ═══


class XqRobotDRProvider(LocomotionDRProvider):
    """Domain Randomization Provider — 负责 reset 和 interval DR

    核心职责:
    1. build_reset_plan: 每次 episode 开始时采样初始位姿和命令
    2. _sample_commands:  生成带解耦训练和逆比约束的命令
    3. _compute_reset_obs: reset 后立即计算初始观测 (填充历史)
    """

    _LEG_GEOM_NAMES = [
        "left_link_1_collision",
        "left_link_2_collision",
        "left_link_3_collision",
        "right_link_1_collision",
        "right_link_2_collision",
        "right_link_3_collision",
    ]

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        """构建 reset 计划 — 为即将重置的 env 生成新初始状态

        步骤:
        1. 复制默认 qpos/qvel
        2. 添加 XY 位置噪声 (±0.5m) + 地形原点偏移
        3. 随机 yaw 旋转 (域随机化)
        4. 采样新命令 (带解耦)
        5. 重置 current/last actions 为零
        """
        num_reset = len(env_ids)
        qpos = np.tile(env._init_qpos, (num_reset, 1))
        qvel = np.tile(env._init_qvel, (num_reset, 1))
        qpos[:, 0:2] += np.random.uniform(-0.5, 0.5, (num_reset, 2))
        qpos[:, 0:3] += env._spawn.origins_for(env_ids)
        yaw = self._sample_reset_yaw(env, num_reset)
        qpos[:, 3:7] = np_quat_mul(qpos[:, 3:7], np_yaw_to_quat(yaw))


        randomization = build_common_reset_randomization(env, num_reset)


        commands = self._sample_commands(env, num_reset)
        info_updates: dict[str, Any] = {
            "commands": commands,
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
        }
        if hasattr(env, "_spawn") and hasattr(env._spawn, "record_episode_start"):
            env._spawn.record_episode_start(env_ids, qpos[:, 0:3])
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=randomization,
        )

    def validate(self, env: Any, capabilities: DomainRandomizationCapabilities) -> None:
        """验证后端是否支持间歇推搡 DR"""
        validate_interval_push_support(env, capabilities)

    def build_interval_randomization_plan(self, env: Any, step_counter: int):
        """间歇 DR: 定期施加外力推搡 (当前 push_robots=False, 不触发)"""
        return build_interval_push_plan(env, step_counter)

    def _compute_reset_obs(self, env, env_ids, info_updates, linvel, gyro, gravity, dof_pos, dof_vel):
        """reset 后立即计算观测 — delegate 给 env._compute_obs

        这确保历史缓冲区被正确初始化 (填满 9 帧相同内容)
        """
        info_updates["_reset_ids"] = env_ids
        return env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel)

    def _sample_reset_yaw(self, env: Any, num_reset: int) -> np.ndarray:
        """采样初始 yaw 角 — 启用 DR 时随机, 否则为 0"""
        domain_rand = env._cfg.domain_rand
        if not domain_rand.randomize_init_yaw:
            return np.zeros((num_reset,), dtype=get_global_dtype())
        low, high = float(min(domain_rand.init_yaw_range)), float(max(domain_rand.init_yaw_range))
        return np.asarray(np.random.uniform(low, high, size=(num_reset,)), dtype=get_global_dtype())

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        """采样命令向量 — 5D [vx, vy, vyaw, tsk, height]

        两个关键约束:
        1. 逆比约束 (inverse_linx_angv): |vyaw| ≤ 2.0 / |vx|
           防止低速下发出不切实际的急转弯命令
        2. 解耦训练 (decoupling): 每次随机只激活 Vx 或 Vy 一个轴
           避免策略学出 Vx/Vy 串扰 (比如前进时莫名侧移)
        """
        low = np.asarray(env._cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(env._cfg.commands.vel_limit[1], dtype=get_global_dtype())
        cmds = np.asarray(
            np.random.uniform(low=low, high=high, size=(num_reset, low.shape[0])),
            dtype=get_global_dtype(),
        )
        # 逆比约束: R = 2.0 / |vx|, 确保角速度与线速度匹配 (类似转弯半径限制)
        safe_linv = np.maximum(np.abs(cmds[:, 0]), 1e-4)
        angv_limit = 2.0 / safe_linv
        cmds[:, 2] = np.clip(cmds[:, 2], -angv_limit, angv_limit)
        # 解耦训练: 每次只激活一个运动轴, 避免策略学出 Vx/Vy 串扰
        # 当 Vy 范围为 [0,0] 时, 跳过解耦仅保留 Vx
        vy_has_range = (high[1] - low[1]) > 1e-6
        for i in range(num_reset):
            if vy_has_range:
                axis = np.random.choice([0, 1])
            else:
                axis = 0
            if axis == 0:
                cmds[i, 1] = 0.0
            else:
                cmds[i, 0] = 0.0
        return cmds


# ═══ 环境类: 平地行走 ═══


@registry.env("XqRobotV2WalkFlat", sim_backend="mujoco")
class XqRobotV2WalkFlatEnv(XqRobotBaseEnv):
    """平地行走环境 — 所有 xqrobotV2 任务变体的基类

    这是一个 registered env: 通过 @registry.env 自动注册到框架,
    训练时框架根据 task_name + sim_backend 自动查找并实例化。

    核心流程 (每步):
    1. apply_action — 策略输出 → 关节目标值 (带 scale + default 偏置)
    2. backend.step — 推进物理仿真
    3. update_state — 读取传感器 → 计算终止/奖励/观测

    关键内部状态:
    - _obs_history / _critic_history: 9 帧滑动窗口 (历史堆叠)
    - _curriculum_step_count: 课程学习步数计数器
    - _tracking_err_buf: 跟踪误差累积缓冲
    """
    _cfg: XqRobotV2WalkFlatCfg

    def __init__(self, cfg: XqRobotV2WalkFlatCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        # 创建 MuJoCo 后端 (解析 XML, 分配环境, 设置仿真参数)
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=getattr(cfg.domain_rand, "push_body_name", None),
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._np_dtype = get_global_dtype()
        self._reward_cfg = cfg.reward_config
        self._enable_reward_log = True
        self._init_reward_functions()
        self._init_domain_randomization(XqRobotDRProvider())

        # 腿长域随机化 (当前关闭, 因为需要每次编译 XML 太重)
        if cfg.domain_rand.randomize_leg_length:
            leg_names = getattr(XqRobotDRProvider, "_LEG_GEOM_NAMES", [])
            if leg_names:
                low, high = (
                    float(cfg.domain_rand.leg_length_scale_range[0]),
                    float(cfg.domain_rand.leg_length_scale_range[1]),
                )
                scale = float(np.random.uniform(low, high))
                for name in leg_names:
                    geom_id = backend.get_geom_id(name)
                    backend._model.geom_size[geom_id] = (
                        backend._model.geom_size[geom_id].copy() * scale
                    )

        # 缓存左右轮 body ID (用于计算脚距)
        import mujoco as _mj

        if hasattr(backend, "_model"):  # type: ignore[union-attr]
            self._left_wheel_bid = _mj.mj_name2id(
                backend._model, _mj.mjtObj.mjOBJ_BODY, "left_link_wheel"
            )  # type: ignore[union-attr]
            self._right_wheel_bid = _mj.mj_name2id(
                backend._model, _mj.mjtObj.mjOBJ_BODY, "right_link_wheel"
            )  # type: ignore[union-attr]
        else:
            self._left_wheel_bid = -1
            self._right_wheel_bid = -1

        # ── 历史堆叠缓冲区 ──
        # (num_envs, 9, frame_dim) — 9 帧滑动窗口, 提供 90ms 历史信息
        # 参数含义见 _compute_obs 注释
        self._hist_len = _HISTORY_LEN
        self._obs_frame_dim = 33    # 5D cmd: gyro(3)+grav_neg(3)+leg_diff(6)+leg_vel(6)+wheel_vel(2)+last_act(8)+cmd(5)=33
        self._critic_frame_dim = 36 # 同上 + linvel(3) = 36 (critic 有特权速度信息)
        self._obs_history = np.zeros((num_envs, self._hist_len, self._obs_frame_dim), dtype=self._np_dtype)
        self._critic_history = np.zeros((num_envs, self._hist_len, self._critic_frame_dim), dtype=self._np_dtype)

        # ── 课程学习状态 ──
        self._curriculum_step_count = 0
        self._tracking_err_buf = np.zeros((num_envs,), dtype=self._np_dtype)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        """观测组维度声明 — 框架据此分配网络输入层大小

        - "obs": actor 观测, 297 = 33 × 9 (无特权信息)
        - "critic": critic 观测, 324 = 36 × 9 (含 linvel 特权)
        """
        return {"obs": self._obs_frame_dim * self._hist_len, "critic": self._critic_frame_dim * self._hist_len}

    # ── 动作执行 ──────────────────────────────────────────────────

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        """将策略输出 (8 维 [-1,1]) 转换为 MuJoCo 关节目标值

        关键: MuJoCo actuator 顺序为 [L_leg×3, L_wheel, R_leg×3, R_wheel]
        必须严格匹配 xqrobotV2.xml 中 actuator 的定义顺序!

        腿部: output × action_scale(0.25) + DEFAULT_ANGLES → 位置目标 (rad)
        轮子: output × wheel_action_scale(10.0) → 速度目标 (rad/s)
              轮子没有 default 偏置, 因为速度不需要偏置
        """
        clipped_actions = np.asarray(
            np.clip(
                actions,
                -self._cfg.control_config.clip_actions,
                self._cfg.control_config.clip_actions,
            ),
            dtype=self._np_dtype,
        )
        # 记录动作历史 (用于 action_rate 惩罚和平滑)
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(clipped_actions))
        state.info["current_actions"] = clipped_actions

        # 动作延迟模拟: 若开启, 使用上一帧的动作 (模拟 1 步控制延迟)
        exec_actions = (
            state.info["last_actions"]
            if self._cfg.control_config.simulate_action_latency
            else clipped_actions
        )
        if getattr(self._cfg.control_config, "action_smoothing", 0.0) > 0.0:
            alpha = float(self._cfg.control_config.action_smoothing)
            prev = getattr(self, "_prev_filtered_action", exec_actions)
            exec_actions = alpha * prev + (1.0 - alpha) * exec_actions
            self._prev_filtered_action = exec_actions

        # 腿: position control — 缩放 + 默认偏置
        leg_targets = (
            exec_actions[:, :NUM_LEG_ACTIONS] * self._cfg.control_config.action_scale
            + DEFAULT_ANGLES[:NUM_LEG_ACTIONS]
        )
        # 轮: velocity control — 仅缩放 (无偏置)
        wheel_targets = exec_actions[:, NUM_LEG_ACTIONS:] * self._cfg.control_config.wheel_action_scale

        # 重排为 MuJoCo actuator 顺序: [left_leg(0-2), left_wheel, right_leg(3-5), right_wheel]
        half_legs = NUM_LEG_ACTIONS // 2  # 3
        return np.concatenate([
            leg_targets[:, :half_legs],          # L_leg (3): hip, thigh, calf
            wheel_targets[:, :1],                 # L_wheel
            leg_targets[:, half_legs:],           # R_leg (3): hip, thigh, calf
            wheel_targets[:, 1:],                 # R_wheel
        ], axis=1, dtype=self._np_dtype)

    # ── 状态更新 ──────────────────────────────────────────────────

    def update_state(self, state: NpEnvState) -> NpEnvState:
        """每步主循环 — 读取传感器 → 计算终止/奖励/观测 → 更新课程

        调用链:
        _update_commands → 读传感器 → _update_feet_distance → _compute_terminated
        → _compute_reward → _compute_obs → _update_curriculum
        """
        self._update_commands(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        self._update_feet_distance(state.info)
        self._compute_costs(state.info, dof_pos, dof_vel)
        terminated = self._compute_terminated(gravity, dof_pos)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        self._update_curriculum(state.info)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    # ── 终止条件 ──────────────────────────────────────────────────

    def _compute_terminated(self, gravity: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        """判断 robot 是否终止 — 四个条件 (任一满足即终止):

        1. 倾角过大: arccos(gravity_z) > max_tilt(60°) — 机器人倒了
        2. 高度过低: base_height < min_base_height(0.20m) — 坐地上了
        3. 大腿塌陷: L/R thigh < 0.02 rad — 大腿角度太直 (趴了/后弯塌了)
        4. 小腿极限: |L/R calf| > 0.85 rad — 小腿过度弯曲 (跪了)

        注意: 与 rough.py 相比, 这里多了 base_height 检查 (平地有固定高度)
        """
        base_height = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        max_tilt = np.deg2rad(self._reward_cfg.max_tilt_deg)
        terminated = np.logical_or(tilt > max_tilt, base_height < self._reward_cfg.min_base_height)
        # 关节塌陷: 大腿过直 (角度太小) = 趴了
        thigh_collapsed = (dof_pos[:, 1] < 0.02) | (dof_pos[:, 4] < 0.02)
        # 小腿过度弯曲 (角度绝对值太大) = 跪了
        calf_extreme = (np.abs(dof_pos[:, 2]) > 0.85) | (np.abs(dof_pos[:, 5]) > 0.85)
        terminated |= thigh_collapsed
        terminated |= calf_extreme
        return terminated

    # ── 脚距计算 ──────────────────────────────────────────────────

    def _update_feet_distance(self, info: dict) -> None:
        """计算左右轮 body 的世界坐标 Y 方向距离 (横向间距)

        用于 _reward_feet_distance 检查间距是否在 [0.3, 0.6]m 内。
        如果后端无法读取 body 位置, 回退为 0.45m (默认值, 不做惩罚)
        """
        try:
            body_ids = np.array([self._left_wheel_bid, self._right_wheel_bid], dtype=np.int32)
            pos = self._backend.get_body_pos_w(body_ids)
            info["feet_distance"] = np.abs(pos[1, :, 1] - pos[0, :, 1]).astype(np.float64)
        except Exception:
            info["feet_distance"] = np.full((self._num_envs,), 0.45, dtype=np.float64)


    def _compute_costs(self, info: dict, dof_pos: np.ndarray, dof_vel: np.ndarray) -> None:
        """Compute per-step cost violations from physics sensor data.

        Mirrors Tita RL constraint functions — costs are 0/1 binary or small continuous
        violations computed from actual physics, NOT heuristic observation extraction.
        """
        num_envs = dof_pos.shape[0]
        costs = np.zeros((num_envs, 6), dtype=np.float64)

        # [0] orientation: tilt > 0.3 rad (≈17°)
        gravity = np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.upvector), dtype=np.float64
        )
        tilt = np.sqrt(gravity[:, 0] ** 2 + gravity[:, 1] ** 2)
        costs[:, 0] = (tilt > 0.3).astype(np.float64)

        # [1] joint velocity: mean |dof_vel| > 5 rad/s
        leg_vel = np.abs(dof_vel[:, :NUM_LEG_ACTIONS])
        costs[:, 1] = (leg_vel.mean(axis=1) > 5.0).astype(np.float64)

        # [2] joint acceleration: max |dof_vel - last_dof_vel|/dt > 800 rad/s²
        last_vel = info.get("_last_dof_vel", dof_vel.copy())
        acc = (
            np.abs(dof_vel[:, :NUM_LEG_ACTIONS] - last_vel[:, :NUM_LEG_ACTIONS]) / self._cfg.ctrl_dt
        )
        info["_last_dof_vel"] = dof_vel.copy()
        costs[:, 2] = (acc.max(axis=1) > 800.0).astype(np.float64)

        # [3] torque: approximated from leg force sensors
        try:
            lt = np.asarray(self._backend.get_sensor_data("left_thigh_torque"), dtype=np.float64)
            rt = np.asarray(self._backend.get_sensor_data("right_thigh_torque"), dtype=np.float64)
            lc = np.asarray(self._backend.get_sensor_data("left_calf_torque"), dtype=np.float64)
            rc = np.asarray(self._backend.get_sensor_data("right_calf_torque"), dtype=np.float64)
            torque_mag = np.stack(
                [
                    np.linalg.norm(lt.reshape(num_envs, -1), axis=1),
                    np.linalg.norm(rt.reshape(num_envs, -1), axis=1),
                    np.linalg.norm(lc.reshape(num_envs, -1), axis=1),
                    np.linalg.norm(rc.reshape(num_envs, -1), axis=1),
                ],
                axis=1,
            )
            costs[:, 3] = (torque_mag.max(axis=1) > 4.0).astype(np.float64)
        except Exception:
            pass

        # [4] foot contact force: wheel force > 500N
        try:
            lf = np.asarray(self._backend.get_sensor_data("left_wheel_force"), dtype=np.float64)
            rf = np.asarray(self._backend.get_sensor_data("right_wheel_force"), dtype=np.float64)
            wheel_f = np.stack(
                [
                    np.linalg.norm(lf.reshape(num_envs, -1), axis=1),
                    np.linalg.norm(rf.reshape(num_envs, -1), axis=1),
                ],
                axis=1,
            )
            costs[:, 4] = (wheel_f.max(axis=1) > 500.0).astype(np.float64)
        except Exception:
            pass

        # [5] stumble: horizontal force > 5 * vertical force
        try:
            lf_arr = lf.reshape(num_envs, -1)
            rf_arr = rf.reshape(num_envs, -1)
            l_horiz = np.sqrt(lf_arr[:, 0] ** 2 + lf_arr[:, 1] ** 2)
            r_horiz = np.sqrt(rf_arr[:, 0] ** 2 + rf_arr[:, 1] ** 2)
            l_stumble = l_horiz > 5 * np.abs(lf_arr[:, 2])
            r_stumble = r_horiz > 5 * np.abs(rf_arr[:, 2])
            costs[:, 5] = (l_stumble | r_stumble).astype(np.float64)
        except Exception:
            pass

        info["np3o_costs"] = costs

    def _update_leg_forces(self, info: dict) -> None:
        try:
            for name in (
                "left_thigh_force",
                "left_calf_force",
                "right_thigh_force",
                "right_calf_force",
            ):
                info[name] = np.asarray(self._backend.get_sensor_data(name), dtype=np.float64)
        except Exception:
            pass

    def _update_wheel_contact(self, info: dict) -> None:
        try:
            left_f = np.asarray(self._backend.get_sensor_data("left_wheel_force"), dtype=np.float64)
            right_f = np.asarray(
                self._backend.get_sensor_data("right_wheel_force"), dtype=np.float64
            )
            left_mag = np.linalg.norm(left_f.reshape(self._num_envs, -1), axis=1)
            right_mag = np.linalg.norm(right_f.reshape(self._num_envs, -1), axis=1)
            info["wheel_contact"] = np.stack(
                [
                    (left_mag > 1.0).astype(np.float64),
                    (right_mag > 1.0).astype(np.float64),
                ],
                axis=1,
            )
        except Exception:
            info["wheel_contact"] = np.ones((self._num_envs, 2), dtype=np.float64)

    def _update_base_z_history(self, info: dict) -> None:
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        info["_prev_base_z"] = info.get("current_base_z", base_pos[:, 2].copy())
        info["current_base_z"] = base_pos[:, 2].copy()

    # ── 奖励函数注册 ──────────────────────────────────────────────

    def _init_reward_functions(self) -> None:
        """注册 14 个奖励函数到 dispatch 表 (含 1 个示例, 默认关闭)

        其中 tracking_lin_vel, tracking_ang_vel, lin_vel_z, ang_vel_xy,
        base_height, orientation, alive 来自框架共享库 (common/rewards.py)
        其余是 xqrobotV2 自定义函数 (定义在本文件顶部)

        ── 添加新奖励的方法 ──
        在下面字典中加一行 "your_name": _your_function,
        然后在 YAML 的 reward.scales 中加 your_name: -1.0 (或其他权重)
        名字必须一致!
        """
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "lin_vel_z": rewards.lin_vel_z,
            "ang_vel_xy": rewards.ang_vel_xy,
            "base_height": rewards.base_height,
            "orientation": rewards.orientation,
            "joint_action_rate": _reward_joint_action_rate,
            "wheel_action_rate": _reward_wheel_action_rate,
            "similar_calf": _reward_similar_calf,
            "tsk": _reward_tsk,
            "hip_roll": _reward_hip_roll,
            "wheel_symmetry": _reward_wheel_symmetry,
            "feet_distance": _reward_feet_distance,
            "collision": _reward_collision,
            "alive": rewards.alive,
            "example_smooth_height": _reward_example_smooth_height,  # ★ 示例: 权重=0.0(关闭)
        }

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        """计算总奖励 — 构建 RewardContext → 分派到各 reward 函数 → 加权求和

        RewardContext 包含:
        - 所有传感器原始数据 (linvel, gyro, gravity, dof_pos, dof_vel)
        - info 字典中的元数据 (commands, actions)
        - 参考值 (default_angles, tracking_sigma, base_height_target)

        最终奖励 = Σ(scale_i × fn_i(ctx)), 其中 scale 来自 Hydra YAML 注入
        """
        dtype = get_global_dtype()
        num_obs = linvel.shape[0]
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos[:, :NUM_LEG_ACTIONS],
            dof_vel=dof_vel[:, :NUM_LEG_ACTIONS],
            num_envs=num_obs,
            default_angles=DEFAULT_ANGLES[:NUM_LEG_ACTIONS].astype(dtype),
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=self._reward_cfg.base_height_target,
            base_height=self._base_height_values(num_obs),
            gravity=gravity,
            joint_range=None,
        )
        return rewards.run_reward_dispatch(
            scales=self._reward_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._reward_cfg.only_positive_rewards,
        )

    # ── 课程学习 ──────────────────────────────────────────────────

    def _update_curriculum(self, info: dict) -> None:
        """对称课程学习 — 跟踪误差驱动速度范围扩展

        算法 (每 update_interval=25 步执行一次):
        1. 累积当前 Vx 跟踪误差到 buffer
        2. 计算存活率 = mean_ep_len / max_steps
        3. 若存活率 < 50% → 跳过 (策略还不够稳定)
        4. 若平均跟踪误差 < err_threshold(0.35) → 对称扩展速度范围
           - low[0] -= vel_step, high[0] += vel_step (Vx)
           - low[2] -= ang_vel_step, high[2] += ang_vel_step (Vyaw)
        5. 重置误差 buffer, 重置计数器

        设计意图: 从小速度开始训练, 策略稳定后逐步加速,
        避免训练初期面对大速度命令直接崩溃 (reward → 0, 无法学习)
        """
        cc = self._cfg.curriculum
        if not cc.enabled:
            return
        commands = info.get("commands")
        linvel = self.get_local_linvel()
        if commands is not None and linvel is not None:
            self._tracking_err_buf += np.abs(commands[:, 0] - linvel[:, 0])
        self._curriculum_step_count += 1
        if self._curriculum_step_count < cc.update_interval:
            return
        self._curriculum_step_count = 0

        # 存活率 = 平均 episode 长度 / 最大可能长度
        ep_steps = info.get("steps", np.zeros((self._num_envs,)))
        mean_ep_len = ep_steps[ep_steps > 0].mean() if np.any(ep_steps > 0) else 0.0
        max_steps = int(self._cfg.max_episode_seconds / self._cfg.ctrl_dt)
        survive_ratio = mean_ep_len / max_steps if max_steps > 0 else 0.0
        if survive_ratio < 0.5:
            return

        # 平均跟踪误差 (只统计存活过的 env)
        active = ep_steps[ep_steps > 0]
        mean_err = (
            self._tracking_err_buf[ep_steps > 0].mean() / active.mean()
            if len(active) > 0
            else 999.0
        )
        self._tracking_err_buf[:] = 0.0

        # 对称扩展: 同时扩大上下界, 保持 [-range, +range] 对称
        low = np.array(self._cfg.commands.vel_limit[0], dtype=self._np_dtype)
        high = np.array(self._cfg.commands.vel_limit[1], dtype=self._np_dtype)
        vx_range = max(abs(low[0]), abs(high[0]))
        vyaw_range = max(abs(low[2]), abs(high[2]))
        if mean_err < cc.err_threshold:
            low[0] = max(low[0] - cc.vel_step, -vx_range)
            high[0] = min(high[0] + cc.vel_step, vx_range)
            low[2] = max(low[2] - cc.ang_vel_step, -vyaw_range)
            high[2] = min(high[2] + cc.ang_vel_step, vyaw_range)
        self._cfg.commands.vel_limit[0] = low.tolist()
        self._cfg.commands.vel_limit[1] = high.tolist()

    # ── 命令重采样 ────────────────────────────────────────────────

    def _update_commands(self, info: dict) -> None:
        """定期重采样命令 — 每 resampling_time(3s) 生成新命令

        与 _sample_commands 相同的约束:
        - 逆比约束: |vyaw| ≤ 2.0 / |vx| (防止低速急转)
        - 解耦训练: 每次只激活 Vx 或 Vy 一个轴
        """
        commands = info.get("commands")
        if commands is None:
            return
        commands_arr = np.asarray(commands, dtype=get_global_dtype())
        resampling_time = float(getattr(self._cfg.commands, "resampling_time", 0.0))
        if resampling_time > 0.0:
            interval_steps = max(int(round(resampling_time / self._cfg.ctrl_dt)), 1)  # 3.0/0.01 = 300 步
            steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
            resample_mask = (steps > 0) & ((steps % interval_steps) == 0)
            if np.any(resample_mask):
                num_resample = int(np.count_nonzero(resample_mask))
                low = np.asarray(self._cfg.commands.vel_limit[0], dtype=get_global_dtype())
                high = np.asarray(self._cfg.commands.vel_limit[1], dtype=get_global_dtype())
                sampled = np.random.uniform(
                    low=low, high=high, size=(num_resample, low.shape[0])
                ).astype(get_global_dtype())
                safe_linv = np.maximum(np.abs(sampled[:, 0]), 1e-4)
                angv_limit = 2.0 / safe_linv
                sampled[:, 2] = np.clip(sampled[:, 2], -angv_limit, angv_limit)
                vy_has_range = (high[1] - low[1]) > 1e-6
                for i in range(num_resample):
                    if vy_has_range:
                        axis = np.random.choice([0, 1])
                    else:
                        axis = 0
                    if axis == 0:
                        sampled[i, 1] = 0.0
                    else:
                        sampled[i, 0] = 0.0
                commands_arr[resample_mask] = sampled
        info["commands"] = commands_arr

    # ── 观测 + 历史堆叠 ──────────────────────────────────────────

    def _compute_obs(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> dict[str, np.ndarray]:
        """构建观测 — 单帧 + 9 帧历史堆叠

        单帧观测 (33 维):
        - gyro(3):      机体角速度 (带噪声)
        - -gravity(3):  机体 z 轴在世界系投影 (即上向量, 隐含姿态)
                        用负号翻转, 使直立时所有元素为正 (更有物理意义)
        - leg_diff(6):  腿关节角度 - 默认姿态 (相对角度)
        - leg_vel(6):   腿关节角速度 (带噪声)
        - wheel_vel(2): 轮子速度 (带噪声)
        - last_act(8):  上一帧策略输出 (提供历史动作信息)
        - commands(5):  当前目标命令 [vx, vy, vyaw, tsk, height]

        Critic 单帧 (36 维): 同上 + linvel(3) (特权信息, 本体速度)

        历史堆叠: 9 帧滑动窗口
        - Reset 时: 所有 9 个 slot 填入当前帧 (避免空历史导致偏移)
        - Step 时: 左移 1 slot, 新帧填入末尾
        - 最终输出: (batch, 33*9=297) 或 (batch, 36*9=324)

        噪声只加在 actor obs 上 (提供鲁棒性), critic 用干净数据
        """
        noise_cfg = self._cfg.noise_config
        # 相对角度: 当前角度 - 默认站姿 → 物理含义更直观
        leg_diff = dof_pos[:, :NUM_LEG_ACTIONS] - DEFAULT_ANGLES[:NUM_LEG_ACTIONS]
        leg_vel = dof_vel[:, :NUM_LEG_ACTIONS]
        wheel_vel = dof_vel[:, NUM_LEG_ACTIONS:]
        # actor 观测加噪声 (域随机化 — 模拟传感器误差)
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_leg_diff = self._obs_noise(leg_diff, noise_cfg.scale_joint_angle)
        noisy_leg_vel = self._obs_noise(leg_vel, noise_cfg.scale_joint_vel)
        noisy_wheel_vel = self._obs_noise(wheel_vel, noise_cfg.scale_wheel_vel)
        last_actions = info.get("current_actions", np.zeros((linvel.shape[0], NUM_ACTIONS)))


        # Actor 单帧 (33 维): 带噪声, 无 linvel 特权
        obs_frame = np.concatenate([
            noisy_gyro, -noisy_gravity,
            noisy_leg_diff, noisy_leg_vel, noisy_wheel_vel,
            last_actions, info["commands"],
        ], axis=1, dtype=get_global_dtype())

        # Critic 单帧 (36 维): 无噪声, 含 linvel 特权
        critic_frame = np.concatenate([
            gyro, -gravity,
            leg_diff, leg_vel, wheel_vel,
            last_actions, info["commands"], linvel,
        ], axis=1, dtype=get_global_dtype())

        batch_size = obs_frame.shape[0]
        steps_val = int(info.get("steps", np.zeros(1, dtype=np.uint32))[0])

        if steps_val <= 1:
            # reset: 所有历史帧填入当前观测 (避免滑动后出现零帧偏置)
            for i in range(self._hist_len):
                self._obs_history[:batch_size, i, :] = obs_frame
                self._critic_history[:batch_size, i, :] = critic_frame
        else:
            # step: 滑动窗口 — 丢弃最旧一帧, 末尾加入新帧
            self._obs_history[:batch_size, :-1, :] = self._obs_history[:batch_size, 1:, :]
            self._obs_history[:batch_size, -1, :] = obs_frame
            self._critic_history[:batch_size, :-1, :] = self._critic_history[:batch_size, 1:, :]
            self._critic_history[:batch_size, -1, :] = critic_frame

        # 展平历史 → 策略输入
        obs = self._obs_history[:batch_size].reshape(batch_size, -1)
        critic = self._critic_history[:batch_size].reshape(batch_size, -1)
        return {"obs": obs, "critic": critic}

    def _base_height_values(self, num_obs: int) -> np.ndarray:
        """读取 base_link 的 Z 坐标 (世界系) — 用于 base_height 奖励

        如果维度不匹配 (例如后端还没初始化), 返回全零数组
        """
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
        if base_pos.shape[0] != num_obs:
            return np.zeros((num_obs,), dtype=get_global_dtype())
        return np.asarray(base_pos[:, 2], dtype=get_global_dtype())


# ── Motrix 后端注册 ─────────────────────────────────────────────
# 同一个 XqRobotV2WalkFlatEnv 类, 但绑定到 motrix 后端
# 训练时用 task=xqrobotV2_walk_flat/motrix 即可切换
registry.register_env("XqRobotV2WalkFlat", XqRobotV2WalkFlatEnv, sim_backend="motrix")
