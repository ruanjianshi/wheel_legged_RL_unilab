"""xqrobotwl 跌倒恢复 FTSR 环境 — 力引导 + 高度分阶段奖励 (CPO 约束)

基于论文: Hou et al. 2026, "Robust Fall Recovery for Armless Bipedal-Wheeled
Robots via Force-Guided Learning" (FTSR)。与早期"贴地后空翻"方案不同,
**无脚本轨迹, 纯学习恢复**:

  1. **多姿态复位**: supine/prone/左躺/右躺 四种倒地姿态 + 姿态扰动 +
     关节角扰动 → 策略从任意姿态自己学起身。
  2. **力引导学习**: 训练期施加与高度直接相关的外部辅助力 F (向上) 和
     力矩 T (对齐直立), F/T 同时作为 CPO 约束 (C1=F, C2=T, d→0) —
     策略被引导逐步降低对辅助的依赖, 学会无臂自恢复。
  3. **高度分阶段奖励**: 按批次高度统计切换阶段
     ru (上半身直立, 目标 h_cmd1) → rs (站起, 目标 h_cmd2)。
     每阶段独立奖励集 (论文 Table II 裁剪: 无行走阶段, 恢复后站住平衡)。
  4. **贴地终止**: 机身贴地超过 idle_ground_time (10s) 终止 (防闲置死点)。
  5. **观测 = 基础 297/324** (commands[4] = 当前阶段目标高度), 热启动自 walk。

Joint order (policy): [L_hip_roll, L_hip_pitch, L_knee, R_hip_roll, R_hip_pitch, R_knee, L_wheel, R_wheel]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.dr import ResetPlan
from unilab.dr.dr_utils import build_common_reset_randomization, zero_actions
from unilab.dtype_config import get_global_dtype
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.xqrobotwl.base import (
    DEFAULT_LEG_ANGLES,
    NUM_LEG_ACTIONS,
    NUM_WHEEL_ACTIONS,
)
from unilab.envs.locomotion.xqrobotwl.joystick import XqRobotWLWalkFlatEnv
from unilab.envs.locomotion.xqrobotwl.jump_srl import (
    XqRobotWLJumpDRProvider,
    XqRobotWLJumpSRLFlatCfg,
    XqRobotWLJumpSRLFlatEnv,
)

# ── 常数 ──
WHEEL_R = 0.11
G = 9.81
# 倒地姿态 base_z (贴地): 躯干厚度~0.12-0.16
_LYING_Z = 0.15
# 关节极限 (防随机复位超限)
_HIP_PITCH_LIM = 1.0
_KNEE_LIM = 0.85
# 空闲贴地判据 (0.25 杀掉"撑到 0.26 躲终止"的局部最优; 合法恢复会快速穿过)
_IDLE_Z = 0.25


def _quat_from_euler(roll, pitch, yaw):
    """按 ZYX (yaw→pitch→roll) 内旋构造四元数 [qw,qx,qy,qz] (与 MuJoCo euler 一致)."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def _pose_quat(pose: int) -> np.ndarray:
    """4 种倒地姿态的 base 朝向 (世界系, 加扰动前的基准):
    0 supine(背贴地, 前胸朝上), 1 prone(前胸贴地), 2 左躺, 3 右躺.
    """
    if pose == 0:  # supine: R_y(-90°), up=-x
        return _quat_from_euler(0.0, -math.pi / 2, 0.0)
    if pose == 1:  # prone: R_y(+90°), up=+x
        return _quat_from_euler(0.0, math.pi / 2, 0.0)
    if pose == 2:  # 左躺: R_x(+90°), up=-y
        return _quat_from_euler(math.pi / 2, 0.0, 0.0)
    return _quat_from_euler(-math.pi / 2, 0.0, 0.0)  # 右躺: R_x(-90°), up=+y


# ========== 奖励函数 (原始项, 阶段权重由 scales_ru/rs 决定) ==========


def _reward_orient(ctx: RewardContext) -> np.ndarray:
    """方向: ||g_xy||^2 (重力投影, 直立=0) — ru/rs 都激活 (负权重=罚倾覆)."""
    g = np.asarray(ctx.gravity, dtype=get_global_dtype())[: ctx.num_envs]
    return g[:, 0] ** 2 + g[:, 1] ** 2


def _reward_base_height(ctx: RewardContext) -> np.ndarray:
    """基准高度: 线性进度 (h 从贴地 _IDLE_Z 线性升到 h_cmd 得满分),
    超过 h_cmd 后线性归零 — 防"升直腿超高"站姿 (恢复后高度应≈h_cmd).

    exp(-8.3*(h-h_cmd)^2) 在低处饱和 (0.26 处得 0.94, 与 0.35 几乎同分) —
    会诱导"撑到 0.26 混分"局部最优。线性梯度 0.26 只得 0.55, 给强上升推力。
    但线性部分在 h≥h_cmd 处恒为 1.0, 无超高惩罚 — 策略学会升直腿到 0.60-0.67m
    (与 0.55m 同分)。故 h_cmd 以上 0.12m 内线性归零 (0.67m 以上=0) —
    band 0.12 不杀起身自然冲高 (0.60 仍得 0.58), 但罚 0.596+ 的过伸站姿。
    """
    h_cmd = ctx.info.get("h_cmd", 0.45)
    h = ctx.base_height[: ctx.num_envs]
    rise = np.clip((h - _IDLE_Z) / (h_cmd - _IDLE_Z + 1e-6), 0.0, 1.0)
    over = np.clip((h - h_cmd) / 0.12, 0.0, 1.0)  # 高于 h_cmd 0.12m 内归零
    return np.clip(rise * (1.0 - over), 0.0, 1.0)


def _reward_act_rate(ctx: RewardContext) -> np.ndarray:
    cur = ctx.info["current_actions"]
    lst = ctx.info["last_actions"]
    return np.sum(np.square(cur - lst), axis=1)


def _reward_dof_pos(ctx: RewardContext) -> np.ndarray:
    """自由度位置: 罚偏离自然站姿构型 (rs 阶段回归站姿).

    目标用 info["standing_angles"] (walk 模型实测自然站姿, base_z≈0.518),
    不是 DEFAULT_LEG_ANGLES (膝盖符号相反, 会拉向错误姿态 0.474m)."""
    nominal = np.asarray(
        ctx.info.get("standing_angles", ctx.default_angles[:NUM_LEG_ACTIONS]),
        dtype=get_global_dtype(),
    )[:NUM_LEG_ACTIONS]
    return np.sum(np.square(ctx.dof_pos - nominal), axis=1)


def _reward_stand_pose(ctx: RewardContext) -> np.ndarray:
    """门控站姿正奖励 (治蹲姿): 直立 × 接近站立高度 × exp(-||dof-standing_angles||²/σ).

    ★ v7: v6 的蹲姿 (L_pitch -0.83, R_knee +0.91) 产生持续 yaw 力矩 → 转圈。
    dof_pos 是全局惩罚 (起身期也生效, 强了压坏起身 v4 教训, 弱了压不住蹲姿)。
    这个是**正奖励 + height_ok×up_ok 门控** — 只在站立期激活 (起身期≈0 不影响),
    站对自然站姿给分, 把蹲姿拉直。σ² = 2*(0.25)²."""
    h = ctx.base_height[: ctx.num_envs]
    h_cmd = ctx.info.get("h_cmd", 0.55)
    height_ok = np.clip(1.0 - np.abs(h - h_cmd) / 0.15, 0.0, 1.0)
    up_ok = np.clip(np.asarray(ctx.gravity, dtype=get_global_dtype())[:, 2], 0.0, 1.0)
    nominal = np.asarray(
        ctx.info.get("standing_angles", ctx.default_angles[:NUM_LEG_ACTIONS]),
        dtype=get_global_dtype(),
    )[:NUM_LEG_ACTIONS]
    err = np.sum(np.square(ctx.dof_pos - nominal), axis=1)
    pose_ok = np.exp(-err / (2.0 * 0.25**2))
    return (height_ok * up_ok * pose_ok).astype(get_global_dtype())


def _reward_leg_bias(ctx: RewardContext) -> np.ndarray:
    """腿部偏置: 罚左右腿不对称 (镜像误差)."""
    l = ctx.dof_pos[:, 0:3]
    r = ctx.dof_pos[:, 3:6]
    hip_err = np.abs(l[:, 0] + r[:, 0])
    pitch_err = np.abs(l[:, 1] + r[:, 1])
    knee_err = np.abs(l[:, 2] - r[:, 2])
    return hip_err + pitch_err + knee_err


def _reward_no_fly(ctx: RewardContext) -> np.ndarray:
    """禁止飞行: 双轮同时离地 = 1 (倒地恢复期不应腾空)."""
    contact = ctx.info.get("wheel_contact", np.zeros((ctx.num_envs, 2)))
    both_off = (1.0 - contact).prod(axis=1)
    return both_off


def _reward_wheel_force(ctx: RewardContext) -> np.ndarray:
    """腿部使用: 轮地接触力 ≈ 总重 M 时给奖 (腿撑起机身)."""
    f = ctx.info.get("wheel_force_total", np.zeros(ctx.num_envs))
    M = ctx.info.get("robot_mass", 18.65) * G
    return np.exp(-np.square(f - M) / (2.0 * 100.0**2))


def _reward_alive(ctx: RewardContext) -> np.ndarray:
    return np.ones(ctx.num_envs, dtype=get_global_dtype())


def _reward_upright(ctx: RewardContext) -> np.ndarray:
    """躯干直立正奖励: up·[0,0,1] (躺地=0, 完全直立=1).

    只有 orient 罚项(||g_xy||²)时, 机器人半撑到 0.29m 就够规避罚项 —
    加正奖励直接驱动躯干转正 (恢复的关键动作)。
    """
    return np.clip(np.asarray(ctx.gravity, dtype=get_global_dtype())[:, 2], 0.0, 1.0)


def _reward_recover_complete(ctx: RewardContext) -> np.ndarray:
    """恢复完成(锁存): 站立 (base_z>recover_height + 直立>0.85 + 双轮着地) 连续保持
    0.5s → 一次性大奖。制造"必须完成恢复"的奖励悬崖, 打破半撑混分的局部最优."""
    done = ctx.info.get("recover_completed", np.zeros(ctx.num_envs, dtype=bool))
    return done.astype(np.float64)


def _reward_rise(ctx: RewardContext) -> np.ndarray:
    """中间里程碑(锁存): base_z>rise_height + 直立>0.80 + 双轮着地 连续保持
    rise_hold → 一次性中奖。桥接 0.30(半撑)→0.45(站立) 的奖励梯度死区,
    给"再推一把腿"一个够得着的小目标 (与 recover_complete 形成阶梯)."""
    done = ctx.info.get("rise_completed", np.zeros(ctx.num_envs, dtype=bool))
    return done.astype(np.float64)


def _reward_rise_vel(ctx: RewardContext) -> np.ndarray:
    """上升速度(密集, 起身过渡期): 向上 base_z 速度 (双轮着地 + base_z<cap 时), 封顶 1.0m/s.

    直接奖"推腿自举"动作 — 力辅助时代策略从不需主动举升, 力撤走后缺失此技能.
    逐帧密集反馈, 给半撑→站立的最小推力一个即时梯度 (脚手架).
    到站立高度后归零 → 顶部不再鼓励继续推 (防跳动/冲过头)."""
    return ctx.info.get("rise_vel", np.zeros(ctx.num_envs, dtype=get_global_dtype()))


def _reward_settle(ctx: RewardContext) -> np.ndarray:
    """稳定站立(连续): 直立 × 接近站立高度 × 垂直静止 × 角速度静止 — 教"站稳".

    rise_vel 教推力, 这个是补缺失的平衡技能: 高度到位 + 躯干转正 + 垂直几乎不动 +
    不摇摆 = 满分. 躺地/跳动/倾倒时接近 0 — 直接针对"冲到高度但站不住"的失败模式.
    ★ 加角速度静止项 (ctx.gyro): model_4000 诊断发现站立时 |gyro| 高达 6-9 rad/s
    (剧烈摇摆), 但旧 settle 只罚垂直速度不罚角速度 → 摇摆不受罚, 几帧就倒.
    需 info["abs_rise_vel"] (垂直速度绝对值) + ctx.gyro (角速度)."""
    h = ctx.base_height[: ctx.num_envs]
    h_cmd = ctx.info.get("h_cmd", 0.55)
    height_ok = np.clip(1.0 - np.abs(h - h_cmd) / 0.15, 0.0, 1.0)
    up_ok = np.clip(np.asarray(ctx.gravity, dtype=get_global_dtype())[:, 2], 0.0, 1.0)
    avz = ctx.info.get("abs_rise_vel", np.zeros(ctx.num_envs, dtype=get_global_dtype()))
    still_v = 1.0 - np.clip(np.abs(avz) / 1.0, 0.0, 1.0)
    # 角速度静止: exp(-|gyro|/2.0) — 指数衰减给全程梯度 (0.5 硬门是"全有或全无",
    # gyro 6→3→1 全程 0 分学不到; 指数型 gyro=0→1, 2→0.37, 4→0.14, 6→0.05, 持续压低摇摆)
    gyro = np.linalg.norm(np.asarray(ctx.gyro, dtype=get_global_dtype())[: ctx.num_envs], axis=1)
    still_g = np.exp(-gyro / 2.0)
    return (height_ok * up_ok * still_v * still_g).astype(get_global_dtype())


def _reward_stand_still(ctx: RewardContext) -> np.ndarray:
    """水平静止 (独立项, 防一直后退): 直立 × 接近站立高度 × 水平速度小 = 满分.

    settle 只管垂直静止; 这个是补"水平漂移"的梯度 — 恢复后 base 本地水平速度
    < 0.5 m/s 才给分, 直接针对用户反馈"恢复后一直后退" (实测漂移 2.48-4.62m).
    独立加法项 (不乘死 settle), 避免起身阶段的水平运动把站稳信号一起抹掉."""
    h = ctx.base_height[: ctx.num_envs]
    h_cmd = ctx.info.get("h_cmd", 0.55)
    height_ok = np.clip(1.0 - np.abs(h - h_cmd) / 0.15, 0.0, 1.0)
    up_ok = np.clip(np.asarray(ctx.gravity, dtype=get_global_dtype())[:, 2], 0.0, 1.0)
    vxy = np.linalg.norm(
        np.asarray(ctx.linvel, dtype=get_global_dtype())[: ctx.num_envs, :2], axis=1
    )
    still_h = 1.0 - np.clip(vxy / 0.5, 0.0, 1.0)
    return (height_ok * up_ok * still_h).astype(get_global_dtype())


def _reward_no_yaw(ctx: RewardContext) -> np.ndarray:
    """朝向静止 (防转圈): 直立 × 接近站立高度 × yaw 角速度小 = 满分.

    针对用户反馈"以右腿为圆心转圈" (实测站立时 yaw 累计旋转 260°).
    gyro z 分量 = 本地 yaw 角速度, 指数衰减给全程梯度 (转越快分越低).
    ★ v8.2: 加死区 — |gyro_z| < no_yaw_deadzone(1.0) 免费, 超出才罚.
    倒立摆平衡需要小幅 yaw 微调, 直接罚 yaw 角速度会杀平衡自由度 (v7 站立 0.63s 教训);
    但完全放松又让持续打转回归 (v8 转圈 350°/resume 6000-7000 400-680°).
    死区 = 允许平衡微调, 专打持续转圈 (>1 rad/s 重罚)."""
    h = ctx.base_height[: ctx.num_envs]
    h_cmd = ctx.info.get("h_cmd", 0.55)
    height_ok = np.clip(1.0 - np.abs(h - h_cmd) / 0.15, 0.0, 1.0)
    up_ok = np.clip(np.asarray(ctx.gravity, dtype=get_global_dtype())[:, 2], 0.0, 1.0)
    gyro_z = np.abs(np.asarray(ctx.gyro, dtype=get_global_dtype())[: ctx.num_envs, 2])
    dead = ctx.info.get("no_yaw_deadzone", 1.0)
    still_yaw = np.exp(-np.clip(gyro_z - dead, 0.0, None) / 0.8)
    return (height_ok * up_ok * still_yaw).astype(get_global_dtype())


def _reward_wheel_symmetry(ctx: RewardContext) -> np.ndarray:
    """轮速对称 (治差速转圈): exp(-|wL-wR|/12) — 轮速差小则 1, 差大则 0.

    ★ v6 决定性根因: 站立时右轮 -565 rad/s 狂转、左轮 -91 rad/s, 差速 474 rad/s →
    机器人以右轮为轴空转打转。v6 用 /20 阈值抓到狂转 (474→3.9), 但**残留小差速
    (L恒+1.2, 方向恒定) 累积成转圈 yaw 299°** → 收紧到 /5 抓"小而恒定"差速。
    ★ v8 回松到 /12: 倒立摆平衡**本质需要轮子差速微调**, /5 把平衡自由度一起罚掉
    → v7 站立塌到 0.63-0.81s。净旋转改由 _reward_stand_anchor (yaw 锚点) 专管,
    此处只拦狂转/明显差速, 不再抓平衡用的小差速。"""
    wv = ctx.info.get("wheel_vel", np.zeros((ctx.num_envs, 2), dtype=get_global_dtype()))
    diff = np.abs(wv[:, 0] - wv[:, 1])
    return np.exp(-diff / 12.0)


def _reward_wheel_speed(ctx: RewardContext) -> np.ndarray:
    """轮速过大 (防空转): exp(-(|wL|+|wR|)/100) — 轮速小则 1, 狂转则 0.

    站立应几乎静止 (轮速≈0); 空转不打转也不产生推进, 纯浪费."""
    wv = ctx.info.get("wheel_vel", np.zeros((ctx.num_envs, 2), dtype=get_global_dtype()))
    speed = np.abs(wv[:, 0]) + np.abs(wv[:, 1])
    return np.exp(-speed / 100.0)


def _reward_stand_anchor(ctx: RewardContext) -> np.ndarray:
    """锚点站立 (净位移/净旋转): 恢复锁存后, 罚离开站立点的净水平位移与净 yaw 旋转.

    ★ v8: 解耦"平衡微调"与"净漂移"。v7 用 wheel_symmetry/5 + no_yaw + stand_still
    同时罚掉倒立摆平衡必需的轮子微调 (前后来回/差速/小幅 yaw) → 站立时长塌到 0.63-0.81s;
    而这些瞬时项又只罚速度不罚净位移 → 慢漂仍超 0.5m (评估口径 max|x−x₀|, 附录A <0.5m)。
    锚点项直接对净位移/净旋转: 轮子可自由微调平衡, 但 base 离开恢复点净位移
    >σ_xy 或净 yaw 旋转 >σ_yaw 则失分。只在恢复锁存后激活 (起身期≈0 不影响恢复)。"""
    h = ctx.base_height[: ctx.num_envs]
    h_cmd = ctx.info.get("h_cmd", 0.55)
    height_ok = np.clip(1.0 - np.abs(h - h_cmd) / 0.15, 0.0, 1.0)
    up_ok = np.clip(np.asarray(ctx.gravity, dtype=get_global_dtype())[:, 2], 0.0, 1.0)
    anchored = ctx.info.get("anchor_active", np.zeros(ctx.num_envs, dtype=bool))
    sig_xy = ctx.info.get("stand_anchor_sigma_xy", 0.25)
    sig_yaw = ctx.info.get("stand_anchor_sigma_yaw", 0.35)
    dx = ctx.info.get("anchor_dx", np.zeros(ctx.num_envs, dtype=get_global_dtype()))
    dy = ctx.info.get("anchor_dy", np.zeros(ctx.num_envs, dtype=get_global_dtype()))
    dyaw = ctx.info.get("anchor_dyaw", np.zeros(ctx.num_envs, dtype=get_global_dtype()))
    d2 = (dx**2 + dy**2) / sig_xy**2 + (dyaw**2) / sig_yaw**2
    anchored_ok = np.exp(-d2)
    return (height_ok * up_ok * anchored * anchored_ok).astype(get_global_dtype())


# ========== 配置 ==========


@dataclass
class XqRobotWLFallRecoveryCommands(Commands):
    """5D 命令: [vx, vy, vyaw, tsk, height] — 恢复期 vx/vy/vyaw/tsk=0, height=h_cmd"""

    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0, 0.0, 0.30], [0.0, 0.0, 0.0, 0.0, 0.55]]
    )
    resampling_time: float = 4.0


@dataclass
class XqRobotWLFallRecoveryRewardConfig:
    scales_ru: dict[str, float]  # 阶段1: 上半身直立
    scales_rs: dict[str, float]  # 阶段2: 站起
    only_positive_rewards: bool = False
    tracking_sigma: float = 0.3
    max_tilt_deg: float = 55.0
    min_base_height: float = 0.15
    # 高度分阶段
    h_cmd1: float = 0.32  # ru 目标/ru→rs 阈值
    h_cmd2: float = 0.55  # rs 目标 (站立)
    stage_fraction: float = 2.0 / 3.0  # 超过阈值代理比例达到该值才切阶段
    # 恢复里程碑 (阶梯): rise 中间小目标 → recover_complete 最终大奖
    recover_height: float = 0.45  # 最终站立高度阈值 (base_z)
    rise_height: float = 0.35  # 中间里程碑高度 (桥接半撑→站立梯度死区)
    rise_hold: float = 0.3  # 中间里程碑保持时间 (s)
    rise_vel_height_cap: float = 0.45  # rise_vel 仅在此高度以下生效 (顶部不鼓励推)
    rise_vel_up_gate: float = 0.35  # ★ v8.1: rise_vel 按躯干直立度门控 (up/0.35, 躺着推=0)
    no_yaw_deadzone: float = (
        1.0  # ★ v8.2: no_yaw 死区 (|gyro_z|<1rad/s 免费, 保平衡微调; 超出重罚持续打转)
    )
    # ★ 自然站姿 (dof_pos 目标): walk 模型实测站立腿角 [L_roll,L_pitch,L_knee,R_roll,R_pitch,R_knee]
    #   → base_z≈0.518 (与 h_cmd2=0.52 兼容)。DEFAULT_LEG_ANGLES 膝盖符号相反 (0.474m),
    #   若用它做 dof_pos 目标会把腿拉向错误姿态 (一前一后/别扭)。
    standing_angles: list[float] = field(
        default_factory=lambda: [0.1021, 0.0828, -0.0789, 0.0127, -0.1083, 0.0188]
    )
    # 力引导 (CPO 约束 C1=F, C2=T)
    force_assist_enabled: bool = True
    force_max: float = 160.0  # Fmax (N, 向上)
    torque_max: float = 15.0  # Tmax (Nm, 对齐直立)
    mu: float = 6.0  # 高度系数
    force_end_iters: int = 3000  # 辅助力全局撤除的训练 iter
    # ★ v8 锚点站立 (净位移/净旋转): 恢复锁存后罚离开站立点的净漂移/净转圈
    #   评估口径 = episode 内 max 水平位移 <0.5m (附录A), 故 σ_xy 取 0.25 给足梯度
    stand_anchor_sigma_xy: float = 0.25  # 净水平位移 σ (m)
    stand_anchor_sigma_yaw: float = 0.35  # 净 yaw 旋转 σ (rad)
    # 终止
    idle_ground_time: float = 6.0  # 贴地超过即终止 (防死点; < max_episode 10s)


@registry.envcfg("XqRobotWLFallRecoveryFlat")
@dataclass
class XqRobotWLFallRecoveryFlatCfg(XqRobotWLJumpSRLFlatCfg):
    commands: XqRobotWLFallRecoveryCommands = field(default_factory=XqRobotWLFallRecoveryCommands)  # type: ignore[assignment]  # 子类收窄命令/奖励配置类型
    reward_config: XqRobotWLFallRecoveryRewardConfig | None = None  # type: ignore[assignment]
    max_episode_seconds: float = 10.0


class XqRobotWLFallRecoveryDRProvider(XqRobotWLJumpDRProvider):
    """多姿态复位: 4 种倒地姿态随机 + 姿态/关节扰动, 命令恒 height=h_cmd1."""

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        cmds = np.zeros((num_reset, 5), dtype=get_global_dtype())
        cmds[:, 4] = getattr(env._cfg.reward_config, "h_cmd1", 0.35)
        return cmds

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        rng = np.random.default_rng()
        pose = rng.integers(0, 4, size=num_reset)
        base_z = np.full(num_reset, _LYING_Z, dtype=get_global_dtype())
        base_z += rng.uniform(-0.02, 0.02, size=num_reset)
        # 姿态扰动: 基准倒地 + euler 扰动 U(-0.3, 0.3)
        quats = np.zeros((num_reset, 4), dtype=get_global_dtype())
        for i in range(num_reset):
            q = _pose_quat(int(pose[i]))
            dq = _quat_from_euler(
                rng.uniform(-0.3, 0.3),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-0.3, 0.3),
            )
            quats[i] = _quat_mul(q, dq)
        # 关节扰动: 默认腿角 * U(0.5, 1.5), 裁剪到关节极限
        leg_scale = rng.uniform(0.5, 1.5, size=(num_reset, NUM_LEG_ACTIONS))
        legs = np.clip(DEFAULT_LEG_ANGLES * leg_scale, -_KNEE_LIM, _KNEE_LIM)
        legs[:, 0] = np.clip(legs[:, 0], 0.0, 3.14)  # L_hip_roll ≥ 0
        legs[:, 3] = np.clip(legs[:, 3], -3.14, 0.0)  # R_hip_roll ≤ 0

        qpos = np.zeros((num_reset, 15), dtype=get_global_dtype())
        qpos[:, 0] = rng.uniform(-0.1, 0.1, size=num_reset)
        qpos[:, 1] = rng.uniform(-0.1, 0.1, size=num_reset)
        qpos[:, 2] = base_z
        qpos[:, 3:7] = quats
        # 关节序 (MuJoCo): [L_roll,L_pitch,L_knee,L_wheel, R_roll,R_pitch,R_knee,R_wheel]
        qpos[:, 7] = legs[:, 0]
        qpos[:, 8] = legs[:, 1]
        qpos[:, 9] = legs[:, 2]
        qpos[:, 11] = legs[:, 3]
        qpos[:, 12] = legs[:, 4]
        qpos[:, 13] = legs[:, 5]
        qvel = np.zeros((num_reset, 14), dtype=get_global_dtype())
        randomization = build_common_reset_randomization(env, num_reset)
        commands = self._sample_commands(env, num_reset)
        info_updates: dict[str, Any] = {
            "commands": commands,
            "current_actions": zero_actions(num_reset, env._num_action),
            "last_actions": zero_actions(num_reset, env._num_action),
        }
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=randomization,
        )


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """四元数乘法 (Hamilton, [qw,qx,qy,qz]) — 返回与 a 同 dtype."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=a.dtype,
    )


@registry.env("XqRobotWLFallRecoveryFlat", sim_backend="mujoco")
class XqRobotWLFallRecoveryFlatEnv(XqRobotWLJumpSRLFlatEnv):
    """xqrobotwl 跌倒恢复 FTSR 环境 — 力引导约束 + 分阶段奖励 + 多姿态复位"""

    _cfg: XqRobotWLFallRecoveryFlatCfg
    _jump_cfg: XqRobotWLFallRecoveryRewardConfig  # type: ignore[assignment]  # 收窄基类奖励配置类型

    def __init__(self, cfg: XqRobotWLFallRecoveryFlatCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        self._jump_cfg = cfg.reward_config
        self._total_env_steps = 0
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._dr_manager._provider = XqRobotWLFallRecoveryDRProvider()  # type: ignore[union-attr]
        # FTSR 状态
        self._stage = 0  # 全局阶段 0=ru 1=rs
        self._idle_time = np.zeros(num_envs, dtype=np.float64)
        self._has_recovered = np.zeros(num_envs, dtype=bool)  # base_z 达 h_cmd1 锁存
        self._recover_hold = np.zeros(num_envs, dtype=np.float64)  # 站立保持计时
        self._recover_completed = np.zeros(num_envs, dtype=bool)  # 恢复完成锁存
        self._rise_hold = np.zeros(num_envs, dtype=np.float64)  # 中间里程碑保持计时
        self._rise_completed = np.zeros(num_envs, dtype=bool)  # 中间里程碑锁存
        self._prev_base_z = np.zeros(num_envs, dtype=np.float64)  # 上一帧 base_z (rise_vel 用)
        self._constraint_costs = np.zeros((num_envs, 2), dtype=get_global_dtype())  # [F_mag, T_mag]
        self._robot_mass = 18.65
        # ★ v8 锚点站立: 恢复锁存上升沿锁存站立点 (位置+yaw), 之后罚净漂移/净旋转
        self._anchor_xy = np.zeros((num_envs, 2), dtype=np.float64)
        self._anchor_yaw = np.zeros(num_envs, dtype=np.float64)
        self._anchor_latched = np.zeros(num_envs, dtype=bool)
        # ★ step_counter 是批次步 (np_env.step 每次所有 env 同步 +1, 每 iter 共 24 次),
        #   不是 env 步数 — 若乘 num_envs 力将永不衰减 (需 num_envs× 倍 iter).
        #   全局撤除步数 = iters × 每 iter 批次数 (24)
        self._force_end_steps = int(getattr(self._jump_cfg, "force_end_iters", 3000) * 24)
        # base_link body id (冷路径解析)
        self._base_body_id = self._resolve_base_body_id()

    def _resolve_base_body_id(self) -> int:
        try:
            import mujoco as _mj

            if hasattr(self._backend, "_model"):
                return int(
                    _mj.mj_name2id(
                        self._backend._model,
                        _mj.mjtObj.mjOBJ_BODY,
                        "base_link",  # type: ignore[union-attr]
                    )
                )
        except Exception:
            pass
        return -1

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # 基础 297/324 (commands[4] = h_cmd), 无额外特征 — 热启动自 walk 零填充
        return {"obs": 297, "critic": 324}

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, object] = {
            "orient": _reward_orient,
            "base_height": _reward_base_height,
            "upright": _reward_upright,
            "recover_complete": _reward_recover_complete,
            "rise": _reward_rise,
            "rise_vel": _reward_rise_vel,
            "settle": _reward_settle,
            "stand_still": _reward_stand_still,
            "no_yaw": _reward_no_yaw,
            "wheel_symmetry": _reward_wheel_symmetry,
            "wheel_speed": _reward_wheel_speed,
            "stand_anchor": _reward_stand_anchor,
            "act_rate": _reward_act_rate,
            "dof_pos": _reward_dof_pos,
            "stand_pose": _reward_stand_pose,
            "leg_bias": _reward_leg_bias,
            "no_fly": _reward_no_fly,
            "wheel_force": _reward_wheel_force,
            "alive": _reward_alive,
        }

    # ── reset: 硬重置阶段/空闲计时/约束代价 ──

    def _sync_prev_base_z(self, idx: np.ndarray) -> None:
        try:
            self._prev_base_z[idx] = np.asarray(
                self._backend.get_base_pos(), dtype=get_global_dtype()
            )[:, 2][idx]
        except Exception:
            pass

    def reset(self, env_indices):
        out = super().reset(env_indices)
        idx = np.asarray(env_indices, dtype=np.int64)
        self._idle_time[idx] = 0.0
        self._has_recovered[idx] = False
        self._recover_hold[idx] = 0.0
        self._recover_completed[idx] = False
        self._rise_hold[idx] = 0.0
        self._rise_completed[idx] = False
        self._constraint_costs[idx] = 0.0
        self._anchor_xy[idx] = 0.0
        self._anchor_yaw[idx] = 0.0
        self._anchor_latched[idx] = False
        self._sync_prev_base_z(idx)
        self._stage = 0
        return out

    def _reset_done_envs(self) -> None:
        assert self._state is not None
        done = self._state.terminated | self._state.truncated
        idx = np.flatnonzero(done).astype(np.int32)
        super()._reset_done_envs()
        self._idle_time[idx] = 0.0
        self._has_recovered[idx] = False
        self._recover_hold[idx] = 0.0
        self._recover_completed[idx] = False
        self._rise_hold[idx] = 0.0
        self._rise_completed[idx] = False
        self._constraint_costs[idx] = 0.0
        self._anchor_xy[idx] = 0.0
        self._anchor_yaw[idx] = 0.0
        self._anchor_latched[idx] = False
        self._sync_prev_base_z(idx)

    # ── step: 先施加力引导 wrench, 再走物理 ──

    def step(self, actions):
        self._apply_force_assist()
        return XqRobotWLWalkFlatEnv.step(self, actions)

    def _apply_force_assist(self) -> None:
        """论文 Eq.4: F (向上) + T (对齐直立), 与高度相关且全局衰减.

        F = (1-e^{-μ·(h_cmd-h)}) · sat(1 - steps/force_end) · F_max
        T = (1-e^{-μ·(h_cmd-h)}) · sat(...) · T_max · rot_vec(h→upright)
        约束代价 C = [|F|, |T|] → info (CPO 读取).
        """
        self._constraint_costs[:] = 0.0
        if not getattr(self._jump_cfg, "force_assist_enabled", True):
            return
        num = self._num_envs
        base_z = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:num, 2]
        gravity = np.asarray(
            self._backend.get_sensor_data(self._cfg.sensor.upvector), dtype=get_global_dtype()
        )[:num]
        # 力辅助始终指向完整站立目标 h_cmd2: 若随批次阶段(h_cmd1)走, 阶段门控在
        # 恢复扩散前卡死 → 力在 ~0.30m 处衰减, 机器人卡半撑. 全程目标站立才有力跨过起身坎.
        h_cmd = float(self._jump_cfg.h_cmd2)
        height_term = 1.0 - np.exp(-self._jump_cfg.mu * np.maximum(0.0, h_cmd - base_z))
        t_coeff = np.clip(1.0 - self.step_counter / max(self._force_end_steps, 1), 0.0, 1.0)
        strength = height_term * t_coeff

        # 力矩: 旋转向量 up→[0,0,1], 幅值=倾角 (裁剪到 1 rad)
        up = -gravity
        up_n = up / (np.linalg.norm(up, axis=1, keepdims=True) + 1e-8)
        target = np.array([0.0, 0.0, 1.0])
        cross = np.cross(up_n, target)
        dot = np.clip(np.sum(up_n * target, axis=1), -1.0, 1.0)
        angle = np.arccos(dot)
        axis_norm = np.linalg.norm(cross, axis=1)
        rot_vec = np.zeros((num, 3), dtype=get_global_dtype())
        nz = axis_norm > 1e-6
        rot_vec[nz] = (cross[nz] / axis_norm[nz, None]) * np.minimum(angle[nz], 1.0)[:, None]

        F = self._jump_cfg.force_max * strength
        T = self._jump_cfg.torque_max * strength[:, None] * rot_vec

        wrench = np.zeros((num, 1, 6), dtype=get_global_dtype())
        wrench[:, 0, 2] = F
        wrench[:, 0, 3:6] = T
        if self._base_body_id >= 0:
            try:
                self._backend.apply_body_wrench(
                    np.array([self._base_body_id], dtype=np.int32), wrench
                )
            except NotImplementedError:
                pass
        # 归一化到 [0,1] (避免 cost critic 预测大数值发散)
        self._constraint_costs[:, 0] = F / max(self._jump_cfg.force_max, 1e-6)
        self._constraint_costs[:, 1] = np.linalg.norm(T, axis=1) / max(
            self._jump_cfg.torque_max, 1e-6
        )

    # ── 状态更新 ──

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        self._total_env_steps += self._num_envs

        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()
        base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[
            : self._num_envs
        ]
        base_z = base_pos[:, 2]

        # 阶段 (批次高度统计): |S1| > 2/3 N → rs — 仅切换次要奖励权重集.
        # 高度命令/奖励不随阶段走 (恒 h_cmd2), 否则阶段门控卡死会连高度目标一起卡死.
        n1 = float(np.mean(base_z > self._jump_cfg.h_cmd1))
        self._stage = int(n1 > self._jump_cfg.stage_fraction)
        h_cmd = float(self._jump_cfg.h_cmd2)
        state.info["stage"] = self._stage
        state.info["h_cmd"] = h_cmd
        # 自然站姿 (dof_pos 目标) — walk 模型实测
        state.info["standing_angles"] = np.asarray(
            self._jump_cfg.standing_angles, dtype=get_global_dtype()
        )
        # 命令: 恢复期只保持, height 命令 = 站立目标
        state.info["commands"][:, :4] = 0.0
        state.info["commands"][:, 4] = h_cmd

        # 约束代价 (本步施加的 F/T 幅值) → CPO
        state.info["constraint_costs"] = self._constraint_costs.copy()

        self._update_wheel_contact(state.info)
        state.info["wheel_force_total"] = self._wheel_force_total()
        # 轮子速度 (索引 6,7): 差速转圈/空转惩罚用 (诊断发现站立时右轮 -565 rad/s 狂转)
        state.info["wheel_vel"] = dof_vel[:, NUM_LEG_ACTIONS : NUM_LEG_ACTIONS + 2].copy()

        # 上升速度奖励信号: 向上 base_z 速度, 双轮着地时生效 (防跳起刷分),
        # 且仅在起身过渡期 (base_z < rise_vel_height_cap) — 到站立高度后不再奖推, 防顶部跳动
        # ★ v8.1: 乘躯干直立门控 clip(up/rise_vel_up_gate,0,1) — 堵"桥式半撑"局部最优
        #   (v8 诊断: 策略撑起骨盆到 0.30m 但躯干躺平 up≈0, 靠 rise_vel 无限刷分,
        #   永远够不到 rise 里程碑 0.35m+直立0.80, 4000 iter 从未恢复)。门控后躺着推=0,
        #   只有边转正边推才计分, 逼策略学会"边转正边自举".
        dbz = base_z - self._prev_base_z
        self._prev_base_z = base_z.copy()
        vz = dbz / self._cfg.ctrl_dt
        wheel_on_mask = (
            np.min(state.info.get("wheel_contact", np.ones((self._num_envs, 2))), axis=1) > 0.5
        )
        up_gate = np.clip(
            gravity[:, 2] / getattr(self._jump_cfg, "rise_vel_up_gate", 0.35), 0.0, 1.0
        )
        state.info["rise_vel"] = (
            np.clip(vz, 0.0, 1.0)
            * wheel_on_mask
            * up_gate
            * (base_z < getattr(self._jump_cfg, "rise_vel_height_cap", 0.45))
        ).astype(get_global_dtype())
        # settle 用: 垂直速度绝对值 + 水平速度 (教"站稳不动")
        state.info["abs_rise_vel"] = np.abs(vz).astype(get_global_dtype())

        # 空闲贴地计时 + 恢复锁存
        lying = base_z < _IDLE_Z
        self._idle_time[lying] += self._cfg.ctrl_dt
        self._idle_time[~lying] = 0.0
        self._has_recovered |= base_z > self._jump_cfg.h_cmd1
        # 中间里程碑: base_z>rise_height + 直立>0.80 + 双轮着地 保持 rise_hold 锁存
        wheel_on = (
            np.min(state.info.get("wheel_contact", np.ones((self._num_envs, 2))), axis=1) > 0.5
        )
        rising = (base_z > self._jump_cfg.rise_height) & (gravity[:, 2] > 0.80) & wheel_on
        self._rise_hold[rising] += self._cfg.ctrl_dt
        self._rise_hold[~rising] = 0.0
        self._rise_completed |= self._rise_hold >= self._jump_cfg.rise_hold
        # 恢复完成: 站立 (base_z>recover_height + 直立>0.85 + 双轮着地) 连续保持 0.5s 锁存
        standing = (base_z > self._jump_cfg.recover_height) & (gravity[:, 2] > 0.85) & wheel_on
        self._recover_hold[standing] += self._cfg.ctrl_dt
        self._recover_hold[~standing] = 0.0
        self._recover_completed |= self._recover_hold >= 0.5
        # ★ v8 锚点站立: 恢复锁存上升沿锁存站立点 (base_xy + yaw), 之后罚净漂移/净旋转
        base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())[
            : self._num_envs
        ]
        _qw, _qx, _qy, _qz = base_quat[:, 0], base_quat[:, 1], base_quat[:, 2], base_quat[:, 3]
        base_yaw = np.arctan2(2.0 * (_qw * _qz + _qx * _qy), 1.0 - 2.0 * (_qy * _qy + _qz * _qz))
        just = self._recover_completed & ~self._anchor_latched
        self._anchor_xy[just] = base_pos[just, :2]
        self._anchor_yaw[just] = base_yaw[just]
        self._anchor_latched[just] = True
        state.info["anchor_active"] = self._anchor_latched.copy()
        state.info["anchor_dx"] = (base_pos[:, 0] - self._anchor_xy[:, 0]).astype(
            get_global_dtype()
        )
        state.info["anchor_dy"] = (base_pos[:, 1] - self._anchor_xy[:, 1]).astype(
            get_global_dtype()
        )
        state.info["anchor_dyaw"] = np.arctan2(
            np.sin(base_yaw - self._anchor_yaw), np.cos(base_yaw - self._anchor_yaw)
        ).astype(get_global_dtype())
        state.info["stand_anchor_sigma_xy"] = getattr(self._jump_cfg, "stand_anchor_sigma_xy", 0.25)
        state.info["stand_anchor_sigma_yaw"] = getattr(
            self._jump_cfg, "stand_anchor_sigma_yaw", 0.35
        )
        state.info["no_yaw_deadzone"] = getattr(self._jump_cfg, "no_yaw_deadzone", 1.0)
        state.info["idle_time"] = self._idle_time.copy()
        state.info["has_recovered"] = self._has_recovered.copy()
        state.info["recover_completed"] = self._recover_completed.copy()
        state.info["rise_completed"] = self._rise_completed.copy()

        terminated = self._compute_terminated(gravity, dof_pos)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _update_wheel_contact(self, info: dict) -> None:
        try:
            left = self._backend.get_sensor_data("left_wheel_force")
            right = self._backend.get_sensor_data("right_wheel_force")
            lf = np.asarray(left, dtype=get_global_dtype()).reshape(-1, 3)[: self._num_envs]
            rf = np.asarray(right, dtype=get_global_dtype()).reshape(-1, 3)[: self._num_envs]
            info["wheel_contact"] = np.stack(
                [
                    (np.linalg.norm(lf, axis=1) > 10.0).astype(np.float64),
                    (np.linalg.norm(rf, axis=1) > 10.0).astype(np.float64),
                ],
                axis=1,
            )
        except (KeyError, AttributeError):
            info["wheel_contact"] = np.zeros((self._num_envs, 2), dtype=np.float64)

    def _wheel_force_total(self) -> np.ndarray:
        try:
            left = np.asarray(
                self._backend.get_sensor_data("left_wheel_force"), dtype=get_global_dtype()
            ).reshape(-1, 3)[: self._num_envs]
            right = np.asarray(
                self._backend.get_sensor_data("right_wheel_force"), dtype=get_global_dtype()
            ).reshape(-1, 3)[: self._num_envs]
            return np.linalg.norm(left, axis=1) + np.linalg.norm(right, axis=1)
        except (KeyError, AttributeError):
            return np.zeros(self._num_envs, dtype=get_global_dtype())

    def _compute_terminated(self, gravity: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        """终止 (分阶段): 恢复前(未达 h_cmd1)不按倾覆终止 — 倒地是合法起始态;
        恢复后(has_recovered) 倾覆/塌缩才终止. 贴地超时(死点)始终终止."""
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        max_tilt = np.deg2rad(self._jump_cfg.max_tilt_deg)
        base_z = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        terminated = self._idle_time > self._jump_cfg.idle_ground_time
        # 恢复后再倒: 倾覆或塌缩 (倒地态由 idle 处理, 不立即杀)
        fell = (tilt > max_tilt) | (base_z < self._jump_cfg.min_base_height)
        terminated |= fell & self._has_recovered & (base_z > _IDLE_Z)
        thigh_collapsed = (dof_pos[:, 1] < -1.0) | (dof_pos[:, 4] > 1.0)
        calf_extreme = (np.abs(dof_pos[:, 2]) > 0.95) | (np.abs(dof_pos[:, 5]) > 0.95)
        terminated |= (thigh_collapsed | calf_extreme) & self._has_recovered
        return terminated

    def _compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        # 基础 297/324, 跳过 jump_srl 的 FSM 追加
        return XqRobotWLWalkFlatEnv._compute_obs(
            self, info, linvel, gyro, gravity, dof_pos, dof_vel
        )

    def _compute_reward(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        dtype = get_global_dtype()
        num_obs = linvel.shape[0]
        h_cmd = info.get("h_cmd", self._jump_cfg.h_cmd1)
        scales = [self._jump_cfg.scales_ru, self._jump_cfg.scales_rs][self._stage]
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos[:, :NUM_LEG_ACTIONS],
            dof_vel=dof_vel[:, :NUM_LEG_ACTIONS],
            num_envs=num_obs,
            default_angles=DEFAULT_LEG_ANGLES.astype(dtype),
            tracking_sigma=self._jump_cfg.tracking_sigma,
            base_height_target=h_cmd,
            base_height=np.asarray(self._backend.get_base_pos(), dtype=dtype)[:, 2][:num_obs],
            gravity=gravity,
            joint_range=None,
        )
        ctx.info.setdefault("robot_mass", self._robot_mass)
        return rewards.run_reward_dispatch(
            scales=scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._jump_cfg.only_positive_rewards,
        )
