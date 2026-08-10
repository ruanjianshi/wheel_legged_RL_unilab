"""xqrobotwl 单腿平衡(单轮支撑)环境 — FSM前馈过渡 + PPO 横滚平衡

基于 P1 验证 (tools/xqrobotwl/single_leg_balance_feasibility.py) 固化的
4 状态 FSM:
  -1站立(两轮) → 0折腿(FF收膝) → 1单轮平衡(RL核心) → 2落腿(FF) → 回 -1

与后空翻的核心区别:
  - 后空翻是弹道(开环可脚本化), 单腿平衡的**保持**是倒立摆调节(必须闭环反馈)
  - 故 FF 只做可脚本化的**折腿/落腿过渡**(状态0/2, 策略屏蔽为0), RL 专注状态1
    的横滚主动平衡 — 这才是最难的自由度
  - P1 结论: 折腿机制用**收膝**(L_knee→0.87, L_hip_pitch→0.30), 不用髋外展
    (髋外展把 CoM 甩离支撑轮, 过渡即倒)

核心设计:
  1. FSM 前馈 = 收膝折腿目标渐变 (状态0/2), 支撑腿保持默认
  2. 平衡奖励: balance_upright 奖机身对齐平衡倾角(roll_ref=-28°), 非直立
  3. 自由轮离地奖励: wheel_off (防作弊: 靠支撑轮撑)
  4. 折腿姿态奖励: fold_pose (自由腿贴收膝目标)
  5. balance_complete 单轮平衡达标志 (周期级锁存, 一次性大奖)
  6. 状态1横滚倾角终止放宽 (平衡位本身侧倾 28°)

Joint order (policy): [L_hip_roll, L_hip_pitch, L_knee, R_hip_roll, R_hip_pitch, R_knee, L_wheel, R_wheel]
支撑侧: 默认右腿支撑, 左腿折叠 (P1 验证配置)
"""

from __future__ import annotations

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
    NUM_LEG_ACTIONS,
    NUM_WHEEL_ACTIONS,
)
from unilab.envs.locomotion.xqrobotwl.joystick import XqRobotWLWalkFlatEnv
from unilab.envs.locomotion.xqrobotwl.jump_srl import (
    XqRobotWLJumpDRProvider,
    XqRobotWLJumpSRLFlatCfg,
    XqRobotWLJumpSRLFlatEnv,
)

# ── P1 验证参数 ──
WHEEL_R = 0.065
DEFAULT_LEGS = np.array([0.1, 0.15, 0.15, -0.1, -0.15, -0.15], dtype=np.float64)

# FSM 相位时长 (s)
_FSM_DUR = {0: 1.00, 2: 1.00}  # 折腿/落腿过渡 (慢: 给 RL 时间把重量移到支撑腿)

# 自由腿姿态 (用户决定 2026-08-06): 30°侧压 + 自由腿展开当配重
# 机身侧压 30° 使 CoM 精确落支撑轮 (|dy|=0.000 验证), 自由腿 L_hip_roll 展开
# 当配重 (像走钢丝者持杆) — RL 用它持续微调 CoM, 这是动态平衡的关键通道。
# 自由腿 pitch/knee 保持自然微屈 (不深折), L_hip_roll 交给 RL 调节。
_FOLD_KNEE = 0.30
_FOLD_PITCH = 0.10
# 自由腿配重初始摆位 (L_hip_roll, 展开当配重); RL 在平衡期可自由调节 ±
_FREE_LEG_ROLL_INIT = -0.5
# 平衡倾角: 30° 侧压 (P1 验证甜点位, CoM 精确落支撑轮)。up_y=sin30°=+0.5 向支撑轮。
_ROLL_REF_RAD = np.deg2rad(30.0)


def _compute_feedforward(fsm_state, fsm_timer, action_scale: float) -> np.ndarray:
    """按 FSM 状态取前馈 (policy 空间, envs×8) — 收膝折腿 + 支撑腿伸直渐变

    自由腿(左) 膝/髋前倾折腿; 支撑腿(右) 髋/膝伸直 (P1 平衡位: R 腿伸直撑住机身,
    否则折腿时 base 会 squat 塌缩到 z<0.40, balance_complete 的 height_ok 永不满足)。
    这是纯几何目标, 可脚本化 — RL 负责支撑腿 hip_roll(横滚移轮) + 轮子(俯仰)。
    状态0 折腿: knee 0.15→0.60, pitch 0.15→0.20 + R 伸直 (1.0s)
    状态1 保持: 钉住折腿+伸直目标 (RL 平衡期不能塌回去)
    状态2 落腿: 反向渐变回默认
    换算: 目标 T = ff*scale*flip + default → ff = (T-default)/(flip*scale)
    L_knee flip=-1, L_hip_pitch flip=+1, R_hip_pitch flip=-1, R_knee flip=+1。
    """
    ff = np.zeros((fsm_state.shape[0], 8), dtype=np.float64)
    for s, ramp in {0: _FSM_DUR[0], 1: _FSM_DUR[0], 2: _FSM_DUR[2]}.items():
        m = fsm_state == s
        if not m.any():
            continue
        if s == 2:
            r = np.clip(fsm_timer[m] / ramp, 0.0, 1.0)
            l_knee_t = _FOLD_KNEE + (0.15 - _FOLD_KNEE) * r
            l_pitch_t = _FOLD_PITCH + (0.15 - _FOLD_PITCH) * r
            r_pitch_t = 0.0 + (-0.15 - 0.0) * r
            r_knee_t = 0.0 + (-0.15 - 0.0) * r
        else:  # 0/1: 折腿 + 支撑腿伸直 + 保持 (1 直接钉满)
            r = 1.0 if s == 1 else np.clip(fsm_timer[m] / ramp, 0.0, 1.0)
            l_knee_t = 0.15 + (_FOLD_KNEE - 0.15) * r
            l_pitch_t = 0.15 + (_FOLD_PITCH - 0.15) * r
            r_pitch_t = -0.15 + (0.0 + 0.15) * r
            r_knee_t = -0.15 + (0.0 + 0.15) * r
        ff[m, 1] = (l_pitch_t - 0.15) / action_scale  # L_hip_pitch, flip=+1
        ff[m, 2] = (l_knee_t - 0.15) / (-action_scale)  # L_knee, flip=-1
        ff[m, 4] = (r_pitch_t - (-0.15)) / (-action_scale)  # R_hip_pitch, flip=-1
        ff[m, 5] = (r_knee_t - (-0.15)) / (action_scale)  # R_knee, flip=+1
    return ff


def _update_fsm(
    fsm_state: np.ndarray,
    fsm_timer: np.ndarray,
    sl_trigger: np.ndarray,
    balance_done: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """时间驱动 4 状态 FSM 转移"""
    fsm_timer += dt
    for s in (-1, 0, 1, 2):
        m = fsm_state == s
        if not m.any():
            continue
        if s == -1:
            t = (sl_trigger[m] > 0.5) & (fsm_timer[m] > 0.05)
            nxt_state = 0
        elif s == 0:
            t = fsm_timer[m] > _FSM_DUR[0]
            nxt_state = 1
        elif s == 1:
            # 触发结束 或 本轮平衡完成 → 落腿
            t = (sl_trigger[m] < 0.5) | balance_done[m]
            nxt_state = 2
        else:  # s == 2
            t = fsm_timer[m] > _FSM_DUR[2]
            nxt_state = -1
        nxt = np.zeros_like(fsm_state, dtype=bool)
        nxt[m] = t
        fsm_state[nxt] = nxt_state
        fsm_timer[nxt] = 0.0
    return fsm_state, fsm_timer


# ========== 奖励函数 (相位门控) ==========


def _reward_balance_upright(ctx: RewardContext) -> np.ndarray:
    """单轮平衡核心奖励: 机身对齐平衡方向 (roll 向支撑轮 +28° 且 pitch 直立), 状态 0/1/2

    平衡位: up 向量应对齐 up_ref=(0, sin28°, cos28°) — roll 侧倾 28° 使 CoM 投影落
    支撑轮 (不能直立, 直立 CoM 在两轮间单轮撑不住), 同时 pitch 保持直立。
    用 dot 的锐化曲线: dot>0.85 (≈32° 内) 线性爬升到 1, 低于 0.85 归零 —
    pitch 和 roll 任一偏太多都罚, 梯度陡。状态0/2(折腿/落腿过渡)也激活:
    过渡中就要对齐平衡方向, 否则折腿时 CoM 前移 → 俯冲 (实测 pitch 冲到 -58° 摔死)。
    """
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [0, 1, 2]).astype(np.float64)
    up_ref = np.array([0.0, np.sin(_ROLL_REF_RAD), np.cos(_ROLL_REF_RAD)])
    g = np.asarray(ctx.gravity, dtype=get_global_dtype())
    dot = g @ up_ref
    # 锐化: dot=1 (对齐) → 1; dot=0.883 (直立 roll=0) → 0.22; dot≤0.85 → 0
    r = np.clip((dot - 0.85) / 0.15, 0.0, 1.0)
    return r * active


def _reward_wheel_off(ctx: RewardContext) -> np.ndarray:
    """自由轮(左)必须离地: 状态 0/1/2 奖着地=0 离地=1

    防作弊: 策略靠自由轮偷撑就不给分。左轮 idx0, 右轮 idx1。
    """
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [0, 1, 2]).astype(np.float64)
    contact = ctx.info.get("wheel_contact", np.zeros((ctx.num_envs, 2)))
    free = contact[:, 0]
    return (1.0 - free) * active


def _reward_fold_pose(ctx: RewardContext) -> np.ndarray:
    """姿态保持: 状态 0/1/2 罚支撑腿膝/髋偏离伸直 + 自由腿微屈偏离

    支撑腿伸直撑住机身 (否则 base 塌缩 z<0.40), 自由腿 pitch/knee 保持自然微屈
    (配重靠 L_hip_roll, 不深折)。dof_pos 前6 = 腿关节。
    """
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [0, 1, 2]).astype(np.float64)
    # 支撑腿(右) pitch/knee 应伸直(≈0): idx 4=R_pitch 5=R_knee
    stance_err = np.square(ctx.dof_pos[:, 4]) + np.square(ctx.dof_pos[:, 5])
    # 自由腿(左) pitch/knee 应微屈(≈_FOLD_PITCH/_FOLD_KNEE): idx 1/2
    free_err = np.square(ctx.dof_pos[:, 1] - _FOLD_PITCH) + np.square(
        ctx.dof_pos[:, 2] - _FOLD_KNEE
    )
    return -(stance_err * 3.0 + free_err) * active


def _reward_balance_complete(ctx: RewardContext) -> np.ndarray:
    """单轮平衡达成(周期级锁存): 一次性大奖"""
    done = ctx.info.get("balance_completed", np.zeros(ctx.num_envs, dtype=bool))
    return done.astype(np.float64)


def _reward_stance_height(ctx: RewardContext) -> np.ndarray:
    """支撑高度: 状态 1 罚 base_z 偏离目标"""
    fsm = ctx.info["fsm_state"]
    active = (fsm == 1).astype(np.float64)
    return -np.square(ctx.base_height - ctx.base_height_target) * active


def _reward_roll_rate(ctx: RewardContext) -> np.ndarray:
    """角速度阻尼: 状态 0/1/2 罚机身角速度 (gyro 前两分量 = roll+pitch 速率)

    倒立摆保持需要主动阻尼震荡 — 只奖对齐不够, 还得罚抖/晃, 策略才学会平稳持住。
    gyro 是机身系角速度, 前两分量对应绕 x/y (roll/pitch 速率)。
    返回正误差 (rate²), config scale 用负值 → 惩罚 (对齐 joint_action_rate 惯例)。
    """
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [0, 1, 2]).astype(np.float64)
    gyro = np.asarray(ctx.gyro, dtype=get_global_dtype())
    rate = gyro[:, 0] ** 2 + gyro[:, 1] ** 2
    return rate * active


def _reward_counterweight(ctx: RewardContext) -> np.ndarray:
    """配重范围: 状态 0/1/2 自由腿 L_hip_roll 保持在有效配重区间给分

    30° 侧压姿态下, 自由腿展开当配重调节 CoM。L_hip_roll 需在展开区间
    (负=向内压向支撑轮侧, 正=向外) 内才能有效移 CoM。超出区间(甩飞/收死)
    配重失效 → 不给分。dof_pos idx0 = L_hip_roll。
    """
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [0, 1, 2]).astype(np.float64)
    l_roll = ctx.dof_pos[:, 0]
    # 有效配重区间: -1.2 ~ +0.5 (展开能移 CoM 的范围), 中心 ≈ -0.5
    in_range = (l_roll > -1.2) & (l_roll < 0.5)
    r = in_range.astype(np.float64) * np.exp(-0.5 * np.square(l_roll + 0.5))
    return r * active


def _reward_stand_balance(ctx: RewardContext) -> np.ndarray:
    """站住不倒: 状态 -1/2 温和奖励直立 (up=cos tilt)"""
    assert ctx.gravity is not None
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [-1, 2]).astype(np.float64)
    up = np.clip(ctx.gravity[:, 2], 0.0, 1.0)
    return up * active


def _reward_stand_height(ctx: RewardContext) -> np.ndarray:
    """不倒: 状态 -1/2 奖励高度>0.25 (轮子撑住不折叠)"""
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [-1, 2]).astype(np.float64)
    r = np.clip((ctx.base_height - 0.25) / 0.2, 0.0, 1.0)
    return r * active


def _reward_action_magnitude(ctx: RewardContext) -> np.ndarray:
    return np.sum(np.square(ctx.info["current_actions"]), axis=1)


def _reward_leg_mirror(ctx: RewardContext) -> np.ndarray:
    hip_error = np.abs(ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3])
    pitch_error = np.abs(ctx.dof_pos[:, 1] + ctx.dof_pos[:, 4])
    knee_error = np.abs(ctx.dof_pos[:, 2] - ctx.dof_pos[:, 5])
    return hip_error + pitch_error + knee_error


# ========== 配置 ==========


@dataclass
class XqRobotWLSingleLegCommands(Commands):
    """5D 命令: [vx, vy, vyaw, tsk, sl_trigger]"""

    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.3, 0.0, -0.5, -0.1, 0], [0.3, 0.0, 0.5, 0.1, 1]]
    )
    resampling_time: float = 4.0


@dataclass
class XqRobotWLSingleLegRewardConfig:
    scales: dict[str, float]
    only_positive_rewards: bool = False
    tracking_sigma: float = 0.3
    base_height_target: float = 0.55
    max_tilt_deg: float = 55.0
    min_base_height: float = 0.20
    sl_trigger_prob: float = 0.7
    sl_warmup_iters: int = 100
    balance_hold_time: float = 0.5  # 单轮平衡保持时长 (s) 才算完成
    start_in_balance: bool = False  # True=直接从单轮平衡位起步(跳过渡, 先学保持)


@registry.envcfg("XqRobotWLSingleLegFlat")
@dataclass
class XqRobotWLSingleLegFlatCfg(XqRobotWLJumpSRLFlatCfg):
    commands: XqRobotWLSingleLegCommands = field(default_factory=XqRobotWLSingleLegCommands)  # type: ignore[assignment]
    reward_config: XqRobotWLSingleLegRewardConfig | None = None  # type: ignore[assignment]
    max_episode_seconds: float = 8.0


class XqRobotWLSingleLegDRProvider(XqRobotWLJumpDRProvider):
    """采样命令: 站立/单轮平衡混合 (sl_trigger∈{0,1})

    start_in_balance=True: 直接从单轮平衡位起步 (跳过渡段, 先学"保持"),
    此时 reset 姿态 = P1 折叠平衡位, sl_trigger 恒 1。
    """

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        cmds = np.zeros((num_reset, 5), dtype=get_global_dtype())
        if getattr(env._cfg.reward_config, "start_in_balance", False):
            cmds[:, 4] = 1.0  # 恒触发单轮平衡
            return cmds
        prob = getattr(env._cfg.reward_config, "sl_trigger_prob", 0.7)
        sl = np.random.uniform(0.0, 1.0, num_reset) < prob
        cmds[:, 4] = sl.astype(get_global_dtype())
        return cmds

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        if not getattr(env._cfg.reward_config, "start_in_balance", False):
            return super().build_reset_plan(env, env_ids)
        # 直接置为 30° 侧压平衡位: base 侧倾 -30° (up_y=+0.5 向支撑轮, CoM 精确落轮),
        # 自由腿 L_hip_roll 展开当配重, 支撑腿伸直。qpos 顺序: [x,y,z, qw,qx,qy,qz,
        # L_roll,L_pitch,L_knee,L_wheel, R_roll,R_pitch,R_knee,R_wheel]
        num_reset = len(env_ids)
        lean = np.radians(-30.0)
        c, s = np.cos(lean / 2), np.sin(lean / 2)
        qpos = np.tile(
            np.array(
                [
                    0.0,
                    0.0,
                    0.55,
                    c,
                    s,
                    0.0,
                    0.0,
                    _FREE_LEG_ROLL_INIT,
                    _FOLD_PITCH,
                    _FOLD_KNEE,
                    0.0,
                    -0.1,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=get_global_dtype(),
            ),
            (num_reset, 1),
        )
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


@registry.env("XqRobotWLSingleLegFlat", sim_backend="mujoco")
class XqRobotWLSingleLegFlatEnv(XqRobotWLJumpSRLFlatEnv):
    """xqrobotwl 单腿平衡环境 — FSM前馈过渡 + balance_upright + 相位门控奖励"""

    _cfg: XqRobotWLSingleLegFlatCfg
    _jump_cfg: XqRobotWLSingleLegRewardConfig  # type: ignore[assignment]  # 收窄基类奖励配置类型

    def __init__(self, cfg: XqRobotWLSingleLegFlatCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        self._jump_cfg = cfg.reward_config
        self._total_env_steps = 0
        self._warmup_progress = 0.0
        self._sl_warmup_env_steps = cfg.reward_config.sl_warmup_iters * 24 * num_envs
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._dr_manager._provider = XqRobotWLSingleLegDRProvider()  # type: ignore[union-attr]
        # 单腿平衡专用状态
        self._fsm_state = -np.ones(num_envs, dtype=np.int32)
        self._fsm_timer = np.zeros(num_envs, dtype=np.float64)
        self._balance_hold = np.zeros(num_envs, dtype=np.float64)  # 状态1连续保持时长
        self._balance_completed = np.zeros(num_envs, dtype=bool)  # 本轮平衡完成锁存
        self._policy_actions = np.zeros((num_envs, 8), dtype=np.float64)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # base 297/324 + 额外 4×9=36 (fsm_state, timer/0.8, wheel_contact×2)
        return {"obs": 297 + 36, "critic": 324 + 36}

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, object] = {
            "balance_upright": _reward_balance_upright,
            "wheel_off": _reward_wheel_off,
            "fold_pose": _reward_fold_pose,
            "counterweight": _reward_counterweight,
            "balance_complete": _reward_balance_complete,
            "stance_height": _reward_stance_height,
            "roll_rate": _reward_roll_rate,
            "stand_balance": _reward_stand_balance,
            "stand_height": _reward_stand_height,
            "joint_action_rate": self._reward_joint_action_rate,
            "wheel_action_rate": self._reward_wheel_action_rate,
            "action_magnitude": _reward_action_magnitude,
            "leg_mirror": _reward_leg_mirror,
            "alive": rewards.alive,
        }

    def _reward_joint_action_rate(self, ctx: RewardContext) -> np.ndarray:
        cur = ctx.info["current_actions"][:, :NUM_LEG_ACTIONS]
        lst = ctx.info["last_actions"][:, :NUM_LEG_ACTIONS]
        return np.sum(np.square(cur - lst), axis=1)

    def _reward_wheel_action_rate(self, ctx: RewardContext) -> np.ndarray:
        cur = ctx.info["current_actions"][:, NUM_LEG_ACTIONS:]
        lst = ctx.info["last_actions"][:, NUM_LEG_ACTIONS:]
        return np.sum(np.square(cur - lst), axis=1)

    # ── reset: 硬重置 FSM/平衡状态 ──

    def _reset_done_envs(self) -> None:
        assert self._state is not None
        done = self._state.terminated | self._state.truncated
        idx = np.flatnonzero(done).astype(np.int32)
        super()._reset_done_envs()
        # start_in_balance: 直接从单轮平衡位起步 → FSM 置状态 1
        if getattr(self._jump_cfg, "start_in_balance", False):
            self._fsm_state[idx] = 1
        else:
            self._fsm_state[idx] = -1
        self._fsm_timer[idx] = 0.0
        self._balance_hold[idx] = 0.0
        self._balance_completed[idx] = False

    # ── step: FSM 前馈 + PPO 反馈融合 ──

    def step(self, actions):
        ff = _compute_feedforward(
            self._fsm_state,
            self._fsm_timer,
            self._cfg.control_config.action_scale,
        )
        # ★ 确定性过渡 (用户决定 2026-08-06): 平衡期(FSM 0/1/2) 自由腿 pitch/knee(1-2)
        # 和支撑腿 pitch/knee(4-5) 由 ff 钉住 (自由腿微屈保持自然 + 支撑腿伸直撑住机身),
        # 策略屏蔽 — 否则位置执行器会把姿态拉回默认。
        # **自由腿 hip_roll(0) 策略自由当配重**: 30° 侧压姿态下, 自由腿 L_hip_roll 展开
        # 作为可动配重, RL 持续微调移 CoM (像走钢丝者持杆) — 这是动态平衡的关键通道。
        # 支撑腿 hip_roll(3) + 轮子(6-7) 也策略自由 (横滚/俯仰闭环)。
        fold_mask = np.isin(self._fsm_state, [0, 1, 2]).astype(np.float64)[:, None]
        rl_mask = np.ones_like(actions)
        rl_mask[:, 1:3] *= 1.0 - fold_mask  # 自由腿 pitch/knee(1-2)钉住
        rl_mask[:, 4:6] *= 1.0 - fold_mask  # 支撑腿 pitch/knee(4-5)钉住
        actions_masked = actions * rl_mask
        fused = ff + actions_masked
        self._policy_actions = actions_masked
        return XqRobotWLWalkFlatEnv.step(self, fused)

    # ── 状态更新 ──

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)

        # 预热: 前 N 轮纯站立, 之后斜坡放开 sl_trigger
        self._total_env_steps += self._num_envs
        if self._sl_warmup_env_steps <= 0:
            self._warmup_progress = 1.0
        else:
            self._warmup_progress = np.clip(
                (self._total_env_steps - self._sl_warmup_env_steps) / self._sl_warmup_env_steps,
                0.0,
                1.0,
            )

        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        # 平衡保持计时: 状态1中 对齐平衡倾角 + 自由轮离地 + 高度达标 → 累计
        self._update_wheel_contact(state.info)
        in_balance = self._fsm_state == 1
        up_ref = np.array([0.0, np.sin(_ROLL_REF_RAD), np.cos(_ROLL_REF_RAD)])
        dot = gravity[: self._num_envs] @ up_ref
        # 平衡保持判据 (先放宽让大奖可触发, 之后课程收紧):
        # - 对齐平衡倾角: dot>0.88 (偏差 <28°), 倒立摆必然震荡, 0.93 太严
        # - 自由轮离地 + 高度>0.40 (折腿平衡位 base_z≈0.45-0.55)
        upright = dot > 0.88
        free_off = state.info["wheel_contact"][:, 0] < 0.5
        base_z = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        height_ok = base_z > 0.40
        held_ok = upright & free_off & height_ok
        self._balance_hold[in_balance & held_ok] += self._cfg.ctrl_dt
        self._balance_hold[in_balance & ~held_ok] = 0.0
        # 本轮平衡完成: 连续保持达标
        just_done = (
            self._balance_hold >= self._jump_cfg.balance_hold_time
        ) & ~self._balance_completed
        self._balance_completed |= just_done
        state.info["balance_completed"] = self._balance_completed.copy()

        # FSM 更新
        jt = state.info["commands"][:, 4] * self._warmup_progress
        self._fsm_state, self._fsm_timer = _update_fsm(
            self._fsm_state, self._fsm_timer, jt, self._balance_completed, self._cfg.ctrl_dt
        )
        state.info["fsm_state"] = self._fsm_state.copy()
        state.info["fsm_timer"] = self._fsm_timer.copy()
        state.info["wheel_vel"] = dof_vel[:, NUM_LEG_ACTIONS:]
        state.info["policy_actions"] = self._policy_actions.copy()

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

    def _compute_terminated(self, gravity: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        """终止 (分阶段): 状态1平衡位本身侧倾 28°, 倾角阈值放宽
        - 非平衡态(-1/0/2): tilt > max 持续 0.15s 终止
        - 平衡态(1): tilt > max+20° 持续 0.15s 终止 (给调节空间)
        - base_z < min 持续 0.3s
        - 关节极端"""
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        max_tilt = np.deg2rad(self._jump_cfg.max_tilt_deg) + (20.0 * (self._fsm_state == 1))
        base_z = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        sustained = self._fsm_timer > 0.15
        terminated = (base_z < self._jump_cfg.min_base_height) & sustained
        terminated |= (tilt > max_tilt) & sustained
        thigh_collapsed = (dof_pos[:, 1] < -1.0) | (dof_pos[:, 4] > 1.0)
        calf_extreme = (np.abs(dof_pos[:, 2]) > 0.95) | (np.abs(dof_pos[:, 5]) > 0.95)
        terminated |= thigh_collapsed | calf_extreme
        return terminated

    def _compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        base = XqRobotWLWalkFlatEnv._compute_obs(
            self, info, linvel, gyro, gravity, dof_pos, dof_vel
        )
        batch = linvel.shape[0]
        fsm_feat = self._fsm_state[:batch].astype(np.float64).reshape(-1, 1) / 4.0
        timer_feat = np.clip(self._fsm_timer[:batch].reshape(-1, 1) / 0.8, 0.0, 1.0)
        contact = info.get("wheel_contact", np.zeros((batch, 2), dtype=get_global_dtype()))[:batch]
        extra = np.tile(
            np.concatenate([fsm_feat, timer_feat, contact], axis=1, dtype=get_global_dtype())[
                :, None, :
            ],
            (1, self._hist_len, 1),
        ).reshape(batch, -1)
        base["obs"] = np.concatenate([base["obs"], extra], axis=1)
        base["critic"] = np.concatenate([base["critic"], extra], axis=1)
        return base

    def _compute_reward(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        dtype = get_global_dtype()
        num_obs = linvel.shape[0]
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos[:, :NUM_LEG_ACTIONS],
            dof_vel=dof_vel[:, :NUM_LEG_ACTIONS],
            num_envs=num_obs,
            default_angles=DEFAULT_LEGS.astype(dtype),
            tracking_sigma=self._jump_cfg.tracking_sigma,
            base_height_target=self._jump_cfg.base_height_target,
            base_height=np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2][
                :num_obs
            ],
            gravity=gravity,
            joint_range=None,
        )
        return rewards.run_reward_dispatch(
            scales=self._jump_cfg.scales,
            fns=self._reward_fns,
            ctx=ctx,
            info=info,
            enable_log=self._enable_reward_log,
            ctrl_dt=self._cfg.ctrl_dt,
            only_positive=self._jump_cfg.only_positive_rewards,
        )
