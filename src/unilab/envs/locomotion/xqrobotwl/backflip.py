"""xqrobotwl 后空翻环境 — FSM前馈引导 + PPO 强化

基于 P1 开环验证(tools/xqrobotwl/backflip_feasibility.py)固化的 7 状态 FSM:
  -1站立 → 0蹲 → 1蹬 → 2飞(收腿) → 3展 → 4缓冲 → 5恢复 → 回到 -1

核心设计:
  1. FSM 前馈 = P1 验证的 raw MuJoCo 目标反算到 policy 空间 (flip*ff*scale+default=raw)
     → ff 单独执行即复现已验证后空翻 (360°, 落地 3.8°), PPO 在此基础上强化
  2. flip_progress 追踪: up 向量在 XZ 平面无环绕俯仰角 (负=后翻方向)
  3. 相位门控奖励: 飞行期奖翻转进度, 落地期奖直立/轮地匹配, 站立期奖姿态
  4. 翻转专用终止: 飞行状态不按倾角终止
  5. reset 时硬重置 FSM/翻转状态 (一次性动作必须干净开始)

Joint order (policy): [L_hip_roll, L_hip_pitch, L_knee, R_hip_roll, R_hip_pitch, R_knee, L_wheel, R_wheel]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.base import registry
from unilab.base.np_env import NpEnvState
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

# ── 后空翻参数 (P1 验证) ──
WHEEL_R = 0.065
FLIP_TARGET = 2 * np.pi  # 360°
DEFAULT_LEGS = np.array([0.1, 0.15, 0.15, -0.1, -0.15, -0.15], dtype=np.float64)

# FSM 相位时长 (s), 与 P1 脚本一致
_FSM_DUR = {0: 0.30, 1: 0.25, 2: 0.45, 3: 0.18, 4: 0.15, 5: 0.40}

# ── 后空翻 FSM 前馈 (policy 空间) ──
# 直接复现 P1 开环序列 (获胜参数 devlog 01): W=0, crouch_hip=-0.25, launch_lean=0.60, tuck=0.70
# 关键: P1 是目标渐变 blend (launch 膝 0.07s 内 0.45→-0.87 猛伸, 髋先探 +0.10 再后仰 -0.60),
# 瞬时跳变复现不出后旋角动量 → 空中只转一半, 落地倒挂。这里是 P1 式每状态内 blend。
# 转换到 policy 空间: ff = (T - default) / (flip * scale)
# flip=[1,1,-1,1,-1,1], action_scale=0.6, default=[0.1,0.15,0.15,-0.1,-0.15,-0.15]
# 索引: 0 L_roll 1 L_pitch 2 L_knee 3 R_roll 4 R_pitch 5 R_knee 6 L_wheel 7 R_wheel
# 各相位 blend 时长 (s), 与 P1 脚本一致
_RAMP_S = {0: 0.08, 1: 0.07, 2: 0.15, 3: 0.12, 4: 0.08, 5: 0.20}

_FLIP6 = np.array([1, 1, -1, 1, -1, 1], dtype=np.float64)


def _compute_feedforward(
    fsm_state, fsm_timer, dof_pos, linvel, action_scale: float, wheel_scale: float
) -> np.ndarray:
    """按 FSM 状态取前馈 (policy 空间, envs×8) — P1 开环序列直接复现

    每状态内目标渐变 blend (对齐 P1), 轮子 W=0 不加速 (轮子加速=前翻, 禁用)。
    恢复态(5): 当前→默认 渐变 (避免瞬跳打断旋转)。
    """
    ff = np.zeros((fsm_state.shape[0], 8), dtype=np.float64)
    for s in range(0, 5):
        m = fsm_state == s
        if not m.any():
            continue
        r = np.clip(fsm_timer[m] / _RAMP_S[s], 0.0, 1.0)  # (batch,)
        if s == 0:  # 蹲: 髋后仰固定 -0.25, 膝 0.15→0.45
            hip = np.full_like(r, -0.25)
            knee = 0.15 + (0.45 - 0.15) * r
        elif s == 1:  # 蹬: 髋 +0.10→-0.60 (先探再后仰), 膝 0.45→-0.87 猛伸
            hip = 0.10 + (-0.60 - 0.10) * r
            knee = 0.45 + (-0.87 - 0.45) * r
        elif s == 2:  # 飞: 收腿 tuck 0→0.70, 髋 0.10
            hip = np.full_like(r, 0.10)
            knee = 0.0 + (0.70 - 0.0) * r
        elif s == 3:  # 展: 伸腿 0.70→0.10, 髋 0.30
            hip = np.full_like(r, 0.30)
            knee = 0.70 + (0.10 - 0.70) * r
        else:  # s == 4 缓冲: 弯膝 0.10→0.50, 髋 0.15
            hip = np.full_like(r, 0.15)
            knee = 0.10 + (0.50 - 0.10) * r
        legs = np.stack(
            [np.full_like(r, 0.1), hip, knee, np.full_like(r, -0.1), -hip, -knee], axis=1
        )
        ff[m, :6] = (legs - DEFAULT_LEGS) / (_FLIP6 * action_scale)
        # 轮子: crouch/launch/flight/land W=0 (P1 获胜 W=0); deploy 轮 [5,5] rad/s (落地滚到轮上)
        ff[m, 6:] = np.array([0.5, -0.5]) if s == 3 else 0.0
    # 恢复态(5): 当前→默认 渐变 (避免瞬跳打断旋转)
    m5 = fsm_state == 5
    if m5.any():
        r = np.clip(fsm_timer[m5] / _RAMP_S[5], 0.0, 1.0)[:, None]
        cur = dof_pos[m5, :6]
        # MuJoCo 目标 T = flip*ff*scale + default; 要 T=blend(cur,default,r)
        # → ff = (cur - default)*(1-r) / (flip*scale)
        ff[m5, :6] = (cur - DEFAULT_LEGS) * (1.0 - r) / (_FLIP6 * action_scale)
        ff[m5, 6:] = 0.0
    return ff


def _update_fsm(
    fsm_state: np.ndarray,
    fsm_timer: np.ndarray,
    flip_trigger: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """时间驱动 7 状态 FSM 转移"""
    fsm_timer += dt
    for s in range(-1, 6):
        m = fsm_state == s
        if not m.any():
            continue
        if s == -1:
            t = (flip_trigger[m] > 0.5) & (fsm_timer[m] > 0.05)
        else:
            t = fsm_timer[m] > _FSM_DUR[s]
        nxt = np.zeros_like(fsm_state, dtype=bool)
        nxt[m] = t
        nxt_state = s + 1 if s < 5 else -1
        fsm_state[nxt] = nxt_state
        fsm_timer[nxt] = 0.0
    return fsm_state, fsm_timer


# ========== 奖励函数 (相位门控) ==========


def _reward_flip_progress(ctx: RewardContext) -> np.ndarray:
    """翻转进度: 奖励后翻方向进展, 状态 1/2/3; 超过 360° 封顶(防过度旋转)"""
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [1, 2, 3]).astype(np.float64)
    delta = ctx.info["flip_progress_delta"]  # 后翻 delta>0
    not_over = (ctx.info["flip_progress"] < FLIP_TARGET).astype(np.float64)  # 超 360° 不再奖励
    return np.clip(delta, 0.0, None) * active * not_over


def _reward_launch_thrust(ctx: RewardContext) -> np.ndarray:
    """蹬地升力: 状态 1 奖 vz"""
    fsm = ctx.info["fsm_state"]
    active = (fsm == 1).astype(np.float64)
    return np.clip(ctx.linvel[:, 2], 0.0, 2.0) * active


def _reward_upright_landing(ctx: RewardContext) -> np.ndarray:
    """落地直立: 状态 4/5 奖重力对齐 body z"""
    assert ctx.gravity is not None
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [4, 5]).astype(np.float64)
    return np.clip(ctx.gravity[:, 2], 0.0, 1.0) * active


def _reward_landing_soft(ctx: RewardContext) -> np.ndarray:
    """着陆缓冲: 状态 3/4 奖低冲击"""
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [3, 4]).astype(np.float64)
    return np.exp(-np.abs(ctx.linvel[:, 2]) / 0.5) * active


def _reward_wheel_ground_matching(ctx: RewardContext) -> np.ndarray:
    """轮地速度匹配: 状态 3/4"""
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [3, 4]).astype(np.float64)
    wheel_vel = ctx.info.get("wheel_vel", np.zeros((ctx.num_envs, 2)))
    lin_x = ctx.linvel[:, 0:1]
    error = np.sum(np.square(wheel_vel * WHEEL_R - lin_x), axis=1)
    return -error * active


def _reward_base_height_landing(ctx: RewardContext) -> np.ndarray:
    """落地高度: 状态 4/5 奖接近目标"""
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [4, 5]).astype(np.float64)
    return -np.square(ctx.base_height - ctx.base_height_target) * active


def _reward_posture_stand(ctx: RewardContext) -> np.ndarray:
    """站立姿态: 状态 -1/5 罚髋后仰/伸腿"""
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [-1, 5]).astype(np.float64)
    hip_fwd_L = ctx.dof_pos[:, 1]
    hip_fwd_R = -ctx.dof_pos[:, 4]
    knee_bend_L = ctx.dof_pos[:, 2]
    knee_bend_R = -ctx.dof_pos[:, 5]
    p_hip = np.clip(-hip_fwd_L, 0, 1) + np.clip(-hip_fwd_R, 0, 1)
    p_knee = np.clip(-knee_bend_L, 0, 1) ** 2 + np.clip(-knee_bend_R, 0, 1) ** 2
    return -(p_hip + p_knee) * active


def _reward_stand_balance(ctx: RewardContext) -> np.ndarray:
    """站住不倒: 状态 -1/4/5 温和奖励直立 (up=cos tilt), 不罚倾角 — 只需不倒, 不要求完美平衡"""
    assert ctx.gravity is not None
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [-1, 4, 5]).astype(np.float64)
    up = np.clip(ctx.gravity[:, 2], 0.0, 1.0)  # 直立度, 45°→0.7, 60°→0.5
    return up * active


def _reward_stand_height(ctx: RewardContext) -> np.ndarray:
    """不倒: 状态 -1/4/5 奖励高度>0.25 (轮子撑住不折叠), 不要求站直 0.65"""
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [-1, 4, 5]).astype(np.float64)
    # 高度 0.25→奖励0(贴地), 0.45→奖励1(不倒), 之后封顶
    r = np.clip((ctx.base_height - 0.25) / 0.2, 0.0, 1.0)
    return r * active


def _reward_ff_tracking(ctx: RewardContext) -> np.ndarray:
    """参考跟踪(文献 OPT-Mimic): 翻转中惩罚策略偏离 ff — 防甩腿伪造旋转/腿扭麻花
    策略输出大 = 偏离开环 ff → 翻转变乱。让策略输出小, ff 驱动干净翻转。"""
    fsm = ctx.info["fsm_state"]
    active = np.isin(fsm, [1, 2, 3]).astype(np.float64)  # 蹬/飞/展
    policy_actions = ctx.info.get("policy_actions", np.zeros((ctx.num_envs, 8)))
    return -np.sum(np.square(policy_actions), axis=1) * active


def _reward_flip_complete(ctx: RewardContext) -> np.ndarray:
    """翻转完成奖励(锁存): 旋转已完成(_flip_completed) 且之后站起来(z>0.25) 且不摔 (up>0.6)
    先转完(可在低处), 之后站起来即算成功 — 匹配"后空翻后不倒"需求"""
    assert ctx.gravity is not None
    flip_done = ctx.info.get("flip_completed", np.zeros(ctx.num_envs, dtype=bool))
    up = ctx.gravity[:, 2]
    not_fallen = ctx.base_height > 0.25  # 轮子撑住, 不折叠贴地
    done = flip_done & (up > 0.6) & not_fallen
    return done.astype(np.float64)


def _reward_action_magnitude(ctx: RewardContext) -> np.ndarray:
    return np.sum(np.square(ctx.info["current_actions"]), axis=1)


def _reward_leg_mirror(ctx: RewardContext) -> np.ndarray:
    hip_error = np.abs(ctx.dof_pos[:, 0] + ctx.dof_pos[:, 3])
    pitch_error = np.abs(ctx.dof_pos[:, 1] + ctx.dof_pos[:, 4])
    knee_error = np.abs(ctx.dof_pos[:, 2] - ctx.dof_pos[:, 5])
    return hip_error + pitch_error + knee_error


# ========== 配置 ==========


@dataclass
class XqRobotWLBackflipCommands(Commands):
    """5D 命令: [vx, vy, vyaw, tsk, flip_trigger]"""

    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.3, 0.0, -0.5, -0.1, 0], [0.3, 0.0, 0.5, 0.1, 1]]
    )
    resampling_time: float = 4.0


@dataclass
class XqRobotWLBackflipRewardConfig:
    scales: dict[str, float]
    only_positive_rewards: bool = False
    tracking_sigma: float = 0.3
    base_height_target: float = 0.55
    max_tilt_deg: float = 45.0
    min_base_height: float = 0.15
    flip_target_rad: float = 6.283
    ff_gain: float = 1.0
    flip_trigger_prob: float = 0.7
    flip_warmup_iters: int = 100


@registry.envcfg("XqRobotWLBackflipFlat")
@dataclass
class XqRobotWLBackflipFlatCfg(XqRobotWLJumpSRLFlatCfg):
    commands: XqRobotWLBackflipCommands = field(default_factory=XqRobotWLBackflipCommands)  # type: ignore[assignment]
    reward_config: XqRobotWLBackflipRewardConfig | None = None  # type: ignore[assignment]
    max_episode_seconds: float = 4.0


class XqRobotWLBackflipDRProvider(XqRobotWLJumpDRProvider):
    """采样命令: 站立/翻转混合 (flip_trigger∈{0,1})"""

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        cmds = np.zeros((num_reset, 5), dtype=get_global_dtype())
        prob = getattr(env._cfg.reward_config, "flip_trigger_prob", 0.7)
        flip = np.random.uniform(0.0, 1.0, num_reset) < prob
        cmds[:, 4] = flip.astype(get_global_dtype())
        return cmds


@registry.env("XqRobotWLBackflipFlat", sim_backend="mujoco")
class XqRobotWLBackflipFlatEnv(XqRobotWLJumpSRLFlatEnv):
    """xqrobotwl 后空翻环境 — FSM前馈 + flip_progress + 相位门控奖励"""

    _cfg: XqRobotWLBackflipFlatCfg
    _jump_cfg: XqRobotWLBackflipRewardConfig  # type: ignore[assignment]  # 收窄基类奖励配置类型

    def __init__(self, cfg: XqRobotWLBackflipFlatCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        self._jump_cfg = cfg.reward_config
        self._total_env_steps = 0
        self._warmup_progress = 0.0
        self._ff_gain = cfg.reward_config.ff_gain
        self._flip_warmup_env_steps = cfg.reward_config.flip_warmup_iters * 24 * num_envs
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._dr_manager._provider = XqRobotWLBackflipDRProvider()  # type: ignore[union-attr]
        # 后空翻专用状态
        self._fsm_state = -np.ones(num_envs, dtype=np.int32)
        self._fsm_timer = np.zeros(num_envs, dtype=np.float64)
        self._flip_progress = np.zeros(num_envs, dtype=np.float64)
        self._flip_progress_delta = np.zeros(num_envs, dtype=np.float64)
        self._prev_theta = np.zeros(num_envs, dtype=np.float64)
        self._theta_init = np.zeros(num_envs, dtype=bool)
        self._flip_completed = np.zeros(num_envs, dtype=bool)  # 旋转完成锁存
        self._flip_done = np.zeros(num_envs, dtype=bool)  # 本 episode 已翻过一次 (单次翻转)
        self._policy_actions = np.zeros((num_envs, 8), dtype=np.float64)  # ff_tracking 参考跟踪

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # base 297/324 + 额外 3×9=27 (flip_progress_norm, fsm_state/5, timer/0.8)
        return {"obs": 297 + 27, "critic": 324 + 27}

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, object] = {
            "flip_progress": _reward_flip_progress,
            "launch_thrust": _reward_launch_thrust,
            "upright_landing": _reward_upright_landing,
            "landing_soft": _reward_landing_soft,
            "wheel_ground_matching": _reward_wheel_ground_matching,
            "base_height_landing": _reward_base_height_landing,
            "posture_stand": _reward_posture_stand,
            "stand_balance": _reward_stand_balance,
            "stand_height": _reward_stand_height,
            "ff_tracking": _reward_ff_tracking,
            "flip_complete": _reward_flip_complete,
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

    # ── reset: 硬重置翻转状态 (一次性动作必须干净开始) ──

    def _reset_done_envs(self) -> None:
        assert self._state is not None
        done = self._state.terminated | self._state.truncated
        idx = np.flatnonzero(done).astype(np.int32)
        super()._reset_done_envs()
        self._fsm_state[idx] = -1
        self._fsm_timer[idx] = 0.0
        self._flip_progress[idx] = 0.0
        self._flip_progress_delta[idx] = 0.0
        self._prev_theta[idx] = 0.0
        self._theta_init[idx] = False
        self._flip_completed[idx] = False
        self._flip_done[idx] = False

    # ── step: FSM 前馈 + PPO 反馈融合 ──

    def step(self, actions):
        dof_pos = self.get_dof_pos()
        linvel = self.get_local_linvel()
        ff = _compute_feedforward(
            self._fsm_state,
            self._fsm_timer,
            dof_pos,
            linvel,
            self._cfg.control_config.action_scale,
            self._cfg.control_config.wheel_action_scale,
        )
        # ★ 确定性翻转 (用户决定 2026-08-05): 翻转期(FSM 0-5)策略动作强制为 0, 纯 ff 驱动翻转
        # (200Hz ff 已验证 16/16 可靠翻转); RL 只在站立态(-1)生效, 专注两轮足平衡。
        # 反复实验证明策略会被奖励诱导偏离 ff 把翻转搞坏 (flip_progress 刷分), 故翻转期不容策略干扰。
        stand_mask = (self._fsm_state == -1).astype(np.float64)[:, None]
        actions_masked = actions * stand_mask
        fused = ff * self._ff_gain + actions_masked
        self._policy_actions = actions_masked  # ff_tracking 反映实际贡献 (翻转期=0)
        # 直接调 WalkFlatEnv.step — 绕过 jump_srl 的 SLIP 前馈叠加 (super().step 会再叠 jump_ff*0.15,
        # 污染干净的后空翻轨迹, crouch 相位膝目标被改 ~20%)
        return XqRobotWLWalkFlatEnv.step(self, fused)

    # ── 状态更新 ──

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)

        # 预热: 前 N 轮纯站立, 之后斜坡放开 flip_trigger
        # ★ 不修改原始 trigger (避免累乘复合衰减), 只在 FSM 决策时用 warmup_progress 缩放
        self._total_env_steps += self._num_envs
        if self._flip_warmup_env_steps <= 0:
            self._warmup_progress = 1.0
        else:
            self._warmup_progress = np.clip(
                (self._total_env_steps - self._flip_warmup_env_steps) / self._flip_warmup_env_steps,
                0.0,
                1.0,
            )

        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        # flip_progress: 世界系 Y 角速度积分 (正=后翻, 鲁棒于横滚/偏航)
        self._update_flip_progress(self._backend.get_base_ang_vel(), linvel.shape[0])
        state.info["flip_progress"] = self._flip_progress.copy()
        state.info["flip_progress_delta"] = self._flip_progress_delta.copy()
        # 旋转完成锁存: 物理事件判据 — 飞行态(1/2/3)中翻过身(up_z<-0.3, >107°) 且后翻方向
        # 有实际进度(flip_progress>1.5, 排除前翻/侧滚)。不依赖旋转积分阈值:
        # 之前 fp>=5.98 因世界系 Y 角速度测量低估(实际~5.2)+站立态重置而物理上不可达,
        # 导致 flip_complete 在所有训练 run 中恒为 0。翻过身后再站起来(z>0.25,up>0.6)即触发奖励。
        airborne = np.isin(self._fsm_state, [1, 2, 3])
        flipped = airborne & (gravity[:, 2] < -0.3) & (self._flip_progress > 1.5)
        self._flip_completed |= flipped
        self._flip_done |= flipped  # 单次翻转: 完成后不再触发
        state.info["flip_completed"] = self._flip_completed.copy()

        # FSM 更新
        # 单次翻转: 已翻过一次则抑制 trigger (翻完站立到超时, 不再连翻)
        jt = state.info["commands"][:, 4] * self._warmup_progress * (~self._flip_done)
        self._fsm_state, self._fsm_timer = _update_fsm(
            self._fsm_state, self._fsm_timer, jt, self._cfg.ctrl_dt
        )
        state.info["fsm_state"] = self._fsm_state.copy()
        state.info["fsm_timer"] = self._fsm_timer.copy()
        # 站立态重置 flip_progress (避免站姿晃动累积污染翻转进度 / 触发 not_over 封顶)
        # 保留此重置: 翻转完成锁存已改用物理事件判据(飞行态翻过身), 不依赖 fp 绝对阈值;
        # 若改成 ep 级累计, 站立期(命令重采样 4s)后向晃动会把 fp 抬近 FLIP_TARGET,
        # 使飞行期 flip_progress 奖励被 not_over 提前封顶。
        standing = self._fsm_state == -1
        self._flip_progress[standing] = 0.0
        self._flip_progress_delta[standing] = 0.0
        state.info["wheel_vel"] = dof_vel[:, NUM_LEG_ACTIONS:]
        state.info["policy_actions"] = self._policy_actions.copy()  # ff_tracking 参考跟踪
        self._update_wheel_contact(state.info)

        terminated = self._compute_terminated(gravity, dof_pos)
        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        # flip_complete 单次触发后重置锁存 (避免持续刷奖励)
        fired = state.info.get("flip_completed", np.zeros(self._num_envs, dtype=bool))
        if fired.any():
            up_fired = gravity[:, 2] > 0.6
            base_z = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
            self._flip_completed[fired & up_fired & (base_z > 0.25)] = False
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        return state.replace(obs=obs, reward=reward, terminated=terminated)

    def _update_flip_progress(self, ang_vel_w: np.ndarray, num_obs: int) -> None:
        """世界系 Y 角速度积分 (P1 方法): flip_progress = -∫ω_y dt, 后翻为正
        ★ 世界系测量对横滚/偏航鲁棒 — 机身系 gyro / up-向量 XZ 在乱翻时都失真"""
        ang_vel_w = np.asarray(ang_vel_w, dtype=get_global_dtype())[:num_obs]
        self._flip_progress_delta[:num_obs] = -ang_vel_w[:, 1] * self._cfg.ctrl_dt
        self._flip_progress[:num_obs] += self._flip_progress_delta[:num_obs]

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
        """终止 (分阶段: 先放宽让翻转学成): 非飞行态(-1/4/5)下
        - 机身/小腿碰地: base_z < min 持续 0.3s 才终止 (给落地恢复窗口)
        - 倾倒: tilt > max 持续 0.15s
        - 关节塌陷: 髋/膝极端
        飞行态(1/2/3)不按倾角/高度终止"""
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        grounded = np.isin(self._fsm_state, [-1, 4, 5])
        max_tilt = np.deg2rad(self._jump_cfg.max_tilt_deg)
        base_z = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        sustained = self._fsm_timer > 0.15
        # 机身/小腿碰地: 持续 0.3s 才终止 (给落地 0.3s 恢复窗口)
        terminated = (base_z < self._jump_cfg.min_base_height) & grounded & (self._fsm_timer > 0.3)
        # 倾倒: 持续 0.15s
        terminated |= (tilt > max_tilt) & grounded & sustained
        thigh_collapsed = (dof_pos[:, 1] < -1.0) | (dof_pos[:, 4] > 1.0)
        # 膝范围 ±0.87, 阈值 0.8。仅站立/恢复态(-1/5)触发
        calf_extreme = ((np.abs(dof_pos[:, 2]) > 0.8) | (np.abs(dof_pos[:, 5]) > 0.8)) & np.isin(
            self._fsm_state, [-1, 5]
        )
        terminated |= thigh_collapsed | calf_extreme
        return terminated

    def _compute_obs(self, info, linvel, gyro, gravity, dof_pos, dof_vel):
        # 直接调 joystick 基类 (跳过 jump_srl 的额外追加, 避免双重叠加)
        base = XqRobotWLWalkFlatEnv._compute_obs(
            self, info, linvel, gyro, gravity, dof_pos, dof_vel
        )
        batch = linvel.shape[0]
        flip_norm = np.clip(self._flip_progress[:batch] / FLIP_TARGET, 0.0, 2.0).reshape(-1, 1)
        fsm_feat = self._fsm_state[:batch].astype(np.float64).reshape(-1, 1) / 6.0
        timer_feat = np.clip(self._fsm_timer[:batch].reshape(-1, 1) / 0.8, 0.0, 1.0)
        extra = np.tile(
            np.concatenate([flip_norm, fsm_feat, timer_feat], axis=1, dtype=get_global_dtype())[
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
