"""xqrobotwl 可移动单轮平衡环境 — 独轮车式控制 (独立 task, 从零训练)

与 single_leg.py (静态保持 FSM) 完全独立:
  - 无 FSM / 无后空翻遗产 (直接继承 XqRobotWLWalkFlatEnv 最底层)
  - 命令 = vx (前进/后退), 独轮车式: 支撑轮控 pitch+移动, 自由腿配重控 roll
  - start_in_balance: episode 直接从单轮平衡位起步, 全程单轮

经典控制物理验证 (devlog 07) 指导:
  1. kp=60 柔性腿静态能保持 2s (执行器够用), 高 kp 刚性反而差
  2. 30° 侧压结构力矩大偏难, 22° 折中 (仍有明显侧压感)
  3. 最优策略 = 微扰动保持姿态 (奖励温和, 不引导大幅动作)
  4. 平衡参考 = 机身对齐 22° 侧压 (up_ref=[0,sin22°,cos22°])

Joint order (policy): [L_hip_roll, L_hip_pitch, L_knee, R_hip_roll, R_hip_pitch, R_knee, L_wheel, R_wheel]
支撑侧: 右腿支撑, 左腿配重微屈
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
from unilab.envs.locomotion.xqrobotwl.base import NUM_LEG_ACTIONS, NUM_WHEEL_ACTIONS
from unilab.envs.locomotion.xqrobotwl.joystick import (
    XqRobotWLDRProvider,
    XqRobotWLRewardConfig,
    XqRobotWLWalkFlatCfg,
    XqRobotWLWalkFlatEnv,
)

# ── 平衡位参数 (经典控制物理验证 devlog 07 + 自由轮离地扫描) ──
# 30° 侧压 + 配重微屈 = 自由轮 96% 离地 (22° 只 46%, 物理上自由轮着地)
_LEAN_DEG = 30.0
_ROLL_REF_RAD = np.deg2rad(_LEAN_DEG)
# 平衡位姿态: 自由腿(左) L_hip_roll 展开当配重, pitch/knee 微屈; 支撑腿(右) 伸直
_FREE_LEG_ROLL_INIT = -0.5
_FOLD_PITCH = 0.10
_FOLD_KNEE = 0.30
# 钉住关节动作值 (target = act·flip·scale + default):
#   L_pitch(1): flip+1 → (0.10-0.15)/0.6=-0.083; L_knee(2): flip-1 → (0.30-0.15)/0.6/-... = -0.25
#   R_pitch(4): flip-1 → (0-(-0.15))/(-0.6)=-0.25; R_knee(5): flip+1 → (0-(-0.15))/0.6=0.25
_PINNED = np.array([0.0, -0.083, -0.25, 0.0, -0.25, 0.25, 0.0, 0.0], dtype=np.float64)
# mask: 1=RL 自由, 0=钉住。配重 L_hip_roll(0)+支撑腿 R_hip_roll(3)+双轮(6,7) 自由;
# 自由腿 pitch/knee(1,2) + 支撑腿 pitch/knee(4,5) 钉住保持姿态
_RL_MASK = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
_NUM_CMD_DIM = 5

# ── LQR 参考控制器 (P3 验证: 10s 稳定, 突破 PD 1.35s 物理上限) ──
# a_cmd = -(k1·θ + k2·θ̇) - k3·vx  (轮子线加速度, vx 反馈是关键)
# u_r = -0.5 - (kr1·(φ+30°) + kr2·φ̇)  (配重目标)
_WHEEL_R = 0.11
_LQR_K1 = 180.0
_LQR_K2 = 22.0
_LQR_K3 = 10.0  # ★ vx 反馈: 无它只有 2.28s, 有它 10s 稳定
_LQR_KR1 = 1.5
_LQR_KR2 = 0.4


@dataclass
class XqRobotWLSingleLegMoveCommands(Commands):
    # 5D: [vx, vy, vyaw, tsk, height] — 只开 vx (±0.3), 其余固定
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-0.3, 0.0, 0.0, 0.0, 0.55], [0.3, 0.0, 0.0, 0.0, 0.55]]
    )
    resampling_time: float = 3.0


@dataclass
class XqRobotWLSingleLegMoveRewardConfig(XqRobotWLRewardConfig):
    scales: dict[str, float] = field(default_factory=dict)
    tracking_sigma: float = 0.3
    base_height_target: float = 0.55
    only_positive_rewards: bool = False
    max_tilt_deg: float = 60.0
    min_base_height: float = 0.20


@registry.envcfg("XqRobotWLSingleLegMove")
@dataclass
class XqRobotWLSingleLegMoveCfg(XqRobotWLWalkFlatCfg):
    commands: XqRobotWLSingleLegMoveCommands = field(default_factory=XqRobotWLSingleLegMoveCommands)
    reward_config: XqRobotWLSingleLegMoveRewardConfig | None = None
    max_episode_seconds: float = 8.0


# ── 奖励函数 (微扰动保持 + 移动, 参考 single_leg 但温和) ──


def _reward_balance_upright(ctx: RewardContext, lean_rad: float) -> np.ndarray:
    """机身对齐平衡参考 (22° 侧压): dot = gravity·up_ref, 锐化点积。"""
    assert ctx.gravity is not None
    up_ref = np.array([0.0, np.sin(lean_rad), np.cos(lean_rad)])
    dot = ctx.gravity[: ctx.num_envs] @ up_ref
    r = np.clip((dot - 0.85) / 0.15, 0.0, 1.0)
    return r**2


def _reward_wheel_off(ctx: RewardContext) -> np.ndarray:
    """单腿判定 = 自由腿配重展开 (L_hip_roll < -0.3) — 防收回自由腿两轮作弊。

    ⚠️ 不用 wheel_force (free force sensor 含惯性虚拟力, 自由轮离地仍 ~20-138N 误判)。
    """
    l_roll = ctx.dof_pos[:, 0]
    return (l_roll < -0.25).astype(np.float64)


def _reward_fold_pose(ctx: RewardContext) -> np.ndarray:
    """姿态保持: 支撑腿伸直 + 自由腿微屈 (温和二次罚)。"""
    dp = ctx.dof_pos
    r_pitch = dp[:, 4]  # 支撑腿 pitch 目标 0
    r_knee = dp[:, 5]  # 支撑腿 knee 目标 0
    l_pitch = dp[:, 1] - _FOLD_PITCH  # 自由腿 pitch 目标 0.10
    l_knee = dp[:, 2] - _FOLD_KNEE  # 自由腿 knee 目标 0.30
    return np.sum(np.square(np.stack([r_pitch, r_knee, l_pitch, l_knee], axis=1)), axis=1)


def _reward_roll_rate(ctx: RewardContext) -> np.ndarray:
    """角速度阻尼: 罚 roll/pitch 速率 (微扰动保持, 不引导大幅动作)。"""
    gyro = ctx.gyro[: ctx.num_envs, :2]
    return np.sum(np.square(gyro), axis=1)


def _reward_lin_vel(ctx: RewardContext) -> np.ndarray:
    """水平速度惩罚 (防漂移 — LQR 突破的关键: vx 反馈)。

    ⚠️ 单轮平衡靠速度反馈稳定 (LQR k3·vx 10s), 无速度惩罚时 RL 允许缓慢漂移倾覆。
    """
    lv = ctx.linvel[: ctx.num_envs, :2]
    return np.sum(np.square(lv), axis=1)


def _reward_lqr_ref(ctx: RewardContext) -> np.ndarray:
    """LQR 参考引导: 罚 RL 动作偏离 LQR 控制器 (配重+支撑轮)。

    让 RL 学 LQR 的显式状态反馈 (obs→动作映射), 再增强鲁棒。
    """
    lqr = ctx.info.get("lqr_action")
    if lqr is None:
        return np.zeros(ctx.num_envs, dtype=np.float64)
    act = ctx.info["current_actions"][:, :8]
    diff = np.square(act[:, 0] - lqr[:, 0]) + np.square(act[:, 7] - lqr[:, 7])
    return diff


def _reward_balance_complete(ctx: RewardContext) -> np.ndarray:
    """单轮平衡持续达成: 连续 hold 步单轮+平衡 → 一次性大奖 (给 RL 明确目标)。

    读取 env 维护的 _single_leg_hold (连续满足条件步数), 达标即奖并清零。
    """
    done = np.asarray(ctx.info.get("_single_leg_done", np.zeros(ctx.num_envs, dtype=bool)))
    return done.astype(np.float64)


def _reward_joint_action_rate(ctx: RewardContext) -> np.ndarray:
    cur = ctx.info["current_actions"][:, :NUM_LEG_ACTIONS]
    lst = ctx.info["last_actions"][:, :NUM_LEG_ACTIONS]
    return np.sum(np.square(cur - lst), axis=1)


def _reward_wheel_action_rate(ctx: RewardContext) -> np.ndarray:
    cur = ctx.info["current_actions"][:, NUM_LEG_ACTIONS:]
    lst = ctx.info["last_actions"][:, NUM_LEG_ACTIONS:]
    return np.sum(np.square(cur - lst), axis=1)


class XqRobotWLSingleLegMoveDRProvider(XqRobotWLDRProvider):
    """reset 置 22° 侧压平衡位 (start_in_balance), 命令只 vx。"""

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        low = np.asarray(env._cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(env._cfg.commands.vel_limit[1], dtype=get_global_dtype())
        cmds = np.zeros((num_reset, _NUM_CMD_DIM), dtype=get_global_dtype())
        cmds[:, 0] = np.random.uniform(low[0], high[0], num_reset)
        cmds[:, 3:] = np.tile(low[3:], (num_reset, 1))  # tsk/height 固定
        return cmds

    def build_reset_plan(self, env: Any, env_ids: np.ndarray) -> ResetPlan:
        num_reset = len(env_ids)
        lean = np.radians(-_LEAN_DEG)
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


@registry.env("XqRobotWLSingleLegMove", sim_backend="mujoco")
class XqRobotWLSingleLegMoveEnv(XqRobotWLWalkFlatEnv):
    """可移动单轮平衡 — 独轮车式 (支撑轮控 pitch+移动, 配重控 roll)。"""

    _cfg: XqRobotWLSingleLegMoveCfg

    def __init__(self, cfg: XqRobotWLSingleLegMoveCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        self._dr_manager._provider = XqRobotWLSingleLegMoveDRProvider()  # type: ignore[union-attr]
        self._lean_rad = _ROLL_REF_RAD
        # 单轮平衡持续计数 (防两轮作弊: 自由轮离地+平衡+高度连续 hold 才给奖)
        self._single_leg_hold = np.zeros(num_envs, dtype=np.float64)
        self._single_leg_done = np.zeros(num_envs, dtype=bool)
        self._balance_hold_time = 0.3
        # 自由轮着地计时: 着地>阈值终止 (两轮=死路, 强制单轮)
        self._free_down = np.zeros(num_envs, dtype=np.float64)
        self._free_down_limit = 0.3
        # LQR 参考: 轮速积分状态
        self._lqr_wheel_vel = np.zeros(num_envs, dtype=np.float64)

    def _reset_done_envs(self) -> None:
        assert self._state is not None
        done = self._state.terminated | self._state.truncated
        idx = np.flatnonzero(done).astype(np.int32)
        super()._reset_done_envs()
        self._single_leg_hold[idx] = 0.0
        self._single_leg_done[idx] = False
        self._free_down[idx] = 0.0
        self._lqr_wheel_vel[idx] = 0.0

    def update_state(self, state: NpEnvState) -> NpEnvState:
        # 单轮平衡保持计数 (在 super 计算奖励前更新, 供 _reward_balance_complete 读取)
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        base_z = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        dof_pos = self.get_dof_pos()
        # 单腿判定 = 自由腿配重展开 (L_hip_roll<-0.3)。⚠️ 不用 wheel_force (不可靠)
        free_off = dof_pos[:, 0] < -0.25
        # 自由腿收回计时 (两轮=死路)
        self._free_down = np.where(dof_pos[:, 0] > -0.1, self._free_down + self._cfg.ctrl_dt, 0.0)
        upright = (
            gravity[: self._num_envs]
            @ np.array([0.0, np.sin(self._lean_rad), np.cos(self._lean_rad)])
        ) > 0.9
        height_ok = base_z > 0.40
        held = free_off & upright & height_ok
        self._single_leg_hold = np.where(held, self._single_leg_hold + self._cfg.ctrl_dt, 0.0)
        just_done = (self._single_leg_hold >= self._balance_hold_time) & ~self._single_leg_done
        self._single_leg_done |= just_done
        state.info["_single_leg_done"] = just_done.astype(np.float64)

        # ── LQR 参考动作 (P3 验证: vx 反馈 10s 稳定) ──
        gyro = self.get_gyro()
        linvel = self.get_local_linvel()
        num = linvel.shape[0]
        theta = np.arctan2(gravity[:num, 0], gravity[:num, 2])
        phi = np.arctan2(gravity[:num, 1], gravity[:num, 2])
        a_cmd = -(_LQR_K1 * theta + _LQR_K2 * gyro[:num, 1]) - _LQR_K3 * linvel[:num, 0]
        self._lqr_wheel_vel = np.clip(
            self._lqr_wheel_vel + (a_cmd / _WHEEL_R) * self._cfg.ctrl_dt, -25.0, 25.0
        )
        u_r = _FREE_LEG_ROLL_INIT - (_LQR_KR1 * (phi - np.radians(30.0)) + _LQR_KR2 * gyro[:num, 0])
        # LQR 动作 (current_actions 空间, 只填自由关节 配重0 + 支撑轮7)
        lqr = np.zeros((num, 8), dtype=get_global_dtype())
        rl_cw = (u_r + 0.5) / 0.3  # 反解配重动作映射 target=-0.5+0.3·rl
        lqr[:, 0] = -1.0 + 0.5 * np.clip(rl_cw, -1.0, 1.0)
        lqr[:, 7] = self._lqr_wheel_vel / self._cfg.control_config.wheel_action_scale
        state.info["lqr_action"] = lqr
        state = super().update_state(state)
        # 达到单轮平衡后允许重置 (给 env 刷新, 重新挑战)
        if just_done.any():
            self._single_leg_done[just_done] = False
            self._single_leg_hold[just_done] = 0.0
        return state

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {
            "obs": self._obs_frame_dim * self._hist_len,
            "critic": self._critic_frame_dim * self._hist_len,
        }

    def _init_reward_functions(self) -> None:
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "balance_upright": lambda ctx: _reward_balance_upright(ctx, self._lean_rad),
            "wheel_off": _reward_wheel_off,
            "fold_pose": _reward_fold_pose,
            "roll_rate": _reward_roll_rate,
            "lin_vel": _reward_lin_vel,
            "lqr_ref": _reward_lqr_ref,
            "balance_complete": _reward_balance_complete,
            "joint_action_rate": _reward_joint_action_rate,
            "wheel_action_rate": _reward_wheel_action_rate,
            "alive": rewards.alive,
        }

    def step(self, actions):
        # 钉住自由腿 pitch/knee + 支撑腿 pitch/knee (保持平衡姿态),
        # 放开配重 L_hip_roll + 支撑腿 R_hip_roll + 双轮 (独轮车控制通道)
        actions = actions * _RL_MASK + _PINNED * (1.0 - _RL_MASK)
        # ★ L_hip_roll(0) 动作映射: RL 0 → 配重展开位 -0.5, ±0.5·rl 微调
        # (否则策略需探索到输出 -1 才能保持展开, std 不够探索不到)
        # 传给 apply_action 的动作: target = act·0.6+0.1 = -0.5+0.3·rl → act = -1.0+0.5·rl
        actions[:, 0] = -1.0 + 0.5 * np.clip(actions[:, 0], -1.0, 1.0)
        return super().step(actions)

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        dtype = get_global_dtype()
        num_obs = linvel.shape[0]
        ctx = RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos[:, :NUM_LEG_ACTIONS],
            dof_vel=dof_vel[:, :NUM_LEG_ACTIONS],
            num_envs=num_obs,
            default_angles=np.zeros(NUM_LEG_ACTIONS, dtype=dtype),
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

    def _compute_terminated(self, gravity: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        # 单轮平衡: 宽容终止 (侧压平衡位本身 tilt~22°, 允许摆动)
        base_z = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())[:, 2]
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        max_tilt = np.deg2rad(self._reward_cfg.max_tilt_deg)
        terminated = np.logical_or(tilt > max_tilt, base_z < self._reward_cfg.min_base_height)
        # 腿极端屈曲 (kp=60 柔性腿允许小幅屈, 只拦极端)
        calf_extreme = (np.abs(dof_pos[:, 2]) > 1.4) | (np.abs(dof_pos[:, 5]) > 1.4)
        terminated |= calf_extreme
        # ★ 自由轮着地持续 >阈值 → 终止 (防两轮支撑作弊)
        terminated |= self._free_down > self._free_down_limit
        return terminated
