"""MPC×SAC 融合分支 — 低层线性 MPC 控制器 (冻结执行器, 分支自带).

移植经典轨 scripts/classic_control/mpc/controller.py (sagittal QP) +
scripts/classic_control/common/base.py (act 骨架: 腿目标/偏航/轮映射) —
逻辑逐行一致, 独立拥有, 不 import classic_control/mpc。

约定: act(sensors, cmd) → a8 (RL 空间 8D 归一化动作, 直接进 env.step)。
相位: phase=3 (flat 速度+腿高), phase=4 (rough 地形 per-leg)。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from scripts.classic_control.common.config import DEFAULT_LEG_ANGLES, WHEEL_R
from scripts.classic_control.common.leg_control import fk_legs, standing_joint_targets

from scripts.fusion_control.mpc_sac.mpc.dynamics import load_plant_model
from scripts.fusion_control.mpc_sac.mpc.qp import build_mpc_matrices, dlqr_riccati, solve_qp

_FLIP = np.array([1, 1, -1, 1, -1, 1, 1, -1], dtype=np.float64)


class MpcController:
    """低层 MPC: 滚动时域求解轮速命令 (rad/s) + 腿目标 + 偏航差分 → a8."""

    def __init__(
        self,
        phase: int,
        params: dict[str, Any] | None = None,
        action_scale: float = 0.6,
        wheel_action_scale: float = 10.0,
        dt: float = 0.01,
    ) -> None:
        self.phase = phase
        self.dt = dt
        self.action_scale = action_scale
        self.wsa = wheel_action_scale
        params = {} if params is None else params
        self.params = params

        # 控制/命令参数 (来自 conf/fusion_control/mpc_sac, 经 commands_params)
        self._smoothing = float(params.get("smoothing", 0.85))  # 轮速命令平滑
        self.track = float(params.get("track_width", 0.38))
        self.k_yaw = float(params.get("k_yaw", 3.0))
        self.sign = float(params.get("sign", 1.0))  # 控制方向标定
        self.wheel_vel_max = float(params.get("wheel_vel_max", 25.0))

        # 运行时状态
        self._w_c = 0.0
        self._cmd_smooth = 0.0
        self._cmd_smooth_init = False
        self._height_int = 0.0
        self._h_cmd_smooth: float | None = None
        self._xpos = 0.0
        self._z_int = 0.0
        self._prev_base_z: float | None = None
        self._lam0: np.ndarray | None = None
        self._mpc_u_prev = 0.0
        self._solve_time_ms = 0.0

        self._integrate_sagittal()

    # ── 构造 (MPC 矩阵预计算) ──
    def _integrate_sagittal(self) -> None:
        """轮速命令模型 (w_c convention) + 5 态积分增广 + 预计算 P/H/G."""
        use_bb = self.phase >= 2
        loaded = False
        if self.params.get("A_d") is not None and self.params.get("B_d") is not None:
            self._mpc_A = np.asarray(self.params["A_d"], dtype=np.float64)
            self._mpc_B = np.asarray(self.params["B_d"], dtype=np.float64).reshape(-1, 1)
            loaded = True
        elif use_bb and self.params.get("model_file"):
            self._mpc_A, self._mpc_B = load_plant_model(
                float(self.params.get("alpha", 24.4)),
                float(self.params.get("beta", -2.5)),
                float(self.params.get("tau", 0.059)),
                self.dt,
                str(self.params["model_file"]),
            )
            loaded = True
        if not loaded:
            self._mpc_A, self._mpc_B = load_plant_model(
                float(self.params.get("alpha", 24.4)),
                float(self.params.get("beta", -2.5)),
                float(self.params.get("tau", 0.059)),
                self.dt,
                None,
            )
        N = int(self.params.get("mpc_horizon", 20))
        if self.phase >= 2:
            # ★ 5 态积分增广 z=∫(v−v_ref) (offset-free MPC)
            self._mpc_nx = 5
            A_aug = np.zeros((5, 5))
            A_aug[:4, :4] = self._mpc_A
            A_aug[4, 2] = self.dt
            A_aug[4, 4] = 1.0
            B_aug = np.vstack([self._mpc_B, [[0.0]]])
            self._mpc_A = A_aug
            self._mpc_B = B_aug
            Q_mpc = np.diag(
                [
                    float(self.params.get("q_theta", 100.0)),
                    float(self.params.get("q_theta_dot", 20.0)),
                    float(self.params.get("q_v", 80.0)),
                    float(self.params.get("q_x", 30.0)),
                    float(self.params.get("q_z", 8.0)),
                ]
            ).astype(np.float64)
        else:
            self._mpc_nx = 4
            Q_mpc = np.diag(
                [
                    float(self.params.get("q_theta", 100.0)),
                    float(self.params.get("q_theta_dot", 20.0)),
                    float(self.params.get("q_v", 80.0)),
                    float(self.params.get("q_x", 30.0)),
                ]
            ).astype(np.float64)
        R = np.asarray([[float(self.params.get("r", 1.0))]], dtype=np.float64)
        P_term = None
        if bool(self.params.get("terminal_lqr", True)):
            P_term = dlqr_riccati(self._mpc_A, self._mpc_B, Q_mpc, R)
        self._mpc_N = N
        self._mpc = build_mpc_matrices(
            self._mpc_A,
            self._mpc_B,
            N,
            Q_mpc,
            R,
            P_term,
            r_rate=float(self.params.get("r_rate", 0.5)),
        )
        self._mpc_u_prev = 0.0

    # ── 矢状面 (QP) ──
    def _sagittal_u(self, x: np.ndarray, v_ref: float) -> float:
        nx = self._mpc_nx
        mpc = self._mpc
        if nx == 5:
            self._z_int += (x[2] - v_ref) * self.dt
            v_ref_eff = v_ref - float(self.params.get("integral_gain", 1.5)) * self._z_int
            x_aug = np.concatenate([x, [self._z_int]])
            x_ref = np.array([0.0, 0.0, v_ref_eff, 0.0, 0.0], dtype=np.float64)
            c = np.zeros(self._mpc_N * nx)
            for k in range(1, self._mpc_N + 1):
                c[(k - 1) * nx + 4] = -self.dt * v_ref_eff * k
            cN = np.zeros(nx)
            cN[4] = -self.dt * v_ref_eff * self._mpc_N
        else:
            x_aug = x
            x_ref = np.array([0.0, 0.0, v_ref, 0.0], dtype=np.float64)
            c = None
            cN = np.zeros(nx)
        X_ref = np.tile(x_ref, (self._mpc_N, 1)).reshape(-1)
        pred = mpc["P"] @ x_aug + (c if c is not None else 0.0)
        g = mpc["H"].T @ mpc["Q_bar"] @ (pred - X_ref)
        if mpc["P_term"] is not None:
            g = g + mpc["H_N"].T @ mpc["P_term"] @ (mpc["P_N"] @ x_aug + cN - x_ref)
        if float(self.params.get("r_rate", 0.5)) > 0.0:
            g = g.copy()
            g[0] -= float(self.params.get("r_rate", 0.5)) * self._mpc_u_prev
        d_vals: list[float] = []
        theta_max = float(self.params.get("theta_max", 0.35))
        v_max = float(self.params.get("v_max", 2.5))
        for k in range(self._mpc_N):
            Pk = mpc["P"][k * nx : (k + 1) * nx]
            for pi, lim in ((0, theta_max), (2, v_max)):
                d_vals.append(lim - Pk[pi, :] @ x_aug)
                d_vals.append(lim + Pk[pi, :] @ x_aug)
        d_arr = np.array(d_vals, dtype=np.float64)
        t0 = time.perf_counter()
        u0, lam, _stats = solve_qp(
            mpc, g, d_arr, float(self.params.get("u_max", 30.0)), self._lam0
        )
        self._solve_time_ms = (time.perf_counter() - t0) * 1000.0
        self._lam0 = lam
        self._mpc_u_prev = u0
        return u0

    # ── 共享 act 骨架 (腿目标/偏航/轮映射 → a8) ──
    def act(self, sensors: dict[str, Any], cmds: np.ndarray) -> np.ndarray:
        theta = float(sensors["theta"])
        theta_dot = float(sensors["theta_dot"])
        v = float(sensors["v"])
        base_z = float(sensors["base_z"])
        omega_z = float(sensors["omega_z"])
        v_ref_raw = float(cmds[0])
        vyaw = float(cmds[2])
        h_cmd_raw = float(cmds[4]) if len(cmds) >= 5 else 0.518

        # 高度命令平滑 (P3+)
        if self.phase >= 3:
            if self._h_cmd_smooth is None:
                self._h_cmd_smooth = h_cmd_raw
            hs = float(self.params.get("height_smoothing", 0.97))
            self._h_cmd_smooth = hs * self._h_cmd_smooth + (1.0 - hs) * h_cmd_raw
            h_cmd = self._h_cmd_smooth
        else:
            h_cmd = h_cmd_raw

        # 指令斜坡 (★ 首步从 0 起步 — 修复经典轨 cmd_ramp 首步跳变 bug)
        ramp = float(self.params.get("cmd_ramp_s", 1.5))
        if ramp > 0:
            alpha = 1.0 - self.dt / ramp
            if not self._cmd_smooth_init:
                self._cmd_smooth = 0.0  # 从 0 起步, 避免首步直接跳到命令 (起步失衡)
                self._cmd_smooth_init = True
            self._cmd_smooth = alpha * self._cmd_smooth + (1.0 - alpha) * v_ref_raw
            v_ref = self._cmd_smooth
        else:
            v_ref = v_ref_raw

        # 状态 x
        self._xpos += v * self.dt
        x = np.array([theta, theta_dot, v, self._xpos], dtype=np.float64)
        u = self.sign * self._sagittal_u(x, v_ref)

        # 轮速命令 + 平滑
        w_new = float(np.clip(u, -self.wheel_vel_max, self.wheel_vel_max))
        self._w_c = self._smoothing * self._w_c + (1.0 - self._smoothing) * w_new

        # 偏航差分
        delta_w = (self.track / WHEEL_R) * vyaw + self.k_yaw * (vyaw - omega_z)
        w_L = -self._w_c - delta_w / 2.0
        w_R = +self._w_c - delta_w / 2.0
        w_L = float(np.clip(w_L, -self.wheel_vel_max, self.wheel_vel_max))
        w_R = float(np.clip(w_R, -self.wheel_vel_max, self.wheel_vel_max))

        # 腿目标 (P3/P4 高度控制)
        q_leg = standing_joint_targets().copy()
        if self.phase >= 3:
            if self._prev_base_z is None:
                self._prev_base_z = base_z
            base_z_dot = (base_z - self._prev_base_z) / self.dt
            self._prev_base_z = base_z
            err = h_cmd - base_z
            if -0.08 < err < 0.08:
                self._height_int += err * self.dt
            knee_adj = (
                float(self.params.get("height_kp", 0.8)) * err
                + float(self.params.get("height_kd", 0.3)) * base_z_dot
                + float(self.params.get("height_ki", 0.2)) * self._height_int
            )
            knee_adj = float(np.clip(knee_adj, -0.15, 0.15))
            q_leg[2] += knee_adj  # L_knee
            q_leg[5] -= knee_adj  # R_knee
            q_leg[1] = max(q_leg[1], 0.05)
            q_leg[4] = min(q_leg[4], -0.05)
            q_leg[2] = float(np.clip(q_leg[2], -0.45, -0.12))
            q_leg[5] = float(np.clip(q_leg[5], 0.12, 0.45))
            leg_kp = float(self.params.get("leg_balance_kp", -0.25))
            q_leg[1] += leg_kp * theta
            q_leg[4] += leg_kp * theta

        # 映射到归一化动作 a8 (RL 空间)
        a8 = np.zeros(8, dtype=np.float64)
        for i in range(6):
            a8[i] = (q_leg[i] - DEFAULT_LEG_ANGLES[i]) / (_FLIP[i] * self.action_scale)
        a8[6] = w_L / (self.wsa)
        a8[7] = w_R / (-self.wsa)
        return a8

    def reset(self) -> None:
        """复位内部状态 (交互 Backspace / 评估跨 episode / 训练 reset)."""
        self._w_c = 0.0
        self._xpos = 0.0
        self._z_int = 0.0
        self._height_int = 0.0
        self._cmd_smooth = 0.0
        self._cmd_smooth_init = False
        self._prev_base_z = None
        self._h_cmd_smooth = None
        self._lam0 = None
        self._mpc_u_prev = 0.0

    # ── 诊断 ──
    @property
    def wheel_cmd(self) -> float:
        return float(self._w_c)

    @property
    def last_solve_ms(self) -> float:
        return self._solve_time_ms
