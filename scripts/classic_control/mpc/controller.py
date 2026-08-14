"""MPC 平衡控制器 (独立任务轨) — 线性 MPC (Hildreth QP) + LQR 末端代价.

与 LQR 完全独立 (仅共享 common 只读机器人接口). 配置 conf/mpc (自包含)。
模型来源优先级: 参数 A_d/B_d → config model_file (黑箱) → 解析 (α,β,τ)。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from scripts.classic_control.common.base import BaseController
from scripts.classic_control.common.config import ROOT, WHEEL_R, task_key_for_phase


class MpcController(BaseController):
    """MPC: 滚动时域求解轮速命令 (rad/s), 与 LQR 同输出约定."""

    def __init__(
        self,
        phase: int,
        params: dict[str, Any] | None = None,
        action_scale: float = 0.6,
        wheel_action_scale: float = 10.0,
        dt: float = 0.01,
    ) -> None:
        self._lam0: np.ndarray | None = None
        self._solve_time_ms = 0.0
        self._mpc_stats: dict = {}
        super().__init__(phase, params, action_scale, wheel_action_scale, dt)

    def _build_config(self, phase: int, params: dict[str, Any]) -> tuple[Any, dict]:
        from scripts.classic_control.mpc.config import build_config

        return build_config(task_key_for_phase(phase), params)

    def reset(self) -> None:
        super().reset()
        self._lam0 = None
        self._mpc_u_prev = 0.0

    # ── MPC 专用状态 ──
    @property
    def last_solve_ms(self) -> float:
        return self._solve_time_ms

    # ── 构造 ──
    def _integrate_sagittal(self) -> None:
        from scripts.classic_control.common.dynamics import velocity_command_model
        from scripts.classic_control.mpc.qp import (
            build_mpc_matrices,
            dlqr_riccati,
        )

        # ★ 轮速命令模型 (w_c convention): v̇=(−uR−v)/τ, θ̈=αθ+β·v̇.
        #   与 LQR 同输出 (轮速命令 rad/s), 消除"加速模型输出当速度命令"单位错配.
        loaded = False
        # ★ P1 平衡用解析模型 (黑箱 v-pole 1.02 引入漂移 0.4>0.2 不达标);
        #   P2+ 速度/地形用黑箱 (速度跟踪更准). 参数 A_d/B_d 强制注入.
        use_bb = self.phase >= 2
        if self._merged.get("A_d") is not None and self._merged.get("B_d") is not None:
            self._mpc_A = np.asarray(self._merged["A_d"], dtype=np.float64)
            self._mpc_B = np.asarray(self._merged["B_d"], dtype=np.float64).reshape(-1, 1)
            loaded = True
        elif use_bb and self.cfg.model_file:
            _mf = ROOT / self.cfg.model_file
            if _mf.exists():
                _d = np.load(_mf)
                self._mpc_A = np.asarray(_d["A_d"], dtype=np.float64)
                self._mpc_B = np.asarray(_d["B_d"], dtype=np.float64).reshape(-1, 1)
                loaded = True
        if not loaded:
            self._mpc_A, self._mpc_B = velocity_command_model(
                self.cfg.alpha, self.cfg.beta, self.cfg.tau, self.dt, WHEEL_R
            )[2:]
        N = self.cfg.mpc_horizon
        if self.phase >= 2:
            # ★ P2 速度跟踪: 模型内积分增广 z=∫(v−v_ref) (offset-free MPC).
            #   配合黑箱模型 (稳态 u→v 增益准确); 解析模型 v 行失配会抖.
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
                    self.cfg.q_theta,
                    self.cfg.q_theta_dot,
                    self.cfg.q_v,
                    self.cfg.q_x,
                    self.cfg.q_z,
                ]
            ).astype(np.float64)
        else:
            self._mpc_nx = 4
            Q_mpc = np.diag(
                [self.cfg.q_theta, self.cfg.q_theta_dot, self.cfg.q_v, self.cfg.q_x]
            ).astype(np.float64)
        R = np.asarray([[self.cfg.r]], dtype=np.float64)
        P_term = None
        if self.cfg.terminal_lqr:
            P_term = dlqr_riccati(self._mpc_A, self._mpc_B, Q_mpc, R)
        self._mpc_N = N
        self._mpc_Q = Q_mpc
        self._mpc_R = R
        self._mpc = build_mpc_matrices(
            self._mpc_A, self._mpc_B, N, Q_mpc, R, P_term, r_rate=self.cfg.r_rate
        )
        self._mpc_u_prev = 0.0

    # ── 矢状面控制 ──
    def _sagittal_u(self, x: np.ndarray, v_ref: float) -> float:
        from scripts.classic_control.mpc.qp import solve_qp

        mpc = self._mpc
        nx = self._mpc_nx
        if nx == 5:
            # P2 积分: z=∫(v−v_ref) 作参考偏置 (v_ref_eff = v_ref − k_i·z)
            self._z_int += (x[2] - v_ref) * self.dt
            v_ref_eff = v_ref - self.cfg.integral_gain * self._z_int
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
        if self.cfg.r_rate > 0.0:
            g = g.copy()
            g[0] -= self.cfg.r_rate * self._mpc_u_prev
        d_vals: list[float] = []
        for k in range(self._mpc_N):
            Pk = mpc["P"][k * nx : (k + 1) * nx]
            for hi, pi, lim in ((0, 0, self.cfg.theta_max), (2, 2, self.cfg.v_max)):
                d_vals.append(lim - Pk[pi, :] @ x_aug)
                d_vals.append(lim + Pk[pi, :] @ x_aug)
        d_arr = np.array(d_vals, dtype=np.float64)
        t0 = time.perf_counter()
        u0, lam, stats = solve_qp(mpc, g, d_arr, self.cfg.u_max, self._lam0)
        self._solve_time_ms = (time.perf_counter() - t0) * 1000.0
        self._mpc_stats = stats
        self._lam0 = lam
        self._mpc_u_prev = u0
        return u0
