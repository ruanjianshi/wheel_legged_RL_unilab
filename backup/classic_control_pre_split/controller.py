"""BalanceController: 矢状面(LQR/MPC) + 偏航差分 + 腿控 → 8D 归一化动作.

输入 sensors dict (单 env), cmds (5D); 输出 a8 (env.step 归一化动作空间).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from scripts.classic_control.config import (
    DEFAULT_LEG_ANGLES,
    WHEEL_R,
    build_cfg_and_overrides,
    task_key_for_phase,
)
from scripts.classic_control.config import (
    ROOT as _CFG_ROOT,
)
from scripts.classic_control.leg_control import (
    fk_legs,
    height_targets,
    joints_from_l0,
    standing_joint_targets,
)

_FLIP = np.array([1, 1, -1, 1, -1, 1, 1, -1], dtype=np.float64)


class BalanceController:
    """kind: 'lqr' | 'mpc';  phase: 1..4."""

    def __init__(
        self,
        kind: str,
        phase: int,
        A_d: np.ndarray,
        B_d: np.ndarray,
        params: dict[str, Any] | None = None,
        action_scale: float = 0.6,
        wheel_action_scale: float = 10.0,
        dt: float = 0.01,
    ) -> None:
        self.kind = kind
        self.phase = phase
        self.A_d = A_d
        self.B_d = B_d
        # ★ 默认参数来自 conf/classic_control YAML; CLI 覆盖合并其上
        params = {} if params is None else params
        self.cfg, merged = build_cfg_and_overrides(task_key_for_phase(phase), params)
        self._merged = merged
        self.dt = dt
        self.action_scale = action_scale
        self.wsa = wheel_action_scale

        self._lqr_K: np.ndarray | None = None
        self._lqr_K_aug: np.ndarray | None = None
        self._mpc_G_cache: Any = None
        self._integrate_sagittal()

        # 运行时状态
        self._w_c = 0.0  # 共同轮角速度 (rad/s)
        self._smoothing = float(merged.get("smoothing", 0.85))  # ★ 轮速命令平滑 (消 yaw/振荡)
        self._cmd_smooth = 0.0  # 指令斜坡状态 (v_ref 平滑)
        self._cmd_smooth_init = False
        self._height_int = 0.0  # 高度积分 (P3 膝伺服消稳态误差)
        self._h_cmd_smooth = None  # 高度命令平滑状态
        self._xpos = 0.0  # 积分轮位置
        self._z_int = 0.0  # LQI 速度误差积分
        self._prev_base_z: float | None = None
        self._lam0: np.ndarray | None = None  # MPC warm-start
        self._solve_time_ms = 0.0

        # 参数 (可覆盖)
        self.track = float(merged.get("track_width", self.cfg.track_width))
        self.k_yaw = float(merged.get("k_yaw", self.cfg.k_yaw))
        self.sign = float(merged.get("sign", 1.0))  # 控制方向标定
        self.wheel_vel_max = float(self.cfg.wheel_vel_max)

    def reset(self) -> None:
        """重置控制器内部状态 (交互 Backspace 重置用)."""
        self._w_c = 0.0
        self._xpos = 0.0
        self._z_int = 0.0
        self._height_int = 0.0
        self._cmd_smooth = 0.0
        self._cmd_smooth_init = False
        self._prev_base_z = None
        self._lam0 = None
        self._h_cmd_smooth = None

    def _integrate_sagittal(self) -> None:
        """矢状面控制器. P1 用权重直接作增益 (实测最优, 修正方向后 15s 平衡)."""
        if self.kind == "lqr":
            # ★ 修正方向后实测最优: 权重直接作增益 (非 dlqr), 输出 = 轮速命令 (rad/s)
            self._lqr_K = np.array(
                [self.cfg.q_theta, self.cfg.q_theta_dot, self.cfg.q_v, self.cfg.q_x],
                dtype=np.float64,
            )
            if self.phase >= 2:
                # P2 指令: 加速度误差积分 z=∫(v-v_ref)dt (LQI 风格)
                self._lqr_K_aug = np.array(
                    [
                        self.cfg.q_theta,
                        self.cfg.q_theta_dot,
                        self.cfg.q_v,
                        self.cfg.q_x,
                        self.cfg.q_z,
                    ],
                    dtype=np.float64,
                )
        elif self.kind == "mpc":
            from scripts.classic_control.dynamics import velocity_command_model
            from scripts.classic_control.mpc import (
                build_mpc_matrices,
                dlqr_riccati,
            )

            # ★ 轮速命令模型 (w_c convention): v̇=(u·R−v)/τ, θ̈=αθ+β·v̇.
            #   实测 α/τ + 解析 β; 与 LQR 同输出 (轮速命令 rad/s), 消除了
            #   "加速模型输出当速度命令用" 的单位错配 (旧 MPC 失败根因).
            #   模型来源优先级: 参数 A_d/B_d → config model_file → 解析模型.
            loaded = False
            # ★ P1 平衡用解析模型 (黑箱 v-pole 1.02 引入漂移 0.4>0.2 不达标);
            #   P2+ 速度/地形用黑箱 (速度跟踪更准). 可用参数 A_d/B_d 强制注入.
            use_bb = self.phase >= 2
            if self._merged.get("A_d") is not None and self._merged.get("B_d") is not None:
                self._mpc_A = np.asarray(self._merged["A_d"], dtype=np.float64)
                self._mpc_B = np.asarray(self._merged["B_d"], dtype=np.float64).reshape(-1, 1)
                loaded = True
            elif use_bb and self.cfg.model_file:
                _mf = _CFG_ROOT / self.cfg.model_file
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
                #   z[k+1]=z[k]+dt·v[k]; 仿射项 −dt·v_ref·k 预测时补偿.
                #   配合黑箱模型 (稳态 u→v 增益准确) 使用; 解析模型 v 行失配会抖.
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
                self._mpc_A,
                self._mpc_B,
                N,
                Q_mpc,
                R,
                P_term,
                r_rate=self.cfg.r_rate,
            )
            self._mpc_u_prev = 0.0
        else:  # pragma: no cover
            raise ValueError(f"unknown kind {self.kind}")

    # ── 矢状面控制 u (轮速命令 rad/s) ──
    def _sagittal_u(self, x: np.ndarray, v_ref: float) -> float:
        if self.kind == "lqr":
            if self.phase >= 2:
                # P2 指令: q_v·(v-v_ref) 直接驱动速度 + 积分 z 消稳态误差
                self._z_int += (x[2] - v_ref) * self.dt
                return float(
                    -(
                        self.cfg.q_theta * x[0]
                        + self.cfg.q_theta_dot * x[1]
                        + self.cfg.q_v * (x[2] - v_ref)
                        + self.cfg.q_z * self._z_int
                    )
                )
            return float(-self._lqr_K @ x)  # type: ignore[operator]
        # MPC: 滚动时域求解轮速命令 u (rad/s) — 与 LQR 同为轮速命令, 方向映射一致.
        # ★ 预计算 (build_mpc_matrices) + 向量化 QP → 亚毫秒级, 满足 10ms 周期.
        from scripts.classic_control.mpc import solve_qp

        mpc = self._mpc
        nx = self._mpc_nx
        if nx == 5:
            # P2 积分: z=∫(v−v_ref) 作参考偏置 (v_ref_eff = v_ref − k_i·z), 模型内增广
            #   避免积分在模型外抖动; 仿射偏移 c[k, z行]=−dt·v_ref_eff·k
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
            # 变化率线性项: −r_rate·u₋₁·e₀ (惩罚 (u₀−u₋₁)² 的一次项)
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
        self._mpc_u_prev = u0  # 变化率惩罚参考上一拍输出
        return u0

    # ── 主入口 ──
    def act(self, sensors: dict[str, Any], cmds: np.ndarray) -> np.ndarray:
        theta = float(sensors["theta"])
        theta_dot = float(sensors["theta_dot"])
        # ★ MPC 与 LQR 共用同一状态 v=linvel[0] (LQR 实测最优; v_wheel 会破坏闭环)
        v = float(sensors["v"])
        base_z = float(sensors["base_z"])
        omega_z = float(sensors["omega_z"])
        v_ref_raw = float(cmds[0])
        vyaw = float(cmds[2])
        h_cmd_raw = float(cmds[4]) if len(cmds) >= 5 else 0.518
        # ★ 高度命令平滑 (Q/E 快速按 → 高度阶跃 → 膝猛变 → 不稳)
        if self.phase >= 3:
            if self._h_cmd_smooth is None:
                self._h_cmd_smooth = h_cmd_raw
            self._h_cmd_smooth = (
                self.cfg.height_smoothing * self._h_cmd_smooth
                + (1 - self.cfg.height_smoothing) * h_cmd_raw
            )
            h_cmd = self._h_cmd_smooth
        else:
            h_cmd = h_cmd_raw
        # ★ 指令斜坡: v_ref 平滑过渡 (阶跃→暴力刹车→倒; 斜坡 1-2s 消除)
        if self.cfg.cmd_ramp_s > 0:
            alpha = 1.0 - self.dt / self.cfg.cmd_ramp_s
            self._cmd_smooth = (
                v_ref_raw
                if not self._cmd_smooth_init
                else alpha * self._cmd_smooth + (1 - alpha) * v_ref_raw
            )
            self._cmd_smooth_init = True
            v_ref = self._cmd_smooth
        else:
            v_ref = v_ref_raw

        # 状态 x
        self._xpos += v * self.dt
        x = np.array([theta, theta_dot, v, self._xpos], dtype=np.float64)

        u = self.sign * self._sagittal_u(x, v_ref)

        # ★ 直接轮速命令 (rad/s, 不积分): _sagittal_u 返回 -(weights@x) = 前倾→负 (后驱)
        w_new = float(np.clip(u, -self.wheel_vel_max, self.wheel_vel_max))
        # ★ 平滑 (α=0.85 实测最优, 消除轮命令突变激发的 yaw/振荡)
        self._w_c = self._smoothing * self._w_c + (1.0 - self._smoothing) * w_new

        # 偏航差分
        delta_w = (self.track / WHEEL_R) * vyaw + self.k_yaw * (vyaw - omega_z)
        # ★ 修正根因: L 轮取负 (镜像轴 + 方向); yaw 差分 R 侧取负 (前向差速)
        #   w_L=-(w_c+δ), w_R=(w_c-δ) → L前向快/R前向慢 → 转向
        w_L = -self._w_c - delta_w / 2.0
        w_R = +self._w_c - delta_w / 2.0
        w_L = float(np.clip(w_L, -self.wheel_vel_max, self.wheel_vel_max))
        w_R = float(np.clip(w_R, -self.wheel_vel_max, self.wheel_vel_max))

        # 腿目标
        if self.phase <= 2:
            q_leg = standing_joint_targets()
        else:
            # ★ P3/P4 腿长控制: 补偿基座 + 膝高度伺服 (L0-IK 受 kp=60 下垂限制不可靠)
            #   膝目标随高度误差调整 (符号: L膝负方向弯=低; 双膝对称)
            q_leg = standing_joint_targets().copy()
            if self._prev_base_z is None:
                self._prev_base_z = base_z
            base_z_dot = (base_z - self._prev_base_z) / self.dt
            self._prev_base_z = base_z
            err = h_cmd - base_z
            # ★ 高度积分 anti-windup: 超出可达范围时截断积分 (防持续下沉)
            if -0.08 < err < 0.08:
                self._height_int += err * self.dt
            knee_adj = (
                self.cfg.height_kp * err
                + self.cfg.height_kd * base_z_dot
                + self.cfg.height_ki * self._height_int
            )
            knee_adj = float(np.clip(knee_adj, -0.15, 0.15))  # ★ 膝调整幅度限制 (防猛变)
            q_leg[2] += knee_adj  # L_knee
            q_leg[5] -= knee_adj  # R_knee (镜像)
            # 安全钳位 (防 kp=60 下垂塌陷)
            q_leg[1] = max(q_leg[1], 0.05)  # L_pitch > 0.05
            q_leg[4] = min(q_leg[4], -0.05)  # R_pitch < -0.05
            q_leg[2] = float(np.clip(q_leg[2], -0.45, -0.12))  # L_knee 安全区
            q_leg[5] = float(np.clip(q_leg[5], 0.12, 0.45))  # R_knee 安全区
            # ★ 腿协助平衡: hip_pitch 随倾角偏移 CoM (双腿同向)
            q_leg[1] += self.cfg.leg_balance_kp * theta
            q_leg[4] += self.cfg.leg_balance_kp * theta

        # 映射到归一化动作
        a8 = np.zeros(8, dtype=np.float64)
        for i in range(6):
            a8[i] = (q_leg[i] - DEFAULT_LEG_ANGLES[i]) / (_FLIP[i] * self.action_scale)
        a8[6] = w_L / (self.wsa)
        a8[7] = w_R / (-self.wsa)
        return a8

    @property
    def last_solve_ms(self) -> float:
        return self._solve_time_ms

    @property
    def wheel_cmd(self) -> float:
        """当前施加的轮速命令 w_c (rad/s, 平滑后). 供辨识/诊断记录."""
        return float(self._w_c)
