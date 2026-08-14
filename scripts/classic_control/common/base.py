"""经典控制共享控制器基类 — 机器人接口 (腿目标/偏航/动作映射) 只读共享.

LQR / MPC 两条独立任务轨共用此基类 (开发规范 §3.2 共享只读资源):
- 公共 act() 骨架: 指令斜坡 / 高度平滑 / 偏航差分 / 腿目标 / 8D 动作映射
- 算法差异点 (`_integrate_sagittal` / `_sagittal_u`) 由子类实现 → 两轨互不干涉

子类必须实现:
- `_build_config(phase, params)` → (cfg dataclass, merged dict)
- `_integrate_sagittal()`  构造控制器 (增益/MPC 矩阵)
- `_sagittal_u(x, v_ref)`  矢状面控制命令 (轮速命令 rad/s)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.classic_control.common.config import DEFAULT_LEG_ANGLES, WHEEL_R
from scripts.classic_control.common.leg_control import standing_joint_targets

_FLIP = np.array([1, 1, -1, 1, -1, 1, 1, -1], dtype=np.float64)


class BaseController:
    """kind 由子类决定;  phase: 1..4."""

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
        # ★ 算法配置来自各自 conf (LQR→conf/lqr, MPC→conf/mpc)
        self.cfg, self._merged = self._build_config(phase, params)

        self._smoothing = float(self._merged.get("smoothing", 0.85))  # 轮速命令平滑
        self.track = float(self._merged.get("track_width", self.cfg.track_width))
        self.k_yaw = float(self._merged.get("k_yaw", self.cfg.k_yaw))
        self.sign = float(self._merged.get("sign", 1.0))  # 控制方向标定
        self.wheel_vel_max = float(self.cfg.wheel_vel_max)

        # 运行时状态
        self._w_c = 0.0  # 共同轮角速度 (rad/s)
        self._cmd_smooth = 0.0  # 指令斜坡状态
        self._cmd_smooth_init = False
        self._height_int = 0.0  # 高度积分 (P3)
        self._h_cmd_smooth: float | None = None
        self._xpos = 0.0  # 积分轮位置
        self._z_int = 0.0  # LQI 速度误差积分
        self._prev_base_z: float | None = None

        self._integrate_sagittal()  # 子类实现

    # ── 子类需实现 ──
    def _build_config(self, phase: int, params: dict[str, Any]) -> tuple[Any, dict]:
        raise NotImplementedError

    def _integrate_sagittal(self) -> None:
        raise NotImplementedError

    def _sagittal_u(self, x: np.ndarray, v_ref: float) -> float:
        raise NotImplementedError

    # ── 共享 ──
    def reset(self) -> None:
        """重置控制器内部状态 (交互 Backspace 重置 / 评估跨 episode)."""
        self._w_c = 0.0
        self._xpos = 0.0
        self._z_int = 0.0
        self._height_int = 0.0
        self._cmd_smooth = 0.0
        self._cmd_smooth_init = False
        self._prev_base_z = None
        self._h_cmd_smooth = None

    def act(self, sensors: dict[str, Any], cmds: np.ndarray) -> np.ndarray:
        theta = float(sensors["theta"])
        theta_dot = float(sensors["theta_dot"])
        v = float(sensors["v"])
        base_z = float(sensors["base_z"])
        omega_z = float(sensors["omega_z"])
        v_ref_raw = float(cmds[0])
        vyaw = float(cmds[2])
        h_cmd_raw = float(cmds[4]) if len(cmds) >= 5 else 0.518
        # ★ 高度命令平滑 (阶跃 → 膝猛变 → 不稳)
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
        # ★ 指令斜坡: v_ref 平滑过渡
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

        # 轮速命令 (rad/s) + 平滑 (消 yaw/振荡)
        w_new = float(np.clip(u, -self.wheel_vel_max, self.wheel_vel_max))
        self._w_c = self._smoothing * self._w_c + (1.0 - self._smoothing) * w_new

        # 偏航差分
        delta_w = (self.track / WHEEL_R) * vyaw + self.k_yaw * (vyaw - omega_z)
        w_L = -self._w_c - delta_w / 2.0
        w_R = +self._w_c - delta_w / 2.0
        w_L = float(np.clip(w_L, -self.wheel_vel_max, self.wheel_vel_max))
        w_R = float(np.clip(w_R, -self.wheel_vel_max, self.wheel_vel_max))

        # 腿目标
        if self.phase <= 2:
            q_leg = standing_joint_targets()
        else:
            q_leg = standing_joint_targets().copy()
            if self._prev_base_z is None:
                self._prev_base_z = base_z
            base_z_dot = (base_z - self._prev_base_z) / self.dt
            self._prev_base_z = base_z
            err = h_cmd - base_z
            if -0.08 < err < 0.08:
                self._height_int += err * self.dt
            knee_adj = (
                self.cfg.height_kp * err
                + self.cfg.height_kd * base_z_dot
                + self.cfg.height_ki * self._height_int
            )
            knee_adj = float(np.clip(knee_adj, -0.15, 0.15))
            q_leg[2] += knee_adj  # L_knee
            q_leg[5] -= knee_adj  # R_knee (镜像)
            q_leg[1] = max(q_leg[1], 0.05)  # L_pitch 安全区
            q_leg[4] = min(q_leg[4], -0.05)  # R_pitch 安全区
            q_leg[2] = float(np.clip(q_leg[2], -0.45, -0.12))
            q_leg[5] = float(np.clip(q_leg[5], 0.12, 0.45))
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
    def wheel_cmd(self) -> float:
        """当前施加的轮速命令 w_c (rad/s, 平滑后). 供辨识/诊断记录."""
        return float(self._w_c)
