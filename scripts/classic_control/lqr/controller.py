"""LQR 平衡控制器 (独立任务轨) — 状态反馈, 权重直接作增益.

与 MPC 完全独立 (仅共享 common 只读机器人接口). 配置 conf/lqr (自包含)。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.classic_control.common.base import BaseController
from scripts.classic_control.common.config import task_key_for_phase


class LqrController(BaseController):
    """LQR: u = −K·x (P1), LQI 积分增广 z=∫(v−v_ref) (P2+)."""

    def _build_config(self, phase: int, params: dict[str, Any]) -> tuple[Any, dict]:
        from scripts.classic_control.lqr.config import build_config

        return build_config(task_key_for_phase(phase), params)

    def _integrate_sagittal(self) -> None:
        # ★ 修正方向后实测最优: 权重直接作增益 (非 dlqr), 输出 = 轮速命令 (rad/s)
        self._lqr_K = np.array(
            [self.cfg.q_theta, self.cfg.q_theta_dot, self.cfg.q_v, self.cfg.q_x],
            dtype=np.float64,
        )
        self._lqr_K_aug: np.ndarray | None = None
        if self.phase >= 2:
            self._lqr_K_aug = np.array(
                [self.cfg.q_theta, self.cfg.q_theta_dot, self.cfg.q_v, self.cfg.q_x, self.cfg.q_z],
                dtype=np.float64,
            )

    def _sagittal_u(self, x: np.ndarray, v_ref: float) -> float:
        if self.phase >= 2:
            # P2 指令: q_v·(v−v_ref) 直接驱动速度 + 积分 z 消稳态误差
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
