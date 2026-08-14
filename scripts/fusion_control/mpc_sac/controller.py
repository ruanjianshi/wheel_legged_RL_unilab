"""MPC×SAC 融合控制 — 高层控制器 (AugMPC 层次化: SAC 决策 → MPC 执行).

MpcSacController: 组合 高层 SAC 策略 (policy_loader.HighLevelPolicy) +
低层分支 MPC (mpc.controller.MpcController)。

act(sensors, des_cmd) → a8:
  1. build_high_obs(sensors, des, prev_a)  高层状态
  2. policy.select_action(obs) → 归一化动作
  3. denorm_action(a) → 实际命令 cmd
  4. low_mpc.act(sensors, cmd) → a8 (RL 空间 8D)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from scripts.fusion_control.mpc_sac.config import build_config
from scripts.fusion_control.mpc_sac.mpc.controller import MpcController
from scripts.fusion_control.mpc_sac.obs import action_dim, build_high_obs, cmd_dim, denorm_action

PolicyFactory = Callable[[], Any]


class MpcSacController:
    """融合控制器: 高层 SAC 发命令, 低层 MPC 执行."""

    def __init__(
        self,
        task_key: str = "walk_flat",
        params: dict[str, Any] | None = None,
        action_scale: float = 0.6,
        wheel_action_scale: float = 10.0,
        dt: float = 0.01,
        policy_factory: PolicyFactory | None = None,
        policy=None,
    ) -> None:
        self.task_key = task_key
        self.dt = dt
        self.cfg, self._merged = build_config(task_key, params)
        self.phase = int(self.cfg.phase_flat if task_key == "walk_flat" else self.cfg.phase_rough)

        # 低层 MPC (分支自带)
        self.low = MpcController(
            self.phase,
            params=params,
            action_scale=action_scale,
            wheel_action_scale=wheel_action_scale,
            dt=dt,
        )
        self.action_dim = action_dim(task_key)
        self._cmd_dim = cmd_dim(task_key)
        self._policy = policy
        self._policy_factory = policy_factory
        self._prev_a = np.zeros(self.action_dim, dtype=np.float64)
        self._des = np.zeros(self._cmd_dim, dtype=np.float64)

    def _load_policy(self) -> None:
        if self._policy is None and self._policy_factory is not None:
            self._policy = self._policy_factory()

    def reset(self) -> None:
        self.low.reset()
        self._prev_a = np.zeros(self.action_dim, dtype=np.float64)
        self._des = np.zeros(self._cmd_dim, dtype=np.float64)

    def act(self, sensors: dict[str, Any], des_cmd: np.ndarray) -> np.ndarray:
        """des_cmd: 期望命令 (5D flat / 4D rough). 返回 a8."""
        self._load_policy()
        self._des = np.asarray(des_cmd, dtype=np.float64).reshape(-1)
        obs = build_high_obs(sensors, self._des, self._prev_a, self.cfg, self.task_key)
        a_hi = self._policy.select_action(obs, deterministic=True)
        cmd = denorm_action(a_hi, self.cfg, self.task_key, self._des)
        a8 = self.low.act(sensors, cmd)
        self._prev_a = a_hi
        return a8

    @property
    def last_cmd(self) -> np.ndarray:
        """高层实际发出的命令 (des + 残差校正)."""
        return denorm_action(self._prev_a, self.cfg, self.task_key, self._des)

    @property
    def last_solve_ms(self) -> float:
        return self.low.last_solve_ms
