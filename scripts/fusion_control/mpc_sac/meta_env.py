"""MPC×SAC 融合分支 — 高层训练 meta-env (AugMPC 层次化: MPC 固定执行器).

numpy 向量化 env: obs (N, obs_dim), action (N, action_dim) 命令, reward/done (N,)。
每 env 一个分支 MPC (各自内部状态: lam0 热启动/平滑/积分不串扰)。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.fusion_control.mpc_sac.config import MpcSacConfig, build_config
from scripts.fusion_control.mpc_sac.env import build_env, read_sensors_batch
from scripts.fusion_control.mpc_sac.mpc.controller import MpcController
from scripts.fusion_control.mpc_sac.obs import (
    action_dim,
    build_high_obs_batch,
    cmd_dim,
    compute_reward,
    denorm_action,
    obs_dim,
    sample_desired,
)


class MpcSacMetaEnv:
    """高层 SAC 训练环境: 策略输出命令, 低层 MPC 执行."""

    def __init__(
        self,
        task_key: str = "walk_flat",
        num_envs: int = 64,
        params: dict[str, Any] | None = None,
        cfg: MpcSacConfig | None = None,
        seed: int = 42,
        lock_hip_roll: bool = True,
    ) -> None:
        self.task_key = task_key
        self.num_envs = num_envs
        self.cfg, self._merged = (cfg, params) if isinstance(cfg, MpcSacConfig) else build_config(task_key, params)
        if cfg is None:
            self.cfg, self._merged = build_config(task_key, params)
        self.phase = int(self.cfg.phase_flat if task_key == "walk_flat" else self.cfg.phase_rough)
        self.action_dim = action_dim(task_key)
        self.cmd_dim = cmd_dim(task_key)
        self.obs_dim = obs_dim(task_key, self.cfg)
        self.rng = np.random.default_rng(seed)

        self.low_env = build_env(task_key, num_envs, cmd=None, lock_hip_roll=lock_hip_roll)
        ascale = float(self.low_env._cfg.control_config.action_scale)
        wscale = float(self.low_env._cfg.control_config.wheel_action_scale)
        dt = float(self.low_env._cfg.ctrl_dt)
        self.dt = dt
        low_params = {k: v for k, v in self._merged.items()}
        self.low: list[MpcController] = [
            MpcController(
                self.phase, params=low_params, action_scale=ascale, wheel_action_scale=wscale, dt=dt
            )
            for _ in range(num_envs)
        ]

        self._des = np.zeros((num_envs, self.cmd_dim), dtype=np.float64)
        self._prev_a = np.zeros((num_envs, self.action_dim), dtype=np.float64)
        self._steps = np.zeros(num_envs, dtype=int)
        self._last_obs = np.zeros((num_envs, self.obs_dim), dtype=np.float64)

    # ── 环境接口 ──
    def reset(self, indices: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        if indices is None:
            indices = np.arange(self.num_envs, dtype=np.int32)
        else:
            indices = np.asarray(indices, dtype=np.int32)
        if self.low_env.state is None:
            self.low_env.init_state()
        self.low_env.reset(indices)
        for i in indices:
            self.low[int(i)].reset()
        self._resample_des(indices)
        self._prev_a[indices] = 0.0
        self._steps[indices] = 0
        self._last_obs[indices] = self._collect_obs(indices)
        return self._last_obs.copy(), {}

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """actions: (N, action_dim) 归一化命令 → (obs2, reward, done, info)."""
        a = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        cmds = np.stack(
            [denorm_action(a[i], self.cfg, self.task_key, self._des[i]) for i in range(self.num_envs)]
        )
        sens = read_sensors_batch(self.low_env)
        a8 = np.stack(
            [
                self.low[i].act({k: v[i] for k, v in sens.items()}, cmds[i])
                for i in range(self.num_envs)
            ]
        )
        self.low_env.state.info["commands"][:] = cmds
        st = self.low_env.step(a8)
        done = (st.terminated | st.truncated)
        rew = compute_reward(sens, self._des, a, self._prev_a, done, self.cfg, self.task_key)
        self._prev_a = a.copy()
        self._steps += 1
        done_idx = np.nonzero(done)[0]
        if len(done_idx):
            # ★ auto-reset 终止 env (站姿复位 + MPC 复位 + 重采样期望)
            self._resample_des(done_idx)
            self._steps[done_idx] = 0
            self.low_env.reset(done_idx.astype(np.int32))
            for i in done_idx:
                self.low[int(i)].reset()
        self._last_obs = self._collect_obs(np.arange(self.num_envs))
        return self._last_obs.copy(), rew, done, {"time_outs": st.truncated.copy()}

    def close(self) -> None:
        try:
            self.low_env.close()
        except Exception:
            pass

    # ── 内部 ──
    def _resample_des(self, indices: np.ndarray) -> None:
        for i in indices:
            self._des[int(i)] = sample_desired(self.task_key, self.rng, self.cfg)

    def _collect_obs(self, indices: np.ndarray) -> np.ndarray:
        sens = read_sensors_batch(self.low_env)
        out = np.zeros((self.num_envs, self.obs_dim), dtype=np.float64)
        idx = np.asarray(indices, dtype=np.int64)
        s = {k: v[idx] for k, v in sens.items()}
        out[idx] = build_high_obs_batch(s, self._des[idx], self._prev_a[idx], self.cfg, self.task_key)
        return out
