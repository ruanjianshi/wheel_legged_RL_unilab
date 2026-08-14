"""MPC×SAC 融合分支 — 高层 obs / 动作反归一化 / 期望命令 / 奖励.

高层策略 (SAC) 只关心"发什么命令", 由低层 MPC 执行。obs 取自传感器 + 期望命令,
不依赖底层 RL 的 297/288 堆叠 obs。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.fusion_control.mpc_sac.config import MpcSacConfig


def action_dim(task_key: str) -> int:
    return 3 if task_key == "walk_flat" else 2


def obs_dim(task_key: str, cfg: MpcSacConfig) -> int:
    return cfg.obs_dim_flat if task_key == "walk_flat" else cfg.obs_dim_rough


def cmd_dim(task_key: str) -> int:
    return 5 if task_key == "walk_flat" else 4


def denorm_action(a: np.ndarray, cfg: MpcSacConfig, task_key: str) -> np.ndarray:
    """归一化动作 [-1,1] → 实际命令 (flat 5D / rough 4D)."""
    a = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
    if task_key == "walk_flat":
        return np.array(
            [
                float(a[0]) * cfg.vx_flat,
                0.0,
                float(a[1]) * cfg.vyaw_flat,
                cfg.tsk_flat,
                cfg.height_mid + float(a[2]) * cfg.height_half,
            ],
            dtype=np.float64,
        )
    return np.array(
        [float(a[0]) * cfg.vx_rough, 0.0, float(a[1]) * cfg.vyaw_rough, 0.0],
        dtype=np.float64,
    )


def build_high_obs(
    sensors: dict[str, np.ndarray],
    des: np.ndarray,
    prev_a: np.ndarray,
    cfg: MpcSacConfig,
    task_key: str,
) -> np.ndarray:
    """高层 obs (1D, 来自单 env 传感器). flat 11D / rough 9D."""
    if task_key == "walk_flat":
        return np.array(
            [
                float(sensors["theta"]),
                float(sensors["theta_dot"]),
                float(sensors["v"]),
                float(sensors["omega_z"]),
                float(sensors["base_z"]),
                float(des[0]),
                float(des[2]),
                float(des[4]),
                float(prev_a[0]),
                float(prev_a[1]),
                float(prev_a[2]),
            ],
            dtype=np.float64,
        )
    return np.array(
        [
            float(sensors["theta"]),
            float(sensors["theta_dot"]),
            float(sensors["v"]),
            float(sensors["omega_z"]),
            float(sensors["base_z"]),
            float(des[0]),
            float(des[2]),
            float(prev_a[0]),
            float(prev_a[1]),
        ],
        dtype=np.float64,
    )


def sample_desired(task_key: str, rng: np.random.Generator, cfg: MpcSacConfig) -> np.ndarray:
    """随机期望命令 (训练任务): flat 5D / rough 4D."""
    if task_key == "walk_flat":
        vx = float(rng.uniform(*cfg.vx_range_flat))
        vyaw = float(rng.uniform(*cfg.vyaw_range_flat))
        height = float(rng.uniform(*cfg.height_range_flat))
        return np.array([vx, 0.0, vyaw, 0.0, height], dtype=np.float64)
    vx = float(rng.uniform(*cfg.vx_range_rough))
    vyaw = float(rng.uniform(*cfg.vyaw_range_rough))
    return np.array([vx, 0.0, vyaw, 0.0], dtype=np.float64)


def build_high_obs_batch(
    sensors: dict[str, np.ndarray],
    des: np.ndarray,
    prev_a: np.ndarray,
    cfg: MpcSacConfig,
    task_key: str,
) -> np.ndarray:
    """批量高层 obs (N, obs_dim). sensors/des/prev_a 均为 (N, ...)."""
    n = np.asarray(sensors["theta"]).shape[0]
    if task_key == "walk_flat":
        return np.stack(
            [
                sensors["theta"],
                sensors["theta_dot"],
                sensors["v"],
                sensors["omega_z"],
                sensors["base_z"],
                des[:, 0],
                des[:, 2],
                des[:, 4],
                prev_a[:, 0],
                prev_a[:, 1],
                prev_a[:, 2],
            ],
            axis=-1,
        ).astype(np.float64).reshape(n, -1)
    return np.stack(
        [
            sensors["theta"],
            sensors["theta_dot"],
            sensors["v"],
            sensors["omega_z"],
            sensors["base_z"],
            des[:, 0],
            des[:, 2],
            prev_a[:, 0],
            prev_a[:, 1],
        ],
        axis=-1,
    ).astype(np.float64).reshape(n, -1)


def compute_reward(
    sensors: dict[str, np.ndarray],
    des: np.ndarray,
    a: np.ndarray,
    prev_a: np.ndarray,
    done: np.ndarray,
    cfg: MpcSacConfig,
    task_key: str,
) -> np.ndarray:
    """奖励 (向量化, 对 MPC 执行后的轨迹). des 为 (N, cmd_dim)."""
    des = np.asarray(des, dtype=np.float64)
    r = cfg.w_alive * (1.0 - done.astype(np.float64))
    v = np.asarray(sensors["v"], dtype=np.float64)
    omega_z = np.asarray(sensors["omega_z"], dtype=np.float64)
    theta = np.asarray(sensors["theta"], dtype=np.float64)
    r = r + cfg.w_vx * np.exp(-((v - des[:, 0]) ** 2) / (2 * cfg.sigma_vx**2))
    r = r + cfg.w_vyaw * np.exp(-((omega_z - des[:, 2]) ** 2) / (2 * cfg.sigma_vyaw**2))
    if task_key == "walk_flat":
        base_z = np.asarray(sensors["base_z"], dtype=np.float64)
        r = r + cfg.w_h * np.exp(-((base_z - des[:, 4]) ** 2) / (2 * cfg.sigma_h**2))
    r = r + cfg.w_theta * theta**2
    r = r + cfg.w_energy * (np.asarray(sensors["dof_vel"], dtype=np.float64)[:, 6] ** 2)
    r = r + cfg.w_cmd_rate * np.sum((np.asarray(a) - np.asarray(prev_a)) ** 2, axis=-1)
    return r
