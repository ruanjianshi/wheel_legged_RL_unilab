"""MPC×SAC 融合分支 — 高层 SAC 策略加载 (推理).

训练产出的 ckpt (sac/trainer 保存: actor/q/normalizer) → 推理包装。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from scripts.fusion_control.mpc_sac.config import MpcSacConfig
from scripts.fusion_control.mpc_sac.obs import action_dim, obs_dim
from scripts.fusion_control.mpc_sac.sac.trainer import SacHyper, SACTrainer


class HighLevelPolicy:
    """高层策略推理接口: obs_np (1D) → action_np (1D)."""

    def __init__(self, trainer: SACTrainer, obs_dim_: int, action_dim_: int, device: str) -> None:
        self.trainer = trainer
        self.obs_dim = obs_dim_
        self.action_dim = action_dim_
        self.device = device

    def select_action(self, obs_np: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs = np.asarray(obs_np, dtype=np.float32).reshape(-1)
        assert obs.shape[0] == self.obs_dim, f"obs dim {obs.shape[0]} != {self.obs_dim}"
        a = self.trainer.select_action(obs, deterministic=deterministic)
        return np.asarray(a, dtype=np.float64).reshape(-1)


def load_policy(
    cfg: MpcSacConfig,
    task_key: str,
    checkpoint: str | Path,
    device: str = "cpu",
) -> HighLevelPolicy:
    """从 ckpt 加载高层 SAC 策略 (含 obs_normalizer)."""
    od = obs_dim(task_key, cfg)
    ad = action_dim(task_key)
    h = SacHyper(
        obs_dim=od,
        action_dim=ad,
        actor_hidden_dims=cfg.actor_hidden_dims,
        critic_hidden_dims=cfg.critic_hidden_dims,
        activation=cfg.activation,
        gamma=cfg.gamma,
        tau=cfg.tau,
        init_noise_std=cfg.init_noise_std,
        batch_size=cfg.batch_size,
        replay_buffer_capacity=cfg.replay_buffer_capacity,
        obs_normalization=cfg.obs_normalization,
        device=device,
    )
    trainer = SACTrainer(h)
    trainer.load(checkpoint)
    trainer.actor.eval()
    return HighLevelPolicy(trainer, od, ad, device)


def get_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
