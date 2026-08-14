"""MPC×SAC 融合分支 — SAC trainer (标准 SAC + 可选 obs 归一化 + 温度自学习)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from scripts.fusion_control.mpc_sac.sac.networks import GaussianActor, TwinQ
from scripts.fusion_control.mpc_sac.sac.replay import ReplayBuffer


@dataclass
class SacHyper:
    obs_dim: int
    action_dim: int
    actor_hidden_dims: tuple[int, ...] = (256, 128)
    critic_hidden_dims: tuple[int, ...] = (256, 128)
    activation: str = "relu"
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    alpha_init: float = 0.2
    target_entropy_ratio: float = 1.0
    gamma: float = 0.99
    tau: float = 0.005
    init_noise_std: float = 0.6
    batch_size: int = 256
    replay_buffer_capacity: int = 200000
    updates_per_step: int = 2
    obs_normalization: bool = True
    device: str = "cpu"


class ObsNormalizer:
    """RunningMeanStd: 训练期在线更新, 推理期冻结."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.eps = eps
        self._mean = np.zeros(dim, dtype=np.float64)
        self._var = np.ones(dim, dtype=np.float64)
        self._count = 0.0

    def update(self, obs: np.ndarray) -> None:
        batch_mean = obs.mean(axis=0)
        batch_var = obs.var(axis=0)
        n = float(obs.shape[0])
        total = self._count + n
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * (n / total)
        self._var = (self._var * self._count + batch_var * n) / total + (
            delta**2
        ) * (self._count * n / total**2)
        self._count = total

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return ((np.asarray(obs, dtype=np.float64) - self._mean) / np.sqrt(self._var + self.eps)).astype(
            np.float32
        )

    def state(self) -> dict:
        return {"mean": self._mean, "var": self._var, "count": self._count}

    def load(self, d: dict) -> None:
        self._mean = np.asarray(d["mean"], dtype=np.float64)
        self._var = np.asarray(d["var"], dtype=np.float64)
        self._count = float(d["count"])


class SACTrainer:
    """紧凑 SAC: GaussianActor + TwinQ + 温度自学习."""

    def __init__(self, h: SacHyper) -> None:
        self.h = h
        self.device = h.device
        self.actor = GaussianActor(
            h.obs_dim, h.action_dim, h.actor_hidden_dims, h.activation, h.init_noise_std
        ).to(self.device)
        self.q1 = TwinQ(h.obs_dim, h.action_dim, h.critic_hidden_dims, h.activation).to(self.device)
        self.q2 = TwinQ(h.obs_dim, h.action_dim, h.critic_hidden_dims, h.activation).to(self.device)
        self.q1_target = TwinQ(h.obs_dim, h.action_dim, h.critic_hidden_dims, h.activation).to(
            self.device
        )
        self.q2_target = TwinQ(h.obs_dim, h.action_dim, h.critic_hidden_dims, h.activation).to(
            self.device
        )
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        log_alpha = np.log(max(h.alpha_init, 1e-6))
        self.log_alpha = torch.tensor([log_alpha], dtype=torch.float32, requires_grad=True, device=self.device)
        self.target_entropy = -h.target_entropy_ratio * h.action_dim

        self.opt_actor = Adam(self.actor.parameters(), lr=h.actor_lr)
        self.opt_critic = Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=h.critic_lr)
        self.opt_alpha = Adam([self.log_alpha], lr=h.alpha_lr)

        self.normalizer = ObsNormalizer(h.obs_dim) if h.obs_normalization else None
        self.replay = ReplayBuffer(h.replay_buffer_capacity, h.obs_dim, h.action_dim, "cpu")
        self.step_count = 0

    # ── 推理 ──
    def select_action(self, obs_np: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.select_action_batch(np.asarray(obs_np, dtype=np.float32)[None, :], deterministic)[0]

    def select_action_batch(
        self, obs_np: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        obs = self._norm_obs(np.asarray(obs_np, dtype=np.float32))
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            a = self.actor(obs_t, deterministic=deterministic)
        return a.detach().cpu().numpy()

    def _norm_obs(self, obs_np: np.ndarray) -> np.ndarray:
        if self.normalizer is None:
            return obs_np
        return self.normalizer.normalize(obs_np).astype(np.float32)

    # ── 学习 ──
    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs, a = batch["obs"].to(self.device), batch["a"].to(self.device)
        r, obs2, done = (
            batch["r"].to(self.device).unsqueeze(-1),
            batch["obs2"].to(self.device),
            batch["done"].to(self.device).unsqueeze(-1),
        )
        alpha = self.log_alpha.exp()

        # critic target: min(Q1', Q2') − α log π(a'|s')
        with torch.no_grad():
            a2, logp2 = self.actor.sample_log_prob(obs2)
            q1t = self.q1_target.min(obs2, a2)
            q2t = self.q2_target.min(obs2, a2)
            q_tgt = torch.min(q1t, q2t) - alpha * logp2
            y = r + (1.0 - done) * self.h.gamma * q_tgt

        q1v, q2v = self.q1.both(obs, a)
        loss_q = F.mse_loss(q1v, y) + F.mse_loss(q2v, y)
        self.opt_critic.zero_grad()
        loss_q.backward()
        self.opt_critic.step()

        # actor: maximize Q − α log π
        a_new, logp_new = self.actor.sample_log_prob(obs)
        q_pi = self.q1.min(obs, a_new)
        loss_a = (alpha.detach() * logp_new - q_pi).mean()
        self.opt_actor.zero_grad()
        loss_a.backward()
        self.opt_actor.step()

        # temperature
        loss_alpha = -(self.log_alpha * (logp_new.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        loss_alpha.backward()
        self.opt_alpha.step()

        # soft target update
        for tp, p in zip(
            self.q1_target.parameters(), self.q1.parameters(), strict=True
        ):
            tp.data.copy_(self.h.tau * p.data + (1.0 - self.h.tau) * tp.data)
        for tp, p in zip(
            self.q2_target.parameters(), self.q2.parameters(), strict=True
        ):
            tp.data.copy_(self.h.tau * p.data + (1.0 - self.h.tau) * tp.data)

        self.step_count += 1
        return {
            "loss_q": float(loss_q.item()),
            "loss_a": float(loss_a.item()),
            "alpha": float(alpha.item()),
            "q1": float(q1v.mean().item()),
        }

    def train_step(self, obs_np: np.ndarray, a_np: np.ndarray, r_np: np.ndarray, obs2_np: np.ndarray, done_np: np.ndarray) -> dict[str, float]:
        """在线推入回放 + 更新若干步."""
        if self.normalizer is not None:
            self.normalizer.update(obs_np)
        obs_n = self._norm_obs(obs_np)
        obs2_n = self._norm_obs(obs2_np)
        self.replay.push(obs_n, a_np, r_np, obs2_n, done_np)
        if self.replay.size < self.h.batch_size:
            return {}
        info: dict[str, float] = {}
        for _ in range(self.h.updates_per_step):
            b = self.replay.sample(self.h.batch_size)
            info = self.update(b)
        return info

    # ── 存取 ──
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "log_alpha": self.log_alpha.detach(),
                "normalizer": self.normalizer.state() if self.normalizer else None,
                "step_count": self.step_count,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        d = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(d["actor"])
        self.q1.load_state_dict(d["q1"])
        self.q2.load_state_dict(d["q2"])
        self.q1_target.load_state_dict(d["q1_target"])
        self.q2_target.load_state_dict(d["q2_target"])
        self.log_alpha = d["log_alpha"].to(self.device).requires_grad_()
        if self.normalizer and d.get("normalizer"):
            self.normalizer.load(d["normalizer"])
        self.step_count = int(d.get("step_count", 0))
