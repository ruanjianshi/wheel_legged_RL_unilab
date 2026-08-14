"""MPC×SAC 融合分支 — SAC 高层策略网络 (紧凑实现, 自包含).

GaussianActor (tanh 压缩) + Twin Q-critic. 高层策略小 (obs 11/9, action 2/3),
标准 SAC (Haarnoja et al. 2018) 即可, MPC 已扛平衡。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def build_mlp(
    dims: list[int], activation: str = "relu", out_activation: str | None = None
) -> nn.Sequential:
    layers: list[nn.Module] = []
    act_cls: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
    }
    act = act_cls.get(activation, nn.ReLU)
    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(act())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if out_activation == "tanh":
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """高斯随机策略: 输出均值为 tanh 压缩, 标准差独立可学."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str = "relu",
        init_std: float = 0.6,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.trunk = build_mlp([obs_dim, *hidden_dims, action_dim], activation)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_std)).log(), requires_grad=True)

    def _dist(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tanh(self.trunk(obs))
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mean, log_std = self._dist(obs)
        if deterministic:
            return mean
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        u = normal.sample()
        a = torch.tanh(u)
        return a

    def sample_log_prob(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (action, log_prob), tanh 修正雅可比."""
        mean, log_std = self._dist(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()
        a = torch.tanh(u)
        # log π = log N(u) − Σ log(1 − tanh²(u)) + eps
        log_prob = normal.log_prob(u).sum(-1, keepdim=True)
        log_prob = log_prob - torch.log(1.0 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        return a, log_prob


class TwinQ(nn.Module):
    """双 Q 网络 (Clipped Double-Q)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.q1 = build_mlp([obs_dim + action_dim, *hidden_dims, 1], activation)
        self.q2 = build_mlp([obs_dim + action_dim, *hidden_dims, 1], activation)

    def both(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.both(obs, action)
        return torch.min(q1, q2)
