"""MPC×SAC 融合分支 — 经验回放缓冲 (numpy 存储, torch 采样)."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str = "cpu") -> None:
        self.capacity = int(capacity)
        self.device = device
        self._obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._a = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self._r = np.zeros(self.capacity, dtype=np.float32)
        self._obs2 = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self._done = np.zeros(self.capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def push(
        self,
        obs: np.ndarray,
        a: np.ndarray,
        r: np.ndarray,
        obs2: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """向量化推入 (批量)."""
        n = len(obs)
        idx = (np.arange(self._ptr, self._ptr + n) % self.capacity).astype(np.int64)
        self._obs[idx] = obs
        self._a[idx] = a
        self._r[idx] = r
        self._obs2[idx] = obs2
        self._done[idx] = done
        self._ptr = (self._ptr + n) % self.capacity
        self._size = min(self._size + n, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = np.random.randint(0, self._size, size=int(batch_size))
        to_t = lambda x: torch.as_tensor(x[idx], dtype=torch.float32, device=self.device)  # noqa: E731
        return {
            "obs": to_t(self._obs),
            "a": to_t(self._a),
            "r": to_t(self._r),
            "obs2": to_t(self._obs2),
            "done": to_t(self._done),
        }

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size
