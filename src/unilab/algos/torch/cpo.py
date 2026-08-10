"""CPO: Constrained Policy Optimization (penalty method) — 力引导跌倒恢复用.

Reference: FTSR (Hou et al. 2026, "Robust Fall Recovery for Armless Bipedal-
Wheeled Robots via Force-Guided Learning") — 将外部辅助力 F 和力矩 T 作为
可优化约束, 引导策略逐步降低对辅助的依赖 (d_i → 0)。

与 NP3O 的区别:
  - NP3O 从物理观测提取二进制 cost violation;
  - CPO 的约束代价由 **env 提供** (extras["constraint_costs"], 即辅助力/力矩
    幅值 [F_mag, T_mag]), 是连续值, 由 rsl_rl wrapper 从 state.info 转发。
  - 罚因子 β 代替 NP3O 的 K 退火 (论文固定 β ≈ 0.001)。

公式 (惩罚函数法, 论文 Eq.5/8):
  cost_returns = GAE(costs, cost_critic, γ, λ)
  cost_adv     = cost_returns - cost_values → 归一化
  cost_viol    = (1-γ)*(cost_returns - d)   → 归一化
  cost_surr    = max(cost_adv*ratio, cost_adv*clip(ratio)) → mean(0)
  viol_loss    = Σ_i β_i * ReLU(cost_surr_i + mean(cost_viol_i))
  loss = surrogate + value_coef*v_loss + cv_coef*cv_loss + penalty_coef*viol - entropy_coef*entropy
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from unilab.algos.torch.rsl_rl_ppo import FinalObservationAwarePPO


def _build_cost_critic(
    input_dim: int, hidden_dims: list[int], output_dim: int
) -> torch.nn.Sequential:
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(torch.nn.Linear(prev, h))
        layers.append(torch.nn.ELU())
        prev = h
    layers.append(torch.nn.Linear(prev, output_dim))
    return torch.nn.Sequential(*layers)


class CPO(FinalObservationAwarePPO):
    """惩罚函数法 CPO — env 提供约束代价 (辅助力/力矩幅值), 罚其依赖."""

    _cost_buf: torch.Tensor
    _cost_accum: torch.Tensor
    _cost_val_buf: torch.Tensor
    _cost_ret: torch.Tensor
    _cost_adv: torch.Tensor
    _cost_viol: torch.Tensor
    _cache_beta: torch.Tensor

    def __init__(
        self,
        *args: Any,
        num_constraints: int = 2,
        constraint_critic_hidden_dims: list[int] | None = None,
        constraint_value_loss_coef: float = 1.0,
        constraint_penalty_coef: float = 1.0,
        beta_init: float | list[float] = 0.001,
        beta_max: float = 1.0,
        beta_growth: float = 1.0001,
        d_values: list[float] | None = None,
        enable_compile: bool = False,
        **kwargs: Any,
    ) -> None:
        # Pop CPO-specific kwargs before passing to PPO parent
        cpo_keys = {
            "num_constraints",
            "constraint_critic_hidden_dims",
            "constraint_value_loss_coef",
            "constraint_penalty_coef",
            "beta_init",
            "beta_max",
            "beta_growth",
            "d_values",
            "cost_gamma",
            "cost_lam",
            "cost_max_grad_norm",
            "cost_learning_rate",
        }
        ppo_kwargs = {k: v for k, v in kwargs.items() if k not in cpo_keys}
        super().__init__(*args, enable_compile=enable_compile, **ppo_kwargs)

        if constraint_critic_hidden_dims is None:
            constraint_critic_hidden_dims = [512, 256, 128]

        self.num_constraints = int(num_constraints)
        self.constraint_value_loss_coef = float(constraint_value_loss_coef)
        self.constraint_penalty_coef = float(constraint_penalty_coef)
        if isinstance(beta_init, (int, float)):
            self.beta_inits = [float(beta_init)] * self.num_constraints
        else:
            self.beta_inits = [float(v) for v in beta_init]
        self.beta_max = float(beta_max)
        self.beta_growth = float(beta_growth)
        self._iter_counter = 0

        self.d_values = (
            torch.tensor(d_values, device=self.device, dtype=torch.float32)
            if d_values is not None
            else torch.zeros(self.num_constraints, device=self.device)
        )

        critic_obs_dim = self._infer_critic_obs_dim()

        self.constraint_critic = _build_cost_critic(
            critic_obs_dim, constraint_critic_hidden_dims, self.num_constraints
        ).to(self.device)

        for param in self.constraint_critic.parameters():
            self.optimizer.add_param_group({"params": param})

    # ── helpers ──────────────────────────────────────────────────────

    def _infer_critic_obs_dim(self) -> int:
        if hasattr(self.critic, "mlp"):
            for m in self.critic.mlp.modules():
                if isinstance(m, torch.nn.Linear):
                    return int(m.in_features)
        raise RuntimeError("Cannot infer critic obs dim from critic.mlp")

    def _critic_obs_tensor(self, obs: TensorDict) -> torch.Tensor:
        groups = getattr(self.critic, "obs_groups", None)
        if groups:
            tensors = [obs[g] for g in groups]
            return torch.cat(tensors, dim=-1) if len(tensors) > 1 else tensors[0]
        return self._model_obs_tensor(self.critic, obs)

    # ── beta annealing (温和增长, 约束难满足时放宽) ──────────────────

    @property
    def beta_current(self) -> torch.Tensor:
        if not hasattr(self, "_cache_beta"):
            self._cache_beta = torch.tensor(self.beta_inits, device=self.device)
        return cast(torch.Tensor, self._cache_beta)

    def _anneal_beta(self) -> None:
        v = torch.tensor(self.beta_inits, device=self.device) * (
            self.beta_growth ** max(self._iter_counter, 1)
        )
        self._cache_beta = torch.min(torch.ones_like(v) * self.beta_max, v)

    # ── override process_env_step ────────────────────────────────────

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor | TensorDict],
    ) -> None:
        E = dones.shape[0]
        S = self.storage.num_transitions_per_env

        if not hasattr(self, "_cost_buf") or self._cost_buf.shape[1] != E:
            self._cost_buf = torch.zeros(S, E, self.num_constraints, device=self.device)
            self._cost_accum = torch.zeros(E, self.num_constraints, device=self.device)
            self._cost_val_buf = torch.zeros(S, E, self.num_constraints, device=self.device)

        # 约束代价 = env 施加的辅助力/力矩幅值 (extras 由 rsl_rl wrapper 转发)
        cost_src = extras.get("constraint_costs")
        if cost_src is None:
            raw = torch.zeros(E, self.num_constraints, device=self.device)
        else:
            raw = cost_src.to(self.device).float().reshape(E, -1)
            if raw.shape[1] < self.num_constraints:
                raw = F.pad(raw, (0, self.num_constraints - raw.shape[1]))
            raw = raw[:, : self.num_constraints]

        critic_obs = self._critic_obs_tensor(obs)
        with torch.no_grad():
            cv = self.constraint_critic(critic_obs)

        # 累计代价, done 重置
        self._cost_accum = (
            self._cost_accum[:E] * (1.0 - dones.float().squeeze(-1).unsqueeze(-1)) + raw
        )
        self._cost_accum[dones.squeeze(-1).bool()] = 0.0

        super().process_env_step(obs, rewards, dones, extras)

        step = (self.storage.step - 1) % S
        self._cost_buf[step] = self._cost_accum.clone()
        self._cost_val_buf[step] = cv

    # ── override compute_returns ─────────────────────────────────────

    def compute_returns(self, obs: TensorDict) -> None:
        super().compute_returns(obs)

        critic_obs = self._critic_obs_tensor(obs)
        with torch.no_grad():
            last_cv = self.constraint_critic(critic_obs)

        S, E, NC = self._cost_val_buf.shape
        gamma, lam = self.gamma, self.lam

        self._cost_ret = torch.zeros_like(self._cost_val_buf)
        self._cost_adv = torch.zeros_like(self._cost_val_buf)
        self._cost_viol = torch.zeros_like(self._cost_val_buf)

        adv = torch.zeros(E, NC, device=self.device)
        for step in reversed(range(S)):
            nxt = last_cv if step == S - 1 else self._cost_val_buf[step + 1]
            ntm = 1.0 - self.storage.dones[step].squeeze(-1).float()
            delta = (
                self._cost_buf[step] + ntm.unsqueeze(-1) * gamma * nxt - self._cost_val_buf[step]
            )
            adv = delta + ntm.unsqueeze(-1) * gamma * lam * adv
            self._cost_ret[step] = adv + self._cost_val_buf[step]

        self._cost_adv = self._cost_ret - self._cost_val_buf
        flat = self._cost_adv.view(E * S, -1)
        mu = flat.mean(0)
        std = flat.std(0) + 1e-8
        self._cost_adv = (self._cost_adv - mu.view(1, 1, -1)) / std.view(1, 1, -1)
        self._cost_viol = (
            (1.0 - gamma) * (self._cost_ret - self.d_values.view(1, 1, -1)) + mu.view(1, 1, -1)
        ) / std.view(1, 1, -1)

        # Clone to break inference tensor tracking for use in backward pass
        self._cost_ret = self._cost_ret.clone()
        self._cost_adv = self._cost_adv.clone()
        self._cost_viol = self._cost_viol.clone()
        self._cost_val_buf = self._cost_val_buf.clone()

    # ── override update ──────────────────────────────────────────────

    def update(self) -> dict[str, float]:
        self._iter_counter += 1
        self._anneal_beta()

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_cv_loss = 0.0
        mean_viol = 0.0
        mean_entropy = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        S, E, NC = self._cost_ret.shape
        cost_ret_f = torch.empty_like(self._cost_ret).copy_(self._cost_ret).flatten(0, 1)
        cost_adv_f = torch.empty_like(self._cost_adv).copy_(self._cost_adv).flatten(0, 1)
        cost_val_f = torch.empty_like(self._cost_val_buf).copy_(self._cost_val_buf).flatten(0, 1)
        cost_viol_f = torch.empty_like(self._cost_viol).copy_(self._cost_viol).flatten(0, 1)

        for batch in generator:
            act = cast(torch.Tensor, batch.actions)
            val = cast(torch.Tensor, batch.values)
            adv = cast(torch.Tensor, batch.advantages)
            ret = cast(torch.Tensor, batch.returns)
            old_logp = cast(torch.Tensor, batch.old_actions_log_prob)
            old_mu, old_sigma = cast(Any, batch.old_distribution_params)

            actor_obs = self._model_obs_tensor(self.actor, batch.observations).clone()
            critic_obs = self._critic_obs_tensor(batch.observations).clone()

            mu, sigma = self._actor_mean_std(actor_obs)
            logp = self._gaussian_log_prob(act, mu, sigma)
            values = self._critic_value(critic_obs)
            entropy = self._gaussian_entropy(sigma).mean()

            r = torch.exp(logp - old_logp.squeeze(-1))
            surr = -adv.squeeze(-1) * r
            surr_clip = -adv.squeeze(-1) * torch.clamp(
                r, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surr, surr_clip).mean()

            if self.use_clipped_value_loss:
                v_clip = val.squeeze(-1) + (values - val.squeeze(-1)).clamp(
                    -self.clip_param, self.clip_param
                )
                v_loss = torch.max(
                    (values - ret.squeeze(-1)).pow(2),
                    (v_clip - ret.squeeze(-1)).pow(2),
                ).mean()
            else:
                v_loss = (ret.squeeze(-1) - values).pow(2).mean()

            kl = torch.sum(
                torch.log(sigma / old_sigma + 1e-5)
                + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2))
                - 0.5,
                dim=-1,
            ).mean()

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_val = float(kl.detach())
                    if kl_val > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_val < self.desired_kl / 2.0 and kl_val > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate

            bs = act.shape[0]
            cv_now = self.constraint_critic(critic_obs)
            cv_targ = cost_val_f[:bs]
            # 目标乘 (1-γ) 缩到 ~1 尺度 (论文 cost_viol 同理), 防 cost critic 发散
            cv_ret = cost_ret_f[:bs] * (1.0 - self.gamma)

            if self.use_clipped_value_loss:
                cv_clip = cv_targ + (cv_now - cv_targ).clamp(-self.clip_param, self.clip_param)
                cv_loss = torch.max((cv_now - cv_ret).pow(2), (cv_clip - cv_ret).pow(2)).mean()
            else:
                cv_loss = (cv_ret - cv_now).pow(2).mean()

            # 约束罚项 (惩罚函数法): Σ β_i * ReLU(约束surrogate + 违反量)
            ca = cost_adv_f[:bs]
            rs = r.unsqueeze(-1)
            cs = ca * rs
            cs_clip = ca * torch.clamp(rs, 1.0 - self.clip_param, 1.0 + self.clip_param)
            cs_per_dim = torch.max(cs, cs_clip).mean(0)
            viol_batch_mean = cost_viol_f[:bs].mean()
            viol = torch.sum(self.beta_current * torch.relu(cs_per_dim + viol_batch_mean))

            loss = (
                surrogate_loss
                + self.value_loss_coef * v_loss
                + self.constraint_value_loss_coef * cv_loss
                + self.constraint_penalty_coef * viol
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.constraint_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += v_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_cv_loss += cv_loss.item()
            mean_viol += viol.item()
            mean_entropy += entropy.item()

        n = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        return {
            "value": mean_value_loss / n,
            "surrogate": mean_surrogate_loss / n,
            "constraint_value": mean_cv_loss / n,
            "constraint_viol": mean_viol / n,
            "entropy": mean_entropy / n,
        }
