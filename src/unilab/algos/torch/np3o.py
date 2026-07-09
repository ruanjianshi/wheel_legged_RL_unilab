"""NP3O: Neural Proximal Policy Optimization with constraints.

Reference: Tita RL — cost critic + viol_loss + K-multiplier annealing.
Formulas:
  cost_returns  = GAE(costs, cost_values, γ, λ)
  cost_adv      = (cost_returns - cost_values) → normalize per-cost-dim
  cost_viol     = (1-γ)*(cost_returns - d) → normalize
  cost_surr     = max(cost_adv*ratio, cost_adv*clip(ratio)) → mean(0)
  viol_loss     = Σ k * ReLU(cost_surr + mean(cost_viol))
  k             = min(1, k_init * k_growth^iter)
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


class NP3O(FinalObservationAwarePPO):
    def __init__(
        self,
        *args: Any,
        num_costs: int = 6,
        cost_critic_hidden_dims: list[int] | None = None,
        cost_value_loss_coef: float = 1.0,
        cost_viol_loss_coef: float = 1.0,
        k_init: float | list[float] = 0.3,
        k_growth: float = 1.0004,
        k_max: float = 1.0,
        d_values: list[float] | None = None,
        enable_compile: bool = False,
        **kwargs: Any,
    ) -> None:
        # Pop NP3O-specific kwargs before passing to PPO parent
        np3o_keys = {
            "num_costs",
            "cost_critic_hidden_dims",
            "cost_value_loss_coef",
            "cost_viol_loss_coef",
            "k_init",
            "k_growth",
            "k_max",
            "d_values",
            "cost_gamma",
            "cost_lam",
            "cost_max_grad_norm",
            "cost_learning_rate",
        }
        ppo_kwargs = {k: v for k, v in kwargs.items() if k not in np3o_keys}
        super().__init__(*args, enable_compile=enable_compile, **ppo_kwargs)

        if cost_critic_hidden_dims is None:
            cost_critic_hidden_dims = [512, 256, 128]

        self.num_costs = int(num_costs)
        self.cost_value_loss_coef = float(cost_value_loss_coef)
        self.cost_viol_loss_coef = float(cost_viol_loss_coef)
        if isinstance(k_init, (int, float)):
            self.k_inits = [float(k_init)] * self.num_costs
        else:
            self.k_inits = [float(v) for v in k_init]
        self.k_growth = float(k_growth)
        self.k_max = float(k_max)
        self._iter_counter = 0

        self.d_values = (
            torch.tensor(d_values, device=self.device, dtype=torch.float32)
            if d_values is not None
            else torch.zeros(self.num_costs, device=self.device)
        )

        critic_obs_dim = self._infer_critic_obs_dim()

        self.cost_critic = _build_cost_critic(
            critic_obs_dim, cost_critic_hidden_dims, self.num_costs
        ).to(self.device)

        for param in self.cost_critic.parameters():
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

    def _extract_costs(self, obs: TensorDict) -> torch.Tensor:
        B = obs.batch_size[0] if hasattr(obs, "batch_size") else obs.shape[0]
        c = torch.zeros(B, self.num_costs, device=self.device)
        critic = obs.get("critic")
        actor = obs.get("actor")
        if critic is not None:
            c[:, 0] = torch.sqrt(critic[:, 0] ** 2 + critic[:, 1] ** 2)
            if critic.shape[-1] >= 17:
                c[:, 1] = torch.abs(critic[:, 11:17]).mean(dim=-1)
                c[:, 2] = torch.abs(critic[:, 11:17]).mean(dim=-1) * 0.5
        if actor is not None:
            if actor.shape[-1] >= 6:
                c[:, 3] = torch.abs(actor[:, 3:6]).mean(dim=-1) * 0.1
            if actor.shape[-1] >= 8:
                c[:, 4] = torch.abs(actor[:, 6:8]).mean(dim=-1) * 0.1
            c[:, 5] = (actor.abs().mean(dim=-1) > 0.5).float() * 0.5
        return c

    # ── K annealing ──────────────────────────────────────────────────

    @property
    def k_current(self) -> torch.Tensor:
        if not hasattr(self, "_cache_k"):
            self._cache_k = torch.tensor(self.k_inits, device=self.device)
        return cast(torch.Tensor, self._cache_k)

    def _anneal_k(self) -> None:
        k_init_t = torch.tensor(self.k_inits, device=self.device)
        v = k_init_t * (self.k_growth ** max(self._iter_counter, 1))
        self._cache_k = torch.min(torch.ones_like(v) * self.k_max, v)

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
            self._cost_buf = torch.zeros(S, E, self.num_costs, device=self.device)
            self._cost_accum = torch.zeros(E, self.num_costs, device=self.device)
            self._cost_val_buf = torch.zeros(S, E, self.num_costs, device=self.device)

        critic_obs = self._critic_obs_tensor(obs)
        with torch.no_grad():
            raw = self._extract_costs(obs)
            cv = self.cost_critic(critic_obs)

        # Accumulate costs per env, reset on done
        self._cost_accum = self._cost_accum[:E] * (1.0 - dones.float().squeeze(-1).unsqueeze(-1)) + raw
        self._cost_accum[dones.squeeze(-1).bool()] = 0.0

        super().process_env_step(obs, rewards, dones, extras)

        step = (self.storage.step - 1) % S
        self._cost_buf[step] = self._cost_accum.clone()
        self._cost_val_buf[step] = cv

    # ── override compute_returns ─────────────────────────────────────

    def compute_returns(self, last_obs: TensorDict) -> None:
        super().compute_returns(last_obs)

        critic_obs = self._critic_obs_tensor(last_obs)
        with torch.no_grad():
            last_cv = self.cost_critic(critic_obs)

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
        self._anneal_k()

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
            surr_clip = -adv.squeeze(-1) * torch.clamp(r, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surr, surr_clip).mean()

            if self.use_clipped_value_loss:
                v_clip = val.squeeze(-1) + (values - val.squeeze(-1)).clamp(-self.clip_param, self.clip_param)
                v_loss = torch.max(
                    (values - ret.squeeze(-1)).pow(2),
                    (v_clip - ret.squeeze(-1)).pow(2),
                ).mean()
            else:
                v_loss = (ret.squeeze(-1) - values).pow(2).mean()

            kl = torch.sum(
                torch.log(sigma / old_sigma + 1e-5)
                + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (2.0 * sigma.pow(2))
                - 0.5, dim=-1,
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
            cv_now = self.cost_critic(critic_obs)
            cv_targ = cost_val_f[:bs]
            cv_ret = cost_ret_f[:bs]

            if self.use_clipped_value_loss:
                cv_clip = cv_targ + (cv_now - cv_targ).clamp(-self.clip_param, self.clip_param)
                cv_loss = torch.max(
                    (cv_now - cv_ret).pow(2), (cv_clip - cv_ret).pow(2)
                ).mean()
            else:
                cv_loss = (cv_ret - cv_now).pow(2).mean()

            ca = cost_adv_f[:bs]
            rs = r.unsqueeze(-1)
            cs = ca * rs
            cs_clip = ca * torch.clamp(rs, 1.0 - self.clip_param, 1.0 + self.clip_param)
            cs_per_dim = torch.max(cs, cs_clip).mean(0)
            viol_batch_mean = cost_viol_f[:bs].mean()
            viol = torch.sum(self.k_current * torch.relu(cs_per_dim + viol_batch_mean))

            loss = (
                surrogate_loss
                + self.value_loss_coef * v_loss
                + self.cost_value_loss_coef * cv_loss
                + self.cost_viol_loss_coef * viol
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
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
            "cost_value": mean_cv_loss / n,
            "viol": mean_viol / n,
            "entropy": mean_entropy / n,
        }
