"""MPC×SAC 融合控制训练 — 高层 SAC 策略 (低层 MPC 冻结执行).

用法:
  uv run mjpython scripts/fusion_control/mpc_sac/train_mpc_sac.py --task walk_flat --max_iterations 3000
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from scripts.fusion_control.mpc_sac.config import build_config
from scripts.fusion_control.mpc_sac.meta_env import MpcSacMetaEnv
from scripts.fusion_control.mpc_sac.sac.trainer import SacHyper, SACTrainer


def main() -> int:
    ap = argparse.ArgumentParser(description="MPC×SAC 高层策略训练 (低层 MPC 冻结)")
    ap.add_argument("--task", choices=["walk_flat", "walk_rough"], default="walk_flat")
    ap.add_argument("--num_envs", type=int, default=None)
    ap.add_argument("--max_iterations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tag", type=str, default="", help="run 目录后缀")
    ap.add_argument("--resume", type=str, default=None, help="恢复训练 ckpt 路径")
    args = ap.parse_args()

    task_key = args.task
    cfg, merged = build_config(task_key)
    num_envs = args.num_envs or cfg.num_envs
    max_iter = args.max_iterations or cfg.max_iterations
    seed = args.seed if args.seed is not None else cfg.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = MpcSacMetaEnv(task_key, num_envs, params=merged, cfg=cfg, seed=seed)
    h = SacHyper(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        actor_hidden_dims=cfg.actor_hidden_dims,
        critic_hidden_dims=cfg.critic_hidden_dims,
        activation=cfg.activation,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        alpha_lr=cfg.alpha_lr,
        alpha_init=cfg.alpha_init,
        target_entropy_ratio=cfg.target_entropy_ratio,
        gamma=cfg.gamma,
        tau=cfg.tau,
        init_noise_std=cfg.init_noise_std,
        batch_size=cfg.batch_size,
        replay_buffer_capacity=cfg.replay_buffer_capacity,
        updates_per_step=cfg.updates_per_step,
        obs_normalization=cfg.obs_normalization,
        device=device,
    )
    trainer = SACTrainer(h)
    if args.resume:
        trainer.load(args.resume)
        print(f"[resume] {args.resume} (step={trainer.step_count})")

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = ROOT / cfg.log_dir / task_key / f"{ts}{('_' + args.tag) if args.tag else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_final = run_dir / "model_final.pt"
    print(f"[MPC×SAC] {task_key}  num_envs={num_envs}  max_iter={max_iter}  device={device}")
    print(f"[run] {run_dir}")

    obs, _ = env.reset()
    ep_rew = np.zeros(num_envs, dtype=np.float64)
    ep_len = np.zeros(num_envs, dtype=np.float64)
    cum_rew_window: list[float] = []
    total_steps = 0

    for it in range(1, max_iter + 1):
        a = trainer.select_action_batch(obs, deterministic=False)
        obs2, r, done, _info = env.step(a)
        info = trainer.train_step(obs, a, r, obs2, done)
        ep_rew += r
        ep_len += 1.0
        done_ep = np.nonzero(done)[0]
        if len(done_ep):
            cum_rew_window.extend(ep_rew[done_ep].tolist())
            ep_rew[done_ep] = 0.0
            ep_len[done_ep] = 0.0
        obs = obs2
        total_steps += num_envs

        if it % 100 == 0 or it == 1:
            tail = cum_rew_window[-50:]
            mean_ret = float(np.mean(tail)) if tail else float("nan")
            mean_len = float(np.mean(ep_len)) if len(ep_len) else 0.0
            alpha = float(info.get("alpha", float("nan"))) if info else float("nan")
            q1 = float(info.get("q1", float("nan"))) if info else float("nan")
            print(
                f"[{it:5d}] steps={total_steps:8d} mean_ret(50)={mean_ret:6.3f} "
                f"ep_len={mean_len:6.1f} alpha={alpha:.3f} q1={q1:.2f} "
                f"rbuf={len(trainer.replay)}"
            )

        if it % cfg.eval_interval == 0:
            _quick_eval(env, trainer, task_key, cfg, merged, it, run_dir)
        if it % cfg.save_interval == 0:
            trainer.save(run_dir / f"model_{it:05d}.pt")

    trainer.save(model_final)
    print(f"\n[done] 最终模型 → {model_final}")
    env.close()
    return 0


def _quick_eval(env, trainer, task_key, cfg, merged, it, run_dir) -> None:
    """训练中快速评估: 单条恒定命令 vx=0.4, 统计存活/时长 (只读复用 eval 逻辑)."""
    try:
        from scripts.fusion_control.mpc_sac.controller import MpcSacController
        from scripts.fusion_control.mpc_sac.env import build_env
        from scripts.fusion_control.mpc_sac.policy_loader import HighLevelPolicy
        from scripts.fusion_control.mpc_sac.runner import make_schedule, run_episode

        env2 = build_env(task_key, num_envs=1, cmd=None, lock_hip_roll=True)
        ascale = float(env2._cfg.control_config.action_scale)
        wscale = float(env2._cfg.control_config.wheel_action_scale)
        dt = float(env2._cfg.ctrl_dt)
        hl = HighLevelPolicy(trainer, env.obs_dim, env.action_dim, trainer.device)
        ctrl = MpcSacController(
            task_key, params=merged, action_scale=ascale, wheel_action_scale=wscale,
            dt=dt, policy=hl,
        )
        schedule = make_schedule(task_key, "vx=0.4,vyaw=0.0")
        ctrl.reset()
        rec, stats = run_episode(env2, ctrl, schedule, 15.0, dt)
        env2.close()
        surv = 1.0 if stats["ep_len_s"] > 10.0 else stats["ep_len_s"] / 10.0
        print(f"    [eval@{it}] surv={surv*100:.0f}%  len={stats['ep_len_s']:.1f}s  steps={stats['n_steps']}")
    except Exception as exc:  # pragma: no cover
        print(f"    [eval@{it}] failed: {exc}")


if __name__ == "__main__":
    sys.exit(main())
