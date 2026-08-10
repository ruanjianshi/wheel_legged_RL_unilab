"""诊断: 用训练的真实运行时 (OnPolicyRunner) 跑 rollout, 对比手写评估.

用法:
  uv run mjpython tools/xqrobotwl/diag_runtime_rollout.py \
      --run 2026-08-07_10-54-48_mujoco --ckpt model_6000.pt [--num_envs 32] [--steps 800]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from unilab.algos.torch.rsl_rl_runtime import resolve_rsl_rl_ppo_runtime  # noqa: E402
from unilab.base import registry  # noqa: E402
from unilab.base.registry import ensure_registries  # noqa: E402
from unilab.training import create_env, get_log_root, parse_checkpoint_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--num_envs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()

    ensure_registries()
    cfg = OmegaConf.load(ROOT / "conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml")
    # 组织成 play 所需结构 (algo dict)
    algo = cfg.algo
    rl_cfg = dict(algo)
    rl_cfg.pop("algo", None)
    # 加载 checkpoint
    run_dir = ROOT / "logs/rsl_rl_cpo/XqRobotWLFallRecoveryFlat" / args.run
    ckpt_path = run_dir / (args.ckpt or "model_6000.pt")
    from unilab.training.rsl_rl import RslRlVecEnvWrapper

    resolve_rsl_rl_ppo_runtime(rl_cfg, default_wrapper_cls=RslRlVecEnvWrapper)

    env_cfg_override = {"reward_config": dict(cfg.reward)}
    env_cfg_override["reward_config"]["force_assist_enabled"] = False
    env_cfg_override.update(dict(cfg.env))
    env = registry.make(
        "XqRobotWLFallRecoveryFlat",
        sim_backend="mujoco",
        num_envs=args.num_envs,
        env_cfg_override=env_cfg_override,
    )
    wrapped = RslRlVecEnvWrapper(env, device="cpu")
    train_cfg = dict(rl_cfg)
    train_cfg["runner"] = {"logger": "none"}
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device="cpu")
    runner.load(str(ckpt_path), map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

    obs, _ = env.reset(np.arange(args.num_envs))
    maxz = np.zeros(args.num_envs)
    rec = np.zeros(args.num_envs, dtype=bool)
    for i in range(args.steps):
        with torch.no_grad():
            a = policy(obs).detach().cpu().numpy().astype(np.float64)
        st = wrapped.step(a)
        bz = env._backend.get_base_pos()[:, 2]
        maxz = np.maximum(maxz, bz)
        rec |= bz > 0.45
        obs = st.obs
    print(
        f"runtime rollout: max_z={maxz.mean():.2f}  达0.45: {rec.mean() * 100:.0f}%  达0.55: {(maxz > 0.55).mean() * 100:.0f}%"
    )


if __name__ == "__main__":
    main()
