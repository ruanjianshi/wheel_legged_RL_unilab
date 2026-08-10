"""确定性评估 XqRobotWLSingleLegMove checkpoint — 单轮平衡 + vx 跟踪

指标:
  - episode_length: 平均时长 (倒得越晚越好)
  - wheel_off_rate: 自由轮(左)离地率 (需 ≈1)
  - vx_tracking: vx 命令 vs 实际 RMSE (需小)
  - survived: 跑满 max_steps 的 episode 比例

用法:
  uv run mjpython tools/xqrobotwl/eval_single_leg_move.py \
      --ckpt logs/rsl_rl_ppo/XqRobotWLSingleLegMove/<run>/model_1000.pt \
      --episodes 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from unilab.base import registry
from unilab.base.registry import ensure_registries

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "conf/ppo/task/xqrobotwl_single_leg_move/mujoco.yaml"


class ActorMLP(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        return self.mlp(x)


def main() -> None:
    ap = argparse.ArgumentParser(description="评估 single_leg_move")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=600, help="单 episode 上限步数")
    ap.add_argument("--num_envs", type=int, default=10, help="并行 env (每次 reset)")
    args = ap.parse_args()

    ensure_registries()
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    override = {"reward_config": cfg["reward"]}
    override.update(cfg.get("env", {}))

    env = registry.make(
        "XqRobotWLSingleLegMove",
        sim_backend="mujoco",
        num_envs=args.num_envs,
        env_cfg_override=override,
    )
    obs, _ = env.reset(np.arange(args.num_envs))

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pol = ActorMLP(obs["obs"].shape[1])
    pol.load_state_dict({k: v for k, v in ckpt["actor_state_dict"].items() if k.startswith("mlp.")})
    pol.eval()

    cmd_hist = np.zeros((args.num_envs, args.max_steps))
    vx_hist = np.zeros((args.num_envs, args.max_steps))
    wheel_off = np.zeros((args.num_envs, args.max_steps))
    ep_len = np.zeros(args.num_envs)
    done_flags = np.zeros(args.num_envs, dtype=bool)

    for i in range(args.max_steps):
        obs_t = torch.tensor(obs["obs"], dtype=torch.float32)
        with torch.no_grad():
            a = pol(obs_t).numpy().astype(np.float64)
        st = env.step(a)
        cmd_hist[:, i] = obs["obs"][:, -5]  # commands[0]=vx (obs 帧末5列)
        linvel = env.get_local_linvel()
        vx_hist[:, i] = linvel[:, 0]
        # 单腿判据: 自由腿配重展开 (L_hip_roll<-0.3)。⚠️ 不用 wheel_force (不可靠)
        dof_pos = env.get_dof_pos()
        wheel_off[:, i] = (dof_pos[:, 0] < -0.3).astype(np.float64)
        ep_len += 1.0
        done = st.terminated | st.truncated
        done_flags |= done
        # 已 done 的 env 静置 (不再统计)
        if done.any() and i > 0:
            ep_len[done] = np.minimum(ep_len[done], i + 1)
            cmd_hist[done, i:] = 0
            vx_hist[done, i:] = 0
        obs = st.obs

    print("=" * 60)
    print(f"评估: {args.ckpt}")
    print("=" * 60)
    print(
        f"  episode 长度: {ep_len.mean():.1f} 步 / {args.max_steps}  ({(ep_len >= args.max_steps).mean() * 100:.0f}% 跑满)"
    )
    off_all = wheel_off[:, : args.max_steps].mean()
    print(f"  自由轮离地率: {off_all * 100:.1f}%  (需 ≈100%)")
    # vx 跟踪 (只统计活跃步)
    mask = vx_hist != 0
    cmd_active = cmd_hist[mask]
    vx_active = vx_hist[mask]
    if len(cmd_active) > 0:
        rmse = np.sqrt(np.mean((cmd_active - vx_active) ** 2))
        print(f"  vx 跟踪 RMSE: {rmse:.3f} m/s  (cmd range ±0.3)")
        for vc in [-0.3, 0.0, 0.3]:
            m = np.abs(cmd_active - vc) < 0.01
            if m.sum() > 0:
                print(f"    vx_cmd={vc:+.1f}: 实际 {vx_active[m].mean():+.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
