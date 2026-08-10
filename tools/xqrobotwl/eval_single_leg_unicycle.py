"""确定性评估 XqRobotWLSingleLegUnicycle checkpoint — 改进结构单轮平衡

指标:
  - episode_length: 平均时长 (平衡越久越好, max_steps=600 = 6s)
  - survived: 跑满比例
  - wheel_on_ground: 左轮贴地率 (需 ≈100%)
  - pitch_rmse / roll_rmse: 机身横躺保持误差 (base长轴·[1,0,0] 与 up·[0,-1,0])

用法:
  uv run mjpython tools/xqrobotwl/eval_single_leg_unicycle.py \
      --ckpt logs/rsl_rl_ppo/XqRobotWLSingleLegUnicycle/<run>/model_XXXX.pt \
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
CONFIG = ROOT / "conf/ppo/task/xqrobotwl_single_leg_unicycle/mujoco.yaml"
OBS_DIM = 324  # 36 × 9 帧


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
    ap = argparse.ArgumentParser(description="评估 single_leg_unicycle")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--num_envs", type=int, default=10)
    args = ap.parse_args()

    ensure_registries()
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    override = {"reward_config": cfg["reward"]}
    override.update(cfg.get("env", {}))

    env = registry.make(
        "XqRobotWLSingleLegUnicycle",
        sim_backend="mujoco",
        num_envs=args.num_envs,
        env_cfg_override=override,
    )
    obs, _ = env.reset(np.arange(args.num_envs))

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pol = ActorMLP(obs["obs"].shape[1])
    pol.load_state_dict({k: v for k, v in ckpt["actor_state_dict"].items() if k.startswith("mlp.")})
    pol.eval()

    ep_len = np.zeros(args.num_envs)
    active = np.ones(args.num_envs, dtype=bool)
    wheel_down = np.zeros((args.num_envs, args.max_steps))
    pitch_a = np.zeros((args.num_envs, args.max_steps))
    roll_a = np.zeros((args.num_envs, args.max_steps))

    for i in range(args.max_steps):
        obs_t = torch.tensor(obs["obs"], dtype=torch.float32)
        with torch.no_grad():
            a = pol(obs_t).numpy().astype(np.float64)
        st = env.step(a)
        lw_z = np.asarray(env._backend.get_sensor_data("left_wheel_world_pos"))[:, 2]
        wheel_down[:, i] = (lw_z < 0.11 + 0.02).astype(np.float64)
        up = env._backend.get_sensor_data("upvector")
        bx = env._backend.get_sensor_data("basexvector")
        roll_a[:, i] = up @ np.array([0.0, -1.0, 0.0])
        pitch_a[:, i] = bx @ np.array([1.0, 0.0, 0.0])
        ep_len[active] += 1.0
        done = st.terminated | st.truncated
        if done.any():
            new_done = done & active
            ep_len[new_done] = np.minimum(ep_len[new_done], i + 1)
            active &= ~done  # ⚠️ done env 自动 reset 后不再累加, 冻结其长度
        obs = st.obs

    print("=" * 60)
    print(f"评估: {args.ckpt}")
    print("=" * 60)
    print(
        f"  episode 长度: {ep_len.mean():.1f} 步 / {args.max_steps}  "
        f"({(ep_len >= args.max_steps).mean() * 100:.0f}% 跑满 = 平衡 {args.max_steps / 100:.1f}s)"
    )
    active = wheel_down[:, : max(int(ep_len.max()), 1)]
    print(f"  左轮贴地率: {active.mean() * 100:.1f}%  (需 ≈100%)")
    active = pitch_a[:, : max(int(ep_len.max()), 1)]
    print(f"  pitch 对齐 (base长轴·[1,0,0]): {active.mean():.3f}  (1=横躺)")
    active = roll_a[:, : max(int(ep_len.max()), 1)]
    print(f"  roll 对齐 (up·[0,-1,0]): {active.mean():.3f}  (1=横躺)")


if __name__ == "__main__":
    main()
