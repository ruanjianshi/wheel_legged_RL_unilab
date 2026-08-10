"""渲染 XqRobotWLSingleLegUnicycle checkpoint 平衡视频 (横躺视角).

用法:
  uv run tools/xqrobotwl/render_trained_unicycle.py \
      --ckpt logs/rsl_rl_ppo/XqRobotWLSingleLegUnicycle/<run>/model_XXXX.pt \
      --out video/single_leg/2026-08-06_unicycle_balance.mp4 --steps 600
"""

from __future__ import annotations

import argparse
import subprocess
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
    ap = argparse.ArgumentParser(description="渲染 unicycle 平衡")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="video/single_leg/2026-08-06_unicycle_balance.mp4")
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    ensure_registries()
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    override = {"reward_config": cfg["reward"]}
    override.update(cfg.get("env", {}))

    env = registry.make(
        "XqRobotWLSingleLegUnicycle",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override=override,
    )
    obs, _ = env.reset(np.arange(1))

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pol = ActorMLP(obs["obs"].shape[1])
    pol.load_state_dict({k: v for k, v in ckpt["actor_state_dict"].items() if k.startswith("mlp.")})
    pol.eval()

    backend = env._backend  # type: ignore[attr-defined]
    states = []
    done_at = args.steps
    for i in range(args.steps):
        obs_t = torch.tensor(obs["obs"], dtype=torch.float32)
        with torch.no_grad():
            a = pol(obs_t).numpy().astype(np.float64)
        st = env.step(a)
        obs = st.obs
        states.append(backend._physics_state.copy())  # type: ignore[attr-defined]
        if (st.terminated | st.truncated)[0] and done_at == args.steps:
            done_at = i

    from unilab.visualization.render_many import render_states_get_frames_tracking

    frames = render_states_get_frames_tracking(
        states,
        str(ROOT / "src/unilab/assets/robots/xqrobotwl/scene_flat.xml"),
        width=640,
        height=480,
        tracking_env_idx=0,
        max_extra_envs=0,
        cam_distance=2.6,
        cam_elevation=-12,
        cam_azimuth=90,
        render_spacing=0.0,
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f"_frames_{out_path.stem}"
    tmp.mkdir(exist_ok=True)
    for i, fr in enumerate(frames):
        (tmp / f"f_{i:05d}.ppm").write_bytes(
            b"P6\n%d %d\n255\n" % (fr.shape[1], fr.shape[0]) + fr.tobytes()
        )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "50",
            "-i",
            str(tmp / "f_%05d.ppm"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    print(f"视频: {out_path}  (平衡 {done_at * 0.01:.2f}s / 渲染 {len(frames)} 帧)")


if __name__ == "__main__":
    main()
