#!/usr/bin/env python3
"""Render a repeated-jump video for §7.5 (§2.4 视角内, camera tracking).

Usage:
  uv run tools/xqrobotwl/render_jump_repeat.py \
      --task XqRobotWLJumpSRLFlat \
      --checkpoint logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat/<run>/model_15000.pt \
      --out video/jump/<date>_jump_srl_repeat.mp4 \
      --jumps 4 --on 100 --off 120 --settle 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify_jump import load_actor, trained_env_overrides  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jumps", type=int, default=4)
    ap.add_argument("--settle", type=int, default=50)
    ap.add_argument("--on", type=int, default=100)
    ap.add_argument("--off", type=int, default=120)
    ap.add_argument("--hidden", default="512,512,256,128")
    ap.add_argument("--cam_distance", type=float, default=2.6)
    ap.add_argument("--cam_elevation", type=float, default=-15.0)
    ap.add_argument("--cam_azimuth", type=float, default=90.0)
    args = ap.parse_args()
    hidden = [int(x) for x in args.hidden.split(",")]

    from unilab.base import registry
    from unilab.base.registry import ensure_registries

    ensure_registries()
    ov = trained_env_overrides(args.checkpoint)
    if ov is None:
        ctrl = (
            {"action_scale": 0.6, "wheel_action_scale": 10.0, "clip_actions": 1.0}
            if "VMC" in args.task
            else {"action_scale": 0.6, "wheel_action_scale": 10.0, "clip_actions": 100.0}
        )
        ov = {"control_config": ctrl}
    env = registry.make(args.task, num_envs=1, sim_backend="mujoco", env_cfg_override=ov)
    try:
        obs_dim = env.obs_groups_spec["obs"]
        actor = load_actor(args.checkpoint, obs_dim, 8, hidden)

        total = args.settle + args.jumps * (args.on + args.off)
        counter = {"n": 0}

        def _step(obs):
            i = counter["n"]
            if i < args.settle:
                trig = 0.0
            else:
                j = (i - args.settle) % (args.on + args.off)
                trig = 1.0 if j < args.on else 0.0
            env.state.info["commands"][:, 4] = trig
            counter["n"] += 1
            a = actor(torch.tensor(obs, dtype=torch.float32)).detach().numpy()
            st = env.step(a)
            if st.terminated[0]:
                # keep stepping to render the fall; stop arming trigger
                env.state.info["commands"][:, 4] = 0.0
            return st.obs["obs"]

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        path = env.run_playback_mode(
            play_render_mode="auto",
            play_steps=total,
            output_video=str(out),
            render_spacing=0.0,
            initialize=lambda: (env.init_state(), env.reset(np.arange(1, dtype=np.int32))[0]["obs"])[1],
            step=_step,
            camera_kwargs={
                "cam_distance": args.cam_distance,
                "cam_elevation": args.cam_elevation,
                "cam_azimuth": args.cam_azimuth,
                "cam_lookat": None,
                "cam_tracking": True,
                "cam_tracking_env_idx": 0,
                "cam_tracking_extra_envs": 0,
            },
        )
        print(f"video -> {path}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
