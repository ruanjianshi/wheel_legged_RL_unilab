#!/usr/bin/env python3
"""Render a single static frame of xqrobotwl in its default standing pose.

Used to produce the robot image in the jumping paper (fig:robot).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from unilab.base import registry  # noqa: E402
from unilab.base.registry import ensure_registries  # noqa: E402


def main() -> int:
    ensure_registries()
    with open(ROOT / "conf/ppo/task/xqrobotwl_walk_flat/mujoco.yaml") as f:
        cfg = yaml.safe_load(f)
    override = {"reward_config": cfg["reward"]}
    override.update(cfg.get("env", {}))
    env = registry.make(
        "XqRobotWLWalkFlat",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override=override,
    )
    env.reset(np.array([0]))
    backend = env._backend
    state = backend._physics_state.copy()
    env.close()

    from unilab.visualization.render_many import render_states_get_frames_tracking

    frames = render_states_get_frames_tracking(
        [state],
        str(ROOT / "src/unilab/assets/robots/xqrobotwl/scene_flat.xml"),
        width=800,
        height=600,
        tracking_env_idx=0,
        max_extra_envs=0,
        cam_distance=1.6,
        cam_elevation=-14,
        cam_azimuth=90,
        render_spacing=0.0,
    )
    fr = frames[0]  # (H, W, 3) RGB uint8
    out = ROOT / "latex/Wheeled-SRL-Jumping/figures/xqrobotwl_render.png"
    from PIL import Image

    Image.fromarray(fr).save(out)
    print(f"机器人渲染图 -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
