#!/usr/bin/env python3
"""渲染双模式点足行走演示视频 — 站立 → 抬腿前进 → 抬腿侧移 → 抬腿转向 → 切回站立.

用法:
  uv run mjpython tools/xqrobotwl/render_toe_walk_mode.py \
      [--run <run_dir>] [--ckpt model_9999.pt] \
      [--out video/toe_walk/<date>_双模式演示.mp4]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _devlog.assess import engine, tasks  # noqa: E402

DT = 0.01
MODE_STAND, MODE_LIFT = 0.0, 1.0
SEQUENCE = [
    ("1_站立", MODE_STAND, (0.0, 0.0, 0.0), 3.0),
    ("2_抬腿前进", MODE_LIFT, (0.2, 0.0, 0.0), 4.0),
    ("3_抬腿侧移", MODE_LIFT, (0.0, 0.1, 0.0), 4.0),
    ("4_抬腿转向", MODE_LIFT, (0.0, 0.0, 0.3), 4.0),
    ("5_抬腿后退", MODE_LIFT, (-0.2, 0.0, 0.0), 4.0),
    ("6_切回站立", MODE_STAND, (0.0, 0.0, 0.0), 3.0),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=str, default=None, help="run 目录名 (默认最新)")
    ap.add_argument("--ckpt", type=str, default="model_9999.pt")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--cam_distance", type=float, default=2.6)
    args = ap.parse_args()

    task = tasks.get("toe_walk_mode")
    if args.run is not None:
        run_dir = engine.resolve_run_dir(args.run, task.log_root)
    else:
        run_dir = sorted(Path(ROOT / task.log_root).glob("*/"), key=lambda p: p.stat().st_mtime)[-1]
    ckpt_path = engine.find_checkpoint(run_dir, args.ckpt)
    env = engine.build_env(task, num_envs=1, ckpt_path=ckpt_path)
    policy = engine.load_policy(ckpt_path, env.obs_groups_spec["obs"], task.num_actions)
    try:
        env.init_state()
        backend = env._backend
        states = []
        for name, mode, cmd, dur in SEQUENCE:
            print(f"  渲染段: {name}  (mode={int(mode)}, cmd={cmd}, {dur}s)")
            for _ in range(int(dur / DT)):
                env.state.info["commands"][:, :3] = np.asarray(cmd, dtype=np.float64)
                env.state.info["commands"][:, 4] = mode
                obs_t = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                with torch.no_grad():
                    a = policy(obs_t).numpy().astype(np.float64)
                env.step(a)
                states.append(backend._physics_state.copy())
    finally:
        env.close()

    from unilab.visualization.render_many import render_states_get_frames_tracking

    frames = render_states_get_frames_tracking(
        states,
        str(ROOT / "src/unilab/assets/robots/xqrobotwl/scene_flat.xml"),
        width=720,
        height=480,
        tracking_env_idx=0,
        max_extra_envs=0,
        cam_distance=args.cam_distance,
        cam_elevation=-10,
        cam_azimuth=90,
        render_spacing=0.0,
    )

    out_path = (
        Path(args.out)
        if args.out
        else ROOT / "video" / "toe_walk" / f"{run_dir.name}_双模式点足演示.mp4"
    )
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
    for f in tmp.glob("f_*.ppm"):
        f.unlink()
    tmp.rmdir()
    print(f"✅ 视频已生成: {out_path}")


if __name__ == "__main__":
    main()
