#!/usr/bin/env python3
"""渲染单模式点足行走视频 (xqrobotwl_toe_walk_flat, 4D 命令) + 同步采集轮力时序.

用法:
  uv run python tools/xqrobotwl/render_toe_walk_single.py \
      [--run <run_dir>] [--ckpt model_9999.pt] [--out video/toe_walk/x.mp4]
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
SEQ = [("站立微动", (0.0, 0.0, 0.0, 0.0), 3.0), ("点足原地踏步", (0.0, 0.0, 0.0, 0.0), 6.0)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=str, default=None)
    ap.add_argument("--ckpt", type=str, default="model_9999.pt")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    task = tasks.get("toe_walk")
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
        forces = {"L": [], "R": []}
        for name, cmd, dur in SEQ:
            print(f"  段: {name} ({dur}s)  cmd={cmd}")
            for _ in range(int(dur / DT)):
                env.state.info["commands"][:, :4] = np.asarray(cmd, dtype=np.float64)
                obs_t = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                with torch.no_grad():
                    a = policy(obs_t).numpy().astype(np.float64)
                env.step(a)
                states.append(backend._physics_state.copy())
                fl = np.asarray(backend.get_sensor_data("left_wheel_force"), dtype=np.float64)[0]
                fr = np.asarray(backend.get_sensor_data("right_wheel_force"), dtype=np.float64)[0]
                forces["L"].append(float(np.linalg.norm(fl)))
                forces["R"].append(float(np.linalg.norm(fr)))
    finally:
        env.close()

    forces = {k: np.asarray(v) for k, v in forces.items()}
    print(f"  轮力 L: min {forces['L'].min():.0f} p50 {np.median(forces['L']):.0f} N | R: min {forces['R'].min():.0f} p50 {np.median(forces['R']):.0f} N")
    print(f"  轮力<10N 占比: L {(forces['L']<10).mean()*100:.0f}% R {(forces['R']<10).mean()*100:.0f}%")

    from unilab.visualization.render_many import render_states_get_frames_tracking

    frames = render_states_get_frames_tracking(
        states,
        str(ROOT / "src/unilab/assets/robots/xqrobotwl/scene_flat.xml"),
        width=720,
        height=480,
        tracking_env_idx=0,
        max_extra_envs=0,
        cam_distance=2.6,
        cam_elevation=-10,
        cam_azimuth=90,
        render_spacing=0.0,
    )

    out_path = Path(args.out) if args.out else ROOT / "video" / "toe_walk" / f"{run_dir.name}_点足演示.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f"_frames_{out_path.stem}"
    tmp.mkdir(exist_ok=True)
    for i, fr in enumerate(frames):
        (tmp / f"f_{i:05d}.ppm").write_bytes(
            b"P6\n%d %d\n255\n" % (fr.shape[1], fr.shape[0]) + fr.tobytes()
        )
    ff = subprocess.run(
        ["python", "-c", "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        [ff, "-y", "-framerate", "50", "-i", str(tmp / "f_%05d.ppm"), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-preset", "fast", str(out_path)],
        check=True,
        capture_output=True,
    )
    for f in tmp.glob("f_*.ppm"):
        f.unlink()
    tmp.rmdir()
    print(f"✅ 视频已生成: {out_path}")


if __name__ == "__main__":
    main()