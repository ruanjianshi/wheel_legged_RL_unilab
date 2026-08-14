#!/usr/bin/env python3
"""离屏渲染 xqrobotwl 机器人模型视频 (视觉 / 碰撞 / 惯量, 绕机一圈).

用法:
  uv run tools/mujoco/render_robot_models.py --mode visual     # 视觉网格 (mesh)
  uv run tools/mujoco/render_robot_models.py --mode collision  # 碰撞体 (box/cyl)
  uv run tools/mujoco/render_robot_models.py --mode inertia    # 惯量椭球 + CoM
  默认输出: video/robot_<mode>.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mujoco  # noqa: E402
from tools.mujoco.show_collision_model import (  # noqa: E402
    SCENE,
    colorize_collision,
    standing_qpos,
)


def _build_opt(mode: str) -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1  # 地面
    if mode == "visual":
        opt.geomgroup[2] = 1  # 视觉网格 (group 2)
    elif mode == "collision":
        opt.geomgroup[3] = 1  # 碰撞体 (group 3)
    elif mode == "inertia":
        opt.geomgroup[2] = 1  # 显示视觉网格作参照
        opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = 1  # 惯量椭球
        opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 1  # 质心点
    return opt


def main() -> int:
    ap = argparse.ArgumentParser(description="渲染机器人模型视频")
    ap.add_argument("--mode", choices=["visual", "collision", "inertia"], default="visual")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=120, help="绕机帧数 (360° → 每帧 3°)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    data.qpos[:] = standing_qpos()
    mujoco.mj_forward(model, data)
    colorize_collision(model)

    opt = _build_opt(args.mode)
    renderer = mujoco.Renderer(model, args.height, args.width)
    focus = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    frames = []
    for f in range(args.frames):
        az = 30.0 + 360.0 * f / args.frames
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 1.6
        cam.elevation = -15
        cam.azimuth = az
        if focus >= 0:
            cam.lookat[:] = data.xpos[focus]
            cam.lookat[2] += 0.1
        renderer.update_scene(data, cam, scene_option=opt)
        frames.append(renderer.render())
    renderer.close()

    out = Path(args.out) if args.out else ROOT / "video" / f"robot_{args.mode}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio

    with imageio.get_writer(str(out), fps=30, macro_block_size=None) as w:
        for fr in frames:
            w.append_data(fr)
    print(f"[{args.mode}] 视频已保存: {out} ({len(frames)} 帧)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
