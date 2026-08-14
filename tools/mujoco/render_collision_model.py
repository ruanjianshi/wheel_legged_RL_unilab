#!/usr/bin/env python3
"""离屏渲染 xqrobotwl 碰撞体模型视频 (仅碰撞体视角, 绕机一圈).

用法:
  uv run tools/mujoco/render_collision_model.py                 # → video/collision_model.mp4
  uv run tools/mujoco/render_collision_model.py --out video/classic/collision_align.mp4
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
import mujoco.viewer  # noqa: E402  # 确保 viewer 子模块可用 (同 show 脚本)
from tools.mujoco.show_collision_model import (  # noqa: E402
    _PALETTE,
    BASE_Z,
    SCENE,
    STANDING_ANGLES,
    colorize_collision,
    standing_qpos,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="渲染碰撞体模型视频")
    ap.add_argument("--out", default=str(ROOT / "video" / "collision_model.mp4"))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=120, help="绕机帧数 (360° → 每帧 3°)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    data.qpos[:] = standing_qpos()
    mujoco.mj_forward(model, data)
    colorize_collision(model)

    # 离屏渲染器: 只显示碰撞体 (group 3) + 地面 (group 0)
    renderer = mujoco.Renderer(model, args.height, args.width)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1  # 地面
    opt.geomgroup[3] = 1  # 碰撞体

    focus = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    frames = []
    for f in range(args.frames):
        az = 30.0 + 360.0 * f / args.frames  # 绕机一圈
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

    # 写视频
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio

        with imageio.get_writer(str(out), fps=30, macro_block_size=None) as w:
            for fr in frames:
                w.append_data(fr)
        print(f"视频已保存: {out}  ({len(frames)} 帧, 360° 绕机)")
    except Exception as e:  # pragma: no cover
        # 回退: 存 PNG 序列
        png_dir = out.with_suffix("")
        png_dir.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames):
            imageio.imwrite(png_dir / f"{i:04d}.png", fr)
        print(f"imageio 写视频失败 ({e}), 已存 PNG 序列: {png_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
