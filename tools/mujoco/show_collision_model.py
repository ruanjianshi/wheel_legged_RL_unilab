#!/usr/bin/env python3
"""显示 xqrobotwl 机器人的碰撞体模型 (MuJoCo viewer) — 验证碰撞几何是否正确.

视觉网格 (group 2) 默认隐藏, 碰撞体 (group 3) 高亮显示, 各连杆异色便于区分。
站姿 = 自然站姿 (standing_angles + base_z≈0.512), 与经典控制/RL 一致。

用法:
  uv run tools/mujoco/show_collision_model.py          # Linux
  uv run mjpython tools/mujoco/show_collision_model.py # macOS

键盘:
  C   仅碰撞体 (默认)   V   仅视觉网格   B   两者   R   重置站姿   Space 暂停
相机: 鼠标拖拽 / 滚轮缩放
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402  # 显式导入 viewer 子模块 (launch_passive)

# ★ 镜像对称站姿 (左腿=右腿镜像; 原 RL 测得站姿左右不对称会显示"左腿歪")
#   验证: 对称站姿下左右轮世界 |y| 完全相等 (±0.171)
STANDING_ANGLES = [0.0, 0.1083, -0.0188, 0.0, -0.1083, 0.0188]
BASE_Z = 0.512
SCENE = ROOT / "src/unilab/assets/robots/xqrobotwl/scene_flat.xml"

# 各连杆碰撞体颜色 (L/R 镜像同色, 便于按连杆识别)
_PALETTE = {
    "base": (0.90, 0.30, 0.30, 1.0),  # 红: 机身
    "left_link_1": (0.30, 0.70, 0.40, 1.0),  # 绿: 左髋
    "left_link_2": (0.35, 0.50, 0.95, 1.0),  # 蓝: 左大腿
    "left_link_3": (0.95, 0.70, 0.25, 1.0),  # 橙: 左小腿
    "left_link_wheel": (0.30, 0.30, 0.30, 1.0),  # 灰: 左轮
    "right_link_1": (0.30, 0.70, 0.40, 1.0),
    "right_link_2": (0.35, 0.50, 0.95, 1.0),
    "right_link_3": (0.95, 0.70, 0.25, 1.0),
    "right_link_wheel": (0.30, 0.30, 0.30, 1.0),
}

_GEOM_TYPES = {
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_MESH: "mesh",
    mujoco.mjtGeom.mjGEOM_PLANE: "plane",
}


def standing_qpos() -> np.ndarray:
    """直立站姿 qpos (15D): [x,y,z, quat(wxyz), 6腿, 2轮]."""
    qpos = np.zeros(15, dtype=np.float64)
    qpos[2] = BASE_Z
    qpos[3] = 1.0  # 直立四元数
    qpos[7:15] = [*STANDING_ANGLES[:3], 0.0, *STANDING_ANGLES[3:], 0.0]
    return qpos


def colorize_collision(model: mujoco.MjModel) -> None:
    """按连杆给碰撞体 (group 3) 着色; 非碰撞体置为不渲染 (group 外)."""
    for i in range(model.ngeom):
        if model.geom_group[i] != 3:
            continue
        body = model.body(model.geom_bodyid[i]).name or ""
        key = (
            body if body in _PALETTE else next((k for k in _PALETTE if body.startswith(k)), "base")
        )
        model.geom_rgba[i] = _PALETTE[key]


def print_collision_table(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """打印碰撞体清单: 名称/类型/尺寸/本地位姿/世界位置/接触属性."""
    print("=" * 118)
    print("XqRobotWL 碰撞体清单 (group=3)")
    print(
        f"{'geom':<28}{'type':<10}{'size(half)':<22}{'local pos':<24}{'world pos':<22}{'contype/ca':<12}"
    )
    print("-" * 118)
    n_col = 0
    for i in range(model.ngeom):
        if model.geom_group[i] != 3:
            continue
        name = model.geom(i).name or f"geom{i}"
        gtype = _GEOM_TYPES.get(int(model.geom_type[i]), "?")
        size = np.asarray(model.geom_size[i])
        size_s = " ".join(f"{s:.3f}" for s in size[:3])
        pos = np.asarray(model.geom_pos[i])
        pos_s = " ".join(f"{p:+.3f}" for p in pos)
        wpos = data.geom_xpos[i]
        wpos_s = " ".join(f"{p:+.3f}" for p in wpos)
        cc = f"{int(model.geom_contype[i])}/{int(model.geom_conaffinity[i])}"
        print(f"{name:<28}{gtype:<10}{size_s:<22}{pos_s:<24}{wpos_s:<22}{cc:<12}")
        n_col += 1
    print("-" * 118)
    print(
        f"共 {n_col} 个碰撞体; 接触规则: 机器人 contype=0(自身不发起接触), "
        f"conaffinity=1(接受地面 contact) → 轮地接触正常生成"
    )
    print("=" * 118)


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    model.opt.gravity[:] = [0.0, 0.0, -9.81]

    data.qpos[:] = standing_qpos()
    mujoco.mj_forward(model, data)
    colorize_collision(model)
    print_collision_table(model, data)

    focus = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    paused = False
    mode = "collision"

    def _set_mode(m: str) -> None:
        nonlocal mode
        mode = m
        # geomgroup: 0=默认(地面) 2=视觉 3=碰撞
        viewer.opt.geomgroup[:] = 0
        viewer.opt.geomgroup[0] = 1  # 地面
        if m in ("collision", "both"):
            viewer.opt.geomgroup[3] = 1
        if m in ("visual", "both"):
            viewer.opt.geomgroup[2] = 1
        print(f"[collision] 显示: {m}")

    def _on_key(keycode: int) -> int | None:
        nonlocal paused
        ch = chr(keycode) if 32 <= keycode < 128 else ""
        if ch in ("C", "c"):
            _set_mode("collision")
        elif ch in ("V", "v"):
            _set_mode("visual")
        elif ch in ("B", "b"):
            _set_mode("both")
        elif ch in ("R", "r"):
            data.qpos[:] = standing_qpos()
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            print("[collision] 重置站姿")
        elif keycode == ord(" "):
            paused = not paused
            print(f"[collision] {'paused' if paused else 'resumed'}")
        else:
            return None
        return 1

    print("[collision] C=仅碰撞体 V=仅视觉 B=两者 R=重置站姿 Space=暂停")
    with mujoco.viewer.launch_passive(model, data, key_callback=_on_key) as viewer:
        viewer.cam.distance = 1.6
        viewer.cam.elevation = -18
        viewer.cam.azimuth = 150
        if focus >= 0:
            viewer.cam.lookat[:] = data.xpos[focus]
            viewer.cam.lookat[2] += 0.1
        _set_mode("collision")
        while viewer.is_running():
            viewer.sync()
            # 暂停时不推进; 纯显示脚本无动力学推进
    return 0


if __name__ == "__main__":
    sys.exit(main())
