#!/usr/bin/env python3
"""经典 LQR 平衡控制器交互回放 — MuJoCo 键盘控制 (前进/后退/转向/腿长).

用法:
  uv run mjpython scripts/classic_control/play_classic.py
  uv run mjpython scripts/classic_control/play_classic.py --terrain rough

键盘:
  ↑ / ↓       : 前进 / 后退 (vx)
  ← / →       : 左转 / 右转 (vyaw)
  Q / E       : 降低 / 升高机身 (腿长)
  Enter       : 停止
  Space       : 暂停
  相机: 鼠标拖拽/滚轮
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402  # 显式导入 viewer 子模块 (launch_passive)

from scripts.classic_control import rollout as rollout_mod  # noqa: E402
from scripts.classic_control.controller import BalanceController  # noqa: E402
from scripts.classic_control.run import get_dynamics  # noqa: E402
from unilab.base.backend.mujoco.playback import resolve_render_play_model_files  # noqa: E402
from unilab.visualization.interactive_playback import KeyboardCommander  # noqa: E402

_KEY_UP, _KEY_DOWN, _KEY_LEFT, _KEY_RIGHT = 265, 264, 263, 262
_KEY_ENTER, _KEY_KP_ENTER = 257, 335
_KEY_BACKSPACE = 259


def main() -> int:
    ap = argparse.ArgumentParser(description="经典 LQR/MPC 平衡交互控制 (键盘)")
    ap.add_argument("--terrain", choices=["flat", "rough"], default="flat")
    ap.add_argument("--controller", choices=["lqr", "mpc"], default="lqr")
    ap.add_argument("--q_z", type=float, default=15.0)
    ap.add_argument("--speed", type=float, default=1.0, help="仿真速度倍率")
    args = ap.parse_args()

    task_key = "walk_rough" if args.terrain == "rough" else "walk_flat"
    env = rollout_mod.build_env(task_key=task_key, num_envs=1)
    A_d, B_d = get_dynamics()
    phase = 3 if task_key == "flat" else 4  # flat 支持腿长 (P3), rough 地形 (P4)
    ctl = BalanceController(
        args.controller,
        phase,
        A_d,
        B_d,
        params={"q_z": args.q_z},
        action_scale=float(env._cfg.control_config.action_scale),
        wheel_action_scale=float(env._cfg.control_config.wheel_action_scale),
        dt=float(env._cfg.ctrl_dt),
    )
    env.init_state()

    # 查看器模型/数据 (★ 用解析视觉模型, get_playback_model 缺视觉网格→白屏)
    import tempfile

    def _load_viewer_model_file(path: str) -> mujoco.MjModel:
        return (
            mujoco.MjModel.from_binary_path(path)
            if Path(path).suffix.lower() == ".mjb"
            else mujoco.MjModel.from_xml_path(path)
        )

    mj_model = None
    try:
        with tempfile.TemporaryDirectory(prefix="classic-viewer-") as _tmp:
            model_files = resolve_render_play_model_files(env, num_envs=1, tmp_dir=_tmp)
            mf = model_files[0] if isinstance(model_files, list) else model_files
            mj_model = _load_viewer_model_file(str(mf))
    except Exception:
        mj_model = None
    if mj_model is None:
        mj_model = env.get_playback_model(0)
    # ★ 只显示视觉 geom (group 2) → group 0; 碰撞模型 (group 3) 不显示
    for _i in range(mj_model.ngeom):
        if int(mj_model.geom_group[_i]) == 2:
            mj_model.geom_group[_i] = 0
        elif int(mj_model.geom_group[_i]) == 3:
            mj_model.geom_group[_i] = 3  # 保持不显示
    viz_data = mujoco.MjData(mj_model)
    cmd_dim = env.state.info["commands"].shape[1]

    vel_limit = np.asarray(env._cfg.commands.vel_limit, dtype=np.float64)
    commander = KeyboardCommander.from_vel_limit(vel_limit, step_lin=0.1, step_ang=0.2)
    if cmd_dim >= 5:
        commander.height_target = float(env._cfg.commands.vel_limit[0][4]) if False else 0.518
    commander.height_step = 0.02
    commander.height_min, commander.height_max = 0.48, 0.55  # ★ 可达高度范围 (0.60 会塌)

    paused = False
    ctrl_dt = float(env._cfg.ctrl_dt)
    focus_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    step_counter = 0

    def _on_key(keycode: int) -> int | None:
        nonlocal paused
        if keycode == ord(" "):
            paused = not paused
            print(f"[classic] {'paused' if paused else 'resumed'} (space)")
            return 1
        elif keycode in (_KEY_ENTER, _KEY_KP_ENTER):
            commander.zero()
            print("[classic] stop")
            return 1
        elif keycode == _KEY_UP:
            commander.nudge(commander.AXIS_VX, +1.0)
        elif keycode == _KEY_DOWN:
            commander.nudge(commander.AXIS_VX, -1.0)
        elif keycode == _KEY_LEFT:
            commander.nudge(commander.AXIS_VYAW, +1.0)
        elif keycode == _KEY_RIGHT:
            commander.nudge(commander.AXIS_VYAW, -1.0)
        elif keycode == ord("Q"):
            commander.nudge_height(-1.0)
        elif keycode == ord("E"):
            commander.nudge_height(+1.0)
        elif keycode == _KEY_BACKSPACE:
            # ★ Backspace 重置 (和 RL 一致): 重置机器人 + 控制器 + 命令
            ctl.reset()
            commander.zero()
            env.init_state()
            print("[classic] reset (backspace)")
            return 1
        else:
            return None
        print(f"[classic] {commander.describe()}")
        return 1

    print("[classic] 经典 LQR 平衡交互 (键盘):")
    print("  ↑/↓ 前进/后退 · ←/→ 转向 · Q/E 腿长 · Enter 停止 · Space 暂停 · Backspace 重置")
    print(f"  terrain={args.terrain} phase=P{phase}")

    with mujoco.viewer.launch_passive(mj_model, viz_data, key_callback=_on_key) as viewer:
        # ★ 初始化相机对准机器人 (lookat 起点 = base 位置, 否则画面空白)
        phys0 = np.asarray(env._backend.get_physics_state(), dtype=np.float64)[0]
        mujoco.mj_setState(mj_model, viz_data, phys0[1:], mujoco.mjtState.mjSTATE_PHYSICS)
        mujoco.mj_forward(mj_model, viz_data)
        viewer.cam.distance = 2.6
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135
        if focus_body_id >= 0:
            viewer.cam.lookat[:] = viz_data.xpos[focus_body_id]
            viewer.cam.lookat[2] += 0.3
        while viewer.is_running():
            # 命令: [vx, vy, vyaw, tsk, height]
            cmd = np.zeros(cmd_dim, dtype=np.float64)
            cmd[:3] = commander.command
            if cmd_dim >= 5:
                cmd[4] = commander.height_target
            sensors = rollout_mod.read_sensors(env)
            a = ctl.act(sensors, cmd)
            if not paused:
                env.state.info["commands"][:] = np.tile(cmd, (1, 1))
                st = env.step(np.asarray(a, dtype=np.float64)[None, :])
                if st.terminated[0]:
                    print("[classic] ⚠️ 机器人跌倒 — 按 Backspace 重置")
            # 同步查看器 (★ 不覆盖相机 lookat, 让用户手动拖视角)
            phys = np.asarray(env._backend.get_physics_state(), dtype=np.float64)[0]
            mujoco.mj_setState(mj_model, viz_data, phys[1:], mujoco.mjtState.mjSTATE_PHYSICS)
            mujoco.mj_forward(mj_model, viz_data)
            viewer.sync()  # ★ 强制刷新 viewer 画面 (否则机器人"钉死")
            if step_counter % 100 == 0:
                print(
                    f"  t={step_counter * ctrl_dt:.1f}s vx_cmd={cmd[0]:+.2f} "
                    f"base_x={phys[1]:+.2f} base_z={phys[3]:.3f} v={sensors['v']:+.2f}"
                )
            step_counter += 1
            time.sleep(ctrl_dt / args.speed * 0.8)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
