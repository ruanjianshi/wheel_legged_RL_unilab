"""经典平衡控制器交互回放 (共享) — MuJoCo 键盘控制 (前进/后退/转向/腿长).

共享只读 (开发规范 §3.2); 控制器类由各轨 CLI 传入 → 互不干涉。

键盘:
  ↑ / ↓       : 前进 / 后退 (vx)
  ← / →       : 左转 / 右转 (vyaw)
  Q / E       : 降低 / 升高机身 (腿长)
  Enter       : 停止
  Space       : 暂停
  相机: 鼠标拖拽/滚轮
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from scripts.classic_control.common import rollout as rollout_mod  # noqa: E402
from unilab.base.backend.mujoco.playback import resolve_render_play_model_files  # noqa: E402
from unilab.visualization.interactive_playback import KeyboardCommander  # noqa: E402

_KEY_UP, _KEY_DOWN, _KEY_LEFT, _KEY_RIGHT = 265, 264, 263, 262
_KEY_ENTER, _KEY_KP_ENTER = 257, 335
_KEY_BACKSPACE = 259


def run_play(
    controller_cls,
    label: str = "classic",
    terrain: str = "flat",
    speed: float = 1.0,
) -> int:
    task_key = "walk_rough" if terrain == "rough" else "walk_flat"
    env = rollout_mod.build_env(task_key=task_key, num_envs=1)
    phase = 3 if task_key == "flat" else 4  # flat 支持腿长 (P3), rough 地形 (P4)
    ctl = controller_cls(
        phase,
        action_scale=float(env._cfg.control_config.action_scale),
        wheel_action_scale=float(env._cfg.control_config.wheel_action_scale),
        dt=float(env._cfg.ctrl_dt),
    )
    env.init_state()

    # 查看器模型/数据 (★ 用解析视觉模型, get_playback_model 缺视觉网格→白屏)
    def _load_viewer_model_file(path: str) -> mujoco.MjModel:
        return (
            mujoco.MjModel.from_binary_path(path)
            if Path(path).suffix.lower() == ".mjb"
            else mujoco.MjModel.from_xml_path(path)
        )

    mj_model = None
    try:
        with tempfile.TemporaryDirectory(prefix=f"{label}-viewer-") as _tmp:
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
    commander.height_target = 0.518
    commander.height_step = 0.02
    commander.height_min, commander.height_max = 0.48, 0.55  # ★ 可达高度范围

    paused = False
    ctrl_dt = float(env._cfg.ctrl_dt)
    focus_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    step_counter = 0

    def _on_key(keycode: int) -> int | None:
        nonlocal paused
        if keycode == ord(" "):
            paused = not paused
            print(f"[{label}] {'paused' if paused else 'resumed'} (space)")
            return 1
        elif keycode in (_KEY_ENTER, _KEY_KP_ENTER):
            commander.zero()
            print(f"[{label}] stop")
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
            ctl.reset()
            commander.zero()
            env.init_state()
            print(f"[{label}] reset (backspace)")
            return 1
        else:
            return None
        print(f"[{label}] {commander.describe()}")
        return 1

    print(f"[{label}] 经典 {label} 平衡交互 (键盘):")
    print("  ↑/↓ 前进/后退 · ←/→ 转向 · Q/E 腿长 · Enter 停止 · Space 暂停 · Backspace 重置")
    print(f"  terrain={terrain} phase=P{phase}")

    with mujoco.viewer.launch_passive(mj_model, viz_data, key_callback=_on_key) as viewer:
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
                    print(f"[{label}] ⚠️ 机器人跌倒 — 按 Backspace 重置")
            phys = np.asarray(env._backend.get_physics_state(), dtype=np.float64)[0]
            mujoco.mj_setState(mj_model, viz_data, phys[1:], mujoco.mjtState.mjSTATE_PHYSICS)
            mujoco.mj_forward(mj_model, viz_data)
            viewer.sync()  # ★ 强制刷新 viewer 画面
            if step_counter % 100 == 0:
                print(
                    f"  t={step_counter * ctrl_dt:.1f}s vx_cmd={cmd[0]:+.2f} "
                    f"base_x={phys[1]:+.2f} base_z={phys[3]:.3f} v={sensors['v']:+.2f}"
                )
            step_counter += 1
            time.sleep(ctrl_dt / speed * 0.8)
    env.close()
    return 0
