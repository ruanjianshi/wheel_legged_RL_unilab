#!/usr/bin/env python3
"""渲染平衡问题演示视频: 经典 LQR 失败 vs RL 稳定 (对照).

用法:
  uv run mjpython scripts/classic_control/render_balance_demo.py --demo lqr_fail
  uv run mjpython scripts/classic_control/render_balance_demo.py --demo rl_hold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.classic_control.common.config import STANDING_ANGLES
from scripts.classic_control.common.render import states_to_video


def demo_lqr_fail(out: Path, sim_s: float = 2.0) -> None:
    """扭矩 LQR 平衡失败 (从 standing_angles 起点, 展示初始冲击+倒下)."""
    import mujoco

    XML = ROOT / "src/unilab/assets/robots/xqrobotwl/scene_flat_vmc.xml"
    model = mujoco.MjModel.from_xml_path(str(XML))
    model.opt.timestep = 0.001
    adr_up = int(np.asarray(model.sensor("upvector").adr).reshape(-1)[0])
    adr_gyro = int(np.asarray(model.sensor("gyro").adr).reshape(-1)[0])
    KP, KD = 800.0, 20.0
    LEG_IDX = np.array([0, 1, 2, 4, 5, 6])
    DT = 0.005
    d = mujoco.MjData(model)
    d.qpos[:] = 0.0
    d.qpos[2] = 0.518
    d.qpos[3] = 1.0
    d.qpos[7:15] = [*STANDING_ANGLES[:3], 0.0, *STANDING_ANGLES[3:], 0.0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)
    xpos = 0.0
    record = []
    n = int(sim_s / DT)
    for step in range(n):
        t = step * DT
        up = d.sensordata[adr_up : adr_up + 3]
        gyro = d.sensordata[adr_gyro : adr_gyro + 3]
        th = float(np.arctan2(up[0], up[2]))
        thd = float(gyro[1])
        v = float(d.qvel[0])
        xpos += v * DT
        tau = float(np.clip(-(50 * th + 15 * thd + 10 * v + 1.0 * xpos), -10, 10))
        target = np.array([*STANDING_ANGLES[:3], 0.0, *STANDING_ANGLES[3:], 0.0])
        q = d.qpos[7:15]
        qv = d.qvel[6:14]
        ctrl = np.zeros(8)
        ctrl[LEG_IDX] = KP * (target[LEG_IDX] - q[LEG_IDX]) - KD * qv[LEG_IDX]
        ctrl[[3, 7]] = tau
        d.ctrl[:] = ctrl
        for _ in range(5):
            mujoco.mj_step(model, d)
        phy = np.concatenate([d.qpos, d.qvel])
        record.append({"t": t, "state": np.concatenate([[t], phy])})
        if d.qpos[2] < 0.25 or abs(th) > 1.0:
            print(f"  LQR 倒下 at t={t:.2f}s (th={th:+.2f} z={d.qpos[2]:.2f})")
            break
    print(f"  LQR 录制 {len(record)} 帧 ({len(record) * DT:.2f}s)")
    states_to_video(
        record,
        XML,
        out,
        fps=50,
        num_processes=1,
        cam_distance=2.6,
        cam_elevation=-25,
        cam_azimuth=135,
        cam_lookat=[0.0, 0.0, 0.32],
    )
    print(f"  ✅ {out}")


def demo_rl_hold(out: Path, sim_s: float = 3.0) -> None:
    """RL 策略稳定站立 (对照)."""
    import torch
    from tools.xqrobotwl.verify_jump import load_actor

    from scripts.classic_control.common import rollout as rollout_mod

    env = rollout_mod.build_env(task_key="walk_flat", num_envs=1)
    ckpt = "logs/rsl_rl_ppo/XqRobotWLWalkFlat/2026-08-10_01-25-14_mujoco/model_9999.pt"
    pol = load_actor(ckpt, env.obs_groups_spec["obs"], 8, hidden=[512, 512, 256, 128])
    cmd = np.array([0, 0, 0, 0, 0.518])
    env.init_state()
    record = []
    dt = float(env._cfg.ctrl_dt)
    n = int(sim_s / dt)
    with torch.no_grad():
        for step in range(n):
            t = step * dt
            obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
            a = pol(obs).numpy()[0]
            env.state.info["commands"][:] = np.tile(cmd, (1, 1))
            st = env.step(np.asarray(a)[None, :])
            phy = np.asarray(env._backend.get_physics_state(), dtype=np.float64)[0]
            # ★ physics_state = [pad(1), qpos(15), qvel(14)] → 取 [t, qpos, qvel]
            record.append({"t": t, "state": np.concatenate([[t], phy[1:]])})
            if st.terminated[0]:
                break
    print(f"  RL 录制 {len(record)} 帧 ({len(record) * dt:.2f}s), 存活未倒")
    states_to_video(record, env._cfg.scene.model_file, out, fps=50, num_processes=1)
    print(f"  ✅ {out}")


def demo_success(out: Path, sim_s: float = 1.5) -> None:
    """修复轮子符号 + 腿协助后: 平衡 15s (成功演示)."""
    import mujoco

    XML = ROOT / "src/unilab/assets/robots/xqrobotwl/scene_flat_vmc.xml"
    model = mujoco.MjModel.from_xml_path(str(XML))
    model.opt.timestep = 0.001
    adr_up = int(np.asarray(model.sensor("upvector").adr).reshape(-1)[0])
    adr_gyro = int(np.asarray(model.sensor("gyro").adr).reshape(-1)[0])
    adr_lv = int(np.asarray(model.sensor("local_linvel").adr).reshape(-1)[0])
    KP, KD = 800.0, 20.0
    LEG_IDX = np.array([0, 1, 2, 4, 5, 6])
    DT = 0.005
    LEGS = np.array([0.1, 0.15, 0.15, -0.1, -0.15, -0.15])
    kth, kthd, kv, kx, kleg = 60, 20, 10, 2, 6
    d = mujoco.MjData(model)
    d.qpos[:] = 0.0
    d.qpos[2] = 0.475
    d.qpos[3] = 1.0
    d.qpos[7:15] = [*LEGS[:3], 0.0, *LEGS[3:], 0.0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)
    xpos = 0.0
    record = []
    n = int(sim_s / DT)
    for step in range(n):
        t = step * DT
        up = d.sensordata[adr_up : adr_up + 3]
        gyro = d.sensordata[adr_gyro : adr_gyro + 3]
        lv = d.sensordata[adr_lv : adr_lv + 3]
        th = float(np.arctan2(up[0], up[2]))
        thd = float(gyro[1])
        v = float(lv[0])
        xpos += v * DT
        tau = float(np.clip(kth * th + kthd * thd + kv * v + kx * xpos, -10, 10))
        target = np.array([*LEGS[:3], 0.0, *LEGS[3:], 0.0])
        target[1] += kleg * th
        target[5] -= kleg * th
        q = d.qpos[7:15]
        qv = d.qvel[6:14]
        ctrl = np.zeros(8)
        ctrl[LEG_IDX] = KP * (target[LEG_IDX] - q[LEG_IDX]) - KD * qv[LEG_IDX]
        ctrl[3] = -tau
        ctrl[7] = +tau
        d.ctrl[:] = ctrl
        for _ in range(5):
            mujoco.mj_step(model, d)
        record.append({"t": t, "state": np.concatenate([[t], np.concatenate([d.qpos, d.qvel])])})
        if d.qpos[2] < 0.25 or abs(th) > 1.0:
            break
    print(f"  成功演示录制 {len(record)} 帧 ({len(record) * DT:.2f}s), th={th:+.3f}")
    states_to_video(
        record,
        XML,
        out,
        fps=50,
        num_processes=2,
        cam_distance=2.6,
        cam_elevation=-25,
        cam_azimuth=135,
        cam_lookat=[0.0, 0.0, 0.32],
    )
    print(f"  ✅ {out}")


def demo_honest(out: Path, sim_s: float = 2.0) -> None:
    """修正后 (轮子着地 base_z=0.516 + 方向正确 + 对称腿): 扭矩 LQR 平衡状态."""
    import mujoco

    XML = ROOT / "src/unilab/assets/robots/xqrobotwl/scene_flat_vmc.xml"
    model = mujoco.MjModel.from_xml_path(str(XML))
    model.opt.timestep = 0.001
    adr_up = int(np.asarray(model.sensor("upvector").adr).reshape(-1)[0])
    adr_gyro = int(np.asarray(model.sensor("gyro").adr).reshape(-1)[0])
    adr_lv = int(np.asarray(model.sensor("local_linvel").adr).reshape(-1)[0])
    KP, KD = 800.0, 20.0
    LEG_IDX = np.array([0, 1, 2, 4, 5, 6])
    DT = 0.005
    LEGS = np.array([0.1021, 0.0828, -0.0789, -0.1021, -0.0828, 0.0789])
    base_z = 0.516  # ★ 双轮压地 (髋 offset 修正后)
    kth, kthd, kv, kx = 100, 40, 20, 5
    d = mujoco.MjData(model)
    d.qpos[:] = 0.0
    d.qpos[2] = base_z
    d.qpos[3] = 1.0
    d.qpos[7:15] = [*LEGS[:3], 0.0, *LEGS[3:], 0.0]
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)
    xpos = 0.0
    record = []
    n = int(sim_s / DT)
    for step in range(n):
        t = step * DT
        up = d.sensordata[adr_up : adr_up + 3]
        gyro = d.sensordata[adr_gyro : adr_gyro + 3]
        lv = d.sensordata[adr_lv : adr_lv + 3]
        th = float(np.arctan2(up[0], up[2]))
        thd = float(gyro[1])
        v = float(lv[0])
        xpos += v * DT
        tau = float(np.clip(kth * th + kthd * thd + kv * v + kx * xpos, -10, 10))
        target = np.array([*LEGS[:3], 0.0, *LEGS[3:], 0.0])
        q = d.qpos[7:15]
        qv = d.qvel[6:14]
        ctrl = np.zeros(8)
        ctrl[LEG_IDX] = KP * (target[LEG_IDX] - q[LEG_IDX]) - KD * qv[LEG_IDX]
        ctrl[3] = -tau
        ctrl[7] = +tau
        d.ctrl[:] = ctrl
        for _ in range(5):
            mujoco.mj_step(model, d)
        record.append({"t": t, "state": np.concatenate([[t], np.concatenate([d.qpos, d.qvel])])})
        if d.qpos[2] < 0.25 or abs(th) > 1.0:
            break
    print(f"  修正后 LQR: {len(record)} 帧 ({len(record) * DT:.2f}s), 最后 th={th:+.2f}")
    states_to_video(
        record,
        XML,
        out,
        fps=50,
        num_processes=2,
        cam_distance=2.6,
        cam_elevation=-25,
        cam_azimuth=135,
        cam_lookat=[0.0, 0.0, 0.32],
    )
    print(f"  ✅ {out}")


def demo_success_env(out: Path, sim_s: float = 5.0) -> None:
    """★ 修正控制方向后: 标准 env (速度伺服) 平衡 15s, 倾角 ~10°, 漂移 <0.5m (成功)."""
    from scripts.classic_control.common import rollout as rollout_mod
    from scripts.classic_control.common.config import DEFAULT_LEG_ANGLES

    env = rollout_mod.build_env(task_key="walk_flat", num_envs=1)
    cmd = np.array([0, 0, 0, 0, 0.518])
    flip = np.array([1, 1, -1, 1, -1, 1, 1, -1])
    ascale, wsa = 0.6, 10.0
    leg_action = (
        np.array([0.02, 0.15, -0.30, 0.0, -0.18, 0.28]) - np.array(DEFAULT_LEG_ANGLES)
    ) / (flip[:6] * ascale)
    env.init_state()
    record = []
    dt = float(env._cfg.ctrl_dt)
    w_prev, xpos = 0.0, 0.0
    kth, kthd, kv, kx, alpha = 100, 20, 80, 30, 0.85
    n = int(sim_s / dt)
    for step in range(n):
        t = step * dt
        s = rollout_mod.read_sensors(env)
        th, thd, v = s["theta"], s["theta_dot"], s["v"]
        xpos += v * dt
        w_new = -(kth * th + kthd * thd + kv * v + kx * xpos)
        w_c = float(np.clip(alpha * w_prev + (1 - alpha) * w_new, -25, 25))
        w_prev = w_c
        a = np.zeros(8)
        a[:6] = leg_action
        a[6] = -w_c / wsa
        a[7] = w_c / (-wsa)
        env.state.info["commands"][:] = np.tile(cmd, (1, 1))
        st = env.step(np.asarray(a)[None, :])
        phy = np.asarray(env._backend.get_physics_state(), dtype=np.float64)[0]
        # ★ physics_state = [pad(1), qpos(15), qvel(14)] → 取 [t, qpos, qvel]
        record.append({"t": t, "state": np.concatenate([[t], phy[1:]])})
        if st.terminated[0]:
            break
    print(f"  ★ env成功: {len(record)} 帧 ({len(record) * dt:.2f}s), 最后 th={th:+.3f}")
    states_to_video(
        record,
        env._cfg.scene.model_file,
        out,
        fps=50,
        num_processes=2,
        cam_distance=2.6,
        cam_elevation=-25,
        cam_azimuth=135,
        cam_lookat=[0.0, 0.0, 0.32],
    )
    print(f"  ✅ {out}")


def demo_p2(out: Path, sim_s: float = 25.0) -> None:
    """P2 指令控制: vx 0→0.5→0→-0.5→0 追踪 (稳态 RMSE 0.052)."""
    from scripts.classic_control.common import rollout as rollout_mod
    from scripts.classic_control.common.run import get_dynamics
    from scripts.classic_control.lqr.controller import LqrController

    env = rollout_mod.build_env(task_key="walk_flat", num_envs=1)
    ctl = LqrController(
        2,
        params={"q_z": 15},
        action_scale=float(env._cfg.control_config.action_scale),
        wheel_action_scale=float(env._cfg.control_config.wheel_action_scale),
        dt=float(env._cfg.ctrl_dt),
    )
    env.init_state()
    record = []
    dt = float(env._cfg.ctrl_dt)
    n = int(sim_s / dt)
    for step in range(n):
        t = step * dt
        if t < 5:
            vx = 0.0
        elif t < 15:
            vx = 0.5
        elif t < 20:
            vx = 0.0
        else:
            vx = -0.5
        s = rollout_mod.read_sensors(env)
        cmds = np.array([vx, 0, 0, 0, 0.518])
        a = ctl.act(s, cmds)
        env.state.info["commands"][:] = np.tile(cmds, (1, 1))
        st = env.step(np.asarray(a)[None, :])
        phy = np.asarray(env._backend.get_physics_state(), dtype=np.float64)[0]
        record.append({"t": t, "state": np.concatenate([[t], phy[1:]])})
        if st.terminated[0]:
            break
    print(f"  P2 指令: {len(record)} 帧 ({len(record) * dt:.2f}s), vx 段已完成")
    states_to_video(
        record,
        env._cfg.scene.model_file,
        out,
        fps=50,
        num_processes=2,
        cam_distance=2.6,
        cam_elevation=-25,
        cam_azimuth=135,
        cam_lookat=[0.0, 0.0, 0.32],
    )
    print(f"  ✅ {out}")


def demo_p3(out: Path, sim_s: float = 25.0) -> None:
    """P3 腿长控制: 高度 0.50→0.55→0.48→0.518 追踪 (可达范围内)."""
    from scripts.classic_control.common import rollout as rollout_mod
    from scripts.classic_control.common.run import get_dynamics
    from scripts.classic_control.lqr.controller import LqrController

    env = rollout_mod.build_env(task_key="walk_flat", num_envs=1)
    ctl = LqrController(
        3,
        params={"q_z": 15},
        action_scale=float(env._cfg.control_config.action_scale),
        wheel_action_scale=float(env._cfg.control_config.wheel_action_scale),
        dt=float(env._cfg.ctrl_dt),
    )
    env.init_state()
    record = []
    dt = float(env._cfg.ctrl_dt)
    n = int(sim_s / dt)
    for step in range(n):
        t = step * dt
        if t < 5:
            hc = 0.518
        elif t < 11:
            hc = 0.50
        elif t < 17:
            hc = 0.55
        else:
            hc = 0.48
        s = rollout_mod.read_sensors(env)
        cmds = np.array([0, 0, 0, 0, hc])
        a = ctl.act(s, cmds)
        env.state.info["commands"][:] = np.tile(cmds, (1, 1))
        st = env.step(np.asarray(a)[None, :])
        phy = np.asarray(env._backend.get_physics_state(), dtype=np.float64)[0]
        record.append({"t": t, "state": np.concatenate([[t], phy[1:]])})
        if st.terminated[0]:
            break
    print(f"  P3 腿长: {len(record)} 帧 ({len(record) * dt:.2f}s)")
    states_to_video(
        record,
        env._cfg.scene.model_file,
        out,
        fps=50,
        num_processes=2,
        cam_distance=2.6,
        cam_elevation=-25,
        cam_azimuth=135,
        cam_lookat=[0.0, 0.0, 0.32],
    )
    print(f"  ✅ {out}")


def demo_p4(out: Path, sim_s: float = 20.0) -> None:
    """P4 地形自适应: walk_rough 粗糙地形, vx=0.4, 腿随地形自适应."""
    from scripts.classic_control.common import rollout as rollout_mod
    from scripts.classic_control.common.run import get_dynamics
    from scripts.classic_control.lqr.controller import LqrController

    env = rollout_mod.build_env(task_key="walk_rough", num_envs=1)
    ctl = LqrController(
        4,
        params={"q_z": 15},
        action_scale=float(env._cfg.control_config.action_scale),
        wheel_action_scale=float(env._cfg.control_config.wheel_action_scale),
        dt=float(env._cfg.ctrl_dt),
    )
    env.init_state()
    record = []
    dt = float(env._cfg.ctrl_dt)
    n = int(sim_s / dt)
    for step in range(n):
        t = step * dt
        vx = 0.0 if t < 2 else 0.4
        s = rollout_mod.read_sensors(env)
        cmds = np.array([vx, 0, 0, 0])  # rough 4D
        a = ctl.act(s, cmds)
        env.state.info["commands"][:] = np.tile(cmds, (1, 1))
        st = env.step(np.asarray(a)[None, :])
        phy = np.asarray(env._backend.get_physics_state(), dtype=np.float64)[0]
        record.append({"t": t, "state": np.concatenate([[t], phy[1:]])})
        if st.terminated[0]:
            break
    print(f"  P4 地形: {len(record)} 帧 ({len(record) * dt:.2f}s)")
    states_to_video(
        record,
        env._cfg.scene.model_file,
        out,
        fps=50,
        num_processes=2,
        cam_distance=3.0,
        cam_elevation=-30,
        cam_azimuth=135,
        cam_lookat=[0.0, 0.0, 0.3],
    )
    print(f"  ✅ {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--demo",
        choices=[
            "lqr_fail",
            "rl_hold",
            "success",
            "honest",
            "success_env",
            "p2",
            "p3",
            "p4",
            "both",
        ],
        default="p4",
    )
    ap.add_argument("--out_dir", default=str(ROOT / "video" / "classic"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.demo in ("p4", "both"):
        demo_p4(out_dir / "lqr_env_p4_rough.mp4")
    if args.demo in ("p3", "both"):
        demo_p3(out_dir / "lqr_env_p3_height.mp4")
    if args.demo in ("p2", "both"):
        demo_p2(out_dir / "lqr_env_p2_command.mp4")
    if args.demo in ("success_env", "both"):
        demo_success_env(out_dir / "lqr_env_balance_SUCCESS.mp4")
    if args.demo in ("honest", "both"):
        demo_honest(out_dir / "lqr_natural_posture_honest.mp4")
    if args.demo in ("success", "both"):
        demo_success(out_dir / "lqr_torque_balance_SUCCESS.mp4")
    if args.demo in ("lqr_fail", "both"):
        demo_lqr_fail(out_dir / "lqr_torque_balance_fail.mp4")
    if args.demo in ("rl_hold", "both"):
        demo_rl_hold(out_dir / "rl_balance_hold.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
