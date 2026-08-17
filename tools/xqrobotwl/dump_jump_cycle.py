#!/usr/bin/env python3
"""Dump a per-step jump-cycle pose CSV for §7.5 phase-chain / wheel-speed audit.

Pulses the jump trigger once (settle -> ON -> tail) and writes every control
step's full pose: base_z, 6 leg joints, base euler, linvel, gyro, wheel
angular velocity, up_z, wheel_contact (geometric) and jump_phase.

Usage:
    uv run tools/xqrobotwl/dump_jump_cycle.py \
        --task XqRobotWLJumpFlat \
        --checkpoint logs/rsl_rl_ppo/XqRobotWLJumpFlat/<run>/model_N.pt \
        --out logs/pose_data/<prefix>_jump_cycle.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.xqrobotwl.verify_jump import (  # noqa: E402
    _DR,
    _REWARD,
    load_actor,
    trained_env_overrides,
)

COLUMNS = [
    "step",
    "trigger",
    "base_z",
    "L_hip_roll",
    "L_hip_pitch",
    "L_knee",
    "R_hip_roll",
    "R_hip_pitch",
    "R_knee",
    "L_wheel_rads",
    "R_wheel_rads",
    "base_roll",
    "base_pitch",
    "base_yaw",
    "linvel_x",
    "linvel_y",
    "linvel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "up_z",
    "wheel_contact_L",
    "wheel_contact_R",
    "jump_phase",
]


def _quat_to_euler(q) -> np.ndarray:
    w, x, y, z = q[0], q[1], q[2], q[3]
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return np.array([roll, pitch, yaw])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--settle", type=int, default=60)
    p.add_argument("--pulse", type=int, default=140)
    p.add_argument("--tail", type=int, default=160)
    p.add_argument("--wheel_radius", type=float, default=0.11)
    p.add_argument("--hidden", default="512,512,256,128")
    args = p.parse_args()
    hidden = [int(x) for x in args.hidden.split(",")]

    from unilab.base import registry
    from unilab.training import ensure_registries

    ensure_registries()
    ov = trained_env_overrides(args.checkpoint)
    if ov is None:
        vmc = "VMC" in args.task
        ctrl = (
            {"action_scale": 0.6, "wheel_action_scale": 10.0, "clip_actions": 1.0}
            if vmc
            else {"action_scale": 0.6, "wheel_action_scale": 10.0, "clip_actions": 100.0}
        )
        ov = {"reward_config": _REWARD, "domain_rand": _DR, "control_config": ctrl}
    env = registry.make(args.task, num_envs=1, sim_backend="mujoco", env_cfg_override=ov)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        actor = load_actor(args.checkpoint, env.obs_groups_spec["obs"], 8, hidden)
        env.init_state()
        total = args.settle + args.pulse + args.tail
        rows = []
        with torch.no_grad():
            for step in range(total):
                trigger = 1.0 if args.settle <= step < args.settle + args.pulse else 0.0
                env.state.info["commands"][:, 4] = trigger
                obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                action = actor(obs).numpy()
                st = env.step(action)
                dof = env.get_dof_pos()[0]
                dof_vel = env.get_dof_vel()[0]
                base_pos = np.asarray(env._backend.get_base_pos())[0]
                quat = env._backend.get_base_quat()[0] if hasattr(env._backend, "get_base_quat") else None
                euler = _quat_to_euler(quat) if quat is not None else np.zeros(3)
                linvel = env.get_local_linvel()[0]
                gyro = np.asarray(env._backend.get_sensor_data("gyro")).reshape(-1, 3)[0]
                up_z = float(np.asarray(env._backend.get_sensor_data("upvector")).reshape(-1, 3)[0, 2])
                lz = float(np.asarray(env._backend.get_sensor_data("left_wheel_world_pos")).reshape(-1, 3)[0, 2])
                contact = 1.0 if lz < args.wheel_radius + 0.02 else 0.0
                phase = float(st.info.get("jump_phase", np.zeros(1))[0])
                rows.append([
                    step, trigger, round(float(base_pos[2]), 3),
                    round(float(dof[0]), 3), round(float(dof[1]), 3), round(float(dof[2]), 3),
                    round(float(dof[3]), 3), round(float(dof[4]), 3), round(float(dof[5]), 3),
                    round(float(dof_vel[6]), 3), round(float(dof_vel[7]), 3),
                    round(float(euler[0]), 3), round(float(euler[1]), 3), round(float(euler[2]), 3),
                    round(float(linvel[0]), 3), round(float(linvel[1]), 3), round(float(linvel[2]), 3),
                    round(float(gyro[0]), 3), round(float(gyro[1]), 3), round(float(gyro[2]), 3),
                    round(up_z, 3), int(contact), int(contact), int(phase),
                ])
                if st.terminated[0]:
                    break
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            w.writerows(rows)
        z = [r[2] for r in rows]
        print(f"wrote {out} rows={len(rows)} max_base_z={max(z):.3f} terminated={st.terminated[0]}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
