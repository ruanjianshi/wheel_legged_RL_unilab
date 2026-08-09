#!/usr/bin/env python3
"""Record a single jump trajectory (base_z + joint angles + FSM phase) for one
trained checkpoint, for paper time-series figures.

Triggers ONE jump (settle -> crouch -> thrust -> flight -> landing) and saves
the per-ctrl-step time series to an .npz file:

    t         : time (s), starting at the settle window
    base_z    : base link height (m)
    hip_pitch : [L, R] hip pitch joint angles (rad)
    knee      : [L, R] knee joint angles (rad)
    phase     : SLIP-FSM phase id (jump_vmc/srl envs only; -1 = none)

Usage:
    uv run python scripts/record_jump_trajectory.py \
        --task XqRobotWLJumpVMC \
        --checkpoint logs/rsl_rl_ppo/.../model_9999.pt \
        --out jump_management/results/jump_traj_vmc.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify_jump import load_actor, trained_env_overrides  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--settle", type=int, default=40, help="steps trigger-off before pulse")
    parser.add_argument("--pulse", type=int, default=140, help="steps trigger-on")
    parser.add_argument("--tail", type=int, default=120, help="steps trigger-off after pulse")
    parser.add_argument("--hidden", default="512,512,256,128")
    args = parser.parse_args()
    hidden = [int(x) for x in args.hidden.split(",")]

    from unilab.base import registry
    from unilab.training import ensure_registries

    ensure_registries()
    ov = trained_env_overrides(args.checkpoint)
    if ov is None:
        ctrl = (
            {"action_scale": 0.6, "wheel_action_scale": 10.0, "clip_actions": 1.0}
            if "VMC" in args.task
            else {"action_scale": 0.6, "wheel_action_scale": 10.0, "clip_actions": 100.0}
        )
        ov = {"control_config": ctrl}
    env = registry.make(
        args.task, num_envs=1, sim_backend="mujoco", env_cfg_override=ov
    )
    try:
        obs_dim = env.obs_groups_spec["obs"]
        actor = load_actor(args.checkpoint, obs_dim, 8, hidden)
        env.init_state()

        total = args.settle + args.pulse + args.tail
        t, base_z, hip, knee, phase = [], [], [], [], []
        with torch.no_grad():
            for step in range(total):
                trigger = 1.0 if args.settle <= step < args.settle + args.pulse else 0.0
                env.state.info["commands"][:, 4] = trigger
                obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                action = actor(obs).numpy()
                state = env.step(action)
                dof_pos = env.get_dof_pos()
                t.append(step * float(env._cfg.ctrl_dt))
                base_z.append(float(np.asarray(env._backend.get_base_pos())[0, 2]))
                # dof order: [L_roll, L_pitch, L_knee, R_roll, R_pitch, R_knee, L_wheel, R_wheel]
                hip.append([float(dof_pos[0, 1]), float(dof_pos[0, 4])])
                knee.append([float(dof_pos[0, 2]), float(dof_pos[0, 5])])
                phase.append(float(getattr(env, "_fsm_state", np.array([-1.0]))[0]))
                if state.terminated[0]:
                    break

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            t=np.array(t),
            base_z=np.array(base_z),
            hip_pitch=np.array(hip),
            knee=np.array(knee),
            phase=np.array(phase),
            ctrl_dt=float(env._cfg.ctrl_dt),
            task=args.task,
            checkpoint=str(args.checkpoint),
        )
        print(f"task={args.task} steps={len(t)} max_base_z={max(base_z):.3f} "
              f"standing={np.median(base_z[: args.settle]):.3f} -> {out}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
