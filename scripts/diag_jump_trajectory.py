#!/usr/bin/env python3
"""Diagnose trained jump policy: dump base_z, wheel_contact, jump_phase, FSM
per step to find why reward/jump_height is high but verify_jump air is low."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verify_jump import _ActorMLP, load_actor, trained_env_overrides  # reuse loader


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--jump_every", type=int, default=200)
    p.add_argument("--hidden", default="512,512,256,128")
    args = p.parse_args()
    hidden = [int(x) for x in args.hidden.split(",")]

    from unilab.base import registry
    from unilab.training import ensure_registries

    ensure_registries()
    ov = trained_env_overrides(args.checkpoint)
    if ov is None:
        vmc_task = "VMC" in args.task
        ctrl = (
            {"action_scale": 0.6, "wheel_action_scale": 10.0, "clip_actions": 1.0}
            if vmc_task
            else {"action_scale": 0.6, "wheel_action_scale": 10.0, "clip_actions": 100.0}
        )
        ov = {
            "reward_config": {
                "scales": {
                    "jump_height": 12.0,
                    "vertical_thrust": 30.0,
                    "crouch_prep": 4.0,
                    "landing_soft": 15.0,
                    "wheel_air_time": 20.0,
                    "crouch_depth": 4.0,
                    "alive": 1.0,
                    "base_height": -60.0,
                    "orientation": -5.0,
                    "joint_action_rate": -0.1,
                    "wheel_action_rate": -0.015,
                    "leg_mirror": -12.0,
                    "tsk": -2.0,
                    "anti_loiter": 12.0,
                    "lean_forward": 5.0,
                    "tracking_lin_vel": 2.0,
                    "tracking_ang_vel": 1.0,
                    "lin_vel_z": -0.2,
                    "ang_vel_xy": -0.05,
                },
                "tracking_sigma": 0.3,
                "base_height_target": 0.55,
                "jump_height_target": 1.0,
                "crouch_height_target": 0.40,
                "max_tilt_deg": 60.0,
                "min_base_height": 0.15,
                "jump_curriculum_start": 0,
                "jump_curriculum_end": 0,
            },
            "domain_rand": {
                "randomize_base_mass": False,
                "randomize_ground_friction": False,
                "randomize_kp": False,
                "randomize_kd": False,
                "random_com": False,
                "randomize_leg_length": False,
            },
            "control_config": ctrl,
        }
    env = registry.make(
        args.task,
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override=ov,
    )
    obs_dim = env.obs_groups_spec["obs"]
    actor = load_actor(args.checkpoint, obs_dim, 8, hidden)
    env.init_state()

    print(f"task={args.task} checkpoint={Path(args.checkpoint).name}")
    print("step  trig  base_z  wl_contact wr_contact  phase  fsm   maxz")
    maxz = 0.0
    with torch.no_grad():
        for step in range(args.steps):
            trigger = 1.0 if (step % args.jump_every) < (args.jump_every // 2) else 0.0
            env.state.info["commands"][:, 4] = trigger
            obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
            action = actor(obs).numpy()
            state = env.step(action)
            base_z = float(np.asarray(env._backend.get_base_pos())[0, 2])
            maxz = max(maxz, base_z)
            wc = state.info.get("wheel_contact", np.zeros((1, 2)))
            phase = float(state.info.get("jump_phase", np.zeros(1))[0])
            fsm = float(
                getattr(env, "_fsm_state", np.zeros(1))[0] if hasattr(env, "_fsm_state") else 0
            )
            if step % 10 == 0 or base_z > 0.75:
                print(
                    f"{step:5d}  {int(trigger)}   {base_z:.3f}  "
                    f"{int(wc[0, 0])}          {int(wc[0, 1])}          "
                    f"{phase:4.0f}   {fsm:.1f}   {maxz:.3f}"
                )
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
