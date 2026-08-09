#!/usr/bin/env python3
"""Diagnose VMC thrust mechanics: trace L0 tracking, VMC force, phase over a
single jump. Answers WHY the VMC variants jump lower than joint-space SRL.

Dumps, at each ctrl step:
    phase       : SLIP-FSM phase
    L0_ref      : commanded virtual-leg length (m)
    L0          : actual virtual-leg length (m)
    L0_dot      : virtual-leg length rate (m/s)
    force_L0    : VMC leg-length force (N), pre-Jacobian
    kp_eff      : effective kp_l0 (default or phase-scaled)
    kd_eff      : effective kd_l0
    ff_eff      : effective feedforward (N)
    base_z      : base height (m)
    wheel_contact: [L,R] contact bool
"""

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
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--settle", type=int, default=40)
    p.add_argument("--pulse", type=int, default=140)
    p.add_argument("--tail", type=int, default=120)
    args = p.parse_args()

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
    env = registry.make(args.task, num_envs=1, sim_backend="mujoco", env_cfg_override=ov)
    try:
        actor = load_actor(args.checkpoint, env.obs_groups_spec["obs"], 8, [512, 512, 256, 128])
        env.init_state()
        total = args.settle + args.pulse + args.tail
        rows = []
        with torch.no_grad():
            for step in range(total):
                trigger = 1.0 if args.settle <= step < args.settle + args.pulse else 0.0
                env.state.info["commands"][:, 4] = trigger
                obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                action = actor(obs).numpy()
                state = env.step(action)
                dof_pos = env.get_dof_pos()
                dof_vel = env.get_dof_vel()
                # VMC kinematics (left leg index 0)
                vmc = env._vmc
                theta1, theta2, theta0, L0, theta0_dot, L0_dot = vmc.compute_kinematics(
                    dof_pos, dof_vel
                )
                L0_l = float(L0[0, 0])
                L0dot_l = float(L0_dot[0, 0])
                # effective gains per phase (env override, NOT raw vmc defaults)
                if hasattr(env, "get_l0_control_parameters"):
                    kp, kd, ff = env.get_l0_control_parameters()
                else:
                    kp, kd, ff = vmc.get_l0_control_parameters()
                kp_l, kd_l, ff_l = float(kp[0, 0]), float(kd[0, 0]), float(ff[0, 0])
                # L0 reference: env stores the physical policy_ctrl used this step
                base_z = float(np.asarray(env._backend.get_base_pos())[0, 2])
                wc = state.info.get("wheel_contact", np.zeros((1, 2)))
                phase = float(getattr(env, "_fsm_state", np.array([-1.0]))[0])
                # reconstruct L0_ref (physical) from the action the policy commanded.
                # For SRL+VMC the FSM reference is blended in step(); the actual
                # physical L0_ref is exactly what compute_torques saw, but we don't
                # store it. Approximate from kinematics target via L0_ref = L0 +
                # (force_L0 - ff)/(kp): not exact. Instead log raw action L0 channel.
                raw_act = action[0, 2]  # L0_L normalized action
                rows.append(
                    (
                        step,
                        phase,
                        L0_l,
                        L0dot_l,
                        base_z,
                        int(wc[0, 0]),
                        int(wc[0, 1]),
                        kp_l,
                        kd_l,
                        ff_l,
                        raw_act,
                    )
                )
        # find thrust window (phase==1 or phase==2) and print focused view
        print(f"task={args.task} step phase L0    L0dot  base_z  Lc Rc  kp_eff kd_eff ff_eff L0act")
        for r in rows:
            step, phase, L0l, L0d, bz, lc, rc, kp, kd, ff, act = r
            if args.settle <= step and phase >= 0:
                print(
                    f"{step:4d} {int(phase):5d} {L0l:5.3f} {L0d:6.2f} {bz:5.3f}  {lc}  {rc}  "
                    f"{kp:6.0f} {kd:5.1f} {ff:6.0f} {act:5.2f}"
                )
            elif step in (args.settle - 1, args.settle, args.settle + 5, total - 1):
                print(
                    f"{step:4d} {int(phase):5d} {L0l:5.3f} {L0d:6.2f} {bz:5.3f}  {lc}  {rc}  "
                    f"{kp:6.0f} {kd:5.1f} {ff:6.0f} {act:5.2f}"
                )
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
