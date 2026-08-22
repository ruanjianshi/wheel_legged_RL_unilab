#!/usr/bin/env python3
"""Record the full jump trajectory per algorithm's best checkpoint.

Reuses verify_jump.py's env-build / actor-loading helpers so the evaluation
matches training exactly (trained_env_overrides from run_config.json).

For each algorithm it runs ONE jump cycle (settle -> trigger ON -> tail) and
saves the per-ctrl-step time series in the same schema as
record_jump_trajectory.py (plus standing_z / terminated):

    t          : time (s), from step 0
    base_z     : base link height (m)
    hip_pitch  : [L, R] hip pitch joint angles (rad)
    hip_roll   : [L, R] hip roll joint angles (rad)      [v2 added]
    knee       : [L, R] knee joint angles (rad)
    base_euler : [roll, pitch, yaw] base ZYX euler (rad) [v2 added]
    linvel     : base local linear velocity (m/s)        [v2 added]
    phase      : SLIP-FSM phase id (jump_srl envs only; -1 = none)
    standing_z : median grounded base_z with trigger off (m)
    terminated : whether the episode terminated before the window ended
    ctrl_dt    : control period (s)

Output: jump_management/results/jump_traj_{srl,ppo,vmc,srlvmc}.npz
Fed by scripts/plot_jump_trajectory.py for the reference-style paper figure.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.xqrobotwl.dump_pose_data import (  # noqa: E402
    _quat_to_euler,
    _world_to_local,
)
from tools.xqrobotwl.verify_jump import (  # noqa: E402
    _DR,
    _REWARD,
    load_actor,
    trained_env_overrides,
)

# Final checkpoints used by the paper's unified 2026-08-17 evaluation.
# (npz stem, env task, checkpoint path)
JOBS: dict[str, tuple[str, str, str]] = {
    "SRL": (
        "jump_traj_srl",
        "XqRobotWLJumpSRLFlat",
        "logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat/2026-08-17_00-55-51_mujoco_final10000/model_9999.pt",
    ),
    "PPO": (
        "jump_traj_ppo",
        "XqRobotWLJumpFlat",
        "logs/rsl_rl_ppo/XqRobotWLJumpFlat/2026-08-17_00-54-30_mujoco_final10000/model_9999.pt",
    ),
    "PPO+VMC": (
        "jump_traj_vmc",
        "XqRobotWLJumpVMC",
        "logs/rsl_rl_ppo/XqRobotWLJumpVMC/2026-08-17_00-54-30_mujoco_final10000/model_9999.pt",
    ),
    "SRL+VMC": (
        "jump_traj_srlvmc",
        "XqRobotWLJumpSRLVMC",
        "logs/rsl_rl_ppo/XqRobotWLJumpSRLVMC/2026-08-17_12-50-27_mujoco_v8e5_10000/model_9999.pt",
    ),
}

SETTLE = 50  # trigger-off steps first (let the reset drop settle)
# Keep the command window identical to the repeated-jump protocol.  A longer
# trigger window can start a second FSM cycle and is therefore unsuitable for
# a representative single-jump trace.
ON = 100  # trigger-on steps (one crouch + thrust + flight cycle)
TAIL = 120  # trailing trigger-off steps (landing + recover)
HIDDEN = [512, 512, 256, 128]
CTRL_DT = 0.01  # xqrobotwl base.py default; overwritten from env at runtime
DOF = {"hip_roll": [0, 3], "hip": [1, 4], "knee": [2, 5]}  # [L, R] indices in dof order


def main() -> int:
    from unilab.base import registry
    from unilab.training import ensure_registries

    ensure_registries()

    for algo, (stem, task, ckpt) in JOBS.items():
        path = ROOT / ckpt
        if not path.exists():
            print(f"SKIP {algo}: no checkpoint {path}")
            continue
        trained_ov = trained_env_overrides(str(path))
        if trained_ov is None:
            trained_ov = {
                "reward_config": _REWARD,
                "domain_rand": _DR,
                "control_config": (
                    {"action_scale": 0.6, "wheel_action_scale": 10.0, "clip_actions": 1.0}
                    if "VMC" in task
                    else {"action_scale": 0.6, "wheel_action_scale": 10.0, "clip_actions": 100.0}
                ),
            }
        env = registry.make(task, num_envs=1, sim_backend="mujoco", env_cfg_override=trained_ov)
        try:
            dt = float(getattr(env._cfg, "ctrl_dt", CTRL_DT))
            obs_dim = env.obs_groups_spec["obs"]
            actor = load_actor(str(path), obs_dim, 8, HIDDEN)
            env.init_state()

            t: list[float] = []
            base_z: list[float] = []
            hip: list[list[float]] = []
            hip_roll: list[list[float]] = []
            knee: list[list[float]] = []
            euler: list[list[float]] = []
            linvel: list[list[float]] = []
            phase: list[float] = []
            stand_samples: list[float] = []
            terminated = False

            total = SETTLE + ON + TAIL
            with torch.no_grad():
                for step in range(total):
                    trigger = 1.0 if SETTLE <= step < SETTLE + ON else 0.0
                    env.state.info["commands"][:, 4] = trigger
                    obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                    action = actor(obs).numpy()
                    state = env.step(action)
                    dof_pos = env.get_dof_pos()
                    t.append(step * dt)
                    base_z.append(float(np.asarray(env._backend.get_base_pos())[0, 2]))
                    hip.append([float(dof_pos[0, i]) for i in DOF["hip"]])
                    hip_roll.append([float(dof_pos[0, i]) for i in DOF["hip_roll"]])
                    knee.append([float(dof_pos[0, i]) for i in DOF["knee"]])
                    quat = np.asarray(env._backend.get_base_quat())[0]
                    euler.append(_quat_to_euler(quat).tolist())
                    # 本地系前向速度 (x 为机身前向)
                    lv = _world_to_local(quat[None, :], np.asarray(env._backend.get_base_lin_vel()))
                    linvel.append(lv[0].tolist())
                    phase.append(float(getattr(env, "_fsm_state", np.array([-1.0]))[0]))
                    wc = state.info.get("wheel_contact", np.zeros((1, 2)))
                    if trigger == 0.0 and np.mean(wc) > 0.99:
                        stand_samples.append(base_z[-1])
                    if state.terminated[0]:
                        terminated = True
                        break

            standing_z = float(np.median(stand_samples)) if stand_samples else float(base_z[0])
            dest = ROOT / "latex" / "Wheeled-SRL-Jumping" / "data" / f"{stem}.npz"
            np.savez_compressed(
                dest,
                t=np.asarray(t),
                base_z=np.asarray(base_z),
                hip_pitch=np.asarray(hip),
                hip_roll=np.asarray(hip_roll),
                knee=np.asarray(knee),
                base_euler=np.asarray(euler),
                linvel=np.asarray(linvel),
                phase=np.asarray(phase),
                standing_z=np.asarray([standing_z]),
                terminated=np.asarray([terminated]),
                ctrl_dt=dt,
                task=task,
                checkpoint=str(ckpt),
            )
            peak = max(base_z) - standing_z
            print(
                f"{algo:<8} steps={len(t):3d} standing_z={standing_z:.3f} "
                f"peak={peak:.3f} m  terminated={terminated} -> {dest.name}"
            )
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
