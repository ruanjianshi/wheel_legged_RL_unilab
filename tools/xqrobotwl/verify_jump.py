#!/usr/bin/env python3
"""Verify a trained jump checkpoint can jump: measure max height / air time / survival.

Usage:
    uv run scripts/verify_jump.py --task XqRobotWLJumpVMC \
        --checkpoint logs/rsl_rl_ppo/XqRobotWLJumpVMC/<run>/model_3000.pt

The reward scales are shared by all four jump tasks (phase-gated jump rewards),
so one minimal config builds any of them.  Deterministic policy (action mean) is
used; the jump trigger is pulsed every 2 s like the training command schedule.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_REWARD = {
    "scales": {
        "tracking_lin_vel": 2.0,
        "tracking_ang_vel": 1.0,
        "lin_vel_z": -0.2,
        "ang_vel_xy": -0.05,
        "base_height": -60.0,
        "orientation": -5.0,
        "joint_action_rate": -0.1,
        "wheel_action_rate": -0.015,
        "leg_mirror": -12.0,
        "tsk": -2.0,
        "alive": 1.0,
        "jump_height": 12.0,
        "crouch_prep": 4.0,
        "landing_soft": 15.0,
        "wheel_air_time": 20.0,
        "vertical_thrust": 30.0,
        "crouch_depth": 4.0,
        "anti_loiter": 12.0,
        "lean_forward": 5.0,
    },
    "tracking_sigma": 0.3,
    "base_height_target": 0.55,
    "jump_height_target": 1.0,
    "crouch_height_target": 0.40,
    "max_tilt_deg": 60.0,
    "min_base_height": 0.15,
    "jump_curriculum_start": 0,
    "jump_curriculum_end": 0,
}
_DR = {
    "randomize_base_mass": False,
    "randomize_ground_friction": False,
    "randomize_kp": False,
    "randomize_kd": False,
    "random_com": False,
    "randomize_leg_length": False,
}


class _ActorMLP(torch.nn.Module):
    """Mirror of the rsl_rl MLP actor stored in the checkpoint."""

    def __init__(self, obs_dim: int, hidden: list[int], out_dim: int):
        super().__init__()
        dims = [obs_dim, *hidden, out_dim]
        layers: list[torch.nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(torch.nn.ELU())
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def load_actor(checkpoint_path: str, obs_dim: int, num_actions: int, hidden: list[int]):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor_state = ckpt.get("actor_state_dict", ckpt.get("model_state_dict"))
    actor = _ActorMLP(obs_dim, hidden, num_actions)
    # checkpoint keys: 'mlp.0.weight', 'mlp.0.bias', ... (distribution.std_param ignored)
    weights = {k: v for k, v in actor_state.items() if k.startswith("mlp.")}
    actor.load_state_dict(weights)
    actor.eval()
    return actor


def trained_env_overrides(checkpoint_path: str) -> dict | None:
    """Rebuild env from the training run's run_config.json, so evaluation runs
    under the exact trained vmc/reward/feedback_gain config (not stale defaults).

    The checkpoint lives in ``logs/rsl_rl_ppo/<Task>/<run>/model_N.pt``; the
    sibling ``run_config.json`` records the Hydra config snapshot at launch.
    Returns an ``env_cfg_override`` dict, or None if no run_config exists.
    """
    run_dir = Path(checkpoint_path).parent
    rc = run_dir / "run_config.json"
    if not rc.exists():
        return None
    import json

    data = json.loads(rc.read_text())
    cfg = data.get("config", {})
    env_cfg = cfg.get("env", {})
    reward_cfg = cfg.get("reward")
    if not env_cfg or not reward_cfg:
        return None
    overrides = {"reward_config": reward_cfg, "domain_rand": env_cfg.get("domain_rand", {})}
    for key in ("control_config", "commands", "curriculum", "vmc", "post_step_forward_sensor"):
        if key in env_cfg:
            overrides[key] = env_cfg[key]
    return overrides


def config_env_overrides(config_path: str) -> dict:
    """Build evaluation overrides from a current Hydra task YAML."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    env_cfg = cfg.get("env", {})
    overrides = {
        "reward_config": cfg.get("reward", {}),
        "domain_rand": env_cfg.get("domain_rand", {}),
    }
    for key in ("control_config", "commands", "curriculum", "vmc", "post_step_forward_sensor"):
        if key in env_cfg:
            overrides[key] = env_cfg[key]
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Registered env name")
    parser.add_argument("--checkpoint", required=True, help="Path to model_*.pt")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--jump_every", type=int, default=200, help="trigger pulse every N steps")
    parser.add_argument("--hidden", default="512,512,256,128")
    args = parser.parse_args()

    hidden = [int(x) for x in args.hidden.split(",")]

    from unilab.base import registry
    from unilab.training import ensure_registries

    ensure_registries()
    # Prefer the training snapshot so evaluation matches training exactly.
    trained_ov = trained_env_overrides(args.checkpoint)
    if trained_ov is not None:
        ctrl = trained_ov.get("control_config", {})
        # The trained snapshot already carries clip_actions; only fall back to
        # per-task defaults when no snapshot exists.
    else:
        # VMC tasks use their own apply_action (clip_actions from env.vmc); joint-space
        # tasks (pure PPO / SRL) train with clip_actions=100 and must NOT be clipped.
        vmc_task = "VMC" in args.task
        ctrl = (
            {"action_scale": 0.6, "wheel_action_scale": 10.0, "clip_actions": 1.0}
            if vmc_task
            else {"action_scale": 0.6, "wheel_action_scale": 10.0, "clip_actions": 100.0}
        )
        trained_ov = {
            "reward_config": _REWARD,
            "domain_rand": _DR,
            "control_config": ctrl,
        }
    env = registry.make(
        args.task,
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override=trained_ov,
    )
    try:
        obs_dim = env.obs_groups_spec["obs"]
        actor = load_actor(args.checkpoint, obs_dim, 8, hidden)
        env.init_state()  # initialise NpEnvState and perform the first reset

        max_base_z = 0.0
        base_z_log: list[float] = []  # (trigger, base_z) samples for standing estimate
        air_steps = 0
        survived_steps = 0
        terminated = False

        with torch.no_grad():
            for step in range(args.steps):
                # Pulse jump trigger (half the window on, matching the eval runner).
                trigger = 1.0 if (step % args.jump_every) < (args.jump_every // 2) else 0.0
                # Set the command before the step so the FSM/reward see it.
                env.state.info["commands"][:, 4] = trigger
                obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                # Do NOT clip: the env's apply_action handles scaling/clipping.
                action = actor(obs).numpy()
                state = env.step(action)
                base_z = float(np.asarray(env._backend.get_base_pos())[0, 2])
                max_base_z = max(max_base_z, base_z)
                wheel_contact = state.info.get("wheel_contact", np.zeros((1, 2)))
                # Standing height = stable base_z while the trigger is OFF and
                # wheels are grounded (step-0 may be mid-air / hover, so it is
                # NOT a valid standing baseline).
                if trigger == 0.0 and np.mean(wheel_contact) > 0.99:
                    base_z_log.append(base_z)
                air_steps += int(1.0 - np.mean(wheel_contact))
                if state.terminated[0]:
                    terminated = True
                    break
                survived_steps += 1

        standing_z = float(np.median(base_z_log)) if base_z_log else 0.0
        jump_height = max_base_z - standing_z
        print(
            f"task={args.task} survived={survived_steps}/{args.steps} "
            f"terminated={terminated} max_base_z={max_base_z:.3f} "
            f"standing_z={standing_z:.3f} jump_height={max(jump_height, 0.0):.3f} "
            f"air_frac={air_steps / max(survived_steps, 1):.2f}"
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
