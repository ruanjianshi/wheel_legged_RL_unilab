#!/usr/bin/env python3
"""Compare the four jump algorithms across speeds.

Usage:
    uv run scripts/compare_jump.py \
        --algo "XqRobotWLJumpFlat=logs/.../model_9999.pt" \
        --algo "XqRobotWLJumpSRLFlat=logs/.../model_9999.pt" \
        --algo "XqRobotWLJumpVMC=logs/.../model_9999.pt" \
        --algo "XqRobotWLJumpSRLVMC=logs/.../model_9999.pt" \
        --episodes 5 --vx "0,0.3,0.6,1.0"

Metrics per algorithm (averaged over episodes x speeds):
    max_height   max base_z - initial hover (m)
    air_frac     fraction of steps with both wheels off the ground (real jump)
    survival     fraction of episodes that did not terminate
    jump_success fraction of episodes that got airborne AND survived
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


def _load_actor(path: str, obs_dim: int, hidden: list[int]):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    actor_state = ckpt.get("actor_state_dict", ckpt.get("model_state_dict"))
    actor = _ActorMLP(obs_dim, hidden, 8)
    actor.load_state_dict({k: v for k, v in actor_state.items() if k.startswith("mlp.")})
    actor.eval()
    return actor


def eval_algo(task: str, checkpoint: str, vx: float, steps: int = 600) -> dict:
    from unilab.base import registry
    from unilab.training import ensure_registries

    ensure_registries()
    from verify_jump import trained_env_overrides  # same training-snapshot logic

    ov = trained_env_overrides(checkpoint)
    if ov is None:
        vmc_task = "VMC" in task
        ctrl = (
            {"action_scale": 0.6, "wheel_action_scale": 10.0, "clip_actions": 1.0}
            if vmc_task
            else {"action_scale": 0.6, "wheel_action_scale": 10.0, "clip_actions": 100.0}
        )
        ov = {"reward_config": _REWARD, "domain_rand": _DR, "control_config": ctrl}
    env = registry.make(
        task,
        num_envs=1,
        sim_backend="mujoco",
        env_cfg_override=ov,
    )
    try:
        actor = _load_actor(checkpoint, env.obs_groups_spec["obs"], [512, 512, 256, 128])
        env.init_state()
        max_height, air_steps, survived = 0.0, 0, True
        run_steps = 0
        base_z_log = []  # (trigger, z) while grounded with trigger off -> standing
        with torch.no_grad():
            for step in range(steps):
                trigger = 1.0 if (step % 200) < 100 else 0.0
                env.state.info["commands"][:, 0] = vx  # forward speed
                env.state.info["commands"][:, 4] = trigger
                obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                action = actor(obs).numpy()
                state = env.step(action)
                run_steps += 1
                z = float(np.asarray(env._backend.get_base_pos())[0, 2])
                max_height = max(max_height, z)
                wc = state.info.get("wheel_contact", np.zeros((1, 2)))
                if trigger == 0.0 and np.mean(wc) > 0.99:
                    base_z_log.append(z)
                air_steps += int(1.0 - np.mean(wc))
                if state.terminated[0]:
                    survived = False
                    break
        standing = float(np.median(base_z_log)) if base_z_log else 0.0
        return {
            "jump_height": max_height - standing,
            "air_frac": air_steps / max(run_steps, 1),  # over steps actually run
            "survived": survived,
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", action="append", required=True, help="task=checkpoint")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--vx", default="0,0.3,0.6,1.0")
    args = parser.parse_args()

    vxs = [float(x) for x in args.vx.split(",")]
    rows = []
    for spec in args.algo:
        task, ckpt = spec.split("=", 1)
        heights, airs, surv = [], [], []
        for vx in vxs:
            for _ in range(args.episodes):
                r = eval_algo(task, ckpt, vx)
                heights.append(r["jump_height"])
                airs.append(r["air_frac"])
                surv.append(r["survived"])
        jumped = sum(1 for a in airs if a > 0.03)
        rows.append(
            {
                "algo": task,
                "avg_jump_height": float(np.mean(heights)),
                "avg_air_frac": float(np.mean(airs)),
                "survival": float(np.mean(surv)),
                "jump_success": jumped / len(airs),
            }
        )

    print("\n=== 四算法跳跃对比 ===")
    print(f"{'算法':<24}{'跳高(m)':>10}{'air(真跳)':>10}{'存活率':>10}{'跳跃成功率':>10}")
    for r in sorted(rows, key=lambda x: -x["avg_air_frac"]):
        print(
            f"{r['algo']:<24}{r['avg_jump_height']:>10.3f}"
            f"{r['avg_air_frac']:>10.3f}{r['survival']:>10.2f}{r['jump_success']:>10.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
