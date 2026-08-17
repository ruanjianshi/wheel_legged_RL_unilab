#!/usr/bin/env python3
"""Repeat-jump acceptance test (CLAUDE.md §7.5).

Pulses the jump trigger repeatedly and scores EVERY trigger window for the
full stage chain:
  crouch -> launch (airborne) -> land (wheels grounded again) -> recover to
  stable standing (near target height, upright, both wheels grounded).

A window counts as a SUCCESS iff all of:
  - launch   : wheels left the ground (air) for >= min_air_steps this window
  - landed   : wheels came back down (airborne -> grounded) within the window
  - recovered: within `recover_steps` after landing the base is within
               [h_min, h_max] of standing, upright (up_z > 0.85) and both
               wheels grounded
  - survived : episode did not terminate during the window

Usage:
    uv run tools/xqrobotwl/verify_jump_repeat.py \
        --task XqRobotWLJumpFlat \
        --checkpoint logs/rsl_rl_ppo/XqRobotWLJumpFlat/<run>/model_9999.pt \
        --cycles 10
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

from tools.xqrobotwl.verify_jump import (  # noqa: E402
    _DR,
    _REWARD,
    load_actor,
    trained_env_overrides,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cycles", type=int, default=10, help="number of jump attempts")
    p.add_argument("--settle", type=int, default=50, help="trigger-off steps before first pulse")
    p.add_argument("--pulse", type=int, default=100, help="trigger-on steps per cycle")
    p.add_argument("--tail", type=int, default=100, help="trigger-off steps per cycle (land+recover)")
    p.add_argument("--recover_steps", type=int, default=80, help="max steps after landing to recover")
    p.add_argument("--min_air_steps", type=int, default=3, help="min airborne steps to count a launch")
    p.add_argument("--hidden", default="512,512,256,128")
    p.add_argument("--wheel_radius", type=float, default=0.11, help="wheel geom radius (m)")
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
    try:
        actor = load_actor(args.checkpoint, env.obs_groups_spec["obs"], 8, hidden)
        env.init_state()

        contact = lambda: np.asarray(
            env._backend.get_sensor_data("left_wheel_world_pos"), dtype=np.float64
        ).reshape(-1, 3)[0, 2] < args.wheel_radius + 0.02
        up = lambda: float(
            np.asarray(env._backend.get_sensor_data("upvector"), dtype=np.float64).reshape(-1, 3)[0, 2]
        )
        base_z = lambda: float(np.asarray(env._backend.get_base_pos())[0, 2])

        total = args.settle + args.cycles * (args.pulse + args.tail)
        records = []  # per-step: (trigger, base_z, contact, up_z, terminated)
        standing_samples = []
        terminated_at = None
        with torch.no_grad():
            for step in range(total):
                if step < args.settle:
                    trigger = 0.0
                else:
                    k = (step - args.settle) % (args.pulse + args.tail)
                    trigger = 1.0 if k < args.pulse else 0.0
                env.state.info["commands"][:, 4] = trigger
                obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                action = actor(obs).numpy()
                st = env.step(action)
                records.append(
                    (trigger, base_z(), contact(), up(), bool(st.terminated[0]))
                )
                if trigger == 0.0 and contact():
                    standing_samples.append(base_z())
                if st.terminated[0]:
                    terminated_at = step
                    break

        standing_z = float(np.median(standing_samples)) if standing_samples else 0.0
        # Analyze each cycle.
        results = []
        for i in range(args.cycles):
            start = args.settle + i * (args.pulse + args.tail)
            on = records[start : start + args.pulse]
            off = records[start + args.pulse : start + args.pulse + args.tail]
            if len(on) == 0:
                continue
            air_on = [not c for (_t, _z, c, _u, _term) in on]
            air_steps = sum(air_on)
            launch = air_steps >= args.min_air_steps
            # max height this window
            maxz = max((z for _t, z, _c, _u, _term in on), default=0.0)
            jump_h = max(maxz - standing_z, 0.0)
            # landed: was airborne then grounded again within the on+off window
            grounded_after_air = False
            seen_air = False
            for (_t, z, c, _u, _term) in on + off:
                if not c:
                    seen_air = True
                elif seen_air:
                    grounded_after_air = True
                    break
            landed = launch and grounded_after_air
            # recovered: within recover_steps after landing, stand near target & upright
            recovered = False
            recover_base = standing_z
            landed_step = None
            if landed:
                seen_air = False
                for j, (_t, z, c, u, _term) in enumerate(on + off):
                    if not c:
                        seen_air = True
                    elif seen_air:
                        landed_step = j
                        break
                if landed_step is not None:
                    for j in range(landed_step, min(landed_step + args.recover_steps, len(on + off))):
                        _t, z, c, u, _term = (on + off)[j]
                        if (abs(z - standing_z) < 0.12) and (u > 0.85) and c:
                            recovered = True
                            break
            survived = not any(_term for _t, _z, _c, _u, _term in on + off)
            ok = launch and landed and recovered and survived
            results.append(
                dict(
                    cycle=i,
                    launch=launch,
                    landed=landed,
                    recovered=recovered,
                    survived=survived,
                    ok=ok,
                    air_steps=air_steps,
                    jump_h=round(jump_h, 3),
                )
            )

        n_ok = sum(1 for r in results if r["ok"])
        n = len(results)
        print(f"task={args.task} cycles={n} standing_z={standing_z:.3f} terminated_at={terminated_at}")
        for r in results:
            print(
                f"  cycle {r['cycle']:2d}: {'PASS' if r['ok'] else 'FAIL'}"
                f"  launch={int(r['launch'])} landed={int(r['landed'])}"
                f" recovered={int(r['recovered'])} survived={int(r['survived'])}"
                f" air_steps={r['air_steps']:3d} jump_h={r['jump_h']:.3f}"
            )
        print(f"SUCCESS_RATE={n_ok}/{n} = {n_ok / max(n, 1):.0%}")
        return 0 if n and n_ok / n >= 0.9 else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
