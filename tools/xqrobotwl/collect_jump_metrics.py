#!/usr/bin/env python3
"""Collect final-model jump metrics across N repeated evals → JSON.

Reuses eval_jump_detail's evaluation loop (deterministic env, but initial
condition varies run-to-run), returns per-metric lists for mean±std bars.

Usage:
    uv run tools/xqrobotwl/collect_jump_metrics.py --n 5 --out logs/pose_data/jump_final_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.xqrobotwl.verify_jump import load_actor, trained_env_overrides  # noqa: E402

ALGOS = {
    "PPO": ("XqRobotWLJumpFlat",
            "logs/rsl_rl_ppo/XqRobotWLJumpFlat/2026-08-16_01-53-39_mujoco/model_9999.pt"),
    "PPO+VMC": ("XqRobotWLJumpVMC",
                "logs/rsl_rl_ppo/XqRobotWLJumpVMC/2026-08-16_01-53-43_mujoco/model_1000.pt"),
    "SRL": ("XqRobotWLJumpSRLFlat",
            "logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat/2026-08-16_13-36-25_mujoco/model_3999.pt"),
    "SRL+VMC": ("XqRobotWLJumpSRLVMC",
                "logs/rsl_rl_ppo/XqRobotWLJumpSRLVMC/2026-08-16_14-08-53_mujoco/model_3999.pt"),
}

SETTLE, PULSE, TAIL = 100, 160, 200


def _euler(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def one_episode(task, ckpt, hidden):
    from unilab.base import registry
    from unilab.training import ensure_registries

    ensure_registries()
    ov = trained_env_overrides(ckpt)
    env = registry.make(task, num_envs=1, sim_backend="mujoco", env_cfg_override=ov)
    try:
        actor = load_actor(ckpt, env.obs_groups_spec["obs"], 8, hidden)
        env.init_state()
        total = SETTLE + PULSE + TAIL
        recs = []
        with torch.no_grad():
            for step in range(total):
                trig = 1.0 if SETTLE <= step < SETTLE + PULSE else 0.0
                env.state.info["commands"][:, 4] = trig
                obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                action = actor(obs).numpy()
                st = env.step(action)
                bp = np.asarray(env._backend.get_base_pos())[0]
                lz = float(np.asarray(env._backend.get_sensor_data("left_wheel_world_pos")).reshape(-1, 3)[0, 2])
                gyro = np.asarray(env._backend.get_sensor_data("gyro")).reshape(-1, 3)[0]
                recs.append((step, trig, float(bp[2]), float(np.linalg.norm(gyro)), 1.0 if lz < 0.13 else 0.0))
                if st.terminated[0]:
                    break
        n = len(recs)
        # jump height: peak z during trigger - standing z
        standing_z = np.median([r[2] for r in recs[:SETTLE] if r[4] > 0.5]) if n > SETTLE else 0.52
        on = [r for r in recs if r[1] > 0.5]
        peak = max(r[2] for r in on) if on else standing_z
        jump_height = peak - standing_z
        # standing gyro
        pre = [r[3] for r in recs[:SETTLE] if r[4] > 0.5]
        stand_gyro = float(np.mean(pre)) if pre else float("nan")
        # air steps
        air = [r for r in on if r[4] < 0.5]
        air_steps = len(air)
        # recovery: find first post-pulse step that is standing near target
        post = recs[SETTLE + PULSE:] if n > SETTLE + PULSE else []
        rec_i = None
        for i, r in enumerate(post):
            if abs(r[2] - 0.52) < 0.12 and r[4] > 0.5:
                rec_i = i
                break
        recover_steps = rec_i if rec_i is not None else -1
        recover_gyro = float(np.mean([r[3] for r in post])) if post else float("nan")
        return dict(jump_height=jump_height, stand_gyro=stand_gyro,
                    recover_gyro=recover_gyro, air_steps=air_steps,
                    recover_steps=recover_steps, stand_z=standing_z)
    finally:
        env.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--out", default="logs/pose_data/jump_final_metrics.json")
    p.add_argument("--hidden", default="512,512,256,128")
    p.add_argument("--algos", default="PPO,PPO+VMC,SRL,SRL+VMC")
    args = p.parse_args()
    hidden = [int(x) for x in args.hidden.split(",")]
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {}
    for algo in args.algos.split(","):
        task, ckpt = ALGOS[algo]
        print(f"[{algo}] collecting {args.n} episodes...", flush=True)
        ep = []
        for i in range(args.n):
            m = one_episode(task, ckpt, hidden)
            ep.append(m)
            print(f"  ep{i+1}: jump={m['jump_height']:.3f} stand_gyro={m['stand_gyro']:.2f} "
                  f"rec_gyro={m['recover_gyro']:.2f} air={m['air_steps']}", flush=True)
        result[algo] = ep
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
