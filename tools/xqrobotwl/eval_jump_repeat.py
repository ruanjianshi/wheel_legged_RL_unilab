#!/usr/bin/env python3
"""Repeated-jump evaluation for §7.5 平地跳跃 success rate.

For each episode the robot is triggered to jump ``--jumps`` times (trigger ON
``--on`` steps, then OFF ``--off`` steps for landing recovery).  Per jump we
record:

  standing_z   median base_z just before the trigger turns ON (settle window)
  max_z        peak base_z during the trigger-ON window
  airborne     both wheels left the ground at any point in the trigger window
  air_steps    airborne steps in the trigger window
  spin_air     peak wheel |dof_vel| (rad/s) while airborne
  recovered    reached a stable standing pose inside the trigger-OFF window:
               both wheels contact AND 0.38 <= base_z <= 0.70 AND up_z > 0.8,
               held for ``--hold`` consecutive steps
  drift        horizontal displacement during the trigger-OFF window (m)

Success rate = recovered / jump_attempts (jump_attempts = episodes x jumps).

Usage:
  uv run tools/xqrobotwl/eval_jump_repeat.py --task XqRobotWLJumpSRLFlat \
      --checkpoint logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat/<run>/model_9999.pt \
      --jumps 10 --episodes 2
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

from verify_jump import load_actor, trained_env_overrides  # noqa: E402


def _recovered(
    wc: np.ndarray,
    base_z: float,
    up_z: float,
) -> bool:
    both = bool(np.min(wc) > 0.5)
    z_ok = 0.38 <= base_z <= 0.70
    up_ok = up_z > 0.80
    return both and z_ok and up_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jumps", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--settle", type=int, default=50, help="trigger-off settle steps before 1st jump")
    ap.add_argument("--on", type=int, default=100, help="trigger-ON steps per jump")
    ap.add_argument("--off", type=int, default=120, help="trigger-OFF (recovery) steps per jump")
    ap.add_argument("--hold", type=int, default=10, help="consecutive steps that define stable standing")
    ap.add_argument("--hidden", default="512,512,256,128")
    args = ap.parse_args()
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
    env = registry.make(args.task, num_envs=1, sim_backend="mujoco", env_cfg_override=ov)
    try:
        obs_dim = env.obs_groups_spec["obs"]
        actor = load_actor(args.checkpoint, obs_dim, 8, hidden)
        env.init_state()

        attempts = args.episodes * args.jumps
        results = []  # one dict per jump
        with torch.no_grad():
            for ep in range(args.episodes):
                env.reset(np.arange(1, dtype=np.int32))
                for j in range(args.jumps):
                    standing_log = []
                    # --- settle: trigger OFF, record standing height ---
                    for _ in range(args.settle):
                        env.state.info["commands"][:, 4] = 0.0
                        obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                        action = actor(obs).numpy()
                        st = env.step(action)
                        z = float(np.asarray(env._backend.get_base_pos())[0, 2])
                        wc = st.info.get("wheel_contact", np.zeros((1, 2)))
                        if np.mean(wc) > 0.99:
                            standing_log.append(z)
                        if st.terminated[0]:
                            break
                    if st.terminated[0]:
                        results.append(
                            {"jump": f"ep{ep}.{j}", "recovered": False, "airborne": False,
                             "terminated": True, "max_z": 0.0, "stand": 0.0,
                             "air_steps": 0, "spin_air": 0.0, "drift": 0.0}
                        )
                        continue
                    standing_z = float(np.median(standing_log)) if standing_log else 0.0

                    x0, y0 = env._backend.get_base_pos()[0, :2]
                    max_z = 0.0
                    air_steps = 0
                    spin_air = 0.0
                    airborne = False
                    # --- trigger ON: jump window ---
                    for _ in range(args.on):
                        env.state.info["commands"][:, 4] = 1.0
                        obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                        action = actor(obs).numpy()
                        st = env.step(action)
                        z = float(np.asarray(env._backend.get_base_pos())[0, 2])
                        max_z = max(max_z, z)
                        wc = st.info.get("wheel_contact", np.zeros((1, 2)))
                        air = 1.0 - float(np.mean(wc))
                        if air > 0.5:
                            airborne = True
                            air_steps += 1
                            wv = np.asarray(env.get_dof_vel())[0, 6:8]
                            spin_air = max(spin_air, float(np.abs(wv).max()))
                        if st.terminated[0]:
                            break
                    if st.terminated[0]:
                        results.append(
                            {"jump": f"ep{ep}.{j}", "recovered": False, "airborne": airborne,
                             "terminated": True, "max_z": max_z, "stand": standing_z,
                             "air_steps": air_steps, "spin_air": spin_air, "drift": 0.0}
                        )
                        continue

                    # --- trigger OFF: landing + recovery window ---
                    recovered = False
                    drift = 0.0
                    streak = 0
                    # drift is measured once stable standing is reached, else at window end
                    for s in range(args.off):
                        env.state.info["commands"][:, 4] = 0.0
                        obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                        action = actor(obs).numpy()
                        st = env.step(action)
                        if st.terminated[0]:
                            break
                        z = float(np.asarray(env._backend.get_base_pos())[0, 2])
                        wc = st.info.get("wheel_contact", np.zeros((1, 2)))
                        up = np.asarray(env._backend.get_sensor_data("upvector"), dtype=np.float64)[0]
                        up_z = float(up[2])
                        if _recovered(wc, z, up_z):
                            streak += 1
                            if streak >= args.hold:
                                recovered = True
                                x1, y1 = env._backend.get_base_pos()[0, :2]
                                drift = float(np.hypot(x1 - x0, y1 - y0))
                                # consume the rest of the window
                                for _ in range(args.off - s - 1):
                                    env.state.info["commands"][:, 4] = 0.0
                                    obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                                    action = actor(obs).numpy()
                                    if env.step(action).terminated[0]:
                                        break
                                break
                        else:
                            streak = 0
                    if not recovered:
                        x1, y1 = env._backend.get_base_pos()[0, :2]
                        drift = float(np.hypot(x1 - x0, y1 - y0))

                    results.append(
                        {"jump": f"ep{ep}.{j}", "recovered": recovered,
                         "airborne": airborne, "terminated": bool(st.terminated[0]),
                         "max_z": max_z, "stand": standing_z,
                         "air_steps": air_steps, "spin_air": spin_air, "drift": drift}
                    )

        n = len(results)
        n_air = sum(1 for r in results if r["airborne"])
        n_rec = sum(1 for r in results if r["recovered"])
        n_term = sum(1 for r in results if r["terminated"])
        heights = [max(r["max_z"] - r["stand"], 0.0) for r in results if r["airborne"]]
        drifts = [r["drift"] for r in results if r["airborne"] and not r["terminated"]]
        spins = [r["spin_air"] for r in results if r["airborne"]]
        print(f"\n=== 重复跳跃评估 {args.task} ===")
        print(f"checkpoint={args.checkpoint}")
        print(f"attempts={n}  airborne(真跳)={n_air}/{n}  recovered(恢复站立)={n_rec}/{n}  "
              f"terminated={n_term}/{n}")
        print(f"成功恢复率 = {n_rec / max(n, 1):.2f}  (空中跳恢复率 = "
              f"{sum(1 for r in results if r['airborne'] and r['recovered']) / max(n_air, 1):.2f})")
        if heights:
            print(f"跳高(max_z-stand) = {np.mean(heights):.3f} ± {np.std(heights):.3f} m  "
                  f"(min {np.min(heights):.3f}, max {np.max(heights):.3f})")
        if drifts:
            print(f"恢复窗口漂移 = {np.mean(drifts):.3f} ± {np.std(drifts):.3f} m")
        if spins:
            print(f"空中轮速峰值 = {np.max(spins):.1f} rad/s (≈ {np.max(spins) * 0.065:.3f} m/s)")
        print("\n逐跳明细:")
        print(f"{'jump':<10}{'air':<5}{'rec':<5}{'term':<5}{'stand':>7}{'maxz':>7}{'height':>8}"
              f"{'air_st':>7}{'spin':>7}{'drift':>7}")
        for r in results:
            h = max(r["max_z"] - r["stand"], 0.0)
            print(f"{r['jump']:<10}{str(r['airborne']):<5}{str(r['recovered']):<5}"
                  f"{str(r['terminated']):<5}{r['stand']:>7.3f}{r['max_z']:>7.3f}{h:>8.3f}"
                  f"{r['air_steps']:>7d}{r['spin_air']:>7.1f}{r['drift']:>7.3f}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
