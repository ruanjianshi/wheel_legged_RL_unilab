#!/usr/bin/env python3
"""Analyze a jump-cycle CSV (dump_jump_cycle.py) against §7.5 stage-chain.

Segments the cycle into the five phases and prints the posture at each:
  crouch : base_z drops, knees bend, hip_pitch forward
  launch : knees extend rapidly, base_z rising, wheels still grounded
  air    : wheels off ground, base_z near peak
  land   : wheels re-contact, knees absorb
  recover: upright standing near target height

Also audits wheel-speed matching: at the land step the wheel angular velocity
should be near the ground speed (standing jump -> ~0), not spinning/slipping.

Usage:
    uv run tools/xqrobotwl/analyze_jump_cycle.py --csv logs/pose_data/xxx.csv \
        [--standing 0.55] [--wheel_radius 0.11]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

# Joint-index columns in dump_jump_cycle.csv
COL = {
    "base_z": 2,
    "Lhip": 3, "Lpitch": 4, "Lknee": 5,
    "Rhip": 6, "Rpitch": 7, "Rknee": 8,
    "Lw": 9, "Rw": 10,
    "up_z": 20, "contL": 21, "contR": 22, "phase": 23,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--standing", type=float, default=0.55)
    p.add_argument("--wheel_radius", type=float, default=0.11)
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        print("empty csv")
        return 1
    step = [int(r["step"]) for r in rows]
    trig = [float(r["trigger"]) for r in rows]
    z = [float(r["base_z"]) for r in rows]
    Lk = [float(r["L_knee"]) for r in rows]
    Rk = [float(r["R_knee"]) for r in rows]
    Lp = [float(r["L_hip_pitch"]) for r in rows]
    Rp = [float(r["R_hip_pitch"]) for r in rows]
    up = [float(r["up_z"]) for r in rows]
    contL = [float(r["wheel_contact_L"]) for r in rows]
    Lw = [float(r["L_wheel_rads"]) for r in rows]
    Rw = [float(r["R_wheel_rads"]) for r in rows]
    n = len(rows)

    def pose_at(i):
        if i is None:
            return None
        return (
            f"z={z[i]:.3f} up={up[i]:.3f} "
            f"knee=[{Lk[i]:+.2f},{Rk[i]:+.2f}] hip=[{Lp[i]:+.2f},{Rp[i]:+.2f}] "
            f"wrads=[{Lw[i]:+.1f},{Rw[i]:+.1f}] cont=[{contL[i]:.0f}]"
        )

    on = [i for i in range(n) if trig[i] > 0.5]
    if not on:
        print("no trigger-on steps")
        return 1
    first_on, last_on = on[0], on[-1]

    # crouch: deepest grounded base_z inside the trigger window
    crouch_i = None
    crouch_z = 99.0
    for i in on:
        if contL[i] > 0.5 and z[i] < crouch_z:
            crouch_z = z[i]
            crouch_i = i

    # launch: the step before wheels leave ground (last grounded step rising)
    launch_i = None
    for i in on:
        if contL[i] > 0.5:
            launch_i = i  # keep updating to last grounded step
        else:
            break
    # but launch is the grounded step with max z before air
    air_start = None
    for i in on:
        if contL[i] < 0.5:
            air_start = i
            break
    # land: first grounded step after air
    land_i = None
    seen_air = False
    for i in range(on[0], n):
        if contL[i] < 0.5:
            seen_air = True
        elif seen_air:
            land_i = i
            break
    # recover: after land, find a step with standing posture
    recover_i = None
    if land_i is not None:
        for i in range(land_i, min(land_i + 80, n)):
            if (abs(z[i] - args.standing) < 0.12) and up[i] > 0.85 and contL[i] > 0.5:
                recover_i = i
                break

    max_i = int(np.argmax(z))
    air_steps = sum(1 for i in range(n) if contL[i] < 0.5)
    max_h = max(z) - min(z)

    print(f"rows={n} max_base_z={max(z):.3f} min_base_z={min(z):.3f} max_rise={max_h:.3f} air_steps={air_steps}")
    print(f"  crouch  @step {step[crouch_i] if crouch_i is not None else 'NA'}: {pose_at(crouch_i)}")
    print(f"  launch  @step {step[air_start] if air_start is not None else 'NA'}: {pose_at(air_start)}")
    print(f"  peak    @step {step[max_i]}: {pose_at(max_i)}")
    print(f"  land    @step {step[land_i] if land_i is not None else 'NA'}: {pose_at(land_i)}")
    print(f"  recover @step {step[recover_i] if recover_i is not None else 'NA'}: {pose_at(recover_i)}")

    # Wheel speed matching at landing (standing jump -> wheels should be ~0 rad/s)
    if land_i is not None:
        pre = max(0, land_i - 3)
        avg_w = (sum(abs(Lw[pre:land_i + 1])) + sum(abs(Rw[pre:land_i + 1]))) / (2 * (land_i - pre + 1))
        slip = abs(Lw[land_i]) > 5.0 or abs(Rw[land_i]) > 5.0
        print(f"  wheel@land: L={Lw[land_i]:+.1f} R={Rw[land_i]:+.1f} rad/s (avg|w| last4={avg_w:.1f}) slip={slip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
