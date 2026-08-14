#!/usr/bin/env python3
"""MPC 经典平衡控制器 CLI (独立任务轨).

用法:
  uv run mjpython scripts/classic_control/balance_mpc.py --phase 1 --sim_time 15 --mpc_horizon 20 --render video/classic/mpc_p1_balance.mp4
  uv run mjpython scripts/classic_control/balance_mpc.py --phase 2 --cmd "vx=0.4,vyaw=0.3" --sim_time 30 --render video/classic/mpc_p2_turn.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    ap = argparse.ArgumentParser(description="MPC 平衡控制器 (P1 平衡/P2 指令/P3 腿长/P4 地形)")
    ap.add_argument("--phase", type=int, choices=[1, 2, 3, 4], default=1)
    ap.add_argument("--sim_time", type=float, default=None)
    ap.add_argument("--cmd", type=str, default=None, help='"vx=0.4,vyaw=0.3"')
    ap.add_argument("--render", type=str, default=None)
    ap.add_argument("--report", type=str, default=None)
    ap.add_argument("--mpc_horizon", type=int, default=20)
    ap.add_argument("--u_max", type=float, default=None)
    ap.add_argument("--theta_max", type=float, default=None)
    ap.add_argument("--q_theta", type=float, default=None)
    ap.add_argument("--k_yaw", type=float, default=None)
    ap.add_argument("--sign", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None, help="模型 α (默认 25)")
    ap.add_argument("--beta", type=float, default=None, help="模型 β (默认 -2.5)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    params: dict = {"mpc_horizon": args.mpc_horizon}
    for k in ("u_max", "theta_max", "q_theta", "k_yaw", "sign", "alpha", "beta"):
        v = getattr(args, k)
        if v is not None:
            params[k] = v

    from scripts.classic_control.run import run_phase

    run_phase(
        "mpc",
        args.phase,
        sim_time=args.sim_time,
        cmd_override=args.cmd,
        render_path=args.render,
        report_path=args.report,
        mpc_horizon=args.mpc_horizon,
        params=params,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
