#!/usr/bin/env python3
"""LQR 经典平衡控制器 CLI (独立任务轨).

用法:
  uv run python scripts/classic_control/lqr/balance_lqr.py --phase 1 --sim_time 15 --render video/lqr/lqr_p1_balance.mp4
  uv run python scripts/classic_control/lqr/balance_lqr.py --phase 2 --cmd "vx=0.5,vyaw=0.3" --sim_time 30
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import argparse

from scripts.classic_control.common.run import run_phase
from scripts.classic_control.lqr.controller import LqrController


def main() -> int:
    ap = argparse.ArgumentParser(description="LQR 平衡控制器 (P1 平衡/P2 指令/P3 腿长/P4 地形)")
    ap.add_argument("--phase", type=int, choices=[1, 2, 3, 4], default=1)
    ap.add_argument("--sim_time", type=float, default=None)
    ap.add_argument("--cmd", type=str, default=None, help='"vx=0.4,vyaw=0.3"')
    ap.add_argument("--render", type=str, default=None)
    ap.add_argument("--report", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_phase(
        LqrController,
        args.phase,
        sim_time=args.sim_time,
        cmd_override=args.cmd,
        render_path=args.render,
        report_path=args.report,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
