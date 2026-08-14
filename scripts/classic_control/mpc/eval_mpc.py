#!/usr/bin/env python3
"""MPC 批量评估 CLI (独立任务轨).

用法:
  uv run python scripts/classic_control/mpc/eval_mpc.py --phases 1 2 3 4 --episodes 5 --out logs/classic/report_mpc.md
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import argparse

from scripts.classic_control.common.eval import run_eval
from scripts.classic_control.mpc.controller import MpcController


def main() -> int:
    ap = argparse.ArgumentParser(description="MPC 批量评估")
    ap.add_argument("--phases", type=int, nargs="+", default=[1])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--mpc_horizon", type=int, default=20)
    args = ap.parse_args()
    run_eval(
        MpcController,
        args.phases,
        args.episodes,
        out=args.out,
        params={"mpc_horizon": args.mpc_horizon},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
