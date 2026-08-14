#!/usr/bin/env python3
"""LQR 平衡控制器 交互回放 CLI (独立任务轨).

用法:
  uv run mjpython scripts/classic_control/lqr/play_lqr.py                 # 平地
  uv run mjpython scripts/classic_control/lqr/play_lqr.py --terrain rough # 粗糙地形
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import argparse

from scripts.classic_control.common.play import run_play
from scripts.classic_control.lqr.controller import LqrController


def main() -> int:
    ap = argparse.ArgumentParser(description="LQR 平衡交互控制")
    ap.add_argument("--terrain", choices=["flat", "rough"], default="flat")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()
    return run_play(LqrController, "LQR", terrain=args.terrain, speed=args.speed)


if __name__ == "__main__":
    sys.exit(main())
