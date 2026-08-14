#!/usr/bin/env python3
"""批量评估 CLI: 多 episode 重复跑各阶段, 统计存活率/指标 → markdown 报告.

用法:
  uv run mjpython scripts/classic_control/eval_classic.py --controller lqr --phases 1 2 3 4 --episodes 10 --out logs/classic/report_lqr.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> int:
    ap = argparse.ArgumentParser(description="经典控制器批量评估")
    ap.add_argument("--controller", choices=["lqr", "mpc"], required=True)
    ap.add_argument("--phases", type=int, nargs="+", default=[1])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", type=str, default=None, help="markdown 报告路径")
    ap.add_argument("--mpc_horizon", type=int, default=20)
    args = ap.parse_args()

    from scripts.classic_control import metrics as metrics_mod
    from scripts.classic_control import report as report_mod
    from scripts.classic_control import rollout as rollout_mod
    from scripts.classic_control.run import get_dynamics, make_schedule

    A_d, B_d = get_dynamics()
    phase_survival: dict[int, list[float]] = {}

    for phase in args.phases:
        task_key = "walk_rough" if phase == 4 else "walk_flat"
        env = rollout_mod.build_env(task_key=task_key, num_envs=1)
        from scripts.classic_control.controller import BalanceController

        controller = BalanceController(
            args.controller,
            phase,
            A_d,
            B_d,
            params={"mpc_horizon": args.mpc_horizon} if args.controller == "mpc" else None,
            action_scale=float(env._cfg.control_config.action_scale),
            wheel_action_scale=float(env._cfg.control_config.wheel_action_scale),
            dt=float(env._cfg.ctrl_dt),
        )
        schedule = make_schedule(phase, None)
        dt = float(env._cfg.ctrl_dt)
        sim_time = {1: 12.0, 2: 30.0, 3: 36.0, 4: 25.0}[phase]
        survivals = []
        all_metrics: list[dict] = []
        for ep in range(args.episodes):
            controller.reset()  # ★ 控制器内部状态 (xpos/z_int/平滑/高度积分) 每 ep 复位
            rec, stats = rollout_mod.run_episode(env, controller, schedule, sim_time, dt)
            m = metrics_mod.compute(rec, phase)
            all_metrics.append(m)
            survivals.append(float(m.get("survived", 0.0)))
        phase_survival[phase] = survivals
        # 汇总
        agg: dict[str, float] = {}
        keys = list(all_metrics[0].keys()) if all_metrics else []
        for k in keys:
            vals = [m[k] for m in all_metrics]
            agg[k] = float(np.mean(vals)) if k != "survived" else float(np.mean(vals))
        agg["survival_rate"] = float(np.mean(survivals))
        tag = f"{args.controller.upper()} P{phase} ({args.episodes} ep)"
        report_mod.print_metrics(tag, agg, phase)
        if args.out:
            report_mod.write_report(args.out, f"{tag} — 汇总", agg, phase)
        env.close()

    # 总存活率表
    print("\n=== 存活率汇总 ===")
    for phase, sv in phase_survival.items():
        print(f"  P{phase}: {np.mean(sv) * 100:.0f}% ({int(np.sum(sv))}/{len(sv)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
