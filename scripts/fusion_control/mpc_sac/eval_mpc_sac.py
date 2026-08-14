"""MPC×SAC 融合控制评估 — 批量 episodes, 融合指标 + 存活率 + 可选渲染.

用法:
  uv run mjpython scripts/fusion_control/mpc_sac/eval_mpc_sac.py --task walk_flat --episodes 5 \
      --checkpoint logs/fusion_control/mpc_sac/walk_flat/<run>/model_final.pt \
      --out logs/fusion_control/mpc_sac/report_f1.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from scripts.classic_control.common import metrics as common_metrics
from scripts.classic_control.common import render
from scripts.fusion_control.mpc_sac.config import build_config
from scripts.fusion_control.mpc_sac.controller import MpcSacController
from scripts.fusion_control.mpc_sac.env import build_env
from scripts.fusion_control.mpc_sac.metrics import compute
from scripts.fusion_control.mpc_sac.policy_loader import get_device, load_policy
from scripts.fusion_control.mpc_sac.runner import make_schedule, run_episode

DEFAULT_SIM_TIME = {"walk_flat": 35.0, "walk_rough": 30.0}


def main() -> int:
    ap = argparse.ArgumentParser(description="MPC×SAC 融合控制评估")
    ap.add_argument("--task", choices=["walk_flat", "walk_rough"], default="walk_flat")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--sim_time", type=float, default=None)
    ap.add_argument("--checkpoint", type=str, required=True, help="高层 SAC ckpt (model_final.pt)")
    ap.add_argument("--out", type=str, default=None, help="report md 路径")
    ap.add_argument("--render", type=str, default=None, help="渲染最佳 episode 视频路径")
    ap.add_argument("--cmd", type=str, default=None, help="单条命令 'vx=0.4,vyaw=0.3'")
    ap.add_argument("--device", type=str, default="auto")
    args = ap.parse_args()

    task_key = args.task
    cfg, merged = build_config(task_key)
    device = get_device(args.device)
    sim_time = args.sim_time or DEFAULT_SIM_TIME[task_key]
    phase = int(cfg.phase_flat if task_key == "walk_flat" else cfg.phase_rough)

    policy = lambda: load_policy(cfg, task_key, args.checkpoint, device=device)  # noqa: E731
    env = build_env(task_key, num_envs=1, cmd=None, lock_hip_roll=True)
    ascale = float(env._cfg.control_config.action_scale)
    wscale = float(env._cfg.control_config.wheel_action_scale)
    dt = float(env._cfg.ctrl_dt)
    ctrl = MpcSacController(
        task_key,
        params=merged,
        action_scale=ascale,
        wheel_action_scale=wscale,
        dt=dt,
        policy_factory=policy,
    )
    schedule = make_schedule(task_key, args.cmd)

    all_metrics: list[dict] = []
    records: list[list[dict]] = []
    for ep in range(args.episodes):
        ctrl.reset()
        rec, stats = run_episode(env, ctrl, schedule, sim_time, dt)
        m = compute(rec, phase, cfg)
        m["ep"] = ep + 1
        all_metrics.append(m)
        records.append(rec)

    # 聚合
    keys = [
        "ep_len_s",
        "stand_hold_max_s",
        "gyro_rms",
        "linvel_xy_mean",
        "base_z_mean",
        "vx_rmse",
        "height_err_mean",
        "theta_max_viol",
        "v_max_viol",
        "cmd_track_err",
        "solve_ms_mean",
    ]
    agg: dict[str, float] = {
        "survival_rate": float(np.mean([m["survived"] for m in all_metrics])),
        "ep_len_s_mean": float(np.mean([m["ep_len_s"] for m in all_metrics])),
    }
    for k in keys:
        if k in all_metrics[0]:
            agg[f"{k}_mean"] = float(np.mean([m.get(k, 0.0) for m in all_metrics]))

    print(f"\n=== MPC×SAC 融合评估 [{task_key}] {args.episodes} ep ===")
    print(f"存活率: {agg['survival_rate']*100:.0f}%  平均时长: {agg['ep_len_s_mean']:.1f}s")
    for k in keys:
        if f"{k}_mean" in agg:
            print(f"  {k}: {agg[f'{k}_mean']:.4f}")
    # 阈值对照
    th = common_metrics.threshold_for(phase)
    for name, (lim, direction, unit) in th.items():
        val = agg.get(f"{name}_mean")
        if val is None:
            continue
        ok = val < lim if direction == "<" else val > lim
        print(f"  {name}: {val:.3f} {unit} [{'✅' if ok else '❌'} 阈 {direction}{lim}]")

    if args.render and records:
        best = max(records, key=lambda r: (r[-1]["t"] if r else 0.0))
        model_file = env._cfg.scene.model_file
        Path(args.render).parent.mkdir(parents=True, exist_ok=True)
        render.states_to_video(best, model_file, args.render)
        print(f"渲染 → {args.render}")

    if args.out:
        lines = [
            f"# MPC×SAC 融合评估 [{task_key}] ({args.episodes} ep, ckpt={Path(args.checkpoint).name})",
            "",
            "| 指标 | 值 | 阈值 |",
            "|---|---|---|",
            f"| 存活率 | {agg['survival_rate']*100:.0f}% | ≥95% |",
        ]
        for k in keys:
            if f"{k}_mean" in agg:
                lines.append(f"| {k} | {agg[f'{k}_mean']:.4f} | |")
        for name, (lim, direction, unit) in th.items():
            val = agg.get(f"{name}_mean")
            if val is not None:
                lines.append(f"| {name} | {val:.4f} {unit} | {direction}{lim} |")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"报告 → {args.out}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
