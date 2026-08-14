"""MPC×SAC 融合控制单次运行 + 渲染.

用法:
  uv run mjpython scripts/fusion_control/mpc_sac/balance_mpc_sac.py --task walk_flat \
      --cmd "vx=0.4" --checkpoint logs/fusion_control/mpc_sac/walk_flat/<run>/model_final.pt \
      --render video/fusion_control/mpc_sac/f1_fusion_vx04.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.classic_control.common import render
from scripts.fusion_control.mpc_sac.config import build_config
from scripts.fusion_control.mpc_sac.controller import MpcSacController
from scripts.fusion_control.mpc_sac.env import build_env
from scripts.fusion_control.mpc_sac.metrics import compute
from scripts.fusion_control.mpc_sac.policy_loader import get_device, load_policy
from scripts.fusion_control.mpc_sac.runner import make_schedule, run_episode

DEFAULT_SIM_TIME = {"walk_flat": 35.0, "walk_rough": 30.0}


def main() -> int:
    ap = argparse.ArgumentParser(description="MPC×SAC 融合控制单次运行")
    ap.add_argument("--task", choices=["walk_flat", "walk_rough"], default="walk_flat")
    ap.add_argument("--sim_time", type=float, default=None)
    ap.add_argument("--cmd", type=str, default=None, help="单条命令 'vx=0.4,vyaw=0.3'")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--render", type=str, default=None)
    ap.add_argument("--report", type=str, default=None)
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
        task_key, params=merged, action_scale=ascale, wheel_action_scale=wscale,
        dt=dt, policy_factory=policy,
    )
    schedule = make_schedule(task_key, args.cmd)
    ctrl.reset()
    rec, stats = run_episode(env, ctrl, schedule, sim_time, dt)
    m = compute(rec, phase, cfg)

    print(f"\n=== MPC×SAC [{task_key}] {args.cmd or '默认命令'} ===")
    print(f"时长: {stats['ep_len_s']:.1f}s  终止: {stats['terminated']}")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.render:
        Path(args.render).parent.mkdir(parents=True, exist_ok=True)
        render.states_to_video(rec, env._cfg.scene.model_file, args.render)
        print(f"渲染 → {args.render}")
    if args.report:
        lines = [f"# MPC×SAC [{task_key}] {args.cmd or ''}", "", "| 指标 | 值 |", "|---|---|"]
        for k, v in m.items():
            lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"报告 → {args.report}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
