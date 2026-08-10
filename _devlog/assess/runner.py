#!/usr/bin/env python3
"""八任务统一评估 CLI (CLAUDE.md §1.2 每千轮评估 / §7.0 通用评估).

用法:
  uv run mjpython _devlog/assess/runner.py -t <task> -r <run> -c model_9999.pt \
      [-s decoupling|full|standing] [--num_envs N] [--pose 0-3] \
      [--dump-pose] [--report] [--no-verify]

示例:
  uv run mjpython _devlog/assess/runner.py -t walk_flat -r 2026-08-10_xx_mujoco -c model_9999.pt
  uv run mjpython _devlog/assess/runner.py -t fall_recovery -r <run> --pose 1 --num_envs 20
  uv run mjpython _devlog/assess/runner.py --list-tasks

数据优先 (§1.5): --dump-pose 额外导出 logs/pose_data/ CSV; --report 生成评估报告。
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # 仓库根
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _devlog.assess import engine, pose, report, tasks, verify  # noqa: E402
from _devlog.assess.tasks import TaskDef  # noqa: E402


def _dump_pose_csv(env, policy, task: TaskDef, ckpt_stem: str, steps: int) -> Path:
    """通用姿态数据导出: 单 env 跑 steps 步 → logs/pose_data/<task>_<ckpt>.csv."""
    import numpy as np
    import torch

    dt = env._cfg.ctrl_dt
    env.init_state()  # 初始化 NpEnvState
    samples = []
    with torch.no_grad():
        for step in range(steps):
            obs_t = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
            a = policy(obs_t).numpy().astype(np.float64)
            st = env.step(a)
            samples.append(engine.collect_step(env, st, step, 0, dt))
    out = pose.POSE_OUT_DIR / f"{task.key}_{ckpt_stem}.csv"
    return pose.write_pose_csv(samples, out)


def main() -> int:
    ap = argparse.ArgumentParser(description="八任务统一评估 (CLAUDE.md §1.2/§7.0)")
    ap.add_argument("-t", "--task", required=True, help="任务 key (--list-tasks 查看)")
    ap.add_argument("-r", "--run", required=True, help="run 目录名 (在任务 log_root 下)")
    ap.add_argument("-c", "--ckpt", default=None, help="checkpoint 文件名 (默认最新 model_*.pt)")
    ap.add_argument("-s", "--suite", default="decoupling", help="行走类场景套件")
    ap.add_argument("--num_envs", type=int, default=1, help="并行 env / episode 数")
    ap.add_argument("--max_steps", type=int, default=None, help="每 episode 最大步数")
    ap.add_argument("--pose", type=int, default=None, help="跌倒恢复固定倒地姿态 0-3")
    ap.add_argument("--jump_every", type=int, default=None, help="跳跃/后空翻触发周期 (步)")
    ap.add_argument("--dump-pose", action="store_true", help="导出姿态数据 CSV (logs/pose_data/)")
    ap.add_argument("--report", action="store_true", help="生成 Markdown 评估报告")
    ap.add_argument("--no-verify", action="store_true", help="跳过达标判定")
    ap.add_argument("--list-tasks", action="store_true")
    args = ap.parse_args()

    if args.list_tasks:
        print("八任务评估注册表:")
        for k, t in tasks.list_tasks().items():
            print(f"  {k:<14} {t.name:<16} env={t.env_name}")
        return 0

    task = tasks.get(args.task)
    num_envs = max(args.num_envs, 1)

    run_dir = engine.resolve_run_dir(args.run, task.log_root)
    ckpt_path = engine.find_checkpoint(run_dir, args.ckpt)
    env = engine.build_env(task, num_envs=num_envs, pose=args.pose, ckpt_path=ckpt_path)
    obs_dim = env.obs_groups_spec["obs"]
    policy = engine.load_policy(ckpt_path, obs_dim, task.num_actions)

    try:
        mod = importlib.import_module(f"_devlog.assess.eval.{task.key}")
        metrics = mod.evaluate(env, policy, args)
    finally:
        env.close()

    header = f"评估 · {task.name} ({task.key})  ckpt={ckpt_path.name}  run={run_dir.name}"
    report.print_metrics(task, metrics, header)

    verdicts = None
    if not args.no_verify:
        verdicts = verify.check(task, metrics)
        report.print_verdicts(task, verdicts)

    if args.dump_pose:
        env2 = engine.build_env(task, num_envs=1, pose=args.pose, ckpt_path=ckpt_path)
        try:
            steps = args.max_steps or 900
            pol2 = engine.load_policy(ckpt_path, env2.obs_groups_spec["obs"], task.num_actions)
            out = _dump_pose_csv(env2, pol2, task, ckpt_path.stem, steps)
            print(f"  姿态数据: {out}")
        finally:
            env2.close()

    if args.report:
        session = report.session_dir(task, run_dir.name, ckpt_path.name)
        meta = {"task": task.key, "run": run_dir.name, "ckpt": ckpt_path.name, "suite": args.suite}
        json_path = report.write_json(metrics, meta, session / "metrics.json")
        print(f"  JSON: {json_path}")
        if verdicts is not None:
            md_path = report.REPORTS_DIR / task.key / f"{run_dir.name}_{ckpt_path.stem}.md"
            report.write_markdown(task, metrics, verdicts, meta, md_path)
            print(f"  报告: {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
