"""批量评估共享逻辑: 多 episode 重复跑各阶段, 统计存活率/指标 (两轨复用).

共享只读 (开发规范 §3.2); 控制器类由各轨 CLI 传入 → 互不干涉。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.classic_control.common import metrics as metrics_mod
from scripts.classic_control.common import report as report_mod
from scripts.classic_control.common import rollout as rollout_mod
from scripts.classic_control.common.run import make_schedule


def run_eval(
    controller_cls,
    phases: list[int],
    episodes: int,
    out: str | None = None,
    params: dict | None = None,
) -> dict[int, list[float]]:
    """批量评估 → phase_survival dict (供 CLI 打印汇总)."""
    phase_survival: dict[int, list[float]] = {}

    for phase in phases:
        task_key = "walk_rough" if phase == 4 else "walk_flat"
        env = rollout_mod.build_env(task_key=task_key, num_envs=1)
        controller = controller_cls(
            phase,
            params=dict(params or {}),
            action_scale=float(env._cfg.control_config.action_scale),
            wheel_action_scale=float(env._cfg.control_config.wheel_action_scale),
            dt=float(env._cfg.ctrl_dt),
        )
        schedule = make_schedule(phase, None)
        dt = float(env._cfg.ctrl_dt)
        sim_time = {1: 12.0, 2: 30.0, 3: 36.0, 4: 25.0}[phase]
        survivals = []
        all_metrics: list[dict] = []
        for ep in range(episodes):
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
            agg[k] = float(np.mean(vals))
        agg["survival_rate"] = float(np.mean(survivals))
        tag = f"{controller_cls.__name__} P{phase} ({episodes} ep)"
        report_mod.print_metrics(tag, agg, phase)
        if out:
            report_mod.write_report(out, f"{tag} — 汇总", agg, phase)
        env.close()

    # 总存活率表
    print("\n=== 存活率汇总 ===")
    for phase, sv in phase_survival.items():
        print(f"  P{phase}: {np.mean(sv) * 100:.0f}% ({int(np.sum(sv))}/{len(sv)})")
    return phase_survival
