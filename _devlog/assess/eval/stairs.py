"""抬腿上台阶评估 (CLAUDE.md §7.9).

上台阶成功率≥90% (存活到 max_steps 且机身高度显著爬升) · 上下稳定.
注: 高台阶跳上/低台阶抬上的细分判定待地形语义明确后细化。
"""

from __future__ import annotations

import numpy as np
from _devlog.assess import engine
from _devlog.assess import metrics as M


def _climbed(t: M.Trace, max_steps: int) -> bool:
    """成功上台阶: 存活到 max_steps 且 base_z 净爬升 > 0.2m."""
    if len(t) < max_steps:
        return False
    z = np.array([s.base_pos[2] for s in t])
    return float(np.max(z) - np.min(z)) > 0.2


def evaluate(env, policy, args) -> dict[str, float]:
    dt = env._cfg.ctrl_dt
    num_envs = args.num_envs or 10
    max_steps = args.max_steps or 1000

    def _collect(env, st, step, i, action, dt):
        return engine.collect_step(env, st, step, i, dt)

    traces = engine.run_episodes(env, policy, num_envs, max_steps, _collect, dt)
    return {
        "success_rate": M.task_success_rate(traces, lambda t: _climbed(t, max_steps)),
        "survival_rate": M.survival_rate(traces, max_steps),
        "mean_max_z": float(
            np.mean([M.mean_max_z(t) for t in traces if t]) if traces else float("nan")
        ),
    }
