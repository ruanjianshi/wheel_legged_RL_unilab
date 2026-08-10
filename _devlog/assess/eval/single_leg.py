"""平地单腿平衡评估 (CLAUDE.md §7.7).

形态 A: 单腿保持 ≥5s (直立 + 单轮支撑)
形态 B: 倾斜单腿行走指令追踪 (vx RMSE)
(形态 C 独轮姿态由 xqrobotwl/single_leg_unicycle 单独任务承载)
"""

from __future__ import annotations

import numpy as np
from _devlog.assess import engine
from _devlog.assess import metrics as M


def _hold_time_s(t: M.Trace, dt: float) -> float:
    """单轮支撑连续时长: 直立 (up>0.85) 且恰一轮着地."""
    if not t:
        return 0.0
    up = np.array([s.up_z for s in t])
    wc = np.array([s.wheel_contact for s in t])
    one_wheel = (np.min(wc, axis=1) < 0.5) & (np.max(wc, axis=1) > 0.5)
    mask = (up > 0.85) & one_wheel
    best = cur = 0
    for f in mask:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return float(best * dt)


def evaluate(env, policy, args) -> dict[str, float]:
    dt = env._cfg.ctrl_dt
    num_envs = args.num_envs or 10
    max_steps = args.max_steps or 1000

    def _collect(env, st, step, i, action, dt):
        return engine.collect_step(env, st, step, i, dt)

    traces = engine.run_episodes(env, policy, num_envs, max_steps, _collect, dt)
    holds = [_hold_time_s(t, dt) for t in traces if t]

    # 形态 B: 倾斜单腿行走 vx 追踪
    vx_trace = engine.run_cmd_scenario(env, policy, [0.3, 0.0, 0.0, 0.0, 0.65], 10.0, 2.0, dt)
    rmse = M.tracking_rmse(vx_trace[0], np.array([0.3, 0.0, 0.0]))

    return {
        "hold_time": float(np.mean(holds)) if holds else float("nan"),
        "vx_tracking_rmse": rmse["vx"],
        "survival_rate": M.survival_rate(traces, max_steps),
    }
