"""跌倒恢复评估 (CLAUDE.md §7.8 + 附录 A).

4 姿态恢复率≥80% · 最长连续站立≥0.5s · 水平漂移<0.5m · 站立 |gyro|<1 · 轮子离地率 0
· 转圈 (yaw 累计≈walk 水平~56°) · 轮速差小 (附录A)
复用 eval_fall_recovery 的固定倒地姿态 provider (runner 建 env 时按 --pose 注入)。
"""

from __future__ import annotations

import numpy as np
from _devlog.assess import engine
from _devlog.assess import metrics as M


def evaluate(env, policy, args) -> dict[str, float]:
    dt = env._cfg.ctrl_dt
    num_envs = args.num_envs or 20
    max_steps = args.max_steps or 800

    def _collect(env, st, step, i, action, dt):
        return engine.collect_step(env, st, step, i, dt)

    traces = engine.run_episodes(env, policy, num_envs, max_steps, _collect, dt)

    stand_z = float(getattr(getattr(env, "_jump_cfg", None), "h_cmd2", 0.55))

    def _agg(fn) -> float:
        vals = [fn(t) for t in traces if t]
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "recovery_rate": M.recovery_rate(traces, stand_z),
        "stay_up_rate": M.stay_up_rate(traces, max_steps),
        "longest_stand": _agg(lambda t: M.longest_stand_s(t, dt)),
        "drift": _agg(M.drift),
        "stand_gyro": _agg(M.mean_gyro),
        "wheel_off_rate": _agg(M.wheel_off_rate),
        "yaw_accum": _agg(M.yaw_accum),  # ★ 转圈: 站立期 yaw 累计 (≈walk 56°=0.98rad)
        "wheel_speed_diff": _agg(M.wheel_speed_diff),  # ★ 左右轮速差 (附录A 小)
        "mean_max_z": _agg(M.mean_max_z),
        "mean_max_up": _agg(M.mean_max_up),
        "double_wheel_on": _agg(M.double_wheel_on_rate),
    }
