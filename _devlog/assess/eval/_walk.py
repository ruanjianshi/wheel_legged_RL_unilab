"""共享行走类评估逻辑 (walk_flat / toe_walk / walk_rough).

跑命令场景套件 (decoupling/full) + 恒定追加零指令站立场景,
统计 §7.2-7.4 的追踪误差 / 存活率 / 微动平衡, 并叠加各任务自身指标。
"""

from __future__ import annotations

import math

import numpy as np
from _devlog.assess import engine
from _devlog.assess import metrics as M
from _devlog.assess.scenarios import SHARED_WALK_SUITES, STANDING, get_suite


def _mean(vals: list[float]) -> float:
    clean = [v for v in vals if not math.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


def evaluate_walk(
    env,
    policy,
    args,
    *,
    height_err: bool = False,
    jerk: bool = False,
    rough_std: bool = False,
) -> dict[str, float]:
    """行走类通用评估.

    height_err: 报告站立高度偏离 0.52m (toe_walk §7.3)
    jerk:       报告抬腿平缓度 leg_jerk (toe_walk §7.3)
    rough_std:  报告机身高度波动 base_height_std (rough §7.4)
    """
    dt = env._cfg.ctrl_dt
    suite = get_suite(args.suite, SHARED_WALK_SUITES)
    scenarios = [*suite.scenarios, STANDING.scenarios[0]]  # 恒定追加站立场景

    per_scen: dict[str, dict] = {}
    survived = 0
    for sc in scenarios:
        traces = engine.run_cmd_scenario(env, policy, sc.cmd, sc.duration, sc.warmup, dt)
        trace = traces[0]
        per_scen[sc.name] = {"trace": trace, "cmd": np.asarray(sc.cmd)}
        expected = int((sc.duration - sc.warmup) / dt) - 1
        if len(trace) >= expected:
            survived += 1

    n_scen = len(scenarios)
    metrics: dict[str, float] = {"survival_rate": survived / n_scen if n_scen else float("nan")}

    # 追踪 RMSE (所有命令场景)
    rmse_vx, rmse_vy = [], []
    for d in per_scen.values():
        r = M.tracking_rmse(d["trace"], d["cmd"])
        rmse_vx.append(r["vx"])
        rmse_vy.append(r["vy"])
    metrics["vx_tracking_rmse"] = _mean(rmse_vx)
    metrics["vy_tracking_rmse"] = _mean(rmse_vy)

    # 侧移能力 (§7.2 侧移 Vy): 侧向指令下 |avg vy|
    side = [abs(M.avg_linvel(d["trace"])[1]) for d in per_scen.values() if abs(d["cmd"][1]) > 1e-4]
    metrics["side_vy"] = _mean(side)

    # 微动平衡 (站立场景)
    stand = per_scen.get("stand")
    if stand:
        t = stand["trace"]
        metrics["stand_linvel_xy"] = M.mean_linvel_xy(t)
        metrics["stand_gyro"] = M.mean_gyro(t)
        metrics["yaw_accum"] = M.yaw_accum(t)
        metrics["wheel_off_rate"] = M.wheel_off_rate(t)
        metrics["stand_height"] = M.stand_height(t)
        if height_err:
            metrics["base_height_err"] = M.stand_height_err(t, 0.52)
        if jerk:
            fwd = [d["trace"] for d in per_scen.values() if abs(d["cmd"][0]) > 1e-4]
            metrics["leg_jerk"] = _mean([M.leg_jerk(t_, dt) for t_ in fwd])
        if rough_std:
            fwd = [d["trace"] for d in per_scen.values() if abs(d["cmd"][0]) > 1e-4]
            metrics["base_height_std"] = _mean([M.stand_height_std(t_) for t_ in fwd])
    return metrics
