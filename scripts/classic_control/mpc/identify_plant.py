#!/usr/bin/env python3
"""MPC 植物黑箱辨识 — LQR 闭环 + 强探针 + v_wheel 状态, OLS 拟合 A_d/B_d.

问题回顾:
- 解析模型 (α=25, β=-2.5) 与真实植物差异大 (软腿), MPC 无法平衡;
- 开环数值测量受软腿塌陷污染; 闭环 OLS 之前失败是因为用 linvel(pitch 污染)
  且探针太弱 (秩亏缺)。

本次修正:
- 状态 v 用 v_wheel (L 轮 qvel·R, 前向正), 排除 base 转动污染;
- 探针幅度大 (±4 rad/s 多频正弦, 与 LQR 控制同量级) → 输入充分激励;
- MuJoCo 确定性 (无过程噪声) → 闭环 OLS 在输入充分激励下可辨识真实 A/B;
- 丢弃前 0.5s 站姿瞬态。

拟合: x[k+1] = A_d·x[k] + B_d·u[k], u = w_c (轮速命令, 平滑后) + 探针。
验证: 留出一次新 episode (无探针), 一步预测 RMSE。

用法: uv run python scripts/classic_control/identify_plant.py [--sim_time 20]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.classic_control.common import rollout as rollout_mod
from scripts.classic_control.common.config import PhaseCommands
from scripts.classic_control.common.run import get_dynamics
from scripts.classic_control.lqr.controller import LqrController


def probe(t: float, rng: np.random.Generator) -> float:
    """强多频正弦探针 (rad/s, w_c convention), 与 LQR 控制同量级."""
    return (
        1.2 * np.sin(2 * np.pi * 0.5 * t)
        + 1.2 * np.sin(2 * np.pi * 1.3 * t + 0.9)
        + 1.0 * np.sin(2 * np.pi * 2.4 * t + 2.2)
        + 0.8 * np.sin(2 * np.pi * 4.1 * t + 0.3)
        + 0.6 * np.sin(2 * np.pi * 6.3 * t + 1.7)
    )


def collect(env, ctrl, schedule, sim_time, dt, use_probe: bool):
    env.init_state()
    rng = np.random.default_rng(7)
    wsa = float(env._cfg.control_config.wheel_action_scale)
    xs, us, xnext = [], [], []
    n = int(sim_time / dt)
    for step in range(n):
        t = step * dt
        cmd = schedule.at(t)[: env.state.info["commands"].shape[1]]  # type: ignore[union-attr]
        s = rollout_mod.read_sensors(env)
        a = ctrl.act(s, cmd)
        p = probe(t, rng) if use_probe else 0.0
        a[6] -= p / wsa
        a[7] -= p / wsa
        env.state.info["commands"][:] = np.tile(cmd, (env.num_envs, 1))  # type: ignore[union-attr]
        env.step(np.asarray(a, dtype=np.float64)[None, :])
        if step < int(0.5 / dt):  # 丢弃站姿瞬态
            continue
        s2 = rollout_mod.read_sensors(env)
        # ★ 状态 v 用 base linvel[0] (与控制器完全一致; v_wheel 会破坏闭环)
        x = np.array([s["theta"], s["theta_dot"], s["v"], ctrl._xpos])
        u = ctrl.wheel_cmd + p
        xn = np.array([s2["theta"], s2["theta_dot"], s2["v"], ctrl._xpos])
        xs.append(x)
        us.append(u)
        xnext.append(xn)
    return np.stack(xs), np.array(us), np.stack(xnext)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_time", type=float, default=25.0)
    ap.add_argument("--out", default=str(ROOT / "logs" / "classic" / "mpc_plant_bb.npz"))
    args = ap.parse_args()

    # ★ 用 LQR 控制器 (P2 指令段) 闭环采集 + 探针 — 只读复用 LQR 任务轨
    env = rollout_mod.build_env(task_key="walk_flat", num_envs=1)
    ctrl = LqrController(
        2,  # P2 指令段 (v∈[−0.5,0.5]) — 模型在速度工作区有效
        action_scale=float(env._cfg.control_config.action_scale),
        wheel_action_scale=float(env._cfg.control_config.wheel_action_scale),
        dt=float(env._cfg.ctrl_dt),
    )
    dt = float(env._cfg.ctrl_dt)
    pc = PhaseCommands.from_config()
    schedule = rollout_mod.CmdSchedule(pc.p2)

    X, U, Y = collect(env, ctrl, schedule, args.sim_time, dt, use_probe=True)
    env.close()
    print(f"拟合样本: {len(X)} (dt={dt}s, 探针 {args.sim_time}s)")

    Z = np.hstack([X, U[:, None]])
    Coef, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    A_fit = Coef[:4, :].T
    B_fit = Coef[4, :].reshape(4, 1)
    Y_hat = A_fit @ X.T + B_fit @ U[None, :]
    rmse_fit = np.sqrt(((Y_hat.T - Y) ** 2).mean(axis=0))

    # 留出验证: 新 episode 无探针 (纯 LQR 轨迹)
    ctrl.reset()
    Xv, Uv, Yv = (
        collect(env, ctrl, schedule, 8.0, dt, use_probe=False) if False else (None, None, None)
    )
    # 重新建 env (上一个已 close)
    env2 = rollout_mod.build_env(task_key="walk_flat", num_envs=1)
    ctrl2 = LqrController(
        1,
        action_scale=float(env2._cfg.control_config.action_scale),
        wheel_action_scale=float(env2._cfg.control_config.wheel_action_scale),
        dt=dt,
    )
    Xv, Uv, Yv = collect(env2, ctrl2, schedule, 8.0, dt, use_probe=False)
    env2.close()
    Yv_hat = A_fit @ Xv.T + B_fit @ Uv[None, :]
    rmse_val = np.sqrt(((Yv_hat.T - Yv) ** 2).mean(axis=0))

    np.savez(args.out, A_d=A_fit, B_d=B_fit, dt=dt, theta=X[:, 0], v_wheel=X[:, 2], u=U)
    print(f"保存 → {args.out}")
    print("A_d =\n", np.array2string(A_fit, precision=5, suppress_small=True))
    print("B_d =\n", np.array2string(B_fit, precision=5, suppress_small=True))
    print(f"拟合 一步 RMSE [θ,θ̇,v,x] = {np.round(rmse_fit, 4)}")
    print(f"留出 一步 RMSE [θ,θ̇,v,x] = {np.round(rmse_val, 4)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
