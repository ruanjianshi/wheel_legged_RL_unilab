"""单阶段运行公共逻辑: 动力学加载 → env → 控制器 → rollout → 指标/报告/渲染."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.classic_control import dynamics as dynamics_mod
from scripts.classic_control import metrics as metrics_mod
from scripts.classic_control import render as render_mod
from scripts.classic_control import report as report_mod
from scripts.classic_control import rollout as rollout_mod
from scripts.classic_control.config import PhaseCommands
from scripts.classic_control.controller import BalanceController

ROOT = Path(__file__).resolve().parents[2]
DYNAMICS_PATH = ROOT / "logs" / "classic" / "dynamics_flat.npz"


def get_dynamics(
    dt: float = 0.01, alpha: float | None = None, beta: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """模型动力学: 默认解析倒立摆 (α=g/h≈25, β≈-1/h; 数值测量受软腿污染不可靠).

    可通过 --alpha/--beta 覆盖 (闭环绕调).
    """
    a = 25.0 if alpha is None else alpha
    b = -2.5 if beta is None else beta
    return dynamics_mod.analytic(a, b, dt)


def make_schedule(phase: int, cmd_override: str | None) -> rollout_mod.CmdSchedule:
    pc = PhaseCommands.from_config()
    if phase == 1:
        return rollout_mod.CmdSchedule(pc.p1)
    segments = {2: pc.p2, 3: pc.p3, 4: pc.p4}[phase]
    if cmd_override:
        # 单条恒定命令: "vx=0.5,vyaw=0.3,height=0.65" → 全时段
        cmd = [0.0, 0.0, 0.0, 0.0, 0.518]
        for token in cmd_override.split(","):
            k, _, val = token.strip().partition("=")
            idx = {"vx": 0, "vy": 1, "vyaw": 2, "tsk": 3, "height": 4}.get(k.strip())
            if idx is not None:
                cmd[idx] = float(val)
        total = sum(d for d, _ in segments)
        segments = [(total, cmd)]
    return rollout_mod.CmdSchedule(segments)


def run_phase(
    kind: str,
    phase: int,
    *,
    sim_time: float | None = None,
    cmd_override: str | None = None,
    render_path: str | None = None,
    report_path: str | None = None,
    num_envs: int = 1,
    mpc_horizon: int = 20,
    params: dict | None = None,
    seed: int | None = None,
) -> tuple[list[dict], dict, dict]:
    """返回 (record, metrics, stats)."""
    if seed is not None:
        np.random.seed(seed)
    task_key = "walk_rough" if phase == 4 else "walk_flat"
    params = dict(params or {})
    A_d, B_d = get_dynamics(alpha=params.pop("alpha", None), beta=params.pop("beta", None))
    env = rollout_mod.build_env(task_key=task_key, num_envs=num_envs)
    p = dict(params or {})
    if kind == "mpc":
        p.setdefault("mpc_horizon", mpc_horizon)
    controller = BalanceController(
        kind,
        phase,
        A_d,
        B_d,
        params=p,
        action_scale=float(env._cfg.control_config.action_scale),
        wheel_action_scale=float(env._cfg.control_config.wheel_action_scale),
        dt=float(env._cfg.ctrl_dt),
    )
    schedule = make_schedule(phase, cmd_override)
    dt = float(env._cfg.ctrl_dt)
    default_time = {1: 15.0, 2: 35.0, 3: 40.0, 4: 30.0}[phase]
    sim_time = default_time if sim_time is None else sim_time
    record, stats = rollout_mod.run_episode(env, controller, schedule, sim_time, dt)

    m = metrics_mod.compute(record, phase)
    tag = f"{kind.upper()} P{phase} ({task_key})"
    report_mod.print_metrics(tag, m, phase)
    if report_path:
        report_mod.write_report(report_path, tag, m, phase)

    if render_path and record:
        model_path = env._cfg.scene.model_file
        render_mod.states_to_video(record, model_path, render_path)
    if kind == "mpc" and record:
        print(f"  MPC 求解耗时: 最后 {controller.last_solve_ms:.3f} ms")
    env.close()
    return record, m, stats
