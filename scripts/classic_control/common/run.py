"""共享运行逻辑: 动力学 / 命令表 / 单阶段运行 (参数化控制器类, 两轨复用).

LQR 与 MPC 独立任务轨共用此共享只读模块 (开发规范 §3.2)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.classic_control.common import dynamics as dynamics_mod
from scripts.classic_control.common import metrics as metrics_mod
from scripts.classic_control.common import render as render_mod
from scripts.classic_control.common import report as report_mod
from scripts.classic_control.common import rollout as rollout_mod
from scripts.classic_control.common.config import PhaseCommands

ROOT = Path(__file__).resolve().parents[3]
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
    controller_cls,
    phase: int,
    *,
    sim_time: float | None = None,
    cmd_override: str | None = None,
    render_path: str | None = None,
    report_path: str | None = None,
    num_envs: int = 1,
    params: dict | None = None,
    seed: int | None = None,
) -> tuple[list[dict], dict, dict]:
    """返回 (record, metrics, stats). controller_cls: LqrController | MpcController."""
    if seed is not None:
        np.random.seed(seed)
    task_key = "walk_rough" if phase == 4 else "walk_flat"
    params = dict(params or {})
    env = rollout_mod.build_env(task_key=task_key, num_envs=num_envs)
    controller = controller_cls(
        phase,
        params=params,
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
    tag = f"{controller_cls.__name__} P{phase} ({task_key})"
    report_mod.print_metrics(tag, m, phase)
    if report_path:
        report_mod.write_report(report_path, tag, m, phase)

    if render_path and record:
        model_path = env._cfg.scene.model_file
        render_mod.states_to_video(record, model_path, render_path)
    if hasattr(controller, "last_solve_ms") and record:
        print(f"  求解耗时: 最后 {controller.last_solve_ms:.3f} ms")
    env.close()
    return record, m, stats
