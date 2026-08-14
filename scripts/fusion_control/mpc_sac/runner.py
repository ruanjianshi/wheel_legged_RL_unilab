"""MPC×SAC 融合分支 — episode 运行器 (record 格式对齐 common.metrics).

不能用 common.run.run_episode: 它给控制器直接传命令; 融合控制器需要"期望命令"
(高层策略据此发实际命令)。本运行器每步传期望命令 + 记录高层实际命令 cmd_hi。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.classic_control.common.config import PhaseCommands
from scripts.classic_control.common.rollout import CmdSchedule
from scripts.fusion_control.mpc_sac.config import CONF_DIR
from scripts.fusion_control.mpc_sac.env import read_sensors


def make_schedule(task_key: str = "walk_flat", cmd_override: str | None = None) -> CmdSchedule:
    """命令表 (来自本分支 conf/commands.yaml phases); cmd_override 如 'vx=0.4,vyaw=0.3'."""
    if cmd_override:
        import re

        kv: dict[str, float] = {}
        for m in re.finditer(r"(\w+)=([-\d.]+)", cmd_override):
            kv[m.group(1)] = float(m.group(2))
        if task_key == "walk_flat":
            cmd = [kv.get("vx", 0.0), 0.0, kv.get("vyaw", 0.0), 0.0, kv.get("height", 0.518)]
        else:
            cmd = [kv.get("vx", 0.0), 0.0, kv.get("vyaw", 0.0), 0.0]
        # ★ MPC 起步直接命令失衡 → 先 2s 站姿再发命令 (同经典评估站姿预热)
        stand = np.zeros_like(np.asarray(cmd, dtype=np.float64))
        if task_key == "walk_flat":
            stand[4] = 0.518
        return CmdSchedule([(2.0, stand), (1e9, cmd)])
    phases = PhaseCommands.from_config(task_key, conf_dir=CONF_DIR)
    # flat→P3 (速度+高度), rough→P4 (地形)
    seg = phases.p3 if task_key == "walk_flat" else phases.p4
    return CmdSchedule(seg)


def run_episode(
    env: Any,
    ctrl: Any,
    schedule: CmdSchedule,
    sim_time: float,
    dt: float = 0.01,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """跑一段 episode: 每步传期望命令 → 融合控制器 → a8 → env.step. 终止即断."""
    if env.state is None:
        env.init_state()
    record: list[dict[str, Any]] = []
    n_steps = int(sim_time / dt)
    terminated = False
    for step in range(n_steps):
        t = step * dt
        des = schedule.at(t)
        des = np.asarray(des, dtype=np.float64)[: env.state.info["commands"].shape[1]]
        sensors = read_sensors(env)
        a8 = ctrl.act(sensors, des)
        cmd_hi = ctrl.last_cmd
        env.state.info["commands"][:] = np.tile(cmd_hi, (env.num_envs, 1))
        st = env.step(np.asarray(a8, dtype=np.float64)[None, :])
        solve_ms = float(getattr(ctrl, "last_solve_ms", 0.0))
        phy = np.asarray(env._backend.get_physics_state(), dtype=np.float64)[0]
        record.append(
            {
                "t": t,
                "state": np.concatenate([[t], phy[1:]]),
                "sensors": sensors,
                "cmd": np.asarray(des, dtype=np.float64).copy(),  # 期望命令 (metrics 测端到端跟踪)
                "cmd_hi": np.asarray(cmd_hi, dtype=np.float64).copy(),  # 高层实际命令
                "action": np.asarray(a8, dtype=np.float64).copy(),
                "solve_ms": solve_ms,
                "terminated": bool(st.terminated[0]),
                "truncated": bool(st.truncated[0]),
            }
        )
        if st.terminated[0] or st.truncated[0]:
            terminated = True
            break
    stats = {"ep_len_s": len(record) * dt, "terminated": terminated, "n_steps": len(record)}
    return record, stats
