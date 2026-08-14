"""经典控制共享配置加载 — 每条任务轨配置自包含 (conf/lqr 与 conf/mpc).

LQR/MPC 两条独立任务轨各持有**完整配置目录** (conf/lqr, conf/mpc):
- robot.yaml (机器人常数) + commands.yaml (命令表) + task/ (env 映射) + config.yaml (算法权重)
- 算法权重互不干涉; 机器人/命令表内容两轨相同 (同一台机器人)

本模块为共享加载器: 默认指向 conf/lqr (两轨机器人/命令表一致, 共享代码只读)。
各轨算法配置在 lqr/config.py / mpc/config.py, 显式传自己 conf_dir。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
# ★ 共享代码默认 conf (两轨 robot/commands 相同; 算法权重各轨自取)
DEFAULT_CONF = ROOT / "conf" / "lqr"


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并 dict (override 优先); 非 dict 值直接覆盖."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_common(task_key: str = "walk_flat", conf_dir: Path | None = None) -> dict:
    """合并 机器人常数 + 任务 env 映射 + 命令表 (conf_dir 指定轨)."""
    conf_dir = conf_dir or DEFAULT_CONF
    cfg = _deep_merge(_load_yaml(conf_dir / "robot.yaml"), _load_yaml(conf_dir / "commands.yaml"))
    cfg = _deep_merge(cfg, _load_yaml(conf_dir / "task" / f"xqrobotwl_{task_key}.yaml"))
    return cfg


def load_commands(task_key: str = "walk_flat", conf_dir: Path | None = None) -> dict:
    """命令表段 (yaw/commands/leg_control/phases)."""
    c = load_common(task_key, conf_dir)
    return {k: c[k] for k in ("yaw", "commands", "leg_control", "phases") if k in c}


def task_key_for_phase(phase: int) -> str:
    """阶段 → 任务: P4 地形自适应在 rough, 其余在 flat."""
    return "walk_rough" if phase >= 4 else "walk_flat"


# ── 机器人常数 (walk_flat 任务基线; 两轨同一台机器人, 取自默认 conf) ──
_CFG = load_common("walk_flat")
_robot = _CFG["robot"]

WHEEL_R = float(_robot["wheel_r"])  # 轮半径 (m)
STANDING_ANGLES = [float(x) for x in _robot["standing_angles"]]
DEFAULT_LEG_ANGLES = [float(x) for x in _robot["default_leg_angles"]]
LEG_TARGETS_COMPENSATED = [float(x) for x in _robot["leg_targets_compensated"]]
STANDING_BASE_Z = float(_robot["standing_base_z"])
HIP_Z_OFFSET = float(_robot["hip_z_offset"])
JOINT_LIMITS = {k: [float(x) for x in v] for k, v in _robot["joint_limits"].items()}
IK_MARGIN = float(_robot["ik_margin"])
VMC = {
    k: ([float(x) for x in v] if isinstance(v, list) else float(v))
    for k, v in _robot["vmc"].items()
}


def env_conf(task_key: str = "walk_flat", conf_dir: Path | None = None) -> dict:
    """任务 env 映射 (task_name/command_dim/mujoco_yaml)."""
    return load_common(task_key, conf_dir)["env"]


# ── 命令表 (phases) ──
def _phases(task_key: str = "walk_flat", conf_dir: Path | None = None) -> dict:
    return load_commands(task_key, conf_dir)["phases"]


def _fmt(segments) -> list[tuple[float, list[float]]]:
    return [(float(d), [float(x) for x in c]) for d, c in segments]


@dataclass
class PhaseCommands:
    """各阶段默认命令 (5D: vx, vy, vyaw, tsk, height)。"""

    p1: list[tuple[float, list[float]]] = field(default_factory=list)
    p2: list[tuple[float, list[float]]] = field(default_factory=list)
    p3: list[tuple[float, list[float]]] = field(default_factory=list)
    p4: list[tuple[float, list[float]]] = field(default_factory=list)

    @classmethod
    def from_config(
        cls, task_key: str = "walk_flat", conf_dir: Path | None = None
    ) -> "PhaseCommands":
        ph = _phases(task_key, conf_dir)
        return cls(
            p1=_fmt(ph["p1"]),
            p2=_fmt(ph["p2"]),
            p3=_fmt(ph["p3"]),
            p4=_fmt(ph["p4"]),
        )


def commands_params(task_key: str = "walk_flat", conf_dir: Path | None = None) -> dict:
    """命令参数 (smoothing/leg_control/yaw) → 扁平 dict, 供各轨配置合并."""
    c = load_commands(task_key, conf_dir)
    p: dict = {}
    p.update(c["yaw"])
    p["cmd_ramp_s"] = c["commands"]["cmd_ramp_s"]
    p["smoothing"] = c["commands"]["wheel_smoothing"]
    p.update(c["leg_control"])
    return p
