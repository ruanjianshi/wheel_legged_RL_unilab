"""经典控制轨配置加载: conf/classic_control/*.yaml → 常数 + dataclasses.

结构对齐 RL 算法配置: conf/<algo>/config.yaml (算法超参 + 命令表)
+ conf/<algo>/task/<task>.yaml (任务 env 映射 + 机器人常数)。
手动 YAML deep-merge (不依赖 hydra); robot 常数基线取 walk_flat 任务。

独立于 RL 任务 (只读复用 env/XML), 供 LQR/MPC 平衡控制器共用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT / "conf" / "classic_control"
_BASE_CONF = CONF_DIR / "config.yaml"
_FLAT_TASK = CONF_DIR / "task" / "xqrobotwl_walk_flat.yaml"
_TASK_DIR = CONF_DIR / "task"


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


def load_config(task_key: str = "walk_flat") -> dict:
    """合并 base 算法配置 + 任务配置 → 配置 dict.

    robot 常数基线取 walk_flat 任务 (两任务同一台机器人);
    指定任务仅覆盖差异 (walk_rough 覆盖 env 映射)。
    """
    cfg = _deep_merge(_load_yaml(_BASE_CONF), _load_yaml(_FLAT_TASK))
    if task_key != "walk_flat":
        cfg = _deep_merge(cfg, _load_yaml(_TASK_DIR / f"xqrobotwl_{task_key}.yaml"))
    return cfg


def task_key_for_phase(phase: int) -> str:
    """阶段 → 任务: P4 地形自适应在 rough, 其余在 flat."""
    return "walk_rough" if phase >= 4 else "walk_flat"


# ── 模块常数 (walk_flat 任务基线, 两任务同一台机器人) ──
_CFG = load_config("walk_flat")
_robot = _CFG["robot"]

WHEEL_R = float(_robot["wheel_r"])  # 轮半径 (m), 见 xqrobotwl.xml wheel cylinder
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


def default_params(task_key: str = "walk_flat") -> dict:
    """conf YAML → 扁平参数字典 (含 SagittalConfig 字段 + smoothing 等控制器字段).

    CLI 覆盖合并: build_cfg_and_overrides(task_key, overrides) 返回
    (SagittalConfig, merged) — merged 保留 smoothing/sign 等非 dataclass 字段。
    """
    c = load_config(task_key)
    p: dict = {}
    p.update(c["sagittal"])
    p.update(c["mpc"])
    p.update(c["yaw"])
    p["cmd_ramp_s"] = c["commands"]["cmd_ramp_s"]
    p["smoothing"] = c["commands"]["wheel_smoothing"]  # 轮速命令平滑 α
    p.update(c["leg_control"])
    return p


@dataclass
class SagittalConfig:
    """矢状面倒立摆模型 + 控制器参数 (来自 conf/classic_control/config.yaml)."""

    q_theta: float = 100.0  # MPC θ 代价权重
    q_theta_dot: float = 20.0  # θ̇ 权重
    q_v: float = 80.0  # v 权重 (★ 漂移控制关键)
    q_x: float = 30.0  # 位置权重 (★ 漂移控制关键)
    q_z: float = 5.0  # LQI 积分权重 (速度跟踪)
    r: float = 1.0  # 控制权重

    # MPC (轮速命令模型: v̇=(uR−v)/τ, θ̈=αθ+β·v̇; u=轮速命令 rad/s)
    mpc_horizon: int = 20
    u_max: float = 30.0  # 轮速命令限 (rad/s, 匹配 wheel_vel_max clip)
    theta_max: float = 0.35  # 倾角限 (rad ≈20°)
    v_max: float = 2.5  # 轮速限 (m/s)
    wheel_vel_max: float = 25.0  # 轮角速度 clip (rad/s)
    tau: float = 0.059  # 轮速度伺服时间常数 (s), 实测
    alpha: float = 24.4  # 倒立摆 α (θ̈=αθ), 自由落体实测
    beta: float = -2.5  # 轮加速耦合 β (θ̈=β·v̇_wheel), 解析 −1/L_eff
    terminal_lqr: bool = True  # 末端 LQR 代价 (鲁棒性)
    r_rate: float = 0.5  # 控制变化率惩罚 (消抖动)
    integral_gain: float = 1.5  # P2 积分参考偏置增益
    model_file: str = "logs/classic/mpc_plant_bb.npz"  # 黑箱模型覆盖解析

    # 偏航 (差分轮速)
    track_width: float = 0.38  # 左右轮距 (m), 运行时 get_body_pos_w 实测覆盖
    k_yaw: float = 3.0  # yaw 率 P 增益 (rad/s per rad/s error)
    # ★ 指令斜坡: v_ref 平滑过渡 (阶跃命令会暴力刹车→倒; 斜坡 1-2s 消除)
    cmd_ramp_s: float = 1.5

    # 腿控 (★ P3: 膝高度伺服, 补偿基座 + 积分; L0-IK 受 kp=60 下垂限制不可靠)
    height_kp: float = 0.8  # 高度伺服增益 (膝随高度误差)
    height_kd: float = 0.3  # 高度阻尼 (基高变化率)
    height_ki: float = 0.2  # 高度积分 (消稳态误差)
    # ★ 高度命令平滑 (Q/E 快速按 → 高度阶跃 → 膝猛变 → 不稳; 平滑斜坡消除)
    height_smoothing: float = 0.97
    # ★ 腿协助平衡: hip_pitch 随倾角偏移 CoM (双腿同向)
    #   -0.3 实测改善 max_tilt (0.206→0.191) 且存活; -0.5 过长调度会失稳
    leg_balance_kp: float = -0.3
    l0_margin: float = 0.03  # IK 腿长 clamp margin


_SAGGITAL_FIELDS = frozenset(SagittalConfig.__dataclass_fields__)


def build_cfg_and_overrides(
    task_key: str = "walk_flat", overrides: dict | None = None
) -> tuple[SagittalConfig, dict]:
    """合并 conf 默认 + CLI 覆盖 → (SagittalConfig, merged).

    merged 保留 smoothing/sign 等非 dataclass 字段 (控制器读取), 仅字段键入 dataclass。
    """
    p = default_params(task_key)
    if overrides:
        p = {**p, **overrides}
    cfg = SagittalConfig(**{k: v for k, v in p.items() if k in _SAGGITAL_FIELDS})
    return cfg, p


def _phases(task_key: str = "walk_flat") -> dict:
    """phase → [(时长 s, cmd)] 命令表 (来自 config.yaml phases 段)."""
    return load_config(task_key)["phases"]


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
    def from_config(cls, task_key: str = "walk_flat") -> "PhaseCommands":
        ph = _phases(task_key)
        return cls(
            p1=_fmt(ph["p1"]),
            p2=_fmt(ph["p2"]),
            p3=_fmt(ph["p3"]),
            p4=_fmt(ph["p4"]),
        )
