"""评估场景定义 (CLAUDE.md §7.0 通用评估).

每个场景一个速度命令 + 时长 (行走类任务); 动作类任务 (jump/backflip/fall_recovery/
single_leg/stairs) 由各自 eval 模块定义触发/姿态/地形参数, 不走固定命令场景。

保留旧框架的 EvalScenario/EvalSuite + YAML 加载; 新增八任务共享的行走套件。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvalScenario:
    """单场景: 目标速度命令 [vx, vy, vyaw, tsk, height]."""

    name: str
    cmd: list[float]
    duration: float = 5.0  # 评估时长 (s)
    warmup: float = 1.5  # 稳定期 (s, 不纳入统计)
    description: str = ""


@dataclass
class EvalSuite:
    """场景集合构成一次完整评估."""

    name: str
    description: str = ""
    ctrl_dt: float = 0.01
    scenarios: list[EvalScenario] = field(default_factory=list)


# ── 共享行走套件 (walk_flat / toe_walk / walk_rough) ────────────────────────

WALK_DECOUPLING = EvalSuite(
    name="decoupling",
    description="Vx/Vy 解耦快速测试 (§7.0 验证脚本)",
    scenarios=[
        EvalScenario("fwd_vx=0.6", [0.6, 0.0, 0.0, 0.0, 0.65], duration=5.0, warmup=1.5),
        EvalScenario("fwd_vx=0.3", [0.3, 0.0, 0.0, 0.0, 0.65], duration=5.0, warmup=1.5),
        EvalScenario("fwd_vx=-0.3", [-0.3, 0.0, 0.0, 0.0, 0.65], duration=5.0, warmup=1.5),
        EvalScenario("lat_vy=+0.3", [0.0, 0.3, 0.0, 0.0, 0.65], duration=5.0, warmup=1.5),
        EvalScenario("lat_vy=-0.3", [0.0, -0.3, 0.0, 0.0, 0.65], duration=5.0, warmup=1.5),
        EvalScenario("fwd+lat", [0.3, 0.2, 0.0, 0.0, 0.65], duration=5.0, warmup=1.5),
    ],
)

WALK_FULL = EvalSuite(
    name="full",
    description="全速度扫频评估 (§7.0 长时追踪)",
    scenarios=[
        EvalScenario(f"vx={v}", [v, 0.0, 0.0, 0.0, 0.65], duration=3.0, warmup=1.5)
        for v in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    ]
    + [
        EvalScenario(f"vy={v}", [0.0, v, 0.0, 0.0, 0.65], duration=3.0, warmup=1.5)
        for v in (0.1, 0.2, 0.3)
    ]
    + [
        EvalScenario(f"vyaw={v}", [0.0, 0.0, v, 0.0, 0.65], duration=3.0, warmup=1.5)
        for v in (0.5, 1.0)
    ]
    + [
        EvalScenario("vx=-0.3", [-0.3, 0.0, 0.0, 0.0, 0.65], duration=3.0, warmup=1.5),
        EvalScenario("vx=-0.6", [-0.6, 0.0, 0.0, 0.0, 0.65], duration=3.0, warmup=1.5),
        EvalScenario("diag_vx.3_vy.2", [0.3, 0.2, 0.0, 0.0, 0.65], duration=3.0, warmup=1.5),
    ],
)

STANDING = EvalSuite(
    name="standing",
    description="零指令静止稳定性 (§1.4 微动平衡)",
    scenarios=[
        EvalScenario(
            "stand",
            [0.0, 0.0, 0.0, 0.0, 0.65],
            duration=10.0,
            warmup=2.0,
            description="零指令站立 ≥10s",
        ),
    ],
)

SHARED_WALK_SUITES: dict[str, EvalSuite] = {
    "decoupling": WALK_DECOUPLING,
    "full": WALK_FULL,
    "standing": STANDING,
}


def load_suite(path: str | Path) -> EvalSuite:
    """从 YAML 加载场景套件."""
    with open(path) as f:
        data = yaml.safe_load(f)
    scenarios = [EvalScenario(**s) for s in data.pop("scenarios", [])]
    return EvalSuite(scenarios=scenarios, **data)


def get_suite(name: str, suites: dict[str, EvalSuite]) -> EvalSuite:
    """按名取套件, 未知则报错."""
    if name in suites:
        return suites[name]
    raise KeyError(f"未知场景套件: {name}, 可选: {list(suites)}")
