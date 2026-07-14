"""Test scenario definitions for XqRobotV2 policy evaluation.

Each scenario specifies a velocity command and evaluation duration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvalScenario:
    """Single test case with target velocity command."""

    name: str
    cmd: list[float]  # [vx, vy, vyaw, tsk, height]
    duration: float = 5.0  # evaluation time (seconds)
    warmup: float = 1.5  # settling time before measurement (seconds)
    description: str = ""


@dataclass
class EvalSuite:
    """Collection of scenarios forming a complete evaluation."""

    name: str
    description: str = ""
    ctrl_dt: float = 0.01
    scenarios: list[EvalScenario] = field(default_factory=list)


# ── Preset suites ─────────────────────────────────────────────────────────

FLAT_WALK_DECOUPLING = EvalSuite(
    name="flat_walk_decoupling",
    description="Vx/Vy decoupling test for flat walk policy",
    scenarios=[
        EvalScenario("fwd_vx=0.6", [0.6, 0.0, 0.0, 0.0, 0.65], description="Forward high speed"),
        EvalScenario("fwd_vx=0.3", [0.3, 0.0, 0.0, 0.0, 0.65], description="Forward medium speed"),
        EvalScenario("fwd_vx=-0.3", [-0.3, 0.0, 0.0, 0.0, 0.65], description="Backward"),
        EvalScenario("lat_vy=+0.3", [0.0, 0.3, 0.0, 0.0, 0.65], description="Strafe right"),
        EvalScenario("lat_vy=-0.3", [0.0, -0.3, 0.0, 0.0, 0.65], description="Strafe left"),
        EvalScenario("fwd+lat", [0.3, 0.2, 0.0, 0.0, 0.65], description="Diagonal"),
    ],
)

FLAT_WALK_FULL = EvalSuite(
    name="flat_walk_full",
    description="Comprehensive flat walk evaluation",
    scenarios=[
        # Forward speed sweep
        EvalScenario("vx=0.1", [0.1, 0.0, 0.0, 0.0, 0.65], duration=3.0),
        EvalScenario("vx=0.2", [0.2, 0.0, 0.0, 0.0, 0.65], duration=3.0),
        EvalScenario("vx=0.3", [0.3, 0.0, 0.0, 0.0, 0.65], duration=3.0),
        EvalScenario("vx=0.4", [0.4, 0.0, 0.0, 0.0, 0.65], duration=3.0),
        EvalScenario("vx=0.5", [0.5, 0.0, 0.0, 0.0, 0.65], duration=3.0),
        EvalScenario("vx=0.6", [0.6, 0.0, 0.0, 0.0, 0.65], duration=3.0),
        # Lateral speed sweep
        EvalScenario("vy=0.1", [0.0, 0.1, 0.0, 0.0, 0.65], duration=3.0),
        EvalScenario("vy=0.2", [0.0, 0.2, 0.0, 0.0, 0.65], duration=3.0),
        EvalScenario("vy=0.3", [0.0, 0.3, 0.0, 0.0, 0.65], duration=3.0),
        # Yaw
        EvalScenario("vyaw=0.5", [0.0, 0.0, 0.5, 0.0, 0.65], duration=3.0),
        EvalScenario("vyaw=1.0", [0.0, 0.0, 1.0, 0.0, 0.65], duration=3.0),
        # Backward
        EvalScenario("vx=-0.3", [-0.3, 0.0, 0.0, 0.0, 0.65], duration=3.0),
        EvalScenario("vx=-0.6", [-0.6, 0.0, 0.0, 0.0, 0.65], duration=3.0),
        # Diagonal
        EvalScenario("vx=0.3_vy=0.2", [0.3, 0.2, 0.0, 0.0, 0.65], duration=3.0),
        EvalScenario("vx=0.3_vyaw=0.5", [0.3, 0.0, 0.5, 0.0, 0.65], duration=3.0),
        EvalScenario("vx=0.3_vy=0.2_vyaw=0.5", [0.3, 0.2, 0.5, 0.0, 0.65], duration=3.0),
    ],
)

STANDING_STABILITY = EvalSuite(
    name="standing_stability",
    description="Standing still stability test",
    scenarios=[
        EvalScenario(
            "stand",
            [0.0, 0.0, 0.0, 0.0, 0.65],
            duration=10.0,
            warmup=2.0,
            description="Zero command standing",
        ),
    ],
)

TOE_WALK_TRACKING = EvalSuite(
    name="toe_walk_tracking",
    description="Toe walk trajectory tracking evaluation",
    scenarios=[
        # Add toe walk specific scenarios here
    ],
)

# Suite registry
SUITES: dict[str, EvalSuite] = {
    "decoupling": FLAT_WALK_DECOUPLING,
    "full": FLAT_WALK_FULL,
    "standing": STANDING_STABILITY,
    "toe_walk": TOE_WALK_TRACKING,
}


def load_suite(path: str | Path) -> EvalSuite:
    """Load evaluation suite from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    scenarios = [EvalScenario(**s) for s in data.pop("scenarios", [])]
    return EvalSuite(scenarios=scenarios, **data)


def get_suite(name: str) -> EvalSuite:
    """Get a preset suite by name."""
    if name in SUITES:
        return SUITES[name]
    raise KeyError(f"Unknown suite: {name}. Available: {list(SUITES.keys())}")
