"""跳跃评估场景定义.

评估场景:
  - fix_01m, fix_02m, fix_03m: 固定 vx, 周期触发跳跃
  - random: vx~U[0, 1.0]
  - platform: 跳上 0.15m 平台
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 触发周期: 每2秒触发一次跳跃 (0.01s控制周期)
JUMP_PERIOD = 200  # 2s
JUMP_DUTY = 100    # 1s 触发

# 固定距离场景: vx 映射到目标跳距
FIXED_DIST_SCENARIOS = {
    "fix_01m": {"vx": 0.3,  "target_distance": 0.10},   # 实际跳距还需训练验证
    "fix_02m": {"vx": 0.6,  "target_distance": 0.20},
    "fix_03m": {"vx": 1.0,  "target_distance": 0.30},
}

RANDOM_SCENARIO = {
    "vx_min": 0.0,
    "vx_max": 1.0,
    "target_distance": None,  # 随机, 不做精确距离要求
}

PLATFORM_SCENARIO = {
    "platform_height": 0.15,
    "platform_x": 2.0,
    "vx": 0.6,
}


def generate_jump_commands(
    num_envs: int,
    scenario_name: str,
    step_counter: int,
) -> np.ndarray:
    """生成跳跃命令 [vx, vy, vyaw, tsk, jump_trigger].

    测试时固定: vy=0, vyaw=0, tsk=0
    jump_trigger 按周期触发.
    """
    cmds = np.zeros((num_envs, 5), dtype=np.float64)

    if scenario_name in FIXED_DIST_SCENARIOS:
        cmds[:, 0] = FIXED_DIST_SCENARIOS[scenario_name]["vx"]
        cmds[:, 4] = 1.0 if (step_counter % JUMP_PERIOD) < JUMP_DUTY else 0.0

    elif scenario_name == "random":
        cmds[:, 0] = np.random.uniform(
            RANDOM_SCENARIO["vx_min"], RANDOM_SCENARIO["vx_max"], size=num_envs
        )
        cmds[:, 4] = 1.0 if (step_counter % JUMP_PERIOD) < JUMP_DUTY else 0.0

    elif scenario_name == "platform":
        cmds[:, 0] = PLATFORM_SCENARIO["vx"]
        cmds[:, 4] = 1.0 if (step_counter % JUMP_PERIOD) < JUMP_DUTY else 0.0

    return cmds


@dataclass
class JumpCycleRecord:
    """单次跳跃周期记录."""
    takeoff_x: float = 0.0
    landing_x: float = 0.0
    max_height: float = 0.0
    wheel_slip_at_landing: float = 0.0
    duration_steps: int = 0


@dataclass
class EvalResult:
    """单回合评估结果."""
    scenario: str
    survived: bool
    jump_cycles: list[JumpCycleRecord] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.jump_cycles) > 0 and self.survived

    @property
    def avg_jump_distance(self) -> float:
        if not self.jump_cycles:
            return 0.0
        return float(np.mean([abs(j.landing_x - j.takeoff_x) for j in self.jump_cycles]))

    @property
    def avg_jump_height(self) -> float:
        if not self.jump_cycles:
            return 0.0
        return float(np.mean([j.max_height for j in self.jump_cycles]))

    @property
    def avg_wheel_slip(self) -> float:
        if not self.jump_cycles:
            return float("inf")
        return float(np.mean([j.wheel_slip_at_landing for j in self.jump_cycles]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "survived": self.survived,
            "success": self.success,
            "num_jumps": len(self.jump_cycles),
            "avg_jump_distance": self.avg_jump_distance,
            "avg_jump_height": self.avg_jump_height,
            "avg_wheel_slip": self.avg_wheel_slip,
            "jump_details": [
                {
                    "distance": abs(j.landing_x - j.takeoff_x),
                    "max_height": j.max_height,
                    "wheel_slip": j.wheel_slip_at_landing,
                    "duration_steps": j.duration_steps,
                }
                for j in self.jump_cycles
            ],
        }
