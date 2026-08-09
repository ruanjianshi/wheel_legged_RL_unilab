"""xqrobotwl stairs-only terrain env: 100% pyramid stairs + terrain curriculum."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.envs.locomotion.common.height_scan import HeightScanConfig
from unilab.envs.locomotion.common.terrain_spawn import TerrainCurriculumCfg
from unilab.terrains import SubTerrainCfg, TerrainGeneratorCfg, pyramid_stairs, pyramid_stairs_inv

from .joystick import XqRobotWLWalkFlatCfg
from .rough import (
    RoughTerminationConfig,
    XqRobotWLRoughCommands,
    XqRobotWLWalkRoughEnv,
)


@dataclass(kw_only=True)
class StairsOnlyTerrainCfg(TerrainGeneratorCfg):
    size: tuple[float, float] = (8.0, 8.0)
    num_rows: int = 8
    num_cols: int = 4
    border_width: float = 20.0
    horizontal_scale: float = 0.05

    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            "pyramid_stairs": pyramid_stairs(
                proportion=0.5,
                step_height_range=(0.03, 0.11),
                step_width=0.55,
                platform_width=3.0,
                border_width=0.2,
            ),
            "pyramid_stairs_inv": pyramid_stairs_inv(
                proportion=0.5,
                step_height_range=(0.03, 0.11),
                step_width=0.55,
                platform_width=3.0,
                border_width=0.2,
            ),
        }
    )


@registry.envcfg("XqRobotWLStairs")
@dataclass
class XqRobotWLStairsCfg(XqRobotWLWalkFlatCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "xqrobotwl" / "xqrobotwl.xml"),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "xqrobotwl" / "locomotion_task.xml"),
            ],
            terrain=TerrainSceneCfg(
                generator=StairsOnlyTerrainCfg(),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    commands: XqRobotWLRoughCommands = field(default_factory=XqRobotWLRoughCommands)
    terrain_scan: HeightScanConfig = field(default_factory=HeightScanConfig)
    termination_config: RoughTerminationConfig = field(default_factory=RoughTerminationConfig)
    terrain_curriculum: TerrainCurriculumCfg = field(
        default_factory=lambda: TerrainCurriculumCfg(
            enabled=True, promote_frac=0.5, demote_frac=0.25, cycle_top_frac=0.5
        )
    )


@registry.env("XqRobotWLStairs", sim_backend="mujoco")
class XqRobotWLStairsEnv(XqRobotWLWalkRoughEnv):
    _cfg: XqRobotWLStairsCfg

    def __init__(self, cfg: XqRobotWLStairsCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        # Stairs uses 5D commands [vx, vy, vyaw, tsk, height] (see task YAML), so
        # obs/critic frames match flat (5D) rather than rough (4D).
        self._obs_frame_dim = 33  # 5D cmd: gyro(3)+grav(3)+diff(6)+vel(6)+wheel(2)+act(8)+cmd(5)
        self._critic_frame_dim = 36  # 5D cmd + linvel(3)
        self._obs_history = np.zeros(
            (num_envs, self._hist_len, self._obs_frame_dim), dtype=self._np_dtype
        )
        self._critic_history = np.zeros(
            (num_envs, self._hist_len, self._critic_frame_dim), dtype=self._np_dtype
        )
