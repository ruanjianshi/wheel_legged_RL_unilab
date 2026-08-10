"""xqrobotV2 stairs-only terrain env: 100% pyramid stairs + terrain curriculum."""

from __future__ import annotations

from dataclasses import dataclass, field

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.envs.locomotion.common.height_scan import HeightScanConfig
from unilab.envs.locomotion.common.terrain_spawn import TerrainCurriculumCfg
from unilab.terrains import SubTerrainCfg, TerrainGeneratorCfg, pyramid_stairs, pyramid_stairs_inv

from .joystick import XqRobotV2WalkFlatCfg
from .rough import (
    RoughTerminationConfig,
    XqRobotRoughCommands,
    XqRobotV2WalkRoughEnv,
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


@registry.envcfg("XqRobotV2Stairs")
@dataclass
class XqRobotV2StairsCfg(XqRobotV2WalkFlatCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "xqrobotV2" / "xqrobotV2.xml"),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "xqrobotV2" / "locomotion_task.xml"),
            ],
            terrain=TerrainSceneCfg(
                generator=StairsOnlyTerrainCfg(),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    commands: XqRobotRoughCommands = field(default_factory=XqRobotRoughCommands)
    terrain_scan: HeightScanConfig = field(default_factory=HeightScanConfig)
    termination_config: RoughTerminationConfig = field(default_factory=RoughTerminationConfig)
    terrain_curriculum: TerrainCurriculumCfg = field(
        default_factory=lambda: TerrainCurriculumCfg(
            enabled=True, promote_frac=0.5, demote_frac=0.25, cycle_top_frac=0.5
        )
    )


@registry.env("XqRobotV2Stairs", sim_backend="mujoco")
class XqRobotV2StairsEnv(XqRobotV2WalkRoughEnv):
    _cfg: XqRobotV2StairsCfg  # type: ignore[assignment]  # stairs cfg 继承 walk_flat, 基类 env 需 walk_rough cfg (运行时兼容)
