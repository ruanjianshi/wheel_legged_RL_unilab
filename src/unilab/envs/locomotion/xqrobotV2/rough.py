"""xqrobotV2 rough terrain env: procedural terrain + height scan + terrain termination."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg, TerrainSceneCfg
from unilab.dr import DomainRandomizationManager
from unilab.dr.dr_utils import zero_actions
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_quat_mul, np_yaw_to_quat
from unilab.envs.locomotion.common.commands import Commands
from unilab.envs.locomotion.common.height_scan import (
    HeightScanConfig,
    base_height_from_scan,
    height_scan_obs,
    init_height_scan_sensor,
    terrain_out_of_bounds,
)
from unilab.envs.locomotion.common.terrain_spawn import (
    TerrainCurriculumCfg,
    TerrainSpawnManager,
)
from unilab.terrains import (
    SubTerrainCfg,
    TerrainGeneratorCfg,
    flat,
    hf_pyramid_slope,
    hf_pyramid_slope_inv,
    pyramid_stairs,
    pyramid_stairs_inv,
    random_rough,
    wave_terrain,
)

from .base import NUM_ACTIONS, NUM_LEG_ACTIONS
from .joystick import (
    XqRobotDRProvider,
    XqRobotV2WalkFlatCfg,
    XqRobotV2WalkFlatEnv,
)

_HISTORY_LEN = 9


@dataclass
class XqRobotRoughCommands(Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-1.0, -0.5, -1.5, -0.1, 0.40], [1.0, 0.5, 1.5, 0.1, 0.90]]
    )
    resampling_time: float = 10.0


@dataclass
class RoughTerminationConfig:
    terrain_out_of_bounds: bool = True
    terrain_distance_buffer: float = 3.0


@dataclass(kw_only=True)
class XqRobotRoughTerrainCfg(TerrainGeneratorCfg):
    size: tuple[float, float] = (8.0, 8.0)
    num_rows: int = 6
    num_cols: int = 6
    border_width: float = 20.0
    horizontal_scale: float = 0.1

    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            "flat": flat(proportion=0.0),
            "random_rough": random_rough(
                proportion=0.4,
                noise_range=(0.01, 0.06),
                noise_step=0.01,
                border_width=0.2,
            ),
            "wave_terrain": wave_terrain(
                proportion=0.4,
                amplitude_range=(0.0, 0.10),
                num_waves=4,
                border_width=0.2,
            ),
            "hf_pyramid_slope": hf_pyramid_slope(
                proportion=0.1,
                slope_range=(0.0, 0.15),
                platform_width=2.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
                proportion=0.1,
                slope_range=(0.0, 0.15),
                platform_width=2.0,
                border_width=0.2,
            ),
        }
    )


@registry.envcfg("XqRobotV2WalkRough")
@dataclass
class XqRobotV2WalkRoughCfg(XqRobotV2WalkFlatCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "xqrobotV2" / "xqrobotV2.xml"),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "xqrobotV2" / "locomotion_task.xml"),
            ],
            terrain=TerrainSceneCfg(
                generator=XqRobotRoughTerrainCfg(),
                hfield_name="terrain_hfield",
                geom_name="floor",
            ),
        )
    )
    commands: XqRobotRoughCommands = field(default_factory=XqRobotRoughCommands)
    terrain_scan: HeightScanConfig = field(default_factory=HeightScanConfig)
    termination_config: RoughTerminationConfig = field(default_factory=RoughTerminationConfig)
    terrain_curriculum: TerrainCurriculumCfg = field(default_factory=TerrainCurriculumCfg)


class XqRobotRoughDRProvider(XqRobotDRProvider):
    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        low = np.asarray(env._cfg.commands.vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(env._cfg.commands.vel_limit[1], dtype=get_global_dtype())
        cmds = np.asarray(
            np.random.uniform(low=low, high=high, size=(num_reset, low.shape[0])),
            dtype=get_global_dtype(),
        )
        safe_linv = np.maximum(np.abs(cmds[:, 0]), 1e-4)
        angv_limit = 2.0 / safe_linv
        cmds[:, 2] = np.clip(cmds[:, 2], -angv_limit, angv_limit)
        return cmds


@registry.env("XqRobotV2WalkRough", sim_backend="mujoco")
class XqRobotV2WalkRoughEnv(XqRobotV2WalkFlatEnv):
    _cfg: XqRobotV2WalkRoughCfg
    _height_scan_dim: int = 0

    def __init__(self, cfg: XqRobotV2WalkRoughCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        terrain_origins = getattr(self._backend, "terrain_origins", None)
        terrain_generator = cfg.scene.terrain.generator if cfg.scene.terrain is not None else None
        if terrain_origins is not None and terrain_generator is not None:
            self._spawn = TerrainSpawnManager(
                num_envs,
                terrain_origins,
                cell_size=float(terrain_generator.size[0]),
                cfg=cfg.terrain_curriculum,
                terrain_surface_sampler=getattr(self._backend, "terrain_surface_sampler", None),
            )
        self._dr_manager = DomainRandomizationManager(self, XqRobotRoughDRProvider())
        init_height_scan_sensor(self, cfg.terrain_scan, cfg.asset.base_name)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        base = super().obs_groups_spec
        base["critic"] = base["critic"] + self._height_scan_dim
        return base

    def _base_height_values(self, num_obs: int) -> np.ndarray:
        height = base_height_from_scan(self, num_obs)
        if height.shape[0] != num_obs:
            return super()._base_height_values(num_obs)
        return height

    def _compute_obs(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        leg_diff = dof_pos[:, :NUM_LEG_ACTIONS] - self.default_angles[:NUM_LEG_ACTIONS]
        leg_vel = dof_vel[:, :NUM_LEG_ACTIONS]
        wheel_vel = dof_vel[:, NUM_LEG_ACTIONS:]
        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(-gravity, noise_cfg.scale_gravity)
        noisy_leg_diff = self._obs_noise(leg_diff, noise_cfg.scale_joint_angle)
        noisy_leg_vel = self._obs_noise(leg_vel, noise_cfg.scale_joint_vel)
        noisy_wheel_vel = self._obs_noise(wheel_vel, noise_cfg.scale_wheel_vel)
        last_actions = info.get("current_actions", np.zeros((linvel.shape[0], NUM_ACTIONS)))

        obs_frame = np.concatenate([
            noisy_gyro, noisy_gravity,
            noisy_leg_diff, noisy_leg_vel, noisy_wheel_vel,
            last_actions, info["commands"],
        ], axis=1, dtype=get_global_dtype())

        critic_frame = np.concatenate([
            gyro, -gravity,
            leg_diff, leg_vel, wheel_vel,
            last_actions, info["commands"], linvel,
        ], axis=1, dtype=get_global_dtype())

        batch_size = obs_frame.shape[0]
        steps_val = int(info.get("steps", np.zeros(1, dtype=np.uint32))[0])

        if steps_val <= 1:
            for i in range(self._hist_len):
                self._obs_history[:batch_size, i, :] = obs_frame
                self._critic_history[:batch_size, i, :] = critic_frame
        else:
            self._obs_history[:batch_size, :-1, :] = self._obs_history[:batch_size, 1:, :]
            self._obs_history[:batch_size, -1, :] = obs_frame
            self._critic_history[:batch_size, :-1, :] = self._critic_history[:batch_size, 1:, :]
            self._critic_history[:batch_size, -1, :] = critic_frame

        num_obs = linvel.shape[0]
        obs = self._obs_history[:batch_size].reshape(batch_size, -1)
        critic_base = self._critic_history[:batch_size].reshape(batch_size, -1)
        critic = np.concatenate(
            [critic_base, height_scan_obs(self, self._cfg.terrain_scan, num_obs)],
            axis=1, dtype=get_global_dtype(),
        )
        return {"obs": obs, "critic": critic}

    def _compute_terminated(self, gravity: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        max_tilt = np.deg2rad(self._reward_cfg.max_tilt_deg)
        terminated = tilt > max_tilt
        thigh_collapsed = (dof_pos[:, 1] < 0.02) | (dof_pos[:, 4] < 0.02)
        calf_extreme = (np.abs(dof_pos[:, 2]) > 0.85) | (np.abs(dof_pos[:, 5]) > 0.85)
        terminated |= thigh_collapsed
        terminated |= calf_extreme
        return terminated

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        truncated = super()._compute_truncated(state)
        if self._cfg.termination_config.terrain_out_of_bounds:
            terrain_scene = self._cfg.scene.terrain
            terrain_cfg = terrain_scene.generator if terrain_scene is not None else None
            np.logical_or(
                truncated,
                terrain_out_of_bounds(
                    self,
                    terrain_cfg,
                    float(self._cfg.termination_config.terrain_distance_buffer),
                ),
                out=truncated,
            )
        return truncated
