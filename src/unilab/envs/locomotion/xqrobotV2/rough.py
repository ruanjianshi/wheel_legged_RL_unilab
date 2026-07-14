"""xqrobotV2 崎岖地形环境 — 程序化地形 + 高度扫描 + 出界终止

继承平地行走环境 (joystick.py), 在以下方面有所区别:
- 场景: xqrobotV2.xml + locomotion_task.xml (无 scene_flat)
- 地形: 6×6 网格, 8×8m 单元格, 5 种地形混合
- 命令: vx[-1,1] vyaw[-1.5,1.5], 10s 重采样, 不解耦
- 观测: critic 附加高度扫描
- 终止: 无 base_height 检查 (地形高度变化), 替代为地形出界
- DR: 不解耦, 全 5D 随机命令
"""
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


# ═══ 粗糙地形命令配置 ═══
# 比平地范围更大: vx[-1, +1], vyaw[-1.5, +1.5], 高度[0.40, 0.90]
# 重采样间隔 10 秒 (需要更长时间来穿越地形)


@dataclass
class XqRobotRoughCommands(Commands):
    vel_limit: list[list[float]] = field(
        default_factory=lambda: [[-1.0, -0.5, -1.5, -0.1, 0.40], [1.0, 0.5, 1.5, 0.1, 0.90]]
    )
    resampling_time: float = 10.0


@dataclass
class RoughTerminationConfig:
    """粗糙地形终止配置 — 无 base_height 检查 (地面高度变化)"""
    terrain_out_of_bounds: bool = True         # 跑到地形外缘 → 终止
    terrain_distance_buffer: float = 3.0       # 离边界 3m 内即算越界


# ═══ 程序化地形生成器 ═══


@dataclass(kw_only=True)
class XqRobotRoughTerrainCfg(TerrainGeneratorCfg):
    """地形混合配方 — 6×6 网格, 5 种地形类型混搭

    配比 (总和 = 1.0):
    - flat:               0%   (平地已由 walk_flat 覆盖)
    - pyramid_stairs:    15%   (上楼梯 — step 2-8cm)
    - pyramid_stairs_inv:15%   (下楼梯 — 同参数)
    - random_rough:      30%   (随机粗糙 — 噪声 1-6cm)
    - wave_terrain:      30%   (波浪 — 振幅 0-10cm, 4 个波)
    - hf_pyramid_slope:   5%   (上坡 — 坡度 0-15%)
    - hf_pyramid_slope_inv:5%  (下坡 — 同参数)
    """
    size: tuple[float, float] = (8.0, 8.0)   # 每个单元格 8×8m
    num_rows: int = 6
    num_cols: int = 6
    border_width: float = 20.0               # 边界宽度 (平坦区域)
    horizontal_scale: float = 0.1            # 水平分辨率 10cm

    sub_terrains: dict[str, SubTerrainCfg] = field(
        default_factory=lambda: {
            "flat": flat(proportion=0.2),
            "random_rough": random_rough(
                proportion=0.35,
                noise_range=(0.005, 0.04),
                noise_step=0.01,
                border_width=0.2,
            ),
            "wave_terrain": wave_terrain(
                proportion=0.35,
                amplitude_range=(0.0, 0.12),
                num_waves=4,
                border_width=0.2,
            ),
            "hf_pyramid_slope": hf_pyramid_slope(
                proportion=0.05,
                slope_range=(0.1, 0.35),
                platform_width=2.0,
                border_width=0.2,
            ),
            "hf_pyramid_slope_inv": hf_pyramid_slope_inv(
                proportion=0.05,
                slope_range=(0.1, 0.35),
                platform_width=2.0,
                border_width=0.2,
            ),
        }
    )


@registry.envcfg("XqRobotV2WalkRough")
@dataclass
class XqRobotV2WalkRoughCfg(XqRobotV2WalkFlatCfg):
    """粗糙地形任务配置 — 使用独立 robot.xml + fragment (不含 scene_flat 的地面)"""
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "xqrobotV2" / "xqrobotV2.xml"),
            fragment_files=[
                str(ASSETS_ROOT_PATH / "robots" / "xqrobotV2" / "locomotion_task.xml"),
            ],
            terrain=TerrainSceneCfg(
                generator=XqRobotRoughTerrainCfg(),
                hfield_name="terrain_hfield",    # hfield 名称 — 用于高度查询
                geom_name="floor",               # 地面 geom — 用于渲染
            ),
        )
    )
    commands: XqRobotRoughCommands = field(default_factory=XqRobotRoughCommands)
    terrain_scan: HeightScanConfig = field(default_factory=HeightScanConfig)       # 地形扫描传感器配置
    termination_config: RoughTerminationConfig = field(default_factory=RoughTerminationConfig)
    terrain_curriculum: TerrainCurriculumCfg = field(default_factory=TerrainCurriculumCfg)  # 地形难度课程


class XqRobotRoughDRProvider(XqRobotDRProvider):
    """粗糙地形 DR — 与平地 DR 的唯一区别: 不解耦命令 (不 zero-out Vx/Vy)

    粗糙地形不需要解耦训练, 因为地形本身已经提供了足够的泛化挑战
    """
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
        return cmds  # 不解耦: Vx 和 Vy 同时激活


@registry.env("XqRobotV2WalkRough", sim_backend="mujoco")
class XqRobotV2WalkRoughEnv(XqRobotV2WalkFlatEnv):
    """粗糙地形行走环境

    关键差异 vs 平地:
    1. 初始化 TerrainSpawnManager — 负责根据地形分配机器人初始位置
    2. 初始化高度扫描传感器 — 为 critic 提供地形感知
    3. obs_groups_spec 在 critic 中附加 height_scan_dim
    4. _base_height_values 用 base_height_from_scan (从 hfield 查高度, 非固定 0)
    5. _compute_terminated 无 base_height 检查 (地形高度变化, 固定阈值无意义)
    6. _compute_truncated 增加 terrain_out_of_bounds (跑到边界外 → truncate)
    """
    _cfg: XqRobotV2WalkRoughCfg
    _height_scan_dim: int = 0

    def __init__(self, cfg: XqRobotV2WalkRoughCfg, num_envs=1, backend_type="mujoco"):
        super().__init__(cfg, num_envs=num_envs, backend_type=backend_type)
        # 地形生成管理器: 分配各 env 到不同的地形单元格
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
        # DR Manager 用 Rough 专用 Provider (不解耦命令)
        self._dr_manager = DomainRandomizationManager(self, XqRobotRoughDRProvider())
        # 初始化高度扫描: 在 base_link 上安装射线传感器
        init_height_scan_sensor(self, cfg.terrain_scan, cfg.asset.base_name)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        """critic 附加 terrain_scan 维度 — 让 critic 知道前方地形"""
        base = super().obs_groups_spec
        base["critic"] = base["critic"] + self._height_scan_dim
        return base

    def _base_height_values(self, num_obs: int) -> np.ndarray:
        """地形模式: base_height = 传感器到地形表面的距离 (非世界 Z 坐标)"""
        height = base_height_from_scan(self, num_obs)
        if height.shape[0] != num_obs:
            return super()._base_height_values(num_obs)
        return height

    def _compute_obs(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> dict[str, np.ndarray]:
        """粗糙地形观测 — 与平地相同 + critic 附加高度扫描"""
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

        obs_frame = np.concatenate(
            [
                noisy_gyro,
                noisy_gravity,
                noisy_leg_diff,
                noisy_leg_vel,
                noisy_wheel_vel,
                last_actions,
                info["commands"],
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

        critic_frame = np.concatenate(
            [
                gyro,
                -gravity,
                leg_diff,
                leg_vel,
                wheel_vel,
                last_actions,
                info["commands"],
                linvel,
            ],
            axis=1,
            dtype=get_global_dtype(),
        )

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
        # 附加地形高度扫描到 critic (actor 不感知地形 — 它必须学会鲁棒步态)
        critic = np.concatenate(
            [critic_base, height_scan_obs(self, self._cfg.terrain_scan, num_obs)],
            axis=1,
            dtype=get_global_dtype(),
        )
        return {"obs": obs, "critic": critic}

    def update_state(self, state: NpEnvState) -> NpEnvState:
        from unilab.base.np_env import NpEnvState

        state = super().update_state(state)
        if hasattr(self, "_spawn") and self._cfg.terrain_curriculum.enabled:
            terminated_ids = np.where(state.terminated)[0]
            if len(terminated_ids) > 0:
                base_pos = np.asarray(self._backend.get_base_pos(), dtype=get_global_dtype())
                self._spawn.update_on_done(terminated_ids, base_pos[terminated_ids])
        return state

    def _compute_terminated(self, gravity: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        """粗糙地形终止条件 — 无 base_height 检查

        地形表面高度变化大, 固定高度阈值无意义。
        保留: 倾角 + 大腿塌陷 + 小腿极限
        """
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        max_tilt = np.deg2rad(self._reward_cfg.max_tilt_deg)
        terminated = tilt > max_tilt
        thigh_collapsed = (dof_pos[:, 1] < 0.02) | (dof_pos[:, 4] < 0.02)
        calf_extreme = (np.abs(dof_pos[:, 2]) > 0.85) | (np.abs(dof_pos[:, 5]) > 0.85)
        terminated |= thigh_collapsed
        terminated |= calf_extreme
        return terminated

    def _compute_truncated(self, state: NpEnvState) -> np.ndarray:
        """检查是否跑出地形区域 — 3m buffer 外视为越界"""
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
