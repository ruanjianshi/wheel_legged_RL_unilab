"""通用确定性 rollout 引擎 (评估核心).

职责:
  - build_env   : 按 conf yaml + registry 建环境 (无辅助, 确定性策略)
  - load_policy : 从 checkpoint 加载确定性 actor (复用 tools/xqrobotwl/verify_jump.load_actor)
  - run_steps   : 单段连续运行 (行走场景/跳跃脉冲), per_step_fn 逐行采集
  - run_episodes: 并行 episode 循环 (跌倒恢复/单腿), done env 冻结统计 (active mask)
  - collect_step: 逐行状态采集 → StepSample (与 dump_pose_data 26 列 CSV 对齐)

复用:
  - unilab.base.registry.make + conf/<algo>/task/<task>/mujoco.yaml (eval_fall_recovery 模式)
  - verify_jump.load_actor (hidden 列表 + model_state_dict 回退 + mlp 过滤)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent  # 仓库根
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _devlog.assess.metrics import StepSample, Trace  # noqa: E402
from _devlog.assess.tasks import TaskDef  # noqa: E402

# ── 数学工具 (四元数→欧拉、世界→本地, 与 dump_pose_data 同约定) ──────────────


def quat_to_euler(qwxyz: np.ndarray) -> np.ndarray:
    """MuJoCo 四元数 [w,x,y,z] → ZYX 内旋欧拉角 [roll, pitch, yaw] (rad)."""
    w, x, y, z = qwxyz
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def world_to_local(quats: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """世界系向量 → base 本地系: v_local = R^T @ v_world."""
    w, x, y, z = quats.T
    rot = np.zeros((quats.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 0, 2] = 2 * (x * z + w * y)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 2, 1] = 2 * (y * z + w * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return np.einsum("nji,nj->ni", rot, vecs)


# ── 环境构建 ────────────────────────────────────────────────────────────────


def load_conf(task: TaskDef) -> dict:
    """加载任务 conf mujoco.yaml."""
    path = ROOT / task.cfg_path
    if not path.exists():
        raise FileNotFoundError(f"任务配置不存在: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def build_env(
    task: TaskDef,
    num_envs: int = 1,
    pose: int | None = None,
    cfg_override: dict | None = None,
    ckpt_path: str | Path | None = None,
):
    """按 conf + registry 建环境 (无辅助力, 确定性策略评估).

    配置还原优先级:
      1. ckpt 相邻的 run_config.json (trained_env_overrides) — 评估即训练配置 (最忠实)
      2. conf/<algo>/task/<task>/mujoco.yaml — 无训练快照时的当前任务配置
    pose: 仅 fall_recovery 有效 (0仰卧/1俯卧/2左躺/3右躺).
    """
    from unilab.base import registry
    from unilab.base.registry import ensure_registries

    ensure_registries()

    override: dict | None = None
    if ckpt_path is not None:
        from tools.xqrobotwl.verify_jump import trained_env_overrides

        trained = trained_env_overrides(str(ckpt_path))
        if trained is not None:
            override = dict(trained)
    if override is None:
        cfg = load_conf(task)
        override = {"reward_config": cfg["reward"]}
        override.update(cfg.get("env", {}))

    rc = override.get("reward_config")
    if isinstance(rc, dict) and "force_assist_enabled" in rc:
        rc["force_assist_enabled"] = False  # 评估真实姿态 (无辅助力)
    if cfg_override:
        override.update(cfg_override)

    env = registry.make(
        task.env_name,
        sim_backend="mujoco",
        num_envs=num_envs,
        env_cfg_override=override,
    )
    if pose is not None:
        from tools.xqrobotwl.eval_fall_recovery import _set_fixed_pose_provider

        _set_fixed_pose_provider(env, pose)
    return env


def load_policy(checkpoint_path: str | Path, obs_dim: int, num_actions: int = 8):
    """从 checkpoint 加载确定性 actor (复用 verify_jump.load_actor)."""
    from tools.xqrobotwl.verify_jump import load_actor

    hidden = [512, 512, 256, 128]  # rsl_rl 默认 MLP 结构
    return load_actor(str(checkpoint_path), obs_dim, num_actions, hidden)


# ── run / checkpoint 路径解析 ───────────────────────────────────────────────


def resolve_run_dir(run_arg: str, log_root: str) -> Path:
    """--run 接受完整/相对路径, 或 log_root 下的目录名."""
    p = Path(run_arg)
    if p.is_absolute() or p.exists():
        return p
    cand = ROOT / log_root / run_arg
    if cand.is_dir():
        return cand
    raise SystemExit(f"找不到 run 目录: {run_arg} (在 {log_root}/ 下查找)")


def find_checkpoint(run_dir: Path, ckpt: str | None) -> Path:
    """checkpoint: 显式文件名或默认最新 model_*.pt."""
    if ckpt:
        p = run_dir / ckpt
        if not p.exists():
            raise SystemExit(f"无 checkpoint: {p}")
        return p
    ckpts = sorted(run_dir.glob("model_*.pt"))
    if not ckpts:
        raise SystemExit(f"无 checkpoint: {run_dir}")
    return ckpts[-1]


# ── 逐行采集 ────────────────────────────────────────────────────────────────


def collect_step(env, st, step: int, env_idx: int, ctrl_dt: float = 0.01) -> StepSample:
    """采集单个 env 的一步姿态数据 (与 dump_pose_data 列对齐)."""
    backend = env._backend
    quat = np.asarray(backend.get_base_quat(), dtype=np.float64)[env_idx]
    euler = quat_to_euler(quat)
    base_pos = np.asarray(backend.get_base_pos(), dtype=np.float64)[env_idx]
    linvel = world_to_local(
        quat[None, :], np.asarray(backend.get_base_lin_vel(), dtype=np.float64)
    )[0]
    gyro = world_to_local(quat[None, :], np.asarray(backend.get_base_ang_vel(), dtype=np.float64))[
        0
    ]
    dof_pos = np.asarray(backend.get_dof_pos(), dtype=np.float64)[env_idx]
    dof_vel = np.asarray(backend.get_dof_vel(), dtype=np.float64)[env_idx]
    up_z = float(np.asarray(backend.get_sensor_data("upvector"), dtype=np.float64)[env_idx, 2])
    wc = st.info.get("wheel_contact", np.zeros((env.num_envs, 2)))[env_idx]
    recovered = st.info.get("recover_completed", np.zeros(env.num_envs, dtype=bool))[env_idx]
    cmd = st.info.get("commands", np.zeros((env.num_envs, 5), dtype=np.float64))[env_idx]
    return StepSample(
        step=step,
        time_s=step * ctrl_dt,
        dof_pos=dof_pos[0:6].copy(),
        euler=euler,
        base_pos=base_pos,
        linvel=linvel,
        gyro=gyro,
        wheel_vel=dof_vel[6:8].copy(),
        up_z=up_z,
        wheel_contact=wc.astype(np.float64).copy(),
        recover_completed=float(recovered),
        cmd=np.asarray(cmd, dtype=np.float64).copy(),
    )


# ── rollout 循环 ─────────────────────────────────────────────────────────────


def run_steps(
    env,
    policy,
    steps: int,
    per_step_fn: Callable,
    ctrl_dt: float | None = None,
) -> None:
    """单段连续运行 steps 步; per_step_fn(step, env, st, action) 每步回调.

    用于行走场景 / 跳跃脉冲 / 后空翻触发 — 需任务自定义命令注入的用这个。
    """
    del ctrl_dt  # 周期由 env._cfg.ctrl_dt 提供
    obs, _ = env.reset(np.arange(env.num_envs))
    with torch.no_grad():
        for step in range(steps):
            obs_t = torch.tensor(obs["obs"], dtype=torch.float32)
            action = policy(obs_t).numpy().astype(np.float64)
            st = env.step(action)
            per_step_fn(step, env, st, action)
            obs = st.obs


def run_cmd_scenario(
    env,
    policy,
    cmd: list[float],
    duration: float,
    warmup: float,
    ctrl_dt: float | None = None,
) -> list[Trace]:
    """跑一个命令场景: init_state → 每步注入 5D 命令 → 跑 (warmup+duration) 步.

    仅统计 warmup 之后的测量窗口; 返回每 env 一条 Trace (命令恒定, §7.0 追踪场景).
    采用 verify_jump 的命令注入模式 (env.state 需先 init_state).
    """
    dt = ctrl_dt if ctrl_dt is not None else env._cfg.ctrl_dt
    num_envs = env.num_envs
    env.init_state()  # 首次 reset, 初始化 NpEnvState
    traces: list[Trace] = [[] for _ in range(num_envs)]
    start = int(warmup / dt)
    total = int((duration + warmup) / dt)
    with torch.no_grad():
        for step in range(total):
            env.state.info["commands"][:, :5] = np.asarray(cmd, dtype=np.float64)
            obs_t = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
            a = policy(obs_t).numpy().astype(np.float64)
            st = env.step(a)
            if step >= start:
                for i in range(num_envs):
                    traces[i].append(collect_step(env, st, step, i, dt))
    return traces


def run_episodes(
    env,
    policy,
    num_envs: int,
    max_steps: int,
    collect_fn: Callable,
    ctrl_dt: float | None = None,
) -> list[Trace]:
    """并行 episode 循环: done env 冻结统计 (active mask), env 自动 reset.

    collect_fn(env, st, step, env_idx, action) -> StepSample | None
    返回每 env 一条 Trace (样本直到该 env done 或 max_steps).
    """
    dt = ctrl_dt if ctrl_dt is not None else env._cfg.ctrl_dt
    traces: list[list[StepSample]] = [[] for _ in range(num_envs)]
    obs, _ = env.reset(np.arange(num_envs))
    active = np.ones(num_envs, dtype=bool)
    with torch.no_grad():
        for step in range(max_steps):
            obs_t = torch.tensor(obs["obs"], dtype=torch.float32)
            action = policy(obs_t).numpy().astype(np.float64)
            st = env.step(action)
            for i in range(num_envs):
                if active[i]:
                    sample = collect_fn(env, st, step, i, action[i], dt)
                    if sample is not None:
                        traces[i].append(sample)
            done = st.terminated | st.truncated
            if done.any():
                active &= ~done
                if not active.any():
                    break
            obs = st.obs
    return traces
