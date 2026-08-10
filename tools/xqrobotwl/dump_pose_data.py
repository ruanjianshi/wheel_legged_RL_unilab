"""导出评估 checkpoint 的每步完整姿态数据 CSV(数据优先,供反推姿态)。

规范 CLAUDE.md §1.5.1: 每个评估通过的 checkpoint 导出每一步的完整姿态数据,
视频/图只是展示, 数据 (CSV) 才是评估依据。

用法:
  uv run mjpython tools/xqrobotwl/dump_pose_data.py \
      --run <run_dir> --ckpt model_4000.pt \
      [--pose 0-3] [--steps 900] [--prefix <任务名>]

输出: logs/pose_data/<prefix>_model_XXX.pt_<姿态>.csv
数据保留两位小数 (用户约定); step/接触/恢复锁存为整数列。

CSV 列 (每步一行):
  step / time_s
  L/R_hip_roll_rad, L/R_hip_pitch_rad, L/R_knee_rad   6 腿部关节角度
  base_roll/pitch/yaw_rad                              base 欧拉角
  base_x/y/z                                          base 位置
  linvel_x/y/z                                        本地线速度
  gyro_x/y/z                                          本地角速度
  wheel_L/R_rads                                      轮子角速度
  up_z                                                直立度
  wheel_contact_L/R                                   轮地接触
  recover_completed                                   恢复完成锁存

当前支持 fall_recovery 任务的 checkpoint (与评估脚本同构); 姿态列名与
§1.5.2 反推示例一致 (L_hip_pitch_rad / base_z / up_z / base_yaw_rad)。
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.xqrobotwl.eval_fall_recovery import (  # noqa: E402
    POSES,
    ActorMLP,
    _set_fixed_pose_provider,
)
from unilab.base import registry
from unilab.base.registry import ensure_registries

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml"
RUN_ROOT = ROOT / "logs/rsl_rl_cpo/XqRobotWLFallRecoveryFlat"
OUT_DIR = ROOT / "logs/pose_data"

# CSV 列名 (与规范 §1.5.2 反推示例一致: 角度列带 _rad 后缀)
COLUMNS = [
    "step",
    "time_s",
    "L_hip_roll_rad",
    "L_hip_pitch_rad",
    "L_knee_rad",
    "R_hip_roll_rad",
    "R_hip_pitch_rad",
    "R_knee_rad",
    "base_roll_rad",
    "base_pitch_rad",
    "base_yaw_rad",
    "base_x",
    "base_y",
    "base_z",
    "linvel_x",
    "linvel_y",
    "linvel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "wheel_L_rads",
    "wheel_R_rads",
    "up_z",
    "wheel_contact_L",
    "wheel_contact_R",
    "recover_completed",
]

WXYZ = (0, 1, 2, 3)  # MuJoCo xquat 分量索引 [w, x, y, z]


def _quat_to_euler(qwxyz: np.ndarray) -> np.ndarray:
    """MuJoCo 四元数 [w,x,y,z] → ZYX 内旋欧拉角 [roll, pitch, yaw] (rad).

    与 fall_recovery._quat_from_euler 的约定互为逆变换。
    """
    w, x, y, z = qwxyz
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def _world_to_local(quats: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """把世界系向量转到 base 本地系: v_local = R^T @ v_world.

    quats: (N, 4) MuJoCo [w,x,y,z]; vecs: (N, 3) 世界系.
    """
    w, x, y, z = quats.T
    # R 的列为 body 轴在世界系的表达 → R^T 把世界向量转到 body 系
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


def _resolve_run_dir(run_arg: str) -> Path:
    """--run 接受完整/相对路径, 或直接给 run 目录名 (在 RUN_ROOT 下查找)."""
    p = Path(run_arg)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    cand = RUN_ROOT / run_arg
    if cand.is_dir():
        return cand
    raise SystemExit(f"找不到 run 目录: {run_arg} (相对路径取 {RUN_ROOT})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run",
        type=str,
        required=True,
        help="run 目录 (路径或 logs/rsl_rl_cpo/XqRobotWLFallRecoveryFlat/ 下的目录名)",
    )
    ap.add_argument(
        "--ckpt", type=str, default=None, help="checkpoint 文件名 (默认最新 model_*.pt)"
    )
    ap.add_argument("--pose", type=int, default=-1, help="固定倒地姿态 0-3, -1=随机")
    ap.add_argument("--steps", type=int, default=900, help="导出的步数 (默认 900, 每步一行)")
    ap.add_argument(
        "--prefix", type=str, default="fall_recovery", help="CSV 文件名前缀, 默认任务名"
    )
    args = ap.parse_args()

    ensure_registries()
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    override = {"reward_config": cfg["reward"]}
    override["reward_config"]["force_assist_enabled"] = False  # 无辅助力, 真实姿态
    override.update(cfg.get("env", {}))

    env = registry.make(
        "XqRobotWLFallRecoveryFlat",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override=override,
    )
    if args.pose in (0, 1, 2, 3):
        _set_fixed_pose_provider(env, args.pose)

    run_dir = _resolve_run_dir(args.run)
    if args.ckpt:
        ckpt_path = run_dir / args.ckpt
    else:
        ckpts = sorted(run_dir.glob("model_*.pt"))
        if not ckpts:
            raise SystemExit(f"无 checkpoint: {run_dir}")
        ckpt_path = ckpts[-1]

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pol = ActorMLP(297)
    pol.load_state_dict({k: v for k, v in ckpt["actor_state_dict"].items() if k.startswith("mlp.")})
    pol.eval()

    obs, _ = env.reset(np.arange(1))
    dt = env._cfg.ctrl_dt
    backend = env._backend

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pose_tag = POSES[args.pose] if args.pose in (0, 1, 2, 3) else "随机"
    out_csv = OUT_DIR / f"{args.prefix}_{ckpt_path.stem}_{pose_tag}.csv"

    rows: list[list[float]] = []
    for step in range(args.steps):
        obs_t = torch.tensor(obs["obs"], dtype=torch.float32)
        with torch.no_grad():
            a = pol(obs_t).numpy().astype(np.float64)
        st = env.step(a)

        quat = np.asarray(backend.get_base_quat(), dtype=np.float64)[0]
        euler = _quat_to_euler(quat)
        linvel = _world_to_local(
            quat[None, :], np.asarray(backend.get_base_lin_vel(), dtype=np.float64)
        )[0]
        gyro = _world_to_local(
            quat[None, :], np.asarray(backend.get_base_ang_vel(), dtype=np.float64)
        )[0]
        dof_pos = np.asarray(backend.get_dof_pos(), dtype=np.float64)[0]
        dof_vel = np.asarray(backend.get_dof_vel(), dtype=np.float64)[0]
        up_z = float(np.asarray(backend.get_sensor_data("upvector"), dtype=np.float64)[0, 2])
        wc = st.info.get("wheel_contact", np.zeros((env.num_envs, 2)))[0]
        recovered = st.info.get("recover_completed", np.zeros(env.num_envs, dtype=bool))[0]

        rows.append(
            [
                step,
                round(step * dt, 6),
                *dof_pos[0:6].tolist(),
                *euler.tolist(),
                *np.asarray(backend.get_base_pos(), dtype=np.float64)[0].tolist(),
                *linvel.tolist(),
                *gyro.tolist(),
                *dof_vel[6:8].tolist(),
                up_z,
                int(wc[0]),
                int(wc[1]),
                int(recovered),
            ]
        )
        obs = st.obs

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        # 数据保留两位小数 (用户约定); step/接触/恢复锁存为整数列不受影响
        writer.writerows([round(v, 2) if isinstance(v, float) else v for v in row] for row in rows)

    print("=" * 60)
    print(f"姿态数据导出: {out_csv}")
    print(f"  checkpoint: {ckpt_path}")
    print(f"  姿态: {pose_tag}    步数: {args.steps}   时间: {args.steps * dt:.2f}s")
    print(f"  CSV 列: {len(COLUMNS)} 列 ({COLUMNS[0]} .. {COLUMNS[-1]})")
    print("  反推姿态示例见 CLAUDE.md §1.5.2 (logs/pose_data/xxx.csv)")


if __name__ == "__main__":
    main()
