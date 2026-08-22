"""确定性评估 XqRobotWLSingleLegUnicycle checkpoint — 单轮平衡行走。

指标:
  - episode_length: 平均时长 (平衡越久越好, max_steps=600 = 6s)
  - survived: 跑满比例
  - support_wheel_on_ground / free_wheel_clear: 真正单轮率
  - vx_rmse / signed_distance: 速度跟踪与行走距离
  - pitch_rmse / roll_rmse: 机身横躺保持误差 (base长轴·[1,0,0] 与 up·[0,-1,0])

用法:
  uv run mjpython tools/xqrobotwl/eval_single_leg_unicycle.py \
      --ckpt logs/rsl_rl_ppo/XqRobotWLSingleLegUnicycle/<run>/model_XXXX.pt \
      --episodes 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from unilab.base import registry
from unilab.base.registry import ensure_registries

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "conf/ppo/task/xqrobotwl_single_leg_unicycle/mujoco.yaml"
OBS_DIM = 324  # 36 × 9 帧


class ActorMLP(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        return self.mlp(x)


def main() -> None:
    ap = argparse.ArgumentParser(description="评估 single_leg_unicycle")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--num_envs", type=int, default=10)
    ap.add_argument(
        "--commands",
        default="-0.20,-0.10,0.0,0.10,0.20",
        help="逗号分隔的 vx 命令，循环分配给并行环境",
    )
    ap.add_argument(
        "--curriculum-progress",
        type=float,
        default=1.0,
        help="评估时速度反馈/课程进度；中间 checkpoint 应使用其训练进度",
    )
    args = ap.parse_args()

    ensure_registries()
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    override = {"reward_config": cfg["reward"]}
    override.update(cfg.get("env", {}))

    env = registry.make(
        "XqRobotWLSingleLegUnicycle",
        sim_backend="mujoco",
        num_envs=args.num_envs,
        env_cfg_override=override,
    )
    env.init_state()
    env._command_env_steps = int(
        np.clip(args.curriculum_progress, 0.0, 1.0) * env._reward_cfg.command_curriculum_steps
    )
    obs = env.state.obs
    command_values = np.asarray([float(x) for x in args.commands.split(",")], dtype=np.float64)
    vx_cmd = np.resize(command_values, args.num_envs)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pol = ActorMLP(obs["obs"].shape[1])
    pol.load_state_dict({k: v for k, v in ckpt["actor_state_dict"].items() if k.startswith("mlp.")})
    pol.eval()

    ep_len = np.zeros(args.num_envs)
    active = np.ones(args.num_envs, dtype=bool)
    wheel_down_sum = np.zeros(args.num_envs)
    free_clear_sum = np.zeros(args.num_envs)
    pitch_sum = np.zeros(args.num_envs)
    roll_sum = np.zeros(args.num_envs)
    vx_sqerr_sum = np.zeros(args.num_envs)
    vx_sum = np.zeros(args.num_envs)
    lateral_speed_sum = np.zeros(args.num_envs)
    sample_count = np.zeros(args.num_envs)
    # 后端的并行环境 base position 可能按各自 terrain origin 表示并在内部重心化；
    # 用 episode 内有效的局部纵向速度积分，得到可跨后端比较的有符号路程。
    signed_distance = np.zeros(args.num_envs)

    for i in range(args.max_steps):
        env.state.info["commands"][:, 0] = vx_cmd
        obs_t = torch.tensor(obs["obs"], dtype=torch.float32)
        with torch.no_grad():
            a = pol(obs_t).numpy().astype(np.float64)
        st = env.step(a)
        lw_z = np.asarray(env._backend.get_sensor_data("left_wheel_world_pos"))[:, 2]
        rw_z = np.asarray(env._backend.get_sensor_data("right_wheel_world_pos"))[:, 2]
        up = env._backend.get_sensor_data("upvector")
        bx = env._backend.get_sensor_data("basexvector")
        linvel = env.get_local_linvel()
        wheel_down_sum[active] += (lw_z[active] < 0.13).astype(np.float64)
        free_clear_sum[active] += (rw_z[active] > 0.16).astype(np.float64)
        roll_sum[active] += (up @ np.array([0.0, -1.0, 0.0]))[active]
        pitch_sum[active] += (bx @ np.array([1.0, 0.0, 0.0]))[active]
        vx_sqerr_sum[active] += np.square(linvel[active, 0] - vx_cmd[active])
        vx_sum[active] += linvel[active, 0]
        signed_distance[active] += linvel[active, 0] * env._cfg.ctrl_dt
        lateral_speed_sum[active] += np.abs(linvel[active, 1])
        sample_count[active] += 1.0
        ep_len[active] += 1.0
        done = st.terminated | st.truncated
        if done.any():
            new_done = done & active
            ep_len[new_done] = np.minimum(ep_len[new_done], i + 1)
            active &= ~done  # ⚠️ done env 自动 reset 后不再累加, 冻结其长度
        obs = st.obs

    print("=" * 60)
    print(f"评估: {args.ckpt}")
    print("=" * 60)
    print(
        f"  episode 长度: {ep_len.mean():.1f} 步 / {args.max_steps}  "
        f"({(ep_len >= args.max_steps).mean() * 100:.0f}% 跑满 = 平衡 {args.max_steps / 100:.1f}s)"
    )
    count = np.maximum(sample_count, 1.0)
    print(f"  支撑轮贴地率: {(wheel_down_sum / count).mean() * 100:.1f}%")
    print(f"  自由轮离地率: {(free_clear_sum / count).mean() * 100:.1f}%")
    print(f"  vx 跟踪 RMSE: {np.sqrt(vx_sqerr_sum.sum() / count.sum()):.3f} m/s")
    print(f"  横向速度 |vy|: {(lateral_speed_sum / count).mean():.3f} m/s")
    print(
        f"  pitch 对齐: {(pitch_sum / count).mean():.3f}; roll 对齐: {(roll_sum / count).mean():.3f}"
    )
    print("  分命令结果:")
    for cmd in command_values:
        mask = np.isclose(vx_cmd, cmd)
        if not np.any(mask):
            continue
        rmse = np.sqrt(vx_sqerr_sum[mask].sum() / count[mask].sum())
        mean_vx = vx_sum[mask].sum() / count[mask].sum()
        distance = np.mean(signed_distance[mask])
        survive = np.mean(ep_len[mask] >= args.max_steps) * 100.0
        print(
            f"    vx_cmd={cmd:+.2f}: vx={mean_vx:+.3f}, RMSE={rmse:.3f}, "
            f"distance={distance:+.3f} m, 跑满={survive:.0f}%"
        )


if __name__ == "__main__":
    main()
