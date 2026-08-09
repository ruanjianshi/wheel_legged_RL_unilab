"""确定性评估跌倒恢复 checkpoint — 多姿态倒地 → 起身 → 保持平衡

指标:
  - recovery_rate: base_z 达到 h_cmd2 (0.55, 站立) 的 episode 比例
  - stay_up_rate: episode 跑到 max_steps (未倒/未贴地超时) 比例
  - mean_max_z / mean_ep_len / 平均最长连续站立 / 平均水平漂移

用法:
  uv run mjpython scripts/xqrobotwl/eval_fall_recovery.py \
      --run <run_dir> [--ckpt model_XXX.pt] [--episodes 20]
  # 固定倒地姿态 (0=仰卧 1=俯卧/前倒 2=左躺 3=右躺), 默认随机 4 姿态:
  uv run mjpython scripts/xqrobotwl/eval_fall_recovery.py --run <run_dir> --pose 1
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
from unilab.dr.dr_utils import build_common_reset_randomization, zero_actions

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml"
POSES = {0: "仰卧_supine", 1: "俯卧_前倒_prone", 2: "左躺_left", 3: "右躺_right"}


def _set_fixed_pose_provider(env, pose: int) -> None:
    """把 DR provider 换成固定倒地姿态 (0=仰卧 1=俯卧 2=左躺 3=右躺)."""
    from unilab.envs.locomotion.xqrobotwl import fall_recovery as fr

    class _FixedPoseProvider(fr.XqRobotWLFallRecoveryDRProvider):
        def __init__(self, pose: int) -> None:
            self._pose = pose

        def build_reset_plan(self, env: object, env_ids: np.ndarray) -> fr.ResetPlan:
            num_reset = len(env_ids)
            rng = np.random.default_rng()
            base_z = np.full(num_reset, fr._LYING_Z, dtype=np.float64) + rng.uniform(-0.02, 0.02, size=num_reset)
            quats = np.zeros((num_reset, 4), dtype=np.float64)
            for i in range(num_reset):
                q = fr._pose_quat(self._pose)
                dq = fr._quat_from_euler(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3))
                quats[i] = fr._quat_mul(q, dq)
            legs = np.clip(np.array([0.1, 0.15, 0.15, -0.1, -0.15, -0.15]) * rng.uniform(0.5, 1.5, size=(num_reset, 6)), -0.85, 0.85)
            qpos = np.zeros((num_reset, 15), dtype=np.float64)
            qpos[:, 0] = rng.uniform(-0.1, 0.1, size=num_reset)
            qpos[:, 1] = rng.uniform(-0.1, 0.1, size=num_reset)
            qpos[:, 2] = base_z
            qpos[:, 3:7] = quats
            qpos[:, 7] = legs[:, 0]; qpos[:, 8] = legs[:, 1]; qpos[:, 9] = legs[:, 2]
            qpos[:, 11] = legs[:, 3]; qpos[:, 12] = legs[:, 4]; qpos[:, 13] = legs[:, 5]
            qvel = np.zeros((num_reset, 14), dtype=np.float64)
            randomization = build_common_reset_randomization(env, num_reset)
            return fr.ResetPlan(
                env_ids=env_ids, qpos=qpos, qvel=qvel,
                info_updates={
                    "commands": np.zeros((num_reset, 5), dtype=np.float64),
                    "current_actions": zero_actions(num_reset, env._num_action),
                    "last_actions": zero_actions(num_reset, env._num_action),
                },
                randomization=randomization,
            )

    env._dr_manager._provider = _FixedPoseProvider(pose)  # type: ignore[union-attr]


class ActorMLP(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ELU(),
            nn.Linear(512, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 8),
        )

    def forward(self, x):
        return self.mlp(x)


def main() -> None:
    ap = argparse.ArgumentParser(description="评估跌倒恢复")
    ap.add_argument("--run", type=str, required=True, help="run 目录名 (logs/rsl_rl_cpo/... 下)")
    ap.add_argument("--ckpt", type=str, default=None, help="checkpoint 文件名 (默认最新)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--num_envs", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--pose", type=int, default=-1, help="固定倒地姿态 0-3, -1=随机")
    args = ap.parse_args()

    ensure_registries()
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    override = {"reward_config": cfg["reward"]}
    override["reward_config"]["force_assist_enabled"] = False  # 评估真实(无辅助力)恢复
    override.update(cfg.get("env", {}))

    env = registry.make(
        "XqRobotWLFallRecoveryFlat",
        sim_backend="mujoco",
        num_envs=args.num_envs,
        env_cfg_override=override,
    )
    if args.pose in (0, 1, 2, 3):
        _set_fixed_pose_provider(env, args.pose)

    run_dir = ROOT / "logs/rsl_rl_cpo/XqRobotWLFallRecoveryFlat" / args.run
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

    obs, _ = env.reset(np.arange(args.num_envs))
    ep_len = np.zeros(args.num_envs)
    active = np.ones(args.num_envs, dtype=bool)
    max_z = np.zeros(args.num_envs)
    max_up = np.zeros(args.num_envs)
    wheel_on = np.zeros(args.num_envs)
    reached_h2 = np.zeros(args.num_envs, dtype=bool)
    hold_cur = np.zeros(args.num_envs)  # 当前连续站立帧
    hold_max = np.zeros(args.num_envs)  # 单次最长连续站立帧
    start_x = np.asarray(env._backend.get_base_pos(), dtype=np.float64)[:, 0].copy()
    max_drift = np.zeros(args.num_envs)  # 最大水平漂移距离
    h_cmd2 = env._jump_cfg.h_cmd2

    for i in range(args.max_steps):
        obs_t = torch.tensor(obs["obs"], dtype=torch.float32)
        with torch.no_grad():
            a = pol(obs_t).numpy().astype(np.float64)
        st = env.step(a)
        base_pos = np.asarray(env._backend.get_base_pos(), dtype=np.float64)
        base_z, base_x = base_pos[:, 2], base_pos[:, 0]
        up = np.asarray(env._backend.get_sensor_data("upvector"), dtype=np.float64)[:, 2]
        max_z = np.maximum(max_z, base_z)
        max_up = np.maximum(max_up, up)
        max_drift = np.maximum(max_drift, np.abs(base_x - start_x))
        wc = st.info.get("wheel_contact", np.zeros((env.num_envs, 2)))
        wheel_on += np.min(wc, axis=1)  # 双轮着地累计
        reached_h2 |= base_z > h_cmd2
        # 稳定站立保持: 连续帧数 (z>0.45 + 直立>0.85 + 双轮)
        standing = (base_z > 0.45) & (up > 0.85) & (np.min(wc, axis=1) > 0.5)
        hold_cur[standing] += 1
        hold_cur[~standing] = 0
        hold_max = np.maximum(hold_max, hold_cur)
        ep_len[active] += 1.0
        done = st.terminated | st.truncated
        if done.any():
            new_done = done & active
            ep_len[new_done] = np.minimum(ep_len[new_done], i + 1)
            active &= ~done
        obs = st.obs

    active_steps = ep_len.max() if ep_len.max() > 0 else 1
    pose_tag = f"  (姿态 {args.pose}: {POSES[args.pose]})" if args.pose in (0, 1, 2, 3) else "  (随机 4 姿态)"
    print("=" * 60)
    print(f"评估: {ckpt_path}{pose_tag}")
    print("=" * 60)
    print(f"  episode 数: {args.num_envs}")
    print(f"  恢复率 (base_z 达 {h_cmd2}m 站立): {(reached_h2.mean()*100):.1f}%")
    print(f"  保持率 (跑到 {args.max_steps} 步): {(ep_len >= args.max_steps).mean()*100:.1f}%")
    print(f"  平均最大高度: {max_z.mean():.2f} m  (躺地~0.15, 站立~0.55)")
    print(f"  平均最大躯干直立度: {max_up.mean():.2f}  (躺地~0, 直立=1)")
    print(f"  双轮着地率: {(wheel_on.mean()/active_steps*100):.1f}%")
    print(f"  平均 episode 长度: {ep_len.mean():.1f} 步 / {args.max_steps}")
    hold_s = hold_max.mean() * env._cfg.ctrl_dt  # 平均最长连续站立时间
    print(f"  平均最长连续站立: {hold_s:.2f} s  (需 ≥0.5s 才算稳定站住)")
    print(f"  平均最大水平漂移: {max_drift.mean():.2f} m  (恢复后应 < 0.5m, 防一直后退)")
    print("  " + ("✅ 已学会恢复" if reached_h2.mean() > 0.5 else "⚠️ 恢复未成形"))


if __name__ == "__main__":
    main()
