"""渲染单个机器人的跌倒恢复视频 (无辅助, 确定性).

流程: 随机复位 (或指定姿态) → 跑策略找到能恢复的姿态 (base_z>0.5 保持) → 重放该姿态并录视频.

用法:
  uv run mjpython scripts/xqrobotwl/render_recovery_video.py \
      --run <run_dir> --ckpt model_15000.pt [--out video/fall_recovery/x.mp4] [--steps 1000]
  # 指定倒地姿态 (0=仰卧 1=俯卧/前倒 2=左躺 3=右躺), 默认 -1 随机:
  uv run mjpython scripts/xqrobotwl/render_recovery_video.py \
      --run <run_dir> --ckpt model_5000.pt --pose 1 --out video/fall_recovery/prone.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from unilab.base import registry  # noqa: E402
from unilab.base.registry import ensure_registries  # noqa: E402
from unilab.dr.dr_utils import build_common_reset_randomization, zero_actions  # noqa: E402

CONFIG = ROOT / "conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml"

POSES = {0: "仰卧_supine", 1: "俯卧_前倒_prone", 2: "左躺_left", 3: "右躺_right"}


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="model_15000.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--max_tries", type=int, default=40)
    ap.add_argument("--pose", type=int, default=-1, help="固定倒地姿态 0-3, -1=随机")
    args = ap.parse_args()

    ensure_registries()
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    override = {"reward_config": cfg["reward"]}
    override["reward_config"]["force_assist_enabled"] = False  # 无辅助部署
    override.update(cfg.get("env", {}))

    env = registry.make(
        "XqRobotWLFallRecoveryFlat",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override=override,
    )
    if args.pose in (0, 1, 2, 3):
        _set_fixed_pose_provider(env, args.pose)
        print(f"固定倒地姿态: {args.pose} ({POSES[args.pose]})")

    ckpt_path = ROOT / "logs/rsl_rl_cpo/XqRobotWLFallRecoveryFlat" / args.run / args.ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pol = ActorMLP(297)
    pol.load_state_dict({k: v for k, v in ckpt["actor_state_dict"].items() if k.startswith("mlp.")})
    pol.eval()

    # ── 找能恢复的姿态 ──
    backend = env._backend
    nq = int(backend.nq)
    idx_qpos = int(getattr(backend, "_idx_qpos", 1))
    idx_qvel = int(getattr(backend, "_idx_qvel", 1 + nq))
    nv = int(getattr(backend, "nv", nq - 1))
    saved_qpos = saved_qvel = None
    found = False
    for attempt in range(args.max_tries):
        obs, _ = env.reset(np.array([0], dtype=np.int32))
        st_full = backend.get_physics_state()
        qpos = np.asarray(st_full[0, idx_qpos : idx_qpos + nq], dtype=np.float64)
        qvel = np.asarray(st_full[0, idx_qvel : idx_qvel + nv], dtype=np.float64)
        maxz = 0.0
        up = 0.0
        for i in range(args.steps):
            ot = torch.tensor(obs["obs"], dtype=torch.float32)
            with torch.no_grad():
                a = pol(ot).numpy().astype(np.float64)
            st = env.step(a)
            bz = float(backend.get_base_pos()[0, 2])
            maxz = max(maxz, bz)
            up = max(up, float(backend.get_sensor_data("upvector")[0, 2]))
            obs = st.obs
            if i > 200 and maxz > 0.50:  # 已站起并保持
                break
        if maxz > 0.50:
            saved_qpos, saved_qvel, found = qpos, qvel, True
            print(f"找到恢复姿态: attempt={attempt}, max_z={maxz:.2f}, 躯干直立={up:.2f}")
            break
        if attempt % 5 == 0:
            print(f"尝试 {attempt}/{args.max_tries} ... max_z={maxz:.2f}")

    if not found:
        raise SystemExit(f"未找到恢复姿态 (max_tries={args.max_tries}), 检查 checkpoint 恢复率")

    # ── 重放该姿态并录视频 ──
    pose_tag = f"_{POSES[args.pose]}" if args.pose in (0, 1, 2, 3) else ""
    out = args.out or str(ROOT / "video/fall_recovery" / f"recovery_{args.ckpt}{pose_tag}.mp4")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize():
        # 先 reset 拿合法 obs 结构, 再快照物理态到已恢复姿态 (首步动作偏差 ~1 步, 可忽略)
        obs2, _ = env.reset(np.array([0], dtype=np.int32))
        backend.set_state(np.array([0], dtype=np.int32), saved_qpos, saved_qvel)
        return obs2["obs"]

    def step(obs_np):
        ot = torch.tensor(obs_np, dtype=torch.float32)
        with torch.no_grad():
            a = pol(ot).numpy().astype(np.float64)
        st = env.step(a)
        return st.obs["obs"]

    backend.run_playback(
        env=env,
        initialize=initialize,
        step=step,
        num_steps=args.steps,
        output_video=str(out_path),
        render_spacing=0.0,
        camera_kwargs={
            "cam_tracking": True,  # 相机跟踪机器人, 防出视角
            "cam_distance": 2.5,
            "cam_elevation": -15,
            "cam_azimuth": 90,
        },
    )
    print(f"视频已保存: {out_path}")


if __name__ == "__main__":
    main()
