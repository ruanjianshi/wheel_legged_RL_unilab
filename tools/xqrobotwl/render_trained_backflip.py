"""渲染训练后的后空翻策略回放 — 侧面视角 mp4

用法:
  uv run tools/xqrobotwl/render_trained_backflip.py \
      --ckpt logs/rsl_rl_ppo/XqRobotWLBackflipFlat/<run>/model_9999.pt \
      --out video/backflip/2026-08-04_02_训练策略.mp4 \
      --steps 600
"""

from __future__ import annotations

import argparse
import subprocess
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
CONFIG = ROOT / "conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml"


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
    ap = argparse.ArgumentParser(description="渲染训练后后空翻策略")
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint 路径")
    ap.add_argument("--out", type=str, default="", help="输出 mp4 路径")
    ap.add_argument("--steps", type=int, default=600, help="渲染步数 (0.01s/步)")
    ap.add_argument("--num_envs", type=int, default=4, help="渲染 env 数(网格)")
    args = ap.parse_args()

    ensure_registries()
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    override = {"reward_config": cfg["reward"]}
    override.update(cfg.get("env", {}))
    override["reward_config"]["flip_warmup_iters"] = 0  # 全开翻转
    override["reward_config"]["flip_trigger_prob"] = 1.0

    env = registry.make(
        "XqRobotWLBackflipFlat",
        sim_backend="mujoco",
        num_envs=args.num_envs,
        env_cfg_override=override,
    )
    obs, _ = env.reset(np.arange(args.num_envs))

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ad = ckpt["actor_state_dict"]
    pol = ActorMLP(obs["obs"].shape[1])
    pol.load_state_dict({k: v for k, v in ad.items() if k.startswith("mlp.")})
    pol.eval()

    backend = env._backend
    states = []
    for i in range(args.steps):
        obs_t = torch.tensor(obs["obs"], dtype=torch.float32)
        with torch.no_grad():
            a = pol(obs_t).numpy().astype(np.float64)
        st = env.step(a)
        obs = st.obs
        states.append(backend._physics_state.copy())

    # 渲染 (跟踪相机: 跟随机器人, 防止飞出视角)
    from unilab.visualization.render_many import render_states_get_frames_tracking

    frames = render_states_get_frames_tracking(
        states,
        str(ROOT / "src/unilab/assets/robots/xqrobotwl/scene_flat.xml"),
        width=640,
        height=480,
        tracking_env_idx=0,
        max_extra_envs=0,
        cam_distance=2.6,
        cam_elevation=-12,
        cam_azimuth=90,
        render_spacing=0.0,
    )

    out_path = Path(args.out) if args.out else ROOT / "video" / "backflip" / "trained.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f"_frames_{out_path.stem}"
    tmp.mkdir(exist_ok=True)
    for i, fr in enumerate(frames):
        (tmp / f"f_{i:05d}.ppm").write_bytes(
            b"P6\n%d %d\n255\n" % (fr.shape[1], fr.shape[0]) + fr.tobytes()
        )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "100",
            "-i",
            str(tmp / "f_%05d.ppm"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    for f in tmp.glob("f_*.ppm"):
        f.unlink()
    tmp.rmdir()
    print(f"✅ 视频已生成: {out_path}")


if __name__ == "__main__":
    main()
