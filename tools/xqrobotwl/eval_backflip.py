#!/usr/bin/env python3
"""后空翻任务对称几何评估 — §7.6 对照 (翻转 360° + 落地站立).

触发: flip_trigger_prob=1.0 (复位即触发翻转). 指标: 翻转进度(俯仰积分, 目标 2π),
最高点, 存活(落地站立).

用法:
  uv run python tools/xqrobotwl/eval_backflip.py --ckpt <path> [--episodes 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CONF = ROOT / "conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=800, help="单 episode 步数 (翻转~1.7s + 落地观察)")
    ap.add_argument("--num_envs", type=int, default=1)
    args = ap.parse_args()

    from unilab.base import registry
    from unilab.base.registry import ensure_registries
    from tools.xqrobotwl.render_trained_backflip import ActorMLP  # 复用模型结构

    ensure_registries()
    cfg = yaml.safe_load(open(CONF))
    override = {"reward_config": cfg["reward"]}
    override.update(cfg.get("env", {}))
    override["reward_config"]["flip_warmup_iters"] = 0
    override["reward_config"]["flip_trigger_prob"] = 1.0
    override["curriculum"] = {"enabled": False}
    override["noise_config"] = {"level": 0.0}
    override["domain_rand"] = {
        **override.get("domain_rand", {}),
        "randomize_init_yaw": False, "randomize_ground_friction": False,
        "randomize_kp": False, "randomize_kd": False, "random_com": False,
        "randomize_leg_length": False, "push_robots": False,
    }
    override["max_episode_seconds"] = 1000.0

    env = registry.make("XqRobotWLBackflipFlat", sim_backend="mujoco",
                        num_envs=args.num_envs, env_cfg_override=override)
    env.set_autoreset(False)

    ckpt_obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ad = ckpt_obj.get("actor_state_dict", {})
    obs_dim = int(env.obs_groups_spec["obs"])
    pol = ActorMLP(obs_dim)
    pol.load_state_dict({k: v for k, v in ad.items() if k.startswith("mlp.")})
    pol.eval()
    if any(k.startswith("obs_normalizer.") for k in ad):
        from rsl_rl.modules.normalization import EmpiricalNormalization
        normalizer = EmpiricalNormalization(obs_dim)
        nk = {k.replace("obs_normalizer.", ""): v for k, v in ad.items()
              if k.startswith("obs_normalizer.")}
        normalizer.load_state_dict(nk)
        normalizer.eval()
        _raw = pol

        def pol(x, _r=_raw, _n=normalizer):
            return _r(_n(x))

    dt = float(env._cfg.ctrl_dt)
    results = []
    with torch.no_grad():
        for ep in range(args.episodes):
            env.init_state()
            obs, _ = env.reset(np.arange(args.num_envs))
            flip_progress = 0.0
            max_base_z = 0.0
            prev_up = None
            surv = False
            flip_done = False
            for step in range(args.steps):
                obs_t = torch.tensor(obs["obs"], dtype=torch.float32)
                a = pol(obs_t).numpy().astype(np.float64)
                st = env.step(a)
                backend = env._backend
                up = np.asarray(backend.get_sensor_data("upvector"), dtype=np.float64)[0]
                base_z = float(backend.get_base_pos()[0, 2])
                max_base_z = max(max_base_z, base_z)
                # 俯仰角变化 (up 向量 → pitch, 积分累计翻转)
                pitch = float(np.arctan2(up[0], up[2]))
                if prev_up is not None:
                    d = pitch - prev_up
                    if d > np.pi:
                        d -= 2 * np.pi
                    elif d < -np.pi:
                        d += 2 * np.pi
                    flip_progress += d
                prev_up = pitch
                obs = st.obs
                if abs(flip_progress) >= 2 * np.pi - 0.5:
                    flip_done = True
                if st.terminated[0] or st.truncated[0]:
                    break
                # 落地站立: 翻转完成后恢复直立 + 稳定
                if flip_done and base_z > 0.45 and up[2] > 0.8 and step > 200:
                    surv = True
                    break
            results.append((surv, flip_done, flip_progress, max_base_z, step))
            print(f"[backflip] ep{ep}: {'落地站立✅' if surv else '未完成'} "
                  f"flip_done={flip_done} progress={flip_progress:.2f}rad({flip_progress/6.283*100:.0f}%) "
                  f"max_z={max_base_z:.2f}m steps={step+1}")
    n_ok = sum(r[0] for r in results)
    n_flip = sum(r[1] for r in results)
    print(f"[backflip] 落地站立率 {n_ok/args.episodes*100:.0f}% ({n_ok}/{args.episodes})  "
          f"翻转完成率 {n_flip/args.episodes*100:.0f}% ({n_flip}/{args.episodes})")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
