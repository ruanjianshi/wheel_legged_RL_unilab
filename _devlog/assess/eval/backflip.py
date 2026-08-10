"""平地后空翻评估 (CLAUDE.md §7.6).

翻转完成率≥90% (base 累计翻转 ≥360°) · 落地后稳定站立.
触发脉冲与 jump 一致; 翻转用 pitch 轴累计角度 (后空翻绕机体横轴).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from _devlog.assess import engine
from _devlog.assess import metrics as M

TWO_PI = 2.0 * math.pi


def _unwrap(vals: np.ndarray) -> np.ndarray:
    out = np.empty_like(vals)
    for i, v in enumerate(vals):
        if i == 0:
            out[i] = v
        else:
            d = v - out[i - 1]
            d = (d + math.pi) % TWO_PI - math.pi
            out[i] = out[i - 1] + d
    return out


def evaluate(env, policy, args) -> dict[str, float]:
    dt = env._cfg.ctrl_dt
    steps = args.max_steps or 1500
    resample_s = getattr(getattr(env._cfg, "commands", None), "resampling_time", 4.0)
    flip_every = args.jump_every or max(1, int(resample_s / dt))
    n_env = env.num_envs

    env.init_state()  # 初始化 NpEnvState (命令注入需先 init_state)
    recs: list[tuple[int, float, M.StepSample]] = []
    terminated_any = False
    with torch.no_grad():
        for step in range(steps):
            trig = 1.0 if (step % flip_every) < (flip_every // 2) else 0.0
            env.state.info["commands"][:, 4] = trig
            obs_t = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
            a = policy(obs_t).numpy().astype(np.float64)
            st = env.step(a)
            terminated_any = terminated_any or bool(st.terminated.any())
            for i in range(n_env):
                recs.append((step, trig, engine.collect_step(env, st, step, i, dt)))

    n_windows = steps // flip_every
    flips = 0
    landed = 0
    for k in range(n_windows):
        win = [s for (step, _, s) in recs if k * flip_every <= step < (k + 1) * flip_every]
        if not win:
            continue
        pitch = np.array([s.euler[1] for s in win])
        drift = _unwrap(pitch)
        total = abs(drift[-1] - drift[0]) if len(drift) > 1 else 0.0
        end_up = win[-1].up_z
        if total >= TWO_PI:  # 累计翻转 360°
            flips += 1
        if end_up > 0.85:
            landed += 1

    return {
        "flip_rate": flips / n_windows if n_windows else float("nan"),
        "land_survival": landed / n_windows if n_windows else float("nan"),
        "survival_rate": 0.0 if terminated_any else 1.0,
    }
