"""平地跳跃评估 (CLAUDE.md §7.5).

成功率≥90% (触发后跳出高度且落地站立) · 跳出明显高度 · 有腾空.
触发脉冲与 verify_jump 一致 (每 jump_every 步触发半窗, 重复 ≥10 次).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from _devlog.assess import engine
from _devlog.assess import metrics as M


def evaluate(env, policy, args) -> dict[str, float]:
    dt = env._cfg.ctrl_dt
    steps = args.max_steps or 1200
    # 触发周期对齐训练命令重采样 (resampling_time, 默认 4s)
    resample_s = getattr(getattr(env._cfg, "commands", None), "resampling_time", 4.0)
    jump_every = args.jump_every or max(1, int(resample_s / dt))
    n_env = env.num_envs

    env.init_state()  # 初始化 NpEnvState (命令注入需先 init_state)
    recs: list[tuple[int, float, bool, M.StepSample]] = []  # (step, trig, terminated, sample)
    with torch.no_grad():
        for step in range(steps):
            trig = 1.0 if (step % jump_every) < (jump_every // 2) else 0.0
            env.state.info["commands"][:, 4] = trig
            obs_t = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
            a = policy(obs_t).numpy().astype(np.float64)
            st = env.step(a)
            for i in range(n_env):
                recs.append(
                    (step, trig, bool(st.terminated[i]), engine.collect_step(env, st, step, i, dt))
                )

    # 站立基准: trigger off + 双轮着地的 base_z 中位数
    ground = [
        s.base_pos[2] for (_, trig, _, s) in recs if trig == 0.0 and np.mean(s.wheel_contact) > 0.99
    ]
    standing_z = float(np.median(ground)) if ground else float("nan")

    n_windows = steps // jump_every
    successes = 0
    heights: list[float] = []
    clean_windows = 0
    for k in range(n_windows):
        win = [
            (step, trig, term, s)
            for (step, trig, term, s) in recs
            if k * jump_every <= step < (k + 1) * jump_every
        ]
        if not win:
            continue
        max_z = max(s.base_pos[2] for (_, _, _, s) in win)
        h = (max_z - standing_z) if not math.isnan(standing_z) else 0.0
        heights.append(max(h, 0.0))
        if h > 0.2 and win[-1][3].up_z > 0.85:  # 跳出明显高度且落地站立
            successes += 1
        if not any(term for (_, _, term, _) in win):  # 该窗口未中途终止
            clean_windows += 1

    air = M.air_frac([s for (_, _, _, s) in recs])
    return {
        "success_rate": successes / n_windows if n_windows else float("nan"),
        "jump_height": float(np.mean(heights)) if heights else float("nan"),
        "air_frac": air,
        "survival_rate": clean_windows / n_windows if n_windows else float("nan"),
        "standing_z": standing_z,
    }
