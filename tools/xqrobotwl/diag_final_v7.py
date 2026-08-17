#!/usr/bin/env python3
"""v7 最终复检: 四算法重新诊断 (站立/跳跃/髋外展/膝/相位).

SRL/SRL+VMC 用 v7 最终模型; PPO+VMC/纯PPO 用现有定稿模型 (v7 膝守卫已通过
vmc.py/apply_action 生效)。
用法:
  uv run tools/xqrobotwl/diag_final_v7.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.xqrobotwl.diag_jump_problems import test_self_trigger, test_jump  # noqa: E402
from tools.xqrobotwl.diag_jump_postjump import analyze  # noqa: E402

ALGOS = {
    "纯PPO": ("XqRobotWLJumpFlat",
              "logs/rsl_rl_ppo/XqRobotWLJumpFlat/2026-08-16_01-53-39_mujoco/model_9999.pt"),
    "PPO+VMC": ("XqRobotWLJumpVMC",
                "logs/rsl_rl_ppo/XqRobotWLJumpVMC/2026-08-16_01-53-43_mujoco/model_1000.pt"),
    "SRL(v7)": ("XqRobotWLJumpSRLFlat", None),   # 自动取最新 run 的 model_4000
    "SRL+VMC(v7)": ("XqRobotWLJumpSRLVMC", None),
}
HIDDEN = [512, 512, 256, 128]


def _latest_model(task: str, iters: int = 4000) -> str:
    import os
    runs = sorted(d for d in os.listdir(ROOT / "logs" / "rsl_rl_ppo" / task)
                  if d.endswith("mujoco"))
    for run in reversed(runs):
        for cand in (iters, iters - 1):  # 训练存到 max_iterations-1
            p = ROOT / "logs" / "rsl_rl_ppo" / task / run / f"model_{cand}.pt"
            if p.exists():
                return str(p.relative_to(ROOT))
    return None


def main() -> int:
    result = {}
    for algo, (task, ckpt) in ALGOS.items():
        if ckpt is None:
            ckpt = _latest_model(task)
            if ckpt is None:
                print(f"[{algo}] 未找到最终模型, 跳过")
                continue
        print(f"\n========== [{algo}] {ckpt} ==========", flush=True)
        st = test_self_trigger(task, ckpt, HIDDEN, 300)
        print(f"[站立3s] stand_z={st['stand_z']:.3f} max_z_rel={st['max_z_rel']:.3f} "
              f"jump_events={st['jump_events']} stand|gyro|={st['stand_gyro_mean']:.3f} "
              f"(max {st['stand_gyro_max']:.3f}) end|gyro|={st['end_gyro_mean']:.3f}")
        jm = test_jump(task, ckpt, HIDDEN)
        print(f"[跳跃] jump={jm['jump_height']}m crouch={jm['crouch_depth']} "
              f"hip_roll_max={jm['max_roll']} knee_max={jm['max_knee']} "
              f"overextend={jm['knee_overextend']} air={jm['air_steps']} phase={jm['phase_seq']}")
        r = analyze(algo, task, ckpt, HIDDEN)
        print(f"[跳后5s] pulse_jumps={r['pulse_jumps']} tail_jumps={r['tail_jumps']} "
              f"tail|gyro| mean={r['tail_gyro_mean']:.3f} max={r['tail_gyro_max']:.3f} "
              f"FSM={r['phase_dist']}")
        result[algo] = {"self_trigger": st, "jump": jm, "postjump": r}
    out = ROOT / "logs" / "pose_data" / "jump_final_v7_diag.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
