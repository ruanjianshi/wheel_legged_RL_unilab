#!/usr/bin/env python3
"""2×2 干净消融最终对比 (纯PPO / PPO+VMC / SRL / SRL+VMC).

每对只差输出层:
  纯PPO  vs PPO+VMC  (无 SLIP 参考, 纯PPO 奖励, 297D) — 关节PD vs 虚拟腿VMC
  SRL   vs SRL+VMC  (有 SLIP 参考, SRL 奖励, 315D)    — 关节PD vs 虚拟腿VMC

用法:
  uv run tools/xqrobotwl/diag_2x2_final.py
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

HIDDEN = [512, 512, 256, 128]

ALGOS = {
    "纯PPO": "XqRobotWLJumpFlat",
    "PPO+VMC": "XqRobotWLJumpVMC",
    "SRL": "XqRobotWLJumpSRLFlat",
    "SRL+VMC": "XqRobotWLJumpSRLVMC",
}
# 纯PPO / SRL 用已有定稿模型 (无重训); PPO+VMC / SRL+VMC 用最新 run 的 model_3999
FIXED = {
    "纯PPO": "logs/rsl_rl_ppo/XqRobotWLJumpFlat/2026-08-16_01-53-39_mujoco/model_9999.pt",
    "SRL": "logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat/2026-08-16_15-48-13_mujoco/model_3999.pt",
}


def _latest_model(task: str) -> str | None:
    import os
    runs = sorted(d for d in os.listdir(ROOT / "logs" / "rsl_rl_ppo" / task)
                  if d.endswith("mujoco"))
    for run in reversed(runs):
        for cand in (3999, 4000, 9999):
            p = ROOT / "logs" / "rsl_rl_ppo" / task / run / f"model_{cand}.pt"
            if p.exists():
                return str(p.relative_to(ROOT))
    return None


def main() -> int:
    result = {}
    for algo, task in ALGOS.items():
        ckpt = FIXED.get(algo) or _latest_model(task)
        if ckpt is None:
            print(f"[{algo}] 无模型, 跳过")
            continue
        print(f"\n========== [{algo}] {ckpt} ==========", flush=True)
        st = test_self_trigger(task, ckpt, HIDDEN, 300)
        print(f"[站立3s] z={st['stand_z']:.3f} |gyro|={st['stand_gyro_mean']:.3f} "
              f"(max {st['stand_gyro_max']:.3f}/end {st['end_gyro_mean']:.3f}) 自跳={st['jump_events']}")
        jm = test_jump(task, ckpt, HIDDEN)
        print(f"[跳跃] 跳高={jm['jump_height']}m 腾空={jm['air_steps']} 下蹲={jm['crouch_depth']} "
              f"髋外展={jm['max_roll']} 膝={jm['max_knee']}(过伸={jm['knee_overextend']}) "
              f"相位={jm['phase_seq']}")
        r = analyze(algo, task, ckpt, HIDDEN)
        print(f"[跳后5s] |gyro|={r['tail_gyro_mean']:.3f}(max {r['tail_gyro_max']:.3f}) 自跳={r['tail_jumps']}")
        result[algo] = {"ckpt": ckpt, "stand": st, "jump": jm, "post": r}
    out = ROOT / "logs" / "pose_data" / "jump_2x2_final.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
