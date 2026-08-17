#!/usr/bin/env python3
"""测试"跳完一次后, trigger 归 0 是否还持续弹跳" (P3 SRL / P4 SRL+VMC).

序列: settle 80(触发0) + pulse 120(触发1) + tail 500(触发0)
统计:
  - pulse 内跳跃次数
  - tail(触发0)内 base_z 自跳事件数 + 振荡幅度/频率 (站立偏差量化)
  - tail 期 FSM 相位分布 (是否重复进入 crouch/thrust)

用法:
  uv run tools/xqrobotwl/diag_jump_postjump.py --algos SRL,SRL+VMC
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.xqrobotwl.diag_jump_problems import _run, _find_jump_events, ALGOS  # noqa: E402

SETTLE, PULSE, TAIL = 80, 120, 500


def analyze(algo, task, ckpt, hidden):
    recs = _run(task, ckpt, hidden, SETTLE + PULSE + TAIL,
                lambda s: 1.0 if SETTLE <= s < SETTLE + PULSE else 0.0)
    n = len(recs)
    z = np.array([r["z"] for r in recs])
    g = np.array([r["gyro"] for r in recs])
    phase = np.array([r["phase"] for r in recs])
    stand = float(np.median(z[:30]))
    # pulse 内跳跃次数
    on_z = z[SETTLE:SETTLE + PULSE]
    on_jumps, _ = _find_jump_events(on_z, stand)
    # tail (trigger=0) 内自跳
    tail_z = z[SETTLE + PULSE:]
    tail_jumps, tail_starts = _find_jump_events(tail_z, stand)
    # tail 振荡: 过零(相对 stand)次数
    rel = tail_z - stand
    nzero = 0
    for i in range(1, len(rel)):
        if (rel[i - 1] < 0) != (rel[i] < 0):
            nzero += 1
    tail_dur = len(tail_z) * 0.01  # ctrl_dt=0.01s
    # tail 期 FSM
    tail_phase = phase[SETTLE + PULSE:]
    ph_names = {-1: "idle", 0: "crouch", 1: "thrust", 2: "flight", 3: "preland", 4: "landing"}
    phase_dist = {ph_names[k]: int((tail_phase == k).sum()) for k in ph_names}
    # 首次触发后整段 (pulse+tail) 的相位序列
    seq, last = [], None
    for p in phase[SETTLE:]:
        if int(p) != last:
            seq.append(ph_names.get(int(p), int(p)))
            last = int(p)
    print(f"\n===== [{algo}] =====")
    print(f"stand_z={stand:.3f}  pulse内跳跃={on_jumps}")
    print(f"tail(trigger=0, {tail_dur:.1f}s, {len(tail_z)}步): 自跳={tail_jumps} "
          f"starts={tail_starts} 振荡过零={nzero} 幅度max={rel.max():+.3f} min={rel.min():+.3f}")
    print(f"tail |gyro|: mean={g[SETTLE+PULSE:].mean():.3f} max={g[SETTLE+PULSE:].max():.3f} "
          f"end={g[-1]:.3f}")
    print(f"tail FSM分布: {phase_dist}")
    print(f"触发后相位序列: {seq}")
    print(f"end_z={z[-1]:.3f}")
    return dict(stand_z=stand, pulse_jumps=on_jumps, tail_jumps=tail_jumps,
                tail_starts=tail_starts, tail_zero_cross=nzero, tail_z_amp=float(rel.max()),
                tail_gyro_mean=float(g[SETTLE+PULSE:].mean()), tail_gyro_max=float(g[SETTLE+PULSE:].max()),
                phase_dist=phase_dist, phase_seq=seq)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--algos", default="SRL,SRL+VMC")
    p.add_argument("--hidden", default="512,512,256,128")
    p.add_argument("--out", default="logs/pose_data/jump_postjump_diag.json")
    args = p.parse_args()
    hidden = [int(x) for x in args.hidden.split(",")]
    result = {}
    for algo in args.algos.split(","):
        task, ckpt = ALGOS[algo]
        result[algo] = analyze(algo, task, ckpt, hidden)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
