#!/usr/bin/env python3
"""双模式点足行走 — 确定性评估 (站立 ⇄ 点足抬腿 + 指令追踪 + 切换).

老板验收 (2026-08-18 需求):
  1. 默认站立姿态 (mode=0): 微动平衡 (§1.4), 零指令漂移小
  2. 按键切换成点足抬腿模式 (mode=1): 左右交替 + 抬腿幅度近似 (8-18 门槛)
  3. 抬腿模式下指令追踪: 前进/后退 / 侧移 / 转向
  4. 切换过程稳定: 切换后 0.5s 不跌倒, 1s 内进入新模式行为

脚本化模式序列 (命令通道, 等价键盘 H 键):
  站立 3s → 抬腿+前进 3s → 抬腿+侧移 3s → 抬腿+转向 3s → 切回站立 3s

输出: 姿态数据 CSV + 指标 JSON (logs/pose_data/ + logs/eval/)
用法:
  uv run python tools/xqrobotwl/verify_toe_walk_mode.py \
      [--run <run_dir>] [--ckpt model_9999.pt] [--seed 1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _devlog.assess import engine, pose, tasks  # noqa: E402

from tools.xqrobotwl.verify_toe_walk_symmetry import find_swings  # noqa: E402

DT = 0.01
MODE_STAND, MODE_LIFT = 0.0, 1.0
# 序列: (name, mode, cmd[:3], duration_s)
SEQUENCE = [
    ("stand", MODE_STAND, (0.0, 0.0, 0.0), 5.0),
    ("lift_fwd", MODE_LIFT, (0.2, 0.0, 0.0), 4.0),
    ("lift_lat", MODE_LIFT, (0.0, 0.1, 0.0), 4.0),
    ("lift_turn", MODE_LIFT, (0.0, 0.0, 0.3), 4.0),
    ("lift_back", MODE_LIFT, (-0.2, 0.0, 0.0), 4.0),
    ("stand_back", MODE_STAND, (0.0, 0.0, 0.0), 5.0),
]
WARMUP_S = 1.0  # 每段前 1s 为切换过渡, 不计入段内指标 (但全程保留用于切换判定)
SWITCH_SAFE_S = 0.5  # 切换后 0.5s 内不跌倒


def _lift_stats(samples, dt: float) -> dict:
    """从接触切摆动相, 统计交替/对称 (与 verify_toe_walk_symmetry 同口径)."""
    wc = np.stack([s.wheel_contact for s in samples])
    dof = np.stack([s.dof_pos for s in samples])
    cL, cR = wc[:, 0], wc[:, 1]
    swL = find_swings(cL, dt)
    swR = find_swings(cR, dt)
    out: dict = {"n_swing_L": len(swL), "n_swing_R": len(swR)}
    if len(swL) < 2 or len(swR) < 2:
        out["verdict"] = "FAIL: 未检测到双腿交替抬腿"
        return out
    # 相位差 L→R
    midL = np.array([s["i0"] + (s["i1"] - s["i0"]) / 2 for s in swL]) * dt
    midR = np.array([s["i0"] + (s["i1"] - s["i0"]) / 2 for s in swR]) * dt
    TL = float(np.median(np.diff([s["i0"] for s in swL])) * dt)
    lags = []
    for mR in midR:
        prev_L = midL[midL < mR]
        if prev_L.size == 0:
            continue
        lags.append((mR - prev_L[-1]) % TL / TL)
    out["cycle_T_s"] = TL
    out["phase_lag_frac"] = float(np.mean(lags)) if lags else float("nan")
    # 抬腿幅度 (膝弯幅值: 摆动峰 - 支撑基线)
    knee_L, knee_R = dof[:, 2], -dof[:, 5]
    baseL = float(np.median(knee_L[cL > 0.5])) if (cL > 0.5).any() else float("nan")
    baseR = float(np.median(knee_R[cR > 0.5])) if (cR > 0.5).any() else float("nan")
    ampL = np.mean([np.max(knee_L[s["i0"] : s["i1"] + 1]) - baseL for s in swL])
    ampR = np.mean([np.max(knee_R[s["i0"] : s["i1"] + 1]) - baseR for s in swR])
    out.update(
        {
            "knee_amp_L": float(ampL),
            "knee_amp_R": float(ampR),
            "knee_amp_ratio": float(ampL / ampR) if ampR > 1e-6 else float("nan"),
        }
    )
    durL = np.mean([s["dur"] for s in swL])
    durR = np.mean([s["dur"] for s in swR])
    out["lift_dur_ratio"] = float(durL / durR) if durR > 1e-6 else float("nan")
    alt_ok = 0.35 <= out["phase_lag_frac"] <= 0.65
    sym_ok = 0.8 <= out["knee_amp_ratio"] <= 1.25 and 0.8 <= out["lift_dur_ratio"] <= 1.25
    out["verdict_alternation"] = "PASS" if alt_ok else "FAIL"
    out["verdict_symmetry"] = "PASS" if sym_ok else "FAIL"
    out["verdict"] = "PASS" if (alt_ok and sym_ok) else "FAIL"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="run 目录名 (默认最新)")
    ap.add_argument("--ckpt", default="model_9999.pt")
    args = ap.parse_args()

    task = tasks.get("toe_walk_mode")
    if args.run is not None:
        run_dir = engine.resolve_run_dir(args.run, task.log_root)
    else:
        run_dir = sorted(Path(ROOT / task.log_root).glob("*/"), key=lambda p: p.stat().st_mtime)[-1]
    ckpt_path = engine.find_checkpoint(run_dir, args.ckpt)
    env = engine.build_env(task, num_envs=1, ckpt_path=ckpt_path)
    policy = engine.load_policy(ckpt_path, env.obs_groups_spec["obs"], task.num_actions)
    try:
        env.init_state()
        all_samples: list = []
        seg_meta: list[dict] = []
        for name, mode, cmd, dur in SEQUENCE:
            n_steps = int(dur / DT)
            seg_samples = []
            fell = False
            for step in range(n_steps):
                env.state.info["commands"][:, :3] = np.asarray(cmd, dtype=np.float64)
                env.state.info["commands"][:, 4] = mode
                obs_t = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                with torch.no_grad():
                    a = policy(obs_t).numpy().astype(np.float64)
                st = env.step(a)
                if st.terminated[0] or st.truncated[0]:
                    if step * DT < SWITCH_SAFE_S:
                        fell = True
                # 都以 0-index 相对序列记录绝对 step 号
                idx = len(all_samples)
                s = engine.collect_step(env, st, idx, 0, DT)
                seg_samples.append(s)
                all_samples.append(s)
            seg_meta.append({"name": name, "mode": mode, "cmd": list(cmd), "samples": seg_samples, "fell": fell})
    finally:
        env.close()

    # ── 指标 ─────────────────────────────────────────────
    metrics: dict = {}
    for seg in seg_meta:
        seg_samples = seg["samples"]
        t = np.stack([s.time_s for s in seg_samples]) - seg_samples[0].time_s

        bp = np.stack([s.base_pos for s in seg_samples])
        up = np.array([s.up_z for s in seg_samples])
        lv = np.stack([s.linvel for s in seg_samples])
        m: dict = {"fell_switch": seg["fell"]}
        win = t >= WARMUP_S  # 排除切换过渡窗
        if seg["mode"] == MODE_STAND:
            m["stand_linvel_xy"] = float(np.mean(np.linalg.norm(lv[win, :2], axis=1)))
            m["stand_gyro"] = float(np.mean(np.linalg.norm(np.stack([s.gyro for s in seg_samples])[win], axis=1)))
            m["stand_height"] = float(np.mean(bp[win, 2]))
            m["drift_dx"] = float(bp[-1, 0] - bp[0, 0])
            m["up_z"] = float(np.mean(up[win]))
        else:
            m["tracking_rmse"] = float(
                np.sqrt(np.mean((lv[win, 0] - seg["cmd"][0]) ** 2))
            )
            m["avg_vy"] = float(np.mean(lv[win, 1]))
            m["avg_vyaw"] = float(np.mean(np.abs(np.stack([s.gyro for s in seg_samples])[win, 2])))
            if win.any():
                lift_win = seg_samples[int(np.argmax(win)) :]
                m["lift"] = _lift_stats(lift_win, DT)
            else:
                m["lift"] = {}
        metrics[seg["name"]] = m

    # 抬腿段合并交替/对称判定
    lift_segs = [k for k, v in metrics.items() if k.startswith("lift")]
    lifts = [metrics[k]["lift"] for k in lift_segs]
    for k in ("n_swing_L", "n_swing_R", "cycle_T_s", "phase_lag_frac", "knee_amp_ratio", "lift_dur_ratio", "verdict_alternation", "verdict_symmetry", "verdict"):
        vals = [l.get(k) for l in lifts if l.get(k) is not None]
        if vals:
            metrics["LIFT_ALL_" + k] = vals[0] if isinstance(vals[0], str) else float(np.mean(vals))
    metrics["vx_rmse_avg"] = float(np.mean([metrics[k]["tracking_rmse"] for k in lift_segs]))

    # 切换稳定性: 任何段切换后 0.5s 跌倒即 FAIL
    switches_fell = sum(1 for seg in seg_meta if seg["fell"])
    metrics["switch_stable"] = "PASS" if switches_fell == 0 else "FAIL"

    # 站立达标 (§1.4 / 附录A)
    stand = metrics["stand"]
    s_ok = stand["stand_linvel_xy"] < 0.2 and stand["stand_gyro"] < 1.0 and abs(stand["stand_height"] - 0.52) < 0.07
    metrics["stand_verdict"] = "PASS" if s_ok else "FAIL"

    # ── 输出 ─────────────────────────────────────────────
    out_csv = pose.POSE_OUT_DIR / f"toe_walk_mode_{ckpt_path.stem}_mode_seq.csv"
    pose.write_pose_csv(all_samples, out_csv)
    out_json = ROOT / "logs" / "eval" / f"toe_walk_mode_{ckpt_path.stem}_mode_seq.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=1))

    print("=" * 78)
    print(f"双模式点足 · 确定性评估  ckpt={ckpt_path.name}  run={run_dir.name}")
    print(f"站立模式: {metrics['stand_verdict']}  (linvel_xy={stand['stand_linvel_xy']:.3f} m/s, gyro={stand['stand_gyro']:.3f}, h={stand['stand_height']:.3f}m, 漂移={stand['drift_dx']:.2f}m)")
    print(f"抬腿模式: 交替 {metrics.get('LIFT_ALL_verdict_alternation','-')} / 对称 {metrics.get('LIFT_ALL_verdict_symmetry','-')}")
    print(f"  周期={metrics.get('LIFT_ALL_cycle_T_s',float('nan')):.3f}s 相位差={metrics.get('LIFT_ALL_phase_lag_frac',float('nan')):.2f} 膝弯比={metrics.get('LIFT_ALL_knee_amp_ratio',float('nan')):.2f} L/R抬={metrics.get('LIFT_ALL_n_swing_L','-')}/{metrics.get('LIFT_ALL_n_swing_R','-')}")
    for k in lift_segs:
        m = metrics[k]
        print(f"  [{k}] 追踪RMSE(vx)={m['tracking_rmse']:.3f}  avg_vy={m['avg_vy']:+.3f}  |gyro_z|={m['avg_vyaw']:.2f}")
    print(f"切换稳定(0.5s内不倒): {metrics['switch_stable']}")
    print(f"证据 CSV: {out_csv}\n指标 JSON: {out_json}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
