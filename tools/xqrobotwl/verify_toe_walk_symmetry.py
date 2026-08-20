#!/usr/bin/env python3
"""点足抬腿行走 — 左右交替与对称性验证 (CLAUDE.md §7.3 / §1.3).

评估问题 (老板验收):
  1. 最新模型是否正常"左一下、右一下"交替抬腿?
  2. 左右腿抬腿幅度是否近似 (对称)?

方法:
  - 用训练的 run_config.json 重建环境 (最忠实), 加载 ckpt 确定性 actor
  - 零速度指令下连续跑 N 步 (相位时钟步态持续, 原地点足)
  - 逐帧采集 6 关节角 + 轮地接触 (engine.collect_step 26 列对齐)
  - 从轮地接触切出左右摆动相 (swing) → 统计:
      * 交替性: 摆动相相位差 (期望 ≈0.5 周期)、同时离地/同时着地占比
      * 抬腿幅度: 膝弯幅值 (摆动相峰值-支撑相基线)、大腿摆动幅度、离地时长
      * 对称性: L/R 各指标比值、单侧逐周期一致性 (std/mean)
  - 导出姿态数据 CSV (logs/pose_data/, 数据优先) + 打印报告

用法:
  uv run python tools/xqrobotwl/verify_toe_walk_symmetry.py \
      [--run 2026-08-14_14-15-47_mujoco] [--ckpt model_9999.pt] [--steps 900]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _devlog.assess import engine, pose, tasks  # noqa: E402

MIN_SWING_STEPS = 3  # 摆动相最短步数 (30ms), 过滤接触抖动
WARMUP_S = 1.0  # 跳过启动瞬态


def find_swings(contact: np.ndarray, dt: float) -> list[dict]:
    """从接触序列切出摆动相区间 [{t0,t1,dur,mid,peak_knee,exc}]."""
    swings: list[dict] = []
    n = len(contact)
    i = 0
    while i < n:
        if contact[i] == 0:
            j = i
            while j < n and contact[j] == 0:
                j += 1
            if j - i >= MIN_SWING_STEPS:
                swings.append({"i0": i, "i1": j - 1, "dur": (j - i) * dt})
            i = j
        else:
            i += 1
    return swings


def analyze(samples, dt: float, out_csv: Path) -> dict:
    dof = np.stack([s.dof_pos for s in samples])  # (N,6) L_roll,L_pitch,L_knee,R_roll,R_pitch,R_knee
    wc = np.stack([s.wheel_contact for s in samples])  # (N,2)
    bp = np.stack([s.base_pos for s in samples])
    up = np.array([s.up_z for s in samples])
    cL, cR = wc[:, 0], wc[:, 1]
    t = np.array([s.time_s for s in samples])

    # ── 摆动相切分 (每脚) ──────────────────────────────
    swL = find_swings(cL, dt)  # L 轮离地
    swR = find_swings(cR, dt)

    # 支撑相膝角基线 (该脚着地期 10%-90% 分位中值)
    def stance_med(knee_col: np.ndarray, contact_col: np.ndarray) -> float:
        v = knee_col[contact_col > 0.5]
        return float(np.median(v)) if v.size else float(np.nan)

    knee_L, knee_R = dof[:, 2], -dof[:, 5]  # R 取负 → 双侧都是"正=弯膝"
    pitch_L, pitch_R = dof[:, 1], -dof[:, 4]  # 大腿前摆幅度 (幅值用)
    baseL = stance_med(knee_L, cL)
    baseR = stance_med(knee_R, cR)

    for s in swL:
        s["leg"] = "L"
        s["knee_amp"] = float(np.max(knee_L[s["i0"] : s["i1"] + 1]) - baseL)
        s["pitch_exc"] = float(np.max(pitch_L[s["i0"] : s["i1"] + 1]) - np.min(pitch_L[s["i0"] : s["i1"] + 1]))
        s["peak_knee"] = float(np.max(knee_L[s["i0"] : s["i1"] + 1]))
    for s in swR:
        s["leg"] = "R"
        s["knee_amp"] = float(np.max(knee_R[s["i0"] : s["i1"] + 1]) - baseR)
        s["pitch_exc"] = float(np.max(pitch_R[s["i0"] : s["i1"] + 1]) - np.min(pitch_R[s["i0"] : s["i1"] + 1]))
        s["peak_knee"] = float(np.max(knee_R[s["i0"] : s["i1"] + 1]))

    n_sw = len(swL) + len(swR)
    res: dict = {"n_windows_s": float(len(samples) * dt), "n_swing_L": len(swL), "n_swing_R": len(swR)}

    # 证据 CSV (先写, 任何结论都有数据)
    pose.write_pose_csv(samples, out_csv)
    res["csv"] = str(out_csv)

    if n_sw == 0:
        res["verdict"] = "NO_SWING: 轮子从未离地, 无点足抬腿动作"
        return res

    # ── 交替性 ─────────────────────────────────────────
    # 相位差: 每对相邻 L/R 摆动中点时间差 / 当地周期
    midL = np.array([s["i0"] + (s["i1"] - s["i0"]) / 2 for s in swL]) * dt
    midR = np.array([s["i0"] + (s["i1"] - s["i0"]) / 2 for s in swR]) * dt
    # 周期: 相邻 L 摆动起点间隔
    if len(swL) >= 2:
        T = float(np.median(np.diff([s["i0"] for s in swL])) * dt)
    elif len(swR) >= 2:
        T = float(np.median(np.diff([s["i0"] for s in swR])) * dt)
    else:
        T = float("nan")
    res["cycle_T_s"] = T
    lags = []
    for mR in midR:
        prev_L = midL[midL < mR]
        if prev_L.size == 0:
            continue
        dl = mR - prev_L[-1]
        lags.append((dl % T) / T if T > 0 else float("nan"))
    if len(midL) and len(midR):
        res["phase_lag_LtoR_frac"] = float(np.mean(lags)) if lags else float("nan")
        res["phase_lag_std"] = float(np.std(lags)) if lags else float("nan")
    # 时间分配: 仅一脚离地 / 双离地 / 双着地
    one = float(np.mean((cL == 0) ^ (cR == 0)))
    both_off = float(np.mean((cL == 0) & (cR == 0)))
    both_on = float(np.mean((cL > 0) & (cR > 0)))
    res["frac_exactly_one_air"] = one
    res["frac_both_air"] = both_off
    res["frac_both_ground"] = both_on

    # ── 抬腿幅度 + 对称性 ──────────────────────────────
    def stats(side_swings, key):
        vals = [s[key] for s in side_swings]
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

    for m in ("dur", "knee_amp", "pitch_exc", "peak_knee"):
        res[f"L_{m}_mean"], res[f"L_{m}_std"] = stats(swL, m)
        res[f"R_{m}_mean"], res[f"R_{m}_std"] = stats(swR, m)
        a = res.get(f"L_{m}_mean")
        b = res.get(f"R_{m}_mean")
        if a and b and a > 1e-6:
            res[f"ratio_R/L_{m}"] = b / a  # 1.0 = 完全对称
    res["L_knee_stance_baseline"] = baseL
    res["R_knee_stance_baseline"] = baseR

    # ── 稳定性背景 ─────────────────────────────────────
    res["base_z_mean"] = float(np.mean(bp[:, 2]))
    res["base_z_std"] = float(np.std(bp[:, 2]))
    res["up_z_mean"] = float(np.mean(up))
    res["drift_dx_m"] = float(bp[-1, 0] - bp[0, 0])
    res["drift_dy_m"] = float(bp[-1, 1] - bp[0, 1])
    eul = np.stack([s.euler for s in samples])
    res["yaw_accum_deg"] = float(np.abs(np.rad2deg(eul[-1, 2] - eul[0, 2])))
    lv = np.stack([s.linvel for s in samples])
    res["mean_linvel_xy"] = float(np.mean(np.linalg.norm(lv[:, :2], axis=1)))

    # ── 裁定 ───────────────────────────────────────────
    nL, nR = len(swL), len(swR)
    both_lift = nL >= 2 and nR >= 2
    if not both_lift:
        res["verdict_alternation"] = "FAIL"
        res["verdict_symmetry"] = "FAIL"
        res["note"] = f"单边抬腿: L={nL} 次, R={nR} 次 → 非交替步态"
        return res
    sym_ok = (
        res.get("ratio_R/L_knee_amp", 0) > 0.8
        and res.get("ratio_R/L_knee_amp", 9) < 1.25
        and res.get("ratio_R/L_dur", 0) > 0.8
        and res.get("ratio_R/L_dur", 9) < 1.25
    )
    alt_ok = (
        res.get("phase_lag_LtoR_frac", 0) > 0.35
        and res.get("phase_lag_LtoR_frac", 0) < 0.65
        and res.get("frac_exactly_one_air", 0) > 0.8
        and res.get("frac_both_air", 0) < 0.05
    )
    res["verdict_alternation"] = "PASS" if alt_ok else "FAIL"
    res["verdict_symmetry"] = "PASS" if sym_ok else "FAIL"
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="2026-08-14_14-15-47_mujoco", help="run 目录名")
    ap.add_argument("--ckpt", default="model_9999.pt", help="checkpoint 文件名")
    ap.add_argument("--steps", type=int, default=900, help="连续运行步数 (0.01s/步, 0.7s 周期 → 900 步≈12.8 周期)")
    ap.add_argument("--cmd", default="0,0,0,0", help="速度命令 vx,vy,vyaw,tsk (默认零速; 维度自适应)")
    ap.add_argument("--num_envs", type=int, default=1)
    args = ap.parse_args()

    task = tasks.get("toe_walk")
    run_dir = engine.resolve_run_dir(args.run, task.log_root)
    ckpt_path = engine.find_checkpoint(run_dir, args.ckpt)
    env = engine.build_env(task, num_envs=args.num_envs, ckpt_path=ckpt_path)
    try:
        obs_dim = env.obs_groups_spec["obs"]
        policy = engine.load_policy(ckpt_path, obs_dim, task.num_actions)
        dt = env._cfg.ctrl_dt
        cmd = np.asarray([float(x) for x in args.cmd.split(",")], dtype=np.float64)

        env.init_state()
        samples = []
        with np.errstate(all="ignore"):
            n_cmd = env.state.info["commands"].shape[1]
            cmd_full = np.zeros((env.num_envs, n_cmd), dtype=np.float64)
            cmd_full[:, : min(n_cmd, cmd.size)] = cmd[: min(n_cmd, cmd.size)]
            env.state.info["commands"][:, :n_cmd] = cmd_full
            with torch.no_grad():
                for step in range(args.steps):
                    obs_t = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                    a = policy(obs_t).numpy().astype(np.float64)
                    env.state.info["commands"][:, :n_cmd] = cmd_full
                    st = env.step(a)
                    if step * dt >= WARMUP_S:
                        samples.append(engine.collect_step(env, st, step, 0, dt))
    finally:
        env.close()

    if not samples:
        print("无样本 (episode 全程早死?)")
        return 1

    out_csv = pose.POSE_OUT_DIR / f"toe_walk_{run_dir.name}_{ckpt_path.stem}_symcheck.csv"
    res = analyze(samples, dt, out_csv)

    # ── 打印报告 ───────────────────────────────────────
    print("=" * 72)
    print(f"点足抬腿行走 · 左右交替/对称性验证   ckpt={ckpt_path.name}  run={run_dir.name}")
    print(f"零指令 {res['n_windows_s']:.1f}s 窗口  判定: 交替 {res.get('verdict_alternation','-')} / 对称 {res.get('verdict_symmetry','-')}")
    print("-" * 72)
    if str(res.get("verdict", "")).startswith("NO_SWING"):
        print(res["verdict"])
        return 2
    if res.get("note"):
        print(f"⚠ {res['note']}")
        print(f"步态周期 T = {res['cycle_T_s']:.3f}s  (设计 cycle_time=0.7s)")
        print(f"时间分配: 仅一脚离地 {res['frac_exactly_one_air']*100:.1f}% | 双脚同时离地 {res['frac_both_air']*100:.1f}% | 双脚着地 {res['frac_both_ground']*100:.1f}%")
        print(f"机身: z={res['base_z_mean']:.3f}±{res['base_z_std']:.3f} m  up_z={res['up_z_mean']:.3f}  漂移 dx={res['drift_dx_m']:.2f} m dy={res['drift_dy_m']:.2f} m  yaw累计={res['yaw_accum_deg']:.1f}°  平均线速={res['mean_linvel_xy']:.2f} m/s")
        print(f"证据 CSV: {res['csv']}")
        print("=" * 72)
        return 2
    print(f"步态周期 T = {res['cycle_T_s']:.3f}s  (设计 cycle_time=0.7s)")
    print(f"摆动相相位差 L→R = {res['phase_lag_LtoR_frac']:.2f} 周期 (±{res['phase_lag_std']:.2f})  [≈0.5=严格反相]")
    print(f"时间分配: 仅一脚离地 {res['frac_exactly_one_air']*100:.1f}% | 双脚同时离地 {res['frac_both_air']*100:.1f}% | 双脚着地 {res['frac_both_ground']*100:.1f}%")
    print("-" * 72)
    print(f"{'指标':<22}{'左腿 L':>12}{'右腿 R':>12}{'R/L 比值':>10}")
    for m, unit in (("dur", "s"), ("knee_amp", "rad"), ("pitch_exc", "rad"), ("peak_knee", "rad")):
        print(f"{m:<22}{res.get(f'L_{m}_mean',float('nan')):>10.3f}{res.get(f'R_{m}_mean',float('nan')):>12.3f}{res.get(f'ratio_R/L_{m}',float('nan')):>10.3f}  ({unit})")
    print(f"摆动次数: L={res['n_swing_L']}  R={res['n_swing_R']}")
    print(f"支撑相膝角基线: L={res['L_knee_stance_baseline']:.3f} rad  R={res['R_knee_stance_baseline']:.3f} rad")
    print("-" * 72)
    print(f"机身: z={res['base_z_mean']:.3f}±{res['base_z_std']:.3f} m  up_z={res['up_z_mean']:.3f}  漂移 dx={res['drift_dx_m']:.2f} m dy={res['drift_dy_m']:.2f} m  yaw累计={res['yaw_accum_deg']:.1f}°  平均线速={res['mean_linvel_xy']:.2f} m/s")
    print(f"证据 CSV: {res['csv']}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())