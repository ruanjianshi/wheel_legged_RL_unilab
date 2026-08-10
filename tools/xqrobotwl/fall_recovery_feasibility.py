"""P1: 跌倒恢复(平地倒地→自恢复)物理可行性验证 — xqrobotwl 轮足两足机器人

无策略纯前馈 FSM, 在 MuJoCo 中验证: 机器人横躺地面(仰卧/俯卧)后, 脚本化收腿-甩腿
能否让 18.65kg 机器人把轮子移到 CoM 下方并撑起机身, 最终达到"轮着地 + 机身可直立
+ base 高度足够"的恢复状态。

设计思路 (对齐后空翻 P1): 起身是弹道/几何大动作, 可开环脚本化; 恢复后的两轮足
倒立摆平衡是闭环调节, 交给后续 RL。P1 只回答两个静态/脚本可验证的问题:
  1. 倒地姿态稳定可达: 横躺时躯干贴地, 腿在合理关节范围内, 无自碰撞/侧翻
  2. 脚本起身可达: 收腿→甩腿(角动量)→轮子着地撑起, base_z 升到可平衡高度

FSM 相位: tuck(收腿卷身) → kick(甩腿起身) → catch(轮子落地缓冲) → lift(伸腿撑起)

用法:
  uv run tools/xqrobotwl/fall_recovery_feasibility.py            # 默认(仰卧)
  uv run tools/xqrobotwl/fall_recovery_feasibility.py --prone     # 俯卧
  uv run tools/xqrobotwl/fall_recovery_feasibility.py --render /tmp/fall_rec
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as e:  # pragma: no cover
    raise SystemExit("需要 mujoco, 请 uv sync") from e

ROOT = Path(__file__).resolve().parents[2]
XML = ROOT / "src/unilab/assets/robots/xqrobotwl/scene_flat.xml"

# ── 执行器索引 (MuJoCo order) ──
# [L_hip_roll, L_hip_pitch, L_knee, L_wheel, R_hip_roll, R_hip_pitch, R_knee, R_wheel]
A = dict(L_ROLL=0, L_PITCH=1, L_KNEE=2, L_WHEEL=3, R_ROLL=4, R_PITCH=5, R_KNEE=6, R_WHEEL=7)
LEG_IDX = [A["L_ROLL"], A["L_PITCH"], A["L_KNEE"], A["R_ROLL"], A["R_PITCH"], A["R_KNEE"]]
WHEEL_IDX = [A["L_WHEEL"], A["R_WHEEL"]]

# 默认站立腿角 (MuJoCo order, 对齐 keyframe)
DEFAULT = np.array([0.1, 0.15, 0.15, -0.1, -0.15, -0.15], dtype=np.float64)

WHEEL_R = 0.11
# 站立时轮中心高度 (~轮半径 + 腿长投影), 用于判断"轮已着地可撑起"
STANDING_BASE_Z = 0.55
# 恢复成功的判据阈值
RECOVER_UP = 0.70  # 机身直立度 (up·[0,0,1]) 达到才算恢复直立
RECOVER_Z = 0.40  # base 高度达到才算轮子撑起


def whole_com(model, data) -> np.ndarray:
    com = np.zeros(3)
    tot = 0.0
    for b in range(model.nbody):
        m = model.body_mass[b]
        if m > 0:
            com += m * data.xpos[b]
            tot += m
    return com / tot


def blend(cur, target, r: float) -> np.ndarray:
    return cur + (target - cur) * float(np.clip(r, 0.0, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser(description="开环跌倒恢复可行性验证")
    ap.add_argument("--prone", action="store_true", help="俯卧(前胸贴地) 而非默认仰卧(背贴地)")
    ap.add_argument(
        "--pitch_deg", type=float, default=None, help="显式指定倒地俯仰角(覆盖 --prone)"
    )
    ap.add_argument(
        "--start_z", type=float, default=0.14, help="reset 时 base 高度 m (躯干贴地≈0.05-0.15)"
    )
    ap.add_argument(
        "--legs", type=str, default="tuck", help="倒地时腿姿态: tuck(收腿)/straight(伸直)"
    )
    # ── 起身 FSM 参数 ──
    ap.add_argument("--t_tuck", type=float, default=0.30, help="收腿卷身时长 s")
    ap.add_argument("--tuck_hip", type=float, default=0.90, help="收腿时髋前倾幅度 rad")
    ap.add_argument("--tuck_knee", type=float, default=0.87, help="收腿时膝弯曲幅度 rad")
    ap.add_argument("--t_kick", type=float, default=0.25, help="甩腿起身时长 s")
    ap.add_argument("--kick_hip", type=float, default=-0.60, help="甩腿髋后仰幅度 rad")
    ap.add_argument("--kick_knee", type=float, default=0.30, help="甩腿膝展开幅度 rad")
    ap.add_argument("--t_catch", type=float, default=0.30, help="轮子落地缓冲时长 s")
    ap.add_argument("--catch_knee", type=float, default=0.45, help="落地缓冲膝弯曲 rad")
    ap.add_argument("--t_lift", type=float, default=0.60, help="伸腿撑起时长 s")
    ap.add_argument("--lift_hip", type=float, default=0.15, help="撑起时髋角 rad")
    ap.add_argument(
        "--wheel_spin", type=float, default=8.0, help="起身后期轮子驱动 rad/s (助力前移/平衡)"
    )
    ap.add_argument("--sim_dt", type=float, default=0.005, help="sim 步长 s")
    ap.add_argument("--render", type=str, default="", help="输出目录(非空则离屏渲染→mp4)")
    ap.add_argument("--render_every", type=int, default=2, help="每 N 个 sim 步渲染一帧")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(XML))
    model.opt.timestep = args.sim_dt
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # ── 倒地姿态 (base 绕世界 Y 旋转) ──
    # 仰卧(supine): 背贴地, 前胸朝上 → R_y(-90°); 俯卧(prone): 前胸贴地 → R_y(+90°)
    pitch_deg = args.pitch_deg
    if pitch_deg is None:
        pitch_deg = 90.0 if args.prone else -90.0
    th = np.radians(pitch_deg)
    c, s = np.cos(th / 2), np.sin(th / 2)
    # qpos: [x,y,z, qw,qx,qy,qz, 8 joints]
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = args.start_z
    data.qpos[3:7] = [c, 0.0, s, 0.0]  # 绕世界 Y 旋转
    if args.legs == "tuck":
        legs = np.array([0.1, args.tuck_hip, args.tuck_knee, -0.1, -args.tuck_hip, -args.tuck_knee])
    else:  # straight
        legs = DEFAULT.copy()
    data.qpos[7:15] = [legs[0], legs[1], legs[2], 0.0, legs[3], legs[4], legs[5], 0.0]
    mujoco.mj_forward(model, data)

    renderer = None
    frame_count = 0
    if args.render:
        render_path = Path(args.render)
        mp4_out = (
            render_path
            if render_path.suffix.lower() == ".mp4"
            else render_path / "fall_recovery.mp4"
        )
        render_dir = render_path.parent if render_path.suffix.lower() == ".mp4" else render_path
        render_dir.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, 480, 640)
        print(f"渲染帧到: {render_dir} → {mp4_out}")

    # ── 1) 静态检查: 倒地姿态稳定性 ──
    print("=" * 60)
    print("P1: 跌倒恢复可行性验证")
    print("=" * 60)
    up = data.xmat[1].reshape(3, 3)[:, 2]
    com = whole_com(model, data)
    print(
        f"  1) 倒地姿态 ({'仰卧' if not args.prone and pitch_deg < 0 else '俯卧'} pitch={pitch_deg}°)"
    )
    print(f"     base_z     = {data.qpos[2]:.3f} m  (躯干厚度~0.12, 贴地≈0.05-0.10)")
    print(f"     base up    = ({up[0]:+.2f}, {up[1]:+.2f}, {up[2]:+.2f})")
    print(f"     CoM        = ({com[0]:+.3f}, {com[1]:+.3f}, {com[2]:+.3f})")
    torso_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "base_link_collision")
    torso_z = data.geom_xpos[torso_geom][2]
    print(f"     躯干底面 z = {torso_z - 0.06:.3f} m  (≈0 即贴地)")
    # 轮子相对地面: 轮心 z - WHEEL_R (正=离地, 负=压地)
    lw = data.xpos[data.body("left_link_wheel").id]
    rw = data.xpos[data.body("right_link_wheel").id]
    print(f"     左轮心离地 = {lw[2] - WHEEL_R:+.3f}  右轮心离地 = {rw[2] - WHEEL_R:+.3f}")

    # ── 2) 脚本起身 FSM ──
    print("  2) 脚本起身 FSM (tuck→kick→catch→lift)")
    dt = args.sim_dt
    # 相位: (时长, 名字) — 用绝对时间驱动
    phases: list[tuple[float, str]] = [
        (args.t_tuck, "tuck"),
        (args.t_kick, "kick"),
        (args.t_catch, "catch"),
        (args.t_lift, "lift"),
    ]
    total_steps = int(sum(p for p, _ in phases) / dt)

    max_base_z = 0.0
    best_up = 0.0
    best_z = 0.0
    best_t = 0.0
    wheel_contact_time = 0.0  # 两轮同时着地累计时间
    phase_idx = 0
    phase_t = 0.0
    phase_snapshots: dict[str, tuple[float, float]] = {}
    traj: list[
        tuple[float, float, float, float, float]
    ] = []  # (t, base_z, up, x, wheel_contact_both)
    rec_point: dict[str, float] | None = None  # 首个"轮着地+直立"恢复点 (RL 交接姿态)

    for step in range(total_steps):
        if phase_idx < len(phases) and phase_t >= phases[phase_idx][0]:
            phase_idx += 1
            phase_t = 0.0
        phase_t += dt
        phase_name = phases[min(phase_idx, len(phases) - 1)][1]

        # ── 前馈 ctrl (MuJoCo order) ──
        ctrl = np.zeros(8)
        r = np.clip(phase_t / 0.10, 0, 1)
        if phase_name == "tuck":
            # 从倒地姿态收腿卷身: 髋前倾 + 膝弯曲, 把轮子拉近身体
            ctrl[LEG_IDX] = np.array(
                [
                    0.1,
                    blend(0.15, args.tuck_hip, r),
                    blend(0.15, args.tuck_knee, r),
                    -0.1,
                    blend(-0.15, -args.tuck_hip, r),
                    blend(-0.15, -args.tuck_knee, r),
                ]
            )
        elif phase_name == "kick":
            # 甩腿: 髋猛后仰 + 膝展开 — 用腿的角动量把躯干甩起来 (向轮子方向)
            ctrl[LEG_IDX] = np.array(
                [
                    0.1,
                    blend(args.tuck_hip, args.kick_hip, r),
                    blend(args.tuck_knee, args.kick_knee, r),
                    -0.1,
                    blend(-args.tuck_hip, -args.kick_hip, r),
                    blend(-args.tuck_knee, -args.kick_knee, r),
                ]
            )
            ctrl[WHEEL_IDX] = [args.wheel_spin * r, args.wheel_spin * r]
        elif phase_name == "catch":
            # 轮子落地缓冲: 弯膝吸能, 轮子驱动助力
            ctrl[LEG_IDX] = np.array(
                [
                    0.1,
                    args.kick_hip,
                    blend(args.kick_knee, args.catch_knee, r),
                    -0.1,
                    -args.kick_hip,
                    blend(-args.kick_knee, -args.catch_knee, r),
                ]
            )
            ctrl[WHEEL_IDX] = [args.wheel_spin, args.wheel_spin]
        elif phase_name == "lift":
            # 伸腿撑起: 髋回中立 + 膝伸直 → base 升高到平衡高度
            ctrl[LEG_IDX] = np.array(
                [
                    0.1,
                    blend(args.kick_hip, args.lift_hip, r),
                    blend(args.catch_knee, 0.15, r),
                    -0.1,
                    blend(-args.kick_hip, -args.lift_hip, r),
                    blend(-args.catch_knee, -0.15, r),
                ]
            )
            ctrl[WHEEL_IDX] = [args.wheel_spin, args.wheel_spin]

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        base_z = data.qpos[2]
        upv = data.xmat[1].reshape(3, 3)[:, 2]
        up_align = upv[2]
        lwz = data.xpos[data.body("left_link_wheel").id][2]
        rwz = data.xpos[data.body("right_link_wheel").id][2]
        both_wheels = (lwz - WHEEL_R < 0.03) and (rwz - WHEEL_R < 0.03)
        if both_wheels:
            wheel_contact_time += dt
        # 恢复点: 首个 轮着地 + 直立度>0.9 + 高度够 的时刻 (RL 交接姿态)
        if rec_point is None and both_wheels and up_align > 0.9 and base_z > 0.25:
            rec_point = {
                "t": step * dt,
                "base_z": base_z,
                "up": up_align,
                "x": data.qpos[0],
                "legs": np.round(data.qpos[7:13].copy(), 3),
                "qvel": np.round(np.concatenate([data.qvel[3:7], data.qvel[7:13]]), 2),
            }
        max_base_z = max(max_base_z, base_z)
        if up_align > best_up:
            best_up, best_z, best_t = up_align, base_z, step * dt
        if abs(phase_t - phases[phase_idx][0]) < dt * 0.6:
            phase_snapshots[phase_name] = (up_align, base_z)
        # 每 0.1s 记录轨迹
        if step % max(int(0.1 / dt), 1) == 0:
            traj.append((step * dt, base_z, up_align, data.qpos[0], 1.0 if both_wheels else 0.0))

        if renderer is not None and step % args.render_every == 0:
            renderer.update_scene(data)
            cam = renderer.scene.camera[0]
            rx = data.qpos[0]
            cam.pos[:] = [rx + 0.5, -3.4, 1.2]
            fwd = np.array([0.0, 3.4, -0.5]) - np.array([0.5, 0.0, 0.0])
            cam.forward[:] = fwd / np.linalg.norm(fwd)
            cam.up[:] = [0.0, 0.0, 1.0]
            img = renderer.render()
            with open(render_dir / f"frame_{frame_count:05d}.ppm", "wb") as f:
                h, w = img.shape[:2]
                f.write(f"P6\n{w} {h}\n255\n".encode())
                f.write(img.tobytes())
            frame_count += 1

    # ── 结果 ──
    up_final = data.xmat[1].reshape(3, 3)[:, 2]
    tilt = np.arccos(np.clip(up_final[2], -1, 1)) * 180 / np.pi
    base_z = data.qpos[2]
    lwz = data.xpos[data.body("left_link_wheel").id][2]
    rwz = data.xpos[data.body("right_link_wheel").id][2]
    print("  3) 起身结果")
    print(f"     base_z   = {base_z:.3f} m   (撑起目标 ≈{RECOVER_Z}+, 站立 ≈{STANDING_BASE_Z})")
    print(f"     结束倾角 = {tilt:.1f}°  (直立=0°, 需能稳住才有意义)")
    print(f"     左/右轮  = 离地 {lwz - WHEEL_R:+.2f} / {rwz - WHEEL_R:+.2f} m (≈0=着地)")
    print(f"     最佳直立度 = {best_up:.3f} @ t={best_t:.2f}s z={best_z:.3f}")
    print(f"     最大 base_z = {max_base_z:.3f}")
    print(f"     两轮同时着地累计 = {wheel_contact_time:.2f}s")
    if rec_point is not None:
        print(
            f"     ★ 恢复点 (轮着地+直立>0.9): t={rec_point['t']:.2f}s "
            f"z={rec_point['base_z']:.3f} up={rec_point['up']:.2f} x={rec_point['x']:+.2f}"
        )
        print(
            f"       腿角 qpos[7:13] = {rec_point['legs']}  (MuJoCo序: L_roll L_pitch L_knee R_roll R_pitch R_knee)"
        )
        print(f"       速度 qvel       = {rec_point['qvel']}")
    for ph, (u, z) in phase_snapshots.items():
        print(f"     [{ph:5s}] 结束: up={u:+.2f} base_z={z:.2f}")
    print("     ── 轨迹 (t, base_z, up, x, 双轮着地) ──")
    for t, z, u, x, wc in traj:
        print(f"       t={t:5.2f}  z={z:.3f}  up={u:+.2f}  x={x:+.2f}  wheels={'✓' if wc else '·'}")

    rec_ok = (best_up > RECOVER_UP) and (max_base_z > RECOVER_Z) and (wheel_contact_time > 0.2)
    print("=" * 60)
    if rec_ok:
        print("✅ 起身可行: 轮子着地 + 机身直立度可达 + base 高度足够 → 进入 RL")
        print("   RL 结构: FSM(tuck/kick/catch FF) + 两轮倒立摆平衡(RL闭环)")
    else:
        print("❌ 起身未达标, 需调参数 (先试 tuck_knee/tuck_hip/kick_hip/wheel_spin)")
    print("=" * 60)

    if renderer is not None:
        import subprocess

        fps = round(1.0 / (dt * args.render_every))
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(render_dir / "frame_%05d.ppm"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            str(mp4_out),
        ]
        print(f"组装视频: ffmpeg {fps}fps → {mp4_out}")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ 视频已生成: {mp4_out}")
        for f in render_dir.glob("frame_*.ppm"):
            f.unlink()


if __name__ == "__main__":
    main()
