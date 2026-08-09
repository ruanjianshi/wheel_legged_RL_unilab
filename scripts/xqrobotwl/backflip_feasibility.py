"""P1: 开环脚本后空翻物理可行性验证 — xqrobotwl 轮足两足机器人

无策略纯前馈 FSM, 在 MuJoCo 中验证: 轮子急加速(翘头) + 爆发蹬地能否让
18.65kg 机器人攒够角动量转完 360°, 并校准俯仰方向/起跳时机/扭矩峰值。

执行器顺序 (MuJoCo order): [L_hip_roll, L_hip_pitch, L_knee, L_wheel,
                             R_hip_roll, R_hip_pitch, R_knee, R_wheel]
  6 个位置执行器(kp=30), 2 个速度执行器(kv=1, 目标 rad/s)。

FSM 相位: crouch(蹲+轮加速) → launch(爆发蹬地) → flight(空中收腿tuck)
          → deploy(落地前展开+轮对地匹配) → land(缓冲) → recover(恢复)

用法:
  uv run python scripts/xqrobotwl/backflip_feasibility.py            # 默认参数
  uv run python scripts/xqrobotwl/backflip_feasibility.py --W 45 --tlaunch 0.18
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
A = dict(L_ROLL=0, L_PITCH=1, L_KNEE=2, L_WHEEL=3, R_ROLL=4, R_PITCH=5, R_KNEE=6, R_WHEEL=7)
LEG_IDX = [A["L_ROLL"], A["L_PITCH"], A["L_KNEE"], A["R_ROLL"], A["R_PITCH"], A["R_KNEE"]]
WHEEL_IDX = [A["L_WHEEL"], A["R_WHEEL"]]

# 默认姿态 (MuJoCo order, 对齐 keyframe)
DEFAULT = np.array([0.1, 0.15, 0.15, 0.0, -0.1, -0.15, -0.15, 0.0])


def build_ctrl(ff_legs: np.ndarray, ff_wheels: np.ndarray) -> np.ndarray:
    """组装 8 维 MuJoCo ctrl: [legs..., wheels...]"""
    ctrl = np.zeros(8)
    ctrl[LEG_IDX] = ff_legs
    ctrl[WHEEL_IDX] = ff_wheels
    return ctrl


def blend(cur, target, r: float) -> np.ndarray:
    """插值 cur->target, r∈[0,1]"""
    return cur + (target - cur) * float(np.clip(r, 0.0, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser(description="开环脚本后空翻可行性验证")
    ap.add_argument("--W", type=float, default=45.0, help="轮子急加速目标 rad/s")
    ap.add_argument("--spinup", type=float, default=0.30, help="蹲+轮加速时长 s")
    ap.add_argument("--crouch_knee", type=float, default=0.45, help="蹲下膝弯曲深度 rad")
    ap.add_argument(
        "--crouch_hip", type=float, default=0.10, help="蹲时髋角(负=后仰, 抵消蹲的前翻)"
    )
    ap.add_argument("--tlaunch", type=float, default=0.18, help="爆发蹬地时长 s")
    ap.add_argument("--launch_knee", type=float, default=0.87, help="蹬地膝伸到极限幅度 rad")
    ap.add_argument("--launch_lean", type=float, default=0.50, help="蹬地时髋后仰幅度 rad")
    ap.add_argument("--tflight", type=float, default=0.45, help="飞行收腿时长 s")
    ap.add_argument("--tuck", type=float, default=0.70, help="收腿深度 rad")
    ap.add_argument("--tdeploy", type=float, default=0.18, help="落地前展开时长 s")
    ap.add_argument("--tland", type=float, default=0.15, help="落地缓冲时长 s")
    ap.add_argument("--trecover", type=float, default=0.40, help="恢复时长 s")
    ap.add_argument("--sim_dt", type=float, default=0.005, help="sim 步长 s")
    ap.add_argument(
        "--render", type=str, default="", help="输出目录(非空则离屏渲染→mp4, 如 /tmp/backflip)"
    )
    ap.add_argument("--render_every", type=int, default=2, help="每 N 个 sim 步渲染一帧")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(XML))
    model.opt.timestep = args.sim_dt
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # ── 渲染 (离屏, 侧面视角跟随机器人) ──
    renderer = None
    frame_count = 0
    if args.render:
        render_path = Path(args.render)
        if render_path.suffix.lower() == ".mp4":
            render_dir = render_path.parent
            mp4_out = render_path
        else:
            render_dir = render_path
            mp4_out = render_path / "backflip.mp4"
        render_dir.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, 480, 640)
        print(f"渲染帧到: {render_dir} → {mp4_out} (每 {args.render_every} 步一帧)")

    dt = args.sim_dt

    # FSM 阶段: (时长, ctrl 构造)
    # 用绝对时间驱动, 便于观测
    phases: list[tuple[float, str]] = [
        (args.spinup, "crouch"),
        (args.tlaunch, "launch"),
        (args.tflight, "flight"),
        (args.tdeploy, "deploy"),
        (args.tland, "land"),
        (args.trecover, "recover"),
    ]
    total_steps = int(sum(p for p, _ in phases) / dt)

    flip_progress = 0.0
    max_height = 0.0
    max_vz = 0.0
    max_torque = np.zeros(8)
    t_global = 0.0
    phase_idx = 0
    phase_t = 0.0
    phase_snapshots: dict[str, tuple[float, float, float]] = {}  # 各相位结束时的 (progress, vz, z)
    phase_up: dict[str, np.ndarray] = {}  # 各相位结束时 base up 向量

    # 记录飞行状态
    airborne_steps = 0
    max_wheel_lift = 0.0
    min_wheel_z = 1e9
    wheel_geom_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_link_wheel_collision")
    wheel_geom_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_link_wheel_collision")
    prev_ctrl = build_ctrl(DEFAULT[LEG_IDX], np.zeros(2))

    for step in range(total_steps):
        # 当前相位
        if phase_idx < len(phases) and phase_t >= phases[phase_idx][0]:
            phase_idx += 1
            phase_t = 0.0
        phase_t += dt
        phase_name = phases[min(phase_idx, len(phases) - 1)][1]

        # ── 前馈 ctrl (MuJoCo order) ──
        ctrl = prev_ctrl.copy()
        if phase_name == "crouch":
            # 深蹲(适中) + 轮子急加速(前向→翘头后仰)
            r = np.clip(phase_t / 0.08, 0, 1)
            legs = np.array(
                [
                    0.1,
                    args.crouch_hip,
                    blend(0.15, args.crouch_knee, r),  # L
                    -0.1,
                    -args.crouch_hip,
                    blend(-0.15, -args.crouch_knee, r),
                ]
            )  # R
            wheels = np.array([args.W * r, args.W * r])
            ctrl = build_ctrl(legs, wheels)
        elif phase_name == "launch":
            # 爆发蹬地: 髋后仰(后翻方向) + 膝猛伸到极限推地, 轮保持高速
            r = np.clip(phase_t / 0.07, 0, 1)
            hip_l = blend(0.10, -args.launch_lean, r)
            hip_r = blend(-0.10, args.launch_lean, r)
            legs = np.array(
                [
                    0.1,
                    hip_l,
                    blend(args.crouch_knee, -args.launch_knee, r),
                    -0.1,
                    hip_r,
                    blend(-args.crouch_knee, args.launch_knee, r),
                ]
            )
            wheels = np.array([args.W, args.W])
            ctrl = build_ctrl(legs, wheels)
        elif phase_name == "flight":
            # 空中收腿 tuck: 减小转动惯量→加速旋转; 轮子刹车→角动量传给机体
            r = np.clip(phase_t / 0.15, 0, 1)
            legs = np.array(
                [0.1, 0.10, blend(0.0, args.tuck, r), -0.1, -0.10, blend(0.0, -args.tuck, r)]
            )
            wheels = np.array([0.0, 0.0])
            ctrl = build_ctrl(legs, wheels)
        elif phase_name == "deploy":
            # 落地前展开: 伸腿 + 轮子对准前进方向
            r = np.clip(phase_t / 0.12, 0, 1)
            legs = np.array(
                [0.1, 0.30, blend(args.tuck, 0.10, r), -0.1, -0.30, blend(-args.tuck, -0.10, r)]
            )
            wheels = np.array([5.0, 5.0])
            ctrl = build_ctrl(legs, wheels)
        elif phase_name == "land":
            # 落地缓冲: 弯膝吸能
            r = np.clip(phase_t / 0.08, 0, 1)
            legs = np.array([0.1, 0.15, blend(0.10, 0.50, r), -0.1, -0.15, blend(-0.10, -0.50, r)])
            wheels = np.array([0.0, 0.0])
            ctrl = build_ctrl(legs, wheels)
        elif phase_name == "recover":
            r = np.clip(phase_t / 0.20, 0, 1)
            legs = DEFAULT[LEG_IDX]
            wheels = np.zeros(2)
            ctrl = build_ctrl(legs, wheels)
            # 直接朝向默认
            ctrl = blend(prev_ctrl, ctrl, r)

        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        prev_ctrl = ctrl
        t_global += dt

        # ── 测量 ──
        # 世界系 Y 角速度积分 → 俯仰翻转进度 (后空翻方向需校准符号)
        flip_progress += data.qvel[4] * dt
        base_z = data.qpos[2]
        vz = data.qvel[2]
        max_height = max(max_height, base_z)
        max_vz = max(max_vz, abs(vz))
        max_torque = np.maximum(
            max_torque, np.abs(np.asarray(data.actuator_force, dtype=np.float64))
        )
        # 相位快照
        if phase_idx < len(phases) and abs(phase_t - phases[phase_idx][0]) < dt * 0.6:
            phase_snapshots[phase_name] = (flip_progress, vz, base_z)
            phase_up[phase_name] = data.xmat[1].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])

        # 渲染一帧 (侧面视角跟随机器人)
        if renderer is not None and step % args.render_every == 0:
            renderer.update_scene(data)
            cam = renderer.scene.camera[0]
            rx = data.qpos[0]
            cam.pos[:] = [rx + 0.5, -3.4, 1.2]
            fwd = np.array([0.0, 3.4, -0.5]) - np.array([0.5, 0.0, 0.0])
            cam.forward[:] = fwd / np.linalg.norm(fwd)
            cam.up[:] = [0.0, 0.0, 1.0]
            img = renderer.render()
            # 写 PPM (无依赖), 后续 ffmpeg 组装
            with open(render_dir / f"frame_{frame_count:05d}.ppm", "wb") as f:
                h, w = img.shape[:2]
                f.write(f"P6\n{w} {h}\n255\n".encode())
                f.write(img.tobytes())
            frame_count += 1
        # 腾空判断: 轮中心高于站立接触高度 + 裕量 (轮中心站立时 ~0.29)
        wheel_z = min(data.geom_xpos[wheel_geom_l][2], data.geom_xpos[wheel_geom_r][2])
        if wheel_z > 0.35:
            airborne_steps += 1
        # 轮子中心高度: 跟踪最高与最低
        max_wheel_lift = max(max_wheel_lift, wheel_z)
        min_wheel_z = min(min_wheel_z, wheel_z)

    # ── 结果 ──
    base_z = data.qpos[2]
    up = data.xmat[1].reshape(3, 3)[:, 2]  # base_link 世界系 z 轴 (body id 1 = base_link)
    tilt = np.arccos(np.clip(up[2], -1, 1)) * 180 / np.pi
    print("=" * 56)
    print("开环脚本后空翻结果")
    print("=" * 56)
    print(
        f"  翻转进度(俯仰积分): {flip_progress:+.2f} rad  ({flip_progress / (2 * np.pi) * 100:+.0f}% of 360°)"
    )
    print(f"  最高点 base_z:      {max_height:.2f} m  (站立约 0.55-0.65)")
    print(f"  最大 |vz|:          {max_vz:.2f} m/s")
    print(f"  触地步数:           {airborne_steps} 步 ({airborne_steps * dt:.2f}s 腾空)")
    print(f"  轮中心高度范围:     {min_wheel_z:.2f} ~ {max_wheel_lift:.2f} m (站立时约0.29)")
    for ph, (prog, vz, z) in phase_snapshots.items():
        up = phase_up[ph]
        nose = "后翻" if up[0] < 0 else "前翻"
        print(
            f"    [{ph:7s}] 结束: flip={prog:+.2f}rad vz={vz:+.2f} z={z:.2f} up=({up[0]:+.2f},{up[1]:+.2f},{up[2]:+.2f}) {nose}"
        )
    print(f"  结束 base_z:        {base_z:.2f} m")
    print(f"  结束倾角:           {tilt:.1f}°  (直立=0°)")
    print(f"  最大关节力/扭矩:    {max_torque.round(1).tolist()}")
    print("  (actuator: L_roll L_pitch L_knee L_wheel R_roll R_pitch R_knee R_wheel)")
    # 结论 (负=后空翻方向, 正=前空翻方向, 都取绝对值判断旋转量)
    rot = abs(flip_progress)
    landed_ok = tilt < 20.0 and base_z > 0.30
    print(
        f"  {'后空翻' if flip_progress < 0 else '前空翻'}方向, 旋转 {rot / (2 * np.pi) * 100:.0f}% of 360°"
    )
    if rot >= 2 * np.pi * 0.90 and landed_ok:
        print("  ✅ 完整后空翻: 旋转≥324° 且落地直立(<20°)")
    elif rot >= 2 * np.pi * 0.75:
        print("  ⚠️ 旋转接近完成(270-324°), 落地还需调优")
    elif rot >= 2 * np.pi * 0.5:
        print("  ⚠️ 旋转中等(180-270°), 需加强蹬地/收腿")
    else:
        print("  ❌ 旋转不足")
    print(
        f"  参数: W={args.W} spinup={args.spinup} crouch_hip={args.crouch_hip} "
        f"tlaunch={args.tlaunch} launch_lean={args.launch_lean} "
        f"tflight={args.tflight} tuck={args.tuck}"
    )

    # ── 组装 mp4 ──
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
        # 清理帧
        for f in render_dir.glob("frame_*.ppm"):
            f.unlink()


if __name__ == "__main__":
    main()
