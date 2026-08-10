"""P1: 单腿平衡(单轮支撑)物理可行性验证 — xqrobotwl 轮足两足机器人

与后空翻不同(弹道可开环脚本化), 单腿平衡的**保持**是倒立摆调节问题,
必须闭环反馈 — 这正是 RL 的职责。P1 只回答三个可静态验证的问题:

  1. 静态平衡位是否可达: 收膝折腿 + 机身侧倾后, CoM 是否落在支撑轮上?
     (轮着地 + 自由轮离地 + 质心投影接近支撑轮接触点)
  2. 折腿过渡是否稳定: FF 折腿过程 CoM 是否保持在两轮支撑多边形内
     (不收膝→髋外展会横向甩质心, 过渡中就会倒)
  3. 横滚控制权是否足够: 支撑腿 hip_roll → 轮子横向位移的增益
     (横滚主动倒立摆需要的执行器杠杆)

关键结论 (devlog 09): 收膝折腿(L_knee→0.87)+ 机身侧倾 ~-28° +
支撑腿微调 → CoM 距支撑轮 <0.02m, 轮着地, 自由轮离地 0.38m。
折腿机制: 膝弯曲抬轮且 CoM 横向不动; 髋外展会甩 CoM 离支撑轮, 禁用。

执行器顺序 (MuJoCo order): [L_hip_roll, L_hip_pitch, L_knee, L_wheel,
                             R_hip_roll, R_hip_pitch, R_knee, R_wheel]

用法:
  uv run python scripts/xqrobotwl/single_leg_balance_feasibility.py
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

WHEEL_R = 0.11

# 支撑侧 (默认右腿支撑, 左腿折叠); --side left 则镜像
DEFAULT_STANCE = "right"


def whole_com(model, data) -> np.ndarray:
    com = np.zeros(3)
    tot = 0.0
    for b in range(model.nbody):
        m = model.body_mass[b]
        if m > 0:
            com += m * data.xpos[b]
            tot += m
    return com / tot


def main() -> None:
    ap = argparse.ArgumentParser(description="开环单腿平衡可行性验证")
    ap.add_argument("--fold_knee", type=float, default=0.87, help="自由腿膝弯曲 rad (收膝折腿)")
    ap.add_argument("--fold_pitch", type=float, default=0.30, help="自由腿髋前倾 rad (抬膝)")
    ap.add_argument("--lean_deg", type=float, default=-28.0, help="机身横滚侧倾 deg (向支撑侧)")
    ap.add_argument("--stance_roll", type=float, default=-0.10, help="支撑腿 hip_roll")
    ap.add_argument("--stance_pitch", type=float, default=0.0, help="支撑腿 hip_pitch")
    ap.add_argument("--stance_knee", type=float, default=0.0, help="支撑腿膝")
    ap.add_argument("--sim_dt", type=float, default=0.005)
    ap.add_argument("--render", type=str, default="", help="输出目录(非空则离屏渲染→mp4)")
    ap.add_argument("--render_every", type=int, default=2)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(XML))
    model.opt.timestep = args.sim_dt
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    renderer = None
    frame_count = 0
    if args.render:
        render_path = Path(args.render)
        mp4_out = (
            render_path if render_path.suffix.lower() == ".mp4" else render_path / "single_leg.mp4"
        )
        render_dir = render_path.parent if render_path.suffix.lower() == ".mp4" else render_path
        render_dir.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, 480, 640)
        print(f"渲染帧到: {render_dir} → {mp4_out}")

    # ── 1) 静态平衡位检查 (mj_forward 运动学, 无积分) ──
    lean = np.radians(args.lean_deg)
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.55
    c, s = np.cos(lean / 2), np.sin(lean / 2)
    data.qpos[3:7] = [c, s, 0, 0]  # 绕 X 轴侧倾
    # qpos 关节顺序: [L_roll,L_pitch,L_knee,L_wheel, R_roll,R_pitch,R_knee,R_wheel]
    data.qpos[7:15] = [
        0.1,
        args.fold_pitch,
        args.fold_knee,
        0.0,
        args.stance_roll,
        args.stance_pitch,
        args.stance_knee,
        0.0,
    ]
    mujoco.mj_forward(model, data)
    com = whole_com(model, data)
    rw = data.xpos[data.body("right_link_wheel").id]
    lw = data.xpos[data.body("left_link_wheel").id]
    dy = abs(com[1] - rw[1])
    print("=" * 60)
    print("P1: 单腿平衡可行性验证")
    print("=" * 60)
    print("  1) 静态平衡位 (mj_forward 运动学)")
    print(f"     CoM        = ({com[0]:+.3f}, {com[1]:+.3f}, {com[2]:+.3f})")
    print(f"     支撑轮 y   = {rw[1]:+.3f}   CoM-支撑轮横向距 = {com[1] - rw[1]:+.3f} m")
    print(f"     支撑轮 z   = {rw[2]:.3f} (半径 {WHEEL_R}, 着地即 ≈{WHEEL_R})")
    print(f"     自由轮离地 = {lw[2] - WHEEL_R:+.3f} m  (正值=离地)")
    balance_ok = dy < 0.05 and abs(rw[2] - WHEEL_R) < 0.03 and (lw[2] - WHEEL_R) > 0.05
    print(
        f"     {'✅ 静态平衡位可达 (CoM≈支撑轮, 轮着地, 自由轮离地)' if balance_ok else '❌ 静态平衡位不达标'}"
    )

    # ── 2) FF 折腿过渡: 从站立收膝折腿, 检查 CoM 横向漂移 ──
    print("  2) FF 折腿过渡 (从两轮站立收膝, 0.6s)")
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    max_co_drift = 0.0
    for i in range(int(0.6 / args.sim_dt)):
        r = min(i / (0.5 / args.sim_dt), 1.0)
        ctrl = np.zeros(8)
        ctrl[0] = 0.1
        ctrl[1] = args.fold_pitch * r
        ctrl[2] = args.fold_knee * r
        ctrl[4] = args.stance_roll
        ctrl[5] = args.stance_pitch
        ctrl[6] = args.stance_knee
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        c = whole_com(model, data)
        max_co_drift = max(max_co_drift, abs(c[1]))  # CoM 横向漂移
    print(
        f"     折腿中 CoM 横向最大偏移 = {max_co_drift:.3f} m "
        f"(两轮距 ~0.39m → 半宽 ~0.19m, 超出即倒)"
    )
    fold_ok = max_co_drift < 0.19
    print(
        f"     {'✅ 折腿过渡 CoM 保持在支撑多边形内' if fold_ok else '❌ 折腿过渡 CoM 甩出支撑多边形'}"
    )

    # ── 3) 横滚控制权: 支撑腿 hip_roll → 轮子横向位移范围 (保持轮着地) ──
    print("  3) 横滚控制权 (支撑腿 hip_roll → 轮横向位移范围)")
    wheel_ys, wheel_zs = [], []
    for rr in [-0.10, -0.30, -0.60, -0.90]:
        data = mujoco.MjData(model)
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = 0.55
        c, s = np.cos(lean / 2), np.sin(lean / 2)
        data.qpos[3:7] = [c, s, 0, 0]
        data.qpos[7:15] = [
            0.1,
            args.fold_pitch,
            args.fold_knee,
            0.0,
            rr,
            args.stance_pitch,
            args.stance_knee,
            0.0,
        ]
        mujoco.mj_forward(model, data)
        rw = data.xpos[data.body("right_link_wheel").id]
        wheel_ys.append(rw[1])
        wheel_zs.append(rw[2])
    sweep = wheel_ys[-1] - wheel_ys[0]  # 轮横向可移动范围 m
    grounded = all(abs(z - WHEEL_R) < 0.05 for z in wheel_zs)
    print(
        f"     R_hip_roll -0.10→-0.90: 轮 y {wheel_ys[0]:+.3f}→{wheel_ys[-1]:+.3f} "
        f"(范围 {sweep:.2f} m, 全程保持着地: {grounded})"
    )
    # 横滚扰动幅值 ~CoM 投影偏移, 支撑多边形半宽 ~0.19m; 轮需能扫出 ≥0.2m 覆盖扰动
    auth_ok = sweep > 0.20 and grounded
    print(
        f"     {'✅ 横滚控制权充足 (支撑腿髋可把轮扫 0.2m+ 覆盖质心扰动)' if auth_ok else '❌ 横滚控制权不足'}"
    )

    print("=" * 60)
    if balance_ok and fold_ok and auth_ok:
        print("✅ 单腿平衡物理可行: 平衡位可达 + 过渡可脚本化 + 横滚可控")
        print("   → 进入 RL: FSM(站立→折腿FF→单轮平衡RL→落腿FF→站立), RL 学横滚反馈")
    else:
        print("❌ 单腿平衡不可行, 需调整机器人结构/折腿姿态")
    print("=" * 60)

    # ── 渲染平衡位 ──
    if renderer is not None:
        data = mujoco.MjData(model)
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = 0.55
        c, s = np.cos(lean / 2), np.sin(lean / 2)
        data.qpos[3:7] = [c, s, 0, 0]
        data.qpos[7:15] = [
            0.1,
            args.fold_pitch,
            args.fold_knee,
            0.0,
            args.stance_roll,
            args.stance_pitch,
            args.stance_knee,
            0.0,
        ]
        mujoco.mj_forward(model, data)
        # 渲染过渡: 站立 → 折腿平衡位, 静态序列
        for i in range(int(0.6 / args.sim_dt)):
            r = min(i / (0.5 / args.sim_dt), 1.0)
            ctrl = np.zeros(8)
            ctrl[1] = args.fold_pitch * r
            ctrl[2] = args.fold_knee * r
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            if i % args.render_every == 0:
                renderer.update_scene(data)
                cam = renderer.scene.camera[0]
                cam.pos[:] = [data.qpos[0] + 0.5, -3.0, 1.0]
                cam.forward[:] = np.array([0.0, 3.0, -0.5]) - np.array([0.5, 0.0, 0.0])
                cam.forward[:] = cam.forward / np.linalg.norm(cam.forward)
                cam.up[:] = [0.0, 0.0, 1.0]
                img = renderer.render()
                with open(render_dir / f"frame_{frame_count:05d}.ppm", "wb") as f:
                    h, w = img.shape[:2]
                    f.write(f"P6\n{w} {h}\n255\n".encode())
                    f.write(img.tobytes())
                frame_count += 1
        import subprocess

        fps = round(1.0 / (args.sim_dt * args.render_every))
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
