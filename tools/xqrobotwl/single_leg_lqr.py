"""P3: 单轮平衡 LQR/极点配置 参考控制器 — 数值线性化 + 闭环验证

思路 (文献独轮车经典控制 + 用户决策 LQR 参考引导 RL):
  1. 解耦双倒立摆 (平衡位数值线性化):
     - pitch 轴: θ̈ = α·θ + β·a    (θ 前倾角, a 轮子线加速度)
     - roll 轴:  φ̈ = α_r·φ + β_r·u  (φ 侧压角, u 配重 L_hip_roll 位置)
  2. 极点配置算状态反馈增益 K (设计闭环极点稳定带阻尼)
  3. 闭环仿真验证能否 >1.35s 稳定 (物理上限突破点)
  4. 若可行, K 作为 RL 参考引导 (reward 引导/行为克隆)

用法:
  uv run mjpython tools/xqrobotwl/single_leg_lqr.py               # 测动力学 + 闭环仿真
  uv run mjpython tools/xqrobotwl/single_leg_lqr.py --render ...  # 渲染 mp4
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as e:  # pragma: no cover
    raise SystemExit("需要 mujoco") from e

ROOT = Path(__file__).resolve().parents[2]
XML = ROOT / "src/unilab/assets/robots/xqrobotwl/scene_flat.xml"
WHEEL_R = 0.11
L_ROLL_BASE = -0.5
L_PITCH = 0.10
L_KNEE = 0.30
R_ROLL = -0.1
R_PITCH = 0.0
R_KNEE = 0.0
ROLL_UP_Y = np.sin(np.radians(30.0))  # 30° 侧压平衡位 up_y


def setup_balance(d):
    lean = np.radians(-30.0)
    c, s = np.cos(lean / 2), np.sin(lean / 2)
    d.qpos[:] = 0.0
    d.qpos[2] = 0.55
    d.qpos[3:7] = [c, s, 0, 0]
    d.qpos[7:15] = [L_ROLL_BASE, L_PITCH, L_KNEE, 0.0, R_ROLL, R_PITCH, R_KNEE, 0.0]
    d.qvel[:] = 0.0


def hold_pose(d, wheel_vel=0.0, l_roll=L_ROLL_BASE):
    """姿态锁 + 轮速/配重控制, 一次 mj_step。"""
    d.ctrl[:] = [l_roll, L_PITCH, L_KNEE, 0.0, R_ROLL, R_PITCH, R_KNEE, wheel_vel]


def rot_quat_y(d, ang):
    """绕 base 局部 y 轴转 ang (前倾扰动)。"""
    qw, qx, qy, qz = d.qpos[3:7]
    half = ang / 2
    dq = np.array([np.cos(half), 0.0, np.sin(half), 0.0])
    w1, x1, y1, z1 = qw, qx, qy, qz
    w2, x2, y2, z2 = dq
    d.qpos[3:7] = [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def rot_quat_x(d, ang):
    """绕 base 局部 x 轴转 ang (侧压扰动)。"""
    qw, qx, qy, qz = d.qpos[3:7]
    half = ang / 2
    dq = np.array([np.cos(half), np.sin(half), 0.0, 0.0])
    w1, x1, y1, z1 = qw, qx, qy, qz
    w2, x2, y2, z2 = dq
    d.qpos[3:7] = [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def meas_pitch_alpha_beta(model, dt, window=0.02):
    """pitch 轴: 测 α (θ 重力), β (轮加速度控制增益)。长窗二阶拟合。"""
    adr_up = int(np.asarray(model.sensor("upvector").adr).reshape(-1)[0])
    n = max(int(window / dt), 2)

    def theta(d):
        up = d.sensordata[adr_up : adr_up + 3]
        return np.arctan2(up[0], up[2])

    # α: θ0=0.05 扰动, 无控制, 拟合 θ̈
    d = mujoco.MjData(model)
    setup_balance(d)
    rot_quat_y(d, 0.05)
    mujoco.mj_forward(model, d)
    th0 = theta(d)
    for _ in range(n):
        hold_pose(d, wheel_vel=0.0)
        mujoco.mj_step(model, d)
    thn = theta(d)
    alpha = 2.0 * (thn - th0) / (window * window * th0)

    # β: θ=0, 轮子恒定线加速度 a=1, 拟合 θ̈
    d2 = mujoco.MjData(model)
    setup_balance(d2)
    mujoco.mj_forward(model, d2)
    thb0 = theta(d2)
    a = 1.0
    wv = 0.0
    for _ in range(n):
        wv += a * dt  # 线加速度 → 轮速积分
        hold_pose(d2, wheel_vel=wv)
        mujoco.mj_step(model, d2)
    thb1 = theta(d2)
    beta = 2.0 * (thb1 - thb0) / (window * window * a)
    return alpha, beta


def meas_roll_alpha_beta(model, dt, window=0.02):
    """roll 轴: 测 α_r (φ 重力), β_r (配重目标控制增益)。长窗二阶拟合。"""
    adr_up = int(np.asarray(model.sensor("upvector").adr).reshape(-1)[0])
    n = max(int(window / dt), 2)

    def roll(d):
        up = d.sensordata[adr_up : adr_up + 3]
        return np.arctan2(up[1], up[2])  # 绕 x 侧压角

    # α_r: φ 扰动 0.05, 配重不动
    d = mujoco.MjData(model)
    setup_balance(d)
    rot_quat_x(d, 0.05)
    mujoco.mj_forward(model, d)
    r0 = roll(d)
    for _ in range(n):
        hold_pose(d, l_roll=L_ROLL_BASE)
        mujoco.mj_step(model, d)
    rn = roll(d)
    alpha_r = 2.0 * (rn - r0) / (window * window * r0)

    # β_r: φ=0, 配重目标移 du=0.1, 拟合
    d2 = mujoco.MjData(model)
    setup_balance(d2)
    mujoco.mj_forward(model, d2)
    rb0 = roll(d2)
    du = 0.1
    for _ in range(n):
        hold_pose(d2, l_roll=L_ROLL_BASE + du)
        mujoco.mj_step(model, d2)
    rb1 = roll(d2)
    beta_r = 2.0 * (rb1 - rb0) / (window * window * du)
    return alpha_r, beta_r


def pole_gain(alpha, beta, p1, p2):
    """极点配置: θ̈=αθ+βu, 设计极点 p1,p2 → u = -[k1,k2]·[θ,θ̇]。"""
    k1 = (p1 * p2 + alpha) / beta
    k2 = -(p1 + p2) / beta
    return k1, k2


def run_closed_loop(model, dt, k1, k2, kr1, kr2, sim_time, render_dir=None, k3=0.0):
    """闭环仿真, 返回保持时间。"""
    adr_up = int(np.asarray(model.sensor("upvector").adr).reshape(-1)[0])
    adr_gyro = int(np.asarray(model.sensor("gyro").adr).reshape(-1)[0])
    adr_lv = int(np.asarray(model.sensor("local_linvel").adr).reshape(-1)[0])
    d = mujoco.MjData(model)
    setup_balance(d)
    mujoco.mj_forward(model, d)
    renderer = mujoco.Renderer(model, 480, 640) if render_dir else None
    frame_count = 0
    wheel_vel = 0.0
    n_steps = int(sim_time / dt)
    for i in range(n_steps):
        up = d.sensordata[adr_up : adr_up + 3]
        gyro = d.sensordata[adr_gyro : adr_gyro + 3]
        lv = d.sensordata[adr_lv : adr_lv + 3]
        theta = np.arctan2(up[0], up[2])
        th_dot = gyro[1]
        phi = np.arctan2(up[1], up[2])
        phi_dot = gyro[0]
        a_cmd = -(k1 * theta + k2 * th_dot) - k3 * lv[0]
        u_r = L_ROLL_BASE - (kr1 * (phi - np.radians(30.0)) + kr2 * phi_dot)
        wheel_vel += (a_cmd / WHEEL_R) * dt
        wheel_vel = np.clip(wheel_vel, -25, 25)
        hold_pose(d, wheel_vel=wheel_vel, l_roll=u_r)
        if renderer is not None and i % 4 == 0:
            renderer.update_scene(d)
            cam = renderer.scene.camera[0]
            cam.pos[:] = [d.qpos[0] + 0.6, d.qpos[1] - 3.0, d.qpos[2] + 0.5]
            cam.forward[:] = np.array([d.qpos[0], d.qpos[1] + 3.0, d.qpos[2]]) - cam.pos[:]
            cam.forward[:] = cam.forward / np.linalg.norm(cam.forward)
            cam.up[:] = [0.0, 0.0, 1.0]
            img = renderer.render()
            with open(render_dir / f"f_{frame_count:05d}.ppm", "wb") as f:
                h, w = img.shape[:2]
                f.write(f"P6\n{w} {h}\n255\n".encode())
                f.write(img.tobytes())
            frame_count += 1
        mujoco.mj_step(model, d)
        if d.qpos[2] < 0.25 or up[2] < 0.35:
            return i * dt, i
    return sim_time, n_steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_time", type=float, default=8.0)
    ap.add_argument("--sim_dt", type=float, default=0.005)
    ap.add_argument("--render", type=str, default="")
    ap.add_argument("--search", action="store_true", help="网格搜索闭环增益")
    ap.add_argument("--k1", type=float, default=60.0, help="pitch 位置增益")
    ap.add_argument("--k2", type=float, default=12.0, help="pitch 速率增益")
    ap.add_argument("--kr1", type=float, default=2.0, help="roll 位置增益")
    ap.add_argument("--kr2", type=float, default=0.4, help="roll 速率增益")
    ap.add_argument("--k3", type=float, default=10.0, help="vx 速度反馈增益 (关键!)")
    args = ap.parse_args()
    dt = args.sim_dt

    model = mujoco.MjModel.from_xml_path(str(XML))
    model.opt.timestep = dt

    print("=" * 60)
    print("P3: 单轮平衡 LQR/极点配置 参考控制器")
    print("=" * 60)

    if args.search:
        import itertools

        print("  网格搜索闭环增益 (找 >1.35s 的 LQR 配置)...")
        best = None
        for k1, k2, kr1 in itertools.product(
            [110, 130, 150, 180], [12, 15, 18, 22], [0.5, 1.0, 1.5]
        ):
            hold, _ = run_closed_loop(model, dt, k1, k2, kr1, 0.4, 6.0)
            tag = f"K_p=[{k1:.0f},{k2:.0f}] K_r=[{kr1:.1f},0.4]"
            if best is None or hold > best[1]:
                best = (tag, hold)
            if hold >= 5.0:
                break
        print(f"  🏆 最优: {best[0]} → hold {best[1]:.2f}s" if best else "  无")
        return

    # 单次闭环 (指定增益或默认 PD 配置)
    render_dir = None
    if args.render:
        out = Path(args.render)
        render_dir = out.parent / f"_frames_{out.stem}"
        render_dir.mkdir(parents=True, exist_ok=True)
    hold, _ = run_closed_loop(
        model, dt, args.k1, args.k2, args.kr1, args.kr2, args.sim_time, render_dir, k3=args.k3
    )
    print(
        f"  K_p=[{args.k1},{args.k2}] K_r=[{args.kr1},{args.kr2}] → 保持 {hold:.2f}s / {args.sim_time}s"
    )
    print(f"  {'✅ ≥2s, LQR 参考可行!' if hold >= 2.0 else '⚠️ 未达 2s, 需调增益'}")
    if render_dir is not None and len(list(render_dir.glob("f_*.ppm"))) > 0:
        out = Path(args.render)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                "50",
                "-i",
                str(render_dir / "f_%05d.ppm"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        for f in render_dir.glob("f_*.ppm"):
            f.unlink()
        render_dir.rmdir()
        print(f"  ✅ 视频: {args.render}")


if __name__ == "__main__":
    main()
