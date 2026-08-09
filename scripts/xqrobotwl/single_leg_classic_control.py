"""P2: 经典控制(PD) 单轮平衡 + 前进后退可行性验证 — mujoco 直连

独轮车式控制 (文献 OLEBOT / unicycle LQR):
  - pitch(前后倒立摆) → 支撑轮 R_wheel 速度 PD + vx 反馈
    wheel_acc = Kp_p·(θ − θ_ref) + Kd_p·θ̇ + Kc·(vx_cmd − vx)
    θ_ref = k_θ·vx_cmd (速度命令→前倾参考, 独轮车靠前倾前进)
  - roll(侧向配重)    → 自由腿 L_hip_roll 位置 PD
    ctrl[0] = L_ROLL_BASE − Kp_r·(up_y − 0.5) − Kd_r·φ̇   (平衡 up_y=+0.5, 30°侧压)
  - 姿态钉住: 自由腿微屈(0.10/0.30), 支撑腿伸直(0/0), L_wheel=0

目标: 单轮稳定 ≥2s + vx 命令 ±0.3 m/s 可跟踪 → 证明可移动单轮平衡物理可行,
并标定 PD 增益给后续 RL 作 reference。

用法:
  uv run mjpython scripts/xqrobotwl/single_leg_classic_control.py                # 单次运行
  uv run mjpython scripts/xqrobotwl/single_leg_classic_control.py --search       # 网格搜索增益
  uv run mjpython scripts/xqrobotwl/single_leg_classic_control.py --render video/single_leg/classic.mp4
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as e:  # pragma: no cover
    raise SystemExit("需要 mujoco, 请 uv sync") from e

ROOT = Path(__file__).resolve().parents[2]
XML = ROOT / "src/unilab/assets/robots/xqrobotwl/scene_flat.xml"

WHEEL_R = 0.11  # XML 轮几何半径 (cylind size=0.11)
# 平衡位姿态: 右腿支撑伸直 (R_pitch=0,R_knee=0), 左腿配重微屈
# ⚠️ 实验发现: R_pitch=-0.2,R_knee=-0.3 (CoM x对齐) 反而数值不稳 (kp=1000 也 0.02s 崩);
#    原姿态 (伸直) kp=1000 静态可保持 → 用原姿态, 轮子 PID 补 x 方向动态平衡
L_ROLL_BASE = -0.5
L_PITCH = 0.10
L_KNEE = 0.30
R_ROLL = -0.1
R_PITCH = 0.0
R_KNEE = 0.0
ROLL_UP_Y = np.sin(np.radians(30.0))  # 平衡位 up_y = +0.5 (30° 侧压)


def whole_com(model, data) -> np.ndarray:
    com = np.zeros(3)
    tot = 0.0
    for b in range(model.nbody):
        m = model.body_mass[b]
        if m > 0:
            com += m * data.xpos[b]
            tot += m
    return com / tot


def setup_balance(data, lean_deg: float = 30.0) -> None:
    """置侧压平衡位: 机身绕 x -lean_deg, 右腿支撑伸直, 左腿配重微屈。

    ⚠️ 实验发现: 30° 侧压静态只稳 ~0.9s, 10-25° 稳 2s (Z形腿结构力矩随侧压角增大)。
    """
    lean = np.radians(-lean_deg)
    c, s = np.cos(lean / 2), np.sin(lean / 2)
    data.qpos[:] = 0.0
    data.qpos[2] = 0.55
    data.qpos[3:7] = [c, s, 0, 0]
    data.qpos[7:15] = [L_ROLL_BASE, L_PITCH, L_KNEE, 0.0, R_ROLL, R_PITCH, R_KNEE, 0.0]
    data.qvel[:] = 0.0


def run_once(
    model,
    data,
    kp_p: float,
    kd_p: float,
    kp_r: float,
    kd_r: float,
    kc: float,
    k_theta: float,
    vx_phases: list[float],
    sim_time: float,
    dt: float,
    render_dir: Path | None = None,
    render_every: int = 4,
    wheel_kv: float = 1.0,
    wheel_max: float = 25.0,
    leg_kp: float = 60.0,
    leg_kd: float = 0.0,
    sign: float = 1.0,
    lean_deg: float = 30.0,
) -> dict:
    """单次闭环运行, 返回统计。vx_phases 分段命令均匀切分 sim_time。

    leg_kp: 支撑腿位置执行器刚度 (默认 60 撑不住单腿载荷, 平衡任务需 ~800-1200)。
    """
    adr_up = int(np.asarray(model.sensor("upvector").adr).reshape(-1)[0])
    adr_gyro = int(np.asarray(model.sensor("gyro").adr).reshape(-1)[0])
    adr_lv = int(np.asarray(model.sensor("local_linvel").adr).reshape(-1)[0])
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_link_wheel")

    # 可选: 运行时调高轮子执行器增益 (诊断用)
    if wheel_kv != 1.0:
        model.actuator_gainprm[3, 0] = wheel_kv  # L_wheel
        model.actuator_gainprm[7, 0] = wheel_kv  # R_wheel
    # 支撑腿位置执行器刚度 (单腿载荷需要)
    if leg_kp != 60.0 or leg_kd != 0.0:
        for i in [0, 1, 2, 4, 5, 6]:
            model.actuator_gainprm[i, 0] = leg_kp
            if leg_kd > 0:
                model.actuator_biasprm[i, 1] = leg_kd

    setup_balance(data, lean_deg)
    mujoco.mj_forward(model, data)

    n_phases = len(vx_phases)
    seg = sim_time / n_phases
    n_steps = int(round(sim_time / dt))

    wheel_vel = 0.0
    fall_step = n_steps  # 摔倒步数 (base 撞地 / 倾角过大)
    fall_cause = "ok"
    track = {
        "z": np.zeros(n_steps),
        "up_y": np.zeros(n_steps),
        "theta": np.zeros(n_steps),
        "vx": np.zeros(n_steps),
        "wheel": np.zeros(n_steps),
        "com_dy": np.zeros(n_steps),
        "cmd": np.zeros(n_steps),
    }
    renderer = mujoco.Renderer(model, 480, 640) if render_dir else None
    frame_count = 0

    for i in range(n_steps):
        t = i * dt
        vx_cmd = vx_phases[min(int(t / seg), n_phases - 1)]

        up = data.sensordata[adr_up : adr_up + 3]
        gyro = data.sensordata[adr_gyro : adr_gyro + 3]
        lv = data.sensordata[adr_lv : adr_lv + 3]

        theta = np.arctan2(up[0], up[2])  # 前倾正
        theta_dot = gyro[1]
        roll_err = up[1] - np.sin(np.radians(lean_deg))  # >0 = 更向 +y 侧压
        roll_dot = gyro[0]
        vx = lv[0]

        # ── pitch: 支撑轮速度 PD + vx 反馈 ──
        # wheel_acc = 轮子线加速度 (m/s²), 角加速度 = a/R, 轮速积分 ÷R
        # sign 参数用于确定控制方向 (实验标定)
        theta_ref = k_theta * vx_cmd
        wheel_acc = sign * (
            kp_p * (theta - theta_ref) + kd_p * theta_dot
        ) + kc * (vx_cmd - vx)
        wheel_vel = np.clip(wheel_vel + (wheel_acc / WHEEL_R) * dt, -wheel_max, wheel_max)
        # ── roll: 自由腿配重位置 PD ──
        l_roll = L_ROLL_BASE - kp_r * roll_err - kd_r * roll_dot

        ctrl = np.array(
            [l_roll, L_PITCH, L_KNEE, 0.0, R_ROLL, R_PITCH, R_KNEE, wheel_vel]
        )
        data.ctrl[:] = ctrl

        track["z"][i] = data.qpos[2]
        track["up_y"][i] = up[1]
        track["theta"][i] = theta
        track["vx"][i] = vx
        track["wheel"][i] = wheel_vel
        track["cmd"][i] = vx_cmd
        com = whole_com(model, data)
        track["com_dy"][i] = com[1] - data.xpos[rw_id][1]

        if renderer is not None and i % render_every == 0:
            renderer.update_scene(data)
            cam = renderer.scene.camera[0]
            cam.pos[:] = [data.qpos[0] + 0.6, -3.0, 1.1]
            cam.forward[:] = np.array([0.0, 3.0, -0.6])
            cam.forward[:] = cam.forward / np.linalg.norm(cam.forward)
            cam.up[:] = [0.0, 0.0, 1.0]
            img = renderer.render()
            with open(render_dir / f"f_{frame_count:05d}.ppm", "wb") as f:
                h, w = img.shape[:2]
                f.write(f"P6\n{w} {h}\n255\n".encode())
                f.write(img.tobytes())
            frame_count += 1

        mujoco.mj_step(model, data)

        # 摔倒检测: base 撞地 / 机身倾平
        if track["z"][i] < 0.25:
            fall_step, fall_cause = i, "base_crash"
            break
        if up[2] < 0.35:
            fall_step, fall_cause = i, "over_tilt"
            break

    if renderer is not None:
        renderer.close()

    hold_time = fall_step * dt
    return {
        "hold_time": hold_time,
        "fall_cause": fall_cause,
        "fall_step": fall_step,
        "final_vx": track["vx"][fall_step - 1] if fall_step > 0 else 0.0,
        "mean_com_dy": float(np.mean(np.abs(track["com_dy"][:fall_step]))),
        "track": track,
    }


def render_mp4(render_dir: Path, out_path: Path, fps: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(render_dir / "f_%05d.ppm"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    for f in render_dir.glob("f_*.ppm"):
        f.unlink()
    render_dir.rmdir()


def main() -> None:
    ap = argparse.ArgumentParser(description="经典控制单轮平衡+移动验证")
    ap.add_argument("--kp_p", type=float, default=60.0, help="pitch 轮速位置增益")
    ap.add_argument("--kd_p", type=float, default=12.0, help="pitch 轮速阻尼")
    ap.add_argument("--kp_r", type=float, default=2.0, help="roll 配重位置增益")
    ap.add_argument("--kd_r", type=float, default=0.4, help="roll 配重阻尼")
    ap.add_argument("--kc", type=float, default=4.0, help="vx 速度反馈")
    ap.add_argument("--k_theta", type=float, default=0.25, help="速度→前倾参考")
    ap.add_argument("--sim_time", type=float, default=8.0, help="仿真时长 s")
    ap.add_argument("--sim_dt", type=float, default=0.005)
    ap.add_argument("--vx", type=float, nargs="+", default=[0.0, 0.3, -0.3, 0.0], help="分段 vx 命令")
    ap.add_argument("--wheel_kv", type=float, default=1.0, help="轮子执行器增益(诊断用)")
    ap.add_argument("--wheel_max", type=float, default=25.0, help="轮速限幅 rad/s")
    ap.add_argument("--leg_kp", type=float, default=60.0, help="支撑腿位置执行器刚度 (实验: kp=60 柔性最佳, 高kp刚性反倒)")
    ap.add_argument("--leg_kd", type=float, default=0.0, help="支撑腿位置执行器阻尼 (实验: kd>0 有害, 保持0)")
    ap.add_argument("--sign", type=float, default=-1.0, help="pitch 控制方向 (±1, 实验标定: 后仰→轮子向前)")
    ap.add_argument("--lean_deg", type=float, default=30.0, help="机身侧压角 deg (30°不稳, 10-25°更稳)")
    ap.add_argument("--render", type=str, default="", help="输出 mp4 路径")
    ap.add_argument("--search", action="store_true", help="网格搜索增益")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(XML))
    model.opt.timestep = args.sim_dt
    data = mujoco.MjData(model)

    if args.search:
        best = None
        for kp_p, kd_p, kc in itertools.product([40, 60, 90], [8, 12, 18], [2, 4, 8]):
            for kp_r in [1.0, 2.0, 4.0]:
                r = run_once(
                    model, data, kp_p, kd_p, kp_r, args.kd_r, kc, args.k_theta,
                    args.vx, 6.0, args.sim_dt,
                    wheel_kv=args.wheel_kv, wheel_max=args.wheel_max,
                    leg_kp=args.leg_kp, leg_kd=args.leg_kd, sign=args.sign, lean_deg=args.lean_deg,
                )
                tag = f"kp_p={kp_p} kd_p={kd_p} kc={kc} kp_r={kp_r}"
                print(f"  {tag:52s} → hold {r['hold_time']:5.2f}s ({r['fall_cause']}) |CoM_dy|={r['mean_com_dy']:.4f}")
                if best is None or r["hold_time"] > best[1]["hold_time"]:
                    best = (tag, r)
        print("=" * 72)
        if best is not None:
            tag, r = best
            print(f"🏆 最优: {tag} → hold {r['hold_time']:.2f}s, final_vx={r['final_vx']:.2f}")
        return

    render_dir = None
    if args.render:
        out = Path(args.render)
        render_dir = out.parent / f"_frames_{out.stem}"
        render_dir.mkdir(parents=True, exist_ok=True)
        render_every = max(int(round(1.0 / args.sim_dt / 50.0)), 1)  # ~50fps
    else:
        render_every = 4

    r = run_once(
        model, data, args.kp_p, args.kd_p, args.kp_r, args.kd_r, args.kc,
        args.k_theta, args.vx, args.sim_time, args.sim_dt,
        render_dir, render_every,
        wheel_kv=args.wheel_kv, wheel_max=args.wheel_max,
        leg_kp=args.leg_kp, leg_kd=args.leg_kd, sign=args.sign, lean_deg=args.lean_deg,
    )
    tr = r["track"]
    print("=" * 72)
    print("经典控制 单轮平衡+移动验证")
    print("=" * 72)
    print(f"  参数: kp_p={args.kp_p} kd_p={args.kd_p} kp_r={args.kp_r} kd_r={args.kd_r} kc={args.kc} k_theta={args.k_theta}")
    print(f"  vx 命令: {args.vx}  (分段 {args.sim_time/len(args.vx):.1f}s 每段)")
    print(f"  保持时间: {r['hold_time']:.2f}s / {args.sim_time}s  ({r['fall_cause']})")
    print(f"  |CoM-支撑轮| 横向平均: {r['mean_com_dy']:.4f} m")
    n = r["fall_step"]
    if n > 0:
        print(f"  结束 vx = {r['final_vx']:.2f} m/s")
        print(f"  z 范围: {tr['z'][:n].min():.2f}~{tr['z'][:n].max():.2f}, "
              f"θ 范围: {np.degrees(tr['theta'][:n]).min():.1f}°~{np.degrees(tr['theta'][:n]).max():.1f}°, "
              f"up_y 范围: {tr['up_y'][:n].min():.2f}~{tr['up_y'][:n].max():.2f} (ref {ROLL_UP_Y:.2f})")
        # 分段 vx 跟踪
        seg = len(args.vx)
        for i, vc in enumerate(args.vx):
            s0 = int(n * i / seg)
            s1 = int(n * (i + 1) / seg) if i < seg - 1 else n
            if s1 > s0:
                print(f"    vx_cmd={vc:+.2f}: 实际 {tr['vx'][s0:s1].mean():+.2f} (漂移 {tr['vx'][s0:s1].mean()-vc:+.2f})")
    if r["hold_time"] >= 2.0:
        print("✅ 通过: 单轮稳定 ≥2s")
    else:
        print("❌ 未通过: 需要调增益")
    print("=" * 72)

    if render_dir is not None and len(list(render_dir.glob("f_*.ppm"))) > 0:
        render_mp4(render_dir, Path(args.render), 50)
        print(f"✅ 视频已生成: {args.render}")


if __name__ == "__main__":
    main()
