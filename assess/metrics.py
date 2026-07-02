"""Paper-standard evaluation metrics for legged robot RL policies.

Categories:
  I.   Command Tracking
  II.  Stability & Safety
  III. Motion Quality
  IV.  Energy Efficiency
  V.   Gait Characteristics

References:
  RMA (Kumar+, RSS 2021), ANYmal (Hwangbo+, SciRob 2019),
  Cassie (Xie+, ICRA 2020), IsaacGym (Rudin+, CoRL 2022),
  DreamWaQ (Nahrendra+, RAL 2023)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EvalContext:
    """Data buffer collected during one test scenario evaluation."""
    cmd_vx: float
    cmd_vy: float
    cmd_vyaw: float
    actual_vx: list[float] = field(default_factory=list)
    actual_vy: list[float] = field(default_factory=list)
    actual_vz: list[float] = field(default_factory=list)
    base_z: list[float] = field(default_factory=list)
    gyro_z: list[float] = field(default_factory=list)
    gyro_x: list[float] = field(default_factory=list)
    gyro_y: list[float] = field(default_factory=list)
    leg_positions: list[np.ndarray] = field(default_factory=list)
    leg_velocities: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    torques: list[float] = field(default_factory=list)
    wheel_vel: list[np.ndarray] = field(default_factory=list)
    base_roll: list[float] = field(default_factory=list)
    base_pitch: list[float] = field(default_factory=list)

    def record(self, linvel, gyro, base_z, leg_pos, leg_vel, action, torque, wheel_vel, roll, pitch):
        self.actual_vx.append(float(linvel[0, 0]))
        self.actual_vy.append(float(linvel[0, 1]))
        self.actual_vz.append(float(linvel[0, 2]))
        self.gyro_x.append(float(gyro[0, 0]))
        self.gyro_y.append(float(gyro[0, 1]))
        self.gyro_z.append(float(gyro[0, 2]))
        self.base_z.append(float(base_z[0, 2]))
        self.base_roll.append(float(roll))
        self.base_pitch.append(float(pitch))
        self.leg_positions.append(leg_pos[0, :6].copy())
        self.leg_velocities.append(leg_vel[0, :6].copy())
        self.actions.append(action[0].copy())
        self.torques.append(float(np.mean(np.abs(torque))))
        self.wheel_vel.append(wheel_vel[0].copy())


# ═════════════════════════════════════════════════════════════════════════════
# I. Command Tracking Metrics
# ═════════════════════════════════════════════════════════════════════════════

def vx_tracking_rmse(ctx: EvalContext) -> float:
    """RMS error between commanded and actual forward velocity (RMA, ANYmal)."""
    if not ctx.actual_vx:
        return float("nan")
    err = np.array(ctx.actual_vx) - ctx.cmd_vx
    return float(np.sqrt(np.mean(err ** 2)))


def vy_tracking_rmse(ctx: EvalContext) -> float:
    """RMS error between commanded and actual lateral velocity."""
    if not ctx.actual_vy:
        return float("nan")
    err = np.array(ctx.actual_vy) - ctx.cmd_vy
    return float(np.sqrt(np.mean(err ** 2)))


def vyaw_tracking_rmse(ctx: EvalContext) -> float:
    """RMS error between commanded and actual yaw rate."""
    if not ctx.gyro_z:
        return float("nan")
    err = np.array(ctx.gyro_z) - ctx.cmd_vyaw
    return float(np.sqrt(np.mean(err ** 2)))


def avg_velocity_x(ctx: EvalContext) -> float:
    """Mean achieved forward velocity (m/s)."""
    return float(np.mean(ctx.actual_vx)) if ctx.actual_vx else float("nan")


def avg_velocity_y(ctx: EvalContext) -> float:
    """Mean achieved lateral velocity (m/s)."""
    return float(np.mean(ctx.actual_vy)) if ctx.actual_vy else float("nan")


def vel_tracking_ratio(ctx: EvalContext) -> float:
    """Achieved speed magnitude / commanded speed magnitude."""
    cmd_norm = np.sqrt(ctx.cmd_vx ** 2 + ctx.cmd_vy ** 2)
    if cmd_norm < 1e-6:
        return float(np.mean([np.sqrt(vx**2 + vy**2)
                             for vx, vy in zip(ctx.actual_vx, ctx.actual_vy)]))
    actual = [np.sqrt(vx**2 + vy**2) for vx, vy in zip(ctx.actual_vx, ctx.actual_vy)]
    return float(np.mean(actual) / cmd_norm)


def vel_coupling(ctx: EvalContext) -> float:
    """Vy crosstalk when commanding Vx-only (or vice versa).
    
    If Vx command is non-zero and Vy command is zero, returns |avg(Vy)|.
    If Vy command is non-zero and Vx command is zero, returns |avg(Vx)|.
    """
    if abs(ctx.cmd_vx) > 1e-4 and abs(ctx.cmd_vy) < 1e-4:
        return float(abs(np.mean(ctx.actual_vy)))
    if abs(ctx.cmd_vy) > 1e-4 and abs(ctx.cmd_vx) < 1e-4:
        return float(abs(np.mean(ctx.actual_vx)))
    return 0.0


# ═════════════════════════════════════════════════════════════════════════════
# II. Stability & Safety Metrics
# ═════════════════════════════════════════════════════════════════════════════

def base_height_mean(ctx: EvalContext) -> float:
    """Mean base link height (m). Target: 0.65m."""
    return float(np.mean(ctx.base_z)) if ctx.base_z else float("nan")


def base_height_std(ctx: EvalContext) -> float:
    """Standard deviation of base height — lower is more stable (RMA)."""
    return float(np.std(ctx.base_z)) if len(ctx.base_z) > 1 else float("nan")


def orientation_rmse_deg(ctx: EvalContext) -> float:
    """RMS orientation error in degrees (roll+pitch from level)."""
    if not ctx.base_roll:
        return float("nan")
    roll = np.array(ctx.base_roll)
    pitch = np.array(ctx.base_pitch)
    return float(np.sqrt(np.mean(roll ** 2 + pitch ** 2))) * 180.0 / np.pi


def yaw_stability(ctx: EvalContext) -> float:
    """Std of yaw angular velocity — wobble measure."""
    return float(np.std(ctx.gyro_z)) if len(ctx.gyro_z) > 1 else float("nan")


# ═════════════════════════════════════════════════════════════════════════════
# III. Motion Quality Metrics
# ═════════════════════════════════════════════════════════════════════════════

def action_smoothness(ctx: EvalContext) -> float:
    """Mean L2 norm of action differences between consecutive steps.
    
    Lower = smoother (IsaacGym Envs).
    """
    if len(ctx.actions) < 2:
        return float("nan")
    diffs = [np.linalg.norm(ctx.actions[i] - ctx.actions[i - 1])
             for i in range(1, len(ctx.actions))]
    return float(np.mean(diffs))


def joint_velocity_mean(ctx: EvalContext) -> float:
    """Mean absolute joint velocity (rad/s)."""
    if not ctx.leg_velocities:
        return float("nan")
    abs_vel = np.mean([np.abs(v) for v in ctx.leg_velocities])
    return float(abs_vel)


def gait_symmetry(ctx: EvalContext) -> float:
    """Left-right leg position symmetry: mean(abs(L_pos - mirrored_R_pos)).
    
    Lower = more symmetric (Cassie).
    """
    if not ctx.leg_positions:
        return float("nan")
    pos = np.array(ctx.leg_positions)
    # L: indices 0,1,2  |  R: indices 3,4,5
    l = pos[:, :3]
    r = pos[:, 3:6]
    return float(np.mean(np.abs(l - r)))


# ═════════════════════════════════════════════════════════════════════════════
# IV. Energy Efficiency Metrics
# ═════════════════════════════════════════════════════════════════════════════

def mean_torque(ctx: EvalContext) -> float:
    """Mean absolute joint torque (scalar)."""
    return float(np.mean(ctx.torques)) if ctx.torques else float("nan")


def cost_of_transport(ctx: EvalContext) -> float:
    """Approximate Cost of Transport: mean_torque * mean_velocity / (mass * g * velocity).
    
    Simplified: CoT = mean_torque * mean_vel / (mass * g * speed)
    mass ≈ 5.0 kg, g = 9.81 m/s²
    """
    if not ctx.torques or not ctx.actual_vx:
        return float("nan")
    mass = 5.0
    g = 9.81
    speed = np.mean([np.sqrt(vx**2 + vy**2)
                     for vx, vy in zip(ctx.actual_vx, ctx.actual_vy)])
    if speed < 0.01:
        return 0.0
    power = np.mean(ctx.torques) * np.mean(np.abs(ctx.leg_velocities))
    return float(power / (mass * g * speed))


# ═════════════════════════════════════════════════════════════════════════════
# V. Gait Characteristics
# ═════════════════════════════════════════════════════════════════════════════

def step_frequency(ctx: EvalContext) -> float:
    """Estimated step frequency from leg position periodicity (FFT peak)."""
    if len(ctx.leg_positions) < 10:
        return float("nan")
    # Use L_calf position for frequency estimation
    calf = np.array([p[2] for p in ctx.leg_positions])
    calf_centered = calf - np.mean(calf)
    n = len(calf_centered)
    if n < 4:
        return float("nan")
    fft = np.abs(np.fft.rfft(calf_centered))
    freqs = np.fft.rfftfreq(n, d=0.01)  # ctrl_dt = 0.01
    # Exclude DC and very low frequencies
    peak_idx = np.argmax(fft[1:]) + 1
    if peak_idx >= len(freqs):
        return float("nan")
    return float(freqs[peak_idx])


def leg_workspace_utilization(ctx: EvalContext) -> float:
    """Total range of leg joint motion (rad)."""
    if not ctx.leg_positions:
        return float("nan")
    pos = np.array(ctx.leg_positions)
    ranges = np.max(pos, axis=0) - np.min(pos, axis=0)
    return float(np.sum(ranges))


# ── Registry ────────────────────────────────────────────────────────────────

# Category I: Command Tracking
TRACKING_METRICS = {
    "vx_tracking_rmse": vx_tracking_rmse,
    "vy_tracking_rmse": vy_tracking_rmse,
    "vyaw_tracking_rmse": vyaw_tracking_rmse,
    "avg_vx": avg_velocity_x,
    "avg_vy": avg_velocity_y,
    "vel_tracking_ratio": vel_tracking_ratio,
    "vel_coupling": vel_coupling,
}

# Category II: Stability & Safety
STABILITY_METRICS = {
    "base_height_mean": base_height_mean,
    "base_height_std": base_height_std,
    "orientation_rmse_deg": orientation_rmse_deg,
    "yaw_stability": yaw_stability,
}

# Category III: Motion Quality
MOTION_METRICS = {
    "action_smoothness": action_smoothness,
    "joint_velocity_mean": joint_velocity_mean,
    "gait_symmetry": gait_symmetry,
}

# Category IV: Energy Efficiency
ENERGY_METRICS = {
    "mean_torque": mean_torque,
    "cost_of_transport": cost_of_transport,
}

# Category V: Gait
GAIT_METRICS = {
    "step_frequency": step_frequency,
    "leg_workspace_utilization": leg_workspace_utilization,
}

ALL_METRICS = {
    **TRACKING_METRICS,
    **STABILITY_METRICS,
    **MOTION_METRICS,
    **ENERGY_METRICS,
    **GAIT_METRICS,
}

METRIC_CATEGORIES = {
    "tracking": TRACKING_METRICS,
    "stability": STABILITY_METRICS,
    "motion": MOTION_METRICS,
    "energy": ENERGY_METRICS,
    "gait": GAIT_METRICS,
}
