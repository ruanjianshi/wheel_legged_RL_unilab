"""Runtime data recorder for evaluation scenarios.

Captures full time-series data during each evaluation run for later
visualization and detailed analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ScenarioRecord:
    """Full time-series data for one test scenario."""
    name: str
    cmd: list[float]
    timestamps: list[float] = field(default_factory=list)
    # Velocity
    vx: list[float] = field(default_factory=list)
    vy: list[float] = field(default_factory=list)
    vz: list[float] = field(default_factory=list)
    # Angular velocity
    gyro_x: list[float] = field(default_factory=list)
    gyro_y: list[float] = field(default_factory=list)
    gyro_z: list[float] = field(default_factory=list)
    # Base state
    base_z: list[float] = field(default_factory=list)
    base_roll: list[float] = field(default_factory=list)
    base_pitch: list[float] = field(default_factory=list)
    # Joints (6 leg joints)
    leg_pos: list[np.ndarray] = field(default_factory=list)
    leg_vel: list[np.ndarray] = field(default_factory=list)
    # Actions
    actions: list[np.ndarray] = field(default_factory=list)
    # Wheel
    wheel_vel: list[np.ndarray] = field(default_factory=list)
    # Contact
    contacts: list[int] = field(default_factory=list)

    def record_step(
        self,
        t: float,
        linvel: np.ndarray,
        gyro: np.ndarray,
        base_z: float,
        base_euler: np.ndarray,
        leg_pos: np.ndarray,
        leg_vel: np.ndarray,
        action: np.ndarray,
        wheel_vel: np.ndarray,
        contact: int = 0,
    ):
        self.timestamps.append(t)
        self.vx.append(float(linvel[0, 0]))
        self.vy.append(float(linvel[0, 1]))
        self.vz.append(float(linvel[0, 2]))
        self.gyro_x.append(float(gyro[0, 0]))
        self.gyro_y.append(float(gyro[0, 1]))
        self.gyro_z.append(float(gyro[0, 2]))
        self.base_z.append(float(base_z))
        self.base_roll.append(float(base_euler[0]))
        self.base_pitch.append(float(base_euler[1]))
        self.leg_pos.append(leg_pos[0].copy())
        self.leg_vel.append(leg_vel[0].copy())
        self.actions.append(action[0].copy())
        self.wheel_vel.append(wheel_vel[0].copy())
        self.contacts.append(contact)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cmd": self.cmd,
            "timestamps": self.timestamps,
            "vx": self.vx, "vy": self.vy, "vz": self.vz,
            "gyro_x": self.gyro_x, "gyro_y": self.gyro_y, "gyro_z": self.gyro_z,
            "base_z": self.base_z,
            "base_roll": self.base_roll, "base_pitch": self.base_pitch,
            "leg_pos": [a.tolist() for a in self.leg_pos],
            "leg_vel": [a.tolist() for a in self.leg_vel],
            "actions": [a.tolist() for a in self.actions],
            "wheel_vel": [a.tolist() for a in self.wheel_vel],
            "contacts": self.contacts,
        }

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {k: np.array(v) for k, v in self.to_dict().items()
                if k not in ("name", "cmd") and isinstance(v, list) and len(v) > 0}


class Recorder:
    """Manages recording of one or more scenarios."""

    def __init__(self):
        self.records: dict[str, ScenarioRecord] = {}

    def start_scenario(self, name: str, cmd: list[float]) -> ScenarioRecord:
        rec = ScenarioRecord(name=name, cmd=cmd)
        self.records[name] = rec
        return rec

    def save(self, path: Path | str):
        data = {k: v.to_dict() for k, v in self.records.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path | str) -> dict[str, ScenarioRecord]:
        with open(path) as f:
            data = json.load(f)
        records = {}
        for name, d in data.items():
            rec = ScenarioRecord(name=d["name"], cmd=d["cmd"])
            rec.timestamps = d["timestamps"]
            rec.vx = d["vx"]; rec.vy = d["vy"]; rec.vz = d["vz"]
            rec.gyro_x = d["gyro_x"]; rec.gyro_y = d["gyro_y"]; rec.gyro_z = d["gyro_z"]
            rec.base_z = d["base_z"]
            rec.base_roll = d.get("base_roll", []); rec.base_pitch = d.get("base_pitch", [])
            rec.leg_pos = [np.array(a) for a in d.get("leg_pos", [])]
            rec.leg_vel = [np.array(a) for a in d.get("leg_vel", [])]
            rec.actions = [np.array(a) for a in d.get("actions", [])]
            rec.wheel_vel = [np.array(a) for a in d.get("wheel_vel", [])]
            rec.contacts = d.get("contacts", [])
            records[name] = rec
        self.records = records
        return records
