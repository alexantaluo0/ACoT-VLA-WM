#!/usr/bin/env python3
"""
Convert GenieSim recording data to AgiBotWorldChallenge-2026 dataset format.

Source (per episode):
  record/aligned_joints.h5 (grouped frames with action/ + state/) or
  aligned_joints_all.h5 (flat, legacy sim), camera/, observations/videos/, meta_info.json

Real-robot G2: each frame has separate action/ and state/ groups. Arms, grippers,
head and waist are taken from the matching group; grippers use abs(x) then clip
to [0, pi/4]. Other fields are copied as-is.

Target:
  {task_name}/
  ├── meta/
  │   ├── episodes.jsonl
  │   ├── episodes_stats.jsonl
  │   ├── info.json
  │   └── tasks.jsonl
  ├── data/
  │   └── chunk-000/
  │       ├── episode_000000.parquet
  │       └── ...
  └── videos/
      └── chunk-000/
          ├── observation.images.hand_left/
          │   ├── episode_000000.mp4
          │   └── ...
          ├── observation.images.hand_right/
          └── observation.images.top_head/

Usage:
  python3 scripts/convert_to_challenge_format.py \
      --input_dir  output/recording_data_0513_omnipicker/packaging_phone_line_real_0 \
      --output_dir output/challenge/packaging_phone_line_real_0 \
      --task_name  "packaging_phone_line_real_0"

  # dry-run to preview without writing
  python3 scripts/convert_to_challenge_format.py --input_dir ... --dry_run
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("ERROR: pandas required — pip install pandas pyarrow")

try:
    import pyarrow  # noqa: F401
except ImportError:
    sys.exit("ERROR: pyarrow required — pip install pyarrow")

# ── Camera mapping ───────────────────────────────────────────────────────────
# source video name → challenge observation key
CAMERA_MAP = {
    "hand_left_color.mp4": "observation.images.hand_left",
    "hand_right_color.mp4": "observation.images.hand_right",
    "head_color.mp4": "observation.images.top_head",
}

# mp4 basename (without extension) used in h5 timestamp/camera/{stem} and camera/{stem}/
RGB_CAMERA_STEMS = {name: name.replace(".mp4", "") for name in CAMERA_MAP}

# task4_0526 depth 目录为同视角 RGB 拷贝
DEPTH_FROM_RGB = {
    "observation.images.head_depth": "head_color.mp4",
    "observation.images.hand_left_depth": "hand_left_color.mp4",
    "observation.images.hand_right_depth": "hand_right_color.mp4",
}

RESOLUTION_KEY_MAP = {
    "hand_left_color": "hand_left_rgb",
    "hand_right_color": "hand_right_rgb",
    "head_color": "head_front_rgb",
}

DEPTH_KEYS = [
    "observation.images.hand_left_depth",
    "observation.images.hand_right_depth",
    "observation.images.head_depth",
]

LABEL_KEEP_KEYS = (
    "instruction",
    "instruction_augmentation",
    "start_frame_index",
    "success_frame_index",
    "end_frame_index",
)

VIDEO_FEATURE_ORDER = [
    "observation.images.top_head",
    "observation.images.hand_left",
    "observation.images.hand_right",
    "observation.images.head_depth",
    "observation.images.hand_left_depth",
    "observation.images.hand_right_depth",
]

SCALAR_FEATURE_ORDER = [
    "episode_index",
    "frame_index",
    "index",
    "task_index",
    "timestamp",
]

ARM_INDICES = list(range(14))
LEFT_EFFECTOR_INDEX = 14
RIGHT_EFFECTOR_INDEX = 15
HEAD_INDICES = [16, 17, 18]
WAIST_INDICES = [19, 20, 21, 22, 23]
STATE_DIM = 159
ACTION_DIM = 40
GRIPPER_CLIP_MIN = 0.0
GRIPPER_CLIP_MAX = 0.7853981633974483  # pi / 4

FFMPEG_X264_PRESET = "ultrafast"
FFMPEG_CRF = "18"
DEFAULT_VIDEO_WORKERS = 3
DEFAULT_EPISODE_WORKERS = 1


def clip_gripper(value: float) -> float:
    """Real-robot gripper: abs(raw) then clip to [0, pi/4]."""
    return float(np.clip(abs(value), GRIPPER_CLIP_MIN, GRIPPER_CLIP_MAX))


def _gripper_from_joint(joint_pos: np.ndarray) -> tuple[float, float]:
    left_raw = float(_slice_or_pad(joint_pos, [LEFT_EFFECTOR_INDEX])[0])
    right_raw = float(_slice_or_pad(joint_pos, [RIGHT_EFFECTOR_INDEX])[0])
    return clip_gripper(left_raw), clip_gripper(right_raw)


def _stack_grouped_fields(h5_path: Path, *parts: str, default_len: int) -> np.ndarray | None:
    """Stack one field from grouped aligned_joints.h5 across frames."""
    with h5py.File(h5_path, "r") as f:
        if "0" not in f:
            return None
        frame_keys = sorted((k for k in f.keys() if k.isdigit()), key=int)
        rows = []
        for key in frame_keys:
            path = "/".join((key, *parts))
            if path not in f:
                return None
            rows.append(np.asarray(f[path][()], dtype=np.float32).reshape(-1))
        if not rows:
            return None
        stacked = np.stack(rows, axis=0)
        if stacked.shape[1] < default_len:
            stacked = np.pad(stacked, ((0, 0), (0, default_len - stacked.shape[1])))
        return stacked


def read_state_grippers_from_aligned_joints(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read raw state gripper positions from grouped aligned_joints.h5."""
    left = _stack_grouped_fields(h5_path, "state", "left_effector", "position", default_len=1)
    right = _stack_grouped_fields(h5_path, "state", "right_effector", "position", default_len=1)
    if left is None or right is None:
        raise KeyError(f"Missing state effector fields in {h5_path}")
    return left[:, 0], right[:, 0]


def read_action_fields_from_aligned_joints(h5_path: Path) -> dict[str, np.ndarray]:
    """Read action grippers / head / waist from grouped aligned_joints.h5."""
    fields: dict[str, np.ndarray] = {}
    left = _stack_grouped_fields(h5_path, "action", "left_effector", "position", default_len=1)
    right = _stack_grouped_fields(h5_path, "action", "right_effector", "position", default_len=1)
    head = _stack_grouped_fields(h5_path, "action", "head", "position", default_len=3)
    waist = _stack_grouped_fields(h5_path, "action", "waist", "position", default_len=5)
    if left is not None:
        fields["action_left_effector"] = left[:, 0]
    if right is not None:
        fields["action_right_effector"] = right[:, 0]
    if head is not None:
        fields["action_head_position"] = head
    if waist is not None:
        fields["action_waist_position"] = waist
    return fields


def resolve_episode_paths(ep_dir: Path) -> dict[str, Path]:
    """Return aligned_joints / aligned_joints_all paths for legacy or real-robot layouts."""
    aligned_joints = ep_dir / "aligned_joints.h5"
    if not aligned_joints.is_file():
        aligned_joints = ep_dir / "record" / "aligned_joints.h5"
    return {
        "aligned_joints": aligned_joints,
        "aligned_joints_all": ep_dir / "aligned_joints_all.h5",
    }


def _is_grouped_aligned_joints(h5_path: Path) -> bool:
    with h5py.File(h5_path, "r") as f:
        return "0" in f


def attach_grouped_sidecar_fields(data: dict, aligned_path: Path) -> dict:
    """Fill missing state/action fields from grouped aligned_joints.h5."""
    if not aligned_path.is_file() or not _is_grouped_aligned_joints(aligned_path):
        return data

    n = data["n_frames"]
    if data.get("state_left_effector") is None or data.get("state_right_effector") is None:
        left, right = read_state_grippers_from_aligned_joints(aligned_path)
        if left.shape[0] != n:
            raise ValueError(
                f"Frame count mismatch at {aligned_path.parent}: aligned_joints.h5 has "
                f"{left.shape[0]} frames, episode has {n} frames"
            )
        data["state_left_effector"] = left
        data["state_right_effector"] = right

    if data.get("state_head_position") is None:
        head = _stack_grouped_fields(aligned_path, "state", "head", "position", default_len=3)
        if head is not None:
            data["state_head_position"] = head
    if data.get("state_waist_position") is None:
        waist = _stack_grouped_fields(aligned_path, "state", "waist", "position", default_len=5)
        if waist is not None:
            data["state_waist_position"] = waist

    for key, values in read_action_fields_from_aligned_joints(aligned_path).items():
        if data.get(key) is None:
            if values.shape[0] != n:
                raise ValueError(
                    f"Frame count mismatch at {aligned_path.parent}: {key} has "
                    f"{values.shape[0]} frames, episode has {n} frames"
                )
            data[key] = values

    return data


def _effector_series(arr, n: int) -> np.ndarray | None:
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float32).reshape(n, -1)[:, 0]


def find_episodes(input_dir: Path) -> list[Path]:
    eps = []
    for p in input_dir.iterdir():
        if not p.is_dir():
            continue
        paths = resolve_episode_paths(p)
        if paths["aligned_joints_all"].is_file() and paths["aligned_joints"].is_file():
            eps.append(p)
        elif paths["aligned_joints"].is_file():
            eps.append(p)
    return sorted(eps, key=lambda p: int(p.name) if p.name.isdigit() else p.name)


def read_h5(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        n = f["timestamp"].shape[0]
        timestamps = f["timestamp"][:].astype(np.float64)

        state_jp = f["state/joint/position"][:] if "state/joint/position" in f else np.zeros((n, 0), dtype=np.float32)
        state_jv = f["state/joint/velocity"][:] if "state/joint/velocity" in f else np.zeros((n, 0), dtype=np.float32)
        state_je = f["state/joint/effort"][:] if "state/joint/effort" in f else np.zeros((n, 0), dtype=np.float32)
        action_jp = f["action/joint/position"][:] if "action/joint/position" in f else np.zeros((n, 0), dtype=np.float32)
        action_jv = f["action/joint/velocity"][:] if "action/joint/velocity" in f else np.zeros((n, 0), dtype=np.float32)
        action_je = f["action/joint/effort"][:] if "action/joint/effort" in f else np.zeros((n, 0), dtype=np.float32)

        state_end_pos = f["state/end/position"][:] if "state/end/position" in f else None
        state_end_ori = f["state/end/orientation"][:] if "state/end/orientation" in f else None
        state_arm_pos = f["state/end/arm_position"][:] if "state/end/arm_position" in f else None
        state_arm_ori = f["state/end/arm_orientation"][:] if "state/end/arm_orientation" in f else None
        action_end_pos = f["action/end/position"][:] if "action/end/position" in f else None
        action_end_ori = f["action/end/orientation"][:] if "action/end/orientation" in f else None
        state_head_pos = f["state/head/position"][:] if "state/head/position" in f else None
        state_waist_pos = f["state/waist/position"][:] if "state/waist/position" in f else None
        action_head_pos = f["action/head/position"][:] if "action/head/position" in f else None
        action_waist_pos = f["action/waist/position"][:] if "action/waist/position" in f else None
        state_left_eff = f["state/left_effector/position"][:] if "state/left_effector/position" in f else None
        state_right_eff = f["state/right_effector/position"][:] if "state/right_effector/position" in f else None
        action_left_eff = f["action/left_effector/position"][:] if "action/left_effector/position" in f else None
        action_right_eff = f["action/right_effector/position"][:] if "action/right_effector/position" in f else None

        effector_force = f["state/effector/force"][:] if "state/effector/force" in f else np.zeros(n, dtype=np.float32)
        effector_index = f["state/effector/index"][:] if "state/effector/index" in f else np.zeros(n, dtype=np.float32)

        action_effector_force = f["action/effector/force"][:] if "action/effector/force" in f else np.zeros(n, dtype=np.float32)
        action_effector_index = f["action/effector/index"][:] if "action/effector/index" in f else np.zeros(n, dtype=np.float32)
        state_robot_position = f["state/robot/position"][:] if "state/robot/position" in f else np.zeros((n, 3), dtype=np.float32)
        state_robot_orientation = f["state/robot/orientation"][:] if "state/robot/orientation" in f else np.zeros((n, 4), dtype=np.float32)
        state_robot_position_drift = f["state/robot/position_drift"][:] if "state/robot/position_drift" in f else np.zeros((n, 3), dtype=np.float32)
        state_robot_orientation_drift = f["state/robot/orientation_drift"][:] if "state/robot/orientation_drift" in f else np.zeros((n, 4), dtype=np.float32)
        action_robot_velocity = f["action/robot/velocity"][:] if "action/robot/velocity" in f else np.zeros(n, dtype=np.float32)

    payload = {
        "n_frames": n,
        "timestamps": timestamps,
        "state_joint_position": state_jp,
        "state_joint_velocity": state_jv,
        "state_joint_effort": state_je,
        "action_joint_position": action_jp,
        "action_joint_velocity": action_jv,
        "action_joint_effort": action_je,
        "state_end_position": state_end_pos,
        "state_end_orientation": state_end_ori,
        "state_arm_position": state_arm_pos,
        "state_arm_orientation": state_arm_ori,
        "action_end_position": action_end_pos,
        "action_end_orientation": action_end_ori,
        "state_head_position": state_head_pos,
        "state_waist_position": state_waist_pos,
        "action_head_position": action_head_pos,
        "action_waist_position": action_waist_pos,
        "effector_force": effector_force,
        "effector_index": effector_index,
        "action_effector_force": action_effector_force,
        "action_effector_index": action_effector_index,
        "state_robot_position": state_robot_position,
        "state_robot_orientation": state_robot_orientation,
        "state_robot_position_drift": state_robot_position_drift,
        "state_robot_orientation_drift": state_robot_orientation_drift,
        "action_robot_velocity": action_robot_velocity,
    }
    for key, raw in (
        ("state_left_effector", state_left_eff),
        ("state_right_effector", state_right_eff),
        ("action_left_effector", action_left_eff),
        ("action_right_effector", action_right_eff),
    ):
        series = _effector_series(raw, n)
        if series is not None:
            payload[key] = series
    return payload


def _read_grouped_frame_array(f: h5py.File, frame_key: str, *parts: str) -> np.ndarray | None:
    path = "/".join((frame_key, *parts))
    if path not in f:
        return None
    return np.asarray(f[path][()], dtype=np.float32).reshape(-1)


def read_h5_grouped(h5_path: Path) -> dict:
    """Read per-frame grouped aligned_joints.h5 (real-robot layout, single pass)."""
    with h5py.File(h5_path, "r") as f:
        frame_keys = sorted((k for k in f.keys() if k.isdigit()), key=int)
        n = len(frame_keys)
        if n == 0:
            raise ValueError(f"No frames found in {h5_path}")

        stacks: dict[str, list[np.ndarray]] = {
            "state_jp": [],
            "state_jv": [],
            "state_je": [],
            "action_jp": [],
            "state_head": [],
            "state_waist": [],
            "action_head": [],
            "action_waist": [],
            "state_left_effector": [],
            "state_right_effector": [],
            "action_left_effector": [],
            "action_right_effector": [],
            "state_robot_position": [],
            "state_robot_orientation": [],
            "state_end_position": [],
            "state_end_orientation": [],
            "state_arm_position": [],
            "state_arm_orientation": [],
            "action_end_position": [],
            "action_end_orientation": [],
        }
        specs: dict[str, tuple[tuple[str, ...], int]] = {
            "state_jp": (("state", "joint", "position"), 14),
            "state_jv": (("state", "joint", "velocity"), 14),
            "state_je": (("state", "joint", "effort"), 14),
            "action_jp": (("action", "joint", "position"), 14),
            "state_head": (("state", "head", "position"), 3),
            "state_waist": (("state", "waist", "position"), 5),
            "action_head": (("action", "head", "position"), 3),
            "action_waist": (("action", "waist", "position"), 5),
            "state_left_effector": (("state", "left_effector", "position"), 1),
            "state_right_effector": (("state", "right_effector", "position"), 1),
            "action_left_effector": (("action", "left_effector", "position"), 1),
            "action_right_effector": (("action", "right_effector", "position"), 1),
            "state_robot_position": (("state", "robot", "position"), 3),
            "state_robot_orientation": (("state", "robot", "orientation"), 4),
            "state_end_position": (("state", "end", "position"), 6),
            "state_end_orientation": (("state", "end", "orientation"), 8),
            "state_arm_position": (("state", "end", "arm_position"), 6),
            "state_arm_orientation": (("state", "end", "arm_orientation"), 8),
            "action_end_position": (("action", "end", "position"), 6),
            "action_end_orientation": (("action", "end", "orientation"), 8),
        }

        action_robot_velocity = np.zeros((n, 2), dtype=np.float32)
        timestamps = np.arange(n, dtype=np.float64)
        for i, key in enumerate(frame_keys):
            for name, (parts, default_len) in specs.items():
                value = _read_grouped_frame_array(f, key, *parts)
                if value is None:
                    value = np.zeros(default_len, dtype=np.float32)
                stacks[name].append(value)
            value = _read_grouped_frame_array(f, key, "action", "robot", "velocity")
            if value is not None and value.shape[0] >= 2:
                action_robot_velocity[i] = value[:2]
            ts_key = f"{key}/main_timestamp"
            if ts_key in f:
                timestamps[i] = float(f[ts_key][()]) / 1e9

        def stack(name: str) -> np.ndarray:
            return np.stack(stacks[name], axis=0)

        state_jp = stack("state_jp")
        state_jv = stack("state_jv")
        state_je = stack("state_je")
        action_jp = stack("action_jp")
        state_head = stack("state_head")
        state_waist = stack("state_waist")
        action_head = stack("action_head")
        action_waist = stack("action_waist")
        state_left_effector = stack("state_left_effector")
        state_right_effector = stack("state_right_effector")
        action_left_effector = stack("action_left_effector")
        action_right_effector = stack("action_right_effector")
        state_robot_position = stack("state_robot_position")
        state_robot_orientation = stack("state_robot_orientation")
        state_end_position = stack("state_end_position")
        state_end_orientation = stack("state_end_orientation")
        state_arm_position = stack("state_arm_position")
        state_arm_orientation = stack("state_arm_orientation")
        action_end_position = stack("action_end_position")
        action_end_orientation = stack("action_end_orientation")

    return {
        "n_frames": n,
        "timestamps": timestamps,
        "state_joint_position": state_jp,
        "state_joint_velocity": state_jv,
        "state_joint_effort": state_je,
        "action_joint_position": action_jp,
        "action_joint_velocity": np.zeros((n, 0), dtype=np.float32),
        "action_joint_effort": np.zeros((n, 0), dtype=np.float32),
        "state_end_position": state_end_position,
        "state_end_orientation": state_end_orientation,
        "state_arm_position": state_arm_position,
        "state_arm_orientation": state_arm_orientation,
        "action_end_position": action_end_position,
        "action_end_orientation": action_end_orientation,
        "state_head_position": state_head,
        "state_waist_position": state_waist,
        "action_head_position": action_head,
        "action_waist_position": action_waist,
        "state_left_effector": state_left_effector[:, 0],
        "state_right_effector": state_right_effector[:, 0],
        "action_left_effector": action_left_effector[:, 0],
        "action_right_effector": action_right_effector[:, 0],
        "effector_force": np.zeros(n, dtype=np.float32),
        "effector_index": np.zeros(n, dtype=np.float32),
        "action_effector_force": np.zeros(n, dtype=np.float32),
        "action_effector_index": np.zeros(n, dtype=np.float32),
        "state_robot_position": state_robot_position,
        "state_robot_orientation": state_robot_orientation,
        "state_robot_position_drift": np.zeros((n, 3), dtype=np.float32),
        "state_robot_orientation_drift": np.zeros((n, 4), dtype=np.float32),
        "action_robot_velocity": action_robot_velocity,
    }


def read_episode_data(ep_dir: Path) -> dict:
    paths = resolve_episode_paths(ep_dir)
    if paths["aligned_joints_all"].is_file():
        data = read_h5(paths["aligned_joints_all"])
        return attach_grouped_sidecar_fields(data, paths["aligned_joints"])
    if paths["aligned_joints"].is_file():
        return read_h5_grouped(paths["aligned_joints"])
    raise FileNotFoundError(f"No aligned joints HDF5 found under {ep_dir}")


def _slice_or_pad(arr: np.ndarray, indices: list[int]) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    out = np.zeros(len(indices), dtype=np.float32)
    for out_i, src_i in enumerate(indices):
        if src_i < flat.shape[0]:
            out[out_i] = flat[src_i]
    return out


def _flatten_or_zeros(arr, length: int, frame_index: int) -> np.ndarray:
    if arr is None:
        return np.zeros(length, dtype=np.float32)
    value = np.asarray(arr[frame_index], dtype=np.float32).reshape(-1)
    if value.shape[0] >= length:
        return value[:length]
    return np.pad(value, (0, length - value.shape[0])).astype(np.float32)


def _values_at(
    data: dict,
    key: str,
    frame_index: int,
    joint_pos: np.ndarray,
    indices: list[int],
    length: int,
) -> np.ndarray:
    arr = data.get(key)
    if arr is not None:
        return _flatten_or_zeros(arr, length, frame_index)
    return _slice_or_pad(joint_pos, indices)


def build_action_vector(data: dict, frame_index: int) -> np.ndarray:
    """Build 40-dim action from action/* fields (arms, grippers, head, waist, end, robot)."""
    joint_pos = data["action_joint_position"][frame_index]
    if "action_left_effector" in data and "action_right_effector" in data:
        left_effector = clip_gripper(float(data["action_left_effector"][frame_index]))
        right_effector = clip_gripper(float(data["action_right_effector"][frame_index]))
    else:
        left_effector, right_effector = _gripper_from_joint(joint_pos)
    robot_velocity = np.asarray(data["action_robot_velocity"][frame_index], dtype=np.float32).reshape(-1)
    if robot_velocity.shape[0] < 2:
        robot_velocity = np.pad(robot_velocity, (0, 2 - robot_velocity.shape[0]))

    return np.concatenate(
        [
            np.asarray([left_effector], dtype=np.float32),
            np.asarray([right_effector], dtype=np.float32),
            _flatten_or_zeros(data["action_end_position"], 6, frame_index),
            _flatten_or_zeros(data["action_end_orientation"], 8, frame_index),
            _slice_or_pad(joint_pos, ARM_INDICES),
            _values_at(data, "action_head_position", frame_index, joint_pos, HEAD_INDICES, 3),
            _values_at(data, "action_waist_position", frame_index, joint_pos, WAIST_INDICES, 5),
            robot_velocity[:2].astype(np.float32),
        ]
    ).astype(np.float32)


def build_state_vector(data: dict, frame_index: int) -> np.ndarray:
    """Build 159-dim state from state/* fields (arms, grippers, head, waist, end, robot)."""
    joint_pos = data["state_joint_position"][frame_index]
    joint_vel = data["state_joint_velocity"][frame_index]
    joint_effort = data["state_joint_effort"][frame_index]
    left_effector = clip_gripper(float(data["state_left_effector"][frame_index]))
    right_effector = clip_gripper(float(data["state_right_effector"][frame_index]))
    core = np.concatenate(
        [
            np.asarray([left_effector], dtype=np.float32),
            np.asarray([right_effector], dtype=np.float32),
            _flatten_or_zeros(data["state_end_position"], 6, frame_index),
            _flatten_or_zeros(data["state_end_orientation"], 8, frame_index),
            _flatten_or_zeros(data["state_arm_orientation"], 8, frame_index),
            _flatten_or_zeros(data["state_arm_position"], 6, frame_index),
            _slice_or_pad(joint_pos, ARM_INDICES),
            _slice_or_pad(joint_effort, ARM_INDICES),
            _slice_or_pad(joint_vel, ARM_INDICES),
            _values_at(data, "state_head_position", frame_index, joint_pos, HEAD_INDICES, 3),
            _values_at(data, "state_waist_position", frame_index, joint_pos, WAIST_INDICES, 5),
            _flatten_or_zeros(data["state_robot_position"], 3, frame_index),
            _flatten_or_zeros(data["state_robot_orientation"], 4, frame_index),
        ]
    )

    # The challenge reference reserves the remaining state slots for camera
    # extrinsics: six 3x3 rotation matrices followed by six translation vectors.
    extrinsics = np.concatenate(
        [
            np.eye(3, dtype=np.float32).reshape(-1),
            np.eye(3, dtype=np.float32).reshape(-1),
            np.eye(3, dtype=np.float32).reshape(-1),
            np.eye(3, dtype=np.float32).reshape(-1),
            np.eye(3, dtype=np.float32).reshape(-1),
            np.eye(3, dtype=np.float32).reshape(-1),
            np.zeros(18, dtype=np.float32),
        ]
    )
    vector = np.concatenate([core, extrinsics]).astype(np.float32)
    if vector.shape[0] != STATE_DIM:
        raise ValueError(f"observation.state must be {STATE_DIM} dims, got {vector.shape[0]}")
    return vector


def build_episode_dataframe(data: dict, episode_index: int, global_offset: int, fps: int) -> pd.DataFrame:
    n = data["n_frames"]
    return pd.DataFrame(
        {
            "observation.state": [build_state_vector(data, i) for i in range(n)],
            "action": [build_action_vector(data, i) for i in range(n)],
            "episode_index": np.full(n, episode_index, dtype=np.int64),
            "frame_index": np.arange(n, dtype=np.int64),
            "index": np.arange(global_offset, global_offset + n, dtype=np.int64),
            "task_index": np.zeros(n, dtype=np.int64),
            "timestamp": np.asarray([i / fps for i in range(n)], dtype=np.float32),
        }
    )


def _array_stats(arr: np.ndarray) -> dict:
    arr64 = arr.astype(np.float64)
    return {
        "min": arr64.min(axis=0).tolist(),
        "max": arr64.max(axis=0).tolist(),
        "mean": arr64.mean(axis=0).tolist(),
        "std": np.maximum(arr64.std(axis=0), 1e-6).tolist(),
        "count": [int(arr64.shape[0])],
    }


def _scalar_stats(values: np.ndarray) -> dict:
    values64 = values.astype(np.float64)
    return {
        "min": [values64.min().item()],
        "max": [values64.max().item()],
        "mean": [values64.mean().item()],
        "std": [max(values64.std().item(), 1e-6)],
        "count": [int(values64.shape[0])],
    }


def _video_stats() -> dict:
    zero = [[[0.0]], [[0.0]], [[0.0]]]
    return {
        "min": zero,
        "max": zero,
        "mean": zero,
        "std": zero,
        "count": [0],
    }


def compute_episode_stats(data: dict, episode_index: int, global_offset: int, fps: int) -> dict:
    n = data["n_frames"]
    states = np.stack([build_state_vector(data, i) for i in range(n)], axis=0)
    actions = np.stack([build_action_vector(data, i) for i in range(n)], axis=0)

    stats = {key: _video_stats() for key in VIDEO_FEATURE_ORDER}
    stats["observation.state"] = _array_stats(states)
    stats["action"] = _array_stats(actions)
    stats["episode_index"] = _scalar_stats(np.full(n, episode_index, dtype=np.int64))
    stats["frame_index"] = _scalar_stats(np.arange(n, dtype=np.int64))
    stats["index"] = _scalar_stats(np.arange(global_offset, global_offset + n, dtype=np.int64))
    stats["task_index"] = _scalar_stats(np.zeros(n, dtype=np.int64))
    stats["timestamp"] = _scalar_stats(np.asarray([i / fps for i in range(n)], dtype=np.float32))
    return stats


def find_source_video(ep_dir: Path, video_name: str) -> Path | None:
    stem = video_name.replace(".mp4", "")
    for candidate in (
        ep_dir / "observations" / "videos" / video_name,
        ep_dir / "camera" / stem / video_name,
    ):
        if candidate.is_file():
            return candidate
    return None


def find_camera_bundle(ep_dir: Path, camera_stem: str) -> tuple[Path, Path] | None:
    """Return (video, txt); prefer camera/{stem}.mp4 over .h265 (same frame index as txt)."""
    cam_dir = ep_dir / "camera" / camera_stem
    txt = cam_dir / f"{camera_stem}.txt"
    if not txt.is_file():
        return None
    mp4 = cam_dir / f"{camera_stem}.mp4"
    if mp4.is_file():
        return mp4, txt
    h265 = cam_dir / f"{camera_stem}.h265"
    if h265.is_file():
        return h265, txt
    return None


def find_h265_bundle(ep_dir: Path, camera_stem: str) -> tuple[Path, Path] | None:
    """Backward-compatible alias."""
    return find_camera_bundle(ep_dir, camera_stem)


def _ffmpeg_thread_count(slot_workers: int = 1) -> int:
    cpus = os.cpu_count() or 4
    return max(1, cpus // max(1, slot_workers))


def _require_av():
    try:
        import av  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "h265 对齐抽帧需要 PyAV（pip install av），或使用 openpi .venv 的 Python 运行本脚本"
        ) from exc
    return av


def load_txt_timestamp_to_index(txt_path: Path) -> dict[int, int]:
    """Map camera timestamp (ns) -> line index in h265 decode order."""
    mapping: dict[int, int] = {}
    with txt_path.open() as fh:
        for line_no, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            ts = int(line.split()[0])
            mapping[ts] = line_no
    return mapping


def read_h5_camera_timestamps(aligned_h5: Path, camera_stem: str) -> list[int]:
    """Per-frame timestamps for one camera from grouped aligned_joints.h5."""
    return read_h5_all_camera_timestamps(aligned_h5, [camera_stem])[camera_stem]


def read_h5_all_camera_timestamps(aligned_h5: Path, camera_stems: list[str]) -> dict[str, list[int]]:
    """Per-frame timestamps for multiple cameras in one HDF5 pass."""
    if not _is_grouped_aligned_joints(aligned_h5):
        raise ValueError(f"grouped aligned_joints required for per-camera timestamps: {aligned_h5}")
    result: dict[str, list[int]] = {stem: [] for stem in camera_stems}
    with h5py.File(aligned_h5, "r") as f:
        frame_keys = sorted((k for k in f.keys() if k.isdigit()), key=int)
        for key in frame_keys:
            for stem in camera_stems:
                path = f"{key}/timestamp/camera/{stem}"
                if path not in f:
                    raise KeyError(f"Missing {path} in {aligned_h5}")
                raw = f[path][()]
                result[stem].append(int(raw[0]) if getattr(raw, "shape", ()) else int(raw))
    return result


def resolve_h265_line_indices(timestamps: list[int], txt_path: Path) -> list[int]:
    ts_to_idx = load_txt_timestamp_to_index(txt_path)
    indices: list[int] = []
    missing: list[int] = []
    for ts in timestamps:
        idx = ts_to_idx.get(ts)
        if idx is None:
            missing.append(ts)
            continue
        indices.append(idx)
    if missing:
        raise ValueError(
            f"{txt_path}: {len(missing)} h5 timestamps not found in txt "
            f"(first missing: {missing[0]})"
        )
    return indices


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _probe_video_size(video_path: Path) -> tuple[int, int]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    w, h = proc.stdout.strip().split("x")
    return int(w), int(h)


def _indices_are_strict_contiguous(line_indices: list[int]) -> bool:
    if not line_indices:
        return False
    lo, hi = min(line_indices), max(line_indices)
    return len(line_indices) == hi - lo + 1 and len(set(line_indices)) == len(line_indices)


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _ffmpeg_extract_contiguous_mp4(
    video_path: Path,
    line_indices: list[int],
    out_mp4: Path,
    fps: int,
    threads: int,
) -> int:
    lo, hi = min(line_indices), max(line_indices)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            str(threads),
            "-i",
            str(video_path),
            "-vf",
            f"select='between(n\\,{lo}\\,{hi})',setpts=N/FRAME_RATE/TB",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-preset",
            FFMPEG_X264_PRESET,
            "-crf",
            str(FFMPEG_CRF),
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ],
        check=True,
    )
    return len(line_indices)


def _ffmpeg_stream_aligned_extract(
    video_path: Path,
    line_indices: list[int],
    out_mp4: Path,
    fps: int,
    threads: int,
) -> int:
    """Single-pass decode→encode; O(1) memory, supports duplicate frame indices."""
    if not line_indices:
        raise ValueError("empty line_indices")
    if _indices_are_strict_contiguous(line_indices):
        return _ffmpeg_extract_contiguous_mp4(video_path, line_indices, out_mp4, fps, threads)

    width, height = _probe_video_size(video_path)
    frame_bytes = width * height * 3
    max_idx = max(line_indices)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            str(threads),
            "-i",
            str(video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-threads",
            str(threads),
            "-c:v",
            "libx264",
            "-preset",
            FFMPEG_X264_PRESET,
            "-crf",
            str(FFMPEG_CRF),
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert decoder.stdout is not None
    assert encoder.stdin is not None

    output_i = 0
    decode_i = 0
    try:
        while output_i < len(line_indices) and decode_i <= max_idx:
            chunk = decoder.stdout.read(frame_bytes)
            if len(chunk) < frame_bytes:
                break
            while output_i < len(line_indices) and line_indices[output_i] == decode_i:
                encoder.stdin.write(chunk)
                output_i += 1
            decode_i += 1
    finally:
        decoder.stdout.close()
        decoder.wait()
        encoder.stdin.close()
        enc_code = encoder.wait()

    if output_i != len(line_indices):
        raise ValueError(
            f"{video_path}: aligned extract stopped at output {output_i}/{len(line_indices)} "
            f"(decoded through n={decode_i - 1})"
        )
    if enc_code != 0:
        raise RuntimeError(f"ffmpeg encode failed for {out_mp4} (code {enc_code})")
    return len(line_indices)


def _pyav_stream_aligned_extract(
    video_path: Path,
    line_indices: list[int],
    out_mp4: Path,
    fps: int,
) -> int:
    av = _require_av()
    if not line_indices:
        raise ValueError("empty line_indices")
    max_idx = max(line_indices)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    output_i = 0
    decode_i = 0
    with av.open(str(video_path)) as container_in:
        stream_in = container_in.streams.video[0]
        stream_in.thread_type = "AUTO"
        with av.open(str(out_mp4), mode="w") as container_out:
            stream_out = container_out.add_stream("libx264", rate=fps)
            stream_out.width = stream_in.width
            stream_out.height = stream_in.height
            stream_out.pix_fmt = "yuv420p"
            for frame in container_in.decode(stream_in):
                if decode_i > max_idx:
                    break
                while output_i < len(line_indices) and line_indices[output_i] == decode_i:
                    out_frame = av.VideoFrame.from_ndarray(
                        frame.to_ndarray(format="rgb24"), format="rgb24"
                    )
                    for packet in stream_out.encode(out_frame):
                        container_out.mux(packet)
                    output_i += 1
                decode_i += 1
            for packet in stream_out.encode():
                container_out.mux(packet)

    if output_i != len(line_indices):
        raise ValueError(
            f"{video_path}: pyav aligned extract wrote {output_i}/{len(line_indices)} frames"
        )
    return len(line_indices)


def decode_h265_frames_by_indices(h265_path: Path, line_indices: list[int]) -> list[np.ndarray]:
    av = _require_av()
    needed = set(line_indices)
    max_idx = max(line_indices)
    by_idx: dict[int, np.ndarray] = {}
    with av.open(str(h265_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i in needed:
                by_idx[i] = frame.to_ndarray(format="rgb24")
            if i >= max_idx and len(by_idx) == len(needed):
                break
    missing = [i for i in line_indices if i not in by_idx]
    if missing:
        raise ValueError(
            f"{h265_path}: could not decode frame indices {missing[:5]} "
            f"({len(missing)} missing)"
        )
    return [by_idx[i] for i in line_indices]


def write_mp4_from_frames(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    av = _require_av()
    if not frames:
        raise ValueError(f"no frames to write: {out_path}")
    h, w = frames[0].shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out_path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        for arr in frames:
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def extract_aligned_rgb_mp4(
    src_ep_dir: Path,
    camera_stem: str,
    aligned_h5: Path,
    out_mp4: Path,
    fps: int,
    *,
    camera_timestamps: list[int] | None = None,
    ffmpeg_threads: int | None = None,
) -> int:
    """Decode camera video aligned to h5 timestamps; return frame count written."""
    bundle = find_camera_bundle(src_ep_dir, camera_stem)
    if bundle is None:
        raise FileNotFoundError(f"no camera bundle for {camera_stem} under {src_ep_dir}")
    video_path, txt_path = bundle
    timestamps = camera_timestamps or read_h5_camera_timestamps(aligned_h5, camera_stem)
    line_indices = resolve_h265_line_indices(timestamps, txt_path)
    threads = ffmpeg_threads or _ffmpeg_thread_count(1)

    if _ffmpeg_available():
        return _ffmpeg_stream_aligned_extract(video_path, line_indices, out_mp4, fps, threads)

    return _pyav_stream_aligned_extract(video_path, line_indices, out_mp4, fps)


def _copy_one_rgb_camera(
    src_ep_dir: Path,
    dest_dir: Path,
    episode_index: int,
    src_name: str,
    obs_key: str,
    *,
    aligned_h5: Path | None,
    fps: int,
    use_align: bool,
    camera_ts: dict[str, list[int]],
    ffmpeg_threads: int,
) -> tuple[str, bool, str, str | None]:
    cam_dir = dest_dir / obs_key
    cam_dir.mkdir(parents=True, exist_ok=True)
    dest_file = cam_dir / f"episode_{episode_index:06d}.mp4"
    stem = RGB_CAMERA_STEMS[src_name]
    if use_align and aligned_h5 is not None and find_camera_bundle(src_ep_dir, stem) is not None:
        try:
            n_written = extract_aligned_rgb_mp4(
                src_ep_dir,
                stem,
                aligned_h5,
                dest_file,
                fps,
                camera_timestamps=camera_ts.get(stem),
                ffmpeg_threads=ffmpeg_threads,
            )
            src_kind = "mp4" if (src_ep_dir / "camera" / stem / f"{stem}.mp4").is_file() else "h265"
            return (
                obs_key,
                True,
                f"    {obs_key}: aligned {src_kind} -> {n_written} frames",
                src_name,
            )
        except Exception as exc:
            print(f"    {obs_key}: aligned extract failed ({exc}), trying mp4 copy", flush=True)

    src_file = find_source_video(src_ep_dir, src_name)
    if src_file is None:
        return obs_key, False, f"    {obs_key}: missing source video", None
    shutil.copy2(src_file, dest_file)
    return obs_key, True, f"    {obs_key}: copied {src_file.name}", src_name


def copy_video(
    src_ep_dir: Path,
    dest_dir: Path,
    episode_index: int,
    *,
    aligned_h5: Path | None = None,
    fps: int = 30,
    align_to_h5: bool = True,
    video_workers: int = DEFAULT_VIDEO_WORKERS,
) -> dict[str, bool]:
    """Write episode mp4s. When align_to_h5, frame count matches aligned_joints.h5."""
    result: dict[str, bool] = {}
    written_rgb: dict[str, Path] = {}
    cam_slots = max(1, min(video_workers, len(CAMERA_MAP)))
    ffmpeg_threads = _ffmpeg_thread_count(cam_slots)

    camera_ts: dict[str, list[int]] = {}
    if (
        align_to_h5
        and aligned_h5 is not None
        and aligned_h5.is_file()
        and _is_grouped_aligned_joints(aligned_h5)
    ):
        stems = [RGB_CAMERA_STEMS[name] for name in CAMERA_MAP]
        camera_ts = read_h5_all_camera_timestamps(aligned_h5, stems)

    use_align = (
        align_to_h5
        and aligned_h5 is not None
        and aligned_h5.is_file()
        and _is_grouped_aligned_joints(aligned_h5)
    )

    def run_rgb(src_name: str, obs_key: str) -> tuple[str, bool, str, str | None]:
        return _copy_one_rgb_camera(
            src_ep_dir,
            dest_dir,
            episode_index,
            src_name,
            obs_key,
            aligned_h5=aligned_h5,
            fps=fps,
            use_align=use_align,
            camera_ts=camera_ts,
            ffmpeg_threads=ffmpeg_threads,
        )

    items = list(CAMERA_MAP.items())
    if cam_slots == 1:
        outcomes = [run_rgb(src, obs) for src, obs in items]
    else:
        with ThreadPoolExecutor(max_workers=cam_slots) as pool:
            outcomes = list(pool.map(lambda pair: run_rgb(pair[0], pair[1]), items))

    for obs_key, ok, log_line, src_name in outcomes:
        print(log_line, flush=True)
        result[obs_key] = ok
        if ok and src_name is not None:
            written_rgb[src_name] = dest_dir / obs_key / f"episode_{episode_index:06d}.mp4"

    for depth_key, rgb_name in DEPTH_FROM_RGB.items():
        cam_dir = dest_dir / depth_key
        cam_dir.mkdir(parents=True, exist_ok=True)
        dest_file = cam_dir / f"episode_{episode_index:06d}.mp4"
        src_file = written_rgb.get(rgb_name)
        if src_file is None or not src_file.is_file():
            src_file = find_source_video(src_ep_dir, rgb_name)
        if src_file is None:
            result[depth_key] = False
            continue
        _link_or_copy(src_file, dest_file)
        result[depth_key] = True

    return result


def read_action_intent_label(ep_dir: Path) -> list[dict] | None:
    """Read action_intent_label.json or *_action_intent_label.json."""
    label_path = ep_dir / "action_intent_label.json"
    candidates = sorted(ep_dir.glob("*_action_intent_label.json"))
    if label_path.is_file():
        candidates = [label_path] + [c for c in candidates if c != label_path]
    elif not candidates:
        return None
    with candidates[0].open() as f:
        raw = json.load(f)

    segments = raw.get("instruction_segments", [])
    cleaned: list[dict] = []
    for idx, seg in enumerate(segments):
        entry = {k: seg[k] for k in LABEL_KEEP_KEYS if k in seg}
        if idx > 0:
            prev_end = cleaned[idx - 1]["end_frame_index"]
            if entry.get("start_frame_index") is not None and entry["start_frame_index"] == prev_end:
                entry["start_frame_index"] = prev_end + 1
        cleaned.append(entry)
    return cleaned or None


def _is_valid_instruction_segment(seg: dict) -> bool:
    """单个子步骤是否具备非空 instruction 且帧索引合法。"""
    if not (seg.get("instruction") or "").strip():
        return False
    indices: list[int] = []
    for key in ("start_frame_index", "success_frame_index", "end_frame_index"):
        val = seg.get(key)
        if not isinstance(val, int) or val < 0:
            return False
        indices.append(val)
    start, success, end = indices
    return start <= success <= end


def is_complete_instruction_segments(segments: list[dict], n_frames: int) -> bool:
    """子步骤 prompt 是否完整：每步有 instruction/帧索引，段间无重叠。"""
    if not segments or n_frames <= 0:
        return False
    if not all(_is_valid_instruction_segment(seg) for seg in segments):
        return False
    for i in range(1, len(segments)):
        if segments[i]["start_frame_index"] <= segments[i - 1]["end_frame_index"]:
            return False
    last_end = segments[-1]["end_frame_index"]
    # 标注工具偶发将末帧写成 total_frames（与 h5 帧数相等）；仅允许这一种末帧越界
    if last_end > n_frames - 1 and last_end != n_frames:
        return False
    return True


def has_complete_substep_prompts(ep_dir: Path, n_frames: int) -> bool:
    """原始数据是否具备完整、可用的子步骤 instruction_segments。"""
    segments = read_action_intent_label(ep_dir)
    return segments is not None and is_complete_instruction_segments(segments, n_frames)


def _incomplete_substep_reason(segments: list[dict] | None, n_frames: int) -> str:
    if not segments:
        return "no instruction_segments"
    if not all(_is_valid_instruction_segment(s) for s in segments):
        return "empty instruction or invalid frame indices"
    if any(
        segments[i]["start_frame_index"] <= segments[i - 1]["end_frame_index"]
        for i in range(1, len(segments))
    ):
        return "overlapping segment frame ranges"
    if segments[-1]["end_frame_index"] > n_frames - 1 and segments[-1]["end_frame_index"] != n_frames:
        return (
            f"last segment end ({segments[-1]['end_frame_index']}) "
            f"exceeds episode length ({n_frames})"
        )
    return "unknown"


def get_video_info(src_ep_dir: Path) -> dict:
    """Read video resolution from recording_info.json or camera_resolution.json."""
    info_path = src_ep_dir / "recording_info.json"
    if info_path.is_file():
        with info_path.open() as f:
            data = json.load(f)
        cam_info = data.get("camera_info", {})
        result = {}
        for cam_name, info in cam_info.items():
            size_str = info.get("intrinsic", {}).get("imageSize", "(640, 480)")
            w, h = [int(x.strip()) for x in size_str.strip("()").split(",")]
            result[cam_name] = {"width": w, "height": h}
        if result:
            return result

    cam_res_path = src_ep_dir / "parameters" / "sensor" / "camera_resolution.json"
    if cam_res_path.is_file():
        with cam_res_path.open() as f:
            raw = json.load(f)
        result = {}
        for cam_name, res_key in RESOLUTION_KEY_MAP.items():
            info = raw.get(res_key)
            if info:
                result[cam_name] = {
                    "width": int(info["width"]),
                    "height": int(info["height"]),
                }
        if result:
            return result
    return {}


def make_field_descriptions(fields: list[tuple[str, int]]) -> dict:
    descriptions = {}
    cursor = 0
    for name, dims in fields:
        indices = list(range(cursor, cursor + dims))
        descriptions[name] = {
            "description": "",
            "dimensions": dims,
            "indices": indices,
        }
        cursor += dims
    return descriptions


def make_state_field_descriptions() -> dict:
    fields = [
        ("state/left_effector/position", 1),
        ("state/right_effector/position", 1),
        ("state/end/wrench", 0),
        ("state/end/position", 6),
        ("state/end/velocity", 0),
        ("state/end/orientation", 8),
        ("state/end/arm_orientation", 8),
        ("state/end/arm_position", 6),
        ("state/joint/position", 14),
        ("state/joint/current_value", 0),
        ("state/joint/effort", 14),
        ("state/joint/velocity", 14),
        ("state/head/position", 3),
        ("state/waist/position", 5),
        ("state/robot/position", 3),
        ("state/robot/orientation", 4),
        ("state/operator_event/action_src_status", 0),
        ("state/left_ee_force/controlled", 0),
        ("state/left_ee_force/rows", 0),
        ("state/left_ee_force/cols", 0),
        ("state/left_ee_force/resolution_x", 0),
        ("state/left_ee_force/resolution_y", 0),
        ("state/left_ee_force/normal_force", 0),
        ("state/left_ee_force/shear_force_x", 0),
        ("state/left_ee_force/shear_force_y", 0),
        ("state/left_ee_force/contact", 0),
        ("state/left_ee_force/valid", 0),
        ("state/left_ee_force/err_code", 0),
        ("state/right_ee_force/controlled", 0),
        ("state/right_ee_force/rows", 0),
        ("state/right_ee_force/cols", 0),
        ("state/right_ee_force/resolution_x", 0),
        ("state/right_ee_force/resolution_y", 0),
        ("state/right_ee_force/normal_force", 0),
        ("state/right_ee_force/shear_force_x", 0),
        ("state/right_ee_force/shear_force_y", 0),
        ("state/right_ee_force/contact", 0),
        ("state/right_ee_force/valid", 0),
        ("state/right_ee_force/err_code", 0),
        ("extrinsic_end_T_hand_left_rgbd_aligned/rotation_matrix", 9),
        ("extrinsic_end_T_hand_right_rgbd_aligned/rotation_matrix", 9),
        ("extrinsic_end_T_head_left_fisheye_aligned/rotation_matrix", 9),
        ("extrinsic_end_T_head_right_fisheye_aligned/rotation_matrix", 9),
        ("extrinsic_end_T_head_front_rgbd_aligned/rotation_matrix", 9),
        ("extrinsic_end_T_head_back_fisheye_aligned/rotation_matrix", 9),
        ("extrinsic_end_T_hand_left_rgbd_aligned/translation_vector", 3),
        ("extrinsic_end_T_hand_right_rgbd_aligned/translation_vector", 3),
        ("extrinsic_end_T_head_left_fisheye_aligned/translation_vector", 3),
        ("extrinsic_end_T_head_right_fisheye_aligned/translation_vector", 3),
        ("extrinsic_end_T_head_front_rgbd_aligned/translation_vector", 3),
        ("extrinsic_end_T_head_back_fisheye_aligned/translation_vector", 3),
    ]
    return make_field_descriptions(fields)


def make_action_field_descriptions() -> dict:
    return make_field_descriptions(
        [
            ("action/left_effector/position", 1),
            ("action/right_effector/position", 1),
            ("action/end/position", 6),
            ("action/end/orientation", 8),
            ("action/joint/position", 14),
            ("action/head/position", 3),
            ("action/waist/position", 5),
            ("action/robot/velocity", 2),
        ]
    )


def make_video_feature(shape: list[int], fps: int) -> dict:
    return {
        "dtype": "video",
        "video_info": {
            "video.is_depth_map": False,
            "video.fps": float(fps),
            "video.codec": "hevc",
            "video.pix_fmt": "yuv420p",
            "has_audio": False,
        },
        "shape": shape,
        "names": ["height", "width", "channel"],
    }


def episode_sidecar_maps(
    episodes: list[Path],
    episodes_meta: list[dict],
    instruction: str,
) -> tuple[dict, dict, dict, dict, dict, dict, int]:
    instruction_segments = {}
    key_frame = {}
    high_level_instruction = {}
    take_over = {}
    h5_path = {}
    intervention_info = {}
    labeled_count = 0
    for ep_idx, ep_dir in enumerate(episodes):
        key = str(ep_idx)
        key_frame[key] = {"single": [], "dual": []}
        take_over[key] = []
        intervention_info[key] = {}
        paths = resolve_episode_paths(ep_dir)
        h5_path[key] = {
            "aligned_joints_all": str(paths["aligned_joints_all"])
            if paths["aligned_joints_all"].is_file()
            else "",
            "aligned_joints": str(paths["aligned_joints"]),
        }

        label_segs = read_action_intent_label(ep_dir)
        if label_segs is None:
            raise ValueError(f"{ep_dir}: missing valid action_intent_label.json")
        instruction_segments[key] = label_segs
        labeled_count += 1
        high_level_text = ""
        meta_info = ep_dir / "meta_info.json"
        if meta_info.is_file():
            try:
                data = json.load(meta_info.open())
                text = data.get("text", "")
                if isinstance(text, str) and text:
                    high_level_text = text
            except Exception:
                high_level_text = ""
        high_level_instruction[key] = {"high_level_instruction": high_level_text or instruction}
    return instruction_segments, key_frame, high_level_instruction, take_over, h5_path, intervention_info, labeled_count


@dataclass(frozen=True)
class _EpisodeJob:
    ep_dir: str
    out_ep_idx: int
    global_offset: int
    data_dir: str
    video_dir: str
    task_name: str
    fps: int
    align_to_h5: bool
    video_workers: int


def _process_episode_job(job: _EpisodeJob) -> dict | None:
    ep_dir = Path(job.ep_dir)
    print(f"Processing episode: {ep_dir.name}", flush=True)
    data = read_episode_data(ep_dir)
    n = data["n_frames"]
    df = build_episode_dataframe(data, job.out_ep_idx, job.global_offset, job.fps)
    parquet_path = Path(job.data_dir) / f"episode_{job.out_ep_idx:06d}.parquet"
    df.to_parquet(parquet_path, index=False)

    paths = resolve_episode_paths(ep_dir)
    vid_result = copy_video(
        ep_dir,
        Path(job.video_dir),
        job.out_ep_idx,
        aligned_h5=paths["aligned_joints"],
        fps=job.fps,
        align_to_h5=job.align_to_h5,
        video_workers=job.video_workers,
    )
    ep_stats = compute_episode_stats(data, job.out_ep_idx, job.global_offset, job.fps)
    return {
        "episode_index": job.out_ep_idx,
        "global_offset": job.global_offset,
        "n_frames": n,
        "ndof": int(data["state_joint_position"].shape[1]),
        "ep_meta": {
            "episode_index": job.out_ep_idx,
            "tasks": [job.task_name],
            "length": n,
        },
        "ep_stats": {"episode_index": job.out_ep_idx, "stats": ep_stats},
        "vid_keys": [k for k, v in vid_result.items() if v],
        "ep_dir": job.ep_dir,
        "ep_name": ep_dir.name,
    }


def _build_episode_jobs(
    episodes: list[Path],
    task_name: str,
    data_dir: Path,
    video_dir: Path,
    fps: int,
    align_to_h5: bool,
    video_workers: int,
) -> list[_EpisodeJob]:
    jobs: list[_EpisodeJob] = []
    out_ep_idx = 0
    global_offset = 0
    for ep_dir in episodes:
        if read_action_intent_label(ep_dir) is None:
            print(f"  [SKIP] {ep_dir.name}: missing action_intent_label.json")
            continue
        paths = resolve_episode_paths(ep_dir)
        if paths["aligned_joints"].is_file() and _is_grouped_aligned_joints(paths["aligned_joints"]):
            with h5py.File(paths["aligned_joints"], "r") as f:
                n = len([k for k in f.keys() if k.isdigit()])
        elif paths["aligned_joints_all"].is_file():
            with h5py.File(paths["aligned_joints_all"], "r") as f:
                n = int(f["timestamp"].shape[0])
        else:
            print(f"  [SKIP] {ep_dir.name}: no aligned joints")
            continue
        if not has_complete_substep_prompts(ep_dir, n):
            segments = read_action_intent_label(ep_dir)
            reason = _incomplete_substep_reason(segments, n)
            print(f"  [SKIP] {ep_dir.name}: incomplete sub-step prompts ({reason})")
            continue
        jobs.append(
            _EpisodeJob(
                ep_dir=str(ep_dir),
                out_ep_idx=out_ep_idx,
                global_offset=global_offset,
                data_dir=str(data_dir),
                video_dir=str(video_dir),
                task_name=task_name,
                fps=fps,
                align_to_h5=align_to_h5,
                video_workers=video_workers,
            )
        )
        global_offset += n
        out_ep_idx += 1
    return jobs


def main():
    parser = argparse.ArgumentParser(
        description="Convert GenieSim recordings to AgiBotWorldChallenge-2026 format"
    )
    parser.add_argument("--input_dir", required=True, help="Task recording dir with episode sub-folders")
    parser.add_argument("--output_dir", default=None, help="Output dir (default: input_dir/../challenge/{task_name})")
    parser.add_argument("--task_name", default=None, help="Task name (default: input dir basename)")
    parser.add_argument("--instruction", default=None, help="Instruction text stored in info.json instruction_segments")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_episodes", type=int, default=-1, help="-1 = all")
    parser.add_argument("--dry_run", action="store_true", help="Preview without writing")
    parser.add_argument(
        "--no-align-videos",
        action="store_true",
        help="Copy whole mp4 instead of h5-aligned h265 decode (legacy; may mismatch parquet)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_EPISODE_WORKERS,
        help=f"Parallel episodes (default {DEFAULT_EPISODE_WORKERS})",
    )
    parser.add_argument(
        "--video-workers",
        type=int,
        default=DEFAULT_VIDEO_WORKERS,
        help=f"Parallel RGB cameras per episode (default {DEFAULT_VIDEO_WORKERS})",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    task_name = args.task_name or input_dir.name
    instruction = args.instruction or task_name
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / "challenge" / task_name

    if not input_dir.exists():
        print(f"ERROR: input dir not found: {input_dir}")
        return 1

    episodes = find_episodes(input_dir)
    if not episodes:
        print("ERROR: no episodes found (need aligned_joints.h5 or aligned_joints_all.h5)")
        return 1

    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    print(f"{'=' * 60}")
    print(f"  G2Real → leRobot data Converter")
    print(f"{'=' * 60}")
    print(f"  Input    : {input_dir}")
    print(f"  Output   : {output_dir}")
    print(f"  Task     : {task_name}")
    print(f"  Episodes : {len(episodes)}")
    print(f"  FPS      : {args.fps}")
    print(f"  Dry run  : {args.dry_run}")
    print(f"  Align h5 : {not args.no_align_videos}")
    print(f"  Ep workers: {max(1, args.num_workers)}")
    eff_vw = 1 if max(1, args.num_workers) > 1 else max(1, args.video_workers)
    print(f"  Cam workers: {eff_vw} (requested {max(1, args.video_workers)})")
    print(f"  FFmpeg   : {_ffmpeg_available()}")
    print()

    if args.dry_run:
        for i, ep in enumerate(episodes):
            paths = resolve_episode_paths(ep)
            data = read_episode_data(ep)
            n = data["n_frames"]
            ndof = data["state_joint_position"].shape[1]
            h5_align = (
                not args.no_align_videos
                and paths["aligned_joints"].is_file()
                and _is_grouped_aligned_joints(paths["aligned_joints"])
            )
            bundles = sum(
                1
                for name in CAMERA_MAP
                if find_camera_bundle(ep, RGB_CAMERA_STEMS[name]) is not None
            )
            mp4 = sum(1 for name in CAMERA_MAP if find_source_video(ep, name))
            mode = f"h5+align({bundles})" if h5_align else f"mp4({mp4})"
            print(f"  [{i:03d}] {ep.name}: {n} frames, {ndof} DOF, video={mode}")
        print(f"\n  Would create: {output_dir}/")
        print(f"    meta/ (episodes.jsonl, episodes_stats.jsonl, info.json, tasks.jsonl)")
        print(f"    data/chunk-000/ ({len(episodes)} parquet files)")
        print(f"    videos/chunk-000/ ({len(CAMERA_MAP) + len(DEPTH_FROM_RGB)} camera dirs)")
        return 0

    meta_dir = output_dir / "meta"
    data_dir = output_dir / "data" / "chunk-000"
    video_dir = output_dir / "videos" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    episodes_meta = []
    episodes_stats_list = []
    kept_episodes: list[Path] = []
    global_offset = 0
    total_frames = 0
    ndof = None
    cam_resolutions = {}
    all_cam_keys = set()

    align_to_h5 = not args.no_align_videos
    episode_workers = max(1, args.num_workers)
    video_workers = max(1, args.video_workers)
    if episode_workers > 1 and video_workers > 1:
        video_workers = 1

    jobs = _build_episode_jobs(
        episodes,
        task_name,
        data_dir,
        video_dir,
        args.fps,
        align_to_h5,
        video_workers,
    )

    if episode_workers == 1:
        results: list[dict] = []
        for job in jobs:
            result = _process_episode_job(job)
            if result is not None:
                results.append(result)
                print(
                    f"  wrote {result['n_frames']} frames, {len(result['vid_keys'])} videos",
                    flush=True,
                )
    else:
        results = []
        with ProcessPoolExecutor(max_workers=episode_workers) as pool:
            futures = [pool.submit(_process_episode_job, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                    print(
                        f"  finished episode {result['ep_name']}: "
                        f"{result['n_frames']} frames, {len(result['vid_keys'])} videos",
                        flush=True,
                    )
        results.sort(key=lambda r: r["episode_index"])

    for result in results:
        if ndof is None:
            ndof = result["ndof"]
        if not cam_resolutions:
            cam_resolutions = get_video_info(Path(result["ep_dir"]))
        episodes_meta.append(result["ep_meta"])
        episodes_stats_list.append(result["ep_stats"])
        kept_episodes.append(Path(result["ep_dir"]))
        all_cam_keys.update(result["vid_keys"])
        total_frames += result["n_frames"]

    if not kept_episodes:
        print("ERROR: no episodes with complete sub-step prompts")
        return 1

    # ── meta/episodes.jsonl ──
    with (meta_dir / "episodes.jsonl").open("w") as f:
        for em in episodes_meta:
            f.write(json.dumps(em, ensure_ascii=False) + "\n")

    # ── meta/episodes_stats.jsonl ──
    with (meta_dir / "episodes_stats.jsonl").open("w") as f:
        for es in episodes_stats_list:
            f.write(json.dumps(es, ensure_ascii=False) + "\n")

    # ── meta/tasks.jsonl ──
    with (meta_dir / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_name}, ensure_ascii=False) + "\n")

    # ── meta/info.json ──
    features = {}

    name_to_res = {
        "observation.images.hand_left": cam_resolutions.get("hand_left_color", {}),
        "observation.images.hand_right": cam_resolutions.get("hand_right_color", {}),
        "observation.images.top_head": cam_resolutions.get("head_color", {}),
        "observation.images.hand_left_depth": cam_resolutions.get("hand_left_color", {}),
        "observation.images.hand_right_depth": cam_resolutions.get("hand_right_color", {}),
        "observation.images.head_depth": cam_resolutions.get("head_color", {}),
    }

    for cam_key in VIDEO_FEATURE_ORDER:
        res = name_to_res.get(cam_key, {})
        w = res.get("width", 640)
        h = res.get("height", 480)
        features[cam_key] = make_video_feature([h, w, 3], args.fps)

    features["observation.state"] = {
        "dtype": "float32",
        "shape": [STATE_DIM],
        "field_descriptions": make_state_field_descriptions(),
    }
    features["action"] = {
        "dtype": "float32",
        "shape": [ACTION_DIM],
        "field_descriptions": make_action_field_descriptions(),
    }
    for scalar_key in SCALAR_FEATURE_ORDER:
        features[scalar_key] = {
            "dtype": "float32" if scalar_key == "timestamp" else "int64",
            "shape": [1],
            "names": None,
        }

    # LeRobot v2.1 / lerobot_dataset 依赖 info["chunks_size"]、total_chunks、splits（与 HF 模板一致）
    n_eps = len(episodes_meta)
    chunks_size = 1000
    instruction_segments, key_frame, high_level_instruction, take_over, h5_path, intervention_info, labeled_count = episode_sidecar_maps(
        kept_episodes,
        episodes_meta,
        instruction,
    )
    info = {
        "codebase_version": "v2.1",
        "robot_type": "g2a",
        "total_episodes": n_eps,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": n_eps * len(VIDEO_FEATURE_ORDER),
        "total_chunks": (n_eps + chunks_size - 1) // chunks_size if n_eps else 0,
        "chunks_size": chunks_size,
        "fps": args.fps,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "data_version": "v0.1.4",
        "aid_info": {},
        "instruction_segments": instruction_segments,
        "key_frame": key_frame,
        "high_level_instruction": high_level_instruction,
        "take_over": take_over,
        "h5_path": h5_path,
        "camera_parameters": {},
        "intervention_info": intervention_info,
    }
    with (meta_dir / "info.json").open("w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n{'=' * 60}")
    print(f"  Conversion complete")
    print(f"  Episodes    : {len(episodes_meta)}")
    print(f"  Total frames: {total_frames}")
    print(f"  DOF         : {ndof}")
    print(f"  Cameras     : {', '.join(sorted(all_cam_keys))}")
    print(f"  Labels      : {labeled_count}/{len(episodes_meta)} episodes (complete sub-step prompts)")
    print(f"  Output      : {output_dir}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
