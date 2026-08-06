#!/usr/bin/env python3
"""Query an ExRoMa policy server with one synthetic or recorded observation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "source"
if SOURCE_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SOURCE_ROOT.as_posix())

from exroma_bench.policy import CAMERA_KEYS, RemotePolicyClient

HDF_CAMERA_PATHS = {
    "cam_high": "observations/images/cam_high",
    "cam_left_wrist": "observations/images/cam_left_wrist",
    "cam_right_wrist": "observations/images/cam_right_wrist",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--prompt", default="hold the current robot pose")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def _body_velocity(base_pose: np.ndarray, velocity_w: np.ndarray) -> np.ndarray:
    w, x, y, z = base_pose[3:7] / np.linalg.norm(base_pose[3:7])
    linear = velocity_w[:3]
    angular = velocity_w[3:]
    forward = (
        (1.0 - 2.0 * (y * y + z * z)) * linear[0]
        + 2.0 * (x * y + z * w) * linear[1]
        + 2.0 * (x * z - y * w) * linear[2]
    )
    yaw = (
        2.0 * (x * z + y * w) * angular[0]
        + 2.0 * (y * z - x * w) * angular[1]
        + (1.0 - 2.0 * (x * x + y * y)) * angular[2]
    )
    return np.asarray([forward, yaw], dtype=np.float32)


def _decode_jpeg(value) -> np.ndarray:
    bgr = cv2.imdecode(np.asarray(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Could not decode an HDF5 camera frame.")
    return np.ascontiguousarray(bgr[..., ::-1])


def _recorded_observation(path: Path, frame: int) -> tuple[np.ndarray, dict, str, float]:
    with h5py.File(path.expanduser(), "r") as episode:
        frame_count = len(episode["observations/qpos"])
        index = frame if frame >= 0 else frame_count + frame
        if not 0 <= index < frame_count:
            raise IndexError(f"Frame {frame} is outside an episode with {frame_count} frames.")
        qpos = np.asarray(episode["observations/qpos"][index], dtype=np.float32)
        if qpos.shape != (14,):
            raise ValueError(f"Expected compact qpos[14], got {qpos.shape}.")
        planar = _body_velocity(
            np.asarray(episode["observations/base_pose"][index], dtype=np.float32),
            np.asarray(episode["observations/base_velocity"][index], dtype=np.float32),
        )
        state = np.concatenate((qpos, planar))
        images = {
            key: _decode_jpeg(episode[dataset_path][index])
            for key, dataset_path in HDF_CAMERA_PATHS.items()
        }
        prompt = str(episode.attrs.get("task_instruction", ""))
        timestamp = float(episode["time"][index]) if "time" in episode else float(index)
    return state, images, prompt, timestamp


def _synthetic_observation(prompt: str) -> tuple[np.ndarray, dict, str, float]:
    return (
        np.zeros(16, dtype=np.float32),
        {key: np.zeros((480, 640, 3), dtype=np.uint8) for key in CAMERA_KEYS},
        prompt,
        0.0,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episode is None:
        state, images, prompt, timestamp = _synthetic_observation(args.prompt)
    else:
        state, images, prompt, timestamp = _recorded_observation(args.episode, args.frame)
        if args.prompt != build_parser().get_default("prompt"):
            prompt = args.prompt
    with RemotePolicyClient(
        args.host,
        args.port,
        connect_timeout=args.timeout,
        response_timeout=args.timeout,
    ) as client:
        actions, timing = client.infer(
            state=state,
            images=images,
            prompt=prompt,
            timestamp=timestamp,
        )
        print(
            json.dumps(
                {
                    "server": client.metadata,
                    "state_shape": list(state.shape),
                    "actions_shape": list(actions.shape),
                    "first_action": actions[0].tolist(),
                    "timing": timing,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
