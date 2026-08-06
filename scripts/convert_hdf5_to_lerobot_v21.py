#!/usr/bin/env python3
"""Convert ExRoMa HDF5 episodes to LeRobotDataset v2.1 (Parquet + MP4)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

CODEBASE_VERSION = "v2.1"
CHUNK_SIZE = 1000
CAMERA_PATHS = {
    "cam_high": ("mast/rgb_jpeg", "cam_high"),
    "cam_left_wrist": ("front_left/rgb_jpeg", "cam_left_wrist"),
    "cam_right_wrist": ("front_right/rgb_jpeg", "cam_right_wrist"),
}
BASE_STATE_NAMES = ("base_velocity/forward", "base_velocity/yaw")
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="HDF5 episode or directory of episodes.")
    parser.add_argument("output", type=Path, help="Output LeRobot v2.1 dataset directory.")
    parser.add_argument("--limit", type=int, help="Convert only the first N sorted episodes.")
    parser.add_argument("--robot-type", default="dual_piper_rover")
    parser.add_argument("--video-codec", default="libx264")
    parser.add_argument("--video-crf", type=int, default=23)
    parser.add_argument("--video-preset", default="medium")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include complete episodes whose success attribute is false.",
    )
    return parser


def _json_attr(episode: h5py.File, name: str, default: Any) -> Any:
    value = episode.attrs.get(name)
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def discover_episodes(input_path: Path, *, limit: int | None, include_failed: bool) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        candidates = [input_path]
    elif input_path.is_dir():
        candidates = sorted(input_path.glob("episode_*.hdf5"))
    else:
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    selected = []
    for path in candidates:
        with h5py.File(path, "r") as episode:
            complete = bool(episode.attrs.get("complete", False))
            success = bool(episode.attrs.get("success", False))
        if complete and (success or include_failed):
            selected.append(path)
        if limit is not None and len(selected) >= limit:
            break

    if not selected:
        raise ValueError("No complete successful HDF5 episodes matched the input.")
    return selected


def _camera_dataset(episode: h5py.File, camera_key: str) -> h5py.Dataset:
    images = episode.get("observations/images")
    if images is None:
        raise ValueError(f"Episode has no observations/images group: {episode.filename}")
    for candidate in CAMERA_PATHS[camera_key]:
        if candidate in images:
            dataset = images[candidate]
            if isinstance(dataset, h5py.Dataset):
                return dataset
    raise ValueError(f"Episode is missing camera {camera_key}: {episode.filename}")


def _decode_jpeg(encoded: Any, *, camera_key: str, frame_index: int) -> np.ndarray:
    frame = cv2.imdecode(np.asarray(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not decode {camera_key} frame {frame_index}.")
    return np.ascontiguousarray(frame)


def body_planar_velocity(base_pose: np.ndarray, root_velocity_world: np.ndarray) -> np.ndarray:
    """Return body-frame forward velocity and yaw rate from Isaac world-frame velocity."""

    pose = np.asarray(base_pose, dtype=np.float64)
    velocity = np.asarray(root_velocity_world, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"Expected base pose [N,7], got {pose.shape}.")
    if velocity.ndim != 2 or velocity.shape[1] != 6 or len(velocity) != len(pose):
        raise ValueError(f"Expected matching base velocity [N,6], got {velocity.shape}.")

    quaternion = pose[:, 3:7]
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Base pose contains a zero-length quaternion.")
    w, x, y, z = (quaternion / norm).T

    linear = velocity[:, :3]
    angular = velocity[:, 3:]
    forward = (
        (1.0 - 2.0 * (y * y + z * z)) * linear[:, 0]
        + 2.0 * (x * y + z * w) * linear[:, 1]
        + 2.0 * (x * z - y * w) * linear[:, 2]
    )
    yaw_rate = (
        2.0 * (x * z + y * w) * angular[:, 0]
        + 2.0 * (y * z - x * w) * angular[:, 1]
        + (1.0 - 2.0 * (x * x + y * y)) * angular[:, 2]
    )
    return np.column_stack((forward, yaw_rate)).astype(np.float32)


def compact_state(episode: h5py.File) -> tuple[np.ndarray, list[str]]:
    qpos = np.asarray(episode["observations/qpos"], dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != 14:
        raise ValueError(f"Expected compact qpos [N,14], got {qpos.shape}: {episode.filename}")
    base_velocity = body_planar_velocity(
        np.asarray(episode["observations/base_pose"]),
        np.asarray(episode["observations/base_velocity"]),
    )
    names = list(_json_attr(episode, "joint_names_json", []))
    if len(names) != 14:
        names = [f"joint_{index}" for index in range(14)]
    return np.concatenate((qpos, base_velocity), axis=1), [*names, *BASE_STATE_NAMES]


def compact_action(episode: h5py.File) -> tuple[np.ndarray, list[str]]:
    if "actions/joint_position_target" in episode:
        joint_action = np.asarray(episode["actions/joint_position_target"], dtype=np.float32)
    else:
        joint_action = np.asarray(episode["action"], dtype=np.float32)
    base_action = np.asarray(episode["actions/base_velocity"], dtype=np.float32)
    if joint_action.ndim != 2 or joint_action.shape[1] != 14:
        raise ValueError(f"Expected joint action [N,14], got {joint_action.shape}.")
    if base_action.ndim != 2 or base_action.shape[1] != 2 or len(base_action) != len(joint_action):
        raise ValueError(f"Expected matching base action [N,2], got {base_action.shape}.")
    names = list(_json_attr(episode, "action_names_json", []))
    if len(names) != 16:
        names = [
            *(f"joint_position_target/joint_{index}" for index in range(14)),
            "base_command/linear",
            "base_command/angular",
        ]
    return np.concatenate((joint_action, base_action), axis=1), names


def _numeric_stats(array: np.ndarray) -> dict[str, list]:
    values = np.asarray(array)
    keepdims = values.ndim == 1
    return {
        "min": np.min(values, axis=0, keepdims=keepdims).tolist(),
        "max": np.max(values, axis=0, keepdims=keepdims).tolist(),
        "mean": np.mean(values, axis=0, keepdims=keepdims, dtype=np.float64).tolist(),
        "std": np.std(values, axis=0, keepdims=keepdims, dtype=np.float64).tolist(),
        "count": [len(values)],
    }


class ImageStats:
    def __init__(self) -> None:
        self.frames = 0
        self.pixels = 0
        self.minimum = np.full(3, 255, dtype=np.uint8)
        self.maximum = np.zeros(3, dtype=np.uint8)
        self.total = np.zeros(3, dtype=np.float64)
        self.total_square = np.zeros(3, dtype=np.float64)

    def update_bgr(self, frame: np.ndarray) -> None:
        rgb = frame[..., ::-1]
        flat = rgb.reshape(-1, 3)
        self.minimum = np.minimum(self.minimum, flat.min(axis=0))
        self.maximum = np.maximum(self.maximum, flat.max(axis=0))
        self.total += flat.sum(axis=0, dtype=np.float64)
        self.total_square += np.square(flat.astype(np.float64)).sum(axis=0)
        self.frames += 1
        self.pixels += len(flat)

    def serialize(self) -> dict[str, list]:
        mean = self.total / self.pixels
        variance = np.maximum(self.total_square / self.pixels - mean * mean, 0.0)

        def image_shape(value: np.ndarray) -> list:
            return (value.astype(np.float64).reshape(3, 1, 1) / 255.0).tolist()

        return {
            "min": image_shape(self.minimum),
            "max": image_shape(self.maximum),
            "mean": image_shape(mean),
            "std": image_shape(np.sqrt(variance)),
            "count": [self.frames],
        }


def encode_camera_video(
    dataset: h5py.Dataset,
    output_path: Path,
    *,
    fps: int,
    codec: str,
    crf: int,
    preset: str,
    camera_key: str,
) -> tuple[tuple[int, int], dict[str, list]]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode LeRobot camera videos.")
    if len(dataset) == 0:
        raise ValueError(f"Camera {camera_key} contains no frames.")

    first = _decode_jpeg(dataset[0], camera_key=camera_key, frame_index=0)
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.encoding.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        codec,
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-g",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        temporary.as_posix(),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stats = ImageStats()
    try:
        assert process.stdin is not None
        for frame_index in range(len(dataset)):
            frame = (
                first
                if frame_index == 0
                else _decode_jpeg(
                    dataset[frame_index], camera_key=camera_key, frame_index=frame_index
                )
            )
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Camera {camera_key} frame {frame_index} changed resolution to {frame.shape[:2]}."
                )
            stats.update_bgr(frame)
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {camera_key}: {error.strip()}")
        temporary.replace(output_path)
    except Exception:
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    return (height, width), stats.serialize()


def _fixed_list_array(values: np.ndarray):
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required; install ExRoMa's compatible requirements."
        ) from exc
    values = np.ascontiguousarray(values, dtype=np.float32)
    flat = pa.array(values.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, values.shape[1])


def write_parquet(path: Path, columns: dict[str, np.ndarray]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required; install ExRoMa's compatible requirements."
        ) from exc

    arrays = []
    names = []
    for name, values in columns.items():
        values = np.asarray(values)
        if values.ndim == 2:
            array = _fixed_list_array(values)
        elif values.dtype == np.float32:
            array = pa.array(values, type=pa.float32())
        elif values.dtype == np.int64:
            array = pa.array(values, type=pa.int64())
        else:
            raise TypeError(f"Unsupported Parquet column {name}: {values.dtype} {values.shape}")
        names.append(name)
        arrays.append(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_arrays(arrays, names=names), path, compression="zstd")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonlines(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert_dataset(
    episode_paths: list[Path],
    output: Path,
    *,
    robot_type: str,
    video_codec: str,
    video_crf: int,
    video_preset: str,
    overwrite: bool,
) -> Path:
    output = output.expanduser().resolve()
    temporary = output.with_name(f".{output.name}.converting")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)

    tasks: dict[str, int] = {}
    episode_records = []
    episode_stats = []
    total_frames = 0
    state_names: list[str] | None = None
    action_names: list[str] | None = None
    camera_shapes: dict[str, tuple[int, int]] = {}
    dataset_fps: int | None = None

    try:
        for output_episode_index, source_path in enumerate(episode_paths):
            with h5py.File(source_path, "r") as episode:
                fps_value = float(episode.attrs.get("record_fps", 10.0))
                fps = round(fps_value)
                if not math.isclose(fps_value, fps, abs_tol=1e-6):
                    raise ValueError(f"LeRobot v2.1 requires integer FPS, got {fps_value}.")
                if dataset_fps is None:
                    dataset_fps = fps
                elif dataset_fps != fps:
                    raise ValueError(
                        f"Mixed FPS episodes are unsupported: {dataset_fps} and {fps}."
                    )

                state, current_state_names = compact_state(episode)
                action, current_action_names = compact_action(episode)
                frame_count = len(state)
                if len(action) != frame_count:
                    raise ValueError(f"State/action length mismatch in {source_path}.")
                if state_names is None:
                    state_names = current_state_names
                    action_names = current_action_names
                elif state_names != current_state_names or action_names != current_action_names:
                    raise ValueError(f"State/action names changed in {source_path}.")

                prompt = (
                    str(episode.attrs.get("task_instruction", "")).strip() or "unspecified task"
                )
                task_index = tasks.setdefault(prompt, len(tasks))
                chunk = output_episode_index // CHUNK_SIZE
                columns = {
                    "observation.state": state,
                    "action": action,
                    "timestamp": np.arange(frame_count, dtype=np.float32) / np.float32(fps),
                    "frame_index": np.arange(frame_count, dtype=np.int64),
                    "episode_index": np.full(frame_count, output_episode_index, dtype=np.int64),
                    "index": np.arange(total_frames, total_frames + frame_count, dtype=np.int64),
                    "task_index": np.full(frame_count, task_index, dtype=np.int64),
                }
                parquet_path = (
                    temporary / f"data/chunk-{chunk:03d}/episode_{output_episode_index:06d}.parquet"
                )
                write_parquet(parquet_path, columns)

                stats = {key: _numeric_stats(values) for key, values in columns.items()}
                for camera_key in CAMERA_PATHS:
                    video_path = (
                        temporary
                        / f"videos/chunk-{chunk:03d}/observation.images.{camera_key}"
                        / f"episode_{output_episode_index:06d}.mp4"
                    )
                    shape, camera_stats = encode_camera_video(
                        _camera_dataset(episode, camera_key),
                        video_path,
                        fps=fps,
                        codec=video_codec,
                        crf=video_crf,
                        preset=video_preset,
                        camera_key=camera_key,
                    )
                    if camera_key in camera_shapes and camera_shapes[camera_key] != shape:
                        raise ValueError(
                            f"Camera {camera_key} resolution changed between episodes."
                        )
                    camera_shapes[camera_key] = shape
                    stats[f"observation.images.{camera_key}"] = camera_stats

                episode_records.append(
                    {
                        "episode_index": output_episode_index,
                        "tasks": [prompt],
                        "length": frame_count,
                    }
                )
                episode_stats.append({"episode_index": output_episode_index, "stats": stats})
                total_frames += frame_count
                print(
                    f"[LEROBOT V2.1] episode {output_episode_index}: "
                    f"{source_path.name}, frames={frame_count}, task={prompt!r}",
                    flush=True,
                )

        assert dataset_fps is not None and state_names is not None and action_names is not None
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [16],
                "names": state_names,
            },
            "action": {"dtype": "float32", "shape": [16], "names": action_names},
            **DEFAULT_FEATURES,
        }
        for camera_key, (height, width) in camera_shapes.items():
            features[f"observation.images.{camera_key}"] = {
                "dtype": "video",
                "shape": [3, height, width],
                "names": ["channels", "height", "width"],
                "info": {
                    "video.fps": float(dataset_fps),
                    "video.height": height,
                    "video.width": width,
                    "video.channels": 3,
                    "video.codec": "h264" if video_codec in {"libx264", "h264"} else video_codec,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }

        total_episodes = len(episode_records)
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": robot_type,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(tasks),
            "total_videos": total_episodes * len(camera_shapes),
            "total_chunks": math.ceil(total_episodes / CHUNK_SIZE),
            "chunks_size": CHUNK_SIZE,
            "fps": dataset_fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            ),
            "features": features,
        }
        _write_json(temporary / "meta/info.json", info)
        _write_jsonlines(
            temporary / "meta/tasks.jsonl",
            ({"task_index": index, "task": task} for task, index in tasks.items()),
        )
        _write_jsonlines(temporary / "meta/episodes.jsonl", episode_records)
        _write_jsonlines(temporary / "meta/episodes_stats.jsonl", episode_stats)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        f"[LEROBOT V2.1] complete: episodes={len(episode_records)}, "
        f"frames={total_frames}, output={output}",
        flush=True,
    )
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    episode_paths = discover_episodes(
        args.input,
        limit=args.limit,
        include_failed=args.include_failed,
    )
    convert_dataset(
        episode_paths,
        args.output,
        robot_type=args.robot_type,
        video_codec=args.video_codec,
        video_crf=args.video_crf,
        video_preset=args.video_preset,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
