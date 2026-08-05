"""Create side-by-side RGB preview videos from Mobile ALOHA episodes."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Sequence

import cv2
import h5py
import numpy as np


DEFAULT_VIDEO_CAMERAS = ("mast", "front_left", "front_right")


def _decode_jpeg(encoded: np.ndarray, *, camera_name: str, frame_index: int) -> np.ndarray:
    frame = cv2.imdecode(np.asarray(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not decode {camera_name} frame {frame_index}.")
    return frame


def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (190, 34), (0, 0, 0), thickness=-1)
    cv2.putText(
        output,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    return output


def export_three_view_video(
    episode_path: str | Path,
    *,
    output_path: str | Path | None = None,
    camera_names: Sequence[str] = DEFAULT_VIDEO_CAMERAS,
    overwrite: bool = False,
) -> Path:
    """Export an MP4 with the mast and two wrist views arranged horizontally."""

    episode_path = Path(episode_path).expanduser().resolve()
    if output_path is None:
        output_path = episode_path.with_suffix(".mp4")
    else:
        output_path = Path(output_path).expanduser().resolve()
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output_path.with_name(f".{output_path.stem}.recording.mp4v.mp4")
    encoded_path = output_path.with_name(f".{output_path.stem}.recording.h264.mp4")
    raw_path.unlink(missing_ok=True)
    encoded_path.unlink(missing_ok=True)

    writer: cv2.VideoWriter | None = None
    try:
        with h5py.File(episode_path, "r") as episode:
            images = episode.get("observations/images")
            if images is None:
                raise ValueError(f"Episode has no camera observations: {episode_path}")

            datasets: list[h5py.Dataset] = []
            for camera_name in camera_names:
                dataset_path = f"{camera_name}/rgb_jpeg"
                if dataset_path not in images:
                    raise ValueError(f"Episode is missing observations/images/{dataset_path}: {episode_path}")
                datasets.append(images[dataset_path])

            frame_count = min(len(dataset) for dataset in datasets)
            if frame_count == 0:
                raise ValueError(f"Episode contains no camera frames: {episode_path}")

            first_frames = [
                _decode_jpeg(dataset[0], camera_name=name, frame_index=0)
                for name, dataset in zip(camera_names, datasets)
            ]
            panel_height, panel_width = first_frames[0].shape[:2]
            video_size = (panel_width * len(camera_names), panel_height)
            fps = max(0.1, float(episode.attrs.get("record_fps", 10.0)))
            writer = cv2.VideoWriter(
                raw_path.as_posix(),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                video_size,
            )
            if not writer.isOpened():
                raise RuntimeError(f"OpenCV could not open MP4 writer for {raw_path}")

            for frame_index in range(frame_count):
                frames = []
                for camera_name, dataset in zip(camera_names, datasets):
                    frame = _decode_jpeg(dataset[frame_index], camera_name=camera_name, frame_index=frame_index)
                    if frame.shape[:2] != (panel_height, panel_width):
                        frame = cv2.resize(frame, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
                    frames.append(_label_frame(frame, camera_name))
                writer.write(np.concatenate(frames, axis=1))

        writer.release()
        writer = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raw_path.replace(output_path)
            return output_path
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                raw_path.as_posix(),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                encoded_path.as_posix(),
            ],
            check=True,
        )
        encoded_path.replace(output_path)
        raw_path.unlink(missing_ok=True)
        return output_path
    except Exception:
        if writer is not None:
            writer.release()
        raw_path.unlink(missing_ok=True)
        encoded_path.unlink(missing_ok=True)
        raise
