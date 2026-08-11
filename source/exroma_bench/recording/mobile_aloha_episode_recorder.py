"""HDF5 episode recorder for Mobile ALOHA teleoperation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch

from .episode_video import export_three_view_video


class MobileAlohaEpisodeRecorder:
    """Stream synchronized robot state, commands, object poses, and RGB images to HDF5."""

    FORMAT_VERSION = "exroma_bench.mobile_aloha.episode.v1"
    DUAL_PIPER_COMPACT_ENCODING = "dual_piper_compact"
    DUAL_PIPER_COMPACT_JOINT_NAMES = (
        *(f"fl_joint{index}" for index in range(1, 7)),
        "fl_gripper",
        *(f"fr_joint{index}" for index in range(1, 7)),
        "fr_gripper",
    )

    def __init__(
        self,
        output_dir: str | Path,
        *,
        fps: float = 10.0,
        jpeg_quality: int = 90,
        task_instruction: str = "pick up the water bottle",
        format_name: str | None = None,
        joint_encoding: str = "raw",
        export_video: bool = True,
        keep_hdf5: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = max(0.1, float(fps))
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.task_instruction = task_instruction
        self.format_name = format_name or self.FORMAT_VERSION
        self.export_video = bool(export_video)
        self.keep_hdf5 = bool(keep_hdf5)
        if not self.keep_hdf5 and not self.export_video:
            raise ValueError("keep_hdf5=False requires export_video=True")
        if joint_encoding not in {"raw", self.DUAL_PIPER_COMPACT_ENCODING}:
            raise ValueError(f"Unsupported joint encoding: {joint_encoding}")
        self.joint_encoding = joint_encoding

        self._file: h5py.File | None = None
        self._temp_path: Path | None = None
        self._episode_path: Path | None = None
        self._frame_count = 0
        self._next_sample_time = -1.0
        self._datasets: dict[str, h5py.Dataset] = {}
        self._source_joint_names: list[str] = []
        self._recorded_joint_names: list[str] = []
        self._joint_name_to_id: dict[str, int] = {}

    @property
    def is_recording(self) -> bool:
        return self._file is not None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _next_episode_index(self) -> int:
        indices = []
        for pattern in ("episode_*.hdf5", "episode_*.mp4"):
            for path in self.output_dir.glob(pattern):
                suffix = path.stem.removeprefix("episode_").split("_")[0]
                if suffix.isdigit():
                    indices.append(int(suffix))
        return max(indices, default=-1) + 1

    def start(
        self,
        *,
        metadata: dict[str, Any],
        joint_names: list[str],
        camera_names: list[str],
        task_instruction: str | None = None,
    ) -> Path:
        if self.is_recording:
            raise RuntimeError("An episode is already being recorded.")

        self._configure_joint_encoding(joint_names)

        episode_index = self._next_episode_index()
        self._episode_path = self.output_dir / f"episode_{episode_index:06d}.hdf5"
        self._temp_path = self.output_dir / f".episode_{episode_index:06d}.recording.hdf5"
        self._file = h5py.File(self._temp_path, "w")
        self._datasets = {}
        self._frame_count = 0
        self._next_sample_time = -1.0

        self._file.attrs["format"] = self.format_name
        self._file.attrs["complete"] = False
        self._file.attrs["success"] = False
        episode_instruction = task_instruction or self.task_instruction
        self._file.attrs["task_instruction"] = episode_instruction
        self._file.attrs["record_fps"] = self.fps
        self._file.attrs["jpeg_quality"] = self.jpeg_quality
        self._file.attrs["joint_encoding"] = self.joint_encoding
        self._file.attrs["joint_names_json"] = json.dumps(self._recorded_joint_names)
        self._file.attrs["source_joint_names_json"] = json.dumps(self._source_joint_names)
        self._file.attrs["action_names_json"] = json.dumps(
            [
                *(f"joint_position_target/{name}" for name in self._recorded_joint_names),
                "base_command/linear",
                "base_command/angular",
            ]
        )
        self._file.attrs["camera_names_json"] = json.dumps(camera_names)
        self._file.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False, default=str)
        return self._episode_path

    def _configure_joint_encoding(self, joint_names: list[str]) -> None:
        self._source_joint_names = list(joint_names)
        self._joint_name_to_id = {
            name: index for index, name in enumerate(self._source_joint_names)
        }
        if self.joint_encoding == "raw":
            self._recorded_joint_names = list(self._source_joint_names)
            return

        required_names = [
            *(f"fl_joint{index}" for index in range(1, 9)),
            *(f"fr_joint{index}" for index in range(1, 9)),
        ]
        missing = [name for name in required_names if name not in self._joint_name_to_id]
        if missing:
            raise ValueError(
                "The dual_piper_compact joint encoding is missing source joints: "
                + ", ".join(missing)
            )
        self._recorded_joint_names = list(self.DUAL_PIPER_COMPACT_JOINT_NAMES)

    def _encode_joint_vector(self, value: torch.Tensor) -> torch.Tensor:
        if self.joint_encoding == "raw":
            return value

        encoded = [
            value[self._joint_name_to_id[f"fl_joint{index}"]]
            for index in range(1, 7)
        ]
        encoded.append(
            0.5
            * (
                value[self._joint_name_to_id["fl_joint7"]]
                - value[self._joint_name_to_id["fl_joint8"]]
            )
        )
        encoded.extend(
            value[self._joint_name_to_id[f"fr_joint{index}"]]
            for index in range(1, 7)
        )
        encoded.append(
            0.5
            * (
                value[self._joint_name_to_id["fr_joint7"]]
                - value[self._joint_name_to_id["fr_joint8"]]
            )
        )
        return torch.stack(encoded)

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _append_array(self, path: str, value: Any) -> None:
        if self._file is None:
            return
        array = self._numpy(value)
        if path not in self._datasets:
            if "/" in path:
                parent, name = path.rsplit("/", 1)
                group = self._file.require_group(parent)
            else:
                name = path
                group = self._file
            self._datasets[path] = group.create_dataset(
                name,
                shape=(0, *array.shape),
                maxshape=(None, *array.shape),
                chunks=(1, *array.shape) if array.shape else True,
                dtype=array.dtype,
            )
        dataset = self._datasets[path]
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = array

    def _append_jpeg(self, path: str, rgb: Any) -> None:
        if self._file is None:
            return
        image = self._numpy(rgb)
        if image.ndim == 4:
            image = image[0]
        image = image[..., :3]
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        # Isaac camera tensors are RGB; OpenCV expects BGR before JPEG encoding.
        bgr = np.ascontiguousarray(image[..., ::-1])
        ok, encoded = cv2.imencode(
            ".jpg",
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError(f"Failed to JPEG-encode camera frame for {path}.")

        if path not in self._datasets:
            parent, name = path.rsplit("/", 1)
            group = self._file.require_group(parent)
            dtype = h5py.vlen_dtype(np.dtype("uint8"))
            dataset = group.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)
            dataset.attrs["encoding"] = "jpeg"
            dataset.attrs["decoded_color_order"] = "rgb"
            self._datasets[path] = dataset
        dataset = self._datasets[path]
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = encoded.reshape(-1)

    def capture(
        self,
        *,
        sim_time: float,
        robot,
        base_robot=None,
        joint_targets: torch.Tensor,
        base_command: tuple[float, float],
        cameras: dict[str, Any],
        objects: dict[str, Any],
    ) -> bool:
        if not self.is_recording:
            return False
        if self._next_sample_time < 0.0:
            self._next_sample_time = sim_time
        if sim_time + 1e-9 < self._next_sample_time:
            return False

        sample_period = 1.0 / self.fps
        self._next_sample_time += sample_period
        while self._next_sample_time <= sim_time:
            self._next_sample_time += sample_period
        self._append_array("time", np.asarray(sim_time, dtype=np.float64))
        self._append_array("actions/base_velocity", np.asarray(base_command, dtype=np.float32))
        self._append_array(
            "actions/joint_position_target",
            self._encode_joint_vector(joint_targets[0]).to(dtype=torch.float32),
        )
        self._append_array(
            "observations/qpos",
            self._encode_joint_vector(robot.data.joint_pos[0]).to(dtype=torch.float32),
        )
        self._append_array(
            "observations/qvel",
            self._encode_joint_vector(robot.data.joint_vel[0]).to(dtype=torch.float32),
        )

        base_source = robot if base_robot is None else base_robot
        base_pose = torch.cat((base_source.data.root_pos_w[0], base_source.data.root_quat_w[0])).to(
            dtype=torch.float32
        )
        self._append_array("observations/base_pose", base_pose)
        self._append_array(
            "observations/base_velocity",
            base_source.data.root_link_vel_w[0].to(dtype=torch.float32),
        )

        for object_name, obj in objects.items():
            pose = torch.cat(
                (obj.data.root_pos_w[0], obj.data.root_quat_w[0])
            ).to(dtype=torch.float32)
            self._append_array(f"observations/objects/{object_name}/pose", pose)

        for camera_name, camera in cameras.items():
            rgb = camera.data.output.get("rgb")
            if rgb is not None:
                self._append_jpeg(f"observations/images/{camera_name}/rgb_jpeg", rgb)

        self._frame_count += 1
        if self._frame_count % max(1, int(self.fps * 2)) == 0:
            self._file.flush()
        return True

    def finish(self, *, success: bool) -> Path | None:
        if self._file is None or self._temp_path is None or self._episode_path is None:
            return None

        self._add_robotwin_compatibility_links()
        self._file.attrs["complete"] = True
        self._file.attrs["success"] = bool(success)
        self._file.attrs["num_frames"] = self._frame_count
        self._file.flush()
        self._file.close()
        self._file = None
        self._temp_path.replace(self._episode_path)
        output_path = self._episode_path
        self._clear_paths()
        video_path = None
        if self.export_video:
            try:
                video_path = export_three_view_video(output_path, overwrite=True)
                print(f"[RECORD]: Saved three-view video -> {video_path}", flush=True)
            except Exception as exc:
                print(
                    f"[WARN]: Could not create three-view video for {output_path}: {exc}",
                    flush=True,
                )
        if video_path is not None and not self.keep_hdf5:
            output_path.unlink(missing_ok=True)
            print(f"[RECORD]: Removed temporary episode HDF5 -> {output_path}", flush=True)
            return video_path
        return output_path

    def _add_robotwin_compatibility_links(self) -> None:
        """Expose RoboTwin-style paths without duplicating stored arrays or JPEGs."""

        if self._file is None:
            return
        links = {
            "action": "actions/joint_position_target",
            "observations/images/cam_high": "observations/images/mast/rgb_jpeg",
            "observations/images/cam_left_wrist": (
                "observations/images/front_left/rgb_jpeg"
            ),
            "observations/images/cam_right_wrist": (
                "observations/images/front_right/rgb_jpeg"
            ),
        }
        for destination, source in links.items():
            if source in self._file and destination not in self._file:
                self._file[destination] = self._file[source]
        self._file.attrs["robotwin_compatible_paths"] = True

    def discard(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._temp_path is not None:
            self._temp_path.unlink(missing_ok=True)
        self._clear_paths()

    def close_incomplete(self) -> Path | None:
        if self._file is None or self._temp_path is None or self._episode_path is None:
            return None
        self._file.attrs["complete"] = False
        self._file.attrs["success"] = False
        self._file.attrs["num_frames"] = self._frame_count
        self._file.flush()
        self._file.close()
        self._file = None
        incomplete_path = self._episode_path.with_name(f"{self._episode_path.stem}_incomplete.hdf5")
        self._temp_path.replace(incomplete_path)
        self._clear_paths()
        return incomplete_path

    def _clear_paths(self) -> None:
        self._temp_path = None
        self._episode_path = None
        self._datasets = {}
        self._frame_count = 0
        self._next_sample_time = -1.0
