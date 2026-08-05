"""Camera sensor settings for Mobile ALOHA D435-style cameras."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import CameraCfg

D435_CAMERA_LINK_TO_OPTICAL_QUAT = (0.5, -0.5, 0.5, -0.5)
"""Quaternion for the standard RealSense camera_link -> optical_frame rotation, in (w, x, y, z)."""

D435_PHYSICAL_SIZE_M = (0.02505, 0.090, 0.025)
"""Approximate RealSense D435 body size as (depth, width, height), in meters."""

D435_COLOR_HFOV_DEG = 69.4
D435_COLOR_VFOV_DEG = 42.5
D435_COLOR_OFFSET_M = (0.0, 0.015, 0.0)
D435_DEPTH_OFFSET_M = (0.0, 0.0, 0.0)
D435_INFRA2_OFFSET_M = (0.0, -0.050, 0.0)

D435_MAST_PARENT_TO_COLOR_OFFSET_M = (0.0376, 0.0153, -0.0015)
"""Approximate mast camera color optical center relative to camera_stand_2_Link."""

D435_WRIST_PARENT_TO_COLOR_OFFSET_M = (-0.0344, 0.07699312161647691, 0.05299634422589207)
"""Wrist camera color optical center relative to fl_link6/fr_link6.

This follows the original four-D435 ALOHA URDF:
fl/fr_camera_stand_joint xyz=(0.0, 0.013, 0.03) and
fl/fr_d435_camera_joint xyz=(-0.045, 0.042, -0.004), rpy=(0.52, 0.0, 0.0).
"""

D435_WRIST_PARENT_TO_OPTICAL_QUAT = (
    0.0,
    0.0,
    -0.2570805518921551,
    0.9663899781345132,
)
"""Quaternion for wrist link -> ROS optical frame, aimed along gripper forward with 0.52 rad downward pitch."""

D435_PIPER_WRIST_PARENT_TO_COLOR_OFFSET_M = (
    0.0076,
    0.079243390574347,
    0.045886337652894,
)
"""Color optical center for the newer PiPER wrist bracket relative to link6.

The transform follows the current Mobile ALOHA PiPER assembly:
camera_stand_joint xyz=(0.0, 0.03, 0.0) and
d435_camera_joint xyz=(-0.003, 0.023, 0.023), rpy=(0.35, 0.0, 0.0).
"""

D435_PIPER_WRIST_PARENT_TO_OPTICAL_QUAT = (
    0.0,
    0.0,
    -0.17410813759359595,
    0.9847265389049334,
)
"""ROS optical orientation for the newer PiPER wrist bracket."""

D435_WRIST_GRIPPER_CENTER_OFFSET_M = D435_WRIST_PARENT_TO_COLOR_OFFSET_M

MOBILE_ALOHA_CAMERA_LINKS = {
    "mast": "camera_stand_2_Link",
    "front_left": "fl_link6",
    "front_right": "fr_link6",
}

MOBILE_ALOHA_CAMERA_OFFSETS = {
    "mast": (D435_MAST_PARENT_TO_COLOR_OFFSET_M, D435_CAMERA_LINK_TO_OPTICAL_QUAT),
    "front_left": (D435_WRIST_PARENT_TO_COLOR_OFFSET_M, D435_WRIST_PARENT_TO_OPTICAL_QUAT),
    "front_right": (D435_WRIST_PARENT_TO_COLOR_OFFSET_M, D435_WRIST_PARENT_TO_OPTICAL_QUAT),
}


def add_xyz(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    """Add two xyz vectors."""

    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def normalize_quat(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Normalize a quaternion in wxyz order."""

    norm = math.sqrt(sum(value * value for value in quat))
    if norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(value / norm for value in quat)


def multiply_quat(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Multiply two quaternions in wxyz order."""

    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def ros_camera_pitch_quat(pitch_deg: float) -> tuple[float, float, float, float]:
    """Return a local ROS-camera-frame pitch quaternion; positive values look downward."""

    half_angle = math.radians(pitch_deg) * 0.5
    return (math.cos(half_angle), math.sin(half_angle), 0.0, 0.0)


def d435_color_intrinsic_matrix(width: int = 640, height: int = 480) -> list[float]:
    """Return an approximate D435 color camera intrinsic matrix."""

    fx = width / (2.0 * math.tan(math.radians(D435_COLOR_HFOV_DEG) / 2.0))
    fy = height / (2.0 * math.tan(math.radians(D435_COLOR_VFOV_DEG) / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


def build_d435_camera_cfg(
    *,
    camera_name: str,
    parent_link_name: str,
    width: int = 640,
    height: int = 480,
    update_period: float = 0.0,
    data_types: list[str] | None = None,
    clipping_range: tuple[float, float] = (0.05, 10.0),
    prim_path: str = "/World/envs/env_0/dual_piper",
    wrist_mount_mode: str = "physical",
    extra_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    extra_pitch_deg: float = 0.0,
) -> CameraCfg:
    """Build an IsaacLab camera sensor config attached under a Mobile ALOHA camera link."""

    if data_types is None:
        data_types = ["rgb", "distance_to_image_plane"]
    offset_pos, offset_rot = MOBILE_ALOHA_CAMERA_OFFSETS.get(
        camera_name,
        (D435_COLOR_OFFSET_M, D435_CAMERA_LINK_TO_OPTICAL_QUAT),
    )
    if camera_name in ("front_left", "front_right"):
        if wrist_mount_mode == "piper":
            offset_pos = D435_PIPER_WRIST_PARENT_TO_COLOR_OFFSET_M
            offset_rot = D435_PIPER_WRIST_PARENT_TO_OPTICAL_QUAT
        elif wrist_mount_mode == "gripper_center":
            offset_pos = D435_WRIST_GRIPPER_CENTER_OFFSET_M
    offset_pos = add_xyz(offset_pos, extra_offset)
    if extra_pitch_deg != 0.0:
        offset_rot = normalize_quat(multiply_quat(offset_rot, ros_camera_pitch_quat(extra_pitch_deg)))

    return CameraCfg(
        prim_path=f"{prim_path}/{parent_link_name}/ExRoMa_{camera_name}_d435",
        update_period=update_period,
        height=height,
        width=width,
        data_types=data_types,
        depth_clipping_behavior="max",
        update_latest_camera_pose=True,
        offset=CameraCfg.OffsetCfg(
            pos=offset_pos,
            rot=offset_rot,
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=d435_color_intrinsic_matrix(width=width, height=height),
            width=width,
            height=height,
            focal_length=1.93,
            focus_distance=2.0,
            clipping_range=clipping_range,
        ),
    )
