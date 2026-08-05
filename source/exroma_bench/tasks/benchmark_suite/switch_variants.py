"""Official RoboTwin 056_switch variant metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import xml.etree.ElementTree as ET

from exroma_bench.paths import asset_root


SWITCH_SOURCE_ROOT = asset_root() / "robotwin" / "objects" / "056_switch"
SWITCH_URDF_ROOT = asset_root() / "urdf" / "robotwin" / "056_switch"
SWITCH_VARIANT_IDS = (
    "100880",
    "100901",
    "100905",
    "100906",
    "100907",
    "100914",
    "100933",
    "100937",
)

# RoboTwin's contact annotations are tied to its original gripper geometry.
# These narrow levers need a small PiPER-specific straight-line overtravel so
# the closed fingers reach the moving link instead of straddling it.
_PIPER_CONTACT_ADVANCE = {
    "100905": 0.015,
    "100907": 0.015,
    "100914": 0.015,
}

# RoboTwin maps its annotated contact frame to the gripper frame with A. The
# PiPER planner approaches along local Z, so B remaps RoboTwin local X to Z.
_ROBOTTWIN_GRIPPER_REMAP = (
    (0.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
)
_PIPER_APPROACH_REMAP = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)


@dataclass(frozen=True)
class SwitchVariant:
    object_id: str
    scale: float
    contact_base: str
    contact_offset: tuple[float, float, float]
    contact_rotation: tuple[tuple[float, float, float], ...]
    contact_advance: float


def _matmul(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(left[row][k] * right[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )


def _matvec(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row][col] * vector[col] for col in range(3))
        for row in range(3)
    )


def _rpy_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> tuple[tuple[float, float, float], ...]:
    """Return the URDF fixed-axis roll-pitch-yaw rotation matrix."""

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _root_to_link_transform(
    urdf_path: Path,
    link_name: str,
    scale: float,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[float, float, float],
]:
    """Resolve a URDF link pose in the root-link frame."""

    root = ET.parse(urdf_path).getroot()
    child_joints: dict[
        str,
        tuple[
            str,
            tuple[tuple[float, float, float], ...],
            tuple[float, float, float],
        ],
    ] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        xyz = (0.0, 0.0, 0.0)
        rpy = (0.0, 0.0, 0.0)
        if origin is not None:
            xyz = tuple(float(value) for value in origin.get("xyz", "0 0 0").split())
            rpy = tuple(float(value) for value in origin.get("rpy", "0 0 0").split())
        child_joints[child.get("link")] = (
            parent.get("link"),
            _rpy_matrix(*rpy),
            tuple(value * scale for value in xyz),
        )

    chain = []
    current = link_name
    visited = set()
    while current in child_joints:
        if current in visited:
            raise ValueError(f"Cycle in URDF joint chain at link '{current}': {urdf_path}")
        visited.add(current)
        parent, rotation, translation = child_joints[current]
        chain.append((rotation, translation))
        current = parent

    rotation_total = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    translation_total = (0.0, 0.0, 0.0)
    for rotation, translation in reversed(chain):
        rotated_translation = _matvec(rotation_total, translation)
        translation_total = tuple(
            translation_total[index] + rotated_translation[index]
            for index in range(3)
        )
        rotation_total = _matmul(rotation_total, rotation)
    return rotation_total, translation_total


def load_switch_variant(object_id: str) -> SwitchVariant:
    """Load scale and the original annotated contact frame for one variant."""

    if object_id not in SWITCH_VARIANT_IDS:
        choices = ", ".join(SWITCH_VARIANT_IDS)
        raise ValueError(f"Unsupported 056_switch model '{object_id}'. Available: {choices}")
    model_data_path = SWITCH_SOURCE_ROOT / object_id / "model_data.json"
    if not model_data_path.is_file():
        raise FileNotFoundError(
            f"Missing RoboTwin switch metadata: {model_data_path}. Run "
            "scripts/tools/download_robotwin_task_assets.py --objects 056_switch "
            "--all-variants."
        )
    payload = json.loads(model_data_path.read_text(encoding="utf-8"))
    scale_value = payload["scale"]
    scale = float(scale_value[0] if isinstance(scale_value, list) else scale_value)
    contact_data = payload["contact_points"][0]
    contact_base = str(contact_data["base"])
    contact_matrix = contact_data["matrix"]
    contact_rotation = tuple(
        tuple(float(contact_matrix[row][col]) for col in range(3))
        for row in range(3)
    )
    base_rotation, base_translation = _root_to_link_transform(
        SWITCH_URDF_ROOT / object_id / "mobility.urdf",
        contact_base,
        scale,
    )
    contact_rotation = _matmul(base_rotation, contact_rotation)
    contact_rotation = _matmul(contact_rotation, _ROBOTTWIN_GRIPPER_REMAP)
    contact_rotation = _matmul(contact_rotation, _PIPER_APPROACH_REMAP)
    contact_offset_local = tuple(
        float(contact_matrix[index][3]) * scale for index in range(3)
    )
    rotated_offset = _matvec(base_rotation, contact_offset_local)
    contact_offset = tuple(
        base_translation[index] + rotated_offset[index]
        for index in range(3)
    )
    return SwitchVariant(
        object_id=object_id,
        scale=scale,
        contact_base=contact_base,
        contact_offset=contact_offset,
        contact_rotation=contact_rotation,
        contact_advance=_PIPER_CONTACT_ADVANCE.get(object_id, 0.0),
    )
