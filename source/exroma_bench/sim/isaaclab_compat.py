"""Small Isaac Lab API compatibility helpers."""

from __future__ import annotations

import torch


def get_root_pose_w(data) -> torch.Tensor:
    pose = getattr(data, "root_pose_w", None)
    if pose is not None:
        return pose
    return torch.cat((data.root_pos_w, data.root_quat_w), dim=-1)
