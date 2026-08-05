"""USD placement helpers for assets composed onto procedural terrain."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GroundSnapResult:
    minimum_height: float
    maximum_height: float
    target_bottom_z: float
    shift_z: float


def _sample_mesh_heights(
    stage,
    terrain_path: str,
    sample_points: list[tuple[float, float]],
) -> list[float]:
    from pxr import Gf, Usd, UsdGeom

    terrain_root = stage.GetPrimAtPath(terrain_path)
    if not terrain_root.IsValid():
        return []

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    meshes: list[tuple[np.ndarray, object, object]] = []
    for prim in Usd.PrimRange(terrain_root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points:
            continue
        point_array = np.asarray(points, dtype=np.float64)
        if point_array.ndim != 2 or point_array.shape[1] != 3:
            continue
        local_to_world = xform_cache.GetLocalToWorldTransform(prim)
        meshes.append((point_array, local_to_world, local_to_world.GetInverse()))

    heights: list[float] = []
    for world_x, world_y in sample_points:
        nearest_height = None
        nearest_distance_sq = math.inf
        for points, local_to_world, world_to_local in meshes:
            local_query = world_to_local.Transform(Gf.Vec3d(world_x, world_y, 0.0))
            delta_xy = points[:, :2] - np.asarray(local_query[:2], dtype=np.float64)
            distance_sq = np.einsum("ij,ij->i", delta_xy, delta_xy)
            point_index = int(np.argmin(distance_sq))
            candidate_distance = float(distance_sq[point_index])
            if candidate_distance >= nearest_distance_sq:
                continue
            world_point = local_to_world.Transform(
                Gf.Vec3d(*(float(value) for value in points[point_index]))
            )
            nearest_distance_sq = candidate_distance
            nearest_height = float(world_point[2])
        if nearest_height is not None:
            heights.append(nearest_height)
    return heights


def _shift_root_z(prim, delta_z: float) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    translate_op = next(
        (
            op
            for op in xform.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ),
        None,
    )
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
        current = Gf.Vec3d(0.0)
    else:
        current = translate_op.Get() or Gf.Vec3d(0.0)
    translated = (float(current[0]), float(current[1]), float(current[2]) + delta_z)
    vector_type = (
        Gf.Vec3d
        if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
        else Gf.Vec3f
    )
    translate_op.Set(vector_type(*translated))


def snap_roots_to_terrain(
    stage,
    *,
    terrain_path: str,
    root_paths: tuple[str, ...],
    vertical_offset: float = 0.0,
    embed_depth: float = 0.04,
) -> GroundSnapResult | None:
    """Move several USD roots so their combined lower bound follows terrain."""

    from pxr import Usd, UsdGeom

    roots = [stage.GetPrimAtPath(path) for path in root_paths]
    roots = [prim for prim in roots if prim.IsValid()]
    if not roots:
        return None

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    ranges = [bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange() for prim in roots]
    min_x = min(float(bounds.GetMin()[0]) for bounds in ranges)
    min_y = min(float(bounds.GetMin()[1]) for bounds in ranges)
    min_z = min(float(bounds.GetMin()[2]) for bounds in ranges)
    max_x = max(float(bounds.GetMax()[0]) for bounds in ranges)
    max_y = max(float(bounds.GetMax()[1]) for bounds in ranges)
    max_z = max(float(bounds.GetMax()[2]) for bounds in ranges)
    if not all(
        math.isfinite(value)
        for value in (min_x, min_y, min_z, max_x, max_y, max_z)
    ):
        return None

    sample_x = (
        min_x + 0.12 * (max_x - min_x),
        0.5 * (min_x + max_x),
        max_x - 0.12 * (max_x - min_x),
    )
    sample_y = (
        min_y + 0.12 * (max_y - min_y),
        0.5 * (min_y + max_y),
        max_y - 0.12 * (max_y - min_y),
    )
    heights = _sample_mesh_heights(
        stage,
        terrain_path,
        [(x, y) for x in sample_x for y in sample_y],
    )
    if not heights:
        return None

    target_bottom_z = min(heights) + vertical_offset - embed_depth
    delta_z = target_bottom_z - min_z
    for prim in roots:
        _shift_root_z(prim, delta_z)
    return GroundSnapResult(
        minimum_height=min(heights),
        maximum_height=max(heights),
        target_bottom_z=target_bottom_z,
        shift_z=delta_z,
    )
