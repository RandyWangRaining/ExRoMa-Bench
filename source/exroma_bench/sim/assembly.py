"""Attach the independently controlled arm module to the rover."""

from __future__ import annotations

from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.articulations import move_articulation_root
from isaacsim.core.utils.prims import get_articulation_root_api_prim_path, get_prim_at_path
from pxr import Gf, Sdf, Usd, UsdPhysics


def _is_root_joint(prim: Usd.Prim) -> bool:
    joint = UsdPhysics.Joint(prim)
    return bool(joint) and (
        not joint.GetBody0Rel().GetTargets() or not joint.GetBody1Rel().GetTargets()
    )


def _mask_collisions(base_path: str, attach_path: str) -> None:
    filtering = UsdPhysics.FilteredPairsAPI.Apply(get_prim_at_path(base_path))
    filtering.CreateFilteredPairsRel().AddTarget(Sdf.Path(attach_path))


def assemble_dual_piper(
    *,
    base_path: str = "/World/envs/env_0/robot",
    attach_path: str = "/World/envs/env_0/dual_piper",
    base_mount_frame: str = "body",
    attach_mount_frame: str = "dual_piper_mount",
    offset: tuple[float, float, float] = (0.0, 0.0, 0.42),
    orientation: tuple[float, float, float, float] = (
        0.7071067811865476,
        0.0,
        0.0,
        -0.7071067811865475,
    ),
) -> None:
    """Create a fixed joint while preserving two articulation controllers."""

    base_mount_path = f"{base_path}/{base_mount_frame}"
    attach_mount_path = f"{attach_path}/{attach_mount_frame}"
    if not get_prim_at_path(attach_mount_path).IsValid():
        SingleXFormPrim(attach_mount_path, translation=(0.0, 0.0, 0.0))

    attach_prim = get_prim_at_path(attach_path)
    articulation_root = get_prim_at_path(get_articulation_root_api_prim_path(attach_path))
    if articulation_root.HasAPI(UsdPhysics.ArticulationRootAPI):
        move_articulation_root(articulation_root, attach_prim)

    for prim in Usd.PrimRange(attach_prim):
        if _is_root_joint(prim):
            prim.GetProperty("physics:jointEnabled").Set(False)

    fixed_joint = UsdPhysics.FixedJoint.Define(
        attach_prim.GetStage(), f"{attach_mount_path}/ExRoMaFixedJoint"
    )
    fixed_joint.GetPrim().GetRelationship("physics:body0").SetTargets(
        [Sdf.Path(base_mount_path)]
    )
    fixed_joint.GetPrim().GetRelationship("physics:body1").SetTargets(
        [Sdf.Path(attach_mount_path)]
    )
    fixed_joint.GetLocalPos0Attr().Set(Gf.Vec3f(*offset))
    fixed_joint.GetLocalRot0Attr().Set(Gf.Quatf(*orientation))
    fixed_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.GetLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    fixed_joint.GetExcludeFromArticulationAttr().Set(True)
    _mask_collisions(base_path, attach_path)
