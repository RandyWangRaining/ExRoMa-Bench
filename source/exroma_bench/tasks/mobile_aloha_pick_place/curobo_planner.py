"""cuRobo motion-planning adapter for the two Mobile ALOHA front arms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from exroma_bench.paths import PROJECT_ROOT, asset_root


def _materialize_robot_config(template_path: Path) -> Path:
    """Resolve project and external-asset placeholders for cuRobo."""

    text = template_path.read_text(encoding="utf-8")
    text = text.replace("@PROJECT_ROOT@", PROJECT_ROOT.as_posix())
    text = text.replace("@ASSET_ROOT@", asset_root(require=True).as_posix())
    cache_dir = Path.home() / ".cache" / "exroma" / "curobo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / template_path.name
    output_path.write_text(text, encoding="utf-8")
    return output_path


@dataclass
class CuroboTrajectory:
    """A collision-checked joint trajectory ready for Isaac Lab execution."""

    joint_names: list[str]
    position: torch.Tensor
    velocity: torch.Tensor | None
    dt: float
    solve_time: float
    goalset_index: int


class MobileAlohaCuroboPlanner:
    """Plan one Mobile ALOHA front arm while the other joints remain untouched."""

    def __init__(
        self,
        arm: str,
        *,
        table_center_b: Sequence[float],
        table_quat_b: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        table_dims: Sequence[float],
        robot_config_stem: str = "mobile_aloha",
        interpolation_dt: float = 0.02,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        use_pose_goalset_planning: bool = False,
        self_collision_check: bool = False,
        device: str = "cuda:0",
    ) -> None:
        if arm not in {"fl", "fr"}:
            raise ValueError(f"Unsupported Mobile ALOHA arm: {arm}")
        if not torch.cuda.is_available():
            raise RuntimeError("cuRobo requires a CUDA-capable PyTorch installation.")

        self.arm = arm
        self.use_pose_goalset_planning = use_pose_goalset_planning
        self.joint_names = [f"{arm}_joint{i}" for i in range(1, 7)]
        self.tensor_args = TensorDeviceType(device=torch.device(device))
        side = "left" if arm == "fl" else "right"
        robot_cfg_template = (
            PROJECT_ROOT
            / "configs"
            / "curobo"
            / f"{robot_config_stem}_front_{side}.yml"
        )
        if not robot_cfg_template.is_file():
            raise FileNotFoundError(
                f"cuRobo robot config does not exist: {robot_cfg_template}"
            )
        robot_cfg = _materialize_robot_config(robot_cfg_template)
        self._world = self._make_world(table_center_b, table_quat_b, table_dims)

        config = MotionGenConfig.load_from_robot_config(
            robot_cfg.as_posix(),
            self._world,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
            num_ik_seeds=64,
            num_graph_seeds=2,
            num_trajopt_seeds=4,
            interpolation_dt=interpolation_dt,
            interpolation_steps=2000,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            collision_activation_distance=0.01,
            self_collision_check=self_collision_check,
            use_cuda_graph=True,
        )
        self.motion_gen = MotionGen(config)
        print(f"[CUROBO]: warming up {arm} planner...", flush=True)
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=True, n_goalset=7)
        print(f"[CUROBO]: {arm} planner ready.", flush=True)

    @staticmethod
    def _make_world(
        table_center_b: Sequence[float],
        table_quat_b: Sequence[float],
        table_dims: Sequence[float],
    ) -> WorldConfig:
        return WorldConfig(
            cuboid=[
                Cuboid(
                    name="work_table_top",
                    pose=[
                        *map(float, table_center_b),
                        *map(float, table_quat_b),
                    ],
                    dims=list(map(float, table_dims)),
                )
            ]
        )

    def update_table(
        self,
        table_center_b: Sequence[float],
        table_quat_b: Sequence[float],
        table_dims: Sequence[float],
    ) -> None:
        """Move the fixed-size table obstacle after the mobile base has aligned."""

        self._world = self._make_world(table_center_b, table_quat_b, table_dims)
        self.motion_gen.update_world(self._world)

    def joint_state(self, position: torch.Tensor | Sequence[float]) -> JointState:
        position_tensor = self.tensor_args.to_device(position)
        if position_tensor.ndim == 1:
            position_tensor = position_tensor.unsqueeze(0)
        return self.motion_gen.get_active_js(
            JointState.from_position(position_tensor, joint_names=self.joint_names)
        )

    def forward_pose(self, position: torch.Tensor | Sequence[float]) -> Pose:
        return self.motion_gen.compute_kinematics(self.joint_state(position)).ee_pose

    def orientation_candidates(
        self,
        current_position: torch.Tensor | Sequence[float],
    ) -> torch.Tensor:
        """Generate reachable wrist orientations around the current arm posture."""

        current = self.tensor_args.to_device(current_position).view(1, -1)
        variants = current.repeat(7, 1)
        variants[:, 0] += self.tensor_args.to_device([0.0, -0.10, 0.10, -0.18, 0.18, 0.0, 0.0])
        variants[:, 5] += self.tensor_args.to_device([0.0, 0.0, 0.0, 0.0, 0.0, -0.30, 0.30])
        limits = self.motion_gen.kinematics.get_joint_limits().position
        variants = torch.clamp(variants, limits[0].view(1, -1), limits[1].view(1, -1))
        poses = self.motion_gen.compute_kinematics(
            JointState.from_position(variants, joint_names=self.joint_names)
        ).ee_pose
        return poses.quaternion

    def plan_to_goalset(
        self,
        current_position: torch.Tensor | Sequence[float],
        goal_positions: torch.Tensor | Sequence[Sequence[float]],
        goal_quaternions: torch.Tensor | Sequence[Sequence[float]],
        *,
        max_attempts: int = 4,
        enable_finetune_trajopt: bool = True,
    ) -> CuroboTrajectory | None:
        """Use seeded goal-set IK, then optimize a collision-free joint trajectory."""

        print(f"[CUROBO]: preparing {self.arm} goalset query", flush=True)
        current = self.tensor_args.to_device(current_position).view(1, -1)
        limits = self.motion_gen.kinematics.get_joint_limits().position
        limit_margin = 1.0e-3
        clamped_current = torch.clamp(
            current,
            limits[0].view(1, -1) + limit_margin,
            limits[1].view(1, -1) - limit_margin,
        )
        correction = float(torch.max(torch.abs(clamped_current - current)).item())
        if correction > 0.0:
            print(
                f"[CUROBO]: clamped {self.arm} planning start by "
                f"{correction:.6f} rad to remain inside joint limits",
                flush=True,
            )
        current = clamped_current
        positions = self.tensor_args.to_device(goal_positions).view(1, -1, 3)
        quaternions = self.tensor_args.to_device(goal_quaternions).view(1, -1, 4)
        if positions.shape[1] != quaternions.shape[1]:
            raise ValueError("goal_positions and goal_quaternions must have the same candidate count")

        start_state = self.joint_state(current)
        if self.use_pose_goalset_planning:
            print(
                f"[CUROBO]: planning directly to {positions.shape[1]} Cartesian goal poses...",
                flush=True,
            )
            plan_result = self.motion_gen.plan_goalset(
                start_state,
                Pose(position=positions, quaternion=quaternions),
                MotionGenPlanConfig(
                    max_attempts=max_attempts,
                    enable_graph_attempt=2,
                    enable_finetune_trajopt=enable_finetune_trajopt,
                ),
            )
            if not bool(plan_result.success.item()):
                print(
                    f"[CUROBO]: {self.arm} Cartesian goal-set planning failed: "
                    f"{plan_result.status}; position_error={plan_result.position_error}; "
                    f"rotation_error={plan_result.rotation_error}",
                    flush=True,
                )
                return None
            plan = plan_result.get_interpolated_plan()
            goalset_index = (
                0
                if plan_result.goalset_index is None
                else int(plan_result.goalset_index.reshape(-1)[0].item())
            )
            return CuroboTrajectory(
                joint_names=list(self.joint_names),
                position=plan.position.detach(),
                velocity=None if plan.velocity is None else plan.velocity.detach(),
                dt=float(plan_result.interpolation_dt),
                solve_time=float(plan_result.solve_time),
                goalset_index=goalset_index,
            )

        seed_variants = current.repeat(9, 1)
        seed_variants[1:, 0] += self.tensor_args.to_device(
            [-0.16, -0.08, 0.08, 0.16, 0.0, 0.0, 0.0, 0.0]
        )
        seed_variants[5:, 5] += self.tensor_args.to_device([-0.30, -0.15, 0.15, 0.30])
        seed_variants = torch.clamp(seed_variants, limits[0].view(1, -1), limits[1].view(1, -1))

        print(f"[CUROBO]: solving {positions.shape[1]}-pose seeded IK goalset...", flush=True)
        ik_result = self.motion_gen.ik_solver.solve_goalset(
            Pose(position=positions, quaternion=quaternions),
            retract_config=current,
            seed_config=seed_variants.unsqueeze(0),
            return_seeds=4,
        )
        success = ik_result.success.reshape(-1)
        print(f"[CUROBO]: IK goalset returned {int(success.sum().item())} solutions", flush=True)
        if not bool(success.any().item()):
            print(
                f"[CUROBO]: {self.arm} goal-set IK failed; "
                f"best position error={float(ik_result.position_error.min().item()):.4f} m, "
                f"best rotation error={float(ik_result.rotation_error.min().item()):.4f}",
                flush=True,
            )
            return None

        solutions = ik_result.solution.reshape(-1, current.shape[-1])
        valid_indices = torch.nonzero(success, as_tuple=False).flatten()
        distances = torch.linalg.vector_norm(solutions[valid_indices] - current, dim=-1)
        goalset_indices = ik_result.goalset_index.reshape(-1)
        ordered_indices = valid_indices[torch.argsort(distances)]
        plan_result = None
        selected_goalset = 0
        for solution_rank, selected_tensor in enumerate(ordered_indices):
            selected = int(selected_tensor.item())
            goal_state = JointState.from_position(
                solutions[selected].view(1, -1),
                joint_names=self.joint_names,
            )
            print(
                f"[CUROBO]: optimizing collision-free joint trajectory "
                f"(IK solution {solution_rank + 1}/{len(ordered_indices)})...",
                flush=True,
            )
            candidate_result = self.motion_gen.plan_single_js(
                start_state,
                goal_state,
                MotionGenPlanConfig(
                    max_attempts=max_attempts,
                    enable_graph_attempt=2,
                    enable_finetune_trajopt=enable_finetune_trajopt,
                ),
            )
            if bool(candidate_result.success.item()):
                plan_result = candidate_result
                selected_goalset = int(goalset_indices[selected].item())
                break
            print(
                f"[CUROBO]: {self.arm} IK solution {solution_rank + 1} "
                f"trajectory failed: {candidate_result.status}",
                flush=True,
            )
        if plan_result is None:
            print(
                f"[CUROBO]: {self.arm} trajectory optimization failed for all "
                f"{len(ordered_indices)} IK solutions",
                flush=True,
            )
            return None

        plan = plan_result.get_interpolated_plan()
        return CuroboTrajectory(
            joint_names=list(self.joint_names),
            position=plan.position.detach(),
            velocity=None if plan.velocity is None else plan.velocity.detach(),
            dt=float(plan_result.interpolation_dt),
            solve_time=float(plan_result.solve_time),
            goalset_index=selected_goalset,
        )
