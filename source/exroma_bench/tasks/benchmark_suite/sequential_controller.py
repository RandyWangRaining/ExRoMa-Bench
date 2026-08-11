"""cuRobo expert controllers for the RoboTwin-backed benchmark tasks."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable

import torch
from isaaclab.utils import math as math_utils

from exroma_bench.sim.isaaclab_compat import get_root_pose_w
from exroma_bench.tasks.mobile_aloha_pick_place.curobo_auto_controller import (
    CuroboAutoPickPlaceConfig,
    CuroboAutoPickPlaceController,
)

from .switch_variants import SWITCH_VARIANT_IDS, load_switch_variant
from .robotwin_sampling import sample_blocks_stack_easy


@dataclass
class PickPlaceStep:
    object_name: str
    arm: str
    cfg: CuroboAutoPickPlaceConfig
    after_success: Callable[[], None] | None = None
    spawn_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    spawn_yaw_range: tuple[float, float] | None = None
    spawn_root_z_offset: float | None = None
    arm_by_spawn_x: tuple[str, str] | None = None
    align_grasp_with_spawn_yaw: bool = False


class SequentialBenchmarkController:
    """Compose randomized cuRobo pick/place skills into a task episode."""

    def __init__(
        self,
        robot,
        joint_targets: torch.Tensor,
        scene,
        task_name: str,
        steps: list[PickPlaceStep],
        *,
        final_success: Callable[[], bool],
        post_wait: float = 0.6,
        episode_setup: (
            Callable[[random.Random], dict[str, tuple[float, float, float]]] | None
        ) = None,
        pose_sampler: (
            Callable[
                [random.Random],
                dict[
                    str,
                    tuple[
                        tuple[float, float, float],
                        tuple[float, float, float, float],
                        float,
                    ],
                ],
            ]
            | None
        ) = None,
        hold_active_object_until_grasp: bool = False,
    ):
        self.robot = robot
        self.scene = scene
        self.task_name = task_name
        self.steps = steps
        self.final_success = final_success
        self.post_wait = post_wait
        self.episode_setup = episode_setup
        self.pose_sampler = pose_sampler
        self.hold_active_object_until_grasp = hold_active_object_until_grasp
        self._joint_targets = joint_targets
        self.workers: dict[str, CuroboAutoPickPlaceController] = {}
        for step in steps:
            if step.cfg.arm != step.arm:
                raise ValueError(
                    f"Task {task_name!r} assigns step {step.object_name!r} to "
                    f"arm {step.arm!r}, but its cuRobo config uses {step.cfg.arm!r}."
                )
            if step.arm in self.workers:
                continue
            self.workers[step.arm] = CuroboAutoPickPlaceController(
                robot,
                joint_targets,
                scene[step.object_name],
                step.cfg,
            )
        self.active_step_index = 0
        self.state = "idle"
        self.state_elapsed = 0.0
        self.episode_elapsed = 0.0
        self.success_stable_steps = 0
        self.failure_reason = ""
        self._terminal_reported = False
        self._sampled_poses: dict[str, tuple[float, float, float]] = {}
        self._sampled_yaws: dict[str, float] = {}
        self._pre_manipulation_object_states: dict[str, torch.Tensor] = {}
        self._pending_step_index: int | None = None
        self._anchored_object_states: dict[str, torch.Tensor] = {}
        self._handover_release_started = False
        self._handover_release_elapsed = 0.0

    @property
    def joint_targets(self) -> torch.Tensor:
        return self._joint_targets

    @joint_targets.setter
    def joint_targets(self, value: torch.Tensor) -> None:
        self._joint_targets = value
        for worker in getattr(self, "workers", {}).values():
            worker.joint_targets = value

    @property
    def is_terminal(self) -> bool:
        return self.state in {"done", "failed"}

    @property
    def succeeded(self) -> bool:
        return self.state == "done"

    @property
    def active_worker(self) -> CuroboAutoPickPlaceController:
        return self.workers[self.steps[self.active_step_index].arm]

    def _ensure_worker(self, step: PickPlaceStep) -> CuroboAutoPickPlaceController:
        worker = self.workers.get(step.arm)
        if worker is None:
            worker = CuroboAutoPickPlaceController(
                self.robot,
                self._joint_targets,
                self.scene[step.object_name],
                step.cfg,
            )
            self.workers[step.arm] = worker
        return worker

    def _configure_step_from_pose(
        self,
        step: PickPlaceStep,
        position: tuple[float, float, float],
        yaw: float,
    ) -> None:
        if step.arm_by_spawn_x is not None:
            negative_x_arm, positive_x_arm = step.arm_by_spawn_x
            step.arm = (
                positive_x_arm
                if position[0] >= step.cfg.table_x
                else negative_x_arm
            )
            step.cfg.arm = step.arm
        if step.align_grasp_with_spawn_yaw:
            # PiPER approaches vertically while its horizontal jaw axis tracks
            # the object's planar rotation. The opposite jaw direction is
            # generated automatically by _fixed_axis_orientations_w().
            step.cfg.fixed_grasp_approach_axis_w = (0.0, 0.0, -1.0)
            step.cfg.fixed_grasp_closing_axis_w = (
                math.cos(yaw),
                math.sin(yaw),
                0.0,
            )

    def _set_object_pose(
        self,
        object_name: str,
        position: tuple[float, float, float],
        quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        obj = self.scene[object_name]
        root_state = obj.data.default_root_state.clone()
        root_state[:, :3] = torch.tensor(position, device=root_state.device)
        root_state[:, 3:7] = torch.tensor(quaternion, device=root_state.device)
        root_state[:, 7:] = 0.0
        obj.write_root_state_to_sim(root_state)
        self._pre_manipulation_object_states[object_name] = root_state.detach().clone()

    def _randomize_objects(self, rng: random.Random) -> None:
        seen: set[str] = set()
        self._sampled_poses = {}
        self._sampled_yaws = {}
        self._pre_manipulation_object_states = {}
        sampled = self.pose_sampler(rng) if self.pose_sampler is not None else None
        for step in self.steps:
            if step.object_name in seen:
                continue
            seen.add(step.object_name)
            cfg = step.cfg
            if sampled is not None:
                position, quaternion, yaw = sampled[step.object_name]
            else:
                position = (
                    cfg.table_x + rng.uniform(*cfg.spawn_x_range),
                    cfg.table_y + rng.uniform(*cfg.spawn_y_range),
                    (
                        cfg.table_height + step.spawn_root_z_offset
                        if step.spawn_root_z_offset is not None
                        else cfg.table_height
                        + cfg.object_clearance
                        + 0.5 * cfg.bottle_height
                    ),
                )
                quaternion = step.spawn_quaternion
                yaw = 0.0
                if step.spawn_yaw_range is not None:
                    yaw = rng.uniform(*step.spawn_yaw_range)
                    quaternion = (
                        math.cos(0.5 * yaw),
                        0.0,
                        0.0,
                        math.sin(0.5 * yaw),
                    )
            self._set_object_pose(
                step.object_name,
                position,
                quaternion,
            )
            self._sampled_poses[step.object_name] = position
            self._sampled_yaws[step.object_name] = yaw
            self._configure_step_from_pose(step, position, yaw)

    def hold_pre_manipulation_objects(self) -> None:
        """Keep randomized dynamic assets fixed while the rover approaches."""

        for object_name, root_state in self._pre_manipulation_object_states.items():
            self.scene[object_name].write_root_state_to_sim(root_state)

    def _activate_step(self, index: int) -> None:
        step = self.steps[index]
        self.active_step_index = index
        self._pending_step_index = None
        worker = self._ensure_worker(step)
        worker.target_object = self.scene[step.object_name]
        worker.cfg = step.cfg
        worker.joint_targets = self._joint_targets
        worker.start_from_current_pose()
        self.state = f"step_{index + 1}_{step.object_name}"
        self.state_elapsed = 0.0
        print(
            f"[BENCHMARK]: task={self.task_name}, step={index + 1}/{len(self.steps)}, "
            f"object={step.object_name}, arm={step.arm}",
            flush=True,
        )

    def _begin_inter_step_home(self, next_index: int) -> None:
        previous_step = self.steps[self.active_step_index]
        previous_worker = self.workers[previous_step.arm]
        home_ids = [
            *previous_worker.arm_joint_ids,
            *previous_worker.gripper_joint_ids,
        ]
        self._joint_targets[:, home_ids] = self.robot.data.default_joint_pos[
            :, home_ids
        ]
        self._pending_step_index = next_index
        self.state = "home_between_steps"
        self.state_elapsed = 0.0
        print(
            f"[BENCHMARK]: homing arm={previous_step.arm} before step "
            f"{next_index + 1}",
            flush=True,
        )

    def randomize_and_start(self, rng: random.Random) -> dict[str, object]:
        self._randomize_objects(rng)
        if self.episode_setup is not None:
            self._sampled_poses.update(self.episode_setup(rng))
        self._anchored_object_states = {}
        self._handover_release_started = False
        self._handover_release_elapsed = 0.0
        self.failure_reason = ""
        self._terminal_reported = False
        self.success_stable_steps = 0
        self.episode_elapsed = 0.0
        self.state_elapsed = 0.0
        self._activate_step(0)
        return {
            "task": self.task_name,
            "objects": dict(self._sampled_poses),
            "object_yaws": dict(self._sampled_yaws),
            "arms": {step.object_name: step.arm for step in self.steps},
        }

    def base_command(self) -> tuple[float, float]:
        if (
            self.is_terminal
            or self.state == "post_verify"
            or self.state == "home_between_steps"
        ):
            return 0.0, 0.0
        return self.active_worker.base_command()

    def update(self, dt: float) -> None:
        if self.is_terminal:
            return
        for object_name, root_state in self._anchored_object_states.items():
            self.scene[object_name].write_root_state_to_sim(root_state)
        self.episode_elapsed += dt
        self.state_elapsed += dt
        if self.episode_elapsed > 120.0:
            self.failure_reason = "benchmark episode timeout"
            self.state = "failed"
            return

        if self.state == "post_verify":
            if self.final_success():
                self.success_stable_steps += 1
            else:
                self.success_stable_steps = 0
            if self.success_stable_steps >= 12:
                self.state = "done"
            elif self.state_elapsed >= max(3.0, self.post_wait + 2.0):
                self.failure_reason = "final task geometry did not satisfy success criteria"
                self.state = "failed"
            return

        if self.state == "home_between_steps":
            if self._pending_step_index is None:
                self.failure_reason = "inter-step home target was lost"
                self.state = "failed"
                return
            previous_worker = self.active_worker
            arm_ids = previous_worker.arm_joint_ids
            actual = self.robot.data.joint_pos[0, arm_ids]
            target = self.robot.data.default_joint_pos[0, arm_ids]
            tracking_error = float(torch.max(torch.abs(actual - target)).item())
            if tracking_error <= 0.08:
                self._activate_step(self._pending_step_index)
            elif self.state_elapsed >= 6.0:
                self.failure_reason = (
                    f"arm {self.steps[self.active_step_index].arm} did not home "
                    f"between steps (error={tracking_error:.3f} rad)"
                )
                self.state = "failed"
            return

        worker = self.active_worker
        if self.hold_active_object_until_grasp and worker.state in {
            "settle",
            "plan_pregrasp",
            "execute_pregrasp",
            "execute_grasp",
            "close",
        }:
            object_name = self.steps[self.active_step_index].object_name
            root_state = self._pre_manipulation_object_states.get(object_name)
            if root_state is not None:
                self.scene[object_name].write_root_state_to_sim(root_state)
        if self.task_name == "scan_object" and self.active_step_index == 1:
            scanner_worker = self.workers["fl"]
            scanner_worker._set_gripper(scanner_worker.cfg.closed_gripper)
        if self.task_name == "handover_block" and self.active_step_index == 1:
            giver = self.workers["fl"]
            if not self._handover_release_started:
                giver._set_gripper(giver.cfg.closed_gripper)
            if (
                worker.state == "close"
                and worker.state_elapsed + dt >= worker.cfg.gripper_hold_time
            ):
                if not self._handover_release_started:
                    self._handover_release_started = True
                    self._handover_release_elapsed = 0.0
                    giver._grasp_pose_stabilization_active = False
                    worker._capture_grasp_pose_stabilization()
                    position = self.scene[self.steps[1].object_name].data.root_pos_w[0]
                    print(
                        "[BENCHMARK]: airborne handover: right gripper secured; "
                        f"opening left gripper at z={float(position[2]):.3f} m",
                        flush=True,
                    )
                giver._set_gripper(giver.cfg.open_gripper)
                worker._set_gripper(worker.cfg.closed_gripper)
                worker._apply_grasp_pose_stabilization()
                self._handover_release_elapsed += dt
                if self._handover_release_elapsed < 0.5:
                    return
            if self._handover_release_started:
                giver._set_gripper(giver.cfg.open_gripper)
                if worker.state != "close":
                    self._joint_targets[:, giver.arm_joint_ids] = (
                        self.robot.data.default_joint_pos[:, giver.arm_joint_ids]
                    )
        worker.update(dt)
        if not worker.is_terminal:
            return
        if not worker.succeeded:
            self.failure_reason = (
                f"step {self.active_step_index + 1} failed: {worker.failure_reason}"
            )
            self.state = "failed"
            return

        step = self.steps[self.active_step_index]
        if self.task_name in {"stack_blocks_two", "scan_object", "scan_rock"}:
            anchored_state = self.scene[step.object_name].data.root_state_w.clone()
            anchored_state[:, 7:] = 0.0
            self._anchored_object_states[step.object_name] = anchored_state
        if step.after_success is not None:
            step.after_success()
        next_index = self.active_step_index + 1
        if next_index < len(self.steps):
            if self.task_name == "handover_block" and self.active_step_index == 0:
                position = self.scene[step.object_name].data.root_pos_w[0]
                print(
                    "[BENCHMARK]: left arm holding block at airborne transfer pose "
                    f"z={float(position[2]):.3f} m",
                    flush=True,
                )
                self._activate_step(next_index)
                return
            if self.task_name == "scan_object" and self.active_step_index == 0:
                print(
                    "[BENCHMARK]: scanner held at scan pose; "
                    "moving the opposite arm with the tea box",
                    flush=True,
                )
                self._activate_step(next_index)
                # The rover has already completed its one-time base approach.
                # Holding the scanner introduces small chassis oscillations, so
                # the second arm should not repeat the single-object base-settle
                # gate before planning its pregrasp.
                next_worker = self.active_worker
                next_worker._refresh_table_obstacle()
                next_worker._transition("plan_pregrasp")
                return
            self._begin_inter_step_home(next_index)
            return
        self.state = "post_verify"
        self.state_elapsed = 0.0

    def report_terminal_once(self) -> None:
        if not self.is_terminal or self._terminal_reported:
            return
        self._terminal_reported = True
        print(
            f"[BENCHMARK]: task={self.task_name}, terminal={self.state}, "
            f"success={self.succeeded}, reason={self.failure_reason or 'success'}, "
            f"elapsed={self.episode_elapsed:.2f}s",
            flush=True,
        )


def _cfg(
    *,
    arm: str,
    table_x: float,
    table_y: float,
    table_height: float,
    table_length: float,
    table_width: float,
    spawn_x: tuple[float, float],
    spawn_y: tuple[float, float],
    object_height: float,
    target_x: float,
    target_y: float,
    target_z: float,
    target_radius: float,
    closed_gripper: float,
    strict_collision_check: bool = False,
) -> CuroboAutoPickPlaceConfig:
    return CuroboAutoPickPlaceConfig(
        arm=arm,
        table_x=table_x,
        table_y=table_y,
        table_height=table_height,
        table_length=table_length,
        table_width=table_width,
        target_x=target_x,
        target_y=target_y,
        target_radius=target_radius,
        target_center_z=target_z,
        target_z_tolerance=0.025,
        target_max_linear_speed=0.06,
        robot_config_stem="dual_piper",
        planner_position_threshold=0.008,
        planner_rotation_threshold=0.08,
        planner_self_collision_check=strict_collision_check,
        align_base=False,
        spawn_x_range=spawn_x,
        spawn_y_range=spawn_y,
        bottle_height=object_height,
        object_clearance=0.006,
        grasp_center_z_offset=min(0.025, max(0.005, 0.5 * object_height - 0.005)),
        pregrasp_distance=0.11,
        lift_height=0.12,
        preplace_height=0.14,
        open_gripper=0.040,
        closed_gripper=closed_gripper,
        settle_time=1.2,
        gripper_hold_time=0.9,
        release_hold_time=0.7,
        retreat_distance=0.10,
        trajectory_tracking_tolerance=0.08,
        trajectory_tracking_timeout=5.0,
        overall_timeout=45.0,
        stable_success_steps=10,
    )


def _position(scene, name: str) -> torch.Tensor:
    return scene[name].data.root_pos_w[0]


def _near_xy(scene, name: str, x: float, y: float, tolerance: float) -> bool:
    pos = _position(scene, name)
    return math.hypot(float(pos[0]) - x, float(pos[1]) - y) <= tolerance


def _move_rigid_object(
    scene,
    name: str,
    position: tuple[float, float, float],
) -> None:
    obj = scene[name]
    root_state = obj.data.root_state_w.clone()
    root_state[:, :3] = torch.tensor(position, device=root_state.device)
    root_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(root_state)


class CuroboArticulatedMechanismController:
    """Approach a real RoboTwin mechanism with cuRobo and actuate its joint."""

    def __init__(
        self,
        robot,
        joint_targets: torch.Tensor,
        scene,
        task_name: str,
        *,
        mechanism_name: str,
        contact_offset: tuple[float, float, float],
        joint_fraction: float,
        planning_cfg: CuroboAutoPickPlaceConfig,
        object_cfg: CuroboAutoPickPlaceConfig | None = None,
        physical_contact_only: bool = False,
        contact_quaternion_local: tuple[float, float, float, float] | None = None,
        contact_rotation_offsets_deg: tuple[float, ...] = (0.0,),
        precontact_standoff: float = 0.10,
        contact_advance: float = 0.0,
        planning_max_attempts: int = 4,
        sample_metadata: dict[str, object] | None = None,
        direct_precontact_goalset: bool = False,
        planning_finetune: bool = True,
    ):
        self.robot = robot
        self.scene = scene
        self.task_name = task_name
        self.mechanism = scene[mechanism_name]
        self.contact_offset = contact_offset
        self.joint_fraction = joint_fraction
        self.object_cfg = object_cfg
        self.physical_contact_only = physical_contact_only
        self.contact_quaternion_local = contact_quaternion_local
        self.contact_rotation_offsets_deg = contact_rotation_offsets_deg
        self.precontact_standoff = precontact_standoff
        self.contact_advance = contact_advance
        self.planning_max_attempts = planning_max_attempts
        self.sample_metadata = dict(sample_metadata or {})
        self.direct_precontact_goalset = direct_precontact_goalset
        self.planning_finetune = planning_finetune
        self._joint_targets = joint_targets
        self.contact_worker = CuroboAutoPickPlaceController(
            robot,
            joint_targets,
            self.mechanism,
            planning_cfg,
        )
        self.contact_worker.planner.use_pose_goalset_planning = False
        self.object_worker = None
        if object_cfg is not None:
            self.object_worker = CuroboAutoPickPlaceController(
                robot,
                joint_targets,
                scene["benchmark_cabinet_object"],
                object_cfg,
            )
        self.state = "idle"
        self.state_elapsed = 0.0
        self.episode_elapsed = 0.0
        self.success_stable_steps = 0
        self.failure_reason = ""
        self._terminal_reported = False
        self._contact_position_w: torch.Tensor | None = None
        self._contact_quaternion_w: torch.Tensor | None = None
        self._planned_contact_trajectory = None
        self._mechanism_target: torch.Tensor | None = None
        self._sampled_poses: dict[str, tuple[float, float, float]] = {}

    @property
    def joint_targets(self) -> torch.Tensor:
        return self._joint_targets

    @joint_targets.setter
    def joint_targets(self, value: torch.Tensor) -> None:
        self._joint_targets = value
        self.contact_worker.joint_targets = value
        if self.object_worker is not None:
            self.object_worker.joint_targets = value

    @property
    def is_terminal(self) -> bool:
        return self.state in {"done", "failed"}

    @property
    def succeeded(self) -> bool:
        return self.state == "done"

    def _transition(self, state: str) -> None:
        self.state = state
        self.state_elapsed = 0.0
        self.contact_worker.state_elapsed = 0.0
        print(
            f"[BENCHMARK]: task={self.task_name}, state={state}",
            flush=True,
        )

    def _mesh_center_w(self) -> torch.Tensor:
        offset = torch.tensor(
            self.contact_offset,
            device=self.robot.device,
            dtype=torch.float32,
        ).view(1, 3)
        rotated_offset = math_utils.quat_apply(
            self.mechanism.data.root_quat_w,
            offset,
        )
        return (self.mechanism.data.root_pos_w + rotated_offset)[0]

    def _reset_mechanism(self) -> None:
        joint_pos = self.mechanism.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.mechanism.write_joint_state_to_sim(joint_pos, joint_vel)
        self.mechanism.set_joint_position_target(joint_pos)
        upper = self.mechanism.data.soft_joint_pos_limits[..., 1]
        self._mechanism_target = joint_pos.clone()
        if self.physical_contact_only:
            self._mechanism_target[:, 0] = upper[:, 0] - 0.05
        else:
            self._mechanism_target[:, 0] = upper[:, 0] * self.joint_fraction

    def _randomize_cabinet_object(self, rng: random.Random) -> None:
        if self.object_worker is None or self.object_cfg is None:
            return
        cfg = self.object_cfg
        position = (
            cfg.table_x + rng.uniform(*cfg.spawn_x_range),
            cfg.table_y + rng.uniform(*cfg.spawn_y_range),
            cfg.table_height + cfg.object_clearance + 0.5 * cfg.bottle_height,
        )
        obj = self.scene["benchmark_cabinet_object"]
        root_state = obj.data.default_root_state.clone()
        root_state[:, :3] = torch.tensor(position, device=root_state.device)
        root_state[:, 3:7] = torch.tensor(
            (1.0, 0.0, 0.0, 0.0),
            device=root_state.device,
        )
        root_state[:, 7:] = 0.0
        obj.write_root_state_to_sim(root_state)
        self._sampled_poses["benchmark_cabinet_object"] = position

    def randomize_and_start(self, rng: random.Random) -> dict[str, object]:
        self._joint_targets.copy_(self.robot.data.joint_pos)
        self._reset_mechanism()
        self._sampled_poses = {}
        self._randomize_cabinet_object(rng)
        self.failure_reason = ""
        self._terminal_reported = False
        self.success_stable_steps = 0
        self.episode_elapsed = 0.0
        self._contact_position_w = None
        self._contact_quaternion_w = None
        self._planned_contact_trajectory = None
        self.contact_worker._set_gripper(
            0.006 if self.physical_contact_only else 0.04
        )
        self._transition("settle")
        return {
            "task": self.task_name,
            "objects": dict(self._sampled_poses),
            **self.sample_metadata,
        }

    def base_command(self) -> tuple[float, float]:
        return 0.0, 0.0

    def _plan_contact_pose(self, *, standoff: float) -> bool:
        contact = self._mesh_center_w()
        if self.physical_contact_only:
            root_rotation_w = math_utils.matrix_from_quat(
                self.mechanism.data.root_quat_w
            )[0]
            face_dir_w = -root_rotation_w[:, 0]
            print(
                f"[BENCHMARK]: switch face_dir="
                f"({face_dir_w[0].item():.3f}, {face_dir_w[1].item():.3f}, "
                f"{face_dir_w[2].item():.3f}); arm={self.contact_worker.cfg.arm}",
                flush=True,
            )
        if self.contact_quaternion_local is None:
            quaternions_w = self.contact_worker._table_aligned_grasp_orientations_w(
                contact.clone()
            )
            _, current_quaternion_w = self.contact_worker._current_ee_pose_w()
            quaternions_w = torch.cat(
                (current_quaternion_w.detach(), quaternions_w[:-1]),
                dim=0,
            )
        else:
            base_quaternion_local = torch.tensor(
                self.contact_quaternion_local,
                device=self.robot.device,
                dtype=torch.float32,
            ).view(1, 4)
            offsets = torch.deg2rad(
                torch.tensor(
                    self.contact_rotation_offsets_deg,
                    device=self.robot.device,
                    dtype=torch.float32,
                )
            )
            half_offsets = 0.5 * offsets
            rotations_local_y = torch.stack(
                (
                    torch.cos(half_offsets),
                    torch.zeros_like(offsets),
                    torch.sin(half_offsets),
                    torch.zeros_like(offsets),
                ),
                dim=-1,
            )
            quaternions_local = math_utils.quat_mul(
                base_quaternion_local.repeat(len(offsets), 1),
                rotations_local_y,
            )
            quaternions_w = math_utils.quat_mul(
                self.mechanism.data.root_quat_w.repeat(len(offsets), 1),
                quaternions_local,
            )
        candidate_count = len(quaternions_w)
        approach_w = math_utils.matrix_from_quat(quaternions_w)[:, :, 2]
        contact_positions_w = (
            contact.repeat(candidate_count, 1)
            + self.contact_advance * approach_w
        )
        precontact_positions_w = contact_positions_w - standoff * approach_w
        root_pose_w = get_root_pose_w(self.robot.data).repeat(candidate_count, 1)
        positions_b, quaternions_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            precontact_positions_w,
            quaternions_w,
        )
        print(
            f"[BENCHMARK]: contact mesh center="
            f"({contact[0].item():.3f}, {contact[1].item():.3f}, "
            f"{contact[2].item():.3f}); submitting {candidate_count} "
            f"cuRobo precontact candidates; contact_advance="
            f"{self.contact_advance:.3f} m",
            flush=True,
        )
        planner = self.contact_worker.planner
        planner.use_pose_goalset_planning = self.direct_precontact_goalset
        selected = 0
        try:
            if self.physical_contact_only:
                trajectory = None
                for candidate_index in range(candidate_count):
                    print(
                        f"[BENCHMARK]: planning ordered switch candidate "
                        f"{candidate_index + 1}/{candidate_count} "
                        f"(wrist offset="
                        f"{self.contact_rotation_offsets_deg[candidate_index]:.1f} deg)",
                        flush=True,
                    )
                    trajectory = planner.plan_to_goalset(
                        self.contact_worker._current_arm_position(),
                        positions_b[candidate_index : candidate_index + 1]
                        .detach()
                        .cpu(),
                        quaternions_b[candidate_index : candidate_index + 1]
                        .detach()
                        .cpu(),
                        max_attempts=self.planning_max_attempts,
                        enable_finetune_trajopt=self.planning_finetune,
                    )
                    if trajectory is not None:
                        selected = candidate_index
                        break
            else:
                trajectory = planner.plan_to_goalset(
                    self.contact_worker._current_arm_position(),
                    positions_b.detach().cpu(),
                    quaternions_b.detach().cpu(),
                    max_attempts=self.planning_max_attempts,
                    enable_finetune_trajopt=self.planning_finetune,
                )
        finally:
            # The one-pose contact segment uses seeded joint-space planning;
            # this also avoids changing a warmed seven-pose CUDA graph in place.
            planner.use_pose_goalset_planning = False
        if trajectory is None:
            return False
        self.contact_worker.trajectory = trajectory
        self.contact_worker._capture_trajectory_base_pose()
        if not self.physical_contact_only:
            selected = trajectory.goalset_index
        self._contact_position_w = contact_positions_w[selected]
        self._contact_quaternion_w = quaternions_w[selected]
        print(
            f"[BENCHMARK]: selected precontact candidate={selected}, "
            f"waypoints={len(trajectory.position)}, solve={trajectory.solve_time:.3f}s",
            flush=True,
        )
        return True

    def _execute_contact_trajectory(self) -> bool:
        self.contact_worker.state_elapsed = self.state_elapsed
        completed = self.contact_worker._execute_trajectory()
        if self.contact_worker.state == "failed":
            self.failure_reason = self.contact_worker.failure_reason
            self.state = "failed"
        return completed

    def _mechanism_reached_target(self) -> bool:
        if self._mechanism_target is None:
            return False
        target = self._mechanism_target[:, 0]
        actual = self.mechanism.data.joint_pos[:, 0]
        if self.physical_contact_only:
            return bool(torch.all(actual >= target).item())
        return bool(torch.all(torch.abs(actual - target) <= 0.035).item())

    def update(self, dt: float) -> None:
        if self.is_terminal:
            return
        self.state_elapsed += dt
        self.episode_elapsed += dt
        if self.episode_elapsed >= 120.0:
            self.failure_reason = "articulated benchmark episode timeout"
            self.state = "failed"
            return
        if self.state == "settle":
            if self.state_elapsed >= 1.0:
                self._transition("plan_precontact")
            return
        if self.state == "plan_precontact":
            if self._plan_contact_pose(standoff=self.precontact_standoff):
                self._transition("execute_precontact")
            else:
                self.failure_reason = "cuRobo could not plan to mechanism precontact"
                self.state = "failed"
            return
        if self.state == "execute_precontact":
            if self._execute_contact_trajectory():
                self._transition("plan_contact")
            return
        if self.state == "plan_contact":
            if self._planned_contact_trajectory is not None:
                self.contact_worker.trajectory = self._planned_contact_trajectory
                self.contact_worker._capture_trajectory_base_pose()
                self._planned_contact_trajectory = None
                self._transition("execute_contact")
                return
            if (
                self.physical_contact_only
                and self._contact_position_w is not None
                and self._contact_quaternion_w is not None
            ):
                contact_position_b, contact_quaternion_b = (
                    self.contact_worker._world_pose_to_base(
                        self._contact_position_w,
                        self._contact_quaternion_w,
                    )
                )
                trajectory = self.contact_worker.planner.plan_to_goalset(
                    self.contact_worker._current_arm_position(),
                    contact_position_b.detach().cpu(),
                    contact_quaternion_b.detach().cpu(),
                    max_attempts=self.planning_max_attempts,
                    enable_finetune_trajopt=self.planning_finetune,
                )
                if trajectory is not None:
                    self.contact_worker.trajectory = trajectory
                    self.contact_worker._capture_trajectory_base_pose()
                    self._transition("execute_contact")
                else:
                    self.failure_reason = "cuRobo could not plan the contact motion"
                    self.state = "failed"
                return
            if (
                self._contact_position_w is not None
                and self._contact_quaternion_w is not None
                and self.contact_worker._plan_world_pose(
                    self._contact_position_w,
                    self._contact_quaternion_w,
                )
            ):
                self._transition("execute_contact")
            else:
                self.failure_reason = "cuRobo could not plan the contact motion"
                self.state = "failed"
            return
        if self.state == "execute_contact":
            if self._execute_contact_trajectory():
                self.contact_worker._set_gripper(0.006)
                self._transition(
                    "verify_physical_contact"
                    if self.physical_contact_only
                    else "actuate_mechanism"
                )
            return
        if self.state == "verify_physical_contact":
            self.contact_worker._set_gripper(0.006)
            if self._mechanism_reached_target():
                self.success_stable_steps += 1
            else:
                self.success_stable_steps = 0
            if self.success_stable_steps >= 6:
                self.state = "done"
            elif self.state_elapsed >= 3.0:
                actual = float(self.mechanism.data.joint_pos[0, 0].item())
                target = float(self._mechanism_target[0, 0].item())
                self.failure_reason = (
                    "physical switch contact did not reach joint limit "
                    f"(actual={actual:.3f}, required={target:.3f})"
                )
                self.state = "failed"
            return
        if self.state == "actuate_mechanism":
            if self._mechanism_target is None:
                self.failure_reason = "mechanism target was not initialized"
                self.state = "failed"
                return
            # The current benchmark drives the drawer articulation directly;
            # release the handle so the stationary gripper does not constrain
            # the moving drawer.
            self.contact_worker._set_gripper(0.04)
            self.mechanism.set_joint_position_target(self._mechanism_target)
            if self._mechanism_reached_target():
                self.success_stable_steps += 1
            else:
                self.success_stable_steps = 0
            if self.success_stable_steps < 12:
                if self.state_elapsed >= 8.0:
                    self.failure_reason = "mechanism joint did not reach target"
                    self.state = "failed"
                return
            if self.object_worker is None:
                self.state = "done"
                return
            self.object_worker.joint_targets = self._joint_targets
            self.object_worker.start_from_current_pose()
            self._transition("place_cabinet_object")
            return
        if self.state == "place_cabinet_object":
            assert self.object_worker is not None
            self.object_worker.update(dt)
            if not self.object_worker.is_terminal:
                return
            if self.object_worker.succeeded and self._mechanism_reached_target():
                self.state = "done"
            else:
                self.failure_reason = (
                    self.object_worker.failure_reason
                    or "cabinet object placement failed"
                )
                self.state = "failed"

    def report_terminal_once(self) -> None:
        if not self.is_terminal or self._terminal_reported:
            return
        self._terminal_reported = True
        print(
            f"[BENCHMARK]: task={self.task_name}, terminal={self.state}, "
            f"success={self.succeeded}, reason={self.failure_reason or 'success'}, "
            f"elapsed={self.episode_elapsed:.2f}s",
            flush=True,
        )


def _stack_two_controller(
    robot,
    joint_targets,
    scene,
    geom,
    *,
    strict_collision_check: bool = False,
    planner_enabled: bool = True,
):
    tx, ty, tz, tl, tw = geom
    # RoboTwin's robot faces +Y, while this rover faces -Y at the table. Mirror
    # task-local Y so the target and spawn distribution retain their robot-
    # relative geometry.
    target_x, target_y = tx, ty + 0.10
    names = (
        "benchmark_block_red",
        "benchmark_block_black",
    )
    steps = []
    for index, name in enumerate(names):
        arm = "fl"
        steps.append(
            PickPlaceStep(
                name,
                arm,
                _cfg(
                    arm=arm,
                    table_x=tx,
                    table_y=ty,
                    table_height=tz,
                    table_length=tl,
                    table_width=tw,
                    spawn_x=(-0.25, 0.25),
                    spawn_y=(-0.15, 0.05),
                    object_height=0.05,
                    target_x=target_x,
                    target_y=target_y,
                    target_z=tz + 0.025 + 0.05 * index,
                    target_radius=0.025,
                    closed_gripper=0.006,
                    strict_collision_check=strict_collision_check,
                ),
                arm_by_spawn_x=("fr", "fl"),
                align_grasp_with_spawn_yaw=True,
            )
        )
        steps[-1].cfg.grasp_center_z_offset = 0.008
        steps[-1].cfg.planner_enabled = planner_enabled
        steps[-1].cfg.planner_table_z_offset = -0.012
        steps[-1].cfg.grasp_pose_stabilization = True
        steps[-1].cfg.stabilize_during_release = True
        steps[-1].cfg.compensate_grasp_offset_at_target = True
        steps[-1].cfg.grasp_approach_mode = "top_down"
        # PiPER's wrist limits cannot reproduce RoboTwin's source-arm tool
        # quaternion exactly across the full workspace. Keep the vertical
        # candidates and allow at most 12 degrees of IK orientation error.
        steps[-1].cfg.planner_position_threshold = 0.012
        steps[-1].cfg.planner_rotation_threshold = math.radians(12.0)
        # A square has two equivalent edge-aligned grasp axes. Both candidates
        # remain vertical and parallel to the tabletop.
        steps[-1].cfg.grasp_yaw_offsets_deg = (0.0, 90.0)
        # The local X/Y gripper axes remain horizontal while local Z points
        # straight down, so the gripper plane is parallel to the tabletop.
        steps[-1].cfg.top_down_approach_tilt_deg = 0.0
        steps[-1].cfg.lift_height = 0.10
        steps[-1].cfg.minimum_lift_delta = 0.05
        steps[-1].cfg.preplace_height = (0.08, 0.03)[index]
        steps[-1].cfg.preplace_tracking_tolerance = 0.20

    def retarget_top_block() -> None:
        bottom = _position(scene, names[0])
        steps[1].cfg.target_x = float(bottom[0])
        steps[1].cfg.target_y = float(bottom[1])
        steps[1].cfg.target_z = float(bottom[2]) + 0.05
        print(
            "[BENCHMARK]: bottom block settled; "
            f"top target=({steps[1].cfg.target_x:.3f}, "
            f"{steps[1].cfg.target_y:.3f}, {steps[1].cfg.target_z:.3f})",
            flush=True,
        )

    steps[0].after_success = retarget_top_block

    def success() -> bool:
        positions = sorted(
            (_position(scene, name) for name in names),
            key=lambda position: float(position[2]),
        )
        delta = positions[1] - positions[0]
        return (
            abs(float(delta[0])) < 0.025
            and abs(float(delta[1])) < 0.025
            and abs(float(delta[2]) - 0.05) <= 0.015
            and abs(float(positions[0][0]) - target_x) < 0.025
            and abs(float(positions[0][1]) - target_y) < 0.025
            and min(float(pos[2]) for pos in positions) >= tz + 0.020
        )

    def pose_sampler(rng: random.Random):
        local_poses = sample_blocks_stack_easy(rng)
        samples = {}
        for color, name in zip(("red", "black"), names, strict=True):
            pose = local_poses[color]
            samples[name] = (
                (tx + pose.x, ty - pose.y, tz + 0.025),
                (
                    math.cos(0.5 * pose.yaw),
                    0.0,
                    0.0,
                    math.sin(0.5 * pose.yaw),
                ),
                pose.yaw,
            )
        return samples

    return SequentialBenchmarkController(
        robot,
        joint_targets,
        scene,
        "stack_blocks_two",
        steps,
        final_success=success,
        post_wait=0.8,
        pose_sampler=pose_sampler,
    )


def _handover_controller(
    robot,
    joint_targets,
    scene,
    geom,
    *,
    strict_collision_check: bool = False,
    planner_enabled: bool = True,
):
    tx, ty, tz, tl, tw = geom
    name = "benchmark_handover_block"
    middle_z = tz + 0.28
    first_cfg = _cfg(
        arm="fl",
        table_x=tx,
        table_y=ty,
        table_height=tz,
        table_length=tl,
        table_width=tw,
        spawn_x=(0.15, 0.23),
        spawn_y=(0.04, 0.12),
        object_height=0.20,
        # Hand off on the rover centerline so neither arm crosses deeply into
        # the other arm's workspace.
        target_x=tx,
        target_y=ty + 0.10,
        target_z=middle_z,
        target_radius=0.05,
        closed_gripper=0.013,
        strict_collision_check=strict_collision_check,
    )
    second_cfg = _cfg(
        arm="fr",
        table_x=tx,
        table_y=ty,
        table_height=tz,
        table_length=tl,
        table_width=tw,
        spawn_x=(0.15, 0.23),
        spawn_y=(0.04, 0.12),
        object_height=0.20,
        target_x=tx - 0.20,
        target_y=ty + 0.04,
        target_z=tz + 0.10,
        target_radius=0.03,
        closed_gripper=0.011,
        strict_collision_check=strict_collision_check,
    )
    first_cfg.planner_enabled = planner_enabled
    second_cfg.planner_enabled = planner_enabled
    # The two grippers use vertically separated contact bands on the long block
    # so they can overlap in time without occupying the same end-effector pose.
    first_cfg.grasp_center_z_offset = 0.055
    second_cfg.grasp_center_z_offset = -0.055
    first_cfg.preplace_height = 0.10
    first_cfg.hold_object_at_target = True
    first_cfg.target_z_tolerance = 0.04
    second_cfg.target_z_tolerance = 0.04
    second_cfg.trajectory_tracking_tolerance = 0.11
    second_cfg.trajectory_tracking_timeout = 6.0
    second_cfg.grasp_pose_stabilization = True
    steps = [
        PickPlaceStep(name, "fl", first_cfg),
        PickPlaceStep(name, "fr", second_cfg),
    ]

    def success() -> bool:
        position = _position(scene, name)
        return (
            abs(float(position[0]) - second_cfg.target_x) < 0.03
            and abs(float(position[1]) - second_cfg.target_y) < 0.03
            and abs((float(position[2]) - 0.10) - (tz + 0.005)) < 0.01
        )

    return SequentialBenchmarkController(
        robot,
        joint_targets,
        scene,
        "handover_block",
        steps,
        final_success=success,
    )


def _scan_object_controller(
    robot,
    joint_targets,
    scene,
    geom,
    *,
    strict_collision_check: bool = False,
    planner_enabled: bool = True,
):
    """Reproduce RoboTwin's two-arm scanner and tea-box alignment task."""

    tx, ty, tz, tl, tw = geom
    # Keep the in-air scan pair inside the overlap of both PiPER workspaces.
    # The two object roots remain aligned with an 11 cm scanner-to-box gap.
    scanner_target = (tx - 0.02, ty + 0.10, tz + 0.07)
    object_target = (tx - 0.13, ty + 0.10, tz + 0.07)
    grasp_tilt = math.radians(8.0)
    annotated_topdown = (0.0, -math.sin(grasp_tilt), -math.cos(grasp_tilt))
    transfer_tilt = math.radians(30.0)
    transfer_approach = (
        0.0,
        -math.sin(transfer_tilt),
        -math.cos(transfer_tilt),
    )

    scanner_cfg = _cfg(
        arm="fl",
        table_x=tx,
        table_y=ty,
        table_height=tz,
        table_length=tl,
        table_width=tw,
        spawn_x=(0.15, 0.23),
        spawn_y=(0.04, 0.12),
        object_height=0.08,
        target_x=scanner_target[0],
        target_y=scanner_target[1],
        target_z=scanner_target[2],
        target_radius=0.04,
        closed_gripper=0.016,
        strict_collision_check=strict_collision_check,
    )
    scanner_cfg.grasp_center_offset_object = (0.0, 0.01159, 0.04867)
    scanner_cfg.fixed_grasp_approach_axis_w = annotated_topdown
    scanner_cfg.fixed_grasp_closing_axis_w = (1.0, 0.0, 0.0)
    scanner_cfg.pregrasp_distance = 0.08
    scanner_cfg.lift_along_grasp_approach = True
    scanner_cfg.post_grasp_approach_axis_w = transfer_approach
    scanner_cfg.post_grasp_closing_axis_w = (1.0, 0.0, 0.0)
    scanner_cfg.planner_enable_finetune_trajopt = False
    scanner_cfg.grasp_pose_stabilization = True
    scanner_cfg.compensate_grasp_offset_at_target = True
    scanner_cfg.hold_object_at_target = True
    scanner_cfg.direct_place_after_lift = True
    scanner_cfg.preplace_height = 0.06
    scanner_cfg.target_z_tolerance = 0.04
    scanner_cfg.trajectory_tracking_tolerance = 0.12
    scanner_cfg.trajectory_tracking_timeout = 6.0

    object_cfg = _cfg(
        arm="fr",
        table_x=tx,
        table_y=ty,
        table_height=tz,
        table_length=tl,
        table_width=tw,
        spawn_x=(-0.23, -0.15),
        spawn_y=(0.08, 0.13),
        object_height=0.072,
        target_x=object_target[0],
        target_y=object_target[1],
        target_z=object_target[2],
        target_radius=0.04,
        closed_gripper=0.012,
        strict_collision_check=strict_collision_check,
    )
    scanner_cfg.planner_enabled = planner_enabled
    object_cfg.planner_enabled = planner_enabled
    object_cfg.grasp_center_offset_object = (0.0, 0.0, 0.04545)
    object_cfg.grasp_approach_mode = "top_down"
    object_cfg.top_down_approach_tilt_deg = 8.0
    object_cfg.pregrasp_distance = 0.08
    object_cfg.lift_height = 0.07
    object_cfg.lift_vertical_in_world = True
    object_cfg.planner_enable_finetune_trajopt = False
    object_cfg.planner_rotation_threshold = 0.10
    object_cfg.grasp_pose_stabilization = True
    object_cfg.compensate_grasp_offset_at_target = True
    object_cfg.preserve_grasp_orientation_for_transfer = True
    object_cfg.hold_object_at_target = True
    object_cfg.direct_place_after_lift = True
    object_cfg.preplace_height = 0.06
    object_cfg.target_z_tolerance = 0.04
    object_cfg.trajectory_tracking_tolerance = 0.12
    object_cfg.trajectory_tracking_timeout = 6.0

    steps = [
        PickPlaceStep(
            "benchmark_scanner",
            "fl",
            scanner_cfg,
            spawn_quaternion=(1.0, 0.0, 0.0, 0.0),
            spawn_root_z_offset=0.003,
        ),
        PickPlaceStep(
            "benchmark_scan_object",
            "fr",
            object_cfg,
            spawn_quaternion=(1.0, 0.0, 0.0, 0.0),
            spawn_root_z_offset=0.003,
        ),
    ]

    fl_gripper_ids, _ = robot.find_joints(
        ["fl_joint[7-8]"], preserve_order=True
    )
    fr_gripper_ids, _ = robot.find_joints(
        ["fr_joint[7-8]"], preserve_order=True
    )

    def gripper_closed(joint_ids: list[int]) -> bool:
        opening = torch.max(torch.abs(robot.data.joint_pos[0, joint_ids]))
        return float(opening.item()) <= 0.022

    def success() -> bool:
        scanner = _position(scene, "benchmark_scanner")
        scan_object = _position(scene, "benchmark_scan_object")
        scanner_error = torch.linalg.vector_norm(
            scanner
            - torch.tensor(scanner_target, device=scanner.device)
        )
        object_error = torch.linalg.vector_norm(
            scan_object
            - torch.tensor(object_target, device=scan_object.device)
        )
        scanner_to_object = scanner - scan_object
        aligned = (
            0.045 <= float(scanner_to_object[0]) <= 0.12
            and abs(float(scanner_to_object[1])) <= 0.03
            and abs(float(scanner_to_object[2])) <= 0.03
        )
        return (
            float(scanner_error.item()) <= 0.05
            and float(object_error.item()) <= 0.05
            and aligned
            and gripper_closed(fl_gripper_ids)
            and gripper_closed(fr_gripper_ids)
        )

    return SequentialBenchmarkController(
        robot,
        joint_targets,
        scene,
        "scan_object",
        steps,
        final_success=success,
        post_wait=1.0,
    )


def _scan_rock_controller(
    robot,
    joint_targets,
    scene,
    geom,
    *,
    strict_collision_check: bool = False,
    planner_enabled: bool = True,
):
    """Pick the scanner and aim its scan ray at a scaled OmniLRS ground rock."""

    tx, ty, tz, tl, tw = geom
    scanner_functional_offset = (0.0, -0.0611464, 0.0)
    scanner_scan_axis = (0.0, -0.2741599977, -0.9616799951)
    rock_surface_offset = (0.0, 0.0, 0.030)
    scan_standoff = 0.085

    scanner_cfg = _cfg(
        arm="fl",
        table_x=tx,
        table_y=ty,
        table_height=tz,
        table_length=tl,
        table_width=tw,
        spawn_x=(0.16, 0.30),
        spawn_y=(0.10, 0.22),
        object_height=0.08,
        target_x=tx - 0.03,
        target_y=ty + 0.10,
        target_z=tz + 0.10,
        target_radius=0.035,
        closed_gripper=0.016,
        strict_collision_check=strict_collision_check,
    )
    scanner_cfg.planner_enabled = planner_enabled
    scanner_cfg.grasp_center_offset_object = (0.0, 0.01159, 0.04867)
    scanner_cfg.fixed_grasp_approach_axis_w = (
        0.0,
        -math.sin(math.radians(8.0)),
        -math.cos(math.radians(8.0)),
    )
    scanner_cfg.fixed_grasp_closing_axis_w = (1.0, 0.0, 0.0)
    scanner_cfg.pregrasp_distance = 0.08
    scanner_cfg.lift_height = 0.06
    scanner_cfg.lift_vertical_in_world = True
    scanner_cfg.planner_enable_finetune_trajopt = False
    scanner_cfg.planner_rotation_threshold = 0.16
    scanner_cfg.grasp_pose_stabilization = True
    scanner_cfg.target_functional_offset_object = scanner_functional_offset
    scanner_cfg.target_functional_axis_object = scanner_scan_axis
    scanner_cfg.target_functional_standoff = scan_standoff
    scanner_cfg.hold_object_at_target = True
    scanner_cfg.direct_place_after_lift = True
    scanner_cfg.target_z_tolerance = 0.035
    scanner_cfg.trajectory_tracking_tolerance = 0.12
    scanner_cfg.trajectory_tracking_timeout = 6.0

    step = PickPlaceStep(
        "benchmark_scanner",
        "fl",
        scanner_cfg,
        spawn_quaternion=(1.0, 0.0, 0.0, 0.0),
        spawn_root_z_offset=0.003,
    )
    def setup_episode(rng: random.Random) -> dict[str, tuple[float, float, float]]:
        scanner_position = scene["benchmark_scanner"].data.root_pos_w[0]
        separation = rng.uniform(0.14, 0.18)
        separation_angle = rng.uniform(
            math.radians(-145.0),
            math.radians(-115.0),
        )
        rock_position = (
            float(scanner_position[0]) + separation * math.cos(separation_angle),
            float(scanner_position[1]) + separation * math.sin(separation_angle),
            tz + 0.042,
        )
        rock_yaw = rng.uniform(-math.pi, math.pi)
        rock_quaternion = (
            math.cos(0.5 * rock_yaw),
            0.0,
            0.0,
            math.sin(0.5 * rock_yaw),
        )
        rock = scene["benchmark_scan_rock"]
        rock_state = rock.data.default_root_state.clone()
        rock_state[:, :3] = torch.tensor(
            rock_position, device=rock_state.device
        )
        rock_state[:, 3:7] = torch.tensor(
            rock_quaternion, device=rock_state.device
        )
        rock_state[:, 7:] = 0.0
        rock.write_root_state_to_sim(rock_state)
        scanner_cfg.target_functional_aim_point_w = (
            rock_position[0] + rock_surface_offset[0],
            rock_position[1] + rock_surface_offset[1],
            rock_position[2] + rock_surface_offset[2],
        )
        print(
            "[BENCHMARK]: scan-rock target "
            f"rock=({rock_position[0]:.3f}, {rock_position[1]:.3f}, "
            f"{rock_position[2]:.3f}), standoff={scan_standoff:.3f} m, "
            "scanner_axis=R_object(0.000, -0.274, -0.962)",
            flush=True,
        )
        return {"benchmark_scan_rock": rock_position}

    fl_gripper_ids, _ = robot.find_joints(
        ["fl_joint[7-8]"], preserve_order=True
    )

    def success() -> bool:
        if scanner_cfg.target_functional_aim_point_w is None:
            return False
        scanner = scene["benchmark_scanner"]
        rock = scene["benchmark_scan_rock"]
        functional_offset = torch.tensor(
            [scanner_functional_offset],
            device=scanner.data.root_pos_w.device,
        )
        scan_axis_local = torch.tensor(
            [scanner_scan_axis],
            device=scanner.data.root_pos_w.device,
        )
        functional_point = (
            scanner.data.root_pos_w
            + math_utils.quat_apply(scanner.data.root_quat_w, functional_offset)
        )[0]
        scan_axis_w = math_utils.quat_apply(
            scanner.data.root_quat_w, scan_axis_local
        )[0]
        scan_axis_w = torch.nn.functional.normalize(scan_axis_w, dim=0)
        rock_target = rock.data.root_pos_w[0] + torch.tensor(
            rock_surface_offset, device=rock.data.root_pos_w.device
        )
        scanner_to_rock = rock_target - functional_point
        distance = float(torch.linalg.vector_norm(scanner_to_rock).item())
        if distance <= 1e-5:
            return False
        direction = scanner_to_rock / distance
        alignment = float(torch.dot(direction, scan_axis_w).item())
        along = float(torch.dot(scanner_to_rock, scan_axis_w).item())
        lateral = float(
            torch.linalg.vector_norm(
                scanner_to_rock - along * scan_axis_w
            ).item()
        )
        opening = torch.max(
            torch.abs(robot.data.joint_pos[0, fl_gripper_ids])
        )
        return (
            0.060 <= along <= 0.110
            and lateral <= 0.032
            and alignment >= math.cos(math.radians(20.0))
            and float(opening.item()) <= 0.022
        )

    return SequentialBenchmarkController(
        robot,
        joint_targets,
        scene,
        "scan_rock",
        [step],
        final_success=success,
        post_wait=1.5,
        episode_setup=setup_episode,
        hold_active_object_until_grasp=True,
    )


def _dump_controller(
    robot, joint_targets, scene, geom, *, strict_collision_check: bool = False
):
    tx, ty, tz, tl, tw = geom
    big_x, big_y = tx - 0.62, ty + 0.02
    cfg = _cfg(
        arm="fl",
        table_x=tx,
        table_y=ty,
        table_height=tz,
        table_length=tl,
        table_width=tw,
        spawn_x=(0.14, 0.22),
        spawn_y=(0.04, 0.12),
        object_height=0.13,
        target_x=big_x,
        target_y=big_y,
        target_z=tz + 0.18,
        target_radius=0.18,
        closed_gripper=0.020,
        strict_collision_check=strict_collision_check,
    )
    cfg.grasp_center_z_offset = 0.045
    cfg.target_z_tolerance = 0.16

    def dump_contents() -> None:
        rng = random.Random(0)
        for index in range(5):
            _move_rigid_object(
                scene,
                f"benchmark_garbage_{index}",
                (
                    big_x + rng.uniform(-0.055, 0.055),
                    big_y + rng.uniform(-0.045, 0.045),
                    0.20 + 0.03 * index,
                ),
            )

    step = PickPlaceStep(
        "benchmark_small_bin",
        "fl",
        cfg,
        after_success=dump_contents,
    )

    def success() -> bool:
        for index in range(5):
            pos = _position(scene, f"benchmark_garbage_{index}")
            if (
                abs(float(pos[0]) - big_x) > 0.09
                or abs(float(pos[1]) - big_y) > 0.08
                or not (0.0 <= float(pos[2]) <= 0.72)
            ):
                return False
        return True

    return SequentialBenchmarkController(
        robot,
        joint_targets,
        scene,
        "dump_bin_bigbin",
        [step],
        final_success=success,
        post_wait=1.4,
    )


def _cabinet_controller(
    robot, joint_targets, scene, geom, *, strict_collision_check: bool = False
):
    tx, ty, tz, tl, tw = geom
    planning_cfg = _cfg(
        arm="fr",
        table_x=tx,
        table_y=ty,
        table_height=tz,
        table_length=tl,
        table_width=tw,
        spawn_x=(0.0, 0.0),
        spawn_y=(0.0, 0.0),
        object_height=0.05,
        target_x=tx,
        target_y=ty,
        target_z=tz,
        target_radius=0.04,
        closed_gripper=0.006,
        strict_collision_check=strict_collision_check,
    )
    planning_cfg.planner_position_threshold = 0.035
    planning_cfg.planner_rotation_threshold = 0.40
    planning_cfg.trajectory_tracking_tolerance = 0.35
    planning_cfg.trajectory_tracking_timeout = 7.0
    object_cfg = _cfg(
        arm="fl",
        table_x=tx,
        table_y=ty,
        table_height=tz,
        table_length=tl,
        table_width=tw,
        spawn_x=(0.18, 0.24),
        spawn_y=(0.20, 0.26),
        object_height=0.06,
        target_x=tx,
        target_y=ty + 0.22,
        target_z=tz + 0.13,
        target_radius=0.15,
        closed_gripper=0.012,
        strict_collision_check=strict_collision_check,
    )
    object_cfg.target_z_tolerance = 0.08
    object_cfg.target_max_linear_speed = 0.20
    object_cfg.success_target_x = tx
    object_cfg.success_target_y = ty + 0.09
    object_cfg.grasp_approach_mode = "top_down"
    object_cfg.grasp_yaw_offsets_deg = (
        0.0,
        -45.0,
        45.0,
        -90.0,
        90.0,
        180.0,
    )
    # PiPER's wrist reaches the drawer volume more reliably with the same
    # 30-degree top-down approach used by the validated block-stacking task.
    object_cfg.top_down_approach_tilt_deg = 30.0
    object_cfg.planner_table_z_offset = -0.012
    object_cfg.grasp_pose_stabilization = True
    object_cfg.compensate_grasp_offset_at_target = True
    object_cfg.direct_place_after_lift = True
    object_cfg.trajectory_tracking_tolerance = 0.25
    object_cfg.preplace_tracking_tolerance = 0.35
    object_cfg.trajectory_tracking_timeout = 7.0
    object_cfg.trajectory_time_scale = 1.75
    object_cfg.overall_timeout = 60.0
    return CuroboArticulatedMechanismController(
        robot,
        joint_targets,
        scene,
        "put_object_cabinet",
        mechanism_name="benchmark_cabinet",
        contact_offset=(-0.123507, -0.000100, -0.090338),
        joint_fraction=0.72,
        planning_cfg=planning_cfg,
        object_cfg=object_cfg,
    )


def _switch_controller(
    robot,
    joint_targets,
    scene,
    geom,
    switch_variant: str,
    *,
    strict_collision_check: bool = False,
):
    tx, ty, tz, tl, tw = geom
    variant = load_switch_variant(switch_variant)
    contact_rotation = torch.tensor(
        variant.contact_rotation,
        device=robot.device,
        dtype=torch.float32,
    ).unsqueeze(0)
    contact_quaternion_local = tuple(
        float(value)
        for value in math_utils.quat_from_matrix(contact_rotation)[0].tolist()
    )
    planning_cfg = _cfg(
        # RoboTwin's original right-arm action maps to the PiPER arm mounted
        # on world +X after this rover payload's -90 degree mounting yaw.
        arm="fl",
        table_x=tx,
        table_y=ty,
        table_height=tz,
        table_length=tl,
        table_width=tw,
        spawn_x=(0.18, 0.18),
        spawn_y=(0.0, 0.0),
        object_height=0.05,
        target_x=tx,
        target_y=ty,
        target_z=tz,
        target_radius=0.04,
        closed_gripper=0.006,
        strict_collision_check=strict_collision_check,
    )
    planning_cfg.planner_position_threshold = 0.065
    planning_cfg.planner_rotation_threshold = 0.70
    planning_cfg.max_execution_base_translation = 0.06
    planning_cfg.trajectory_tracking_tolerance = 0.25
    planning_cfg.trajectory_tracking_timeout = 7.0
    planning_cfg.trajectory_time_scale = 1.5
    return CuroboArticulatedMechanismController(
        robot,
        joint_targets,
        scene,
        "turn_switch",
        mechanism_name="benchmark_switch",
        # Use the selected model's original RoboTwin contact annotation.
        contact_offset=variant.contact_offset,
        joint_fraction=0.91,
        planning_cfg=planning_cfg,
        physical_contact_only=True,
        # RoboTwin contact matrix remapped so PiPER's local-Z approach axis
        # matches RoboTwin's local-X grasp approach axis.
        contact_quaternion_local=contact_quaternion_local,
        # Preserve the annotation as the first choice, then relax wrist roll
        # in small increments only when cuRobo cannot optimize that pose.
        contact_rotation_offsets_deg=(
            0.0,
            -15.0,
            15.0,
            -30.0,
            30.0,
            -45.0,
            45.0,
        ),
        precontact_standoff=0.04,
        contact_advance=variant.contact_advance,
        planning_max_attempts=12,
        sample_metadata={
            "switch_model": variant.object_id,
            "robotwin_base": SWITCH_VARIANT_IDS.index(variant.object_id),
        },
        direct_precontact_goalset=False,
        planning_finetune=False,
    )


def create_benchmark_controller(
    robot,
    joint_targets: torch.Tensor,
    scene,
    task_name: str,
    *,
    table_x: float,
    table_y: float,
    table_height: float,
    table_length: float,
    table_width: float,
    switch_variant: str = "100880",
    strict_collision_check: bool = False,
    planner_enabled: bool = True,
):
    geometry = (table_x, table_y, table_height, table_length, table_width)
    builders = {
        "stack_blocks_two": _stack_two_controller,
        "handover_block": _handover_controller,
        "scan_object": _scan_object_controller,
        "scan_rock": _scan_rock_controller,
        "dump_bin_bigbin": _dump_controller,
        "put_object_cabinet": _cabinet_controller,
        "turn_switch": _switch_controller,
    }
    if task_name not in builders:
        raise ValueError(f"Unsupported RoboTwin benchmark task: {task_name}")
    if task_name in {"stack_blocks_two", "handover_block", "scan_object", "scan_rock"}:
        return builders[task_name](
            robot,
            joint_targets,
            scene,
            geometry,
            strict_collision_check=strict_collision_check,
            planner_enabled=planner_enabled,
        )
    if task_name == "turn_switch":
        return builders[task_name](
            robot,
            joint_targets,
            scene,
            geometry,
            switch_variant,
            strict_collision_check=strict_collision_check,
        )
    return builders[task_name](
        robot,
        joint_targets,
        scene,
        geometry,
        strict_collision_check=strict_collision_check,
    )
