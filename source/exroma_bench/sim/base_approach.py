"""Closed-loop rover approach used before tabletop manipulation."""

from __future__ import annotations

import math

import torch


class BasePoseApproachController:
    def __init__(
        self,
        rover,
        *,
        goal_x: float,
        goal_y: float,
        goal_yaw: float = 0.0,
        forward_yaw_offset: float = -0.5 * math.pi,
        position_tolerance: float = 0.015,
        yaw_tolerance: float = math.radians(2.0),
        linear_command: float = 0.65,
        turn_command: float = 0.65,
        timeout: float = 60.0,
    ) -> None:
        self.rover = rover
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_yaw = goal_yaw
        self.forward_yaw_offset = forward_yaw_offset
        self.position_tolerance = position_tolerance
        self.yaw_tolerance = yaw_tolerance
        self.linear_command = linear_command
        self.turn_command = turn_command
        self.timeout = timeout
        self.state = "idle"
        self.elapsed = 0.0
        self.stable_steps = 0
        self.failure_reason = ""
        self.position_reached = False

    @staticmethod
    def _wrap(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _pose(self) -> tuple[float, float, float]:
        position = self.rover.data.root_pos_w[0]
        w, x, y, z = (float(value) for value in self.rover.data.root_quat_w[0])
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return float(position[0]), float(position[1]), yaw

    def _errors(self) -> tuple[float, float, float]:
        x, y, yaw = self._pose()
        dx, dy = self.goal_x - x, self.goal_y - y
        distance = math.hypot(dx, dy)
        desired = (
            math.atan2(dy, dx)
            if distance > self.position_tolerance
            else self.goal_yaw + self.forward_yaw_offset
        )
        heading = self._wrap(desired - (yaw + self.forward_yaw_offset))
        final_yaw = self._wrap(self.goal_yaw - yaw)
        return distance, heading, final_yaw

    @property
    def is_active(self) -> bool:
        return self.state == "approach"

    @property
    def is_failed(self) -> bool:
        return self.state == "failed"

    def reset(self) -> None:
        self.state = "approach"
        self.elapsed = 0.0
        self.stable_steps = 0
        self.failure_reason = ""
        self.position_reached = False

    def base_command(self) -> tuple[float, float]:
        if not self.is_active:
            return 0.0, 0.0
        distance, heading, final_yaw = self._errors()
        if self.position_reached and distance > 0.03:
            self.position_reached = False
        if distance <= self.position_tolerance:
            self.position_reached = True
        if not self.position_reached:
            direction = 1.0
            steering_heading = heading
            if abs(heading) > 0.5 * math.pi:
                direction = -1.0
                steering_heading = self._wrap(
                    heading - math.copysign(math.pi, heading)
                )
            angular = max(
                -self.turn_command,
                min(self.turn_command, 1.4 * steering_heading),
            )
            if abs(steering_heading) > math.radians(50.0):
                return 0.0, angular
            linear = min(self.linear_command, max(0.50, 1.5 * distance))
            return direction * linear * max(0.0, math.cos(steering_heading)), angular
        yaw_rate = float(self.rover.data.root_link_vel_w[0, 5])
        angular = max(
            -self.turn_command,
            min(self.turn_command, 1.8 * final_yaw - 0.8 * yaw_rate),
        )
        if abs(final_yaw) > self.yaw_tolerance and abs(angular) < 0.03:
            angular = math.copysign(0.03, final_yaw)
        return 0.0, angular

    def update(self, dt: float) -> None:
        if not self.is_active:
            return
        self.elapsed += dt
        distance, _, final_yaw = self._errors()
        velocity = self.rover.data.root_link_vel_w[0]
        stable = (
            self.position_reached
            and distance <= 0.025
            and abs(final_yaw) <= self.yaw_tolerance
            and float(torch.linalg.vector_norm(velocity[:3])) <= 0.04
            and float(torch.linalg.vector_norm(velocity[3:])) <= 0.08
        )
        self.stable_steps = self.stable_steps + 1 if stable else 0
        if self.stable_steps >= 10:
            self.state = "done"
            print("[EXROMA][BASE]: reached manipulation pose", flush=True)
        elif self.elapsed >= self.timeout:
            self.failure_reason = (
                f"base approach timeout: distance={distance:.3f} m, "
                f"yaw_error={math.degrees(final_yaw):.1f} deg"
            )
            self.state = "failed"
