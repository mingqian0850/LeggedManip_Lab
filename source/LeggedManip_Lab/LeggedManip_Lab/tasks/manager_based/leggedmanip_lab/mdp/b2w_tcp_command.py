# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""World-frame TCP references derived from reachable B2W + Z1 configurations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class FKReachableWorldPoseCommand(CommandTerm):
    """Generate immutable world TCP goals without a hand-written arm workspace box.

    The arm starts in a valid joint configuration.  Its FK pose relative to the
    root is measured after every reset.  Each episode then samples a planar *ghost
    root* displacement and composes that transform with the valid TCP pose.
    Consequently every final target has a known whole-body solution: move the root
    to the ghost pose and return the arm to its sampled reset configuration.

    By default the final pose remains immutable for the episode.  A task may
    explicitly re-anchor it during a physical settling window; it is frozen
    before motion begins. ``command`` exposes a minimum-jerk reference from the
    settled TCP pose to that final pose so the arm is never hit with a
    discontinuous Cartesian target.
    """

    cfg: FKReachableWorldPoseCommandCfg

    def __init__(self, cfg: FKReachableWorldPoseCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        body_ids, body_names = self.robot.find_bodies(cfg.body_name)
        if body_names != [cfg.body_name]:
            raise ValueError(f"Expected exactly one TCP body named '{cfg.body_name}', found {body_names}.")
        self.body_idx = body_ids[0]

        self.start_pose_w = torch.zeros((self.num_envs, 7), device=self.device)
        self.goal_pose_w = torch.zeros_like(self.start_pose_w)
        self.pose_command_w = torch.zeros_like(self.start_pose_w)
        self.twist_command_w = torch.zeros((self.num_envs, 6), device=self.device)
        self.command_buffer = torch.zeros((self.num_envs, 13), device=self.device)
        self.ghost_root_pose_w = torch.zeros_like(self.start_pose_w)
        self.sampled_tcp_pose_b = torch.zeros_like(self.start_pose_w)
        self.ghost_translation_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.ghost_rotation_b = torch.zeros((self.num_envs, 4), device=self.device)
        self.spatial_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.start_pose_w[:, 3] = 1.0
        self.goal_pose_w[:, 3] = 1.0
        self.pose_command_w[:, 3] = 1.0
        self.ghost_root_pose_w[:, 3] = 1.0
        self.sampled_tcp_pose_b[:, 3] = 1.0
        self.ghost_rotation_b[:, 0] = 1.0
        self.command_buffer[:, :7] = self.pose_command_w
        self.elapsed_s = torch.zeros(self.num_envs, device=self.device)
        self.motion_progress = torch.zeros(self.num_envs, device=self.device)

        self.metrics["reference_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["goal_orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "FKReachableWorldPoseCommand:\n"
            f"\tCommand dimension: {self.command.shape[1]}\n"
            f"\tRadius range: {self.cfg.radius_range}\n"
            f"\tMotion timing: settle={self.cfg.settle_time_s}s, move={self.cfg.motion_time_s}s"
        )

    @property
    def command(self) -> torch.Tensor:
        """World reference as ``(position, quaternion_wxyz, linear/angular twist)``."""
        return self.command_buffer

    @property
    def final_command(self) -> torch.Tensor:
        """World TCP goal, frozen after an optional settling re-anchor."""
        return self.goal_pose_w

    def reference_pose_at_offsets(self, offsets_s: Sequence[float]) -> torch.Tensor:
        """Evaluate future minimum-jerk poses without advancing the command.

        Args:
            offsets_s: Non-negative time offsets relative to the current policy
                step.  The returned tensor has shape ``(num_envs, K, 7)`` and
                stores position plus scalar-first quaternion for each offset.

        This is the task-space trajectory preview consumed by the unified
        whole-body policy.  It is computed from the same immutable episode goal
        as :attr:`command`, so the actor never receives a privileged base pose.
        """
        if len(offsets_s) == 0:
            raise ValueError("offsets_s must contain at least one preview time")
        if any(offset < 0.0 for offset in offsets_s):
            raise ValueError("trajectory preview offsets must be non-negative")
        offsets = torch.as_tensor(offsets_s, device=self.device, dtype=self.elapsed_s.dtype)

        elapsed = self.elapsed_s.unsqueeze(-1) + offsets.unsqueeze(0)
        linear_progress = torch.clamp(
            (elapsed - self.cfg.settle_time_s) / self.cfg.motion_time_s,
            min=0.0,
            max=1.0,
        )
        progress = linear_progress**3 * (10.0 - 15.0 * linear_progress + 6.0 * linear_progress**2)

        poses = torch.zeros((self.num_envs, offsets.numel(), 7), device=self.device)
        poses[..., :3] = self.start_pose_w[:, None, :3] + progress[..., None] * (
            self.goal_pose_w[:, None, :3] - self.start_pose_w[:, None, :3]
        )

        start_quat = self.start_pose_w[:, 3:]
        _, total_rotation = compute_pose_error(
            self.start_pose_w[:, :3],
            start_quat,
            self.goal_pose_w[:, :3],
            self.goal_pose_w[:, 3:],
            rot_error_type="axis_angle",
        )
        total_angle = torch.linalg.vector_norm(total_rotation, dim=-1)
        rotation_axis = total_rotation / torch.clamp(total_angle.unsqueeze(-1), min=1.0e-8)
        preview_angle = progress * total_angle.unsqueeze(-1)
        preview_axis = rotation_axis[:, None, :].expand(-1, offsets.numel(), -1)
        rotation_delta = quat_from_angle_axis(
            preview_angle.reshape(-1),
            preview_axis.reshape(-1, 3),
        ).reshape(self.num_envs, offsets.numel(), 4)
        poses[..., 3:] = quat_mul(
            rotation_delta.reshape(-1, 4),
            start_quat[:, None, :].expand(-1, offsets.numel(), -1).reshape(-1, 4),
        ).reshape(self.num_envs, offsets.numel(), 4)
        return poses

    def _update_metrics(self) -> None:
        tcp_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        tcp_quat_w = self.robot.data.body_quat_w[:, self.body_idx]
        ref_pos_error, ref_rot_error = compute_pose_error(
            tcp_pos_w,
            tcp_quat_w,
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            rot_error_type="axis_angle",
        )
        goal_pos_error, goal_rot_error = compute_pose_error(
            tcp_pos_w,
            tcp_quat_w,
            self.goal_pose_w[:, :3],
            self.goal_pose_w[:, 3:],
            rot_error_type="axis_angle",
        )
        self.metrics["reference_position_error"] = torch.linalg.vector_norm(ref_pos_error, dim=-1)
        self.metrics["reference_orientation_error"] = torch.linalg.vector_norm(ref_rot_error, dim=-1)
        self.metrics["goal_position_error"] = torch.linalg.vector_norm(goal_pos_error, dim=-1)
        self.metrics["goal_orientation_error"] = torch.linalg.vector_norm(goal_rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        count = len(env_ids)
        if count == 0:
            return

        root_pos_w = self.robot.data.root_pos_w[env_ids]
        root_quat_w = self.robot.data.root_quat_w[env_ids]
        # Reset events write root/joint state before the command manager is
        # reset. Accessing body pose here triggers PhysX forward kinematics from
        # that freshly written state, instead of reusing the pre-reset USD pose.
        start_pos_w = self.robot.data.body_pos_w[env_ids, self.body_idx]
        start_quat_w = self.robot.data.body_quat_w[env_ids, self.body_idx]
        sampled_pos_b, sampled_quat_b = subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            start_pos_w,
            start_quat_w,
        )

        radius = torch.empty(count, device=self.device).uniform_(*self.cfg.radius_range)
        if self.cfg.short_radius_probability > 0.0:
            short_mask = torch.rand(count, device=self.device) < self.cfg.short_radius_probability
            short_radius = torch.empty(count, device=self.device).uniform_(*self.cfg.short_radius_range)
            radius = torch.where(short_mask, short_radius, radius)
        bearing = torch.empty(count, device=self.device).uniform_(*self.cfg.bearing_range)
        yaw = torch.empty(count, device=self.device).uniform_(*self.cfg.yaw_range)
        height_offset = torch.zeros(count, device=self.device)
        pitch = torch.zeros(count, device=self.device)
        spatial_mask = torch.zeros(count, dtype=torch.bool, device=self.device)
        if self.cfg.spatial_probability > 0.0:
            height_offset.uniform_(*self.cfg.height_offset_range)
            pitch.uniform_(*self.cfg.pitch_range)
            spatial_mask = torch.rand(count, device=self.device) < self.cfg.spatial_probability
            height_offset = torch.where(spatial_mask, height_offset, torch.zeros_like(height_offset))
            pitch = torch.where(spatial_mask, pitch, torch.zeros_like(pitch))
            if self.cfg.spatial_bearing_range is not None:
                spatial_bearing = torch.empty(count, device=self.device).uniform_(*self.cfg.spatial_bearing_range)
                bearing = torch.where(spatial_mask, spatial_bearing, bearing)
        if self.cfg.stationary_probability > 0.0:
            stationary = torch.rand(count, device=self.device) < self.cfg.stationary_probability
            radius[stationary] = 0.0
            yaw[stationary] = 0.0
            height_offset[stationary] = 0.0
            pitch[stationary] = 0.0
            spatial_mask[stationary] = False

        ghost_translation_b = torch.zeros((count, 3), device=self.device)
        ghost_translation_b[:, 0] = radius * torch.cos(bearing)
        ghost_translation_b[:, 1] = radius * torch.sin(bearing)
        ghost_translation_b[:, 2] = height_offset
        zeros = torch.zeros(count, device=self.device)
        ghost_rotation = quat_from_euler_xyz(zeros, pitch, yaw)
        ghost_root_pos_w, ghost_root_quat_w = combine_frame_transforms(
            root_pos_w,
            root_quat_w,
            ghost_translation_b,
            ghost_rotation,
        )
        goal_pos_w, goal_quat_w = combine_frame_transforms(
            ghost_root_pos_w,
            ghost_root_quat_w,
            sampled_pos_b,
            sampled_quat_b,
        )
        # Preserve the proven planar command geometry bit-for-bit.  Only
        # explicitly selected spatial samples receive a world-vertical TCP
        # displacement, preventing root tilt from reversing a requested low
        # target while avoiding a distribution shift for planar replay.
        goal_pos_w[spatial_mask, 2] = start_pos_w[spatial_mask, 2] + height_offset[spatial_mask]

        self.start_pose_w[env_ids, :3] = start_pos_w
        self.start_pose_w[env_ids, 3:] = start_quat_w
        self.goal_pose_w[env_ids, :3] = goal_pos_w
        self.goal_pose_w[env_ids, 3:] = goal_quat_w
        self.pose_command_w[env_ids] = self.start_pose_w[env_ids]
        self.twist_command_w[env_ids] = 0.0
        self.command_buffer[env_ids, :7] = self.pose_command_w[env_ids]
        self.command_buffer[env_ids, 7:] = 0.0
        self.ghost_root_pose_w[env_ids, :3] = ghost_root_pos_w
        self.ghost_root_pose_w[env_ids, 3:] = ghost_root_quat_w
        self.sampled_tcp_pose_b[env_ids, :3] = sampled_pos_b
        self.sampled_tcp_pose_b[env_ids, 3:] = sampled_quat_b
        self.ghost_translation_b[env_ids] = ghost_translation_b
        self.ghost_rotation_b[env_ids] = ghost_rotation
        self.spatial_mask[env_ids] = spatial_mask
        self.elapsed_s[env_ids] = 0.0
        self.motion_progress[env_ids] = 0.0

    def _recapture_settling_reference(self, env_ids: torch.Tensor) -> None:
        """Anchor the trajectory after the robot has physically settled."""
        if env_ids.numel() == 0:
            return
        root_pos_w = self.robot.data.root_pos_w[env_ids]
        root_quat_w = self.robot.data.root_quat_w[env_ids]
        start_pos_w = self.robot.data.body_pos_w[env_ids, self.body_idx]
        start_quat_w = self.robot.data.body_quat_w[env_ids, self.body_idx]
        sampled_pos_b, sampled_quat_b = subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            start_pos_w,
            start_quat_w,
        )
        ghost_root_pos_w, ghost_root_quat_w = combine_frame_transforms(
            root_pos_w,
            root_quat_w,
            self.ghost_translation_b[env_ids],
            self.ghost_rotation_b[env_ids],
        )
        goal_pos_w, goal_quat_w = combine_frame_transforms(
            ghost_root_pos_w,
            ghost_root_quat_w,
            sampled_pos_b,
            sampled_quat_b,
        )
        spatial_mask = self.spatial_mask[env_ids]
        goal_pos_w[spatial_mask, 2] = (
            start_pos_w[spatial_mask, 2] + self.ghost_translation_b[env_ids[spatial_mask], 2]
        )
        self.start_pose_w[env_ids, :3] = start_pos_w
        self.start_pose_w[env_ids, 3:] = start_quat_w
        self.goal_pose_w[env_ids, :3] = goal_pos_w
        self.goal_pose_w[env_ids, 3:] = goal_quat_w
        self.ghost_root_pose_w[env_ids, :3] = ghost_root_pos_w
        self.ghost_root_pose_w[env_ids, 3:] = ghost_root_quat_w
        self.sampled_tcp_pose_b[env_ids, :3] = sampled_pos_b
        self.sampled_tcp_pose_b[env_ids, 3:] = sampled_quat_b

    def _update_command(self) -> None:
        self.elapsed_s += self._env.step_dt
        if self.cfg.recapture_during_settle:
            settling_env_ids = torch.nonzero(
                self.elapsed_s <= self.cfg.settle_time_s,
                as_tuple=False,
            ).squeeze(-1)
            self._recapture_settling_reference(settling_env_ids)
        linear_progress = torch.clamp(
            (self.elapsed_s - self.cfg.settle_time_s) / self.cfg.motion_time_s,
            min=0.0,
            max=1.0,
        )
        # Quintic minimum-jerk time scaling: zero velocity and acceleration at
        # both ends of the reference motion.
        progress = linear_progress**3 * (10.0 - 15.0 * linear_progress + 6.0 * linear_progress**2)
        progress_rate = (
            30.0 * linear_progress**2 - 60.0 * linear_progress**3 + 30.0 * linear_progress**4
        ) / self.cfg.motion_time_s
        self.motion_progress.copy_(progress)
        blend = progress.unsqueeze(-1)
        self.pose_command_w[:, :3] = self.start_pose_w[:, :3] + blend * (
            self.goal_pose_w[:, :3] - self.start_pose_w[:, :3]
        )

        start_quat = self.start_pose_w[:, 3:]
        _, total_rotation = compute_pose_error(
            self.start_pose_w[:, :3],
            start_quat,
            self.goal_pose_w[:, :3],
            self.goal_pose_w[:, 3:],
            rot_error_type="axis_angle",
        )
        total_angle = torch.linalg.vector_norm(total_rotation, dim=-1)
        rotation_axis = total_rotation / torch.clamp(total_angle.unsqueeze(-1), min=1.0e-8)
        rotation_delta = quat_from_angle_axis(progress * total_angle, rotation_axis)
        self.pose_command_w[:, 3:] = quat_mul(rotation_delta, start_quat)
        self.twist_command_w[:, :3] = progress_rate.unsqueeze(-1) * (self.goal_pose_w[:, :3] - self.start_pose_w[:, :3])
        self.twist_command_w[:, 3:] = progress_rate.unsqueeze(-1) * total_rotation
        self.command_buffer[:, :7] = self.pose_command_w
        self.command_buffer[:, 7:] = self.twist_command_w


@configclass
class FKReachableWorldPoseCommandCfg(CommandTermCfg):
    """Configuration for :class:`FKReachableWorldPoseCommand`."""

    class_type: type = FKReachableWorldPoseCommand
    asset_name: str = MISSING
    body_name: str = MISSING
    radius_range: tuple[float, float] = (0.05, 0.20)
    short_radius_range: tuple[float, float] | None = None
    """Optional replay range used to prevent forgetting earlier short targets."""
    short_radius_probability: float = 0.0
    """Probability of replacing a radius sample with ``short_radius_range``."""
    bearing_range: tuple[float, float] = (-math.pi, math.pi)
    spatial_bearing_range: tuple[float, float] | None = None
    """Optional bearing range used only for height/pitch samples."""
    yaw_range: tuple[float, float] = (-0.25, 0.25)
    height_offset_range: tuple[float, float] = (0.0, 0.0)
    """Ghost-root vertical displacement in the settled root frame."""
    pitch_range: tuple[float, float] = (0.0, 0.0)
    """Ghost-root pitch displacement in radians; positive pitches the nose down."""
    spatial_probability: float = 0.0
    """Fraction of samples using height/pitch; remaining samples stay planar."""
    stationary_probability: float = 0.1
    settle_time_s: float = 1.0
    motion_time_s: float = 2.0
    recapture_during_settle: bool = False
    """Continuously re-anchor start/goal during settle, then freeze them."""

    def __post_init__(self) -> None:
        if self.radius_range[0] < 0.0 or self.radius_range[1] < self.radius_range[0]:
            raise ValueError(f"Invalid radius range: {self.radius_range}")
        if not 0.0 <= self.short_radius_probability <= 1.0:
            raise ValueError("short_radius_probability must lie in [0, 1]")
        if self.short_radius_probability > 0.0:
            if self.short_radius_range is None:
                raise ValueError("short_radius_range is required when short-radius replay is enabled")
            if self.short_radius_range[0] < 0.0 or self.short_radius_range[1] < self.short_radius_range[0]:
                raise ValueError(f"Invalid short radius range: {self.short_radius_range}")
        if not 0.0 <= self.stationary_probability <= 1.0:
            raise ValueError("stationary_probability must lie in [0, 1]")
        if self.height_offset_range[1] < self.height_offset_range[0]:
            raise ValueError(f"Invalid height offset range: {self.height_offset_range}")
        if self.pitch_range[1] < self.pitch_range[0]:
            raise ValueError(f"Invalid pitch range: {self.pitch_range}")
        if self.spatial_bearing_range is not None and self.spatial_bearing_range[1] < self.spatial_bearing_range[0]:
            raise ValueError(f"Invalid spatial bearing range: {self.spatial_bearing_range}")
        if not 0.0 <= self.spatial_probability <= 1.0:
            raise ValueError("spatial_probability must lie in [0, 1]")
        if self.motion_time_s <= 0.0:
            raise ValueError("motion_time_s must be positive")


class PeriodicWorldPoseCommand(FKReachableWorldPoseCommand):
    """Continuous deterministic 6D trajectories for EE-WBC training and evaluation.

    Each environment is assigned one trajectory family by its environment index.
    The reference starts at the physically settled TCP pose and is introduced with
    a quintic ramp, so a benchmark never injects a Cartesian step.  The public
    buffers intentionally match :class:`FKReachableWorldPoseCommand`; existing
    policies therefore keep exactly the same 216 observations.
    """

    cfg: PeriodicWorldPoseCommandCfg
    _SUPPORTED_TYPES = (
        "waypoint_planar",
        "waypoint_low",
        "hold",
        "line_x",
        "line_y",
        "circle_xy",
        "figure8_xy",
        "vertical",
        "yaw_scan",
        "six_d",
    )

    def __init__(self, cfg: PeriodicWorldPoseCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.trajectory_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.frequency_hz = torch.zeros(self.num_envs, device=self.device)
        self.reference_root_pose_w = torch.zeros_like(self.start_pose_w)
        self.reference_root_pose_w[:, 3] = 1.0
        self.reference_frame_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
        self.reference_frame_quat_w[:, 0] = 1.0
        self.reference_center_offset_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.reference_orientation_offset_rpy = torch.zeros((self.num_envs, 3), device=self.device)
        self.is_periodic_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def trajectory_names(self) -> tuple[str, ...]:
        return self.cfg.trajectory_types

    def __str__(self) -> str:
        return (
            "PeriodicWorldPoseCommand:\n"
            f"\tTrajectory types: {self.cfg.trajectory_types}\n"
            f"\tFrequency range: {self.cfg.frequency_range_hz} Hz\n"
            f"\tTiming: settle={self.cfg.settle_time_s}s, ramp={self.cfg.ramp_time_s}s"
        )

    @staticmethod
    def _minimum_jerk(linear_progress: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        progress = linear_progress**3 * (10.0 - 15.0 * linear_progress + 6.0 * linear_progress**2)
        derivative = 30.0 * linear_progress**2 - 60.0 * linear_progress**3 + 30.0 * linear_progress**4
        return progress, derivative

    def _evaluate_reference(
        self, elapsed_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pose, world twist, and ramp progress for ``(N, K)`` times."""
        if elapsed_s.ndim == 1:
            elapsed_s = elapsed_s.unsqueeze(-1)
        count, preview_count = elapsed_s.shape
        active_s = torch.clamp(elapsed_s - self.cfg.settle_time_s, min=0.0)
        linear_ramp = torch.clamp(active_s / self.cfg.ramp_time_s, min=0.0, max=1.0)
        ramp, ramp_shape_rate = self._minimum_jerk(linear_ramp)
        ramp_rate = ramp_shape_rate / self.cfg.ramp_time_s
        omega = (2.0 * math.pi * self.frequency_hz[:count]).unsqueeze(-1)
        phase = omega * active_s
        sin_phase = torch.sin(phase)
        cos_phase = torch.cos(phase)

        offset_b = torch.zeros((count, preview_count, 3), device=self.device)
        offset_rate_b = torch.zeros_like(offset_b)
        rpy = self.reference_orientation_offset_rpy[:count, None, :].expand(
            -1, preview_count, -1
        ).clone()
        rpy_rate = torch.zeros_like(offset_b)
        center = self.reference_center_offset_b[:count, None, :]
        trajectory_ids = self.trajectory_ids[:count]

        def mask_for(name: str) -> torch.Tensor:
            return trajectory_ids == self.cfg.trajectory_types.index(name)

        if "line_x" in self.cfg.trajectory_types:
            mask = mask_for("line_x")
            offset_b[mask, :, 0] = self.cfg.line_amplitude_m * sin_phase[mask]
            offset_rate_b[mask, :, 0] = self.cfg.line_amplitude_m * omega[mask] * cos_phase[mask]
        if "line_y" in self.cfg.trajectory_types:
            mask = mask_for("line_y")
            offset_b[mask, :, 1] = self.cfg.line_amplitude_m * sin_phase[mask]
            offset_rate_b[mask, :, 1] = self.cfg.line_amplitude_m * omega[mask] * cos_phase[mask]
        if "circle_xy" in self.cfg.trajectory_types:
            mask = mask_for("circle_xy")
            offset_b[mask, :, 0] = self.cfg.circle_radius_m * (cos_phase[mask] - 1.0)
            offset_b[mask, :, 1] = self.cfg.circle_radius_m * sin_phase[mask]
            offset_rate_b[mask, :, 0] = -self.cfg.circle_radius_m * omega[mask] * sin_phase[mask]
            offset_rate_b[mask, :, 1] = self.cfg.circle_radius_m * omega[mask] * cos_phase[mask]
        if "figure8_xy" in self.cfg.trajectory_types:
            mask = mask_for("figure8_xy")
            offset_b[mask, :, 0] = self.cfg.figure8_amplitude_m[0] * sin_phase[mask]
            offset_b[mask, :, 1] = self.cfg.figure8_amplitude_m[1] * torch.sin(2.0 * phase[mask])
            offset_rate_b[mask, :, 0] = self.cfg.figure8_amplitude_m[0] * omega[mask] * cos_phase[mask]
            offset_rate_b[mask, :, 1] = (
                2.0 * self.cfg.figure8_amplitude_m[1] * omega[mask] * torch.cos(2.0 * phase[mask])
            )
        if "vertical" in self.cfg.trajectory_types:
            mask = mask_for("vertical")
            offset_b[mask, :, 2] = self.cfg.vertical_amplitude_m * sin_phase[mask]
            offset_rate_b[mask, :, 2] = self.cfg.vertical_amplitude_m * omega[mask] * cos_phase[mask]
        if "yaw_scan" in self.cfg.trajectory_types:
            mask = mask_for("yaw_scan")
            rpy[mask, :, 2] += self.cfg.orientation_amplitude_rpy[2] * sin_phase[mask]
            rpy_rate[mask, :, 2] = self.cfg.orientation_amplitude_rpy[2] * omega[mask] * cos_phase[mask]
        if "six_d" in self.cfg.trajectory_types:
            mask = mask_for("six_d")
            offset_b[mask, :, 0] = self.cfg.circle_radius_m * (cos_phase[mask] - 1.0)
            offset_b[mask, :, 1] = self.cfg.circle_radius_m * sin_phase[mask]
            offset_b[mask, :, 2] = self.cfg.vertical_amplitude_m * torch.sin(0.5 * phase[mask])
            offset_rate_b[mask, :, 0] = -self.cfg.circle_radius_m * omega[mask] * sin_phase[mask]
            offset_rate_b[mask, :, 1] = self.cfg.circle_radius_m * omega[mask] * cos_phase[mask]
            offset_rate_b[mask, :, 2] = (
                0.5 * self.cfg.vertical_amplitude_m * omega[mask] * torch.cos(0.5 * phase[mask])
            )
            roll_amp, pitch_amp, yaw_amp = self.cfg.orientation_amplitude_rpy
            rpy[mask, :, 0] += roll_amp * sin_phase[mask]
            rpy[mask, :, 1] += pitch_amp * torch.sin(0.5 * phase[mask])
            rpy[mask, :, 2] += yaw_amp * torch.sin(0.75 * phase[mask])
            rpy_rate[mask, :, 0] = roll_amp * omega[mask] * cos_phase[mask]
            rpy_rate[mask, :, 1] = 0.5 * pitch_amp * omega[mask] * torch.cos(0.5 * phase[mask])
            rpy_rate[mask, :, 2] = 0.75 * yaw_amp * omega[mask] * torch.cos(0.75 * phase[mask])

        raw_delta_b = center + offset_b
        delta_b = ramp.unsqueeze(-1) * raw_delta_b
        delta_rate_b = ramp_rate.unsqueeze(-1) * raw_delta_b + ramp.unsqueeze(-1) * offset_rate_b
        scaled_rpy = ramp.unsqueeze(-1) * rpy
        scaled_rpy_rate = ramp_rate.unsqueeze(-1) * rpy + ramp.unsqueeze(-1) * rpy_rate

        frame_quat = self.reference_frame_quat_w[:count, None, :].expand(-1, preview_count, -1)
        delta_w = quat_apply(frame_quat.reshape(-1, 4), delta_b.reshape(-1, 3)).reshape(count, preview_count, 3)
        delta_rate_w = quat_apply(
            frame_quat.reshape(-1, 4), delta_rate_b.reshape(-1, 3)
        ).reshape(count, preview_count, 3)
        angular_rate_w = quat_apply(
            frame_quat.reshape(-1, 4), scaled_rpy_rate.reshape(-1, 3)
        ).reshape(count, preview_count, 3)

        local_rotation = quat_from_euler_xyz(
            scaled_rpy[..., 0].reshape(-1),
            scaled_rpy[..., 1].reshape(-1),
            scaled_rpy[..., 2].reshape(-1),
        ).reshape(count, preview_count, 4)
        world_rotation = quat_mul(
            quat_mul(frame_quat.reshape(-1, 4), local_rotation.reshape(-1, 4)),
            quat_inv(frame_quat.reshape(-1, 4)),
        ).reshape(count, preview_count, 4)
        start_quat = self.start_pose_w[:count, None, 3:].expand(-1, preview_count, -1)

        pose = torch.zeros((count, preview_count, 7), device=self.device)
        pose[..., :3] = self.start_pose_w[:count, None, :3] + delta_w
        pose[..., 3:] = quat_mul(
            world_rotation.reshape(-1, 4), start_quat.reshape(-1, 4)
        ).reshape(count, preview_count, 4)
        twist = torch.cat((delta_rate_w, angular_rate_w), dim=-1)
        return pose, twist, ramp

    def reference_pose_at_offsets(self, offsets_s: Sequence[float]) -> torch.Tensor:
        if len(offsets_s) == 0:
            raise ValueError("offsets_s must contain at least one preview time")
        if any(offset < 0.0 for offset in offsets_s):
            raise ValueError("trajectory preview offsets must be non-negative")
        offsets = torch.as_tensor(offsets_s, device=self.device, dtype=self.elapsed_s.dtype)
        elapsed = self.elapsed_s.unsqueeze(-1) + offsets.unsqueeze(0)
        pose, _, _ = self._evaluate_reference(elapsed)
        return pose

    def _capture_reference(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        root_pos_w = self.robot.data.root_pos_w[env_ids]
        root_quat_w = self.robot.data.root_quat_w[env_ids]
        tcp_pos_w = self.robot.data.body_pos_w[env_ids, self.body_idx]
        tcp_quat_w = self.robot.data.body_quat_w[env_ids, self.body_idx]
        sampled_pos_b, sampled_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, tcp_pos_w, tcp_quat_w
        )
        self.start_pose_w[env_ids, :3] = tcp_pos_w
        self.start_pose_w[env_ids, 3:] = tcp_quat_w
        self.goal_pose_w[env_ids] = self.start_pose_w[env_ids]
        self.pose_command_w[env_ids] = self.start_pose_w[env_ids]
        self.reference_root_pose_w[env_ids, :3] = root_pos_w
        self.reference_root_pose_w[env_ids, 3:] = root_quat_w
        self.reference_frame_quat_w[env_ids] = root_quat_w
        self.ghost_root_pose_w[env_ids] = self.reference_root_pose_w[env_ids]
        self.sampled_tcp_pose_b[env_ids, :3] = sampled_pos_b
        self.sampled_tcp_pose_b[env_ids, 3:] = sampled_quat_b

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        if self.cfg.trajectory_weights is None:
            self.trajectory_ids[ids] = ids % len(self.cfg.trajectory_types)
        else:
            weights = torch.as_tensor(self.cfg.trajectory_weights, device=self.device)
            self.trajectory_ids[ids] = torch.multinomial(weights, ids.numel(), replacement=True)
        self.frequency_hz[ids] = torch.empty(ids.numel(), device=self.device).uniform_(
            *self.cfg.frequency_range_hz
        )
        trajectory_ids = self.trajectory_ids[ids]
        non_periodic_ids = [
            self.cfg.trajectory_types.index(name)
            for name in ("hold", "waypoint_planar", "waypoint_low")
            if name in self.cfg.trajectory_types
        ]
        periodic_mask = torch.ones(ids.numel(), dtype=torch.bool, device=self.device)
        for non_periodic_id in non_periodic_ids:
            periodic_mask &= trajectory_ids != non_periodic_id
        self.is_periodic_mask[ids] = periodic_mask

        self.reference_center_offset_b[ids] = torch.as_tensor(
            self.cfg.center_offset_b, device=self.device
        )
        self.reference_orientation_offset_rpy[ids] = 0.0
        for trajectory_name in ("waypoint_planar", "waypoint_low"):
            if trajectory_name not in self.cfg.trajectory_types:
                continue
            local_mask = trajectory_ids == self.cfg.trajectory_types.index(trajectory_name)
            waypoint_ids = ids[local_mask]
            if waypoint_ids.numel() == 0:
                continue
            radius = torch.empty(waypoint_ids.numel(), device=self.device).uniform_(*self.cfg.radius_range)
            if self.cfg.short_radius_probability > 0.0:
                short_mask = torch.rand(waypoint_ids.numel(), device=self.device) < self.cfg.short_radius_probability
                short_radius = torch.empty(waypoint_ids.numel(), device=self.device).uniform_(
                    *self.cfg.short_radius_range
                )
                radius = torch.where(short_mask, short_radius, radius)
            bearing_range = (
                self.cfg.spatial_bearing_range
                if trajectory_name == "waypoint_low" and self.cfg.spatial_bearing_range is not None
                else self.cfg.bearing_range
            )
            bearing = torch.empty(waypoint_ids.numel(), device=self.device).uniform_(*bearing_range)
            self.reference_center_offset_b[waypoint_ids, 0] = radius * torch.cos(bearing)
            self.reference_center_offset_b[waypoint_ids, 1] = radius * torch.sin(bearing)
            if trajectory_name == "waypoint_low":
                self.reference_center_offset_b[waypoint_ids, 2] = torch.empty(
                    waypoint_ids.numel(), device=self.device
                ).uniform_(*self.cfg.height_offset_range)
            yaw = torch.empty(waypoint_ids.numel(), device=self.device).uniform_(*self.cfg.yaw_range)
            self.reference_orientation_offset_rpy[waypoint_ids, 2] = yaw

        spatial_ids = [
            self.cfg.trajectory_types.index(name)
            for name in ("waypoint_low", "vertical", "six_d")
            if name in self.cfg.trajectory_types
        ]
        spatial_mask = torch.zeros(ids.numel(), dtype=torch.bool, device=self.device)
        for spatial_id in spatial_ids:
            spatial_mask |= trajectory_ids == spatial_id
        self.spatial_mask[ids] = spatial_mask
        self.ghost_translation_b[ids] = self.reference_center_offset_b[ids]
        zeros = torch.zeros(ids.numel(), device=self.device)
        self.ghost_rotation_b[ids] = quat_from_euler_xyz(
            zeros,
            self.reference_orientation_offset_rpy[ids, 1],
            self.reference_orientation_offset_rpy[ids, 2],
        )
        self._capture_reference(ids)
        self.elapsed_s[ids] = 0.0
        self.motion_progress[ids] = 0.0
        self.twist_command_w[ids] = 0.0
        self.command_buffer[ids, :7] = self.pose_command_w[ids]
        self.command_buffer[ids, 7:] = 0.0

    def _recapture_settling_reference(self, env_ids: torch.Tensor) -> None:
        self._capture_reference(env_ids)

    def _update_command(self) -> None:
        self.elapsed_s += self._env.step_dt
        if self.cfg.recapture_during_settle:
            settling_ids = torch.nonzero(
                self.elapsed_s <= self.cfg.settle_time_s, as_tuple=False
            ).squeeze(-1)
            self._recapture_settling_reference(settling_ids)
        pose, twist, ramp = self._evaluate_reference(self.elapsed_s.unsqueeze(-1))
        self.pose_command_w.copy_(pose[:, 0])
        self.goal_pose_w.copy_(pose[:, 0])
        self.twist_command_w.copy_(twist[:, 0])
        self.motion_progress.copy_(ramp[:, 0])
        self.command_buffer[:, :7] = self.pose_command_w
        self.command_buffer[:, 7:] = self.twist_command_w

        displacement_w = self.pose_command_w[:, :3] - self.start_pose_w[:, :3]
        self.ghost_root_pose_w[:, :3] = self.reference_root_pose_w[:, :3] + displacement_w
        self.ghost_root_pose_w[:, 3:] = self.reference_root_pose_w[:, 3:]


@configclass
class PeriodicWorldPoseCommandCfg(FKReachableWorldPoseCommandCfg):
    """Configuration for continuous EE trajectory generation."""

    class_type: type = PeriodicWorldPoseCommand
    trajectory_types: tuple[str, ...] = PeriodicWorldPoseCommand._SUPPORTED_TYPES
    trajectory_weights: tuple[float, ...] | None = None
    """Optional categorical training weights; ``None`` assigns types round-robin."""
    frequency_range_hz: tuple[float, float] = (0.10, 0.30)
    center_offset_b: tuple[float, float, float] = (0.25, 0.0, 0.0)
    line_amplitude_m: float = 0.10
    circle_radius_m: float = 0.08
    figure8_amplitude_m: tuple[float, float] = (0.12, 0.06)
    vertical_amplitude_m: float = 0.08
    orientation_amplitude_rpy: tuple[float, float, float] = (0.12, 0.12, 0.25)
    ramp_time_s: float = 2.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.trajectory_types) == 0:
            raise ValueError("trajectory_types must not be empty")
        unknown = set(self.trajectory_types) - set(PeriodicWorldPoseCommand._SUPPORTED_TYPES)
        if unknown:
            raise ValueError(f"Unsupported trajectory types: {sorted(unknown)}")
        if len(set(self.trajectory_types)) != len(self.trajectory_types):
            raise ValueError("trajectory_types must be unique")
        if self.trajectory_weights is not None:
            if len(self.trajectory_weights) != len(self.trajectory_types):
                raise ValueError("trajectory_weights must match trajectory_types")
            if min(self.trajectory_weights) < 0.0 or sum(self.trajectory_weights) <= 0.0:
                raise ValueError("trajectory_weights must be non-negative with positive sum")
        if self.frequency_range_hz[0] <= 0.0 or self.frequency_range_hz[1] < self.frequency_range_hz[0]:
            raise ValueError(f"Invalid frequency range: {self.frequency_range_hz}")
        if len(self.center_offset_b) != 3:
            raise ValueError("center_offset_b must have three elements")
        if len(self.figure8_amplitude_m) != 2:
            raise ValueError("figure8_amplitude_m must have two elements")
        if len(self.orientation_amplitude_rpy) != 3:
            raise ValueError("orientation_amplitude_rpy must have three elements")
        if min(
            self.line_amplitude_m,
            self.circle_radius_m,
            *self.figure8_amplitude_m,
            self.vertical_amplitude_m,
            self.ramp_time_s,
        ) <= 0.0:
            raise ValueError("Trajectory amplitudes and ramp_time_s must be positive")


class WorkspaceSweepWorldPoseCommand(PeriodicWorldPoseCommand):
    """Deterministic pose sequence for finding weak regions of the EE workspace.

    In ``cycle_targets`` mode every environment visits the same named targets in
    order, which is useful for a single review video.  Otherwise each environment
    receives one target and one magnitude variant, so a batch benchmark can test
    every direction independently even if another target terminates early.

    Offsets and RPY rotations are expressed in the settled root frame and are
    introduced with a minimum-jerk transition followed by a constant hold.  The
    command retains the observation interface used by the trained policy.
    """

    cfg: WorkspaceSweepWorldPoseCommandCfg

    def __init__(self, cfg: WorkspaceSweepWorldPoseCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.workspace_target_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.workspace_target_scales = torch.ones(self.num_envs, device=self.device)
        self.current_target_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.segment_elapsed_s = torch.zeros(self.num_envs, device=self.device)
        self.is_holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def target_names(self) -> tuple[str, ...]:
        return self.cfg.target_names

    @property
    def trajectory_names(self) -> tuple[str, ...]:
        # Preserve the interface consumed by the existing benchmark tooling.
        return self.cfg.target_names

    def __str__(self) -> str:
        mode = "sequential video sweep" if self.cfg.cycle_targets else "parallel target grid"
        return (
            "WorkspaceSweepWorldPoseCommand:\n"
            f"\tMode: {mode}\n"
            f"\tTargets: {self.cfg.target_names}\n"
            f"\tTiming: settle={self.cfg.settle_time_s}s, "
            f"transition={self.cfg.transition_time_s}s, hold={self.cfg.hold_time_s}s"
        )

    def _target_state(
        self, elapsed_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return interpolated local offset/RPY, their rates, and target ids."""
        if elapsed_s.ndim == 1:
            elapsed_s = elapsed_s.unsqueeze(-1)
        count, preview_count = elapsed_s.shape
        active_s = torch.clamp(elapsed_s - self.cfg.settle_time_s, min=0.0)

        offsets = torch.as_tensor(self.cfg.target_offsets_b, device=self.device, dtype=elapsed_s.dtype)
        rotations = torch.as_tensor(self.cfg.target_rpy, device=self.device, dtype=elapsed_s.dtype)
        target_count = offsets.shape[0]

        if self.cfg.cycle_targets:
            segment_duration = self.cfg.transition_time_s + self.cfg.hold_time_s
            unwrapped_segment = torch.floor(active_s / segment_duration).to(torch.long)
            target_ids = torch.remainder(unwrapped_segment, target_count)
            previous_ids = torch.remainder(target_ids - 1, target_count)
            segment_elapsed = active_s - unwrapped_segment.to(active_s.dtype) * segment_duration
            scales = torch.ones_like(active_s)
        else:
            target_ids = self.workspace_target_ids[:count, None].expand(-1, preview_count)
            previous_ids = torch.zeros_like(target_ids)
            segment_elapsed = active_s
            scales = self.workspace_target_scales[:count, None].expand(-1, preview_count)

        linear_progress = torch.clamp(segment_elapsed / self.cfg.transition_time_s, min=0.0, max=1.0)
        blend, blend_shape_rate = self._minimum_jerk(linear_progress)
        blend_rate = torch.where(
            segment_elapsed < self.cfg.transition_time_s,
            blend_shape_rate / self.cfg.transition_time_s,
            torch.zeros_like(blend_shape_rate),
        )

        previous_offset = offsets[previous_ids] * scales.unsqueeze(-1)
        target_offset = offsets[target_ids] * scales.unsqueeze(-1)
        previous_rpy = rotations[previous_ids] * scales.unsqueeze(-1)
        target_rpy = rotations[target_ids] * scales.unsqueeze(-1)
        delta_offset = target_offset - previous_offset
        delta_rpy = target_rpy - previous_rpy

        local_offset = previous_offset + blend.unsqueeze(-1) * delta_offset
        local_rpy = previous_rpy + blend.unsqueeze(-1) * delta_rpy
        local_offset_rate = blend_rate.unsqueeze(-1) * delta_offset
        local_rpy_rate = blend_rate.unsqueeze(-1) * delta_rpy
        return local_offset, local_offset_rate, local_rpy, local_rpy_rate, target_ids

    def _evaluate_reference(
        self, elapsed_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if elapsed_s.ndim == 1:
            elapsed_s = elapsed_s.unsqueeze(-1)
        count, preview_count = elapsed_s.shape
        offset_b, offset_rate_b, rpy, rpy_rate, _ = self._target_state(elapsed_s)
        frame_quat = self.reference_frame_quat_w[:count, None, :].expand(-1, preview_count, -1)

        offset_w = quat_apply(
            frame_quat.reshape(-1, 4), offset_b.reshape(-1, 3)
        ).reshape(count, preview_count, 3)
        offset_rate_w = quat_apply(
            frame_quat.reshape(-1, 4), offset_rate_b.reshape(-1, 3)
        ).reshape(count, preview_count, 3)
        angular_rate_w = quat_apply(
            frame_quat.reshape(-1, 4), rpy_rate.reshape(-1, 3)
        ).reshape(count, preview_count, 3)

        local_rotation = quat_from_euler_xyz(
            rpy[..., 0].reshape(-1), rpy[..., 1].reshape(-1), rpy[..., 2].reshape(-1)
        ).reshape(count, preview_count, 4)
        world_rotation = quat_mul(
            quat_mul(frame_quat.reshape(-1, 4), local_rotation.reshape(-1, 4)),
            quat_inv(frame_quat.reshape(-1, 4)),
        ).reshape(count, preview_count, 4)
        start_quat = self.start_pose_w[:count, None, 3:].expand(-1, preview_count, -1)

        pose = torch.zeros((count, preview_count, 7), device=self.device)
        pose[..., :3] = self.start_pose_w[:count, None, :3] + offset_w
        pose[..., 3:] = quat_mul(
            world_rotation.reshape(-1, 4), start_quat.reshape(-1, 4)
        ).reshape(count, preview_count, 4)
        twist = torch.cat((offset_rate_w, angular_rate_w), dim=-1)
        active = torch.clamp(elapsed_s - self.cfg.settle_time_s, min=0.0)
        progress = torch.clamp(active / self.cfg.transition_time_s, min=0.0, max=1.0)
        return pose, twist, progress

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        super()._resample_command(ids)
        target_count = len(self.cfg.target_names)
        variant_count = len(self.cfg.target_scales)
        if self.cfg.cycle_targets:
            self.workspace_target_ids[ids] = 0
            self.workspace_target_scales[ids] = 1.0
        else:
            self.workspace_target_ids[ids] = torch.remainder(ids, target_count)
            variant_ids = torch.remainder(torch.div(ids, target_count, rounding_mode="floor"), variant_count)
            scales = torch.as_tensor(self.cfg.target_scales, device=self.device)
            self.workspace_target_scales[ids] = scales[variant_ids]
        self.trajectory_ids[ids] = self.workspace_target_ids[ids]
        self.is_periodic_mask[ids] = False
        self.current_target_ids[ids] = self.workspace_target_ids[ids]
        self.segment_elapsed_s[ids] = 0.0
        self.is_holding[ids] = False

    def _update_command(self) -> None:
        super()._update_command()
        active_s = torch.clamp(self.elapsed_s - self.cfg.settle_time_s, min=0.0)
        if self.cfg.cycle_targets:
            segment_duration = self.cfg.transition_time_s + self.cfg.hold_time_s
            unwrapped_segment = torch.floor(active_s / segment_duration).to(torch.long)
            self.current_target_ids.copy_(torch.remainder(unwrapped_segment, len(self.cfg.target_names)))
            self.segment_elapsed_s.copy_(active_s - unwrapped_segment.to(active_s.dtype) * segment_duration)
        else:
            self.current_target_ids.copy_(self.workspace_target_ids)
            self.segment_elapsed_s.copy_(active_s)
        self.trajectory_ids.copy_(self.current_target_ids)
        self.is_holding.copy_(self.segment_elapsed_s >= self.cfg.transition_time_s)

        displacement_w = self.pose_command_w[:, :3] - self.start_pose_w[:, :3]
        self.ghost_translation_b.copy_(quat_apply_inverse(self.reference_frame_quat_w, displacement_w))
        _, _, local_rpy, _, _ = self._target_state(self.elapsed_s.unsqueeze(-1))
        current_rpy = local_rpy[:, 0]
        zeros = torch.zeros(self.num_envs, device=self.device)
        self.ghost_rotation_b.copy_(quat_from_euler_xyz(zeros, current_rpy[:, 1], current_rpy[:, 2]))
        self.spatial_mask.copy_(torch.abs(self.ghost_translation_b[:, 2]) > 1.0e-3)


@configclass
class WorkspaceSweepWorldPoseCommandCfg(PeriodicWorldPoseCommandCfg):
    """Configuration for the deterministic Stage-20 workspace audit."""

    class_type: type = WorkspaceSweepWorldPoseCommand
    target_names: tuple[str, ...] = (
        "home",
        "front",
        "left",
        "right",
        "rear",
        "high",
        "low_front",
        "low_left",
        "low_right",
        "far_front",
        "pose_combo",
        "rear_pose",
        "return_home",
    )
    target_offsets_b: tuple[tuple[float, float, float], ...] = (
        (0.00, 0.00, 0.00),
        (0.40, 0.00, 0.00),
        (0.20, 0.32, 0.00),
        (0.20, -0.32, 0.00),
        (-0.35, 0.00, 0.00),
        (0.20, 0.00, 0.16),
        (0.30, 0.00, -0.16),
        (0.25, 0.25, -0.12),
        (0.25, -0.25, -0.12),
        (0.60, 0.00, 0.00),
        (0.20, 0.12, 0.05),
        (-0.25, 0.00, 0.03),
        (0.00, 0.00, 0.00),
    )
    target_rpy: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, math.radians(-10.0), 0.0),
        (0.0, math.radians(10.0), 0.0),
        (math.radians(8.0), math.radians(8.0), math.radians(15.0)),
        (math.radians(-8.0), math.radians(8.0), math.radians(-15.0)),
        (0.0, 0.0, 0.0),
        (math.radians(15.0), math.radians(-10.0), math.radians(25.0)),
        (math.radians(-10.0), math.radians(10.0), math.radians(-25.0)),
        (0.0, 0.0, 0.0),
    )
    target_scales: tuple[float, ...] = (0.85, 1.0, 1.15)
    transition_time_s: float = 2.5
    hold_time_s: float = 1.5
    cycle_targets: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.target_names) == 0:
            raise ValueError("target_names must not be empty")
        if len(self.target_names) != len(self.target_offsets_b) or len(self.target_names) != len(self.target_rpy):
            raise ValueError("target names, offsets, and RPY rotations must have equal length")
        if len(set(self.target_names)) != len(self.target_names):
            raise ValueError("workspace target names must be unique")
        if len(self.target_scales) == 0 or min(self.target_scales) <= 0.0:
            raise ValueError("target_scales must contain positive values")
        if self.transition_time_s <= 0.0 or self.hold_time_s <= 0.0:
            raise ValueError("workspace transition and hold times must be positive")


class DoorHandlePoseCommand(FKReachableWorldPoseCommand):
    """Drive the TCP to a pose anchored to an articulated door handle.

    Stage 11 deliberately uses a collision-free pre-grasp point.  It keeps the
    actor interface identical to the free-space EE-WBC task, which lets a
    validated tracking checkpoint be evaluated before contact, gripper, and
    latch dynamics are introduced.
    """

    cfg: DoorHandlePoseCommandCfg

    def __init__(self, cfg: DoorHandlePoseCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.door: Articulation = env.scene[cfg.door_asset_name]
        body_ids, body_names = self.door.find_bodies(cfg.handle_body_name)
        if body_names != [cfg.handle_body_name]:
            raise ValueError(
                f"Expected exactly one door body named '{cfg.handle_body_name}', found {body_names}."
            )
        self.handle_body_idx = body_ids[0]
        self.target_jitter_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.is_door_target = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def __str__(self) -> str:
        return (
            "DoorHandlePoseCommand:\n"
            f"\tHandle body: {self.cfg.door_asset_name}/{self.cfg.handle_body_name}\n"
            f"\tApproach offset: {self.cfg.approach_offset_handle}\n"
            f"\tMotion timing: settle={self.cfg.settle_time_s}s, move={self.cfg.motion_time_s}s"
        )

    def _handle_goal_position_w(self, env_ids: Sequence[int]) -> torch.Tensor:
        count = len(env_ids)
        offset = torch.tensor(
            self.cfg.approach_offset_handle,
            dtype=self.door.data.body_pos_w.dtype,
            device=self.device,
        ).expand(count, -1)
        identity = torch.zeros((count, 4), device=self.device)
        identity[:, 0] = 1.0
        goal_pos_w, _ = combine_frame_transforms(
            self.door.data.body_pos_w[env_ids, self.handle_body_idx],
            self.door.data.body_quat_w[env_ids, self.handle_body_idx],
            offset,
            identity,
        )
        return goal_pos_w + self.target_jitter_w[env_ids]

    def _capture_door_reference(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        root_pos_w = self.robot.data.root_pos_w[env_ids]
        root_quat_w = self.robot.data.root_quat_w[env_ids]
        start_pos_w = self.robot.data.body_pos_w[env_ids, self.body_idx]
        start_quat_w = self.robot.data.body_quat_w[env_ids, self.body_idx]
        sampled_pos_b, sampled_quat_b = subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            start_pos_w,
            start_quat_w,
        )
        goal_pos_w = self._handle_goal_position_w(env_ids)
        if self.cfg.preserve_start_orientation:
            goal_quat_w = start_quat_w
        else:
            orientation_offset = torch.tensor(
                self.cfg.orientation_offset_handle,
                dtype=start_quat_w.dtype,
                device=self.device,
            ).expand(len(env_ids), -1)
            goal_quat_w = quat_mul(
                self.door.data.body_quat_w[env_ids, self.handle_body_idx],
                orientation_offset,
            )

        tcp_delta_w = goal_pos_w - start_pos_w
        ghost_root_pos_w = root_pos_w + tcp_delta_w
        ghost_root_quat_w = root_quat_w

        self.start_pose_w[env_ids, :3] = start_pos_w
        self.start_pose_w[env_ids, 3:] = start_quat_w
        self.goal_pose_w[env_ids, :3] = goal_pos_w
        self.goal_pose_w[env_ids, 3:] = goal_quat_w
        self.pose_command_w[env_ids] = self.start_pose_w[env_ids]
        self.twist_command_w[env_ids] = 0.0
        self.command_buffer[env_ids, :7] = self.pose_command_w[env_ids]
        self.command_buffer[env_ids, 7:] = 0.0
        self.ghost_root_pose_w[env_ids, :3] = ghost_root_pos_w
        self.ghost_root_pose_w[env_ids, 3:] = ghost_root_quat_w
        self.sampled_tcp_pose_b[env_ids, :3] = sampled_pos_b
        self.sampled_tcp_pose_b[env_ids, 3:] = sampled_quat_b
        self.ghost_translation_b[env_ids] = quat_apply_inverse(root_quat_w, tcp_delta_w)
        self.ghost_rotation_b[env_ids] = 0.0
        self.ghost_rotation_b[env_ids, 0] = 1.0
        self.spatial_mask[env_ids] = torch.abs(tcp_delta_w[:, 2]) > 1.0e-3

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        for axis, bounds in enumerate(self.cfg.position_jitter_range):
            self.target_jitter_w[env_ids, axis] = torch.empty(
                len(env_ids), device=self.device
            ).uniform_(*bounds)
        self._capture_door_reference(env_ids)
        self.elapsed_s[env_ids] = 0.0
        self.motion_progress[env_ids] = 0.0

    def _recapture_settling_reference(self, env_ids: torch.Tensor) -> None:
        self._capture_door_reference(env_ids)


@configclass
class DoorHandlePoseCommandCfg(FKReachableWorldPoseCommandCfg):
    """Configuration for a door-handle anchored pre-grasp command."""

    class_type: type = DoorHandlePoseCommand
    door_asset_name: str = "door"
    handle_body_name: str = "handle_grasp"
    approach_offset_handle: tuple[float, float, float] = (-0.10, 0.0, 0.0)
    position_jitter_range: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ] = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    preserve_start_orientation: bool = True
    orientation_offset_handle: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.approach_offset_handle) != 3:
            raise ValueError("approach_offset_handle must have three elements")
        if len(self.position_jitter_range) != 3:
            raise ValueError("position_jitter_range must contain x/y/z ranges")
        for bounds in self.position_jitter_range:
            if bounds[1] < bounds[0]:
                raise ValueError(f"Invalid door-target jitter range: {bounds}")
        if len(self.orientation_offset_handle) != 4:
            raise ValueError("orientation_offset_handle must be a wxyz quaternion")


class DoorHandleTurnCommand(DoorHandlePoseCommand):
    """Approach a handle, wait for grasp closure, then command a turn arc."""

    cfg: DoorHandleTurnCommandCfg

    def __init__(self, cfg: DoorHandleTurnCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.turn_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.phase_clock_s = torch.zeros(self.num_envs, device=self.device)
        self.turn_goal_pose_w = torch.zeros_like(self.goal_pose_w)
        self.turn_goal_pose_w[:, 3] = 1.0

    def _refresh_turn_goal(self, env_ids: Sequence[int]) -> None:
        translation = torch.tensor(
            self.cfg.turn_translation_w,
            dtype=self.goal_pose_w.dtype,
            device=self.device,
        )
        self.turn_goal_pose_w[env_ids] = self.goal_pose_w[env_ids]
        self.turn_goal_pose_w[env_ids, :3] += translation

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        self.turn_started[env_ids] = False
        self.phase_clock_s[env_ids] = 0.0
        super()._resample_command(env_ids)
        self._refresh_turn_goal(env_ids)

    def _recapture_settling_reference(self, env_ids: torch.Tensor) -> None:
        # Recapture only belongs to the approach phase.  After the phase switch
        # elapsed_s is deliberately reset to settle_time_s, so guard against a
        # floating-point equality re-anchoring the turn goal to the live handle.
        approach_ids = env_ids[~self.turn_started[env_ids]]
        if approach_ids.numel() == 0:
            return
        super()._recapture_settling_reference(approach_ids)
        self._refresh_turn_goal(approach_ids)

    def _update_command(self) -> None:
        self.phase_clock_s += self._env.step_dt
        super()._update_command()
        transition_ids = torch.nonzero(
            (~self.turn_started) & (self.phase_clock_s >= self.cfg.turn_start_s),
            as_tuple=False,
        ).squeeze(-1)
        if transition_ids.numel() == 0:
            return

        current_pos_w = self.robot.data.body_pos_w[transition_ids, self.body_idx]
        current_quat_w = self.robot.data.body_quat_w[transition_ids, self.body_idx]
        self.start_pose_w[transition_ids, :3] = current_pos_w
        self.start_pose_w[transition_ids, 3:] = current_quat_w
        self.goal_pose_w[transition_ids] = self.turn_goal_pose_w[transition_ids]
        if self.cfg.preserve_start_orientation:
            self.goal_pose_w[transition_ids, 3:] = current_quat_w
        self.pose_command_w[transition_ids] = self.start_pose_w[transition_ids]
        self.twist_command_w[transition_ids] = 0.0
        self.command_buffer[transition_ids, :7] = self.pose_command_w[transition_ids]
        self.command_buffer[transition_ids, 7:] = 0.0
        self.elapsed_s[transition_ids] = self.cfg.settle_time_s
        self.motion_progress[transition_ids] = 0.0
        self.turn_started[transition_ids] = True


@configclass
class DoorHandleTurnCommandCfg(DoorHandlePoseCommandCfg):
    """Configuration for the grasp-then-turn command state machine."""

    class_type: type = DoorHandleTurnCommand
    turn_start_s: float = 7.0
    turn_translation_w: tuple[float, float, float] = (0.0, 0.015, -0.075)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.turn_start_s <= self.settle_time_s + self.motion_time_s:
            raise ValueError("turn_start_s must leave time for approach and grasp closure")
        if len(self.turn_translation_w) != 3:
            raise ValueError("turn_translation_w must have three elements")


class DoorHandlePullCommand(DoorHandleTurnCommand):
    """Add a third minimum-jerk phase that pulls an unlatched door open."""

    cfg: DoorHandlePullCommandCfg

    def __init__(self, cfg: DoorHandlePullCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.pull_started = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pull_goal_pose_w = torch.zeros_like(self.goal_pose_w)
        self.pull_goal_pose_w[:, 3] = 1.0

    def _refresh_pull_goal(self, env_ids: Sequence[int]) -> None:
        translation = torch.tensor(
            self.cfg.pull_translation_w,
            dtype=self.goal_pose_w.dtype,
            device=self.device,
        )
        self.pull_goal_pose_w[env_ids] = self.turn_goal_pose_w[env_ids]
        self.pull_goal_pose_w[env_ids, :3] += translation

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        self.pull_started[env_ids] = False
        super()._resample_command(env_ids)
        self._refresh_pull_goal(env_ids)

    def _recapture_settling_reference(self, env_ids: torch.Tensor) -> None:
        super()._recapture_settling_reference(env_ids)
        approach_ids = env_ids[~self.turn_started[env_ids]]
        if approach_ids.numel() > 0:
            self._refresh_pull_goal(approach_ids)

    def _update_command(self) -> None:
        super()._update_command()
        transition_ids = torch.nonzero(
            (~self.pull_started) & (self.phase_clock_s >= self.cfg.pull_start_s),
            as_tuple=False,
        ).squeeze(-1)
        if transition_ids.numel() == 0:
            return

        current_pos_w = self.robot.data.body_pos_w[transition_ids, self.body_idx]
        current_quat_w = self.robot.data.body_quat_w[transition_ids, self.body_idx]
        self.start_pose_w[transition_ids, :3] = current_pos_w
        self.start_pose_w[transition_ids, 3:] = current_quat_w
        self.goal_pose_w[transition_ids] = self.pull_goal_pose_w[transition_ids]
        if self.cfg.preserve_start_orientation:
            self.goal_pose_w[transition_ids, 3:] = current_quat_w
        self.pose_command_w[transition_ids] = self.start_pose_w[transition_ids]
        self.twist_command_w[transition_ids] = 0.0
        self.command_buffer[transition_ids, :7] = self.pose_command_w[transition_ids]
        self.command_buffer[transition_ids, 7:] = 0.0
        self.elapsed_s[transition_ids] = self.cfg.settle_time_s
        self.motion_progress[transition_ids] = 0.0
        self.pull_started[transition_ids] = True


@configclass
class DoorHandlePullCommandCfg(DoorHandleTurnCommandCfg):
    """Configuration for the final pull phase."""

    class_type: type = DoorHandlePullCommand
    pull_start_s: float = 11.0
    pull_translation_w: tuple[float, float, float] = (-0.22, -0.13, 0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pull_start_s <= self.turn_start_s + self.motion_time_s:
            raise ValueError("pull_start_s must leave time to complete the turn phase")
        if len(self.pull_translation_w) != 3:
            raise ValueError("pull_translation_w must have three elements")
