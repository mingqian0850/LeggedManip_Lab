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
    quat_apply_inverse,
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
