# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Observations and rewards for the hierarchical B2W + Z1 TCP task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import compute_pose_error, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _robot_and_tcp(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> tuple[Articulation, torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[asset_cfg.name]
    tcp_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    tcp_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids[0]]
    return robot, tcp_pos_w, tcp_quat_w


def _reference_pose(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)[:, :7]


def _hide_extreme_low_goal_mask(command, minimum_height_offset_m: float) -> torch.Tensor:
    """Mask only the deep-low commands that disturbed the settling phase."""
    if minimum_height_offset_m >= 0.0:
        raise ValueError("minimum_height_offset_m must be negative")
    extreme_low = command.ghost_translation_b[:, 2] <= minimum_height_offset_m
    settling = command.elapsed_s <= command.cfg.settle_time_s
    return extreme_low & settling


def tcp_reference_position_error_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """TCP-to-reference displacement expressed in the current root frame."""
    robot, tcp_pos_w, _ = _robot_and_tcp(env, asset_cfg)
    reference = _reference_pose(env, command_name)
    return quat_apply_inverse(robot.data.root_quat_w, reference[:, :3] - tcp_pos_w)


def tcp_reference_orientation_error_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Shortest TCP orientation error as a root-frame axis-angle vector."""
    robot, tcp_pos_w, tcp_quat_w = _robot_and_tcp(env, asset_cfg)
    reference = _reference_pose(env, command_name)
    _, rotation_error_w = compute_pose_error(
        tcp_pos_w,
        tcp_quat_w,
        reference[:, :3],
        reference[:, 3:],
        rot_error_type="axis_angle",
    )
    return quat_apply_inverse(robot.data.root_quat_w, rotation_error_w)


def tcp_final_position_error_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    hide_during_settle: bool = False,
    minimum_hidden_height_offset_m: float = -0.12,
) -> torch.Tensor:
    """TCP-to-immutable-goal displacement expressed in the current root frame."""
    robot, tcp_pos_w, _ = _robot_and_tcp(env, asset_cfg)
    command = env.command_manager.get_term(command_name)
    goal = command.final_command
    error = quat_apply_inverse(robot.data.root_quat_w, goal[:, :3] - tcp_pos_w)
    if hide_during_settle:
        hidden = _hide_extreme_low_goal_mask(
            command, minimum_hidden_height_offset_m
        )
        error = torch.where(hidden.unsqueeze(-1), torch.zeros_like(error), error)
    return error


def tcp_final_orientation_error_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    hide_during_settle: bool = False,
    minimum_hidden_height_offset_m: float = -0.12,
) -> torch.Tensor:
    """Immutable-goal orientation error as a current-root axis-angle vector."""
    robot, tcp_pos_w, tcp_quat_w = _robot_and_tcp(env, asset_cfg)
    command = env.command_manager.get_term(command_name)
    goal = command.final_command
    _, rotation_error_w = compute_pose_error(
        tcp_pos_w,
        tcp_quat_w,
        goal[:, :3],
        goal[:, 3:],
        rot_error_type="axis_angle",
    )
    error = quat_apply_inverse(robot.data.root_quat_w, rotation_error_w)
    if hide_during_settle:
        hidden = _hide_extreme_low_goal_mask(
            command, minimum_hidden_height_offset_m
        )
        error = torch.where(hidden.unsqueeze(-1), torch.zeros_like(error), error)
    return error


def desired_tcp_twist_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Analytic minimum-jerk TCP twist expressed in the current root axes."""
    robot: Articulation = env.scene[asset_cfg.name]
    twist_w = env.command_manager.get_command(command_name)[:, 7:13]
    linear_b = quat_apply_inverse(robot.data.root_quat_w, twist_w[:, :3])
    angular_b = quat_apply_inverse(robot.data.root_quat_w, twist_w[:, 3:])
    return torch.cat((linear_b, angular_b), dim=-1)


def normalized_arm_joint_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Arm displacement from home normalized by each soft joint half-range."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    half_range = 0.5 * (limits[..., 1] - limits[..., 0])
    return (joint_pos - home) / torch.clamp(half_range, min=1.0e-3)


def scaled_arm_joint_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    scale: float = 0.2,
) -> torch.Tensor:
    """Scaled Z1 joint velocity in explicit joint order."""
    robot: Articulation = env.scene[asset_cfg.name]
    return scale * robot.data.joint_vel[:, asset_cfg.joint_ids]


def root_linear_velocity_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_lin_vel_b


def root_angular_velocity_b(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_ang_vel_b


def root_projected_gravity_b(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.projected_gravity_b


def frozen_policy_last_action(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Expose the 16-D frozen-policy memory used in its next observation."""
    return env.action_manager.get_term(action_name).low_level_actions


def reference_motion_progress(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Minimum-jerk phase in ``[0, 1]``."""
    command_term = env.command_manager.get_term(command_name)
    return command_term.motion_progress.unsqueeze(-1)


def tcp_reference_position_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    error = tcp_reference_position_error_b(env, command_name, asset_cfg)
    return torch.exp(-torch.sum(torch.square(error), dim=-1) / std**2)


def tcp_reference_orientation_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    error = tcp_reference_orientation_error_b(env, command_name, asset_cfg)
    return torch.exp(-torch.sum(torch.square(error), dim=-1) / std**2)


def tcp_twist_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    linear_std: float,
    angular_std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot, _, _ = _robot_and_tcp(env, asset_cfg)
    tcp_linear_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids[0]]
    tcp_angular_w = robot.data.body_ang_vel_w[:, asset_cfg.body_ids[0]]
    desired_twist_w = env.command_manager.get_command(command_name)[:, 7:13]
    linear_error = torch.sum(torch.square(tcp_linear_w - desired_twist_w[:, :3]), dim=-1)
    angular_error = torch.sum(torch.square(tcp_angular_w - desired_twist_w[:, 3:]), dim=-1)
    return torch.exp(-linear_error / linear_std**2 - angular_error / angular_std**2)


def final_tcp_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_threshold: float,
    orientation_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    robot, tcp_pos_w, tcp_quat_w = _robot_and_tcp(env, asset_cfg)
    goal = command_term.final_command
    position_error, orientation_error = compute_pose_error(
        tcp_pos_w,
        tcp_quat_w,
        goal[:, :3],
        goal[:, 3:],
        rot_error_type="axis_angle",
    )
    del robot
    reached = torch.logical_and(
        torch.linalg.vector_norm(position_error, dim=-1) < position_threshold,
        torch.linalg.vector_norm(orientation_error, dim=-1) < orientation_threshold,
    )
    return torch.logical_and(reached, command_term.motion_progress > 0.999).float()


def arm_home_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    normalized_position = normalized_arm_joint_position(env, asset_cfg)
    return torch.sum(torch.square(normalized_position), dim=-1)


def terminal_arm_home_deviation_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_progress: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Resolve whole-body redundancy toward arm home near trajectory completion.

    This is deliberately inactive during the early reference motion so the arm
    can absorb fast tracking error. Near the terminal pose it encourages the
    coordinator to reposition the mobile base, without exposing or rewarding a
    privileged target base pose.
    """
    if not 0.0 <= start_progress < 1.0:
        raise ValueError("start_progress must lie in [0, 1)")
    progress = env.command_manager.get_term(command_name).motion_progress
    phase = torch.clamp((progress - start_progress) / (1.0 - start_progress), 0.0, 1.0)
    smooth_gate = phase * phase * (3.0 - 2.0 * phase)
    return smooth_gate * arm_home_deviation_l2(env, asset_cfg)


def arm_joint_margin_barrier(
    env: ManagerBasedRLEnv,
    margin_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize only the part of an arm configuration close to a soft limit."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    joint_range = torch.clamp(limits[..., 1] - limits[..., 0], min=1.0e-3)
    lower_margin = (joint_pos - limits[..., 0]) / joint_range
    upper_margin = (limits[..., 1] - joint_pos) / joint_range
    nearest_margin = torch.minimum(lower_margin, upper_margin)
    normalized_violation = torch.clamp((margin_threshold - nearest_margin) / margin_threshold, min=0.0)
    return torch.sum(torch.square(normalized_violation), dim=-1)


def arm_joint_velocity_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)


def base_height_error_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_pos_w[:, 2] - target_height)


def coordinator_action_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    action_term = env.action_manager.get_term(action_name)
    return torch.sum(torch.square(action_term.processed_actions), dim=-1)


def settled_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_threshold: float,
    orientation_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    robot, tcp_pos_w, tcp_quat_w = _robot_and_tcp(env, asset_cfg)
    goal = command_term.final_command
    position_error, orientation_error = compute_pose_error(
        tcp_pos_w,
        tcp_quat_w,
        goal[:, :3],
        goal[:, 3:],
        rot_error_type="axis_angle",
    )
    near_goal = torch.logical_and(
        torch.linalg.vector_norm(position_error, dim=-1) < position_threshold,
        torch.linalg.vector_norm(orientation_error, dim=-1) < orientation_threshold,
    )
    tcp_linear_speed_sq = torch.sum(torch.square(robot.data.body_lin_vel_w[:, asset_cfg.body_ids[0]]), dim=-1)
    tcp_angular_speed_sq = torch.sum(torch.square(robot.data.body_ang_vel_w[:, asset_cfg.body_ids[0]]), dim=-1)
    base_speed_sq = torch.sum(torch.square(robot.data.root_lin_vel_b), dim=-1)
    base_yaw_rate_sq = torch.square(robot.data.root_ang_vel_b[:, 2])
    return near_goal.float() * (tcp_linear_speed_sq + 0.1 * tcp_angular_speed_sq + base_speed_sq + base_yaw_rate_sq)
