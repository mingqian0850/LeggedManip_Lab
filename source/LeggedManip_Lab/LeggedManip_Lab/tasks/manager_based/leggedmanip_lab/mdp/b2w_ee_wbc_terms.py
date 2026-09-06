# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Extra observations and regularizers for unified B2W + Z1 EE control."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import compute_pose_error, euler_xyz_from_quat, quat_apply_inverse, quat_inv, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def future_tcp_pose_errors_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    offsets_s: Sequence[float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Future TCP pose errors in the current root frame, flattened as ``K*6``.

    Each preview item is ``[dx, dy, dz, rx, ry, rz]``.  The target is evaluated
    from the immutable world-frame minimum-jerk trajectory; the current TCP is
    intentionally repeated rather than extrapolated.  This tells the actor
    where the task is going while leaving prediction of robot motion to the
    policy and its observation history.
    """
    command_term = env.command_manager.get_term(command_name)
    references = command_term.reference_pose_at_offsets(offsets_s)
    robot: Articulation = env.scene[asset_cfg.name]
    tcp_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    tcp_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids[0]]
    preview_count = references.shape[1]

    current_pos = tcp_pos_w[:, None, :].expand(-1, preview_count, -1).reshape(-1, 3)
    current_quat = tcp_quat_w[:, None, :].expand(-1, preview_count, -1).reshape(-1, 4)
    target_pos = references[..., :3].reshape(-1, 3)
    target_quat = references[..., 3:].reshape(-1, 4)
    position_error_w, orientation_error_w = compute_pose_error(
        current_pos,
        current_quat,
        target_pos,
        target_quat,
        rot_error_type="axis_angle",
    )
    root_quat = robot.data.root_quat_w[:, None, :].expand(-1, preview_count, -1).reshape(-1, 4)
    position_error_b = quat_apply_inverse(root_quat, position_error_w)
    orientation_error_b = quat_apply_inverse(root_quat, orientation_error_w)
    return torch.cat((position_error_b, orientation_error_b), dim=-1).reshape(env.num_envs, -1)


def ghost_root_planar_error_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Privileged ``[x, y, yaw]`` error to the known feasible ghost root pose."""
    robot: Articulation = env.scene[asset_cfg.name]
    ghost_pose_w = env.command_manager.get_term(command_name).ghost_root_pose_w
    position_error_b = quat_apply_inverse(
        robot.data.root_quat_w,
        ghost_pose_w[:, :3] - robot.data.root_pos_w,
    )
    quaternion_error = quat_mul(quat_inv(robot.data.root_quat_w), ghost_pose_w[:, 3:])
    w, x, y, z = quaternion_error.unbind(dim=-1)
    yaw_error = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.cat((position_error_b[:, :2], yaw_error.unsqueeze(-1)), dim=-1)


def wheel_lateral_slip_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Squared lateral wheel speed, active only while a wheel is in contact."""
    robot: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    velocity_w = robot.data.body_lin_vel_w[:, asset_cfg.body_ids]
    body_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids]
    velocity_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        velocity_w.reshape(-1, 3),
    ).reshape(env.num_envs, -1, 3)
    contact_force = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    in_contact = torch.linalg.vector_norm(contact_force, dim=-1) > contact_threshold
    return torch.sum(torch.square(velocity_b[..., 1]) * in_contact.float(), dim=-1)


def wheel_contact_count(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Privileged number of wheels currently carrying contact force."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_force = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return torch.sum(
        (torch.linalg.vector_norm(contact_force, dim=-1) > contact_threshold).float(),
        dim=-1,
        keepdim=True,
    )


def wheel_contact_force_balance_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    minimum_mean_force: float = 1.0,
) -> torch.Tensor:
    """Squared coefficient of variation of the four wheel contact loads."""
    if minimum_mean_force <= 0.0:
        raise ValueError("minimum_mean_force must be positive")
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.linalg.vector_norm(
        sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1
    )
    mean_force = torch.mean(force, dim=-1)
    coefficient_of_variation = torch.std(force, dim=-1) / torch.clamp(
        mean_force, min=minimum_mean_force
    )
    return torch.square(coefficient_of_variation)


def action_term_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Squared raw action for one named actuator group."""
    action = env.action_manager.get_term(action_name).raw_actions
    return torch.sum(torch.square(action), dim=-1)


def joint_default_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Squared deviation from the nominal joint posture for a selected group."""
    robot: Articulation = env.scene[asset_cfg.name]
    deviation = (
        robot.data.joint_pos[:, asset_cfg.joint_ids]
        - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(deviation), dim=-1)


def settling_joint_default_deviation_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Nominal-posture cost active only during the reference settling window."""
    command = env.command_manager.get_term(command_name)
    settling = command.elapsed_s <= command.cfg.settle_time_s
    return settling.float() * joint_default_deviation_l2(env, asset_cfg)


def leg_left_right_symmetry_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize visually crooked left/right leg pairs without tying front to rear.

    The required order is FR, FL, RR, RL with hip/thigh/calf inside each leg.
    Hip abduction has mirrored signs; thigh and calf flexion have equal signs.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_position = robot.data.joint_pos[:, asset_cfg.joint_ids]
    if joint_position.shape[1] != 12:
        raise ValueError("leg_left_right_symmetry_l2 requires exactly 12 ordered leg joints")
    errors = torch.stack(
        (
            joint_position[:, 0] + joint_position[:, 3],
            joint_position[:, 1] - joint_position[:, 4],
            joint_position[:, 2] - joint_position[:, 5],
            joint_position[:, 6] + joint_position[:, 9],
            joint_position[:, 7] - joint_position[:, 10],
            joint_position[:, 8] - joint_position[:, 11],
        ),
        dim=-1,
    )
    return torch.sum(torch.square(errors), dim=-1)


def root_roll_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Roll-only upright cost; pitch remains available for low forward targets."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.square(robot.data.projected_gravity_b[:, 1])


def root_roll_rate_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared body roll rate while leaving pitch and yaw rates unconstrained."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_ang_vel_b[:, 0])


def body_pair_center_distance_barrier(
    env: ManagerBasedRLEnv,
    first_asset_cfg: SceneEntityCfg,
    second_asset_cfg: SceneEntityCfg,
    safe_distance: float,
    collision_distance: float,
) -> torch.Tensor:
    """Smooth early-warning proxy for a calibrated self-collision pair.

    This is deliberately a center-distance barrier rather than a claim about
    exact mesh separation.  ``safe_distance`` and ``collision_distance`` must
    be calibrated from deterministic contact probes for the specific asset.
    """
    if collision_distance < 0.0 or safe_distance <= collision_distance:
        raise ValueError("safe_distance must be greater than a non-negative collision_distance")
    if first_asset_cfg.name != second_asset_cfg.name:
        raise ValueError("body-pair barrier currently requires both bodies on the same articulation")
    robot: Articulation = env.scene[first_asset_cfg.name]
    first_position = robot.data.body_pos_w[:, first_asset_cfg.body_ids[0]]
    second_position = robot.data.body_pos_w[:, second_asset_cfg.body_ids[0]]
    distance = torch.linalg.vector_norm(first_position - second_position, dim=-1)
    normalized_shortfall = torch.clamp(
        (safe_distance - distance) / (safe_distance - collision_distance),
        min=0.0,
    )
    return torch.square(normalized_shortfall)


def base_height_safety_barrier(
    env: ManagerBasedRLEnv,
    safe_height: float,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-sided normalized penalty as root height approaches termination.

    Heights above ``safe_height`` are unpenalized, which leaves room for useful
    whole-body height and tilt motion.  The quadratic reaches one at
    ``minimum_height`` so its reward weight has an intuitive safety scale.
    """
    if minimum_height >= safe_height:
        raise ValueError("minimum_height must be lower than safe_height")
    robot: Articulation = env.scene[asset_cfg.name]
    normalized_shortfall = torch.clamp(
        (safe_height - robot.data.root_pos_w[:, 2]) / (safe_height - minimum_height),
        min=0.0,
    )
    return torch.square(normalized_shortfall)


def spatial_base_pitch_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    maximum_height_offset: float,
    maximum_pitch: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Shape low forward goals toward a safe nose-down whole-body posture.

    The command's signed world-z offset determines the desired pitch.  The term
    is inactive for planar replay and ramps with the minimum-jerk command phase,
    so settling is not disturbed.  Positive pitch is the B2-W nose-down
    convention used by the command curriculum.
    """
    if maximum_height_offset <= 0.0 or maximum_pitch <= 0.0 or std <= 0.0:
        raise ValueError("maximum_height_offset, maximum_pitch, and std must be positive")
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]
    _, root_pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    height_fraction = torch.clamp(
        -command.ghost_translation_b[:, 2] / maximum_height_offset,
        min=0.0,
        max=1.0,
    )
    desired_pitch = maximum_pitch * height_fraction
    phase = command.motion_progress
    active = command.spatial_mask.float()
    return active * phase * torch.exp(-torch.square(root_pitch - desired_pitch) / std**2)


def wheeled_travel_heading_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    activation_radius: float,
    full_radius: float,
    std: float,
    turn_in_end: float = 0.25,
    turn_out_start: float = 0.70,
    maximum_heading: float = math.pi / 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a turn--travel--realign maneuver for distant planar EE goals.

    B2-W cannot translate sideways without wheel scrub.  For a distant lateral
    TCP displacement, this term asks the root to face the direction of travel
    during the middle of the minimum-jerk transition and to recover its initial
    yaw before the EE settles.  Rear goals use the equivalent reverse-driving
    heading, so the robot backs up instead of turning through 180 degrees.

    The radius gate keeps the original close-range manipulation policy intact.
    The returned value is zero below ``activation_radius`` and smoothly reaches
    full strength at ``full_radius``.
    """
    if (
        activation_radius < 0.0
        or full_radius <= activation_radius
        or std <= 0.0
        or maximum_heading <= 0.0
    ):
        raise ValueError("heading radius bounds and std must be positive and ordered")
    if not 0.0 < turn_in_end < turn_out_start < 1.0:
        raise ValueError("turn phases must satisfy 0 < turn_in_end < turn_out_start < 1")

    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]

    planar_offset = command.ghost_translation_b[:, :2]
    planar_radius = torch.linalg.vector_norm(planar_offset, dim=-1)
    radius_gate = torch.clamp(
        (planar_radius - activation_radius) / (full_radius - activation_radius),
        min=0.0,
        max=1.0,
    )

    # Select the forward or reverse direction requiring the smaller yaw change.
    travel_heading = torch.atan2(planar_offset[:, 1], planar_offset[:, 0])
    reverse = torch.cos(travel_heading) < 0.0
    travel_heading = torch.where(
        reverse,
        travel_heading - torch.sign(travel_heading) * math.pi,
        travel_heading,
    )
    travel_heading = torch.clamp(
        travel_heading, min=-maximum_heading, max=maximum_heading
    )

    def smoothstep(value: torch.Tensor) -> torch.Tensor:
        value = torch.clamp(value, min=0.0, max=1.0)
        return value * value * (3.0 - 2.0 * value)

    progress = torch.clamp(command.motion_progress, min=0.0, max=1.0)
    turn_in = smoothstep(progress / turn_in_end)
    turn_out = 1.0 - smoothstep((progress - turn_out_start) / (1.0 - turn_out_start))
    desired_yaw = turn_in * turn_out * travel_heading

    reference_to_root = quat_mul(quat_inv(command.reference_root_pose_w[:, 3:]), robot.data.root_quat_w)
    w, x, y, z = reference_to_root.unbind(dim=-1)
    root_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw_error = root_yaw - desired_yaw
    yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
    return radius_gate * torch.exp(-torch.square(yaw_error) / std**2)


def tcp_goal_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Privileged scalar distance from TCP to the immutable final goal."""
    robot: Articulation = env.scene[asset_cfg.name]
    tcp_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    goal = env.command_manager.get_term(command_name).final_command
    return torch.linalg.vector_norm(goal[:, :3] - tcp_pos_w, dim=-1, keepdim=True)
