# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Extra observations and regularizers for unified B2W + Z1 EE control."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import compute_pose_error, quat_apply_inverse, quat_inv, quat_mul

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


def action_term_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Squared raw action for one named actuator group."""
    action = env.action_manager.get_term(action_name).raw_actions
    return torch.sum(torch.square(action), dim=-1)


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
