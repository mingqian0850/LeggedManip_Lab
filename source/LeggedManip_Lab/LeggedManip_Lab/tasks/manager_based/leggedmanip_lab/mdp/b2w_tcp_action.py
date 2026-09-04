# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hierarchical B2W locomotion plus world-frame Z1 TCP control."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms

from LeggedManip_Lab.controllers.robot_lab_b2w_policy import (
    ROBOT_LAB_B2W_ACTION_SIZE,
    ROBOT_LAB_B2W_JOINT_NAMES,
    ROBOT_LAB_B2W_LEG_ACTION_SIZE,
    RobotLabB2WPolicy,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


class B2WZ1TCPAction(ActionTerm):
    """Map a two-dimensional coordinator action to frozen B2W + Z1 control.

    The trainable action is only ``[forward velocity, yaw rate]``.  A pinned
    Robot Lab policy converts that command into twelve leg position targets and
    four wheel velocity targets at 50 Hz.  At every 200 Hz physics step, a DLS
    differential-IK loop converts the smooth immutable-world TCP reference into
    six Z1 position targets.  Both layers act on one articulation through this
    single term, avoiding last-writer-wins target conflicts.
    """

    cfg: B2WZ1TCPActionCfg

    def __init__(self, cfg: B2WZ1TCPActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = self._asset
        if self.robot.is_fixed_base:
            raise ValueError("B2WZ1TCPAction requires a floating-base articulation.")

        self.policy_joint_ids, policy_joint_names = self.robot.find_joints(
            list(ROBOT_LAB_B2W_JOINT_NAMES), preserve_order=True
        )
        if tuple(policy_joint_names) != ROBOT_LAB_B2W_JOINT_NAMES:
            raise ValueError(f"Unexpected B2W policy joint order: {policy_joint_names}")
        self.leg_joint_ids = self.policy_joint_ids[:ROBOT_LAB_B2W_LEG_ACTION_SIZE]
        self.wheel_joint_ids = self.policy_joint_ids[ROBOT_LAB_B2W_LEG_ACTION_SIZE:]

        self.arm_joint_ids, arm_joint_names = self.robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        if tuple(arm_joint_names) != ARM_JOINT_NAMES:
            raise ValueError(f"Unexpected Z1 joint order: {arm_joint_names}")
        tcp_body_ids, tcp_body_names = self.robot.find_bodies(cfg.body_name)
        if tcp_body_names != [cfg.body_name]:
            raise ValueError(f"Expected exactly one TCP body named '{cfg.body_name}', found {tcp_body_names}.")
        self.tcp_body_id = tcp_body_ids[0]
        self.tcp_jacobian_id = self.tcp_body_id
        # Floating articulations prepend six root generalized coordinates.
        self.arm_jacobian_columns = [joint_id + 6 for joint_id in self.arm_joint_ids]

        jacobian_shape = tuple(self.robot.root_physx_view.get_jacobians().shape)
        expected_shape = (self.robot.num_bodies, self.robot.num_joints + 6)
        if jacobian_shape[1] != expected_shape[0] or jacobian_shape[3] != expected_shape[1]:
            raise ValueError(
                f"Unexpected floating Jacobian shape {jacobian_shape}; expected body/dof axes {expected_shape}."
            )

        self.policy = RobotLabB2WPolicy(
            cfg.checkpoint_path,
            num_envs=self.num_envs,
            device=self.device,
            runtime_checks=cfg.runtime_checks,
        )
        expected_policy_q = self.policy.default_joint_position.expand(self.num_envs, -1)
        actual_policy_q = self.robot.data.default_joint_pos[:, self.policy_joint_ids]
        if not torch.allclose(actual_policy_q, expected_policy_q, atol=1.0e-7, rtol=0.0):
            raise ValueError("B2W asset default joints do not match the frozen policy deployment contract.")

        self.ik = DifferentialIKController(cfg.controller, self.num_envs, self.device)
        self.arm_joint_limits = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids].clone()
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = self.robot.data.default_joint_vel.clone()
        self.zero_joint_effort = torch.zeros_like(self.default_joint_pos)

        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self.velocity_command_b = torch.zeros((self.num_envs, 3), device=self.device)
        self.low_level_actions = torch.zeros((self.num_envs, ROBOT_LAB_B2W_ACTION_SIZE), device=self.device)
        self.leg_position_targets = expected_policy_q[:, :ROBOT_LAB_B2W_LEG_ACTION_SIZE].clone()
        self.wheel_velocity_targets = torch.zeros(
            (self.num_envs, ROBOT_LAB_B2W_ACTION_SIZE - ROBOT_LAB_B2W_LEG_ACTION_SIZE), device=self.device
        )
        self.arm_position_targets = self.default_joint_pos[:, self.arm_joint_ids].clone()
        self._policy_step_counter = 0
        self._force_policy_update = True
        self.joint_limit_clamp_count = torch.zeros((), dtype=torch.int64, device=self.device)
        self.joint_rate_limit_count = torch.zeros((), dtype=torch.int64, device=self.device)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Physical ``[vx, wz]`` command after scaling and zero-drift bias."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"Expected coordinator actions with shape {self._raw_actions.shape}, got {actions.shape}.")
        self._raw_actions.copy_(actions)
        bounded = torch.clamp(actions, -1.0, 1.0)
        self._processed_actions[:, 0] = bounded[:, 0] * self.cfg.max_linear_velocity
        self._processed_actions[:, 0] -= self.cfg.forward_command_bias
        self._processed_actions[:, 1] = bounded[:, 1] * self.cfg.max_yaw_velocity
        self.velocity_command_b[:, 0] = self._processed_actions[:, 0]
        self.velocity_command_b[:, 1] = 0.0
        self.velocity_command_b[:, 2] = self._processed_actions[:, 1]

    def apply_actions(self) -> None:
        if self._force_policy_update or self._policy_step_counter % self.cfg.low_level_decimation == 0:
            policy_action = self.policy.act(
                base_angular_velocity_b=self.robot.data.root_ang_vel_b,
                projected_gravity_b=self.robot.data.projected_gravity_b,
                velocity_command_b=self.velocity_command_b,
                joint_position=self.robot.data.joint_pos[:, self.policy_joint_ids],
                joint_velocity=self.robot.data.joint_vel[:, self.policy_joint_ids],
            )
            self.low_level_actions.copy_(policy_action.raw)
            self.leg_position_targets.copy_(policy_action.leg_position_targets)
            self.wheel_velocity_targets.copy_(policy_action.wheel_velocity_targets)
            self._force_policy_update = False

        arm_target = self._compute_arm_target()

        # Establish deterministic defaults first, then override each controlled
        # group with its appropriate target type.
        self.robot.set_joint_position_target(self.default_joint_pos)
        self.robot.set_joint_velocity_target(self.default_joint_vel)
        self.robot.set_joint_effort_target(self.zero_joint_effort)
        self.robot.set_joint_position_target(self.leg_position_targets, joint_ids=self.leg_joint_ids)
        self.robot.set_joint_velocity_target(self.wheel_velocity_targets, joint_ids=self.wheel_joint_ids)
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_joint_ids)

        gravity_effort = self.robot.root_physx_view.get_gravity_compensation_forces()
        self.robot.set_joint_effort_target(gravity_effort[:, self.arm_jacobian_columns], joint_ids=self.arm_joint_ids)
        self._policy_step_counter += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
            policy_env_ids = None
        else:
            policy_env_ids = env_ids
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids, 0] = -self.cfg.forward_command_bias
        self._processed_actions[env_ids, 1] = 0.0
        self.velocity_command_b[env_ids, 0] = -self.cfg.forward_command_bias
        self.velocity_command_b[env_ids, 1:] = 0.0
        self.low_level_actions[env_ids] = 0.0
        self.leg_position_targets[env_ids] = self.policy.default_joint_position[:, :ROBOT_LAB_B2W_LEG_ACTION_SIZE]
        self.wheel_velocity_targets[env_ids] = 0.0
        self.arm_position_targets[env_ids] = self.default_joint_pos[env_ids][:, self.arm_joint_ids]
        self.policy.reset(policy_env_ids)
        self.ik.reset(policy_env_ids)
        # The low-level cadence is global, while episode resets may affect only a
        # subset of environments. Force one batched inference on the next physics
        # step so no reset environment receives a stale pre-reset target.
        self._force_policy_update = True

    def _compute_arm_target(self) -> torch.Tensor:
        command_term = self._env.command_manager.get_term(self.cfg.command_name)
        target_pose_w = command_term.command[:, :7]
        current_pos_b, current_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.robot.data.body_pos_w[:, self.tcp_body_id],
            self.robot.data.body_quat_w[:, self.tcp_body_id],
        )
        target_pos_b, target_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            target_pose_w[:, :3],
            target_pose_w[:, 3:],
        )

        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.tcp_jacobian_id, :, self.arm_jacobian_columns
        ].clone()
        root_rotation_inverse = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
        jacobian[:, :3] = torch.bmm(root_rotation_inverse, jacobian[:, :3])
        jacobian[:, 3:] = torch.bmm(root_rotation_inverse, jacobian[:, 3:])
        if self.cfg.runtime_checks and not torch.all(torch.isfinite(jacobian)):
            raise FloatingPointError("TCP Jacobian contains NaN or Inf.")

        current_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
        self.ik.set_command(torch.cat((target_pos_b, target_quat_b), dim=-1))
        unconstrained_target = self.ik.compute(current_pos_b, current_quat_b, jacobian, current_joint_pos)
        desired_target = current_joint_pos + self.cfg.ik_gain * (unconstrained_target - current_joint_pos)

        unconstrained_delta = desired_target - self.arm_position_targets
        max_joint_step = self.cfg.max_arm_joint_speed * self._env.physics_dt
        target_delta = torch.clamp(unconstrained_delta, -max_joint_step, max_joint_step)
        self.joint_rate_limit_count += torch.count_nonzero(torch.abs(target_delta - unconstrained_delta) > 1.0e-7)
        limited_target = self.arm_position_targets + target_delta
        clamped_target = torch.maximum(
            torch.minimum(limited_target, self.arm_joint_limits[..., 1]),
            self.arm_joint_limits[..., 0],
        )
        self.joint_limit_clamp_count += torch.count_nonzero(torch.abs(clamped_target - limited_target) > 1.0e-7)
        self.arm_position_targets.copy_(clamped_target)
        return self.arm_position_targets


@configclass
class B2WZ1TCPActionCfg(ActionTermCfg):
    """Configuration for :class:`B2WZ1TCPAction`."""

    class_type: type[ActionTerm] = B2WZ1TCPAction
    asset_name: str = MISSING
    checkpoint_path: str = MISSING
    command_name: str = "tcp_pose"
    body_name: str = "tcp_frame"
    low_level_decimation: int = 4
    max_linear_velocity: float = 0.5
    max_yaw_velocity: float = 0.8
    forward_command_bias: float = 0.05
    max_arm_joint_speed: float = 2.5
    ik_gain: float = 0.25
    runtime_checks: bool = False
    controller: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.03},
    )

    def __post_init__(self) -> None:
        if self.low_level_decimation < 1:
            raise ValueError("low_level_decimation must be positive")
        if self.max_linear_velocity <= 0.0 or self.max_yaw_velocity <= 0.0:
            raise ValueError("Velocity scales must be positive")
        if self.max_arm_joint_speed <= 0.0:
            raise ValueError("max_arm_joint_speed must be positive")
