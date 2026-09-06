"""Action helpers that do not add trainable dimensions to unified EE-WBC."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class LowPassJointPositionAction(JointPositionAction):
    """Apply a first-order low-pass filter to joint-position targets.

    The raw actor action is preserved for PPO bookkeeping.  Only the physical
    PD target is filtered, so existing checkpoints remain shape-compatible and
    can be evaluated before deciding whether retraining is worthwhile.
    """

    cfg: LowPassJointPositionActionCfg

    def __init__(self, cfg: LowPassJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._filtered_actions = torch.zeros_like(self._processed_actions)
        if isinstance(self._offset, torch.Tensor):
            self._filtered_actions.copy_(self._offset)
        else:
            self._filtered_actions.fill_(self._offset)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self._filtered_actions.lerp_(self._processed_actions, self.cfg.alpha)
        self._processed_actions = self._filtered_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        if isinstance(self._offset, torch.Tensor):
            self._filtered_actions[env_ids] = self._offset[env_ids]
        else:
            self._filtered_actions[env_ids] = self._offset


@configclass
class LowPassJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for filtered joint targets; ``alpha=1`` is unfiltered."""

    class_type: type = LowPassJointPositionAction
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")


class MirroredLowPassJointPositionAction(LowPassJointPositionAction):
    """Project four-leg residuals toward a left/right mirrored subspace.

    The expected order is ``FR, FL, RR, RL`` with hip/thigh/calf per leg.
    ``asymmetric_residual_scale`` retains part of the actor's asymmetric
    component so the stance can compensate for an off-centre payload.
    """

    cfg: MirroredLowPassJointPositionActionCfg

    def process_actions(self, actions: torch.Tensor) -> None:
        # Bypass LowPassJointPositionAction.process_actions so projection occurs
        # before temporal filtering while JointPositionAction still records the
        # untouched actor action for PPO observations and action-rate rewards.
        JointPositionAction.process_actions(self, actions)
        if self.action_dim != 12:
            raise ValueError("Mirrored leg projection requires exactly 12 actions")
        offset = self._offset if isinstance(self._offset, torch.Tensor) else float(self._offset)
        residual = self._processed_actions - offset
        mirrored = torch.empty_like(residual)
        for right, left in ((0, 3), (6, 9)):
            hip = 0.5 * (residual[:, right] - residual[:, left])
            thigh = 0.5 * (residual[:, right + 1] + residual[:, left + 1])
            calf = 0.5 * (residual[:, right + 2] + residual[:, left + 2])
            mirrored[:, right] = hip
            mirrored[:, left] = -hip
            mirrored[:, right + 1] = thigh
            mirrored[:, left + 1] = thigh
            mirrored[:, right + 2] = calf
            mirrored[:, left + 2] = calf
        beta = self.cfg.asymmetric_residual_scale
        projected_residual = mirrored + beta * (residual - mirrored)
        self._processed_actions = offset + projected_residual
        self._filtered_actions.lerp_(self._processed_actions, self.cfg.alpha)
        self._processed_actions = self._filtered_actions


@configclass
class MirroredLowPassJointPositionActionCfg(LowPassJointPositionActionCfg):
    """Configuration for payload-aware mirrored leg residual projection."""

    class_type: type = MirroredLowPassJointPositionAction
    asymmetric_residual_scale: float = 0.25

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.asymmetric_residual_scale <= 1.0:
            raise ValueError("asymmetric_residual_scale must lie in [0, 1]")


class FixedJointPositionAction(ActionTerm):
    """Hold selected joints at their reset default without consuming actor output."""

    cfg: FixedJointPositionActionCfg

    def __init__(self, cfg: FixedJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = self._asset
        self.joint_ids, self.joint_names = self.robot.find_joints(cfg.joint_names, preserve_order=True)
        if len(self.joint_ids) == 0:
            raise ValueError(f"No joints matched fixed hold patterns: {cfg.joint_names}")
        self.targets = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
        if cfg.target_position is not None:
            self.targets.fill_(cfg.target_position)
        self._raw_actions = torch.empty((self.num_envs, 0), device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"Expected zero-width hold action, received {tuple(actions.shape)}")

    def apply_actions(self) -> None:
        self.robot.set_joint_position_target(self.targets, joint_ids=self.joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.targets[env_ids] = self.robot.data.default_joint_pos[env_ids][:, self.joint_ids]
        if self.cfg.target_position is not None:
            self.targets[env_ids] = self.cfg.target_position


@configclass
class FixedJointPositionActionCfg(ActionTermCfg):
    class_type: type = FixedJointPositionAction
    joint_names: list[str] = MISSING
    target_position: float | None = None
    """Optional absolute joint target; ``None`` holds the asset default."""


class PhasedJointPositionAction(FixedJointPositionAction):
    """Smoothly close a zero-dimensional gripper after TCP alignment."""

    cfg: PhasedJointPositionActionCfg

    def __init__(self, cfg: PhasedJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._env = env
        self.closure_progress = torch.zeros(self.num_envs, device=self.device)
        self.targets.fill_(cfg.open_position)

    def apply_actions(self) -> None:
        command = self._env.command_manager.get_term(self.cfg.command_name)
        ready = (
            (command.motion_progress >= self.cfg.close_start_progress)
            & (command.metrics["goal_position_error"] <= self.cfg.capture_distance)
        )
        self.closure_progress[ready] = torch.clamp(
            self.closure_progress[ready] + self._env.step_dt / self.cfg.close_duration_s,
            max=1.0,
        )
        blend = self.closure_progress.unsqueeze(-1)
        self.targets[:] = self.cfg.open_position + blend * (
            self.cfg.closed_position - self.cfg.open_position
        )
        self.robot.set_joint_position_target(self.targets, joint_ids=self.joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.closure_progress[env_ids] = 0.0
        self.targets[env_ids] = self.cfg.open_position


@configclass
class PhasedJointPositionActionCfg(FixedJointPositionActionCfg):
    """Configuration for an automatic align-then-close gripper state."""

    class_type: type = PhasedJointPositionAction
    command_name: str = "tcp_pose"
    open_position: float = -1.45
    closed_position: float = 0.0
    close_start_progress: float = 0.90
    capture_distance: float = 0.04
    close_duration_s: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.close_start_progress <= 1.0:
            raise ValueError("close_start_progress must lie in [0, 1]")
        if self.capture_distance <= 0.0:
            raise ValueError("capture_distance must be positive")
        if self.close_duration_s <= 0.0:
            raise ValueError("close_duration_s must be positive")


class DoorLatchAction(ActionTerm):
    """Zero-dimensional latch that releases a door only after handle rotation."""

    cfg: DoorLatchActionCfg

    def __init__(self, cfg: DoorLatchActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.door: Articulation = self._asset
        self.door_joint_ids, names = self.door.find_joints(cfg.door_joint_name)
        self.handle_joint_ids, handle_names = self.door.find_joints(cfg.handle_joint_name)
        if names != [cfg.door_joint_name] or handle_names != [cfg.handle_joint_name]:
            raise ValueError(
                f"Could not resolve door latch joints: door={names}, handle={handle_names}"
            )
        self.door_joint_idx = self.door_joint_ids[0]
        self.handle_joint_idx = self.handle_joint_ids[0]
        self.unlocked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lock_effort = torch.zeros(self.num_envs, device=self.device)
        self._raw_actions = torch.empty((self.num_envs, 0), device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"Expected zero-width latch action, received {tuple(actions.shape)}")

    def apply_actions(self) -> None:
        handle_angle = self.door.data.joint_pos[:, self.handle_joint_idx]
        self.unlocked |= handle_angle >= self.cfg.release_angle
        door_angle = self.door.data.joint_pos[:, self.door_joint_idx]
        door_velocity = self.door.data.joint_vel[:, self.door_joint_idx]
        lock_effort = -self.cfg.lock_stiffness * door_angle - self.cfg.lock_damping * door_velocity
        lock_effort = torch.clamp(lock_effort, -self.cfg.maximum_lock_effort, self.cfg.maximum_lock_effort)
        self.lock_effort.copy_(lock_effort)
        locked_ids = torch.nonzero(~self.unlocked, as_tuple=False).squeeze(-1)
        if locked_ids.numel() > 0:
            self.door.set_joint_effort_target(
                lock_effort[locked_ids].unsqueeze(-1),
                joint_ids=self.door_joint_ids,
                env_ids=locked_ids,
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.unlocked[env_ids] = False
        self.lock_effort[env_ids] = 0.0


@configclass
class DoorLatchActionCfg(ActionTermCfg):
    """Configuration for :class:`DoorLatchAction`."""

    class_type: type = DoorLatchAction
    door_joint_name: str = "door_hinge"
    handle_joint_name: str = "handle_joint"
    release_angle: float = 0.30
    lock_stiffness: float = 500.0
    lock_damping: float = 20.0
    maximum_lock_effort: float = 120.0


class CompliantGraspAction(ActionTerm):
    """Bidirectional Cartesian spring used as a calibrated grasp proxy.

    The current Z1 asset has a coarse single-DOF gripper and an uncalibrated
    TCP.  This term activates only after physical closure near the handle and
    projects the grasp force onto the handle and door generalized coordinates,
    while applying an equal reaction force at the robot TCP.  The robot
    therefore still feels the door load; unlike kinematic attachment, the door
    cannot follow without a reaction force on the base.
    """

    cfg: CompliantGraspActionCfg

    def __init__(self, cfg: CompliantGraspActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._env = env
        self.robot: Articulation = self._asset
        self.door: Articulation = env.scene[cfg.door_asset_name]
        self.tcp_body_idx = self.robot.find_bodies(cfg.tcp_body_name)[0][0]
        self.handle_body_idx = self.door.find_bodies(cfg.handle_body_name)[0][0]
        self.robot_force_body_idx = self.robot.find_bodies(cfg.robot_force_body_name)[0][0]
        self.door_panel_body_idx = self.door.find_bodies(cfg.door_force_body_name)[0][0]
        self.door_joint_idx = self.door.find_joints(cfg.door_joint_name)[0][0]
        self.handle_joint_idx = self.door.find_joints(cfg.handle_joint_name)[0][0]
        self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.relative_offset_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.spring_force_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.maximum_force = torch.zeros(self.num_envs, device=self.device)
        self.door_effort = torch.zeros(self.num_envs, device=self.device)
        self.handle_effort = torch.zeros(self.num_envs, device=self.device)
        self.maximum_door_effort = torch.zeros(self.num_envs, device=self.device)
        self._raw_actions = torch.empty((self.num_envs, 0), device=self.device)
        self._zero_torque = torch.zeros((self.num_envs, 1, 3), device=self.device)
        # Warp wraps torch memory without copying.  Keep the backing tensor
        # alive for the full term lifetime; an inline temporary caused only a
        # subset of replicated environments to receive the reaction wrench.
        self._env_ids_torch = torch.arange(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._env_ids_wp = wp.from_torch(self._env_ids_torch, dtype=wp.int32)
        self._robot_force_body_ids_wp = wp.array(
            [self.robot_force_body_idx], dtype=wp.int32, device=self.device
        )
        self._robot_force_position_w = torch.zeros((self.num_envs, 1, 3), device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"Expected zero-width compliant-grasp action, received {tuple(actions.shape)}")

    def _write_wrenches(self) -> None:
        robot_force = -self.spring_force_w.unsqueeze(1)
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            forces=wp.from_torch(robot_force, dtype=wp.vec3f),
            torques=wp.from_torch(self._zero_torque, dtype=wp.vec3f),
            positions=wp.from_torch(self._robot_force_position_w, dtype=wp.vec3f),
            body_ids=self._robot_force_body_ids_wp,
            env_ids=self._env_ids_wp,
            is_global=True,
        )

    def apply_actions(self) -> None:
        gripper_action = self._env.action_manager.get_term(self.cfg.gripper_action_name)
        tcp_pos_w = self.robot.data.body_pos_w[:, self.tcp_body_idx]
        handle_pos_w = self.door.data.body_pos_w[:, self.handle_body_idx]
        tcp_vel_w = self.robot.data.body_lin_vel_w[:, self.tcp_body_idx]
        handle_vel_w = self.door.data.body_lin_vel_w[:, self.handle_body_idx]

        separation = torch.linalg.vector_norm(handle_pos_w - tcp_pos_w, dim=-1)
        ready = (
            (gripper_action.closure_progress >= self.cfg.activation_closure_progress)
            & (separation <= self.cfg.activation_distance)
        )
        newly_active = ready & ~self.active
        self.relative_offset_w[newly_active] = handle_pos_w[newly_active] - tcp_pos_w[newly_active]
        self.active |= ready

        desired_handle_pos_w = tcp_pos_w + self.relative_offset_w
        position_error = desired_handle_pos_w - handle_pos_w
        constraint_error = torch.linalg.vector_norm(position_error, dim=-1)
        self.active &= constraint_error <= self.cfg.break_distance
        relative_velocity = tcp_vel_w - handle_vel_w
        force = self.cfg.stiffness * position_error + self.cfg.damping * relative_velocity
        magnitude = torch.linalg.vector_norm(force, dim=-1)
        scale = torch.clamp(self.cfg.maximum_force / torch.clamp(magnitude, min=1.0e-6), max=1.0)
        self.spring_force_w = force * scale.unsqueeze(-1) * self.active.unsqueeze(-1)
        self._robot_force_position_w[:, 0] = tcp_pos_w
        self.maximum_force = torch.maximum(
            self.maximum_force, torch.linalg.vector_norm(self.spring_force_w, dim=-1)
        )

        panel_pos_w = self.door.data.body_pos_w[:, self.door_panel_body_idx]
        panel_quat_w = self.door.data.body_quat_w[:, self.door_panel_body_idx]
        pivot_offset = torch.tensor(
            self.cfg.handle_pivot_panel,
            dtype=panel_pos_w.dtype,
            device=self.device,
        ).expand(self.num_envs, -1)
        handle_axis = torch.tensor(
            (1.0, 0.0, 0.0), dtype=panel_pos_w.dtype, device=self.device
        ).expand(self.num_envs, -1)
        handle_pivot_w = panel_pos_w + quat_apply(panel_quat_w, pivot_offset)
        handle_axis_w = quat_apply(panel_quat_w, handle_axis)
        handle_moment_w = torch.cross(
            handle_pos_w - handle_pivot_w, self.spring_force_w, dim=-1
        )
        handle_effort = torch.sum(handle_moment_w * handle_axis_w, dim=-1)
        handle_effort = torch.clamp(
            handle_effort, -self.cfg.maximum_handle_effort, self.cfg.maximum_handle_effort
        )
        self.handle_effort.copy_(handle_effort)
        self.door.set_joint_effort_target(
            handle_effort.unsqueeze(-1), joint_ids=[self.handle_joint_idx]
        )

        root_pos_w = self.door.data.root_pos_w
        root_quat_w = self.door.data.root_quat_w
        door_axis = torch.tensor(
            (0.0, 0.0, 1.0), dtype=root_pos_w.dtype, device=self.device
        ).expand(self.num_envs, -1)
        door_axis_w = quat_apply(root_quat_w, door_axis)
        door_moment_w = torch.cross(handle_pos_w - root_pos_w, self.spring_force_w, dim=-1)
        door_effort = torch.sum(door_moment_w * door_axis_w, dim=-1)
        door_effort = torch.clamp(
            door_effort, -self.cfg.maximum_door_effort, self.cfg.maximum_door_effort
        )
        self.door_effort.copy_(door_effort)
        self.maximum_door_effort = torch.maximum(
            self.maximum_door_effort, torch.abs(door_effort)
        )
        latch_action = self._env.action_manager.get_term(self.cfg.latch_action_name)
        combined_door_effort = torch.where(
            latch_action.unlocked, door_effort, latch_action.lock_effort
        )
        self.door.set_joint_effort_target(
            combined_door_effort.unsqueeze(-1), joint_ids=[self.door_joint_idx]
        )
        self._write_wrenches()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.active[env_ids] = False
        self.relative_offset_w[env_ids] = 0.0
        self.spring_force_w[env_ids] = 0.0
        self.maximum_force[env_ids] = 0.0
        self.door_effort[env_ids] = 0.0
        self.handle_effort[env_ids] = 0.0
        self.maximum_door_effort[env_ids] = 0.0
        self._write_wrenches()


@configclass
class CompliantGraspActionCfg(ActionTermCfg):
    """Configuration for a load-preserving compliant grasp."""

    class_type: type = CompliantGraspAction
    door_asset_name: str = "door"
    tcp_body_name: str = "tcp_frame"
    handle_body_name: str = "handle_grasp"
    robot_force_body_name: str = "gripper_stator"
    door_force_body_name: str = "door_panel"
    door_joint_name: str = "door_hinge"
    handle_joint_name: str = "handle_joint"
    latch_action_name: str = "door_latch"
    handle_pivot_panel: tuple[float, float, float] = (-0.055, 0.72, 1.05)
    gripper_action_name: str = "gripper_hold"
    activation_closure_progress: float = 0.95
    activation_distance: float = 0.06
    stiffness: float = 800.0
    damping: float = 40.0
    maximum_force: float = 80.0
    break_distance: float = 0.15
    maximum_door_effort: float = 120.0
    maximum_handle_effort: float = 20.0
