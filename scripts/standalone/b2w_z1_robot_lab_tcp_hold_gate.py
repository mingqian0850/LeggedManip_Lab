#!/usr/bin/env python3
"""Hold one world-frame B2-W + Z1 TCP pose while a frozen base policy moves.

The test deliberately separates the two controller rates:

* the public ``robot_lab`` B2-W policy runs at 50 Hz;
* Z1 differential IK and physics run at 200 Hz.

The articulation remains floating and all base motion is produced by the
policy's leg-position and wheel-velocity targets.  This script never writes a
root pose or root velocity.  After an initial physical settling period, the
current TCP pose is copied once and held fixed in world coordinates while the
base first translates about 5 cm and then yaws about 5 degrees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

_REPOSITORY = Path(__file__).resolve().parents[2]
_DEFAULT_CHECKPOINT = _REPOSITORY / "third_party/rl_sar_b2w/artifacts/policy.pt"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel physical environments.")
parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT, help="Pinned rl_sar TorchScript policy.")
parser.add_argument("--settle_steps", type=int, default=400, help="Pre-target physical settling steps at 200 Hz.")
parser.add_argument("--initial_hold_steps", type=int, default=100, help="Zero-command world-TCP hold steps.")
parser.add_argument("--forward_steps", type=int, default=300, help="Closed-loop forward-motion steps.")
parser.add_argument("--yaw_steps", type=int, default=400, help="Closed-loop yaw-motion steps.")
parser.add_argument("--terminal_hold_steps", type=int, default=200, help="Hold steps after each base motion.")
parser.add_argument("--window_steps", type=int, default=100, help="Terminal metric window length.")
parser.add_argument("--forward_distance", type=float, default=0.05, help="Desired physical base displacement in m.")
parser.add_argument("--yaw_deg", type=float, default=5.0, help="Desired physical base yaw in degrees.")
parser.add_argument("--position_gain", type=float, default=2.0, help="Base position-error to velocity gain.")
parser.add_argument("--yaw_gain", type=float, default=2.0, help="Base yaw-error to yaw-rate gain.")
parser.add_argument("--max_linear_command", type=float, default=0.20, help="Maximum |vx| command in m/s.")
parser.add_argument("--max_yaw_command", type=float, default=0.30, help="Maximum |wz| command in rad/s.")
parser.add_argument(
    "--forward_command_bias",
    type=float,
    default=0.0,
    help="Optional vx bias subtracted from the position loop; zero preserves the nominal policy.",
)
parser.add_argument(
    "--min_yaw_command",
    type=float,
    default=0.0,
    help="Optional documented yaw dead-zone compensation; zero leaves the nominal policy unchanged.",
)
parser.add_argument(
    "--yaw_stop_tolerance_deg",
    type=float,
    default=0.25,
    help="Yaw error below which optional dead-zone compensation is disabled.",
)
parser.add_argument("--ik_gain", type=float, default=1.0, help="Fraction of each DLS correction to apply.")
parser.add_argument("--max_joint_speed", type=float, default=3.0, help="Z1 joint-target rate limit in rad/s.")
parser.add_argument("--max_dynamic_position_rms", type=float, default=0.025, help="Moving TCP RMS limit in m.")
parser.add_argument(
    "--max_dynamic_orientation_rms_deg", type=float, default=6.0, help="Moving TCP orientation RMS limit."
)
parser.add_argument("--max_terminal_position_rms", type=float, default=0.012, help="Terminal TCP RMS limit in m.")
parser.add_argument(
    "--max_terminal_orientation_rms_deg", type=float, default=3.0, help="Terminal TCP orientation RMS limit."
)
parser.add_argument("--max_forward_error", type=float, default=0.025, help="Final base-x goal error limit in m.")
parser.add_argument("--max_yaw_error_deg", type=float, default=2.5, help="Final base-yaw goal error limit.")
parser.add_argument("--max_lateral_drift", type=float, default=0.025, help="Maximum post-target lateral drift in m.")
parser.add_argument("--min_base_height", type=float, default=0.35, help="Minimum allowed root height in m.")
parser.add_argument("--max_roll_pitch_deg", type=float, default=10.0, help="Maximum absolute root roll/pitch.")
parser.add_argument("--max_arm_home_error", type=float, default=0.10, help="Settled home-joint error limit in rad.")
parser.add_argument("--report", type=Path, default=None, help="Optional JSON report output path.")
parser.add_argument("--debug", action="store_true", help="Print early DIK state for controller diagnosis.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1:
    parser.error("--num_envs must be positive")
if (
    min(
        args_cli.settle_steps,
        args_cli.initial_hold_steps,
        args_cli.forward_steps,
        args_cli.yaw_steps,
        args_cli.terminal_hold_steps,
    )
    < 1
):
    parser.error("all phase lengths must be positive")
if args_cli.window_steps < 2 or args_cli.window_steps > args_cli.terminal_hold_steps:
    parser.error("--window_steps must be in [2, terminal_hold_steps]")
if (
    min(
        args_cli.forward_distance,
        args_cli.yaw_deg,
        args_cli.position_gain,
        args_cli.yaw_gain,
        args_cli.max_linear_command,
        args_cli.max_yaw_command,
        args_cli.ik_gain,
        args_cli.max_joint_speed,
    )
    <= 0.0
):
    parser.error("motion targets, gains, command limits, and IK limits must be positive")
if not 0.0 <= args_cli.min_yaw_command <= args_cli.max_yaw_command:
    parser.error("--min_yaw_command must be in [0, max_yaw_command]")
if abs(args_cli.forward_command_bias) > args_cli.max_linear_command:
    parser.error("absolute --forward_command_bias must not exceed --max_linear_command")
if args_cli.yaw_stop_tolerance_deg < 0.0:
    parser.error("--yaw_stop_tolerance_deg must be non-negative")
if args_cli.max_lateral_drift < 0.0 or args_cli.max_arm_home_error < 0.0:
    parser.error("lateral-drift and arm-home error limits must be non-negative")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402
from LeggedManip_Lab.assets.b2w_z1.b2w_z1_articulation_cfg import (  # noqa: E402
    B2W_Z1_ARM_HOME_JOINT_POS,
    B2W_Z1_ROBOT_LAB_POLICY_CFG,
    B2W_Z1_USD,
)
from LeggedManip_Lab.controllers.robot_lab_b2w_policy import (  # noqa: E402
    ROBOT_LAB_B2W_JOINT_NAMES,
    ROBOT_LAB_B2W_LEG_ACTION_SIZE,
    RobotLabB2WPolicy,
)

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    compute_pose_error,
    euler_xyz_from_quat,
    matrix_from_quat,
    quat_inv,
    subtract_frame_transforms,
)

PHYSICS_DT = 0.005
POLICY_DECIMATION = 4
ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))

CONTACT_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="average",
    restitution_combine_mode="average",
    static_friction=0.8,
    dynamic_friction=0.8,
    restitution=0.0,
)

ROBOT_CFG = B2W_Z1_ROBOT_LAB_POLICY_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=B2W_Z1_ROBOT_LAB_POLICY_CFG.init_state.replace(
        joint_pos={
            **B2W_Z1_ROBOT_LAB_POLICY_CFG.init_state.joint_pos,
            **B2W_Z1_ARM_HOME_JOINT_POS,
        }
    ),
)
ROBOT_CFG.spawn.physics_material = CONTACT_MATERIAL
for _actuator_cfg in ROBOT_CFG.actuators.values():
    if hasattr(_actuator_cfg, "min_delay"):
        _actuator_cfg.min_delay = 0
        _actuator_cfg.max_delay = 0


@configclass
class B2WZ1RobotLabTcpHoldSceneCfg(InteractiveSceneCfg):
    """Flat physical scene for the frozen-policy plus DIK gate."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(physics_material=CONTACT_MATERIAL),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0),
    )
    robot = ROBOT_CFG


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPOSITORY), *arguments], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _rms(values: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(values * values)))


def _per_environment_rms(values: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(values * values, dim=0))


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _pose_in_root(robot: Articulation, tcp_body_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    tcp_pose_w = robot.data.body_pose_w[:, tcp_body_id]
    return subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        tcp_pose_w[:, 0:3],
        tcp_pose_w[:, 3:7],
    )


def _target_in_root(
    robot: Articulation, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        target_pos_w,
        target_quat_w,
    )


def _jacobian_in_root(robot: Articulation, tcp_jacobian_id: int, arm_jacobian_columns: list[int]) -> torch.Tensor:
    jacobian = robot.root_physx_view.get_jacobians()[:, tcp_jacobian_id, :, arm_jacobian_columns].clone()
    root_rotation_inverse = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian[:, :3, :] = torch.bmm(root_rotation_inverse, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(root_rotation_inverse, jacobian[:, 3:, :])
    return jacobian


class CombinedController:
    """Own the 50 Hz frozen base policy and the 200 Hz Z1 DIK loop."""

    def __init__(self, sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
        self.sim = sim
        self.scene = scene
        self.robot: Articulation = scene["robot"]
        if self.robot.is_fixed_base:
            raise AssertionError("this gate requires a floating articulation")

        self.policy_joint_ids, policy_joint_names = self.robot.find_joints(
            list(ROBOT_LAB_B2W_JOINT_NAMES), preserve_order=True
        )
        if tuple(policy_joint_names) != ROBOT_LAB_B2W_JOINT_NAMES:
            raise AssertionError(f"unexpected policy joint order: {policy_joint_names}")
        self.leg_joint_ids = self.policy_joint_ids[:ROBOT_LAB_B2W_LEG_ACTION_SIZE]
        self.wheel_joint_ids = self.policy_joint_ids[ROBOT_LAB_B2W_LEG_ACTION_SIZE:]

        self.arm_joint_ids, arm_joint_names = self.robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        if tuple(arm_joint_names) != ARM_JOINT_NAMES:
            raise AssertionError(f"unexpected arm joint order: {arm_joint_names}")
        tcp_body_ids, tcp_body_names = self.robot.find_bodies("tcp_frame")
        if tcp_body_names != ["tcp_frame"]:
            raise AssertionError(f"unexpected TCP lookup: {tcp_body_names}")
        self.tcp_body_id = tcp_body_ids[0]
        self.tcp_jacobian_id = self.tcp_body_id
        self.arm_jacobian_columns = [joint_id + 6 for joint_id in self.arm_joint_ids]
        jacobian_shape = tuple(self.robot.root_physx_view.get_jacobians().shape)
        if jacobian_shape[1] != self.robot.num_bodies or jacobian_shape[3] != self.robot.num_joints + 6:
            raise AssertionError(
                f"unexpected floating Jacobian shape {jacobian_shape} for "
                f"{self.robot.num_bodies} bodies and {self.robot.num_joints} joints"
            )

        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = self.robot.data.default_joint_vel.clone()
        self.arm_joint_limits = self.robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, :].clone()
        self.policy = RobotLabB2WPolicy(
            args_cli.checkpoint,
            num_envs=args_cli.num_envs,
            device=sim.device,
        )
        expected_policy_q = self.policy.default_joint_position.expand(args_cli.num_envs, -1)
        actual_policy_q = self.default_joint_pos[:, self.policy_joint_ids]
        if not torch.allclose(actual_policy_q, expected_policy_q, atol=1.0e-7, rtol=0.0):
            raise AssertionError(
                "asset defaults do not match checkpoint defaults: "
                f"actual={actual_policy_q[0].tolist()} expected={expected_policy_q[0].tolist()}"
            )
        expected_arm_q = torch.tensor(
            [[B2W_Z1_ARM_HOME_JOINT_POS[name] for name in ARM_JOINT_NAMES]],
            device=sim.device,
            dtype=self.default_joint_pos.dtype,
        ).expand(args_cli.num_envs, -1)
        if not torch.allclose(self.default_joint_pos[:, self.arm_joint_ids], expected_arm_q, atol=1.0e-7, rtol=0.0):
            raise AssertionError("asset did not initialize with the requested Z1 home pose")

        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.03},
            ),
            args_cli.num_envs,
            sim.device,
        )
        self.physics_step_count = 0
        self.base_action = None
        self.arm_position_target = self.default_joint_pos[:, self.arm_joint_ids].clone()
        self.joint_limit_clamp_count = 0
        self.joint_rate_limit_count = 0
        self.minimum_jacobian_singular_value = math.inf
        self.maximum_arm_joint_speed = 0.0

    def initialize_joints(self) -> None:
        """Initialize only joint state; the floating root is never prescribed."""
        self.robot.write_joint_state_to_sim(self.default_joint_pos, self.default_joint_vel)
        self.robot.reset()
        self.policy.reset()

    def begin_world_tracking(self) -> None:
        """Start the DIK command state from the measured arm configuration."""
        self.arm_position_target.copy_(self.robot.data.joint_pos[:, self.arm_joint_ids])
        self.ik.reset()

    def _update_base_policy(self, velocity_command_b: torch.Tensor) -> None:
        policy_joint_pos = self.robot.data.joint_pos[:, self.policy_joint_ids]
        policy_joint_vel = self.robot.data.joint_vel[:, self.policy_joint_ids]
        self.base_action = self.policy.act(
            base_angular_velocity_b=self.robot.data.root_ang_vel_b,
            projected_gravity_b=self.robot.data.projected_gravity_b,
            velocity_command_b=velocity_command_b,
            joint_position=policy_joint_pos,
            joint_velocity=policy_joint_vel,
        )

    def _arm_target(self, target_pos_w: torch.Tensor | None, target_quat_w: torch.Tensor | None) -> torch.Tensor:
        if target_pos_w is None or target_quat_w is None:
            self.arm_position_target.copy_(self.default_joint_pos[:, self.arm_joint_ids])
            return self.arm_position_target

        current_pos_b, current_quat_b = _pose_in_root(self.robot, self.tcp_body_id)
        target_pos_b, target_quat_b = _target_in_root(self.robot, target_pos_w, target_quat_w)
        jacobian = _jacobian_in_root(self.robot, self.tcp_jacobian_id, self.arm_jacobian_columns)
        if not torch.all(torch.isfinite(jacobian)):
            raise FloatingPointError("TCP Jacobian contains NaN or Inf")
        self.minimum_jacobian_singular_value = min(
            self.minimum_jacobian_singular_value,
            float(torch.min(torch.linalg.svdvals(jacobian))),
        )

        current_joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
        self.ik.set_command(torch.cat((target_pos_b, target_quat_b), dim=-1))
        unconstrained_target = self.ik.compute(current_pos_b, current_quat_b, jacobian, current_joint_pos)
        if args_cli.debug and self.physics_step_count < args_cli.settle_steps + 8:
            position_error, orientation_error = compute_pose_error(
                current_pos_b,
                current_quat_b,
                target_pos_b,
                target_quat_b,
                rot_error_type="axis_angle",
            )
            print(
                f"DEBUG step={self.physics_step_count} pos_b={current_pos_b[0].tolist()} "
                f"target_pos_b={target_pos_b[0].tolist()} "
                f"pos_error={position_error[0].tolist()} rot_error={orientation_error[0].tolist()} "
                f"q={current_joint_pos[0].tolist()} q_ik={unconstrained_target[0].tolist()}",
                flush=True,
            )
        desired_target = current_joint_pos + args_cli.ik_gain * (unconstrained_target - current_joint_pos)
        unconstrained_delta = desired_target - self.arm_position_target
        max_joint_step = args_cli.max_joint_speed * PHYSICS_DT
        target_delta = torch.clamp(unconstrained_delta, -max_joint_step, max_joint_step)
        self.joint_rate_limit_count += int(torch.count_nonzero(torch.abs(target_delta - unconstrained_delta) > 1.0e-7))
        limited_target = self.arm_position_target + target_delta
        clamped_target = torch.maximum(
            torch.minimum(limited_target, self.arm_joint_limits[..., 1]),
            self.arm_joint_limits[..., 0],
        )
        self.joint_limit_clamp_count += int(torch.count_nonzero(torch.abs(clamped_target - limited_target) > 1.0e-7))
        self.arm_position_target.copy_(clamped_target)
        return self.arm_position_target

    def step(
        self,
        velocity_command_b: torch.Tensor,
        target_pos_w: torch.Tensor | None,
        target_quat_w: torch.Tensor | None,
    ) -> None:
        """Advance one 5 ms physics step without prescribing root state."""
        if self.physics_step_count % POLICY_DECIMATION == 0:
            self._update_base_policy(velocity_command_b)
        if self.base_action is None:
            raise AssertionError("base policy action was not initialized")
        arm_target = self._arm_target(target_pos_w, target_quat_w)

        self.robot.set_joint_position_target(self.default_joint_pos)
        self.robot.set_joint_velocity_target(self.default_joint_vel)
        self.robot.set_joint_position_target(self.base_action.leg_position_targets, joint_ids=self.leg_joint_ids)
        self.robot.set_joint_velocity_target(self.base_action.wheel_velocity_targets, joint_ids=self.wheel_joint_ids)
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_joint_ids)
        gravity_effort = self.robot.root_physx_view.get_gravity_compensation_forces()
        # PhysX prepends the floating base's six generalized coordinates.  The
        # articulation joint IDs used by Isaac Lab do not include that offset.
        self.robot.set_joint_effort_target(gravity_effort[:, self.arm_jacobian_columns], joint_ids=self.arm_joint_ids)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(PHYSICS_DT)
        self.physics_step_count += 1
        self.maximum_arm_joint_speed = max(
            self.maximum_arm_joint_speed,
            float(torch.max(torch.abs(self.robot.data.joint_vel[:, self.arm_joint_ids]))),
        )


def _relative_root_pose(robot: Articulation, reference_root_pose_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return subtract_frame_transforms(
        reference_root_pose_w[:, 0:3],
        reference_root_pose_w[:, 3:7],
        robot.data.root_pos_w,
        robot.data.root_quat_w,
    )


def _base_command(
    robot: Articulation,
    reference_root_pose_w: torch.Tensor,
    desired_x: float,
    desired_yaw: float,
) -> torch.Tensor:
    relative_pos, relative_quat = _relative_root_pose(robot, reference_root_pose_w)
    _, _, relative_yaw = euler_xyz_from_quat(relative_quat)
    command = torch.zeros((args_cli.num_envs, 3), device=robot.device, dtype=relative_pos.dtype)
    command[:, 0] = torch.clamp(
        args_cli.position_gain * (desired_x - relative_pos[:, 0]) - args_cli.forward_command_bias,
        -args_cli.max_linear_command,
        args_cli.max_linear_command,
    )
    yaw_error = _wrap_to_pi(desired_yaw - relative_yaw)
    yaw_command = args_cli.yaw_gain * yaw_error
    if args_cli.min_yaw_command > 0.0:
        yaw_active = torch.abs(yaw_error) > math.radians(args_cli.yaw_stop_tolerance_deg)
        yaw_command = torch.where(
            yaw_active,
            torch.sign(yaw_command)
            * torch.maximum(
                torch.abs(yaw_command),
                torch.full_like(yaw_command, args_cli.min_yaw_command),
            ),
            torch.zeros_like(yaw_command),
        )
    command[:, 2] = torch.clamp(yaw_command, -args_cli.max_yaw_command, args_cli.max_yaw_command)
    return command


def _sample(
    controller: CombinedController,
    reference_root_pose_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    command: torch.Tensor,
) -> dict[str, torch.Tensor | bool]:
    robot = controller.robot
    tcp_pose_w = robot.data.body_pose_w[:, controller.tcp_body_id]
    position_error, orientation_error = compute_pose_error(
        tcp_pose_w[:, 0:3],
        tcp_pose_w[:, 3:7],
        target_pos_w,
        target_quat_w,
        rot_error_type="axis_angle",
    )
    relative_pos, relative_quat = _relative_root_pose(robot, reference_root_pose_w)
    relative_roll, relative_pitch, relative_yaw = euler_xyz_from_quat(relative_quat)
    root_roll, root_pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    local_root_pos = robot.data.root_pos_w - controller.scene.env_origins
    finite_tensors = (
        robot.data.root_state_w,
        robot.data.joint_pos,
        tcp_pose_w,
        position_error,
        orientation_error,
        command,
    )
    return {
        "position_error": torch.linalg.vector_norm(position_error, dim=-1),
        "orientation_error": torch.linalg.vector_norm(orientation_error, dim=-1),
        "relative_root_position": relative_pos.clone(),
        "relative_root_roll": relative_roll.clone(),
        "relative_root_pitch": relative_pitch.clone(),
        "relative_root_yaw": relative_yaw.clone(),
        "root_roll": root_roll.clone(),
        "root_pitch": root_pitch.clone(),
        "root_height": local_root_pos[:, 2].clone(),
        "command": command.clone(),
        "finite": all(bool(torch.all(torch.isfinite(tensor))) for tensor in finite_tensors),
    }


def _phase_metrics(name: str, samples: list[dict[str, torch.Tensor | bool]]) -> dict[str, object]:
    if not samples:
        raise ValueError(f"phase {name} contains no samples")
    position_error = torch.stack([sample["position_error"] for sample in samples])  # type: ignore[list-item]
    orientation_error = torch.stack([sample["orientation_error"] for sample in samples])  # type: ignore[list-item]
    relative_position = torch.stack([sample["relative_root_position"] for sample in samples])  # type: ignore[list-item]
    relative_yaw = torch.stack([sample["relative_root_yaw"] for sample in samples])  # type: ignore[list-item]
    root_height = torch.stack([sample["root_height"] for sample in samples])  # type: ignore[list-item]
    root_roll = torch.stack([sample["root_roll"] for sample in samples])  # type: ignore[list-item]
    root_pitch = torch.stack([sample["root_pitch"] for sample in samples])  # type: ignore[list-item]
    commands = torch.stack([sample["command"] for sample in samples])  # type: ignore[list-item]
    terminal_count = min(args_cli.window_steps, len(samples))
    terminal_position = position_error[-terminal_count:]
    terminal_orientation = orientation_error[-terminal_count:]
    roll_pitch = torch.maximum(torch.abs(root_roll), torch.abs(root_pitch))
    return {
        "name": name,
        "steps": len(samples),
        "duration_s": len(samples) * PHYSICS_DT,
        "tcp_position_error_rms_m": _rms(position_error),
        "tcp_position_error_rms_worst_env_m": float(torch.max(_per_environment_rms(position_error))),
        "tcp_position_error_p95_m": float(torch.quantile(position_error.flatten(), 0.95)),
        "tcp_position_error_max_m": float(torch.max(position_error)),
        "tcp_orientation_error_rms_deg": math.degrees(_rms(orientation_error)),
        "tcp_orientation_error_rms_worst_env_deg": math.degrees(
            float(torch.max(_per_environment_rms(orientation_error)))
        ),
        "tcp_orientation_error_p95_deg": math.degrees(float(torch.quantile(orientation_error.flatten(), 0.95))),
        "tcp_orientation_error_max_deg": math.degrees(float(torch.max(orientation_error))),
        "terminal_tcp_position_error_rms_worst_env_m": float(torch.max(_per_environment_rms(terminal_position))),
        "terminal_tcp_orientation_error_rms_worst_env_deg": math.degrees(
            float(torch.max(_per_environment_rms(terminal_orientation)))
        ),
        "base_relative_position_final_m": relative_position[-1].tolist(),
        "base_relative_yaw_final_deg": [math.degrees(float(value)) for value in relative_yaw[-1]],
        "base_lateral_displacement_abs_max_m": float(torch.max(torch.abs(relative_position[..., 1]))),
        "base_height_min_m": float(torch.min(root_height)),
        "base_roll_pitch_abs_max_deg": math.degrees(float(torch.max(roll_pitch))),
        "command_abs_max": torch.max(torch.abs(commands), dim=0).values.tolist(),
        "nan_detected": not all(bool(sample["finite"]) for sample in samples),
    }


def _run_phase(
    *,
    name: str,
    steps: int,
    controller: CombinedController,
    reference_root_pose_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    desired_x: float,
    desired_yaw: float,
    zero_command: bool = False,
) -> tuple[dict[str, object], list[dict[str, torch.Tensor | bool]]]:
    samples: list[dict[str, torch.Tensor | bool]] = []
    for _ in range(steps):
        if zero_command:
            command = torch.zeros((args_cli.num_envs, 3), device=controller.sim.device)
        else:
            command = _base_command(controller.robot, reference_root_pose_w, desired_x, desired_yaw)
        controller.step(command, target_pos_w, target_quat_w)
        samples.append(
            _sample(
                controller,
                reference_root_pose_w,
                target_pos_w,
                target_quat_w,
                command,
            )
        )
        if not bool(samples[-1]["finite"]):
            break
    metrics = _phase_metrics(name, samples)
    print(
        f"{name}: TCP={1000.0 * metrics['tcp_position_error_rms_worst_env_m']:.2f} mm/"
        f"{metrics['tcp_orientation_error_rms_worst_env_deg']:.2f} deg, "
        f"base={metrics['base_relative_position_final_m'][0][0]:.3f} m/"
        f"{metrics['base_relative_yaw_final_deg'][0]:.2f} deg",
        flush=True,
    )
    return metrics, samples


def _provenance() -> dict[str, object]:
    script_path = Path(__file__).resolve()
    checkpoint_path = args_cli.checkpoint.expanduser().resolve()
    return {
        "command": [sys.executable, *sys.argv],
        "git_head": _git_value("rev-parse", "HEAD"),
        "relevant_git_status": _git_value(
            "status", "--short", "--", str(script_path.relative_to(_REPOSITORY))
        ).splitlines(),
        "script_path": str(script_path.relative_to(_REPOSITORY)),
        "script_sha256": _sha256(script_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "asset_path": str(Path(B2W_Z1_USD).resolve().relative_to(_REPOSITORY)),
        "asset_sha256": _sha256(Path(B2W_Z1_USD)),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None,
    }


def run_gate() -> dict[str, object]:
    """Run all physical phases and return a JSON-serializable report."""
    start_time = time.perf_counter()
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=PHYSICS_DT,
            device=args_cli.device,
            physx=sim_utils.PhysxCfg(
                min_velocity_iteration_count=1,
                enable_external_forces_every_iteration=True,
            ),
        )
    )
    scene = InteractiveScene(B2WZ1RobotLabTcpHoldSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0))
    sim.reset()
    scene.update(PHYSICS_DT)
    controller = CombinedController(sim, scene)
    pre_settle_root_pose_w = controller.robot.data.root_pose_w.clone()
    controller.initialize_joints()

    zero_command = torch.zeros((args_cli.num_envs, 3), device=sim.device)
    settle_heights: list[torch.Tensor] = []
    settle_attitudes: list[torch.Tensor] = []
    for _ in range(args_cli.settle_steps):
        controller.step(zero_command, None, None)
        root_roll, root_pitch, _ = euler_xyz_from_quat(controller.robot.data.root_quat_w)
        settle_heights.append((controller.robot.data.root_pos_w - controller.scene.env_origins)[:, 2].clone())
        settle_attitudes.append(torch.maximum(torch.abs(root_roll), torch.abs(root_pitch)))
    settling_finite = all(
        bool(torch.all(torch.isfinite(tensor)))
        for tensor in (
            controller.robot.data.root_state_w,
            controller.robot.data.joint_pos,
            controller.robot.data.joint_vel,
            controller.robot.data.body_pose_w[:, controller.tcp_body_id],
        )
    )
    if not settling_finite:
        raise FloatingPointError("settling produced a non-finite robot state")

    reference_root_pose_w = controller.robot.data.root_pose_w.clone()
    settle_relative_pos, settle_relative_quat = subtract_frame_transforms(
        pre_settle_root_pose_w[:, 0:3],
        pre_settle_root_pose_w[:, 3:7],
        reference_root_pose_w[:, 0:3],
        reference_root_pose_w[:, 3:7],
    )
    _, _, settle_relative_yaw = euler_xyz_from_quat(settle_relative_quat)
    settled_arm_position = controller.robot.data.joint_pos[:, controller.arm_joint_ids]
    requested_home_position = controller.default_joint_pos[:, controller.arm_joint_ids]
    settling = {
        "steps": args_cli.settle_steps,
        "duration_s": args_cli.settle_steps * PHYSICS_DT,
        "base_relative_position_final_m": settle_relative_pos.tolist(),
        "base_relative_yaw_final_deg": [math.degrees(float(value)) for value in settle_relative_yaw],
        "base_forward_drift_abs_max_m": float(torch.max(torch.abs(settle_relative_pos[:, 0]))),
        "base_lateral_drift_abs_max_m": float(torch.max(torch.abs(settle_relative_pos[:, 1]))),
        "base_height_min_m": float(torch.min(torch.stack(settle_heights))),
        "base_roll_pitch_abs_max_deg": math.degrees(float(torch.max(torch.stack(settle_attitudes)))),
        "arm_home_joint_error_rms_rad": _rms(settled_arm_position - requested_home_position),
        "arm_home_joint_error_abs_max_rad": float(torch.max(torch.abs(settled_arm_position - requested_home_position))),
        "nan_detected": not settling_finite,
    }
    initial_tcp_pose_w = controller.robot.data.body_pose_w[:, controller.tcp_body_id].clone()
    target_pos_w = initial_tcp_pose_w[:, 0:3].clone()
    target_quat_w = initial_tcp_pose_w[:, 3:7].clone()
    frozen_target = torch.cat((target_pos_w, target_quat_w), dim=-1).clone()
    controller.begin_world_tracking()

    initial_hold, _ = _run_phase(
        name="initial_world_tcp_hold",
        steps=args_cli.initial_hold_steps,
        controller=controller,
        reference_root_pose_w=reference_root_pose_w,
        target_pos_w=target_pos_w,
        target_quat_w=target_quat_w,
        desired_x=0.0,
        desired_yaw=0.0,
        zero_command=True,
    )
    forward_motion, _ = _run_phase(
        name="forward_motion",
        steps=args_cli.forward_steps,
        controller=controller,
        reference_root_pose_w=reference_root_pose_w,
        target_pos_w=target_pos_w,
        target_quat_w=target_quat_w,
        desired_x=args_cli.forward_distance,
        desired_yaw=0.0,
    )
    forward_terminal, _ = _run_phase(
        name="forward_terminal_hold",
        steps=args_cli.terminal_hold_steps,
        controller=controller,
        reference_root_pose_w=reference_root_pose_w,
        target_pos_w=target_pos_w,
        target_quat_w=target_quat_w,
        desired_x=args_cli.forward_distance,
        desired_yaw=0.0,
    )
    desired_yaw = math.radians(args_cli.yaw_deg)
    yaw_motion, _ = _run_phase(
        name="yaw_motion",
        steps=args_cli.yaw_steps,
        controller=controller,
        reference_root_pose_w=reference_root_pose_w,
        target_pos_w=target_pos_w,
        target_quat_w=target_quat_w,
        desired_x=args_cli.forward_distance,
        desired_yaw=desired_yaw,
    )
    yaw_terminal, _ = _run_phase(
        name="yaw_terminal_hold",
        steps=args_cli.terminal_hold_steps,
        controller=controller,
        reference_root_pose_w=reference_root_pose_w,
        target_pos_w=target_pos_w,
        target_quat_w=target_quat_w,
        desired_x=args_cli.forward_distance,
        desired_yaw=desired_yaw,
    )

    target_immutable = bool(torch.equal(torch.cat((target_pos_w, target_quat_w), dim=-1), frozen_target))
    final_relative_pos, final_relative_quat = _relative_root_pose(controller.robot, reference_root_pose_w)
    _, _, final_relative_yaw = euler_xyz_from_quat(final_relative_quat)
    final_forward_error = torch.abs(final_relative_pos[:, 0] - args_cli.forward_distance)
    final_yaw_error = torch.abs(_wrap_to_pi(final_relative_yaw - desired_yaw))
    phases = {
        "initial_world_tcp_hold": initial_hold,
        "forward_motion": forward_motion,
        "forward_terminal_hold": forward_terminal,
        "yaw_motion": yaw_motion,
        "yaw_terminal_hold": yaw_terminal,
    }
    motion_position_rms = max(
        float(initial_hold["tcp_position_error_rms_worst_env_m"]),
        float(forward_motion["tcp_position_error_rms_worst_env_m"]),
        float(yaw_motion["tcp_position_error_rms_worst_env_m"]),
    )
    motion_orientation_rms = max(
        float(initial_hold["tcp_orientation_error_rms_worst_env_deg"]),
        float(forward_motion["tcp_orientation_error_rms_worst_env_deg"]),
        float(yaw_motion["tcp_orientation_error_rms_worst_env_deg"]),
    )
    terminal_position_rms = max(
        float(forward_terminal["terminal_tcp_position_error_rms_worst_env_m"]),
        float(yaw_terminal["terminal_tcp_position_error_rms_worst_env_m"]),
    )
    terminal_orientation_rms = max(
        float(forward_terminal["terminal_tcp_orientation_error_rms_worst_env_deg"]),
        float(yaw_terminal["terminal_tcp_orientation_error_rms_worst_env_deg"]),
    )
    maximum_lateral_drift = max(float(phase["base_lateral_displacement_abs_max_m"]) for phase in phases.values())
    minimum_height = min(
        float(settling["base_height_min_m"]),
        *(float(phase["base_height_min_m"]) for phase in phases.values()),
    )
    maximum_attitude = max(
        float(settling["base_roll_pitch_abs_max_deg"]),
        *(float(phase["base_roll_pitch_abs_max_deg"]) for phase in phases.values()),
    )
    nan_detected = bool(settling["nan_detected"]) or any(bool(phase["nan_detected"]) for phase in phases.values())
    checks = {
        "no_nan": not nan_detected,
        "world_target_immutable": target_immutable,
        "moving_tcp_position": motion_position_rms <= args_cli.max_dynamic_position_rms,
        "moving_tcp_orientation": motion_orientation_rms <= args_cli.max_dynamic_orientation_rms_deg,
        "terminal_tcp_position": terminal_position_rms <= args_cli.max_terminal_position_rms,
        "terminal_tcp_orientation": terminal_orientation_rms <= args_cli.max_terminal_orientation_rms_deg,
        "forward_goal": float(torch.max(final_forward_error)) <= args_cli.max_forward_error,
        "yaw_goal": math.degrees(float(torch.max(final_yaw_error))) <= args_cli.max_yaw_error_deg,
        "base_lateral_drift": maximum_lateral_drift <= args_cli.max_lateral_drift,
        "base_height": minimum_height >= args_cli.min_base_height,
        "base_attitude": maximum_attitude <= args_cli.max_roll_pitch_deg,
        "arm_home_pose": float(settling["arm_home_joint_error_abs_max_rad"]) <= args_cli.max_arm_home_error,
        "arm_joint_limits": controller.joint_limit_clamp_count == 0,
        "arm_joint_speed": controller.maximum_arm_joint_speed <= args_cli.max_joint_speed * 1.05,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "provenance": _provenance(),
        "configuration": {
            "num_envs": args_cli.num_envs,
            "physics_dt_s": PHYSICS_DT,
            "physics_frequency_hz": 1.0 / PHYSICS_DT,
            "policy_decimation": POLICY_DECIMATION,
            "policy_frequency_hz": 1.0 / (PHYSICS_DT * POLICY_DECIMATION),
            "root_state_writes": 0,
            "arm_reset_pose": "B2W_Z1_ARM_HOME_JOINT_POS",
            "forward_goal_m": args_cli.forward_distance,
            "yaw_goal_deg": args_cli.yaw_deg,
            "base_command_components": ["vx", "wz"],
            "lateral_velocity_command": 0.0,
            "forward_command_bias_mps": args_cli.forward_command_bias,
            "contact_material": {
                "friction_combine_mode": "average",
                "static_friction": 0.8,
                "dynamic_friction": 0.8,
                "restitution": 0.0,
            },
            "min_yaw_command_radps": args_cli.min_yaw_command,
            "yaw_stop_tolerance_deg": args_cli.yaw_stop_tolerance_deg,
            "world_target_pose_w": frozen_target.tolist(),
        },
        "joint_mapping": {
            "policy_joint_names": list(ROBOT_LAB_B2W_JOINT_NAMES),
            "policy_joint_ids": controller.policy_joint_ids,
            "leg_joint_ids": controller.leg_joint_ids,
            "wheel_joint_ids": controller.wheel_joint_ids,
            "arm_joint_names": list(ARM_JOINT_NAMES),
            "arm_joint_ids": controller.arm_joint_ids,
            "arm_jacobian_columns": controller.arm_jacobian_columns,
            "tcp_body_id": controller.tcp_body_id,
            "tcp_jacobian_id": controller.tcp_jacobian_id,
        },
        "thresholds": {
            "moving_tcp_position_rms_worst_env_m": args_cli.max_dynamic_position_rms,
            "moving_tcp_orientation_rms_worst_env_deg": args_cli.max_dynamic_orientation_rms_deg,
            "terminal_tcp_position_rms_worst_env_m": args_cli.max_terminal_position_rms,
            "terminal_tcp_orientation_rms_worst_env_deg": args_cli.max_terminal_orientation_rms_deg,
            "final_base_forward_error_max_m": args_cli.max_forward_error,
            "final_base_yaw_error_max_deg": args_cli.max_yaw_error_deg,
            "base_lateral_drift_max_m": args_cli.max_lateral_drift,
            "base_height_min_m": args_cli.min_base_height,
            "base_roll_pitch_abs_max_deg": args_cli.max_roll_pitch_deg,
            "arm_home_joint_error_abs_max_rad": args_cli.max_arm_home_error,
            "arm_joint_speed_abs_max_radps": args_cli.max_joint_speed * 1.05,
            "joint_limit_clamp_count": 0,
        },
        "settling": settling,
        "phases": phases,
        "summary": {
            "moving_tcp_position_rms_worst_env_m": motion_position_rms,
            "moving_tcp_orientation_rms_worst_env_deg": motion_orientation_rms,
            "terminal_tcp_position_rms_worst_env_m": terminal_position_rms,
            "terminal_tcp_orientation_rms_worst_env_deg": terminal_orientation_rms,
            "final_base_relative_position_m": final_relative_pos.tolist(),
            "final_base_relative_yaw_deg": [math.degrees(float(value)) for value in final_relative_yaw],
            "final_forward_error_max_m": float(torch.max(final_forward_error)),
            "final_yaw_error_max_deg": math.degrees(float(torch.max(final_yaw_error))),
            "base_lateral_drift_abs_max_m": maximum_lateral_drift,
            "base_height_min_m": minimum_height,
            "base_roll_pitch_abs_max_deg": maximum_attitude,
            "nan_detected": nan_detected,
            "target_immutable": target_immutable,
            "joint_limit_clamp_count": controller.joint_limit_clamp_count,
            "joint_target_rate_limit_count": controller.joint_rate_limit_count,
            "minimum_jacobian_singular_value": controller.minimum_jacobian_singular_value,
            "maximum_arm_joint_speed_radps": controller.maximum_arm_joint_speed,
        },
        "elapsed_wall_time_s": time.perf_counter() - start_time,
    }


def main() -> int:
    report = run_gate()
    report_text = json.dumps(report, indent=2, sort_keys=True)
    print(report_text, flush=True)
    if args_cli.report is not None:
        report_path = args_cli.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text + "\n", encoding="utf-8")
    print(
        f"{'PASS' if report['passed'] else 'FAIL'}: frozen B2-W policy plus 200 Hz world-TCP hold",
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
