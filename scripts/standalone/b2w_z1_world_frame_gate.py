#!/usr/bin/env python3
"""Validate smooth world-frame TCP tracking while the B2-W root moves.

This is a coordinate/controller gate, not a wheel-dynamics test. The B2-W
root remains a floating articulation so PhysX exposes the floating-base
Jacobian, but a smooth root trajectory is written directly to the simulator.
That isolates three failure-prone details before wheel control is introduced:

* the final TCP goal is stored once in world coordinates;
* every control step re-expresses that same goal in the current root frame;
* the Z1 Jacobian uses the floating-base body row and ``joint_id + 6`` columns.

The gate first tracks a minimum-jerk 6D TCP reference with a stationary root.
It then holds the final world pose while the root translates, changes height,
and pitches. Both signs are exercised across replicated environments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1, help="Number of replicated gate environments.")
parser.add_argument("--settle_steps", type=int, default=100, help="Initial joint settling steps.")
parser.add_argument("--target_motion_steps", type=int, default=400, help="Steps in the minimum-jerk TCP motion.")
parser.add_argument("--target_hold_steps", type=int, default=500, help="Steps holding the final TCP target.")
parser.add_argument("--base_motion_steps", type=int, default=400, help="Steps in each root outbound/return motion.")
parser.add_argument("--base_hold_steps", type=int, default=300, help="Steps at each root-motion endpoint.")
parser.add_argument("--window_steps", type=int, default=100, help="Terminal metric window length.")
parser.add_argument(
    "--initial_yaw_deg", type=float, default=10.0, help="Alternating initial base-yaw magnitude."
)
parser.add_argument("--ik_gain", type=float, default=1.0, help="Per-step gain applied to the DLS correction.")
parser.add_argument("--max_joint_speed", type=float, default=3.0, help="Joint-target rate limit in rad/s.")
parser.add_argument("--max_dynamic_position_rms", type=float, default=0.010, help="Moving RMS position limit in m.")
parser.add_argument(
    "--max_dynamic_orientation_rms_deg", type=float, default=5.0, help="Moving RMS orientation limit."
)
parser.add_argument("--max_terminal_position_rms", type=float, default=0.005, help="Held RMS position limit in m.")
parser.add_argument(
    "--max_terminal_orientation_rms_deg", type=float, default=2.0, help="Held RMS orientation limit."
)
parser.add_argument("--max_terminal_jitter", type=float, default=0.002, help="Held TCP position jitter limit in m.")
parser.add_argument("--max_terminal_linear_speed", type=float, default=0.03, help="Held TCP RMS speed in m/s.")
parser.add_argument("--max_terminal_angular_speed", type=float, default=0.15, help="Held TCP RMS speed in rad/s.")
parser.add_argument("--max_root_position_error", type=float, default=0.005, help="Root write error limit in m.")
parser.add_argument(
    "--max_root_orientation_error_deg", type=float, default=0.5, help="Root write orientation error limit."
)
parser.add_argument("--report", type=Path, default=None, help="Optional JSON report output path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1:
    parser.error("--num_envs must be positive")
if args_cli.window_steps < 2:
    parser.error("--window_steps must be at least two")
if args_cli.window_steps > min(args_cli.target_hold_steps, args_cli.base_hold_steps):
    parser.error("--window_steps must fit inside both hold periods")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    apply_delta_pose,
    compute_pose_error,
    matrix_from_quat,
    quat_inv,
    subtract_frame_transforms,
)

from LeggedManip_Lab.assets.b2w_z1.b2w_z1_articulation_cfg import (  # noqa: E402
    B2W_Z1_TRAINING_CFG,
    B2W_Z1_USD,
)


TCP_GOAL_DELTA = (0.030, 0.020, 0.020, math.radians(3.0), math.radians(-3.0), math.radians(5.0))
BASE_EXCITATIONS = (
    ("base_longitudinal", (0.050, 0.000, 0.000, 0.0, 0.0, 0.0)),
    ("base_lateral", (0.000, 0.040, 0.000, 0.0, 0.0, 0.0)),
    ("base_height", (0.000, 0.000, -0.035, 0.0, 0.0, 0.0)),
    ("base_roll", (0.000, 0.000, 0.000, math.radians(4.0), 0.0, 0.0)),
    ("base_pitch", (0.000, 0.000, 0.000, 0.0, math.radians(5.0), 0.0)),
    ("base_yaw", (0.000, 0.000, 0.000, 0.0, 0.0, math.radians(5.0))),
    ("base_combined", (-0.035, 0.000, 0.020, 0.0, math.radians(-4.0), math.radians(3.0))),
)


@configclass
class B2WZ1WorldFrameGateSceneCfg(InteractiveSceneCfg):
    """Replicated collision-free scene for the world-frame semantics gate."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0),
    )
    robot = B2W_Z1_TRAINING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _rms(values: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(values * values)))


def _p95(values: torch.Tensor) -> float:
    return float(torch.quantile(values.flatten(), 0.95))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance() -> dict[str, object]:
    script_path = Path(__file__).resolve()
    repository = script_path.parents[2]
    tracked_paths = [
        str(script_path.relative_to(repository)),
        str(Path(B2W_Z1_USD).resolve().relative_to(repository)),
    ]
    try:
        git_head = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        relevant_status = subprocess.check_output(
            ["git", "-C", str(repository), "status", "--short", "--", *tracked_paths],
            text=True,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError, ValueError):
        git_head = "unavailable"
        relevant_status = ["unavailable"]
    return {
        "command": [sys.executable, *sys.argv],
        "git_head": git_head,
        "relevant_git_status": relevant_status,
        "script_path": str(script_path.relative_to(repository)),
        "script_sha256": _sha256(script_path),
        "asset_path": tracked_paths[1],
        "asset_sha256": _sha256(Path(B2W_Z1_USD)),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else None,
    }


def _per_environment_rms(values: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(values * values, dim=0))


def _sign_group_metrics(
    position_error_history: torch.Tensor,
    orientation_error_history: torch.Tensor,
    signs: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, dict[str, float | int]] = {}
    for label, mask in (("positive", signs > 0.0), ("negative", signs < 0.0)):
        if not torch.any(mask):
            continue
        position_group = position_error_history[:, mask]
        orientation_group = orientation_error_history[:, mask]
        groups[label] = {
            "environment_count": int(torch.count_nonzero(mask)),
            "position_error_rms_m": _rms(position_group),
            "position_error_rms_worst_env_m": float(
                torch.max(_per_environment_rms(position_group))
            ),
            "orientation_error_rms_deg": math.degrees(_rms(orientation_group)),
            "orientation_error_rms_worst_env_deg": math.degrees(
                float(torch.max(_per_environment_rms(orientation_group)))
            ),
        }
    return groups


def _minimum_jerk(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5


def _signed_delta(values: tuple[float, ...], signs: torch.Tensor, scale: float) -> torch.Tensor:
    delta = torch.tensor(values, device=signs.device, dtype=torch.float32).repeat(signs.shape[0], 1)
    return delta * signs.unsqueeze(-1) * scale


def _pose_in_root(robot: Articulation, tcp_body_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    tcp_pose_w = robot.data.body_pose_w[:, tcp_body_id]
    root_pose_w = robot.data.root_pose_w
    return subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        tcp_pose_w[:, 0:3],
        tcp_pose_w[:, 3:7],
    )


def _target_in_root(
    root_pose_w: torch.Tensor, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        target_pos_w,
        target_quat_w,
    )


def _jacobian_in_root(
    robot: Articulation, tcp_jacobian_id: int, arm_jacobian_cols: list[int]
) -> torch.Tensor:
    jacobian = robot.root_physx_view.get_jacobians()[
        :, tcp_jacobian_id, :, arm_jacobian_cols
    ].clone()
    root_rotation_inverse = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian[:, :3, :] = torch.bmm(root_rotation_inverse, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(root_rotation_inverse, jacobian[:, 3:, :])
    return jacobian


def _window_metrics(
    position_errors: list[torch.Tensor],
    orientation_errors: list[torch.Tensor],
    positions: list[torch.Tensor],
    linear_speeds: list[torch.Tensor],
    angular_speeds: list[torch.Tensor],
    signs: torch.Tensor,
) -> dict[str, object]:
    position_error_history = torch.stack(position_errors[-args_cli.window_steps :])
    orientation_error_history = torch.stack(orientation_errors[-args_cli.window_steps :])
    position_history = torch.stack(positions[-args_cli.window_steps :])
    linear_speed_history = torch.stack(linear_speeds[-args_cli.window_steps :])
    angular_speed_history = torch.stack(angular_speeds[-args_cli.window_steps :])
    position_center = torch.mean(position_history, dim=0, keepdim=True)
    jitter = torch.sqrt(torch.mean(torch.sum((position_history - position_center) ** 2, dim=-1), dim=0))
    return {
        "position_error_rms_m": _rms(position_error_history),
        "position_error_rms_worst_env_m": float(
            torch.max(_per_environment_rms(position_error_history))
        ),
        "position_error_p95_m": _p95(position_error_history),
        "position_error_max_m": float(torch.max(position_error_history)),
        "orientation_error_rms_deg": math.degrees(_rms(orientation_error_history)),
        "orientation_error_rms_worst_env_deg": math.degrees(
            float(torch.max(_per_environment_rms(orientation_error_history)))
        ),
        "orientation_error_p95_deg": math.degrees(_p95(orientation_error_history)),
        "orientation_error_max_deg": math.degrees(float(torch.max(orientation_error_history))),
        "position_jitter_max_m": float(torch.max(jitter)),
        "linear_speed_rms_worst_env_mps": float(
            torch.max(_per_environment_rms(linear_speed_history))
        ),
        "angular_speed_rms_worst_env_radps": float(
            torch.max(_per_environment_rms(angular_speed_history))
        ),
        "by_sign": _sign_group_metrics(position_error_history, orientation_error_history, signs),
    }


def _history_metrics(
    position_errors: list[torch.Tensor], orientation_errors: list[torch.Tensor], signs: torch.Tensor
) -> dict[str, object]:
    position_error_history = torch.stack(position_errors)
    orientation_error_history = torch.stack(orientation_errors)
    return {
        "position_error_rms_m": _rms(position_error_history),
        "position_error_rms_worst_env_m": float(
            torch.max(_per_environment_rms(position_error_history))
        ),
        "position_error_p95_m": _p95(position_error_history),
        "position_error_max_m": float(torch.max(position_error_history)),
        "orientation_error_rms_deg": math.degrees(_rms(orientation_error_history)),
        "orientation_error_rms_worst_env_deg": math.degrees(
            float(torch.max(_per_environment_rms(orientation_error_history)))
        ),
        "orientation_error_p95_deg": math.degrees(_p95(orientation_error_history)),
        "orientation_error_max_deg": math.degrees(float(torch.max(orientation_error_history))),
        "by_sign": _sign_group_metrics(position_error_history, orientation_error_history, signs),
    }


def _step_controller(
    *,
    robot: Articulation,
    sim: sim_utils.SimulationContext,
    controller: DifferentialIKController,
    arm_joint_ids: list[int],
    arm_jacobian_cols: list[int],
    tcp_body_id: int,
    tcp_jacobian_id: int,
    default_joint_pos: torch.Tensor,
    default_joint_vel: torch.Tensor,
    joint_limits: torch.Tensor,
    desired_root_pose_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    float,
    float,
]:
    for label, tensor in (
        ("desired root pose", desired_root_pose_w),
        ("world target position", target_pos_w),
        ("world target orientation", target_quat_w),
    ):
        if not torch.all(torch.isfinite(tensor)):
            raise AssertionError(f"{label} contains NaN or Inf")
    current_pos_b, current_quat_b = _pose_in_root(robot, tcp_body_id)
    target_pos_b, target_quat_b = _target_in_root(robot.data.root_pose_w, target_pos_w, target_quat_w)
    jacobian = _jacobian_in_root(robot, tcp_jacobian_id, arm_jacobian_cols)
    if not torch.all(torch.isfinite(jacobian)):
        raise AssertionError("TCP Jacobian contains NaN or Inf")

    current_joint_pos = robot.data.joint_pos[:, arm_joint_ids]
    controller.set_command(torch.cat((target_pos_b, target_quat_b), dim=-1))
    unconstrained_target = controller.compute(current_pos_b, current_quat_b, jacobian, current_joint_pos)
    max_joint_step = args_cli.max_joint_speed * sim.get_physics_dt()
    unconstrained_delta = args_cli.ik_gain * (unconstrained_target - current_joint_pos)
    target_delta = torch.clamp(
        unconstrained_delta,
        min=-max_joint_step,
        max=max_joint_step,
    )
    rate_limit_count = int(torch.count_nonzero(torch.abs(target_delta - unconstrained_delta) > 1.0e-7))
    limited_target = current_joint_pos + target_delta
    clamped_target = torch.maximum(torch.minimum(limited_target, joint_limits[..., 1]), joint_limits[..., 0])
    clamp_count = int(torch.count_nonzero(torch.abs(clamped_target - limited_target) > 1.0e-7))

    robot.set_joint_position_target(default_joint_pos)
    robot.set_joint_velocity_target(default_joint_vel)
    robot.set_joint_position_target(clamped_target, joint_ids=arm_joint_ids)
    robot.write_root_pose_to_sim(desired_root_pose_w)
    robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.root_vel_w))
    robot.write_data_to_sim()
    sim.step(render=False)
    robot.update(sim.get_physics_dt())

    tcp_pose_w = robot.data.body_pose_w[:, tcp_body_id]
    if not torch.all(torch.isfinite(tcp_pose_w)) or not torch.all(torch.isfinite(robot.data.joint_pos)):
        raise AssertionError("controller step produced a non-finite robot state")
    position_error, orientation_error = compute_pose_error(
        tcp_pose_w[:, 0:3],
        tcp_pose_w[:, 3:7],
        target_pos_w,
        target_quat_w,
        rot_error_type="axis_angle",
    )
    root_position_error, root_orientation_error = compute_pose_error(
        desired_root_pose_w[:, 0:3],
        desired_root_pose_w[:, 3:7],
        robot.data.root_pose_w[:, 0:3],
        robot.data.root_pose_w[:, 3:7],
        rot_error_type="axis_angle",
    )
    minimum_singular_value = float(torch.min(torch.linalg.svdvals(jacobian)))
    maximum_joint_speed = float(torch.max(torch.abs(robot.data.joint_vel[:, arm_joint_ids])))
    return (
        torch.linalg.vector_norm(position_error, dim=-1),
        torch.linalg.vector_norm(orientation_error, dim=-1),
        tcp_pose_w[:, 0:3].clone(),
        torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, tcp_body_id], dim=-1),
        torch.linalg.vector_norm(robot.data.body_ang_vel_w[:, tcp_body_id], dim=-1),
        torch.linalg.vector_norm(root_position_error, dim=-1),
        torch.linalg.vector_norm(root_orientation_error, dim=-1),
        clamp_count,
        rate_limit_count,
        minimum_singular_value,
        maximum_joint_speed,
    )


def _run_segment(
    *,
    scales: list[float],
    root_delta_values: tuple[float, ...],
    target_pose_w: tuple[torch.Tensor, torch.Tensor],
    initial_root_pose_w: torch.Tensor,
    signs: torch.Tensor,
    controller_args: dict,
) -> dict[str, list | int | float]:
    position_errors: list[torch.Tensor] = []
    orientation_errors: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    linear_speeds: list[torch.Tensor] = []
    angular_speeds: list[torch.Tensor] = []
    root_position_errors: list[torch.Tensor] = []
    root_orientation_errors: list[torch.Tensor] = []
    clamp_count = 0
    rate_limit_count = 0
    minimum_singular_value = math.inf
    maximum_joint_speed = 0.0
    target_pos_w, target_quat_w = target_pose_w

    for scale in scales:
        root_delta = _signed_delta(root_delta_values, signs, scale)
        desired_root_pos_w, desired_root_quat_w = apply_delta_pose(
            initial_root_pose_w[:, 0:3], initial_root_pose_w[:, 3:7], root_delta
        )
        outputs = _step_controller(
            desired_root_pose_w=torch.cat((desired_root_pos_w, desired_root_quat_w), dim=-1),
            target_pos_w=target_pos_w,
            target_quat_w=target_quat_w,
            **controller_args,
        )
        (
            position_error,
            orientation_error,
            position,
            linear_speed,
            angular_speed,
            root_position_error,
            root_orientation_error,
            clamps,
            rate_limits,
            singular_value,
            joint_speed,
        ) = outputs
        position_errors.append(position_error)
        orientation_errors.append(orientation_error)
        positions.append(position)
        linear_speeds.append(linear_speed)
        angular_speeds.append(angular_speed)
        root_position_errors.append(root_position_error)
        root_orientation_errors.append(root_orientation_error)
        clamp_count += clamps
        rate_limit_count += rate_limits
        minimum_singular_value = min(minimum_singular_value, singular_value)
        maximum_joint_speed = max(maximum_joint_speed, joint_speed)

    return {
        "position_errors": position_errors,
        "orientation_errors": orientation_errors,
        "positions": positions,
        "linear_speeds": linear_speeds,
        "angular_speeds": angular_speeds,
        "root_position_errors": root_position_errors,
        "root_orientation_errors": root_orientation_errors,
        "joint_limit_clamp_count": clamp_count,
        "joint_target_rate_limit_count": rate_limit_count,
        "minimum_jacobian_singular_value": minimum_singular_value,
        "maximum_arm_joint_speed_radps": maximum_joint_speed,
    }


def _motion_scales() -> list[float]:
    return [
        _minimum_jerk((step + 1) / args_cli.base_motion_steps)
        for step in range(args_cli.base_motion_steps)
    ]


def main() -> None:
    start_time = time.perf_counter()
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = B2WZ1WorldFrameGateSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene_cfg.robot.spawn.rigid_props.disable_gravity = True
    for actuator_cfg in scene_cfg.robot.actuators.values():
        actuator_cfg.min_delay = 0
        actuator_cfg.max_delay = 0
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.update(sim.get_physics_dt())
    robot: Articulation = scene["robot"]

    if robot.is_fixed_base:
        raise AssertionError("The world-frame gate requires a floating-base Jacobian")
    arm_joint_ids, arm_joint_names = robot.find_joints("joint[1-6]")
    tcp_body_ids, tcp_body_names = robot.find_bodies("tcp_frame")
    if arm_joint_names != [f"joint{index}" for index in range(1, 7)]:
        raise AssertionError(f"Unexpected Z1 joint order: {arm_joint_names}")
    if tcp_body_names != ["tcp_frame"]:
        raise AssertionError(f"Unexpected TCP body lookup: {tcp_body_names}")

    tcp_body_id = tcp_body_ids[0]
    tcp_jacobian_id = tcp_body_id
    arm_jacobian_cols = [joint_id + 6 for joint_id in arm_joint_ids]
    jacobian_shape = tuple(robot.root_physx_view.get_jacobians().shape)
    if jacobian_shape[1] != robot.num_bodies or jacobian_shape[3] != robot.num_joints + 6:
        raise AssertionError(
            f"Unexpected floating Jacobian shape {jacobian_shape} for "
            f"{robot.num_bodies} bodies and {robot.num_joints} joints"
        )

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    initial_root_state_w = robot.data.default_root_state.clone()
    # Articulation defaults are expressed in each environment frame.  Isaac
    # Lab state writes use simulation-world coordinates, so every root must be
    # translated by its replicated environment origin before a batched reset.
    initial_root_state_w[:, 0:3] += scene.env_origins
    yaw_signs = torch.where(
        (torch.arange(args_cli.num_envs, device=sim.device) // 2) % 2 == 0,
        torch.ones(args_cli.num_envs, device=sim.device),
        -torch.ones(args_cli.num_envs, device=sim.device),
    )
    initial_root_delta = torch.zeros((args_cli.num_envs, 6), device=sim.device)
    initial_root_delta[:, 5] = yaw_signs * math.radians(args_cli.initial_yaw_deg)
    _, initial_root_quat_w = apply_delta_pose(
        initial_root_state_w[:, 0:3], initial_root_state_w[:, 3:7], initial_root_delta
    )
    initial_root_state_w[:, 3:7] = initial_root_quat_w
    joint_limits = robot.data.soft_joint_pos_limits[:, arm_joint_ids, :].clone()
    controller = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.03},
        ),
        args_cli.num_envs,
        sim.device,
    )

    robot.write_root_state_to_sim(initial_root_state_w)
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.reset()
    for _ in range(args_cli.settle_steps):
        robot.set_joint_position_target(default_joint_pos)
        robot.set_joint_velocity_target(default_joint_vel)
        robot.write_root_state_to_sim(initial_root_state_w)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())

    initial_root_pose_w = robot.data.root_pose_w.clone()
    local_root_position = initial_root_pose_w[:, 0:3] - scene.env_origins
    expected_local_root_position = robot.data.default_root_state[:, 0:3]
    environment_origin_error_max = float(
        torch.max(torch.linalg.vector_norm(local_root_position - expected_local_root_position, dim=-1))
    )
    if environment_origin_error_max > 1.0e-5:
        raise AssertionError(
            f"environment-origin reset error is {environment_origin_error_max:.3e} m"
        )
    if args_cli.num_envs > 1:
        root_distances = torch.cdist(initial_root_pose_w[:, 0:3], initial_root_pose_w[:, 0:3])
        root_distances.fill_diagonal_(math.inf)
        minimum_environment_separation = float(torch.min(root_distances))
        if minimum_environment_separation < 2.0:
            raise AssertionError(
                f"replicated roots overlap: minimum separation {minimum_environment_separation:.3f} m"
            )
    else:
        minimum_environment_separation = None
    initial_tcp_pose_w = robot.data.body_pose_w[:, tcp_body_id].clone()
    signs = torch.where(
        torch.arange(args_cli.num_envs, device=sim.device) % 2 == 0,
        torch.ones(args_cli.num_envs, device=sim.device),
        -torch.ones(args_cli.num_envs, device=sim.device),
    )
    final_goal_delta = _signed_delta(TCP_GOAL_DELTA, signs, 1.0)
    final_target_pos_w, final_target_quat_w = apply_delta_pose(
        initial_tcp_pose_w[:, 0:3], initial_tcp_pose_w[:, 3:7], final_goal_delta
    )
    frozen_final_target = torch.cat((final_target_pos_w, final_target_quat_w), dim=-1).clone()

    controller_args = {
        "robot": robot,
        "sim": sim,
        "controller": controller,
        "arm_joint_ids": arm_joint_ids,
        "arm_jacobian_cols": arm_jacobian_cols,
        "tcp_body_id": tcp_body_id,
        "tcp_jacobian_id": tcp_jacobian_id,
        "default_joint_pos": default_joint_pos,
        "default_joint_vel": default_joint_vel,
        "joint_limits": joint_limits,
    }

    reference_position_errors: list[torch.Tensor] = []
    reference_orientation_errors: list[torch.Tensor] = []
    total_clamp_count = 0
    total_rate_limit_count = 0
    minimum_singular_value = math.inf
    maximum_arm_joint_speed = 0.0
    stationary_root_delta = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    for step in range(args_cli.target_motion_steps):
        scale = _minimum_jerk((step + 1) / args_cli.target_motion_steps)
        active_delta = _signed_delta(TCP_GOAL_DELTA, signs, scale)
        active_target_pos_w, active_target_quat_w = apply_delta_pose(
            initial_tcp_pose_w[:, 0:3], initial_tcp_pose_w[:, 3:7], active_delta
        )
        segment = _run_segment(
            scales=[0.0],
            root_delta_values=stationary_root_delta,
            target_pose_w=(active_target_pos_w, active_target_quat_w),
            initial_root_pose_w=initial_root_pose_w,
            signs=signs,
            controller_args=controller_args,
        )
        reference_position_errors.extend(segment["position_errors"])
        reference_orientation_errors.extend(segment["orientation_errors"])
        total_clamp_count += int(segment["joint_limit_clamp_count"])
        total_rate_limit_count += int(segment["joint_target_rate_limit_count"])
        minimum_singular_value = min(minimum_singular_value, float(segment["minimum_jacobian_singular_value"]))
        maximum_arm_joint_speed = max(
            maximum_arm_joint_speed, float(segment["maximum_arm_joint_speed_radps"])
        )

    target_hold = _run_segment(
        scales=[0.0] * args_cli.target_hold_steps,
        root_delta_values=stationary_root_delta,
        target_pose_w=(final_target_pos_w, final_target_quat_w),
        initial_root_pose_w=initial_root_pose_w,
        signs=signs,
        controller_args=controller_args,
    )
    total_clamp_count += int(target_hold["joint_limit_clamp_count"])
    total_rate_limit_count += int(target_hold["joint_target_rate_limit_count"])
    minimum_singular_value = min(minimum_singular_value, float(target_hold["minimum_jacobian_singular_value"]))
    maximum_arm_joint_speed = max(
        maximum_arm_joint_speed, float(target_hold["maximum_arm_joint_speed_radps"])
    )
    reference_clamp_count = total_clamp_count
    reference_rate_limit_count = total_rate_limit_count
    reference_maximum_arm_joint_speed = maximum_arm_joint_speed

    reference_dynamic = _history_metrics(
        reference_position_errors, reference_orientation_errors, signs
    )
    reference_terminal = _window_metrics(
        target_hold["position_errors"],
        target_hold["orientation_errors"],
        target_hold["positions"],
        target_hold["linear_speeds"],
        target_hold["angular_speeds"],
        signs,
    )
    reference_passed = bool(
        reference_dynamic["position_error_rms_worst_env_m"] <= args_cli.max_dynamic_position_rms
        and reference_dynamic["orientation_error_rms_worst_env_deg"]
        <= args_cli.max_dynamic_orientation_rms_deg
        and reference_terminal["position_error_rms_worst_env_m"]
        <= args_cli.max_terminal_position_rms
        and reference_terminal["orientation_error_rms_worst_env_deg"]
        <= args_cli.max_terminal_orientation_rms_deg
        and reference_terminal["position_jitter_max_m"] <= args_cli.max_terminal_jitter
        and reference_terminal["linear_speed_rms_worst_env_mps"]
        <= args_cli.max_terminal_linear_speed
        and reference_terminal["angular_speed_rms_worst_env_radps"]
        <= args_cli.max_terminal_angular_speed
        and reference_clamp_count == 0
        and reference_maximum_arm_joint_speed <= args_cli.max_joint_speed * 1.05
    )
    print(
        f"{'PASS' if reference_passed else 'FAIL'} smooth_target: "
        f"worst-env dynamic={1000.0 * reference_dynamic['position_error_rms_worst_env_m']:.2f} mm/"
        f"{reference_dynamic['orientation_error_rms_worst_env_deg']:.2f} deg, "
        f"terminal={1000.0 * reference_terminal['position_error_rms_worst_env_m']:.2f} mm/"
        f"{reference_terminal['orientation_error_rms_worst_env_deg']:.2f} deg",
        flush=True,
    )

    excitation_results = []
    outbound_scales = _motion_scales()
    return_scales = [1.0 - value for value in outbound_scales]
    for name, root_delta_values in BASE_EXCITATIONS:
        arm_joint_pos_before = robot.data.joint_pos[:, arm_joint_ids].clone()
        outbound = _run_segment(
            scales=outbound_scales,
            root_delta_values=root_delta_values,
            target_pose_w=(final_target_pos_w, final_target_quat_w),
            initial_root_pose_w=initial_root_pose_w,
            signs=signs,
            controller_args=controller_args,
        )
        outbound_hold = _run_segment(
            scales=[1.0] * args_cli.base_hold_steps,
            root_delta_values=root_delta_values,
            target_pose_w=(final_target_pos_w, final_target_quat_w),
            initial_root_pose_w=initial_root_pose_w,
            signs=signs,
            controller_args=controller_args,
        )
        arm_counter_motion_by_env = torch.linalg.vector_norm(
            robot.data.joint_pos[:, arm_joint_ids] - arm_joint_pos_before, dim=-1
        )
        returning = _run_segment(
            scales=return_scales,
            root_delta_values=root_delta_values,
            target_pose_w=(final_target_pos_w, final_target_quat_w),
            initial_root_pose_w=initial_root_pose_w,
            signs=signs,
            controller_args=controller_args,
        )
        return_hold = _run_segment(
            scales=[0.0] * args_cli.base_hold_steps,
            root_delta_values=root_delta_values,
            target_pose_w=(final_target_pos_w, final_target_quat_w),
            initial_root_pose_w=initial_root_pose_w,
            signs=signs,
            controller_args=controller_args,
        )
        motion_position_errors = outbound["position_errors"] + returning["position_errors"]
        motion_orientation_errors = outbound["orientation_errors"] + returning["orientation_errors"]
        dynamic_metrics = _history_metrics(motion_position_errors, motion_orientation_errors, signs)
        outbound_metrics = _window_metrics(
            outbound_hold["position_errors"],
            outbound_hold["orientation_errors"],
            outbound_hold["positions"],
            outbound_hold["linear_speeds"],
            outbound_hold["angular_speeds"],
            signs,
        )
        return_metrics = _window_metrics(
            return_hold["position_errors"],
            return_hold["orientation_errors"],
            return_hold["positions"],
            return_hold["linear_speeds"],
            return_hold["angular_speeds"],
            signs,
        )
        phase_root_position_errors = (
            outbound["root_position_errors"]
            + outbound_hold["root_position_errors"]
            + returning["root_position_errors"]
            + return_hold["root_position_errors"]
        )
        phase_root_orientation_errors = (
            outbound["root_orientation_errors"]
            + outbound_hold["root_orientation_errors"]
            + returning["root_orientation_errors"]
            + return_hold["root_orientation_errors"]
        )
        root_position_error_max = float(torch.max(torch.stack(phase_root_position_errors)))
        root_orientation_error_max_deg = math.degrees(
            float(torch.max(torch.stack(phase_root_orientation_errors)))
        )
        phase_clamps = sum(
            int(segment["joint_limit_clamp_count"])
            for segment in (outbound, outbound_hold, returning, return_hold)
        )
        phase_rate_limits = sum(
            int(segment["joint_target_rate_limit_count"])
            for segment in (outbound, outbound_hold, returning, return_hold)
        )
        phase_minimum_singular_value = min(
            float(segment["minimum_jacobian_singular_value"])
            for segment in (outbound, outbound_hold, returning, return_hold)
        )
        phase_maximum_arm_joint_speed = max(
            float(segment["maximum_arm_joint_speed_radps"])
            for segment in (outbound, outbound_hold, returning, return_hold)
        )
        minimum_arm_counter_motion = float(torch.min(arm_counter_motion_by_env))
        passed = bool(
            dynamic_metrics["position_error_rms_worst_env_m"] <= args_cli.max_dynamic_position_rms
            and dynamic_metrics["orientation_error_rms_worst_env_deg"]
            <= args_cli.max_dynamic_orientation_rms_deg
            and outbound_metrics["position_error_rms_worst_env_m"]
            <= args_cli.max_terminal_position_rms
            and outbound_metrics["orientation_error_rms_worst_env_deg"]
            <= args_cli.max_terminal_orientation_rms_deg
            and outbound_metrics["position_jitter_max_m"] <= args_cli.max_terminal_jitter
            and outbound_metrics["linear_speed_rms_worst_env_mps"]
            <= args_cli.max_terminal_linear_speed
            and outbound_metrics["angular_speed_rms_worst_env_radps"]
            <= args_cli.max_terminal_angular_speed
            and return_metrics["position_error_rms_worst_env_m"]
            <= args_cli.max_terminal_position_rms
            and return_metrics["orientation_error_rms_worst_env_deg"]
            <= args_cli.max_terminal_orientation_rms_deg
            and return_metrics["position_jitter_max_m"] <= args_cli.max_terminal_jitter
            and return_metrics["linear_speed_rms_worst_env_mps"]
            <= args_cli.max_terminal_linear_speed
            and return_metrics["angular_speed_rms_worst_env_radps"]
            <= args_cli.max_terminal_angular_speed
            and root_position_error_max <= args_cli.max_root_position_error
            and root_orientation_error_max_deg <= args_cli.max_root_orientation_error_deg
            and minimum_arm_counter_motion >= 0.03
            and phase_maximum_arm_joint_speed <= args_cli.max_joint_speed * 1.05
            and phase_clamps == 0
        )
        excitation_results.append(
            {
                "name": name,
                "passed": passed,
                "root_delta": list(root_delta_values),
                "dynamic": dynamic_metrics,
                "outbound_hold": outbound_metrics,
                "return_hold": return_metrics,
                "root_position_error_max_m": root_position_error_max,
                "root_orientation_error_max_deg": root_orientation_error_max_deg,
                "minimum_arm_counter_motion_rad": minimum_arm_counter_motion,
                "joint_limit_clamp_count": phase_clamps,
                "joint_target_rate_limit_count": phase_rate_limits,
                "minimum_jacobian_singular_value": phase_minimum_singular_value,
                "maximum_arm_joint_speed_radps": phase_maximum_arm_joint_speed,
            }
        )
        total_clamp_count += phase_clamps
        total_rate_limit_count += phase_rate_limits
        minimum_singular_value = min(minimum_singular_value, phase_minimum_singular_value)
        maximum_arm_joint_speed = max(maximum_arm_joint_speed, phase_maximum_arm_joint_speed)
        print(
            f"{'PASS' if passed else 'FAIL'} {name}: "
            f"worst-env dynamic={1000.0 * dynamic_metrics['position_error_rms_worst_env_m']:.2f} mm/"
            f"{dynamic_metrics['orientation_error_rms_worst_env_deg']:.2f} deg, "
            f"displaced_hold={1000.0 * outbound_metrics['position_error_rms_worst_env_m']:.2f} mm/"
            f"{outbound_metrics['orientation_error_rms_worst_env_deg']:.2f} deg",
            flush=True,
        )

    if not torch.equal(torch.cat((final_target_pos_w, final_target_quat_w), dim=-1), frozen_final_target):
        raise AssertionError("The immutable world-frame TCP target was mutated")

    report = {
        "gate": "floating_base_kinematic_world_frame_tcp_tracking",
        "passed": bool(
            reference_passed
            and all(result["passed"] for result in excitation_results)
            and total_clamp_count == 0
            and maximum_arm_joint_speed <= args_cli.max_joint_speed * 1.05
        ),
        "device": args_cli.device,
        "num_envs": args_cli.num_envs,
        "provenance": _provenance(),
        "physics_dt_s": sim.get_physics_dt(),
        "steps": {
            "settle": args_cli.settle_steps,
            "target_motion": args_cli.target_motion_steps,
            "target_hold": args_cli.target_hold_steps,
            "base_motion_each_direction": args_cli.base_motion_steps,
            "base_hold_each_endpoint": args_cli.base_hold_steps,
            "terminal_window": args_cli.window_steps,
        },
        "gravity_enabled": False,
        "root_motion": "kinematically prescribed; this is not a wheel-dynamics test",
        "actuator_delay_steps": 0,
        "environment_origins_applied": True,
        "environment_origin_error_max_m": environment_origin_error_max,
        "minimum_environment_root_separation_m": minimum_environment_separation,
        "sign_assignment": "even environment index positive; odd environment index negative",
        "initial_base_yaw_deg": args_cli.initial_yaw_deg,
        "initial_yaw_assignment": "indices 0-1 positive, 2-3 negative, repeating",
        "arm_joint_names": arm_joint_names,
        "arm_joint_ids": arm_joint_ids,
        "arm_jacobian_columns": arm_jacobian_cols,
        "tcp_body_id": tcp_body_id,
        "tcp_jacobian_id": tcp_jacobian_id,
        "jacobian_shape": jacobian_shape,
        "controller": {
            "method": "damped_least_squares",
            "damping": 0.03,
            "per_step_gain": args_cli.ik_gain,
            "joint_target_rate_limit_radps": args_cli.max_joint_speed,
        },
        "reference": {
            "profile": "minimum_jerk_10u3_minus_15u4_plus_6u5",
            "goal_delta": list(TCP_GOAL_DELTA),
            "motion_steps": args_cli.target_motion_steps,
            "hold_steps": args_cli.target_hold_steps,
            "passed": reference_passed,
            "dynamic": reference_dynamic,
            "terminal": reference_terminal,
            "joint_limit_clamp_count": reference_clamp_count,
            "joint_target_rate_limit_count": reference_rate_limit_count,
            "maximum_arm_joint_speed_radps": reference_maximum_arm_joint_speed,
        },
        "base_excitations": excitation_results,
        "thresholds": {
            "dynamic_position_rms_worst_env_m": args_cli.max_dynamic_position_rms,
            "dynamic_orientation_rms_worst_env_deg": args_cli.max_dynamic_orientation_rms_deg,
            "terminal_position_rms_worst_env_m": args_cli.max_terminal_position_rms,
            "terminal_orientation_rms_worst_env_deg": args_cli.max_terminal_orientation_rms_deg,
            "terminal_position_jitter_m": args_cli.max_terminal_jitter,
            "terminal_linear_speed_rms_worst_env_mps": args_cli.max_terminal_linear_speed,
            "terminal_angular_speed_rms_worst_env_radps": args_cli.max_terminal_angular_speed,
            "root_position_error_max_m": args_cli.max_root_position_error,
            "root_orientation_error_max_deg": args_cli.max_root_orientation_error_deg,
            "minimum_arm_counter_motion_rad": 0.03,
            "maximum_arm_joint_speed_radps": args_cli.max_joint_speed * 1.05,
        },
        "total_joint_limit_clamp_count": total_clamp_count,
        "total_joint_target_rate_limit_count": total_rate_limit_count,
        "total_joint_target_rate_limit_fraction": total_rate_limit_count
        / (
            args_cli.num_envs
            * len(arm_joint_ids)
            * (
                args_cli.target_motion_steps
                + args_cli.target_hold_steps
                + len(BASE_EXCITATIONS)
                * 2
                * (args_cli.base_motion_steps + args_cli.base_hold_steps)
            )
        ),
        "minimum_jacobian_singular_value": minimum_singular_value,
        "maximum_arm_joint_speed_radps": maximum_arm_joint_speed,
        "world_target_mutated": False,
        "elapsed_wall_time_s": time.perf_counter() - start_time,
    }
    report_text = json.dumps(report, indent=2, sort_keys=True)
    if args_cli.report is not None:
        args_cli.report.parent.mkdir(parents=True, exist_ok=True)
        args_cli.report.write_text(report_text + "\n", encoding="utf-8")
        print(f"Report: {args_cli.report.resolve()}", flush=True)
    print(report_text, flush=True)
    if not report["passed"]:
        raise AssertionError("B2-W + Z1 floating-base world-frame gate failed")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
