# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Finite smoke test for the trainable B2W + Z1 TCP coordinator task."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="B2W-Z1-TCP-Play")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=180)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--report", type=str, default=None)
parser.add_argument("--policy", type=str, default=None, help="Optional exported 57-D to 2-D TorchScript policy.")
parser.add_argument(
    "--controller",
    choices=("zero", "analytic"),
    default="zero",
    help="Baseline used when --policy is not supplied.",
)
parser.add_argument("--trace_interval", type=int, default=0, help="Print compact metrics every N steps; 0 disables.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim has launched."""

import json
import math
from pathlib import Path

import gymnasium as gym
import LeggedManip_Lab.tasks  # noqa: F401
import torch

from isaaclab.utils.math import compute_pose_error, quat_apply, quat_apply_inverse, quat_inv, quat_mul

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _quaternion_yaw(quaternion: torch.Tensor) -> torch.Tensor:
    """Return yaw for a batch of scalar-first quaternions."""
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _analytic_coordinator_action(task, command_term, action_term) -> torch.Tensor:
    """Track the root pose implied by the smooth TCP reference and reset arm pose.

    This is an interpretable unicycle pose-controller baseline, not an RL input.
    It derives the desired root from the public world TCP reference and the known
    reset arm FK. It never writes root state or bypasses wheel/contact physics.
    """
    robot = task.scene["robot"]
    reference = command_term.command[:, :7]
    sampled_tcp_pose_b = command_term.sampled_tcp_pose_b
    desired_root_quat_w = quat_mul(reference[:, 3:7], quat_inv(sampled_tcp_pose_b[:, 3:7]))
    desired_root_pos_w = reference[:, :3] - quat_apply(desired_root_quat_w, sampled_tcp_pose_b[:, :3])

    position_error_b = quat_apply_inverse(
        robot.data.root_quat_w,
        desired_root_pos_w - robot.data.root_pos_w,
    )
    root_quat_error = quat_mul(quat_inv(robot.data.root_quat_w), desired_root_quat_w)
    final_yaw_error = _wrap_to_pi(_quaternion_yaw(root_quat_error))

    # Local SE(2) feedback: reverse directly for a target behind the body and
    # use lateral/yaw error jointly for steering. This avoids an unnecessary
    # 180-degree turn for a platform that can drive in both directions.
    longitudinal_error = position_error_b[:, 0]
    lateral_error = position_error_b[:, 1]
    desired_vx = torch.clamp(2.5 * longitudinal_error, -0.30, 0.30)
    drive_active = torch.abs(longitudinal_error) > 0.008
    desired_vx = torch.where(
        drive_active,
        torch.sign(desired_vx) * torch.maximum(torch.abs(desired_vx), torch.full_like(desired_vx, 0.12)),
        torch.zeros_like(desired_vx),
    )

    # The frozen plant has a measured low-speed yaw dead zone. Use an explicit
    # minimum command outside the stopping tolerance rather than pretending its
    # velocity command is an ideal actuator.
    yaw_control = 4.0 * lateral_error + 2.0 * final_yaw_error
    desired_wz = torch.clamp(yaw_control, -0.60, 0.60)
    yaw_active = torch.logical_or(
        torch.abs(lateral_error) > 0.010,
        torch.abs(final_yaw_error) > math.radians(3.0),
    )
    desired_wz = torch.where(
        yaw_active,
        torch.sign(desired_wz) * torch.maximum(torch.abs(desired_wz), torch.full_like(desired_wz, 0.45)),
        torch.zeros_like(desired_wz),
    )

    action = torch.empty((task.num_envs, 2), device=task.device)
    # Raw zero is the measured stationary baseline: process_actions applies the
    # configured -0.05 m/s command bias that cancels the plant's forward drift.
    action[:, 0] = desired_vx / action_term.cfg.max_linear_velocity
    action[:, 1] = desired_wz / action_term.cfg.max_yaw_velocity
    return torch.clamp(action, -1.0, 1.0)


def main() -> dict:
    """Instantiate, step, and validate the manager-based task."""
    if args_cli.task not in gym.registry:
        raise RuntimeError(f"Task is not registered: {args_cli.task}")
    if args_cli.num_envs < 1 or args_cli.steps < 1:
        raise ValueError("num_envs and steps must both be positive")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.actions.coordinator.runtime_checks = True
    env = gym.make(args_cli.task, cfg=env_cfg)
    task = env.unwrapped

    try:
        if args_cli.policy is not None and args_cli.controller != "zero":
            raise ValueError("--policy and a non-zero --controller are mutually exclusive")
        observations, _ = env.reset(seed=args_cli.seed)
        policy_observation = observations["policy"]
        action_term = task.action_manager.get_term("coordinator")
        command_term = task.command_manager.get_term("tcp_pose")
        robot = task.scene["robot"]
        tcp_body_id = command_term.body_idx

        initial_goal = command_term.final_command.clone()
        initial_ghost_root_pose_w = command_term.ghost_root_pose_w.clone()
        initial_root_pose_w = robot.data.root_pose_w.clone()
        initial_root_position = initial_root_pose_w[:, :3]
        initial_ghost_displacement_b = quat_apply_inverse(
            initial_root_pose_w[:, 3:7],
            initial_ghost_root_pose_w[:, :3] - initial_root_position,
        )
        initial_ghost_quat_error = quat_mul(
            quat_inv(initial_root_pose_w[:, 3:7]),
            initial_ghost_root_pose_w[:, 3:7],
        )
        initial_ghost_yaw = _wrap_to_pi(_quaternion_yaw(initial_ghost_quat_error))
        initial_tcp_pose_w = robot.data.body_pose_w[:, tcp_body_id].clone()
        initial_reference = command_term.command[:, :7].clone()
        initial_position_error, initial_orientation_error = compute_pose_error(
            initial_tcp_pose_w[:, :3],
            initial_tcp_pose_w[:, 3:7],
            initial_reference[:, :3],
            initial_reference[:, 3:7],
            rot_error_type="axis_angle",
        )
        policy = None
        if args_cli.policy is not None:
            policy_path = Path(args_cli.policy).expanduser().resolve()
            if not policy_path.is_file():
                raise FileNotFoundError(f"Exported coordinator policy not found: {policy_path}")
            policy = torch.jit.load(str(policy_path), map_location=task.device).eval()
            with torch.inference_mode():
                policy_probe = policy(policy_observation)
            if policy_probe.shape != (task.num_envs, 2):
                raise ValueError(f"Expected policy output {(task.num_envs, 2)}, got {tuple(policy_probe.shape)}")

        finite = bool(torch.all(torch.isfinite(policy_observation)))
        minimum_root_height = math.inf
        maximum_tilt = 0.0
        maximum_reference_position_error = 0.0
        maximum_reference_orientation_error = 0.0
        cumulative_reward = torch.zeros(task.num_envs, device=task.device)
        terminated_count = 0
        truncated_count = 0
        absolute_action_sum = torch.zeros(2, device=task.device)
        absolute_processed_command_sum = torch.zeros(2, device=task.device)
        trace = []

        for step_index in range(args_cli.steps):
            with torch.inference_mode():
                if policy is None:
                    if args_cli.controller == "analytic":
                        action = _analytic_coordinator_action(task, command_term, action_term)
                    else:
                        action = torch.zeros((task.num_envs, 2), device=task.device)
                else:
                    action = policy(policy_observation)
                absolute_action_sum += torch.sum(torch.abs(action), dim=0)
                observations, reward, terminated, truncated, _ = env.step(action)
                absolute_processed_command_sum += torch.sum(torch.abs(action_term.processed_actions), dim=0)

            policy_observation = observations["policy"]
            reference = command_term.command[:, :7]
            tcp_pose_w = robot.data.body_pose_w[:, tcp_body_id]
            position_error, orientation_error = compute_pose_error(
                tcp_pose_w[:, :3],
                tcp_pose_w[:, 3:7],
                reference[:, :3],
                reference[:, 3:7],
                rot_error_type="axis_angle",
            )
            tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))

            finite = finite and bool(
                torch.all(torch.isfinite(policy_observation))
                and torch.all(torch.isfinite(action))
                and torch.all(torch.isfinite(reward))
                and torch.all(torch.isfinite(robot.data.joint_pos))
                and torch.all(torch.isfinite(tcp_pose_w))
            )
            minimum_root_height = min(minimum_root_height, _as_float(torch.min(robot.data.root_pos_w[:, 2])))
            maximum_tilt = max(maximum_tilt, _as_float(torch.max(tilt)))
            maximum_reference_position_error = max(
                maximum_reference_position_error,
                _as_float(torch.max(torch.linalg.vector_norm(position_error, dim=-1))),
            )
            maximum_reference_orientation_error = max(
                maximum_reference_orientation_error,
                _as_float(torch.max(torch.linalg.vector_norm(orientation_error, dim=-1))),
            )
            cumulative_reward += reward
            terminated_count += int(torch.count_nonzero(terminated).item())
            truncated_count += int(torch.count_nonzero(truncated).item())
            if args_cli.trace_interval > 0 and (step_index == 0 or (step_index + 1) % args_cli.trace_interval == 0):
                trace_sample = {
                    "step": step_index + 1,
                    "reference_position_error_mean_m": _as_float(
                        torch.mean(torch.linalg.vector_norm(position_error, dim=-1))
                    ),
                    "reference_orientation_error_mean_rad": _as_float(
                        torch.mean(torch.linalg.vector_norm(orientation_error, dim=-1))
                    ),
                    "root_height_min_m": _as_float(torch.min(robot.data.root_pos_w[:, 2])),
                    "arm_target_error_max_rad": _as_float(
                        torch.max(
                            torch.abs(
                                robot.data.joint_pos[:, action_term.arm_joint_ids] - action_term.arm_position_targets
                            )
                        )
                    ),
                }
                trace.append(trace_sample)
                print("B2W_Z1_TCP_TRACE=" + json.dumps(trace_sample, sort_keys=True), flush=True)

        final_tcp_pose_w = robot.data.body_pose_w[:, tcp_body_id]
        final_goal = command_term.final_command
        final_position_error, final_orientation_error = compute_pose_error(
            final_tcp_pose_w[:, :3],
            final_tcp_pose_w[:, 3:7],
            final_goal[:, :3],
            final_goal[:, 3:7],
            rot_error_type="axis_angle",
        )
        root_displacement = torch.linalg.vector_norm(
            robot.data.root_pos_w[:, :2] - initial_root_position[:, :2], dim=-1
        )
        root_goal_position_error = torch.linalg.vector_norm(
            command_term.ghost_root_pose_w[:, :2] - robot.data.root_pos_w[:, :2], dim=-1
        )
        root_goal_quat_error = quat_mul(
            quat_inv(robot.data.root_quat_w),
            command_term.ghost_root_pose_w[:, 3:7],
        )
        root_goal_yaw_error = torch.abs(_wrap_to_pi(_quaternion_yaw(root_goal_quat_error)))
        goal_displacement = torch.linalg.vector_norm(initial_goal[:, :3] - initial_tcp_pose_w[:, :3], dim=-1)
        arm_joint_position = robot.data.joint_pos[:, action_term.arm_joint_ids]
        arm_home = robot.data.default_joint_pos[:, action_term.arm_joint_ids]
        arm_limits = robot.data.soft_joint_pos_limits[:, action_term.arm_joint_ids]
        arm_half_range = 0.5 * (arm_limits[..., 1] - arm_limits[..., 0])
        normalized_arm_home_error = (arm_joint_position - arm_home) / torch.clamp(arm_half_range, min=1.0e-3)
        normalized_arm_home_error_norm = torch.linalg.vector_norm(normalized_arm_home_error, dim=-1)
        goal_immutable = (
            bool(torch.equal(initial_goal, final_goal)) if terminated_count + truncated_count == 0 else False
        )

        final_position_error_norm = torch.linalg.vector_norm(final_position_error, dim=-1)
        final_orientation_error_norm = torch.linalg.vector_norm(final_orientation_error, dim=-1)

        planar_goal_distance = torch.linalg.vector_norm(initial_ghost_displacement_b[:, :2], dim=-1)
        goal_x = initial_ghost_displacement_b[:, 0]
        goal_y = initial_ghost_displacement_b[:, 1]
        stationary = torch.logical_and(planar_goal_distance < 1.0e-5, torch.abs(initial_ghost_yaw) < 1.0e-5)
        moving = torch.logical_not(stationary)
        target_masks = {
            "stationary": stationary,
            "forward": torch.logical_and(moving, goal_x >= torch.abs(goal_y)),
            "reverse": torch.logical_and(moving, -goal_x >= torch.abs(goal_y)),
            "left": torch.logical_and(moving, goal_y > torch.abs(goal_x)),
            "right": torch.logical_and(moving, -goal_y > torch.abs(goal_x)),
            "positive_yaw": initial_ghost_yaw > 1.0e-5,
            "negative_yaw": initial_ghost_yaw < -1.0e-5,
        }

        def summarize_target_family(mask: torch.Tensor) -> dict:
            count = int(torch.count_nonzero(mask).item())
            if count == 0:
                return {"count": 0}
            return {
                "count": count,
                "tcp_position_error_mean_m": _as_float(torch.mean(final_position_error_norm[mask])),
                "tcp_orientation_error_mean_rad": _as_float(torch.mean(final_orientation_error_norm[mask])),
                "arm_home_error_mean": _as_float(torch.mean(normalized_arm_home_error_norm[mask])),
                "root_displacement_mean_m": _as_float(torch.mean(root_displacement[mask])),
                "root_goal_position_error_mean_m": _as_float(torch.mean(root_goal_position_error[mask])),
                "root_goal_yaw_error_mean_rad": _as_float(torch.mean(root_goal_yaw_error[mask])),
                "episode_reward_mean": _as_float(torch.mean(cumulative_reward[mask])),
            }

        checks = {
            "action_shape_is_2": tuple(env.action_space.shape) == (task.num_envs, 2),
            "command_shape_is_13": tuple(command_term.command.shape) == (task.num_envs, 13),
            "finite": finite,
            "goal_immutable_without_reset": goal_immutable,
            "initial_reference_position_matches_tcp": _as_float(
                torch.max(torch.linalg.vector_norm(initial_position_error, dim=-1))
            )
            < 1.0e-4,
            "initial_reference_orientation_matches_tcp": _as_float(
                torch.max(torch.linalg.vector_norm(initial_orientation_error, dim=-1))
            )
            < 1.0e-4,
            "no_early_reset": terminated_count == 0 and truncated_count == 0,
            "policy_observation_shape_is_57": tuple(policy_observation.shape) == (task.num_envs, 57),
            "root_above_termination_height": minimum_root_height >= 0.45,
        }
        report = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "task": args_cli.task,
            "controller": args_cli.controller if policy is None else "torchscript_policy",
            "policy_path": None if args_cli.policy is None else str(Path(args_cli.policy).expanduser().resolve()),
            "seed": args_cli.seed,
            "num_envs": task.num_envs,
            "steps": args_cli.steps,
            "physics_hz": 1.0 / task.physics_dt,
            "coordinator_hz": 1.0 / task.step_dt,
            "checks": checks,
            "first_environment_initial_state": {
                "root_pose_w": initial_root_pose_w[0].detach().cpu().tolist(),
                "tcp_pose_w": initial_tcp_pose_w[0].detach().cpu().tolist(),
                "reference_pose_w": initial_reference[0].detach().cpu().tolist(),
                "sampled_tcp_pose_b": command_term.sampled_tcp_pose_b[0].detach().cpu().tolist(),
            },
            "metrics": {
                "mean_episode_reward": _as_float(torch.mean(cumulative_reward)),
                "initial_reference_position_error_max_m": _as_float(
                    torch.max(torch.linalg.vector_norm(initial_position_error, dim=-1))
                ),
                "initial_reference_orientation_error_max_rad": _as_float(
                    torch.max(torch.linalg.vector_norm(initial_orientation_error, dim=-1))
                ),
                "minimum_root_height_m": minimum_root_height,
                "maximum_root_tilt_rad": maximum_tilt,
                "maximum_reference_position_error_m": maximum_reference_position_error,
                "maximum_reference_orientation_error_rad": maximum_reference_orientation_error,
                "final_goal_position_error_mean_m": _as_float(
                    torch.mean(torch.linalg.vector_norm(final_position_error, dim=-1))
                ),
                "final_goal_position_error_max_m": _as_float(
                    torch.max(torch.linalg.vector_norm(final_position_error, dim=-1))
                ),
                "final_goal_orientation_error_mean_rad": _as_float(
                    torch.mean(torch.linalg.vector_norm(final_orientation_error, dim=-1))
                ),
                "goal_displacement_mean_m": _as_float(torch.mean(goal_displacement)),
                "mean_absolute_normalized_action": (absolute_action_sum / float(task.num_envs * args_cli.steps))
                .detach()
                .cpu()
                .tolist(),
                "mean_absolute_low_level_velocity_command": (
                    absolute_processed_command_sum / float(task.num_envs * args_cli.steps)
                )
                .detach()
                .cpu()
                .tolist(),
                "final_normalized_arm_home_error_mean": _as_float(
                    torch.mean(normalized_arm_home_error_norm)
                ),
                "root_displacement_mean_m": _as_float(torch.mean(root_displacement)),
                "root_goal_position_error_mean_m": _as_float(torch.mean(root_goal_position_error)),
                "root_goal_yaw_error_mean_rad": _as_float(torch.mean(root_goal_yaw_error)),
                "joint_rate_limit_count": int(action_term.joint_rate_limit_count.item()),
                "joint_limit_clamp_count": int(action_term.joint_limit_clamp_count.item()),
                "terminated_count": terminated_count,
                "truncated_count": truncated_count,
            },
            "reward_term_sums_mean": {
                name: _as_float(torch.mean(value)) for name, value in task.reward_manager._episode_sums.items()
            },
            "target_family_metrics": {
                name: summarize_target_family(mask) for name, mask in target_masks.items()
            },
            "trace": trace,
        }

        if args_cli.report is not None:
            report_path = Path(args_cli.report).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, sort_keys=True)
                stream.write("\n")

        print("B2W_Z1_TCP_TASK_SMOKE=" + json.dumps(report, sort_keys=True), flush=True)
        return report
    finally:
        env.close()


if __name__ == "__main__":
    try:
        result = main()
        if result["status"] != "PASS":
            raise RuntimeError("B2W + Z1 TCP task smoke checks failed")
    finally:
        simulation_app.close()
