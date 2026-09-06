"""Finite deterministic checkpoint evaluation for unified B2W + Z1 EE-WBC."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="B2W-Z1-EE-WBC-Flat-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=599)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--report", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import importlib.metadata as metadata
import json
import math
from pathlib import Path

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import LeggedManip_Lab.tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab.utils.math import euler_xyz_from_quat


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.flatten()
    if values.numel() == 0:
        return {"mean": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "mean": _as_float(torch.mean(values)),
        "p95": _as_float(torch.quantile(values, 0.95)),
        "max": _as_float(torch.max(values)),
    }


def _signed_summary(values: torch.Tensor) -> dict[str, float]:
    values = values.flatten()
    if values.numel() == 0:
        return {"mean": math.nan, "p05": math.nan, "p95": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": _as_float(torch.mean(values)),
        "p05": _as_float(torch.quantile(values, 0.05)),
        "p95": _as_float(torch.quantile(values, 0.95)),
        "min": _as_float(torch.min(values)),
        "max": _as_float(torch.max(values)),
    }


def main() -> dict:
    if args_cli.num_envs < 1 or args_cli.steps < 1:
        raise ValueError("num_envs and steps must be positive")

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    agent_cfg.device = args_cli.device
    agent_cfg.seed = args_cli.seed

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    task = env.unwrapped

    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(Path(args_cli.checkpoint).expanduser().resolve()))
        policy = runner.get_inference_policy(device=task.device)
        obs = env.get_observations()

        robot = task.scene["robot"]
        command = task.command_manager.get_term("tcp_pose")
        initial_spatial_mask = getattr(
            command,
            "spatial_mask",
            torch.zeros(task.num_envs, dtype=torch.bool, device=task.device),
        ).clone()
        settle_steps = math.ceil(command.cfg.settle_time_s / task.step_dt)
        alive = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
        initial_root_xy = robot.data.root_pos_w[:, :2].clone()
        target_translation = None
        target_planar_translation = None
        target_vertical_translation = None
        settled_root_pitch = torch.zeros(task.num_envs, device=task.device)
        reference_position_errors = []
        reference_orientation_errors = []
        spatial_reference_position_errors = []
        planar_reference_position_errors = []
        minimum_root_height = math.inf
        maximum_tilt = 0.0
        termination_counts = {name: 0 for name in task.termination_manager.active_terms}
        termination_steps = []
        undesired_cfg = task.termination_manager.get_term_cfg("undesired_contact")
        undesired_sensor_cfg = undesired_cfg.params["sensor_cfg"]
        undesired_sensor = task.scene.sensors[undesired_sensor_cfg.name]
        undesired_body_ids = undesired_sensor_cfg.body_ids
        if isinstance(undesired_body_ids, slice):
            undesired_body_ids = list(range(len(undesired_sensor.body_names)))[undesired_body_ids]
        undesired_body_names = [undesired_sensor.body_names[index] for index in undesired_body_ids]
        undesired_body_counts = {name: 0 for name in undesired_body_names}
        undesired_body_peak_force = {name: 0.0 for name in undesired_body_names}
        monitored_body_pairs = {}
        for first_name, second_name in (
            ("gripper_stator", "lidar_link"),
            ("gripper_mover", "lidar_link"),
            ("link2", "lidar_link"),
            ("link3", "lidar_link"),
        ):
            first_ids = robot.find_bodies(first_name)[0]
            second_ids = robot.find_bodies(second_name)[0]
            if len(first_ids) > 0 and len(second_ids) > 0:
                monitored_body_pairs[f"{first_name}__{second_name}"] = {
                    "first_id": first_ids[0],
                    "second_id": second_ids[0],
                    "minimum_distance": torch.full(
                        (task.num_envs,), math.inf, device=task.device
                    ),
                }
        action_abs_sum = torch.zeros(3, device=task.device)
        action_element_count = torch.zeros(3, device=task.device)
        door = task.scene.articulations.get("door")
        if door is not None:
            handle_body_idx = door.find_bodies("handle_grasp")[0][0]
            door_joint_idx = door.find_joints("door_hinge")[0][0]
            handle_joint_idx = door.find_joints("handle_joint")[0][0]
            gripper_joint_idx = robot.find_joints("gripper_joint")[0][0]
            gripper_sensor_ids = [
                index
                for index, name in enumerate(undesired_sensor.body_names)
                if name in {"gripper_stator", "gripper_mover"}
            ]
            minimum_tcp_handle_distance = torch.full(
                (task.num_envs,), math.inf, device=task.device
            )
            maximum_gripper_contact_force = torch.zeros(task.num_envs, device=task.device)
            gripper_contact_steps = torch.zeros(task.num_envs, device=task.device)

        for step in range(args_cli.steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                dones = dones.bool()
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)

            if step + 1 == settle_steps:
                target_delta = command.goal_pose_w[:, :3] - command.start_pose_w[:, :3]
                target_translation = torch.linalg.vector_norm(target_delta, dim=-1).clone()
                target_planar_translation = torch.linalg.vector_norm(target_delta[:, :2], dim=-1).clone()
                target_vertical_translation = target_delta[:, 2].clone()
                _, settled_pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
                settled_root_pitch.copy_(settled_pitch)

            valid = alive & ~dones
            newly_done = alive & dones
            for pair in monitored_body_pairs.values():
                center_distance = torch.linalg.vector_norm(
                    robot.data.body_pos_w[:, pair["first_id"]]
                    - robot.data.body_pos_w[:, pair["second_id"]],
                    dim=-1,
                )
                pair["minimum_distance"][alive] = torch.minimum(
                    pair["minimum_distance"][alive], center_distance[alive]
                )
            if torch.any(newly_done):
                termination_steps.append(
                    torch.full(
                        (int(torch.count_nonzero(newly_done).item()),),
                        step + 1,
                        dtype=torch.float32,
                        device=task.device,
                    )
                )
                for name in termination_counts:
                    cause = task.termination_manager.get_term(name).bool()
                    termination_counts[name] += int(torch.count_nonzero(newly_done & cause).item())
                undesired_done = newly_done & task.termination_manager.get_term("undesired_contact").bool()
                if torch.any(undesired_done):
                    contact_force = torch.linalg.vector_norm(
                        undesired_sensor.data.net_forces_w_history[:, :, undesired_body_ids, :], dim=-1
                    ).amax(dim=1)
                    for body_index, body_name in enumerate(undesired_body_names):
                        impacted = undesired_done & (contact_force[:, body_index] > undesired_cfg.params["threshold"])
                        undesired_body_counts[body_name] += int(torch.count_nonzero(impacted).item())
                        if torch.any(impacted):
                            undesired_body_peak_force[body_name] = max(
                                undesired_body_peak_force[body_name],
                                _as_float(torch.max(contact_force[impacted, body_index])),
                            )
            if step + 1 > settle_steps and torch.any(valid):
                reference_position_errors.append(command.metrics["reference_position_error"][valid].clone())
                reference_orientation_errors.append(command.metrics["reference_orientation_error"][valid].clone())
                spatial_valid = valid & initial_spatial_mask
                planar_valid = valid & ~initial_spatial_mask
                if torch.any(spatial_valid):
                    spatial_reference_position_errors.append(
                        command.metrics["reference_position_error"][spatial_valid].clone()
                    )
                if torch.any(planar_valid):
                    planar_reference_position_errors.append(
                        command.metrics["reference_position_error"][planar_valid].clone()
                    )

            if door is not None and torch.any(alive):
                tcp_handle_distance = torch.linalg.vector_norm(
                    robot.data.body_pos_w[:, command.body_idx]
                    - door.data.body_pos_w[:, handle_body_idx],
                    dim=-1,
                )
                minimum_tcp_handle_distance[alive] = torch.minimum(
                    minimum_tcp_handle_distance[alive], tcp_handle_distance[alive]
                )
                if gripper_sensor_ids:
                    gripper_force = torch.linalg.vector_norm(
                        undesired_sensor.data.net_forces_w[:, gripper_sensor_ids, :], dim=-1
                    ).amax(dim=-1)
                    maximum_gripper_contact_force[alive] = torch.maximum(
                        maximum_gripper_contact_force[alive], gripper_force[alive]
                    )
                    gripper_contact_steps[alive] += (
                        gripper_force[alive] > 1.0
                    ).float()

            action_abs_sum[0] += torch.sum(torch.abs(actions[alive, :12]))
            action_abs_sum[1] += torch.sum(torch.abs(actions[alive, 12:18]))
            action_abs_sum[2] += torch.sum(torch.abs(actions[alive, 18:22]))
            action_element_count += torch.tensor(
                [12 * int(torch.count_nonzero(alive)), 6 * int(torch.count_nonzero(alive)), 4 * int(torch.count_nonzero(alive))],
                device=task.device,
            )

            if torch.any(valid):
                minimum_root_height = min(
                    minimum_root_height, _as_float(torch.min(robot.data.root_pos_w[valid, 2]))
                )
                tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[valid, 2], -1.0, 1.0))
                maximum_tilt = max(maximum_tilt, _as_float(torch.max(tilt)))
            alive &= ~dones

        final_position_error = command.metrics["goal_position_error"][alive]
        final_orientation_error = command.metrics["goal_orientation_error"][alive]
        final_success = (final_position_error <= 0.03) & (final_orientation_error <= math.radians(6.0))
        root_displacement = torch.linalg.vector_norm(
            robot.data.root_pos_w[alive, :2] - initial_root_xy[alive], dim=-1
        )
        _, final_root_pitch_all, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        final_root_pitch = final_root_pitch_all[alive]
        final_root_pitch_delta = final_root_pitch - settled_root_pitch[alive]
        spatial_alive = alive & initial_spatial_mask
        planar_alive = alive & ~initial_spatial_mask
        spatial_total = int(torch.count_nonzero(initial_spatial_mask).item())
        planar_total = int(task.num_envs - spatial_total)

        report = {
            "task": args_cli.task,
            "checkpoint": str(Path(args_cli.checkpoint).expanduser().resolve()),
            "seed": args_cli.seed,
            "num_envs": task.num_envs,
            "requested_steps": args_cli.steps,
            "step_dt_s": task.step_dt,
            "settle_steps": settle_steps,
            "first_episode_survival": {
                "completed_envs": int(torch.count_nonzero(alive).item()),
                "terminated_envs": int(task.num_envs - torch.count_nonzero(alive).item()),
                "completion_rate": _as_float(torch.mean(alive.float())),
                "minimum_root_height_m": minimum_root_height,
                "maximum_tilt_deg": math.degrees(maximum_tilt),
                "termination_counts": termination_counts,
                "undesired_contact_by_body": {
                    name: {
                        "terminated_env_count": undesired_body_counts[name],
                        "peak_force_n": undesired_body_peak_force[name],
                    }
                    for name in undesired_body_names
                    if undesired_body_counts[name] > 0
                },
                "termination_step": _summary(
                    torch.cat(termination_steps) if termination_steps else torch.empty(0)
                ),
            },
            "first_episode_completion_by_target_type": {
                "spatial": {
                    "completed_envs": int(torch.count_nonzero(spatial_alive).item()),
                    "total_envs": spatial_total,
                    "completion_rate": (
                        _as_float(torch.count_nonzero(spatial_alive).float() / spatial_total)
                        if spatial_total else math.nan
                    ),
                },
                "planar_replay": {
                    "completed_envs": int(torch.count_nonzero(planar_alive).item()),
                    "total_envs": planar_total,
                    "completion_rate": (
                        _as_float(torch.count_nonzero(planar_alive).float() / planar_total)
                        if planar_total else math.nan
                    ),
                },
            },
            "sampled_target_translation_m": _summary(target_translation if target_translation is not None else torch.empty(0)),
            "sampled_target_planar_translation_m": _summary(
                target_planar_translation if target_planar_translation is not None else torch.empty(0)
            ),
            "sampled_target_vertical_translation_m": _signed_summary(
                target_vertical_translation if target_vertical_translation is not None else torch.empty(0)
            ),
            "motion_reference_position_error_m": _summary(
                torch.cat(reference_position_errors) if reference_position_errors else torch.empty(0)
            ),
            "motion_reference_position_error_by_target_type_m": {
                "spatial": _summary(
                    torch.cat(spatial_reference_position_errors)
                    if spatial_reference_position_errors
                    else torch.empty(0)
                ),
                "planar_replay": _summary(
                    torch.cat(planar_reference_position_errors)
                    if planar_reference_position_errors
                    else torch.empty(0)
                ),
            },
            "motion_reference_orientation_error_deg": {
                key: math.degrees(value)
                for key, value in _summary(
                    torch.cat(reference_orientation_errors) if reference_orientation_errors else torch.empty(0)
                ).items()
            },
            "final_goal_position_error_m": _summary(final_position_error),
            "final_goal_orientation_error_deg": {
                key: math.degrees(value) for key, value in _summary(final_orientation_error).items()
            },
            "final_success_rate_3cm_6deg_among_survivors": (
                _as_float(torch.mean(final_success.float())) if final_success.numel() else 0.0
            ),
            "root_planar_displacement_m": _summary(root_displacement),
            "final_root_height_m": _signed_summary(robot.data.root_pos_w[alive, 2]),
            "final_root_pitch_deg": {
                key: math.degrees(value) for key, value in _signed_summary(final_root_pitch).items()
            },
            "final_root_pitch_delta_deg": {
                key: math.degrees(value) for key, value in _signed_summary(final_root_pitch_delta).items()
            },
            "final_root_pitch_delta_by_target_type_deg": {
                "spatial": {
                    key: math.degrees(value)
                    for key, value in _signed_summary(
                        final_root_pitch_all[spatial_alive] - settled_root_pitch[spatial_alive]
                    ).items()
                },
                "planar_replay": {
                    key: math.degrees(value)
                    for key, value in _signed_summary(
                        final_root_pitch_all[planar_alive] - settled_root_pitch[planar_alive]
                    ).items()
                },
            },
            "mean_absolute_normalized_action": {
                "legs": _as_float(action_abs_sum[0] / torch.clamp(action_element_count[0], min=1.0)),
                "arm": _as_float(action_abs_sum[1] / torch.clamp(action_element_count[1], min=1.0)),
                "wheels": _as_float(action_abs_sum[2] / torch.clamp(action_element_count[2], min=1.0)),
            },
            "minimum_monitored_body_pair_center_distance_m": {
                pair_name: {
                    "all_environments": _summary(pair["minimum_distance"]),
                    "completed_environments": _summary(pair["minimum_distance"][alive]),
                    "terminated_environments": _summary(pair["minimum_distance"][~alive]),
                }
                for pair_name, pair in monitored_body_pairs.items()
            },
        }
        if door is not None:
            gripper_action = task.action_manager.get_term("gripper_hold")
            latch_action = (
                task.action_manager.get_term("door_latch")
                if "door_latch" in task.action_manager.active_terms
                else None
            )
            final_tcp_handle_distance = torch.linalg.vector_norm(
                robot.data.body_pos_w[alive, command.body_idx]
                - door.data.body_pos_w[alive, handle_body_idx],
                dim=-1,
            )
            report["door_interaction"] = {
                "final_tcp_handle_center_distance_m": _summary(final_tcp_handle_distance),
                "minimum_tcp_handle_center_distance_m": _summary(
                    minimum_tcp_handle_distance[alive]
                ),
                "maximum_gripper_contact_force_n": _summary(
                    maximum_gripper_contact_force[alive]
                ),
                "gripper_contact_duration_s": _summary(
                    gripper_contact_steps[alive] * task.step_dt
                ),
                "final_gripper_joint_position_rad": _signed_summary(
                    robot.data.joint_pos[alive, gripper_joint_idx]
                ),
                "final_handle_angle_rad": _signed_summary(
                    door.data.joint_pos[alive, handle_joint_idx]
                ),
                "handle_position_target_rad": _signed_summary(
                    door.data.joint_pos_target[alive, handle_joint_idx]
                ),
                "handle_applied_torque_nm": _signed_summary(
                    door.data.applied_torque[alive, handle_joint_idx]
                ),
                "handle_joint_stiffness_nm_per_rad": _signed_summary(
                    door.data.joint_stiffness[alive, handle_joint_idx]
                ),
                "handle_joint_damping_nm_s_per_rad": _signed_summary(
                    door.data.joint_damping[alive, handle_joint_idx]
                ),
                "final_door_angle_rad": _signed_summary(
                    door.data.joint_pos[alive, door_joint_idx]
                ),
            }
            if hasattr(gripper_action, "closure_progress"):
                report["door_interaction"]["gripper_closure_progress"] = _signed_summary(
                    gripper_action.closure_progress[alive]
                )
            if latch_action is not None:
                report["door_interaction"]["latch_unlocked_rate"] = _as_float(
                    torch.mean(latch_action.unlocked.float())
                )
            if "compliant_grasp" in task.action_manager.active_terms:
                compliant_grasp = task.action_manager.get_term("compliant_grasp")
                report["door_interaction"]["compliant_grasp_active_rate"] = _as_float(
                    torch.mean(compliant_grasp.active.float())
                )
                report["door_interaction"]["compliant_grasp_maximum_force_n"] = _summary(
                    compliant_grasp.maximum_force
                )
            if hasattr(command, "pull_started"):
                door_task_success = (
                    alive
                    & (door.data.joint_pos[:, door_joint_idx] >= 0.35)
                    & (door.data.joint_pos[:, handle_joint_idx] >= 0.25)
                    & (
                        torch.linalg.vector_norm(
                            robot.data.body_pos_w[:, command.body_idx]
                            - door.data.body_pos_w[:, handle_body_idx],
                            dim=-1,
                        )
                        <= 0.08
                    )
                )
                report["door_interaction"]["door_task_success"] = {
                    "successful_envs": int(torch.count_nonzero(door_task_success).item()),
                    "total_envs": task.num_envs,
                    "success_rate": _as_float(torch.mean(door_task_success.float())),
                    "criteria": {
                        "door_angle_min_rad": 0.35,
                        "handle_angle_min_rad": 0.25,
                        "tcp_handle_distance_max_m": 0.08,
                        "first_episode_alive": True,
                    },
                }
                if task.num_envs <= 32:
                    jitter = getattr(
                        command,
                        "target_jitter_w",
                        torch.zeros((task.num_envs, 3), device=task.device),
                    )
                    compliant_grasp = task.action_manager.get_term("compliant_grasp")
                    final_distance_all = torch.linalg.vector_norm(
                        robot.data.body_pos_w[:, command.body_idx]
                        - door.data.body_pos_w[:, handle_body_idx],
                        dim=-1,
                    )
                    report["door_interaction"]["per_environment"] = [
                        {
                            "env_id": env_id,
                            "target_jitter_w_m": [float(value) for value in jitter[env_id].cpu().tolist()],
                            "alive": bool(alive[env_id].item()),
                            "door_angle_rad": float(door.data.joint_pos[env_id, door_joint_idx].item()),
                            "handle_angle_rad": float(door.data.joint_pos[env_id, handle_joint_idx].item()),
                            "tcp_handle_distance_m": float(final_distance_all[env_id].item()),
                            "spring_maximum_force_n": float(compliant_grasp.maximum_force[env_id].item()),
                            "door_effort_nm": float(compliant_grasp.door_effort[env_id].item()),
                            "door_effort_target_nm": float(
                                door.data.joint_effort_target[env_id, door_joint_idx].item()
                            ),
                            "door_applied_torque_nm": float(
                                door.data.applied_torque[env_id, door_joint_idx].item()
                            ),
                            "door_joint_friction": float(
                                door.data.joint_friction_coeff[env_id, door_joint_idx].item()
                            ),
                            "door_joint_stiffness": float(
                                door.data.joint_stiffness[env_id, door_joint_idx].item()
                            ),
                            "environment_origin_w_m": [
                                float(value) for value in task.scene.env_origins[env_id].cpu().tolist()
                            ],
                            "maximum_door_effort_nm": float(
                                compliant_grasp.maximum_door_effort[env_id].item()
                            ),
                            "handle_effort_nm": float(compliant_grasp.handle_effort[env_id].item()),
                            "grasp_active": bool(compliant_grasp.active[env_id].item()),
                        }
                        for env_id in range(task.num_envs)
                    ]
        print("B2W_Z1_E2E_EE_WBC_EVAL=" + json.dumps(report, indent=2, sort_keys=True), flush=True)
        if args_cli.report:
            report_path = Path(args_cli.report).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
