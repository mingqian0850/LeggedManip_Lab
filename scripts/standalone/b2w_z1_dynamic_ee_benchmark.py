"""Deterministic time-domain benchmark for B2W + Z1 unified EE-WBC."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="B2W-Z1-EE-WBC-Dynamic-Medium-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=1099)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--maximum_lag_s", type=float, default=1.0)
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.utils.math import euler_xyz_from_quat


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _finite_values(values: torch.Tensor) -> torch.Tensor:
    return values.flatten()[torch.isfinite(values.flatten())]


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = _finite_values(values)
    if values.numel() == 0:
        return {"mean": math.nan, "rms": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "mean": _as_float(torch.mean(values)),
        "rms": _as_float(torch.sqrt(torch.mean(torch.square(values)))),
        "p95": _as_float(torch.quantile(values, 0.95)),
        "max": _as_float(torch.max(values)),
    }


def _masked(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return torch.where(valid, values, torch.full_like(values, math.nan))


def _estimate_position_lag(
    target_position: torch.Tensor,
    actual_position: torch.Tensor,
    valid: torch.Tensor,
    dynamic_envs: torch.Tensor,
    step_dt: float,
    maximum_lag_s: float,
) -> torch.Tensor:
    """Estimate non-negative policy lag from target/actual Cartesian velocity correlation."""
    target_velocity = torch.diff(target_position, dim=0) / step_dt
    actual_velocity = torch.diff(actual_position, dim=0) / step_dt
    pair_valid = valid[1:] & valid[:-1]
    maximum_lag_steps = max(0, int(round(maximum_lag_s / step_dt)))
    lag = torch.full((target_position.shape[1],), math.nan, device=target_position.device)
    for env_id in torch.nonzero(dynamic_envs, as_tuple=False).squeeze(-1).tolist():
        best_score = -math.inf
        best_lag = 0
        for lag_steps in range(maximum_lag_steps + 1):
            if lag_steps == 0:
                target_slice = target_velocity[:, env_id]
                actual_slice = actual_velocity[:, env_id]
                valid_slice = pair_valid[:, env_id]
            else:
                target_slice = target_velocity[:-lag_steps, env_id]
                actual_slice = actual_velocity[lag_steps:, env_id]
                valid_slice = pair_valid[:-lag_steps, env_id] & pair_valid[lag_steps:, env_id]
            if int(torch.count_nonzero(valid_slice).item()) < 20:
                continue
            target_valid = target_slice[valid_slice]
            actual_valid = actual_slice[valid_slice]
            target_valid = target_valid - torch.mean(target_valid, dim=0, keepdim=True)
            actual_valid = actual_valid - torch.mean(actual_valid, dim=0, keepdim=True)
            denominator = torch.linalg.vector_norm(target_valid) * torch.linalg.vector_norm(actual_valid)
            if denominator <= 1.0e-8:
                continue
            score = _as_float(torch.sum(target_valid * actual_valid) / denominator)
            if score > best_score:
                best_score = score
                best_lag = lag_steps
        if best_score > -math.inf:
            lag[env_id] = best_lag * step_dt
    return lag


def main() -> dict:
    if args_cli.num_envs < 1 or args_cli.steps < 1:
        raise ValueError("num_envs and steps must be positive")
    if args_cli.maximum_lag_s < 0.0:
        raise ValueError("maximum_lag_s must be non-negative")

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
        if not hasattr(command, "trajectory_ids"):
            raise TypeError(f"Task {args_cli.task} does not use PeriodicWorldPoseCommand")
        tcp_idx = command.body_idx
        trajectory_ids = command.trajectory_ids.clone()
        trajectory_names = tuple(command.trajectory_names)
        periodic_mask = command.is_periodic_mask.clone()
        frequency_hz = command.frequency_hz.clone()
        settle_steps = math.ceil(command.cfg.settle_time_s / task.step_dt)
        measurement_start_step = math.ceil(
            (command.cfg.settle_time_s + command.cfg.ramp_time_s) / task.step_dt
        )
        if args_cli.steps <= measurement_start_step + 20:
            raise ValueError(
                f"steps must exceed ramp completion ({measurement_start_step}) by at least 20"
            )

        hip_ids, hip_names = robot.find_joints(".*_hip_joint")
        leg_ids = robot.find_joints(".*_(hip|thigh|calf)_joint")[0]
        settled_joint_position = None
        alive = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
        previous_action = torch.zeros((task.num_envs, 22), device=task.device)
        previous_root_velocity = robot.data.root_lin_vel_w.clone()
        termination_counts = {name: 0 for name in task.termination_manager.active_terms}

        position_error_series = []
        orientation_error_series = []
        linear_velocity_error_series = []
        angular_velocity_error_series = []
        tcp_speed_series = []
        root_height_series = []
        root_roll_series = []
        root_pitch_series = []
        root_speed_series = []
        root_acceleration_series = []
        hip_deviation_series = []
        action_delta_series = []
        within_tolerance_series = []
        target_position_series = []
        actual_position_series = []
        valid_series = []

        for step in range(args_cli.steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                dones = dones.bool()
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)

            if step + 1 == settle_steps:
                settled_joint_position = robot.data.joint_pos.clone()

            newly_done = alive & dones
            for name in termination_counts:
                termination_counts[name] += int(
                    torch.count_nonzero(newly_done & task.termination_manager.get_term(name).bool()).item()
                )
            valid = alive & ~dones

            if step + 1 >= measurement_start_step:
                pos_error = command.metrics["reference_position_error"]
                ori_error = command.metrics["reference_orientation_error"]
                tcp_lin_vel = robot.data.body_lin_vel_w[:, tcp_idx]
                tcp_ang_vel = robot.data.body_ang_vel_w[:, tcp_idx]
                root_roll, root_pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
                hip_deviation = torch.linalg.vector_norm(
                    robot.data.joint_pos[:, hip_ids] - robot.data.default_joint_pos[:, hip_ids], dim=-1
                )
                action_delta = torch.linalg.vector_norm(actions - previous_action, dim=-1)
                root_acceleration = torch.linalg.vector_norm(
                    (robot.data.root_lin_vel_w - previous_root_velocity) / task.step_dt, dim=-1
                )
                within_tolerance = (pos_error <= 0.03) & (ori_error <= math.radians(6.0))

                position_error_series.append(_masked(pos_error, valid))
                orientation_error_series.append(_masked(ori_error, valid))
                linear_velocity_error_series.append(
                    _masked(torch.linalg.vector_norm(tcp_lin_vel - command.twist_command_w[:, :3], dim=-1), valid)
                )
                angular_velocity_error_series.append(
                    _masked(torch.linalg.vector_norm(tcp_ang_vel - command.twist_command_w[:, 3:], dim=-1), valid)
                )
                tcp_speed_series.append(_masked(torch.linalg.vector_norm(tcp_lin_vel, dim=-1), valid))
                root_height_series.append(_masked(robot.data.root_pos_w[:, 2], valid))
                root_roll_series.append(_masked(torch.abs(root_roll), valid))
                root_pitch_series.append(_masked(torch.abs(root_pitch), valid))
                root_speed_series.append(
                    _masked(torch.linalg.vector_norm(robot.data.root_lin_vel_w, dim=-1), valid)
                )
                root_acceleration_series.append(_masked(root_acceleration, valid))
                hip_deviation_series.append(_masked(hip_deviation, valid))
                action_delta_series.append(_masked(action_delta, valid))
                within_tolerance_series.append(_masked(within_tolerance.float(), valid))
                target_position_series.append(command.pose_command_w[:, :3].clone())
                actual_position_series.append(robot.data.body_pos_w[:, tcp_idx].clone())
                valid_series.append(valid.clone())

            previous_action.copy_(actions)
            previous_root_velocity.copy_(robot.data.root_lin_vel_w)
            alive &= ~dones

        if settled_joint_position is None:
            raise RuntimeError("The run ended before the settling snapshot")

        series = {
            "position_error_m": torch.stack(position_error_series),
            "orientation_error_rad": torch.stack(orientation_error_series),
            "linear_velocity_error_m_s": torch.stack(linear_velocity_error_series),
            "angular_velocity_error_rad_s": torch.stack(angular_velocity_error_series),
            "tcp_speed_m_s": torch.stack(tcp_speed_series),
            "root_height_m": torch.stack(root_height_series),
            "root_roll_rad": torch.stack(root_roll_series),
            "root_pitch_rad": torch.stack(root_pitch_series),
            "root_speed_m_s": torch.stack(root_speed_series),
            "root_acceleration_m_s2": torch.stack(root_acceleration_series),
            "hip_deviation_norm_rad": torch.stack(hip_deviation_series),
            "action_delta_norm": torch.stack(action_delta_series),
            "within_tolerance": torch.stack(within_tolerance_series),
        }
        target_position = torch.stack(target_position_series)
        actual_position = torch.stack(actual_position_series)
        valid = torch.stack(valid_series)
        position_periodic_mask = torch.zeros_like(periodic_mask)
        for trajectory_name in ("line_x", "line_y", "circle_xy", "figure8_xy", "vertical", "six_d"):
            if trajectory_name in trajectory_names:
                position_periodic_mask |= trajectory_ids == trajectory_names.index(trajectory_name)
        lag_s = _estimate_position_lag(
            target_position,
            actual_position,
            valid,
            position_periodic_mask,
            task.step_dt,
            args_cli.maximum_lag_s,
        )

        def metrics_for(env_mask: torch.Tensor) -> dict:
            selected = {name: values[:, env_mask] for name, values in series.items()}
            tolerance_values = _finite_values(selected["within_tolerance"])
            return {
                "environment_count": int(torch.count_nonzero(env_mask).item()),
                "survived_count": int(torch.count_nonzero(alive & env_mask).item()),
                "position_error_m": _summary(selected["position_error_m"]),
                "orientation_error_deg": {
                    key: math.degrees(value)
                    for key, value in _summary(selected["orientation_error_rad"]).items()
                },
                "linear_velocity_error_m_s": _summary(selected["linear_velocity_error_m_s"]),
                "angular_velocity_error_rad_s": _summary(selected["angular_velocity_error_rad_s"]),
                "fraction_within_3cm_6deg": (
                    _as_float(torch.mean(tolerance_values)) if tolerance_values.numel() else math.nan
                ),
                "position_lag_s": _summary(lag_s[env_mask]),
                "tcp_speed_m_s": _summary(selected["tcp_speed_m_s"]),
                "root_height_m": _summary(selected["root_height_m"]),
                "absolute_root_roll_deg": {
                    key: math.degrees(value) for key, value in _summary(selected["root_roll_rad"]).items()
                },
                "absolute_root_pitch_deg": {
                    key: math.degrees(value) for key, value in _summary(selected["root_pitch_rad"]).items()
                },
                "root_speed_m_s": _summary(selected["root_speed_m_s"]),
                "root_acceleration_m_s2": _summary(selected["root_acceleration_m_s2"]),
                "hip_deviation_norm_rad": _summary(selected["hip_deviation_norm_rad"]),
                "action_delta_norm": _summary(selected["action_delta_norm"]),
            }

        report = {
            "task": args_cli.task,
            "checkpoint": str(Path(args_cli.checkpoint).expanduser().resolve()),
            "seed": args_cli.seed,
            "num_envs": task.num_envs,
            "steps": args_cli.steps,
            "step_dt_s": task.step_dt,
            "measurement_start_step": measurement_start_step,
            "trajectory_frequency_hz": _summary(frequency_hz),
            "first_episode_survival": {
                "completed_envs": int(torch.count_nonzero(alive).item()),
                "completion_rate": _as_float(torch.mean(alive.float())),
                "termination_counts": termination_counts,
            },
            "overall": metrics_for(torch.ones(task.num_envs, dtype=torch.bool, device=task.device)),
            "by_trajectory": {
                name: metrics_for(trajectory_ids == index)
                for index, name in enumerate(trajectory_names)
            },
            "settling_posture": {
                "maximum_joint_change_from_default_rad": _as_float(
                    torch.max(torch.abs(settled_joint_position - robot.data.default_joint_pos))
                ),
                "maximum_leg_change_from_default_rad": _as_float(
                    torch.max(
                        torch.abs(
                            settled_joint_position[:, leg_ids]
                            - robot.data.default_joint_pos[:, leg_ids]
                        )
                    )
                ),
                "maximum_hip_change_from_default_rad": _as_float(
                    torch.max(
                        torch.abs(
                            settled_joint_position[:, hip_ids]
                            - robot.data.default_joint_pos[:, hip_ids]
                        )
                    )
                ),
                "mean_hip_change_from_default_rad": _as_float(
                    torch.mean(
                        torch.abs(
                            settled_joint_position[:, hip_ids]
                            - robot.data.default_joint_pos[:, hip_ids]
                        )
                    )
                ),
                "hip_change_from_default_rad": {
                    name: {
                        "mean": _as_float(torch.mean(delta)),
                        "mean_absolute": _as_float(torch.mean(torch.abs(delta))),
                        "minimum": _as_float(torch.min(delta)),
                        "maximum": _as_float(torch.max(delta)),
                    }
                    for name, delta in zip(
                        hip_names,
                        (
                            settled_joint_position[:, hip_ids]
                            - robot.data.default_joint_pos[:, hip_ids]
                        ).T,
                        strict=True,
                    )
                },
            },
        }
        print("B2W_Z1_DYNAMIC_EE_BENCHMARK=" + json.dumps(report, indent=2, sort_keys=True), flush=True)
        if args_cli.report:
            path = Path(args_cli.report).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
