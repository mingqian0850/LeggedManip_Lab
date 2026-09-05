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
        settle_steps = math.ceil(command.cfg.settle_time_s / task.step_dt)
        alive = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
        initial_root_xy = robot.data.root_pos_w[:, :2].clone()
        target_translation = None
        target_planar_translation = None
        reference_position_errors = []
        reference_orientation_errors = []
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
        action_abs_sum = torch.zeros(3, device=task.device)
        action_element_count = torch.zeros(3, device=task.device)

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

            valid = alive & ~dones
            newly_done = alive & dones
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
            "sampled_target_translation_m": _summary(target_translation if target_translation is not None else torch.empty(0)),
            "sampled_target_planar_translation_m": _summary(
                target_planar_translation if target_planar_translation is not None else torch.empty(0)
            ),
            "motion_reference_position_error_m": _summary(
                torch.cat(reference_position_errors) if reference_position_errors else torch.empty(0)
            ),
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
            "mean_absolute_normalized_action": {
                "legs": _as_float(action_abs_sum[0] / torch.clamp(action_element_count[0], min=1.0)),
                "arm": _as_float(action_abs_sum[1] / torch.clamp(action_element_count[1], min=1.0)),
                "wheels": _as_float(action_abs_sum[2] / torch.clamp(action_element_count[2], min=1.0)),
            },
        }
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
