"""Finite action/observation/reward smoke test for unified B2W + Z1 EE-WBC."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="B2W-Z1-EE-WBC-Flat-Play-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mode", choices=("zero", "mapping"), default="zero")
parser.add_argument("--report", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json
import math
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import LeggedManip_Lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def main() -> dict:
    if args_cli.task not in gym.registry:
        raise RuntimeError(f"Task is not registered: {args_cli.task}")
    if args_cli.num_envs < 1 or args_cli.steps < 1:
        raise ValueError("num_envs and steps must be positive")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    task = env.unwrapped

    try:
        observations, _ = env.reset(seed=args_cli.seed)
        robot = task.scene["robot"]
        action_terms = {
            name: task.action_manager.get_term(name)
            for name in ("leg_position", "arm_position", "wheel_velocity")
        }
        expected_term_dims = {"leg_position": 12, "arm_position": 6, "wheel_velocity": 4}
        initial_joint_position = robot.data.joint_pos.clone()
        zeros = torch.zeros((task.num_envs, 22), device=task.device)

        # Every element is non-zero and unique enough to expose ordering or
        # accidentally omitted actuators.  Keep the magnitude small because
        # this is a mapping test, not a learned controller.
        probe = torch.linspace(-0.12, 0.12, 22, device=task.device)
        probe[torch.abs(probe) < 1.0e-4] = 0.03
        probe = probe.unsqueeze(0).repeat(task.num_envs, 1)
        action = probe if args_cli.mode == "mapping" else zeros

        finite = all(torch.all(torch.isfinite(value)) for value in observations.values())
        minimum_root_height = math.inf
        maximum_tilt_rad = 0.0
        terminated_count = 0
        truncated_count = 0
        termination_counts = {name: 0 for name in task.termination_manager.active_terms}
        cumulative_reward = torch.zeros(task.num_envs, device=task.device)

        for _ in range(args_cli.steps):
            with torch.inference_mode():
                observations, reward, terminated, truncated, _ = env.step(action)
            tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
            finite = finite and bool(
                torch.all(torch.isfinite(action))
                and torch.all(torch.isfinite(reward))
                and torch.all(torch.isfinite(robot.data.joint_pos))
                and all(torch.all(torch.isfinite(value)) for value in observations.values())
            )
            minimum_root_height = min(minimum_root_height, _scalar(torch.min(robot.data.root_pos_w[:, 2])))
            maximum_tilt_rad = max(maximum_tilt_rad, _scalar(torch.max(tilt)))
            terminated_count += int(torch.count_nonzero(terminated).item())
            truncated_count += int(torch.count_nonzero(truncated).item())
            for name in termination_counts:
                termination_counts[name] += int(
                    torch.count_nonzero(task.termination_manager.get_term(name)).item()
                )
            cumulative_reward += reward

        term_dims = {name: term.action_dim for name, term in action_terms.items()}
        processed_action_nonzero = {
            name: int(torch.count_nonzero(torch.abs(term.raw_actions[0]) > 1.0e-6).item())
            for name, term in action_terms.items()
        }
        controlled_joint_delta = torch.abs(robot.data.joint_pos - initial_joint_position)
        reward_terms = {
            name: float(values[0]) for name, values in task.reward_manager.get_active_iterable_terms(0)
        }
        reward_terms_finite = all(math.isfinite(value) for value in reward_terms.values())

        checks = {
            "action_shape_is_22": tuple(env.action_space.shape) == (task.num_envs, 22),
            "action_term_dims_are_12_6_4": term_dims == expected_term_dims,
            "policy_observation_is_matrix": observations["policy"].ndim == 2,
            "privileged_observation_is_matrix": observations["privileged"].ndim == 2,
            "finite": bool(finite),
            "reward_terms_finite": reward_terms_finite,
            "no_early_reset": terminated_count == 0 and truncated_count == 0,
            "root_above_termination_height": minimum_root_height >= 0.45,
        }
        if args_cli.mode == "mapping":
            checks["all_22_action_targets_receive_nonzero_probe"] = processed_action_nonzero == expected_term_dims
        else:
            checks["zero_action_keeps_tilt_below_20_deg"] = maximum_tilt_rad < math.radians(20.0)

        report = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "task": args_cli.task,
            "mode": args_cli.mode,
            "num_envs": task.num_envs,
            "steps": args_cli.steps,
            "step_dt_s": task.step_dt,
            "action_term_order": list(action_terms.keys()),
            "action_term_dims": term_dims,
            "observation_shapes": {name: list(value.shape) for name, value in observations.items()},
            "checks": checks,
            "metrics": {
                "minimum_root_height_m": minimum_root_height,
                "maximum_tilt_deg": math.degrees(maximum_tilt_rad),
                "terminated_count": terminated_count,
                "truncated_count": truncated_count,
                "termination_counts": termination_counts,
                "mean_episode_reward": _scalar(torch.mean(cumulative_reward)),
                "max_joint_motion_rad": _scalar(torch.max(controlled_joint_delta)),
                "nonzero_probe_elements_by_term": processed_action_nonzero,
            },
            "reward_terms_env0": reward_terms,
            "final_joint_state_env0": {
                name: {
                    "position_rad": float(robot.data.joint_pos[0, index].detach().cpu().item()),
                    "default_rad": float(robot.data.default_joint_pos[0, index].detach().cpu().item()),
                    "velocity_rad_s": float(robot.data.joint_vel[0, index].detach().cpu().item()),
                }
                for index, name in enumerate(robot.joint_names)
            },
        }
        print("B2W_Z1_E2E_EE_WBC_SMOKE=" + json.dumps(report, indent=2, sort_keys=True), flush=True)
        if args_cli.report:
            report_path = Path(args_cli.report).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if report["status"] != "PASS":
            raise RuntimeError("Unified EE-WBC smoke test failed")
        return report
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
