#!/usr/bin/env python3
"""Run the deterministic arm-only TCP tracking gate for B2-W + Z1.

The B2-W root is fixed deliberately. This separates Z1 kinematics, Jacobian
indexing, coordinate conventions, actuator response, and terminal jitter from
future wheel/body coordination. The command is an absolute pose in the B2-W
root frame, and the tracked body is the explicit ``tcp_frame``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1, help="Parallel copies of the same gate test.")
parser.add_argument("--settle_steps", type=int, default=100, help="Home-pose settling steps before each goal.")
parser.add_argument("--goal_steps", type=int, default=800, help="Control steps allowed for each goal.")
parser.add_argument("--window_steps", type=int, default=100, help="Terminal window used for RMS/jitter metrics.")
parser.add_argument("--max_position_error", type=float, default=0.005, help="Maximum settled RMS error in metres.")
parser.add_argument(
    "--max_orientation_error_deg", type=float, default=2.0, help="Maximum settled RMS orientation error."
)
parser.add_argument("--max_position_jitter", type=float, default=0.002, help="Maximum terminal TCP jitter in metres.")
parser.add_argument("--max_linear_speed", type=float, default=0.03, help="Maximum terminal TCP RMS speed in m/s.")
parser.add_argument("--max_angular_speed", type=float, default=0.15, help="Maximum terminal TCP RMS speed in rad/s.")
parser.add_argument("--max_joint_speed", type=float, default=2.0, help="DIK joint-target rate limit in rad/s.")
parser.add_argument("--ik_gain", type=float, default=0.2, help="Per-step gain applied to the DLS joint correction.")
parser.add_argument("--report", type=Path, default=None, help="Optional JSON report output path.")
parser.add_argument("--debug", action="store_true", help="Print controller state for the first goal.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1:
    parser.error("--num_envs must be positive")
if args_cli.window_steps < 2 or args_cli.window_steps > args_cli.goal_steps:
    parser.error("--window_steps must be in [2, goal_steps]")

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
)


GOALS = (
    ("forward", (0.040, 0.000, 0.000, 0.000, 0.000, 0.000)),
    ("lateral", (0.000, 0.035, 0.000, 0.000, 0.000, 0.000)),
    ("lower", (0.000, 0.000, -0.035, 0.000, 0.000, 0.000)),
    ("yaw", (0.000, 0.000, 0.000, 0.000, 0.000, math.radians(8.0))),
    ("combined", (-0.025, 0.025, 0.025, math.radians(4.0), -math.radians(4.0), math.radians(5.0))),
)


@configclass
class B2WZ1TcpGateSceneCfg(InteractiveSceneCfg):
    """Replicated B2-W + Z1 scene used by the deterministic gate."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0),
    )
    robot = B2W_Z1_TRAINING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _pose_in_root(robot: Articulation, tcp_body_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    tcp_pose_w = robot.data.body_pose_w[:, tcp_body_id]
    root_pose_w = robot.data.root_pose_w
    return subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        tcp_pose_w[:, 0:3],
        tcp_pose_w[:, 3:7],
    )


def _jacobian_in_root(
    robot: Articulation, tcp_jacobian_id: int, arm_joint_ids: list[int]
) -> torch.Tensor:
    jacobian = robot.root_physx_view.get_jacobians()[:, tcp_jacobian_id, :, arm_joint_ids].clone()
    root_rotation_inverse = matrix_from_quat(quat_inv(robot.data.root_quat_w))
    jacobian[:, :3, :] = torch.bmm(root_rotation_inverse, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(root_rotation_inverse, jacobian[:, 3:, :])
    return jacobian


def _rms(values: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(values * values)))


def _reset_to_home(
    robot: Articulation,
    sim: sim_utils.SimulationContext,
    arm_joint_ids: list[int],
    default_joint_pos: torch.Tensor,
    default_joint_vel: torch.Tensor,
    settle_steps: int,
) -> None:
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.set_joint_position_target(default_joint_pos)
    robot.set_joint_velocity_target(default_joint_vel)
    robot.write_data_to_sim()
    robot.reset()
    for _ in range(settle_steps):
        robot.set_joint_position_target(default_joint_pos)
        robot.set_joint_velocity_target(default_joint_vel)
        gravity_effort = robot.root_physx_view.get_gravity_compensation_forces()
        robot.set_joint_effort_target(gravity_effort[:, arm_joint_ids], joint_ids=arm_joint_ids)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())


def _run_goal(
    *,
    name: str,
    delta_values: tuple[float, float, float, float, float, float],
    robot: Articulation,
    sim: sim_utils.SimulationContext,
    controller: DifferentialIKController,
    arm_joint_ids: list[int],
    tcp_body_id: int,
    tcp_jacobian_id: int,
    default_joint_pos: torch.Tensor,
    default_joint_vel: torch.Tensor,
    joint_limits: torch.Tensor,
) -> dict[str, float | int | str | bool]:
    _reset_to_home(robot, sim, arm_joint_ids, default_joint_pos, default_joint_vel, args_cli.settle_steps)
    start_root_pose = robot.data.root_pose_w.clone()
    start_pos_b, start_quat_b = _pose_in_root(robot, tcp_body_id)

    delta = torch.tensor(delta_values, device=sim.device, dtype=start_pos_b.dtype).repeat(args_cli.num_envs, 1)
    # With multiple environments, exercise both signs of each displacement.
    signs = torch.where(
        torch.arange(args_cli.num_envs, device=sim.device) % 2 == 0,
        torch.ones(args_cli.num_envs, device=sim.device),
        -torch.ones(args_cli.num_envs, device=sim.device),
    )
    delta *= signs.unsqueeze(-1)
    target_pos_b, target_quat_b = apply_delta_pose(start_pos_b, start_quat_b, delta)
    target_command = torch.cat((target_pos_b, target_quat_b), dim=-1)
    frozen_target = target_command.clone()
    controller.reset()
    controller.set_command(target_command)

    position_errors = []
    orientation_errors = []
    positions = []
    linear_speeds = []
    angular_speeds = []
    minimum_singular_value = math.inf
    clamp_count = 0
    max_joint_step = args_cli.max_joint_speed * sim.get_physics_dt()

    for step_index in range(args_cli.goal_steps):
        current_pos_b, current_quat_b = _pose_in_root(robot, tcp_body_id)
        jacobian = _jacobian_in_root(robot, tcp_jacobian_id, arm_joint_ids)
        if not torch.all(torch.isfinite(jacobian)):
            raise AssertionError(f"{name}: TCP Jacobian contains NaN or Inf")
        singular_values = torch.linalg.svdvals(jacobian)
        minimum_singular_value = min(minimum_singular_value, float(torch.min(singular_values)))
        current_joint_pos = robot.data.joint_pos[:, arm_joint_ids]
        unconstrained_target = controller.compute(
            current_pos_b, current_quat_b, jacobian, current_joint_pos
        )
        if args_cli.debug and name == GOALS[0][0] and step_index in (0, 1, 2, 10, 100, args_cli.goal_steps - 1):
            pos_error_debug, rot_error_debug = compute_pose_error(
                current_pos_b,
                current_quat_b,
                target_pos_b,
                target_quat_b,
                rot_error_type="axis_angle",
            )
            print(
                f"DEBUG step={step_index} pos={current_pos_b[0].tolist()} "
                f"target={target_pos_b[0].tolist()} pos_error={pos_error_debug[0].tolist()} "
                f"rot_error={rot_error_debug[0].tolist()} q={current_joint_pos[0].tolist()} "
                f"q_unconstrained={unconstrained_target[0].tolist()} "
                f"jacobian={jacobian[0].tolist()}",
                flush=True,
            )
        # Apply a fraction of the DLS correction around the measured state.
        # Gravity is handled separately as explicit feed-forward effort.
        target_delta = torch.clamp(
            args_cli.ik_gain * (unconstrained_target - current_joint_pos),
            min=-max_joint_step,
            max=max_joint_step,
        )
        limited_target = current_joint_pos + target_delta
        clamped_target = torch.maximum(
            torch.minimum(limited_target, joint_limits[..., 1]), joint_limits[..., 0]
        )
        clamp_count += int(torch.count_nonzero(torch.abs(clamped_target - limited_target) > 1.0e-7))

        robot.set_joint_position_target(default_joint_pos)
        robot.set_joint_velocity_target(default_joint_vel)
        robot.set_joint_position_target(clamped_target, joint_ids=arm_joint_ids)
        gravity_effort = robot.root_physx_view.get_gravity_compensation_forces()
        robot.set_joint_effort_target(gravity_effort[:, arm_joint_ids], joint_ids=arm_joint_ids)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())

        current_pos_b, current_quat_b = _pose_in_root(robot, tcp_body_id)
        pos_error, rot_error = compute_pose_error(
            current_pos_b,
            current_quat_b,
            target_pos_b,
            target_quat_b,
            rot_error_type="axis_angle",
        )
        position_errors.append(torch.linalg.vector_norm(pos_error, dim=-1))
        orientation_errors.append(torch.linalg.vector_norm(rot_error, dim=-1))
        positions.append(current_pos_b.clone())
        linear_speeds.append(
            torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, tcp_body_id], dim=-1)
        )
        angular_speeds.append(
            torch.linalg.vector_norm(robot.data.body_ang_vel_w[:, tcp_body_id], dim=-1)
        )

    if not torch.equal(target_command, frozen_target):
        raise AssertionError(f"{name}: absolute target was mutated during tracking")

    position_error_history = torch.stack(position_errors[-args_cli.window_steps :])
    orientation_error_history = torch.stack(orientation_errors[-args_cli.window_steps :])
    position_history = torch.stack(positions[-args_cli.window_steps :])
    linear_speed_history = torch.stack(linear_speeds[-args_cli.window_steps :])
    angular_speed_history = torch.stack(angular_speeds[-args_cli.window_steps :])
    position_center = torch.mean(position_history, dim=0, keepdim=True)
    position_jitter_by_env = torch.sqrt(
        torch.mean(torch.sum((position_history - position_center) ** 2, dim=-1), dim=0)
    )
    root_pos_error, root_rot_error = compute_pose_error(
        start_root_pose[:, 0:3],
        start_root_pose[:, 3:7],
        robot.data.root_pose_w[:, 0:3],
        robot.data.root_pose_w[:, 3:7],
        rot_error_type="axis_angle",
    )

    metrics = {
        "goal": name,
        "position_error_rms_m": _rms(position_error_history),
        "position_error_max_m": float(torch.max(position_error_history)),
        "orientation_error_rms_deg": math.degrees(_rms(orientation_error_history)),
        "orientation_error_max_deg": math.degrees(float(torch.max(orientation_error_history))),
        "position_jitter_max_m": float(torch.max(position_jitter_by_env)),
        "linear_speed_rms_mps": _rms(linear_speed_history),
        "angular_speed_rms_radps": _rms(angular_speed_history),
        "base_position_drift_max_m": float(torch.max(torch.linalg.vector_norm(root_pos_error, dim=-1))),
        "base_orientation_drift_max_deg": math.degrees(
            float(torch.max(torch.linalg.vector_norm(root_rot_error, dim=-1)))
        ),
        "minimum_jacobian_singular_value": minimum_singular_value,
        "joint_limit_clamp_count": clamp_count,
    }
    metrics["passed"] = bool(
        metrics["position_error_rms_m"] <= args_cli.max_position_error
        and metrics["orientation_error_rms_deg"] <= args_cli.max_orientation_error_deg
        and metrics["position_jitter_max_m"] <= args_cli.max_position_jitter
        and metrics["linear_speed_rms_mps"] <= args_cli.max_linear_speed
        and metrics["angular_speed_rms_radps"] <= args_cli.max_angular_speed
        and metrics["base_position_drift_max_m"] <= 1.0e-5
        and metrics["base_orientation_drift_max_deg"] <= 1.0e-3
    )
    return metrics


def main() -> None:
    start_time = time.perf_counter()
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = B2WZ1TcpGateSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene_cfg.robot.spawn.articulation_props.fix_root_link = True
    # The nominal controller gate is deterministic. Actuation delay belongs in
    # a later robustness test after the nominal servo is known to converge.
    for actuator_cfg in scene_cfg.robot.actuators.values():
        actuator_cfg.min_delay = 0
        actuator_cfg.max_delay = 0
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.update(sim.get_physics_dt())
    robot = scene["robot"]

    if not robot.is_fixed_base:
        raise AssertionError("The arm-only gate must run with a fixed B2-W root")
    arm_joint_ids, arm_joint_names = robot.find_joints("joint[1-6]")
    tcp_body_ids, tcp_body_names = robot.find_bodies("tcp_frame")
    wheel_joint_ids, wheel_joint_names = robot.find_joints(".*_wheel_joint")
    if arm_joint_names != [f"joint{index}" for index in range(1, 7)]:
        raise AssertionError(f"Unexpected Z1 joint order: {arm_joint_names}")
    if tcp_body_names != ["tcp_frame"]:
        raise AssertionError(f"Unexpected TCP body lookup: {tcp_body_names}")
    if len(wheel_joint_names) != 4:
        raise AssertionError(f"Expected four wheel joints, got {wheel_joint_names}")

    tcp_body_id = tcp_body_ids[0]
    tcp_jacobian_id = tcp_body_id - 1
    jacobian_shape = tuple(robot.root_physx_view.get_jacobians().shape)
    if tcp_jacobian_id < 0 or tcp_jacobian_id >= jacobian_shape[1]:
        raise AssertionError(
            f"Invalid TCP Jacobian index {tcp_jacobian_id} for shape {jacobian_shape}"
        )

    controller_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.03},
    )
    controller = DifferentialIKController(controller_cfg, args_cli.num_envs, sim.device)
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    joint_limits = robot.data.soft_joint_pos_limits[:, arm_joint_ids, :].clone()

    goal_results = []
    for goal_name, goal_delta in GOALS:
        result = _run_goal(
            name=goal_name,
            delta_values=goal_delta,
            robot=robot,
            sim=sim,
            controller=controller,
            arm_joint_ids=arm_joint_ids,
            tcp_body_id=tcp_body_id,
            tcp_jacobian_id=tcp_jacobian_id,
            default_joint_pos=default_joint_pos,
            default_joint_vel=default_joint_vel,
            joint_limits=joint_limits,
        )
        goal_results.append(result)
        print(
            f"{'PASS' if result['passed'] else 'FAIL'} {goal_name}: "
            f"position={1000.0 * result['position_error_rms_m']:.2f} mm RMS, "
            f"orientation={result['orientation_error_rms_deg']:.2f} deg RMS, "
            f"jitter={1000.0 * result['position_jitter_max_m']:.2f} mm, "
            f"speed={result['linear_speed_rms_mps']:.4f} m/s",
            flush=True,
        )

    report = {
        "gate": "fixed_base_gravity_compensated_differential_ik",
        "passed": all(bool(result["passed"]) for result in goal_results),
        "device": args_cli.device,
        "num_envs": args_cli.num_envs,
        "physics_dt_s": sim.get_physics_dt(),
        "settle_steps": args_cli.settle_steps,
        "goal_steps": args_cli.goal_steps,
        "terminal_window_steps": args_cli.window_steps,
        "controller": {
            "method": "damped_least_squares",
            "damping": 0.03,
            "per_step_gain": args_cli.ik_gain,
            "joint_target_rate_limit_radps": args_cli.max_joint_speed,
            "gravity_feedforward": True,
            "actuator_delay_steps": 0,
            "fixed_base": True,
        },
        "arm_joint_names": arm_joint_names,
        "wheel_joint_names": wheel_joint_names,
        "tcp_body_id": tcp_body_id,
        "tcp_jacobian_id": tcp_jacobian_id,
        "jacobian_shape": jacobian_shape,
        "thresholds": {
            "position_error_m": args_cli.max_position_error,
            "orientation_error_deg": args_cli.max_orientation_error_deg,
            "position_jitter_m": args_cli.max_position_jitter,
            "linear_speed_mps": args_cli.max_linear_speed,
            "angular_speed_radps": args_cli.max_angular_speed,
        },
        "goals": goal_results,
        "elapsed_wall_time_s": time.perf_counter() - start_time,
    }
    report_text = json.dumps(report, indent=2, sort_keys=True)
    if args_cli.report is not None:
        args_cli.report.parent.mkdir(parents=True, exist_ok=True)
        args_cli.report.write_text(report_text + "\n", encoding="utf-8")
        print(f"Report: {args_cli.report.resolve()}", flush=True)
    print(report_text, flush=True)
    if not report["passed"]:
        failed_goals = [result["goal"] for result in goal_results if not result["passed"]]
        raise AssertionError(f"B2-W + Z1 arm-only TCP gate failed: {failed_goals}")


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
