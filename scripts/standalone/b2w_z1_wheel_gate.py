#!/usr/bin/env python3
"""Validate nominal B2-W wheel signs and flat-ground execution dynamics.

This is a deterministic actuator/contact gate, not a learned locomotion task.
The Z1 is held in its safe stow pose while the four B2-W wheels execute
forward, reverse, counter-clockwise turn, clockwise turn, and active-braking
tests. Turns use forward arcs by default; an in-place skid-steer stress mode
is available separately. Every phase starts from the same reset state.

The gate checks the conventions that a future TCP reachability coordinator
will depend on: +X is forward, +Y is left, every wheel joint uses local +Y,
and a positive yaw command turns counter-clockwise.  It also records effective
rolling radius, wheel contact ratio, estimated contact-patch slip, unintended contacts,
attitude excursion, velocity tracking, and stopping response.
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
parser.add_argument("--num_envs", type=int, default=1, help="Number of replicated test environments.")
parser.add_argument("--settle_steps", type=int, default=400, help="Zero-wheel settling steps before each phase.")
parser.add_argument("--ramp_steps", type=int, default=100, help="Minimum-jerk command ramp steps.")
parser.add_argument("--cruise_steps", type=int, default=300, help="Constant wheel-speed steps after the ramp.")
parser.add_argument("--brake_steps", type=int, default=300, help="Zero-velocity active-braking steps.")
parser.add_argument("--straight_wheel_speed", type=float, default=4.0, help="Straight wheel speed in rad/s.")
parser.add_argument("--turn_wheel_speed", type=float, default=4.0, help="Turn wheel speed in rad/s.")
parser.add_argument(
    "--wheel_velocity_gain",
    type=float,
    default=10.0,
    help="Provisional wheel velocity-error gain in N m per rad/s.",
)
parser.add_argument(
    "--turn_mode",
    choices=["arc", "in_place"],
    default="arc",
    help="Use stable forward arcs by default; in_place is an optional skid-steer stress test.",
)
parser.add_argument(
    "--turn_inner_multiplier",
    type=float,
    default=0.5,
    help="Inner-wheel speed multiplier for arc mode; must be in [0, 1).",
)
parser.add_argument("--wheel_radius", type=float, default=0.113, help="Nominal wheel radius used for slip estimates.")
parser.add_argument("--static_friction", type=float, default=0.8, help="Nominal contact static friction.")
parser.add_argument("--dynamic_friction", type=float, default=0.8, help="Nominal contact dynamic friction.")
parser.add_argument("--contact_force_threshold", type=float, default=20.0, help="Contact threshold in newtons.")
parser.add_argument("--min_straight_distance", type=float, default=0.50, help="Minimum signed straight travel in m.")
parser.add_argument("--max_straight_lateral", type=float, default=0.05, help="Maximum straight lateral drift in m.")
parser.add_argument("--max_straight_yaw_deg", type=float, default=5.0, help="Maximum straight heading drift.")
parser.add_argument("--min_turn_yaw_deg", type=float, default=25.0, help="Minimum signed turn angle.")
parser.add_argument("--min_arc_forward_distance", type=float, default=0.30, help="Minimum forward arc travel in m.")
parser.add_argument("--max_turn_translation", type=float, default=0.20, help="Maximum translation during a turn in m.")
parser.add_argument("--max_roll_pitch_deg", type=float, default=10.0, help="Maximum absolute roll/pitch excursion.")
parser.add_argument("--min_wheel_contact_ratio", type=float, default=0.95, help="Minimum wheel contact fraction.")
parser.add_argument(
    "--max_slip_ratio_p95",
    type=float,
    default=0.30,
    help="Maximum straight-line P95 estimated contact-patch slip ratio.",
)
parser.add_argument(
    "--max_unintended_contact_ratio",
    type=float,
    default=0.01,
    help="Maximum non-wheel contact ratio.",
)
parser.add_argument(
    "--max_wheel_tracking_error", type=float, default=0.15, help="Maximum cruise wheel-speed error ratio."
)
parser.add_argument(
    "--max_turn_wheel_tracking_error",
    type=float,
    default=0.50,
    help="Maximum turn wheel-speed error ratio under skid-steer load.",
)
parser.add_argument("--min_effective_radius", type=float, default=0.100, help="Minimum measured straight radius in m.")
parser.add_argument("--max_effective_radius", type=float, default=0.125, help="Maximum measured straight radius in m.")
parser.add_argument(
    "--max_radius_asymmetry",
    type=float,
    default=0.05,
    help="Maximum forward/reverse radius mismatch.",
)
parser.add_argument("--max_brake_time", type=float, default=1.2, help="Maximum stopping time in seconds.")
parser.add_argument("--max_brake_distance", type=float, default=0.25, help="Maximum travel after braking starts in m.")
parser.add_argument(
    "--brake_speed_threshold",
    type=float,
    default=0.05,
    help="Sustained chassis-stop threshold in equivalent m/s.",
)
parser.add_argument(
    "--brake_hold_steps",
    type=int,
    default=20,
    help="Consecutive steps below the chassis-stop threshold.",
)
parser.add_argument(
    "--yaw_characteristic_radius",
    type=float,
    default=0.235,
    help="Radius converting chassis yaw rate to equivalent linear speed in m.",
)
parser.add_argument(
    "--max_terminal_wheel_speed",
    type=float,
    default=0.75,
    help="Maximum worst-environment wheel-speed RMS over the final 20 brake steps in rad/s.",
)
parser.add_argument(
    "--max_zero_command_wheel_speed",
    type=float,
    default=0.50,
    help="Maximum speed of nominally stationary inner wheels during an arc in rad/s.",
)
parser.add_argument(
    "--phases",
    nargs="+",
    choices=["forward", "reverse", "turn_left", "turn_right"],
    default=None,
    help="Optional subset of command phases; default runs all four.",
)
parser.add_argument("--report", type=Path, default=None, help="Optional JSON report output path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1:
    parser.error("--num_envs must be positive")
if min(args_cli.settle_steps, args_cli.ramp_steps, args_cli.cruise_steps, args_cli.brake_steps) < 1:
    parser.error("all phase step counts must be positive")
if args_cli.brake_steps < args_cli.brake_hold_steps:
    parser.error("--brake_steps must be at least --brake_hold_steps")
if not 0.0 <= args_cli.turn_inner_multiplier < 1.0:
    parser.error("--turn_inner_multiplier must be in [0, 1)")
if min(
    args_cli.straight_wheel_speed,
    args_cli.turn_wheel_speed,
    args_cli.wheel_velocity_gain,
    args_cli.wheel_radius,
    args_cli.brake_speed_threshold,
    args_cli.yaw_characteristic_radius,
) <= 0.0:
    parser.error("wheel speeds, gain, radius, and brake thresholds must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, quat_inv, quat_mul  # noqa: E402

from LeggedManip_Lab.assets.b2w_z1.b2w_z1_articulation_cfg import (  # noqa: E402
    B2W_Z1_CFG,
    B2W_Z1_USD,
)


WHEEL_JOINT_NAMES = [
    "FL_wheel_joint",
    "FR_wheel_joint",
    "RL_wheel_joint",
    "RR_wheel_joint",
]
WHEEL_BODY_NAMES = ["FL_wheel", "FR_wheel", "RL_wheel", "RR_wheel"]
MIN_VELOCITY_ITERATION_COUNT = 1
TURN_COMMANDS = (
    (
        "turn_left",
        (args_cli.turn_inner_multiplier, 1.0, args_cli.turn_inner_multiplier, 1.0)
        if args_cli.turn_mode == "arc"
        else (-1.0, 1.0, -1.0, 1.0),
        "turn",
        1.0,
    ),
    (
        "turn_right",
        (1.0, args_cli.turn_inner_multiplier, 1.0, args_cli.turn_inner_multiplier)
        if args_cli.turn_mode == "arc"
        else (1.0, -1.0, 1.0, -1.0),
        "turn",
        -1.0,
    ),
)
COMMANDS = (
    ("forward", (1.0, 1.0, 1.0, 1.0), "straight", 1.0),
    ("reverse", (-1.0, -1.0, -1.0, -1.0), "straight", -1.0),
    *TURN_COMMANDS,
)

CONTACT_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="average",
    restitution_combine_mode="average",
    static_friction=args_cli.static_friction,
    dynamic_friction=args_cli.dynamic_friction,
    restitution=0.0,
)
ROBOT_CFG = B2W_Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
ROBOT_CFG.spawn.physics_material = CONTACT_MATERIAL
ROBOT_CFG.actuators["wheels"].damping = args_cli.wheel_velocity_gain


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance() -> dict[str, object]:
    script_path = Path(__file__).resolve()
    repository = script_path.parents[2]
    config_path = repository / (
        "source/LeggedManip_Lab/LeggedManip_Lab/assets/b2w_z1/"
        "b2w_z1_articulation_cfg.py"
    )
    asset_path = Path(B2W_Z1_USD).resolve()
    relevant_paths = [script_path, config_path, asset_path]
    try:
        relative_paths = [str(path.relative_to(repository)) for path in relevant_paths]
        git_head = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        relevant_status = subprocess.check_output(
            ["git", "-C", str(repository), "status", "--short", "--", *relative_paths],
            text=True,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError, ValueError):
        git_head = "unavailable"
        relevant_status = ["unavailable"]
        relative_paths = [str(path) for path in relevant_paths]
    return {
        "command": [sys.executable, *sys.argv],
        "git_head": git_head,
        "relevant_git_status": relevant_status,
        "script_path": relative_paths[0],
        "script_sha256": _sha256(script_path),
        "asset_config_path": relative_paths[1],
        "asset_config_sha256": _sha256(config_path),
        "asset_path": relative_paths[2],
        "asset_sha256": _sha256(asset_path),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else None,
    }


@configclass
class B2WZ1WheelGateSceneCfg(InteractiveSceneCfg):
    """Replicated flat-ground scene for deterministic wheel validation."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(physics_material=CONTACT_MATERIAL),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0),
    )
    robot = ROBOT_CFG
    contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        debug_vis=False,
    )


def _minimum_jerk(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5


def _relative_rpy(start_quat_w: torch.Tensor, current_quat_w: torch.Tensor) -> tuple[torch.Tensor, ...]:
    relative_quat = quat_mul(quat_inv(start_quat_w), current_quat_w)
    return euler_xyz_from_quat(relative_quat)


def _percentile(values: torch.Tensor, quantile: float) -> float | None:
    if values.numel() == 0:
        return None
    return float(torch.quantile(values.flatten(), quantile))


def _worst_masked_percentile(
    values: torch.Tensor, mask: torch.Tensor, quantile: float
) -> float | None:
    """Return the worst temporal percentile across every environment and wheel."""
    percentiles = []
    for environment_index in range(values.shape[1]):
        for wheel_index in range(values.shape[2]):
            selected = values[:, environment_index, wheel_index][
                mask[:, environment_index, wheel_index]
            ]
            if selected.numel() == 0:
                return None
            percentiles.append(torch.quantile(selected, quantile))
    return float(torch.max(torch.stack(percentiles)))


def _first_sustained_stop(
    speed_history: torch.Tensor, threshold: float, hold_steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return first sustained stopping index and a stopped flag per environment."""
    stopped = speed_history < threshold
    stop_indices = torch.full(
        (stopped.shape[1],), stopped.shape[0], device=stopped.device, dtype=torch.long
    )
    for start in range(max(0, stopped.shape[0] - hold_steps + 1)):
        sustained = torch.all(stopped[start : start + hold_steps], dim=0)
        newly_stopped = sustained & (stop_indices == stopped.shape[0])
        stop_indices[newly_stopped] = start
    stopped_in_window = stop_indices < stopped.shape[0]
    return stop_indices, stopped_in_window


class WheelGate:
    """Own the simulator state and run repeatable wheel command phases."""

    def __init__(self) -> None:
        sim_cfg = sim_utils.SimulationCfg(
            dt=0.005,
            device=args_cli.device,
            physx=sim_utils.PhysxCfg(
                min_velocity_iteration_count=MIN_VELOCITY_ITERATION_COUNT,
                enable_external_forces_every_iteration=True,
            ),
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)
        scene_cfg = B2WZ1WheelGateSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
        for actuator_cfg in scene_cfg.robot.actuators.values():
            actuator_cfg.min_delay = 0
            actuator_cfg.max_delay = 0
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()
        self.scene.update(self.sim.get_physics_dt())
        self.robot: Articulation = self.scene["robot"]
        self.contacts: ContactSensor = self.scene["contacts"]

        if self.robot.is_fixed_base:
            raise AssertionError("wheel dynamics require a floating-base articulation")
        self.wheel_joint_ids, wheel_joint_names = self.robot.find_joints(
            WHEEL_JOINT_NAMES, preserve_order=True
        )
        self.wheel_body_ids, wheel_body_names = self.robot.find_bodies(
            WHEEL_BODY_NAMES, preserve_order=True
        )
        self.contact_wheel_ids, contact_wheel_names = self.contacts.find_bodies(
            WHEEL_BODY_NAMES, preserve_order=True
        )
        if wheel_joint_names != WHEEL_JOINT_NAMES:
            raise AssertionError(f"unexpected wheel joint order: {wheel_joint_names}")
        if wheel_body_names != WHEEL_BODY_NAMES:
            raise AssertionError(f"unexpected wheel body order: {wheel_body_names}")
        if contact_wheel_names != WHEEL_BODY_NAMES:
            raise AssertionError(f"unexpected contact-sensor body order: {contact_wheel_names}")

        self.wheel_joint_ids_tensor = torch.tensor(self.wheel_joint_ids, device=self.sim.device)
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = self.robot.data.default_joint_vel.clone()
        self.default_root_state = self.robot.data.default_root_state.clone()
        self.nonwheel_contact_ids = [
            index for index, name in enumerate(self.contacts.body_names) if name not in WHEEL_BODY_NAMES
        ]
        self._axis_y = torch.zeros(
            (args_cli.num_envs, len(self.wheel_body_ids), 3), device=self.sim.device
        )
        self._axis_y[..., 1] = 1.0
        self._down = torch.zeros_like(self._axis_y)
        self._down[..., 2] = -1.0
        self._up = -self._down

    def _command_step(self, wheel_velocity: torch.Tensor) -> None:
        self.robot.set_joint_position_target(self.default_joint_pos)
        self.robot.set_joint_velocity_target(self.default_joint_vel)
        self.robot.set_joint_velocity_target(wheel_velocity, joint_ids=self.wheel_joint_ids_tensor)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.sim.get_physics_dt())

    def reset_and_settle(self) -> None:
        root_state = self.default_root_state.clone()
        root_state[:, 0:3] += self.scene.env_origins
        self.robot.write_root_state_to_sim(root_state)
        self.robot.write_joint_state_to_sim(self.default_joint_pos, self.default_joint_vel)
        self.scene.reset()
        zeros = torch.zeros((args_cli.num_envs, 4), device=self.sim.device)
        for _ in range(args_cli.settle_steps):
            self._command_step(zeros)
        if not torch.all(torch.isfinite(self.robot.data.root_state_w)):
            raise AssertionError("settling produced non-finite root state")
        if float(torch.min(self.robot.data.root_pos_w[:, 2])) < 0.35:
            raise AssertionError("robot collapsed during settling")

    def _sample(self, start_pos_w: torch.Tensor, start_quat_w: torch.Tensor) -> dict[str, torch.Tensor]:
        displacement_b = quat_apply_inverse(start_quat_w, self.robot.data.root_pos_w - start_pos_w)
        roll, pitch, yaw = _relative_rpy(start_quat_w, self.robot.data.root_quat_w)
        base_velocity_b = quat_apply_inverse(start_quat_w, self.robot.data.root_link_lin_vel_w)
        base_angular_velocity_b = quat_apply_inverse(
            start_quat_w, self.robot.data.root_link_ang_vel_w
        )

        wheel_force = torch.linalg.vector_norm(
            self.contacts.data.net_forces_w[:, self.contact_wheel_ids, :], dim=-1
        )
        wheel_contact = wheel_force > args_cli.contact_force_threshold
        if self.nonwheel_contact_ids:
            nonwheel_force = torch.linalg.vector_norm(
                self.contacts.data.net_forces_w[:, self.nonwheel_contact_ids, :], dim=-1
            )
            unintended_contact = torch.any(nonwheel_force > args_cli.contact_force_threshold, dim=-1)
        else:
            unintended_contact = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=self.sim.device)

        wheel_quat_w = self.robot.data.body_link_quat_w[:, self.wheel_body_ids]
        wheel_axis_w = quat_apply(wheel_quat_w, self._axis_y)
        down_perpendicular = self._down - torch.sum(self._down * wheel_axis_w, dim=-1, keepdim=True) * wheel_axis_w
        down_perpendicular /= torch.clamp(
            torch.linalg.vector_norm(down_perpendicular, dim=-1, keepdim=True), min=1.0e-6
        )
        contact_offset_w = args_cli.wheel_radius * down_perpendicular
        contact_velocity_w = self.robot.data.body_link_lin_vel_w[:, self.wheel_body_ids] + torch.linalg.cross(
            self.robot.data.body_link_ang_vel_w[:, self.wheel_body_ids], contact_offset_w, dim=-1
        )
        contact_velocity_horizontal = contact_velocity_w - torch.sum(
            contact_velocity_w * self._up, dim=-1, keepdim=True
        ) * self._up
        slip_speed = torch.linalg.vector_norm(contact_velocity_horizontal, dim=-1)
        wheel_speed_scale = args_cli.wheel_radius * torch.abs(
            self.robot.data.joint_vel[:, self.wheel_joint_ids]
        )
        denominator = torch.clamp(wheel_speed_scale, min=0.20)
        slip_ratio = slip_speed / denominator

        return {
            "displacement_b": displacement_b,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "base_velocity_b": base_velocity_b,
            "base_angular_velocity_b": base_angular_velocity_b,
            "base_height": self.robot.data.root_pos_w[:, 2].clone(),
            "wheel_velocity": self.robot.data.joint_vel[:, self.wheel_joint_ids].clone(),
            "wheel_applied_torque": self.robot.data.applied_torque[:, self.wheel_joint_ids].clone(),
            "wheel_contact": wheel_contact,
            "unintended_contact": unintended_contact,
            "slip_ratio": slip_ratio,
        }

    def run_command(
        self,
        name: str,
        signs: tuple[float, float, float, float],
        kind: str,
        expected_direction: float,
    ) -> dict[str, object]:
        self.reset_and_settle()
        start_pos_w = self.robot.data.root_pos_w.clone()
        start_quat_w = self.robot.data.root_quat_w.clone()
        command_signs = torch.tensor(signs, device=self.sim.device).repeat(args_cli.num_envs, 1)
        speed = args_cli.straight_wheel_speed if kind == "straight" else args_cli.turn_wheel_speed

        drive_samples: list[dict[str, torch.Tensor]] = []
        cruise_wheel_errors: list[torch.Tensor] = []
        cruise_integrated_wheel_rotation = torch.zeros_like(command_signs)
        ramp_end_displacement_b = torch.zeros((args_cli.num_envs, 3), device=self.sim.device)
        total_drive_steps = args_cli.ramp_steps + args_cli.cruise_steps
        for step in range(total_drive_steps):
            if step < args_cli.ramp_steps:
                scale = _minimum_jerk((step + 1) / args_cli.ramp_steps)
            else:
                scale = 1.0
            wheel_target = command_signs * speed * scale
            self._command_step(wheel_target)
            sample = self._sample(start_pos_w, start_quat_w)
            drive_samples.append(sample)
            if step == args_cli.ramp_steps - 1:
                ramp_end_displacement_b = sample["displacement_b"].clone()
            if step >= args_cli.ramp_steps:
                cruise_wheel_errors.append(
                    torch.abs(sample["wheel_velocity"] - wheel_target)
                    / torch.clamp(torch.abs(wheel_target), min=0.1)
                )
                cruise_integrated_wheel_rotation += (
                    sample["wheel_velocity"] * self.sim.get_physics_dt()
                )

        drive_end = drive_samples[-1]
        final_displacement_b = drive_end["displacement_b"]
        final_yaw = drive_end["yaw"]
        drive_displacements = torch.stack([sample["displacement_b"] for sample in drive_samples])
        drive_roll = torch.stack([sample["roll"] for sample in drive_samples])
        drive_pitch = torch.stack([sample["pitch"] for sample in drive_samples])
        drive_yaw = torch.stack([sample["yaw"] for sample in drive_samples])
        drive_wheel_velocity = torch.stack([sample["wheel_velocity"] for sample in drive_samples])
        drive_wheel_torque = torch.stack([sample["wheel_applied_torque"] for sample in drive_samples])
        drive_contact = torch.stack([sample["wheel_contact"] for sample in drive_samples])
        drive_unintended = torch.stack([sample["unintended_contact"] for sample in drive_samples])
        drive_slip = torch.stack([sample["slip_ratio"] for sample in drive_samples])

        zeros = torch.zeros_like(command_signs)
        brake_samples: list[dict[str, torch.Tensor]] = []
        for _ in range(args_cli.brake_steps):
            self._command_step(zeros)
            brake_samples.append(self._sample(start_pos_w, start_quat_w))
        brake_chassis_speed = torch.stack(
            [
                torch.maximum(
                    torch.linalg.vector_norm(sample["base_velocity_b"][:, 0:2], dim=-1),
                    args_cli.yaw_characteristic_radius
                    * torch.abs(sample["base_angular_velocity_b"][:, 2]),
                )
                for sample in brake_samples
            ]
        )
        brake_stop_indices, stopped_in_window = _first_sustained_stop(
            brake_chassis_speed,
            threshold=args_cli.brake_speed_threshold,
            hold_steps=args_cli.brake_hold_steps,
        )
        brake_time = (brake_stop_indices.to(torch.float32) + 1.0) * self.sim.get_physics_dt()
        terminal_wheel_speed = torch.stack(
            [sample["wheel_velocity"] for sample in brake_samples[-args_cli.brake_hold_steps :]]
        )
        terminal_wheel_speed_rms = torch.sqrt(torch.mean(terminal_wheel_speed**2, dim=(0, 2)))
        early_brake_wheel_velocity = torch.stack(
            [sample["wheel_velocity"] for sample in brake_samples[: args_cli.brake_hold_steps]]
        )
        early_brake_wheel_torque = torch.stack(
            [sample["wheel_applied_torque"] for sample in brake_samples[: args_cli.brake_hold_steps]]
        )
        moving_during_brake = torch.abs(early_brake_wheel_velocity) > 0.20
        opposing_brake_torque = early_brake_wheel_velocity * early_brake_wheel_torque < 0.0
        brake_opposing_torque_ratio = (
            float(torch.mean(opposing_brake_torque[moving_during_brake].float()))
            if torch.any(moving_during_brake)
            else None
        )
        brake_torque_saturation_ratio = float(
            torch.mean((torch.abs(early_brake_wheel_torque) >= 19.9).float())
        )
        brake_displacement_history = torch.stack(
            [sample["displacement_b"] for sample in brake_samples]
        )
        brake_distance_history = torch.linalg.vector_norm(
            brake_displacement_history[:, :, 0:2] - final_displacement_b.unsqueeze(0)[:, :, 0:2],
            dim=-1,
        )
        brake_distance = torch.max(brake_distance_history, dim=0).values

        commanded_nonzero = torch.abs(command_signs).unsqueeze(0) > 0.01
        cruise_wheel_velocity = drive_wheel_velocity[args_cli.ramp_steps :]
        command_direction = torch.sign(command_signs).unsqueeze(0)
        wheel_direction_correct = cruise_wheel_velocity * command_direction > 0.20
        commanded_direction_is_correct = wheel_direction_correct[
            commanded_nonzero.expand_as(wheel_direction_correct)
        ]
        wheel_direction_ratio = float(torch.mean(commanded_direction_is_correct.float()))
        wheel_direction_temporal_ratio = torch.mean(wheel_direction_correct.float(), dim=0)
        wheel_direction_ratio_min = float(
            torch.min(wheel_direction_temporal_ratio[commanded_nonzero.squeeze(0)])
        )
        contact_slip_values = drive_slip[drive_contact]
        contact_slip_p95_worst = _worst_masked_percentile(drive_slip, drive_contact, 0.95)
        max_roll_pitch = torch.maximum(torch.abs(drive_roll), torch.abs(drive_pitch))
        cruise_wheel_error_history = torch.stack(cruise_wheel_errors)
        tracking_mask = commanded_nonzero.expand_as(cruise_wheel_error_history)
        wheel_tracking_error = float(torch.mean(cruise_wheel_error_history[tracking_mask]))
        wheel_tracking_error_per_env_wheel = torch.mean(cruise_wheel_error_history, dim=0)
        wheel_tracking_error_worst = float(
            torch.max(wheel_tracking_error_per_env_wheel[commanded_nonzero.squeeze(0)])
        )
        zero_command_mask = (~commanded_nonzero).expand_as(cruise_wheel_velocity)
        zero_command_wheel_speed_max = (
            float(torch.max(torch.abs(cruise_wheel_velocity[zero_command_mask])))
            if torch.any(zero_command_mask)
            else 0.0
        )
        metrics: dict[str, object] = {
            "name": name,
            "kind": kind,
            "wheel_command_multipliers": list(signs),
            "wheel_speed_radps": speed,
            "forward_displacement_min_m": float(torch.min(final_displacement_b[:, 0])),
            "signed_forward_displacement_min_m": float(
                torch.min(
                    expected_direction * final_displacement_b[:, 0]
                    if kind == "straight"
                    else final_displacement_b[:, 0]
                )
            ),
            "lateral_displacement_abs_max_m": float(
                torch.max(torch.abs(drive_displacements[:, :, 1]))
            ),
            "signed_yaw_min_deg": math.degrees(float(torch.min(expected_direction * final_yaw))),
            "yaw_abs_max_deg": math.degrees(float(torch.max(torch.abs(drive_yaw)))),
            "translation_max_m": float(
                torch.max(torch.linalg.vector_norm(drive_displacements[:, :, 0:2], dim=-1))
            ),
            "roll_pitch_abs_max_deg": math.degrees(float(torch.max(max_roll_pitch))),
            "wheel_direction_correct_ratio": wheel_direction_ratio,
            "wheel_direction_correct_ratio_min_env_wheel": wheel_direction_ratio_min,
            "wheel_velocity_tracking_error_ratio": wheel_tracking_error,
            "wheel_velocity_tracking_error_ratio_worst_env_wheel": wheel_tracking_error_worst,
            "drive_wheel_torque_abs_max_nm": float(torch.max(torch.abs(drive_wheel_torque))),
            "drive_wheel_torque_saturation_ratio": float(
                torch.mean((torch.abs(drive_wheel_torque) >= 19.9).float())
            ),
            "zero_command_wheel_speed_abs_max_radps": zero_command_wheel_speed_max,
            "wheel_contact_ratio": float(torch.mean(drive_contact.float())),
            "wheel_contact_ratio_min_env_wheel": float(
                torch.min(torch.mean(drive_contact.float(), dim=0))
            ),
            "unintended_contact_ratio": float(torch.mean(drive_unintended.float())),
            "unintended_contact_ratio_max_env": float(
                torch.max(torch.mean(drive_unintended.float(), dim=0))
            ),
            "estimated_contact_patch_slip_ratio_median": _percentile(contact_slip_values, 0.50),
            "estimated_contact_patch_slip_ratio_p95": _percentile(contact_slip_values, 0.95),
            "estimated_contact_patch_slip_ratio_p95_worst_env_wheel": contact_slip_p95_worst,
            "brake_stopped_all_environments": bool(torch.all(stopped_in_window)),
            "brake_stop_time_max_s": float(torch.max(brake_time))
            if torch.all(stopped_in_window)
            else None,
            "brake_initial_chassis_speed_equivalent_max_mps": float(
                torch.max(brake_chassis_speed[0])
            ),
            "brake_initial_wheel_speed_abs_max_radps": float(
                torch.max(torch.abs(early_brake_wheel_velocity[0]))
            ),
            "brake_distance_max_m": float(torch.max(brake_distance)),
            "brake_yaw_travel_abs_max_deg": math.degrees(
                float(
                    torch.max(
                        torch.abs(
                            torch.stack([sample["yaw"] for sample in brake_samples])
                            - final_yaw.unsqueeze(0)
                        )
                    )
                )
            ),
            "brake_terminal_wheel_speed_rms_worst_env_radps": float(
                torch.max(terminal_wheel_speed_rms)
            ),
            "brake_sampled_torque_opposes_same_step_velocity_ratio": brake_opposing_torque_ratio,
            "brake_torque_abs_max_nm": float(torch.max(torch.abs(early_brake_wheel_torque))),
            "brake_torque_saturation_ratio": brake_torque_saturation_ratio,
            "base_height_min_m": float(
                torch.min(
                    torch.stack(
                        [sample["base_height"] for sample in drive_samples + brake_samples]
                    )
                )
            ),
            "cruise_wheel_rotation_abs_mean_rad": float(
                torch.mean(torch.abs(cruise_integrated_wheel_rotation))
            ),
        }
        if kind == "straight":
            cruise_displacement = final_displacement_b[:, 0] - ramp_end_displacement_b[:, 0]
            effective_radius = torch.abs(cruise_displacement) / torch.clamp(
                torch.mean(torch.abs(cruise_integrated_wheel_rotation), dim=-1), min=1.0e-6
            )
            metrics["effective_radius_mean_m"] = float(torch.mean(effective_radius))
            metrics["effective_radius_min_m"] = float(torch.min(effective_radius))
            metrics["effective_radius_max_m"] = float(torch.max(effective_radius))
            motion_passed = bool(
                metrics["signed_forward_displacement_min_m"] >= args_cli.min_straight_distance
                and metrics["lateral_displacement_abs_max_m"] <= args_cli.max_straight_lateral
                and metrics["yaw_abs_max_deg"] <= args_cli.max_straight_yaw_deg
                and metrics["effective_radius_min_m"] >= args_cli.min_effective_radius
                and metrics["effective_radius_max_m"] <= args_cli.max_effective_radius
            )
        else:
            if args_cli.turn_mode == "arc":
                motion_passed = bool(
                    metrics["signed_yaw_min_deg"] >= args_cli.min_turn_yaw_deg
                    and float(torch.min(final_displacement_b[:, 0]))
                    >= args_cli.min_arc_forward_distance
                )
            else:
                motion_passed = bool(
                    metrics["signed_yaw_min_deg"] >= args_cli.min_turn_yaw_deg
                    and metrics["translation_max_m"] <= args_cli.max_turn_translation
                )

        wheel_tracking_limit = (
            args_cli.max_wheel_tracking_error
            if kind == "straight"
            else args_cli.max_turn_wheel_tracking_error
        )
        estimated_slip_passed = bool(
            kind != "straight"
            or (
                metrics["estimated_contact_patch_slip_ratio_p95_worst_env_wheel"] is not None
                and metrics["estimated_contact_patch_slip_ratio_p95_worst_env_wheel"]
                <= args_cli.max_slip_ratio_p95
            )
        )
        brake_time_passed = bool(
            metrics["brake_stop_time_max_s"] is not None
            and metrics["brake_stop_time_max_s"] <= args_cli.max_brake_time
        )
        metrics["passed"] = bool(
            motion_passed
            and metrics["roll_pitch_abs_max_deg"] <= args_cli.max_roll_pitch_deg
            and metrics["wheel_direction_correct_ratio_min_env_wheel"] >= 0.99
            and metrics["wheel_velocity_tracking_error_ratio_worst_env_wheel"]
            <= wheel_tracking_limit
            and metrics["wheel_contact_ratio_min_env_wheel"] >= args_cli.min_wheel_contact_ratio
            and metrics["unintended_contact_ratio_max_env"]
            <= args_cli.max_unintended_contact_ratio
            and estimated_slip_passed
            and metrics["brake_stopped_all_environments"]
            and brake_time_passed
            and metrics["brake_distance_max_m"] <= args_cli.max_brake_distance
            and metrics["brake_terminal_wheel_speed_rms_worst_env_radps"]
            <= args_cli.max_terminal_wheel_speed
            and metrics["zero_command_wheel_speed_abs_max_radps"]
            <= args_cli.max_zero_command_wheel_speed
            and metrics["base_height_min_m"] >= 0.35
        )
        return metrics


def main() -> None:
    start_time = time.perf_counter()
    gate = WheelGate()
    results = []
    selected_commands = [
        command for command in COMMANDS if args_cli.phases is None or command[0] in args_cli.phases
    ]
    for name, signs, kind, expected_direction in selected_commands:
        metrics = gate.run_command(name, signs, kind, expected_direction)
        results.append(metrics)
        if kind == "straight":
            summary = (
                f"travel={metrics['signed_forward_displacement_min_m']:.3f} m, "
                f"radius={metrics['effective_radius_mean_m']:.4f} m"
            )
        else:
            summary = (
                f"yaw={metrics['signed_yaw_min_deg']:.1f} deg, "
                f"translation={metrics['translation_max_m']:.3f} m"
            )
        brake_summary = (
            f"{metrics['brake_stop_time_max_s']:.3f} s"
            if metrics["brake_stop_time_max_s"] is not None
            else f"> {args_cli.brake_steps * gate.sim.get_physics_dt():.3f} s"
        )
        slip_summary = metrics["estimated_contact_patch_slip_ratio_p95"]
        print(
            f"{'PASS' if metrics['passed'] else 'FAIL'} {name}: {summary}, "
            f"contact={100.0 * metrics['wheel_contact_ratio']:.1f}%, "
            f"estimated_slip_p95={slip_summary if slip_summary is not None else 'n/a'}, "
            f"brake={brake_summary}",
            flush=True,
        )

    result_by_name = {str(result["name"]): result for result in results}
    if "forward" in result_by_name and "reverse" in result_by_name:
        forward_radius = float(result_by_name["forward"]["effective_radius_mean_m"])
        reverse_radius = float(result_by_name["reverse"]["effective_radius_mean_m"])
        radius_asymmetry: float | None = abs(forward_radius - reverse_radius) / max(
            0.5 * (forward_radius + reverse_radius), 1.0e-6
        )
        radius_asymmetry_passed = radius_asymmetry <= args_cli.max_radius_asymmetry
    else:
        radius_asymmetry = None
        radius_asymmetry_passed = True
    required_phases = [command[0] for command in COMMANDS]
    executed_phases = [str(result["name"]) for result in results]
    selected_phases_passed = all(bool(result["passed"]) for result in results)
    complete_gate_executed = executed_phases == required_phases
    complete_gate_passed = bool(
        complete_gate_executed and selected_phases_passed and radius_asymmetry_passed
    )
    report = {
        "gate": "b2w_z1_nominal_flat_ground_wheel_dynamics",
        "passed": complete_gate_passed,
        "complete_gate_executed": complete_gate_executed,
        "complete_gate_passed": complete_gate_passed,
        "selected_phases_passed": selected_phases_passed,
        "required_phases": required_phases,
        "executed_phases": executed_phases,
        "device": args_cli.device,
        "num_envs": args_cli.num_envs,
        "provenance": _provenance(),
        "physics_dt_s": gate.sim.get_physics_dt(),
        "physx": {
            "minimum_velocity_iteration_count": MIN_VELOCITY_ITERATION_COUNT,
            "external_forces_every_iteration": True,
        },
        "steps": {
            "settle": args_cli.settle_steps,
            "ramp": args_cli.ramp_steps,
            "cruise": args_cli.cruise_steps,
            "brake": args_cli.brake_steps,
        },
        "actuator_delay_steps": 0,
        "wheel_velocity_gain_nm_per_radps": args_cli.wheel_velocity_gain,
        "nominal_wheel_radius_m": args_cli.wheel_radius,
        "contact_force_threshold_n": args_cli.contact_force_threshold,
        "ground_static_friction": args_cli.static_friction,
        "ground_dynamic_friction": args_cli.dynamic_friction,
        "arm_pose": "photo-matched safe stow",
        "wheel_joint_names": WHEEL_JOINT_NAMES,
        "wheel_joint_ids": gate.wheel_joint_ids,
        "wheel_body_names": WHEEL_BODY_NAMES,
        "wheel_body_ids": gate.wheel_body_ids,
        "contact_sensor_body_names": gate.contacts.body_names,
        "coordinate_convention": "+X forward, +Y left, +Z up, +yaw counter-clockwise",
        "turn_mode": args_cli.turn_mode,
        "turn_inner_multiplier": args_cli.turn_inner_multiplier,
        "command_convention": {
            "forward": [1, 1, 1, 1],
            "reverse": [-1, -1, -1, -1],
            "turn_left": list(TURN_COMMANDS[0][1]),
            "turn_right": list(TURN_COMMANDS[1][1]),
        },
        "phases": results,
        "forward_reverse_effective_radius_asymmetry": radius_asymmetry,
        "thresholds": {
            "min_straight_distance_m": args_cli.min_straight_distance,
            "max_straight_lateral_m": args_cli.max_straight_lateral,
            "max_straight_yaw_deg": args_cli.max_straight_yaw_deg,
            "min_turn_yaw_deg": args_cli.min_turn_yaw_deg,
            "min_arc_forward_distance_m": args_cli.min_arc_forward_distance,
            "max_turn_translation_m": args_cli.max_turn_translation,
            "max_roll_pitch_deg": args_cli.max_roll_pitch_deg,
            "min_wheel_contact_ratio": args_cli.min_wheel_contact_ratio,
            "max_straight_estimated_contact_patch_slip_ratio_p95": args_cli.max_slip_ratio_p95,
            "max_unintended_contact_ratio": args_cli.max_unintended_contact_ratio,
            "max_wheel_velocity_tracking_error_ratio": args_cli.max_wheel_tracking_error,
            "max_turn_wheel_velocity_tracking_error_ratio": args_cli.max_turn_wheel_tracking_error,
            "effective_radius_range_m": [args_cli.min_effective_radius, args_cli.max_effective_radius],
            "max_forward_reverse_radius_asymmetry": args_cli.max_radius_asymmetry,
            "max_brake_stop_time_s": args_cli.max_brake_time,
            "max_brake_distance_m": args_cli.max_brake_distance,
            "brake_chassis_speed_threshold_mps": args_cli.brake_speed_threshold,
            "brake_hold_steps": args_cli.brake_hold_steps,
            "brake_yaw_characteristic_radius_m": args_cli.yaw_characteristic_radius,
            "max_terminal_wheel_speed_rms_radps": args_cli.max_terminal_wheel_speed,
            "max_zero_command_wheel_speed_abs_radps": args_cli.max_zero_command_wheel_speed,
        },
        "elapsed_wall_time_s": time.perf_counter() - start_time,
    }
    report_text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args_cli.report is not None:
        args_cli.report.parent.mkdir(parents=True, exist_ok=True)
        args_cli.report.write_text(report_text + "\n", encoding="utf-8")
        print(f"Report: {args_cli.report.resolve()}", flush=True)
    print(report_text, flush=True)
    if not report["passed"]:
        raise AssertionError("B2-W + Z1 wheel-dynamics gate failed")


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
