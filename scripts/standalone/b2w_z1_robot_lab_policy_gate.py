#!/usr/bin/env python3
"""Run the public robot_lab B2-W policy on the local B2-W + Z1 asset.

This gate validates the frozen low-level locomotion layer before it is connected
to the world-frame TCP controller.  The Z1 is position-held in its safe stow
pose while the learned policy commands all twelve leg joints and four wheels.
Every command phase starts from the same physical reset state.
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

_SCRIPT_PATH = Path(__file__).resolve()
_REPOSITORY = _SCRIPT_PATH.parents[2]
_DEFAULT_CHECKPOINT = _REPOSITORY / "third_party/rl_sar_b2w/artifacts/policy.pt"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=_DEFAULT_CHECKPOINT)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle_seconds", type=float, default=2.0)
parser.add_argument("--ramp_seconds", type=float, default=0.5)
parser.add_argument("--phase_seconds", type=float, default=4.0)
parser.add_argument("--stop_seconds", type=float, default=2.0)
parser.add_argument("--forward_speed", type=float, default=0.3)
parser.add_argument("--yaw_rate", type=float, default=0.3)
parser.add_argument(
    "--phases",
    nargs="+",
    choices=["stand", "forward", "reverse", "turn_left", "turn_right"],
    default=None,
)
parser.add_argument("--arm_pose", choices=["stow", "home"], default="stow")
parser.add_argument("--contact_force_threshold", type=float, default=20.0)
parser.add_argument("--report", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs < 1:
    parser.error("--num_envs must be positive")
if (
    min(
        args_cli.settle_seconds,
        args_cli.ramp_seconds,
        args_cli.phase_seconds,
        args_cli.stop_seconds,
        args_cli.forward_speed,
        args_cli.yaw_rate,
    )
    <= 0.0
):
    parser.error("durations and command magnitudes must be positive")
if args_cli.ramp_seconds >= args_cli.phase_seconds:
    parser.error("--ramp_seconds must be shorter than --phase_seconds")

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
    ROBOT_LAB_B2W_POLICY_SHA256,
    RobotLabB2WPolicy,
)

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_inv, quat_mul  # noqa: E402

PHYSICS_DT = 0.005
POLICY_DECIMATION = 4
POLICY_DT = PHYSICS_DT * POLICY_DECIMATION
ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_joint")
WHEEL_BODY_NAMES = ("FR_wheel", "FL_wheel", "RR_wheel", "RL_wheel")
PHASE_COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "forward": (args_cli.forward_speed, 0.0, 0.0),
    "reverse": (-args_cli.forward_speed, 0.0, 0.0),
    "turn_left": (0.0, 0.0, args_cli.yaw_rate),
    "turn_right": (0.0, 0.0, -args_cli.yaw_rate),
}
REQUIRED_PHASES = tuple(PHASE_COMMANDS)

CONTACT_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="average",
    restitution_combine_mode="average",
    static_friction=0.8,
    dynamic_friction=0.8,
    restitution=0.0,
)
ROBOT_CFG = B2W_Z1_ROBOT_LAB_POLICY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
ROBOT_CFG.spawn.physics_material = CONTACT_MATERIAL
if args_cli.arm_pose == "home":
    ROBOT_CFG.init_state.joint_pos.update(B2W_Z1_ARM_HOME_JOINT_POS)
for actuator_cfg in ROBOT_CFG.actuators.values():
    if hasattr(actuator_cfg, "min_delay"):
        actuator_cfg.min_delay = 0
        actuator_cfg.max_delay = 0


@configclass
class RobotLabPolicyGateSceneCfg(InteractiveSceneCfg):
    """Replicated flat-ground scene for the checkpoint smoke test."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _minimum_jerk(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5


def _relative_rpy(start_quat_w: torch.Tensor, current_quat_w: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return euler_xyz_from_quat(quat_mul(quat_inv(start_quat_w), current_quat_w))


def _first_sustained_stop(
    speed_history: torch.Tensor, threshold: float = 0.05, hold_steps: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    stopped = speed_history < threshold
    stop_indices = torch.full((stopped.shape[1],), stopped.shape[0], device=stopped.device, dtype=torch.long)
    for start in range(max(0, stopped.shape[0] - hold_steps + 1)):
        sustained = torch.all(stopped[start : start + hold_steps], dim=0)
        newly_stopped = sustained & (stop_indices == stopped.shape[0])
        stop_indices[newly_stopped] = start
    return stop_indices, stop_indices < stopped.shape[0]


def _git_provenance(paths: list[Path]) -> dict[str, object]:
    relative_paths = [str(path.resolve().relative_to(_REPOSITORY)) for path in paths]
    try:
        git_head = subprocess.check_output(["git", "-C", str(_REPOSITORY), "rev-parse", "HEAD"], text=True).strip()
        git_status = subprocess.check_output(
            ["git", "-C", str(_REPOSITORY), "status", "--short", "--", *relative_paths],
            text=True,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        git_head = "unavailable"
        git_status = ["unavailable"]
    return {
        "git_head": git_head,
        "relevant_git_status": git_status,
        "files": {relative: _sha256(path) for relative, path in zip(relative_paths, paths, strict=True)},
    }


class RobotLabPolicyGate:
    """Own the simulator, checkpoint state, and reproducible command phases."""

    def __init__(self) -> None:
        sim_cfg = sim_utils.SimulationCfg(
            dt=PHYSICS_DT,
            device=args_cli.device,
            physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.scene = InteractiveScene(RobotLabPolicyGateSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0))
        self.sim.reset()
        self.scene.update(self.sim.get_physics_dt())
        self.robot: Articulation = self.scene["robot"]
        self.contacts: ContactSensor = self.scene["contacts"]
        if self.robot.is_fixed_base:
            raise AssertionError("the locomotion checkpoint requires a floating-base articulation")

        policy_joint_ids, policy_joint_names = self.robot.find_joints(
            list(ROBOT_LAB_B2W_JOINT_NAMES), preserve_order=True
        )
        if tuple(policy_joint_names) != ROBOT_LAB_B2W_JOINT_NAMES:
            raise AssertionError(f"unexpected policy joint order: {policy_joint_names}")
        self.policy_joint_ids = torch.tensor(policy_joint_ids, device=self.sim.device)
        self.leg_joint_ids = self.policy_joint_ids[:12]
        self.wheel_joint_ids = self.policy_joint_ids[12:]

        arm_joint_ids, arm_joint_names = self.robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        if tuple(arm_joint_names) != ARM_JOINT_NAMES:
            raise AssertionError(f"unexpected arm joint order: {arm_joint_names}")
        self.arm_joint_ids = torch.tensor(arm_joint_ids, device=self.sim.device)
        self.arm_position_target = self.robot.data.default_joint_pos[:, self.arm_joint_ids].clone()
        self.arm_velocity_target = torch.zeros_like(self.arm_position_target)

        contact_wheel_ids, contact_wheel_names = self.contacts.find_bodies(list(WHEEL_BODY_NAMES), preserve_order=True)
        if tuple(contact_wheel_names) != WHEEL_BODY_NAMES:
            raise AssertionError(f"unexpected contact wheel order: {contact_wheel_names}")
        self.contact_wheel_ids = contact_wheel_ids
        self.nonwheel_contact_ids = [
            index for index, name in enumerate(self.contacts.body_names) if name not in WHEEL_BODY_NAMES
        ]

        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = self.robot.data.default_joint_vel.clone()
        self.default_root_state = self.robot.data.default_root_state.clone()
        self.zero_command = torch.zeros((args_cli.num_envs, 3), device=self.sim.device)
        self.current_action = None
        self.policy = RobotLabB2WPolicy(
            args_cli.checkpoint,
            num_envs=args_cli.num_envs,
            device=self.sim.device,
        )

    def reset(self) -> None:
        root_state = self.default_root_state.clone()
        root_state[:, 0:3] += self.scene.env_origins
        self.robot.write_root_state_to_sim(root_state)
        self.robot.write_joint_state_to_sim(self.default_joint_pos, self.default_joint_vel)
        self.scene.reset()
        self.policy.reset()
        self.current_action = None
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.sim.get_physics_dt())

    def _policy_step(self, command: torch.Tensor) -> None:
        joint_position = self.robot.data.joint_pos[:, self.policy_joint_ids]
        joint_velocity = self.robot.data.joint_vel[:, self.policy_joint_ids]
        self.current_action = self.policy.act(
            base_angular_velocity_b=self.robot.data.root_link_ang_vel_b,
            projected_gravity_b=self.robot.data.projected_gravity_b,
            velocity_command_b=command,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
        )

    def _physics_step(self, command: torch.Tensor, step: int) -> None:
        if self.current_action is None or step % POLICY_DECIMATION == 0:
            self._policy_step(command)
        self.robot.set_joint_position_target(
            self.current_action.leg_position_targets,
            joint_ids=self.leg_joint_ids,
        )
        self.robot.set_joint_velocity_target(
            torch.zeros_like(self.current_action.leg_position_targets),
            joint_ids=self.leg_joint_ids,
        )
        self.robot.set_joint_velocity_target(
            self.current_action.wheel_velocity_targets,
            joint_ids=self.wheel_joint_ids,
        )
        self.robot.set_joint_position_target(
            self.arm_position_target,
            joint_ids=self.arm_joint_ids,
        )
        self.robot.set_joint_velocity_target(
            self.arm_velocity_target,
            joint_ids=self.arm_joint_ids,
        )
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.sim.get_physics_dt())

    def _sample(self, start_pos_w: torch.Tensor, start_quat_w: torch.Tensor) -> dict[str, torch.Tensor]:
        displacement_b = quat_apply_inverse(start_quat_w, self.robot.data.root_pos_w - start_pos_w)
        roll, pitch, yaw = _relative_rpy(start_quat_w, self.robot.data.root_quat_w)
        wheel_force = torch.linalg.vector_norm(self.contacts.data.net_forces_w[:, self.contact_wheel_ids, :], dim=-1)
        wheel_contact = wheel_force > args_cli.contact_force_threshold
        if self.nonwheel_contact_ids:
            nonwheel_force = torch.linalg.vector_norm(
                self.contacts.data.net_forces_w[:, self.nonwheel_contact_ids, :], dim=-1
            )
            unintended_contact = torch.any(nonwheel_force > args_cli.contact_force_threshold, dim=-1)
        else:
            unintended_contact = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=self.sim.device)
        return {
            "displacement_b": displacement_b,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "base_velocity_b": self.robot.data.root_link_lin_vel_b.clone(),
            "base_angular_velocity_b": self.robot.data.root_link_ang_vel_b.clone(),
            "base_height": self.robot.data.root_pos_w[:, 2].clone(),
            "wheel_contact": wheel_contact,
            "unintended_contact": unintended_contact,
            "leg_target_error": (
                self.robot.data.joint_pos[:, self.leg_joint_ids] - self.current_action.leg_position_targets
            ).clone(),
            "wheel_target_error": (
                self.robot.data.joint_vel[:, self.wheel_joint_ids] - self.current_action.wheel_velocity_targets
            ).clone(),
            "wheel_target": self.current_action.wheel_velocity_targets.clone(),
            "action": self.current_action.raw.clone(),
        }

    def _settle(self) -> None:
        settle_steps = round(args_cli.settle_seconds / PHYSICS_DT)
        for step in range(settle_steps):
            self._physics_step(self.zero_command, step)
        if not torch.all(torch.isfinite(self.robot.data.root_state_w)):
            raise AssertionError("settling produced a non-finite root state")

    def run_phase(self, name: str, command_values: tuple[float, float, float]) -> dict[str, object]:
        self.reset()
        self._settle()
        start_pos_w = self.robot.data.root_pos_w.clone()
        start_quat_w = self.robot.data.root_quat_w.clone()
        command_final = torch.tensor(command_values, device=self.sim.device).repeat(args_cli.num_envs, 1)
        phase_steps = round(args_cli.phase_seconds / PHYSICS_DT)
        ramp_steps = round(args_cli.ramp_seconds / PHYSICS_DT)
        samples = []
        for step in range(phase_steps):
            scale = _minimum_jerk((step + 1) / ramp_steps) if step < ramp_steps else 1.0
            self._physics_step(command_final * scale, step)
            samples.append(self._sample(start_pos_w, start_quat_w))

        displacement = torch.stack([sample["displacement_b"] for sample in samples])
        roll = torch.stack([sample["roll"] for sample in samples])
        pitch = torch.stack([sample["pitch"] for sample in samples])
        yaw = torch.stack([sample["yaw"] for sample in samples])
        linear_velocity = torch.stack([sample["base_velocity_b"] for sample in samples])
        angular_velocity = torch.stack([sample["base_angular_velocity_b"] for sample in samples])
        base_height = torch.stack([sample["base_height"] for sample in samples])
        wheel_contact = torch.stack([sample["wheel_contact"] for sample in samples])
        unintended_contact = torch.stack([sample["unintended_contact"] for sample in samples])
        leg_error = torch.stack([sample["leg_target_error"] for sample in samples])
        wheel_error = torch.stack([sample["wheel_target_error"] for sample in samples])
        wheel_target = torch.stack([sample["wheel_target"] for sample in samples])
        action = torch.stack([sample["action"] for sample in samples])

        steady_start = min(
            phase_steps - 1,
            ramp_steps + round(0.5 / PHYSICS_DT),
        )
        steady_linear_velocity = linear_velocity[steady_start:]
        steady_angular_velocity = angular_velocity[steady_start:]
        max_roll_pitch_deg = math.degrees(float(torch.max(torch.maximum(torch.abs(roll), torch.abs(pitch)))))
        final_displacement = displacement[-1]
        final_yaw = yaw[-1]
        global_passed = bool(
            torch.all(torch.isfinite(linear_velocity))
            and float(torch.min(base_height)) >= 0.40
            and max_roll_pitch_deg <= 10.0
            and float(torch.min(torch.mean(wheel_contact.float(), dim=0))) >= 0.90
            and float(torch.max(torch.mean(unintended_contact.float(), dim=0))) <= 0.01
        )

        metrics: dict[str, object] = {
            "name": name,
            "command_vx_vy_wz": list(command_values),
            "base_height_min_m": float(torch.min(base_height)),
            "roll_pitch_abs_max_deg": max_roll_pitch_deg,
            "wheel_contact_ratio_min_env_wheel": float(torch.min(torch.mean(wheel_contact.float(), dim=0))),
            "unintended_contact_ratio_max_env": float(torch.max(torch.mean(unintended_contact.float(), dim=0))),
            "final_forward_displacement_min_m": float(torch.min(final_displacement[:, 0])),
            "final_lateral_displacement_abs_max_m": float(torch.max(torch.abs(final_displacement[:, 1]))),
            "final_yaw_min_deg": math.degrees(float(torch.min(final_yaw))),
            "final_yaw_abs_max_deg": math.degrees(float(torch.max(torch.abs(final_yaw)))),
            "steady_vx_mean_min_mps": float(torch.min(torch.mean(steady_linear_velocity[:, :, 0], dim=0))),
            "steady_vx_mean_max_mps": float(torch.max(torch.mean(steady_linear_velocity[:, :, 0], dim=0))),
            "steady_vy_rms_max_mps": float(
                torch.max(torch.sqrt(torch.mean(steady_linear_velocity[:, :, 1] ** 2, dim=0)))
            ),
            "steady_wz_mean_min_radps": float(torch.min(torch.mean(steady_angular_velocity[:, :, 2], dim=0))),
            "steady_wz_mean_max_radps": float(torch.max(torch.mean(steady_angular_velocity[:, :, 2], dim=0))),
            "leg_position_target_error_p95_rad": float(torch.quantile(torch.abs(leg_error), 0.95)),
            "wheel_velocity_target_error_p95_radps": float(torch.quantile(torch.abs(wheel_error), 0.95)),
            "wheel_velocity_target_relative_error_p95": float(
                torch.quantile(
                    torch.abs(wheel_error) / torch.clamp(torch.abs(wheel_target), min=0.5),
                    0.95,
                )
            ),
            "raw_action_abs_max": float(torch.max(torch.abs(action))),
        }

        if name == "stand":
            terminal_steps = max(1, round(1.0 / PHYSICS_DT))
            terminal_linear_velocity = linear_velocity[-terminal_steps:]
            terminal_angular_velocity = angular_velocity[-terminal_steps:]
            translation = torch.linalg.vector_norm(final_displacement[:, :2], dim=-1)
            stand_passed = bool(
                float(torch.max(translation)) <= 0.25
                and metrics["final_yaw_abs_max_deg"] <= 15.0
                and float(torch.max(torch.sqrt(torch.mean(terminal_linear_velocity[:, :, :2] ** 2, dim=(0, 2)))))
                <= 0.10
                and float(torch.max(torch.sqrt(torch.mean(terminal_angular_velocity[:, :, 2] ** 2, dim=0)))) <= 0.15
            )
            metrics["passed"] = global_passed and stand_passed
            return metrics

        stop_samples = []
        stop_steps = round(args_cli.stop_seconds / PHYSICS_DT)
        phase_end_position = self.robot.data.root_pos_w.clone()
        for step in range(stop_steps):
            self._physics_step(self.zero_command, step)
            stop_samples.append(self._sample(start_pos_w, start_quat_w))
        stop_speed = torch.stack(
            [
                torch.maximum(
                    torch.linalg.vector_norm(sample["base_velocity_b"][:, :2], dim=-1),
                    0.235 * torch.abs(sample["base_angular_velocity_b"][:, 2]),
                )
                for sample in stop_samples
            ]
        )
        stop_indices, stopped = _first_sustained_stop(stop_speed)
        stop_time = (stop_indices.to(torch.float32) + 1.0) * PHYSICS_DT
        stop_displacement = torch.linalg.vector_norm(
            self.robot.data.root_pos_w[:, :2] - phase_end_position[:, :2], dim=-1
        )
        stop_passed = bool(
            torch.all(stopped)
            and float(torch.max(stop_time)) <= args_cli.stop_seconds
            and float(torch.max(stop_displacement)) <= 0.25
        )
        metrics.update(
            {
                "stop_succeeded_all_envs": bool(torch.all(stopped)),
                "stop_time_max_s": float(torch.max(stop_time)) if torch.all(stopped) else None,
                "stop_displacement_max_m": float(torch.max(stop_displacement)),
            }
        )

        if name in {"forward", "reverse"}:
            direction = 1.0 if name == "forward" else -1.0
            signed_displacement = direction * final_displacement[:, 0]
            signed_velocity = direction * torch.mean(steady_linear_velocity[:, :, 0], dim=0)
            velocity_rmse = torch.sqrt(
                torch.mean(
                    (steady_linear_velocity[:, :, 0] - command_values[0]) ** 2,
                    dim=0,
                )
            )
            motion_passed = bool(
                float(torch.min(signed_displacement)) >= 0.60
                and float(torch.min(signed_velocity)) >= 0.15
                and float(torch.max(velocity_rmse)) <= 0.15
                and metrics["steady_vy_rms_max_mps"] <= 0.10
                and float(torch.max(torch.sqrt(torch.mean(steady_angular_velocity[:, :, 2] ** 2, dim=0)))) <= 0.15
            )
            metrics["signed_forward_displacement_min_m"] = float(torch.min(signed_displacement))
            metrics["signed_steady_vx_mean_min_mps"] = float(torch.min(signed_velocity))
            metrics["vx_tracking_rmse_max_mps"] = float(torch.max(velocity_rmse))
        else:
            direction = 1.0 if name == "turn_left" else -1.0
            signed_yaw = direction * final_yaw
            signed_yaw_rate = direction * torch.mean(steady_angular_velocity[:, :, 2], dim=0)
            yaw_rate_rmse = torch.sqrt(
                torch.mean(
                    (steady_angular_velocity[:, :, 2] - command_values[2]) ** 2,
                    dim=0,
                )
            )
            translation = torch.linalg.vector_norm(final_displacement[:, :2], dim=-1)
            motion_passed = bool(
                math.degrees(float(torch.min(signed_yaw))) >= 30.0
                and float(torch.min(signed_yaw_rate)) >= 0.15
                and float(torch.max(yaw_rate_rmse)) <= 0.15
                and float(torch.max(translation)) <= 0.30
            )
            metrics["signed_yaw_min_deg"] = math.degrees(float(torch.min(signed_yaw)))
            metrics["signed_steady_wz_mean_min_radps"] = float(torch.min(signed_yaw_rate))
            metrics["wz_tracking_rmse_max_radps"] = float(torch.max(yaw_rate_rmse))
            metrics["translation_max_m"] = float(torch.max(translation))

        metrics["passed"] = global_passed and motion_passed and stop_passed
        return metrics


def main() -> None:
    start_time = time.perf_counter()
    gate = RobotLabPolicyGate()
    selected_phases = args_cli.phases or list(REQUIRED_PHASES)
    results = []
    for name in selected_phases:
        metrics = gate.run_phase(name, PHASE_COMMANDS[name])
        results.append(metrics)
        if name == "stand":
            stand_drift = math.hypot(
                metrics["final_forward_displacement_min_m"],
                metrics["final_lateral_displacement_abs_max_m"],
            )
            motion = f"drift={stand_drift:.3f} m"
        elif name in {"forward", "reverse"}:
            motion = f"travel={metrics['signed_forward_displacement_min_m']:.3f} m"
        else:
            motion = f"yaw={metrics['signed_yaw_min_deg']:.1f} deg"
        print(
            f"{'PASS' if metrics['passed'] else 'FAIL'} {name}: {motion}, "
            f"height_min={metrics['base_height_min_m']:.3f} m, "
            f"roll/pitch_max={metrics['roll_pitch_abs_max_deg']:.1f} deg",
            flush=True,
        )

    config_path = _REPOSITORY / ("source/LeggedManip_Lab/LeggedManip_Lab/assets/b2w_z1/b2w_z1_articulation_cfg.py")
    adapter_path = _REPOSITORY / ("source/LeggedManip_Lab/LeggedManip_Lab/controllers/robot_lab_b2w_policy.py")
    complete_gate_executed = selected_phases == list(REQUIRED_PHASES)
    selected_phases_passed = all(bool(result["passed"]) for result in results)
    complete_gate_passed = bool(complete_gate_executed and selected_phases_passed)
    report = {
        "gate": "b2w_z1_robot_lab_frozen_locomotion_policy",
        "passed": selected_phases_passed,
        "complete_gate_executed": complete_gate_executed,
        "complete_gate_passed": complete_gate_passed,
        "selected_phases_passed": selected_phases_passed,
        "required_phases": list(REQUIRED_PHASES),
        "executed_phases": selected_phases,
        "arm_pose": args_cli.arm_pose,
        "num_envs": args_cli.num_envs,
        "device": args_cli.device,
        "physics_dt_s": PHYSICS_DT,
        "policy_dt_s": POLICY_DT,
        "policy_decimation": POLICY_DECIMATION,
        "checkpoint": str(args_cli.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args_cli.checkpoint.resolve()),
        "expected_checkpoint_sha256": ROBOT_LAB_B2W_POLICY_SHA256,
        "policy_joint_names": list(ROBOT_LAB_B2W_JOINT_NAMES),
        "asset_path": str(Path(B2W_Z1_USD).resolve()),
        "commands": {name: list(command) for name, command in PHASE_COMMANDS.items()},
        "durations_s": {
            "settle": args_cli.settle_seconds,
            "ramp": args_cli.ramp_seconds,
            "phase": args_cli.phase_seconds,
            "stop": args_cli.stop_seconds,
        },
        "thresholds": {
            "base_height_min_m": 0.40,
            "roll_pitch_abs_max_deg": 10.0,
            "wheel_contact_ratio_min": 0.90,
            "unintended_contact_ratio_max": 0.01,
            "straight_signed_travel_min_m": 0.60,
            "straight_signed_mean_velocity_min_mps": 0.15,
            "straight_velocity_rmse_max_mps": 0.15,
            "yaw_signed_travel_min_deg": 30.0,
            "yaw_signed_mean_rate_min_radps": 0.15,
            "yaw_rate_rmse_max_radps": 0.15,
            "stop_time_max_s": args_cli.stop_seconds,
            "stop_displacement_max_m": 0.25,
        },
        "phases": results,
        "provenance": _git_provenance([_SCRIPT_PATH, config_path, adapter_path]),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None,
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
        raise AssertionError("B2-W + Z1 robot_lab policy gate failed")


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
