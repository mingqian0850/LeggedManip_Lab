#!/usr/bin/env python3
"""Sweep a diagnostic leg-wheel turn gait on the B2-W + Z1 asset.

This is not a production controller and it does not claim to reproduce
Unitree's closed-source wheeled-sport controller. It tests whether commanded
leg motion can create sustained yaw independently of commanded wheel rolling.
It includes fixed-leg, diagonal, one-foot-at-a-time crawl, and crawl-with-load-
shift controls.

Each replicated environment receives one combination of:

* commanded yaw per support stroke;
* gait frequency and support pattern;
* swing wheel-foot lift height; and
* differential wheel-speed assistance or a zero-speed wheel hold.

The Z1 stays at the asset's safe stow pose.  Results include yaw, translation,
attitude, per-cycle yaw, actual swing contact/clearance, non-wheel contact,
height, joint tracking, and actuator effort. The same candidate index can then
be replayed with the Isaac Sim GUI.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--settle_steps", type=int, default=400)
parser.add_argument("--preload_ramp_steps", type=int, default=100)
parser.add_argument("--preload_hold_steps", type=int, default=50)
parser.add_argument("--ramp_steps", type=int, default=200)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=0,
    help="Run the commanded gait before starting measurement (use whole gait cycles).",
)
parser.add_argument("--drive_steps", type=int, default=1200)
parser.add_argument("--max_candidates", type=int, default=None)
parser.add_argument(
    "--strategies",
    nargs="+",
    choices=[
        "fixed_leg_wheel_only",
        "four_contact_hip_bias",
        "diagonal_reposition",
        "diagonal_refined",
        "diagonal_leg_only_refined",
        "diagonal_factorial",
        "crawl_leg_only",
        "crawl_weight_shift",
    ],
    default=None,
    help="Optional subset of diagnostic strategies to run.",
)
parser.add_argument(
    "--phase_shift_cycles",
    type=float,
    choices=[0.0, 0.5],
    default=None,
    help="For diagonal gait candidates, optionally select which diagonal starts in support.",
)
parser.add_argument(
    "--candidate_index",
    type=int,
    default=None,
    help="Replay exactly one candidate from the stable candidate table.",
)
parser.add_argument(
    "--candidate_indices",
    type=int,
    nargs="+",
    default=None,
    help="Replay several candidates from the stable table in parallel.",
)
parser.add_argument("--static_friction", type=float, default=0.8)
parser.add_argument("--dynamic_friction", type=float, default=0.8)
parser.add_argument("--wheel_velocity_gain", type=float, default=10.0)
parser.add_argument("--contact_force_threshold", type=float, default=20.0)
parser.add_argument("--report", type=Path, default=None)
parser.add_argument(
    "--gui_start_hold_s",
    type=float,
    default=0.0,
    help="Keep the settled start pose visible before advancing physics.",
)
parser.add_argument(
    "--gui_hold_s",
    type=float,
    default=0.0,
    help="Keep the final pose visible without advancing physics.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if min(
    args_cli.settle_steps,
    args_cli.preload_ramp_steps,
    args_cli.preload_hold_steps,
    args_cli.ramp_steps,
    args_cli.drive_steps,
) < 1:
    parser.error("settle, preload, ramp, and drive steps must be positive")
if args_cli.max_candidates is not None and args_cli.max_candidates < 1:
    parser.error("--max_candidates must be positive")
if args_cli.warmup_steps < 0:
    parser.error("--warmup_steps must be non-negative")
if min(args_cli.gui_start_hold_s, args_cli.gui_hold_s) < 0.0:
    parser.error("GUI hold times must be non-negative")
if args_cli.candidate_index is not None and args_cli.candidate_indices is not None:
    parser.error("use either --candidate_index or --candidate_indices, not both")
if min(
    args_cli.static_friction,
    args_cli.dynamic_friction,
    args_cli.contact_force_threshold,
) <= 0.0:
    parser.error("friction and contact threshold must be positive")
if args_cli.wheel_velocity_gain < 0.0:
    parser.error("--wheel_velocity_gain must be non-negative")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_inv, quat_mul  # noqa: E402

from LeggedManip_Lab.assets.b2w_z1.b2w_z1_articulation_cfg import (  # noqa: E402
    B2W_Z1_CFG,
    B2W_Z1_USD,
)


DT = 0.005
UPPER_LEG_M = 0.35
LOWER_LEG_M = 0.35
HIP_LINK_Y_M = (0.11973, -0.11973, 0.11973, -0.11973)
HIP_ORIGIN_X_M = (0.3285, 0.3285, -0.3285, -0.3285)
HIP_ORIGIN_Y_M = (0.072, -0.072, 0.072, -0.072)
LEG_NAMES = ("FL", "FR", "RL", "RR")
LEG_JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in LEG_NAMES
    for joint in ("hip", "thigh", "calf")
]
WHEEL_JOINT_NAMES = [f"{leg}_wheel_joint" for leg in LEG_NAMES]
WHEEL_BODY_NAMES = [f"{leg}_wheel" for leg in LEG_NAMES]
DIAGONAL_PHASE = (0.0, 0.5, 0.5, 0.0)
CRAWL_PHASE = (0.75, 0.25, 0.0, 0.5)
POSITIVE_YAW_WHEEL_SIGNS = (-1.0, 1.0, -1.0, 1.0)


@dataclass(frozen=True)
class Candidate:
    index: int
    strategy: str
    direction: int
    hip_bias_rad: float
    yaw_stride_rad: float
    frequency_hz: float
    lift_height_m: float
    wheel_speed_radps: float
    phase_shift_cycles: float
    duty_factor: float = 0.5
    body_shift_m: float = 0.0


def _candidate_table() -> list[Candidate]:
    """Return a stable table so a reported index can be replayed later."""
    rows = [
        Candidate(0, "fixed_leg_wheel_only", 1, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0),
        Candidate(1, "fixed_leg_wheel_only", -1, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0),
    ]
    for direction, hip_bias in itertools.product(
        (1, -1), (-0.06, -0.04, -0.02, 0.02, 0.04, 0.06)
    ):
        rows.append(
            Candidate(
                index=len(rows),
                strategy="four_contact_hip_bias",
                direction=direction,
                hip_bias_rad=hip_bias,
                yaw_stride_rad=0.0,
                frequency_hz=0.0,
                lift_height_m=0.0,
                wheel_speed_radps=4.0,
                phase_shift_cycles=0.0,
            )
        )
    signed_strides = (0.08, -0.08, 0.16, -0.16, 0.24, -0.24)
    for stride, frequency, lift, wheel_speed in itertools.product(
        signed_strides,
        (0.6, 0.9),
        (0.0, 0.05, 0.09),
        (0.0, 1.5),
    ):
        rows.append(
            Candidate(
                index=len(rows),
                strategy="diagonal_reposition",
                direction=1 if stride > 0.0 else -1,
                hip_bias_rad=0.0,
                yaw_stride_rad=stride,
                frequency_hz=frequency,
                lift_height_m=lift,
                wheel_speed_radps=wheel_speed,
                phase_shift_cycles=0.0,
            )
        )
    # Append the alternate diagonal after the original table so indices from
    # earlier reports remain replayable.
    for stride, frequency, lift, wheel_speed in itertools.product(
        signed_strides,
        (0.6, 0.9),
        (0.0, 0.05, 0.09),
        (0.0, 1.5),
    ):
        rows.append(
            Candidate(
                index=len(rows),
                strategy="diagonal_reposition",
                direction=1 if stride > 0.0 else -1,
                hip_bias_rad=0.0,
                yaw_stride_rad=stride,
                frequency_hz=frequency,
                lift_height_m=lift,
                wheel_speed_radps=wheel_speed,
                phase_shift_cycles=0.5,
            )
        )
    # Refine around the best coarse parameters separately for each direction.
    for stride, frequency, lift, wheel_speed in itertools.product(
        (0.20, 0.24, 0.28, 0.32),
        (0.5, 0.6),
        (0.04, 0.05, 0.06),
        (1.5, 2.0),
    ):
        rows.append(
            Candidate(
                index=len(rows),
                strategy="diagonal_refined",
                direction=1,
                hip_bias_rad=0.0,
                yaw_stride_rad=stride,
                frequency_hz=frequency,
                lift_height_m=lift,
                wheel_speed_radps=wheel_speed,
                phase_shift_cycles=0.5,
            )
        )
    for stride, frequency, lift, wheel_speed in itertools.product(
        (-0.20, -0.24, -0.28),
        (0.5, 0.6),
        (0.07, 0.09),
        (1.0, 1.5, 2.0),
    ):
        rows.append(
            Candidate(
                index=len(rows),
                strategy="diagonal_refined",
                direction=-1,
                hip_bias_rad=0.0,
                yaw_stride_rad=stride,
                frequency_hz=frequency,
                lift_height_m=lift,
                wheel_speed_radps=wheel_speed,
                phase_shift_cycles=0.0,
            )
        )
    # Matched fixed-leg controls for the refined gait's peak wheel speeds.
    for direction, wheel_speed in itertools.product((1, -1), (1.0, 1.5, 2.0)):
        rows.append(
            Candidate(
                index=len(rows),
                strategy="fixed_leg_wheel_only",
                direction=direction,
                hip_bias_rad=0.0,
                yaw_stride_rad=0.0,
                frequency_hz=0.0,
                lift_height_m=0.0,
                wheel_speed_radps=wheel_speed,
                phase_shift_cycles=0.0,
            )
        )
    # Wheel-speed target is zero in this refinement: the wheel velocity loop
    # actively holds/brakes the wheels while the legs reposition them.  This
    # isolates yaw created by leg motion from commanded differential rolling.
    for direction, stride_magnitude, frequency, lift in itertools.product(
        (1, -1),
        (0.04, 0.06, 0.08, 0.10, 0.12),
        (0.4, 0.5, 0.6, 0.7),
        (0.07, 0.09, 0.11),
    ):
        rows.append(
            Candidate(
                index=len(rows),
                strategy="diagonal_leg_only_refined",
                direction=direction,
                hip_bias_rad=0.0,
                yaw_stride_rad=direction * stride_magnitude,
                frequency_hz=frequency,
                lift_height_m=lift,
                wheel_speed_radps=0.0,
                phase_shift_cycles=0.5 if direction > 0 else 0.0,
            )
        )
    # Exact-cycle factorial controls reveal whether yaw follows the requested
    # stride sign or merely the selected starting diagonal / terminal phase.
    for phase_shift, stride, lift in itertools.product(
        (0.0, 0.5),
        (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12),
        (0.06, 0.09),
    ):
        rows.append(
            Candidate(
                index=len(rows),
                strategy="diagonal_factorial",
                direction=1 if stride >= 0.0 else -1,
                hip_bias_rad=0.0,
                yaw_stride_rad=stride,
                frequency_hz=0.625,
                lift_height_m=lift,
                wheel_speed_radps=0.0,
                phase_shift_cycles=phase_shift,
                duty_factor=0.5,
            )
        )
    # A three-wheel-support crawl avoids the diagonal gait's line-support
    # singularity. One wheel-foot is repositioned at a time.
    for stride, lift in itertools.product(
        (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12),
        (0.04, 0.06, 0.08),
    ):
        rows.append(
            Candidate(
                index=len(rows),
                strategy="crawl_leg_only",
                direction=1 if stride >= 0.0 else -1,
                hip_bias_rad=0.0,
                yaw_stride_rad=stride,
                frequency_hz=0.625,
                lift_height_m=lift,
                wheel_speed_radps=0.0,
                phase_shift_cycles=0.0,
                duty_factor=0.75,
            )
        )
    # Shift the base projection away from the next swing corner before lifting
    # it. This gives the remaining three wheels a non-zero static support
    # margin instead of leaving the COM on the support triangle's edge.
    for stride, lift, body_shift in itertools.product(
        (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12),
        (0.03, 0.05, 0.07),
        (0.0, 0.03, 0.05),
    ):
        rows.append(
            Candidate(
                index=len(rows),
                strategy="crawl_weight_shift",
                direction=1 if stride >= 0.0 else -1,
                hip_bias_rad=0.0,
                yaw_stride_rad=stride,
                frequency_hz=0.5,
                lift_height_m=lift,
                wheel_speed_radps=0.0,
                phase_shift_cycles=0.0,
                duty_factor=0.8,
                body_shift_m=body_shift,
            )
        )
    return rows


ALL_CANDIDATES = _candidate_table()
if args_cli.candidate_index is not None:
    if not 0 <= args_cli.candidate_index < len(ALL_CANDIDATES):
        parser.error(f"--candidate_index must be in [0, {len(ALL_CANDIDATES) - 1}]")
    CANDIDATES = [ALL_CANDIDATES[args_cli.candidate_index]]
elif args_cli.candidate_indices is not None:
    invalid_indices = [
        index for index in args_cli.candidate_indices if not 0 <= index < len(ALL_CANDIDATES)
    ]
    if invalid_indices:
        parser.error(
            f"candidate indices must be in [0, {len(ALL_CANDIDATES) - 1}]; "
            f"invalid: {invalid_indices}"
        )
    if len(set(args_cli.candidate_indices)) != len(args_cli.candidate_indices):
        parser.error("--candidate_indices must not contain duplicates")
    CANDIDATES = [ALL_CANDIDATES[index] for index in args_cli.candidate_indices]
else:
    CANDIDATES = [
        candidate
        for candidate in ALL_CANDIDATES
        if args_cli.strategies is None or candidate.strategy in args_cli.strategies
        if (
            args_cli.phase_shift_cycles is None
            or candidate.strategy != "diagonal_reposition"
            or candidate.phase_shift_cycles == args_cli.phase_shift_cycles
        )
    ][: args_cli.max_candidates]


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


@configclass
class TurnSweepSceneCfg(InteractiveSceneCfg):
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


def _minimum_jerk(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, 0.0, 1.0)
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def _relative_rpy(start_quat_w: torch.Tensor, current_quat_w: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return euler_xyz_from_quat(quat_mul(quat_inv(start_quat_w), current_quat_w))


def _git_head(repository: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


class LegWheelTurnSweep:
    def __init__(self) -> None:
        sim_cfg = sim_utils.SimulationCfg(
            dt=DT,
            device=args_cli.device,
            physx=sim_utils.PhysxCfg(
                min_velocity_iteration_count=1,
                enable_external_forces_every_iteration=True,
            ),
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)
        scene_cfg = TurnSweepSceneCfg(num_envs=len(CANDIDATES), env_spacing=3.0)
        for actuator_cfg in scene_cfg.robot.actuators.values():
            actuator_cfg.min_delay = 0
            actuator_cfg.max_delay = 0
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()
        self.scene.update(DT)
        self.robot: Articulation = self.scene["robot"]
        self.contacts: ContactSensor = self.scene["contacts"]

        if not args_cli.headless:
            self.sim.set_camera_view(eye=[2.5, 2.5, 2.0], target=[0.0, 0.0, 0.35])

        self.leg_joint_ids, found_leg_names = self.robot.find_joints(
            LEG_JOINT_NAMES, preserve_order=True
        )
        self.wheel_joint_ids, found_wheel_names = self.robot.find_joints(
            WHEEL_JOINT_NAMES, preserve_order=True
        )
        self.wheel_contact_body_ids, found_contact_wheel_bodies = self.contacts.find_bodies(
            WHEEL_BODY_NAMES, preserve_order=True
        )
        self.wheel_robot_body_ids, found_robot_wheel_bodies = self.robot.find_bodies(
            WHEEL_BODY_NAMES, preserve_order=True
        )
        if found_leg_names != LEG_JOINT_NAMES:
            raise AssertionError(f"unexpected leg joint order: {found_leg_names}")
        if found_wheel_names != WHEEL_JOINT_NAMES:
            raise AssertionError(f"unexpected wheel joint order: {found_wheel_names}")
        if found_contact_wheel_bodies != WHEEL_BODY_NAMES:
            raise AssertionError(
                f"unexpected contact wheel body order: {found_contact_wheel_bodies}"
            )
        if found_robot_wheel_bodies != WHEEL_BODY_NAMES:
            raise AssertionError(
                f"unexpected robot wheel body order: {found_robot_wheel_bodies}"
            )

        self.leg_joint_ids_tensor = torch.tensor(self.leg_joint_ids, device=self.sim.device)
        self.wheel_joint_ids_tensor = torch.tensor(self.wheel_joint_ids, device=self.sim.device)
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()
        self.default_joint_vel = self.robot.data.default_joint_vel.clone()
        self.default_root_state = self.robot.data.default_root_state.clone()
        self.joint_limits = self.robot.data.soft_joint_pos_limits[:, self.leg_joint_ids].clone()
        self.nonwheel_contact_ids = [
            index
            for index, name in enumerate(self.contacts.body_names)
            if name not in WHEEL_BODY_NAMES
        ]

        self.hip_link_y = torch.tensor(HIP_LINK_Y_M, device=self.sim.device).unsqueeze(0)
        self.hip_origin_x = torch.tensor(HIP_ORIGIN_X_M, device=self.sim.device).unsqueeze(0)
        self.hip_origin_y = torch.tensor(HIP_ORIGIN_Y_M, device=self.sim.device).unsqueeze(0)
        diagonal_phase = torch.tensor(DIAGONAL_PHASE, device=self.sim.device).unsqueeze(0)
        crawl_phase = torch.tensor(CRAWL_PHASE, device=self.sim.device).unsqueeze(0)
        self.wheel_yaw_signs = torch.tensor(
            POSITIVE_YAW_WHEEL_SIGNS, device=self.sim.device
        ).unsqueeze(0)

        self.direction = torch.tensor(
            [candidate.direction for candidate in CANDIDATES],
            device=self.sim.device,
            dtype=torch.float32,
        )
        self.yaw_stride = torch.tensor(
            [candidate.yaw_stride_rad for candidate in CANDIDATES],
            device=self.sim.device,
        )
        self.frequency = torch.tensor(
            [candidate.frequency_hz for candidate in CANDIDATES],
            device=self.sim.device,
        )
        self.lift_height = torch.tensor(
            [candidate.lift_height_m for candidate in CANDIDATES],
            device=self.sim.device,
        )
        self.phase_shift = torch.tensor(
            [candidate.phase_shift_cycles for candidate in CANDIDATES],
            device=self.sim.device,
        )
        self.wheel_speed = torch.tensor(
            [candidate.wheel_speed_radps for candidate in CANDIDATES],
            device=self.sim.device,
        )
        self.is_wheel_only = torch.tensor(
            [candidate.strategy == "fixed_leg_wheel_only" for candidate in CANDIDATES],
            device=self.sim.device,
        )
        self.is_hip_bias = torch.tensor(
            [candidate.strategy == "four_contact_hip_bias" for candidate in CANDIDATES],
            device=self.sim.device,
        )
        self.is_gait = torch.tensor(
            [
                candidate.strategy.startswith("diagonal_")
                or candidate.strategy.startswith("crawl_")
                for candidate in CANDIDATES
            ],
            device=self.sim.device,
        )
        self.is_crawl = torch.tensor(
            [candidate.strategy.startswith("crawl_") for candidate in CANDIDATES],
            device=self.sim.device,
        )
        self.is_weight_shift = torch.tensor(
            [candidate.strategy == "crawl_weight_shift" for candidate in CANDIDATES],
            device=self.sim.device,
        )
        self.gait_phase = torch.where(
            self.is_crawl.unsqueeze(1), crawl_phase, diagonal_phase
        )
        self.duty_factor = torch.tensor(
            [candidate.duty_factor for candidate in CANDIDATES],
            device=self.sim.device,
        ).unsqueeze(1)
        self.body_shift = torch.tensor(
            [candidate.body_shift_m for candidate in CANDIDATES],
            device=self.sim.device,
        ).unsqueeze(1)
        self.hip_bias = torch.tensor(
            [candidate.hip_bias_rad for candidate in CANDIDATES],
            device=self.sim.device,
        )

        nominal_leg = self.default_joint_pos[:, self.leg_joint_ids].reshape(-1, 4, 3)
        self.nominal_foot_rel = self._forward_kinematics(nominal_leg)
        self.nominal_foot_abs_x = self.nominal_foot_rel[..., 0] + self.hip_origin_x
        self.nominal_foot_abs_y = self.nominal_foot_rel[..., 1] + self.hip_origin_y
        self.wheel_effort_limit_nm = float(
            scene_cfg.robot.actuators["wheels"].effort_limit_sim
        )

    def _forward_kinematics(self, joint_pos: torch.Tensor) -> torch.Tensor:
        hip = joint_pos[..., 0]
        thigh = joint_pos[..., 1]
        calf = joint_pos[..., 2]
        sagittal_x = -UPPER_LEG_M * torch.sin(thigh) - LOWER_LEG_M * torch.sin(thigh + calf)
        sagittal_z = -UPPER_LEG_M * torch.cos(thigh) - LOWER_LEG_M * torch.cos(thigh + calf)
        lateral_y = torch.cos(hip) * self.hip_link_y - torch.sin(hip) * sagittal_z
        vertical_z = torch.sin(hip) * self.hip_link_y + torch.cos(hip) * sagittal_z
        return torch.stack((sagittal_x, lateral_y, vertical_z), dim=-1)

    def _inverse_kinematics(
        self, sagittal_x: torch.Tensor, lateral_y: torch.Tensor, vertical_z: torch.Tensor
    ) -> torch.Tensor:
        yz_radius_sq = lateral_y.square() + vertical_z.square()
        sagittal_z = -torch.sqrt(torch.clamp(yz_radius_sq - self.hip_link_y.square(), min=1.0e-6))
        hip = torch.atan2(vertical_z, lateral_y) - torch.atan2(sagittal_z, self.hip_link_y)
        hip = torch.atan2(torch.sin(hip), torch.cos(hip))

        reach_sq = sagittal_x.square() + sagittal_z.square()
        cosine_calf = (
            reach_sq - UPPER_LEG_M**2 - LOWER_LEG_M**2
        ) / (2.0 * UPPER_LEG_M * LOWER_LEG_M)
        calf = -torch.acos(torch.clamp(cosine_calf, -0.999, 0.999))
        direction_from_down = torch.atan2(-sagittal_x, -sagittal_z)
        thigh = direction_from_down - torch.atan2(
            LOWER_LEG_M * torch.sin(calf),
            UPPER_LEG_M + LOWER_LEG_M * torch.cos(calf),
        )
        return torch.stack((hip, thigh, calf), dim=-1)

    def _step(self, joint_target: torch.Tensor, wheel_target: torch.Tensor) -> None:
        self.robot.set_joint_position_target(joint_target)
        self.robot.set_joint_velocity_target(self.default_joint_vel)
        self.robot.set_joint_velocity_target(
            wheel_target, joint_ids=self.wheel_joint_ids_tensor
        )
        self.scene.write_data_to_sim()
        self.sim.step(render=not args_cli.headless)
        self.scene.update(DT)

    def _reset_and_settle(self) -> None:
        root_state = self.default_root_state.clone()
        root_state[:, 0:3] += self.scene.env_origins
        self.robot.write_root_state_to_sim(root_state)
        self.robot.write_joint_state_to_sim(self.default_joint_pos, self.default_joint_vel)
        self.scene.reset()
        wheel_zeros = torch.zeros((len(CANDIDATES), 4), device=self.sim.device)
        for _ in range(args_cli.settle_steps):
            self._step(self.default_joint_pos, wheel_zeros)
        if float(torch.min(self.robot.data.root_pos_w[:, 2])) < 0.35:
            raise AssertionError("at least one candidate collapsed during settling")

    def _preload(self) -> None:
        """Smoothly apply static hip-load candidates before measuring yaw."""
        wheel_zeros = torch.zeros((len(CANDIDATES), 4), device=self.sim.device)
        nominal_leg = self.default_joint_pos[:, self.leg_joint_ids].reshape(-1, 4, 3)
        total_steps = args_cli.preload_ramp_steps + args_cli.preload_hold_steps
        for step in range(total_steps):
            progress = min((step + 1) / args_cli.preload_ramp_steps, 1.0)
            progress = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
            desired_leg = nominal_leg.clone()
            desired_leg[..., 0] += (
                self.is_hip_bias.to(torch.float32) * self.hip_bias * progress
            ).unsqueeze(1)
            limits = self.joint_limits.reshape(-1, 4, 3, 2)
            desired_leg = torch.maximum(
                torch.minimum(desired_leg, limits[..., 1]), limits[..., 0]
            )
            joint_target = self.default_joint_pos.clone()
            joint_target[:, self.leg_joint_ids] = desired_leg.reshape(-1, 12)
            self._step(joint_target, wheel_zeros)

    def _targets(self, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_s = step * DT
        envelope_value = min((step + 1) / args_cli.ramp_steps, 1.0)
        envelope_value = 10.0 * envelope_value**3 - 15.0 * envelope_value**4 + 6.0 * envelope_value**5
        envelope = torch.full(
            (len(CANDIDATES), 1), envelope_value, device=self.sim.device
        )

        cycle = torch.remainder(
            time_s * self.frequency.unsqueeze(1)
            + self.gait_phase
            + self.phase_shift.unsqueeze(1),
            1.0,
        )
        stance = cycle < self.duty_factor
        phase_progress = torch.where(
            stance,
            cycle / self.duty_factor,
            (cycle - self.duty_factor) / (1.0 - self.duty_factor),
        )
        smooth_progress = _minimum_jerk(phase_progress)
        path_coordinate = torch.where(
            stance, smooth_progress - 0.5, 0.5 - smooth_progress
        )
        swing_lift = torch.where(
            stance,
            torch.zeros_like(phase_progress),
            torch.sin(math.pi * phase_progress),
        )

        yaw_displacement_x = self.yaw_stride.unsqueeze(1) * self.nominal_foot_abs_y
        yaw_displacement_y = -self.yaw_stride.unsqueeze(1) * self.nominal_foot_abs_x
        swing_center = 0.5 * (self.duty_factor + 1.0)
        distance_to_swing_center = torch.remainder(
            cycle - swing_center + 0.5, 1.0
        ) - 0.5
        shift_half_width = 0.20
        shift_weight = torch.where(
            torch.abs(distance_to_swing_center) < shift_half_width,
            0.5
            * (
                1.0
                + torch.cos(
                    math.pi * distance_to_swing_center / shift_half_width
                )
            ),
            torch.zeros_like(distance_to_swing_center),
        )
        shift_weight *= self.is_weight_shift.unsqueeze(1)
        common_foot_shift_x = self.body_shift * torch.sum(
            shift_weight * torch.sign(self.hip_origin_x), dim=1, keepdim=True
        )
        common_foot_shift_y = self.body_shift * torch.sum(
            shift_weight * torch.sign(self.hip_origin_y), dim=1, keepdim=True
        )
        desired_abs_x = (
            self.nominal_foot_abs_x
            + envelope * path_coordinate * yaw_displacement_x
            + envelope * common_foot_shift_x
        )
        desired_abs_y = (
            self.nominal_foot_abs_y
            + envelope * path_coordinate * yaw_displacement_y
            + envelope * common_foot_shift_y
        )
        desired_rel_z = self.nominal_foot_rel[..., 2] + (
            envelope
            * self.lift_height.unsqueeze(1)
            * swing_lift
        )
        desired_leg = self._inverse_kinematics(
            desired_abs_x - self.hip_origin_x,
            desired_abs_y - self.hip_origin_y,
            desired_rel_z,
        )

        nominal_leg = self.default_joint_pos[:, self.leg_joint_ids].reshape(-1, 4, 3)
        desired_leg = torch.where(self.is_gait[:, None, None], desired_leg, nominal_leg)
        desired_leg[..., 0] += (
            self.is_hip_bias.to(torch.float32) * self.hip_bias
        ).unsqueeze(1)
        limits = self.joint_limits.reshape(-1, 4, 3, 2)
        desired_leg = torch.maximum(
            torch.minimum(desired_leg, limits[..., 1]), limits[..., 0]
        )
        joint_target = self.default_joint_pos.clone()
        joint_target[:, self.leg_joint_ids] = desired_leg.reshape(-1, 12)

        gait_wheel_target = (
            envelope
            * self.direction.unsqueeze(1)
            * self.wheel_speed.unsqueeze(1)
            * self.wheel_yaw_signs
            * stance.to(torch.float32)
        )
        baseline_wheel_target = (
            envelope
            * self.direction.unsqueeze(1)
            * self.wheel_speed.unsqueeze(1)
            * self.wheel_yaw_signs
        )
        wheel_target = torch.where(
            self.is_gait.unsqueeze(1), gait_wheel_target, baseline_wheel_target
        )
        stance = torch.where(
            self.is_gait.unsqueeze(1), stance, torch.ones_like(stance)
        )
        return joint_target, wheel_target, stance

    def run(self) -> list[dict[str, object]]:
        self._reset_and_settle()
        self._preload()
        for step in range(args_cli.warmup_steps):
            joint_target, wheel_target, _ = self._targets(step)
            self._step(joint_target, wheel_target)
        self._hold_gui(args_cli.gui_start_hold_s)
        start_pos_w = self.robot.data.root_pos_w.clone()
        start_quat_w = self.robot.data.root_quat_w.clone()
        start_base_height = self.robot.data.root_pos_w[:, 2].clone()
        start_wheel_center_height = self.robot.data.body_pos_w[
            :, self.wheel_robot_body_ids, 2
        ].clone()

        count = len(CANDIDATES)
        integrated_yaw = torch.zeros(count, device=self.sim.device)
        integrated_yaw_history = torch.zeros(
            (args_cli.drive_steps, count), device=self.sim.device
        )
        root_position_history = torch.zeros(
            (args_cli.drive_steps, count, 3), device=self.sim.device
        )
        max_roll_pitch = torch.zeros(count, device=self.sim.device)
        min_height = self.robot.data.root_pos_w[:, 2].clone()
        wheel_contact_count = torch.zeros((count, 4), device=self.sim.device)
        wheel_fz_sum = torch.zeros((count, 4), device=self.sim.device)
        stance_sample_count = torch.zeros((count, 4), device=self.sim.device)
        stance_contact_count = torch.zeros((count, 4), device=self.sim.device)
        swing_sample_count = torch.zeros((count, 4), device=self.sim.device)
        swing_contact_count = torch.zeros((count, 4), device=self.sim.device)
        swing_center_rise_sum = torch.zeros((count, 4), device=self.sim.device)
        max_swing_center_rise = torch.full(
            (count, 4), -torch.inf, device=self.sim.device
        )
        min_simultaneous_wheel_contacts = torch.full(
            (count,), 4.0, device=self.sim.device
        )
        unintended_contact_count = torch.zeros(count, device=self.sim.device)
        max_leg_torque = torch.zeros(count, device=self.sim.device)
        max_wheel_torque = torch.zeros(count, device=self.sim.device)
        wheel_saturation_count = torch.zeros(count, device=self.sim.device)
        leg_tracking_error_sq_sum = torch.zeros(count, device=self.sim.device)
        wheel_abs_velocity_sum = torch.zeros(count, device=self.sim.device)
        wheel_abs_rotation = torch.zeros(count, device=self.sim.device)
        wheel_positive_work = torch.zeros(count, device=self.sim.device)
        wheel_braking_work = torch.zeros(count, device=self.sim.device)

        for step in range(args_cli.drive_steps):
            joint_target, wheel_target, stance = self._targets(
                args_cli.warmup_steps + step
            )
            self._step(joint_target, wheel_target)
            roll, pitch, _ = _relative_rpy(start_quat_w, self.robot.data.root_quat_w)
            max_roll_pitch = torch.maximum(
                max_roll_pitch, torch.maximum(torch.abs(roll), torch.abs(pitch))
            )
            min_height = torch.minimum(min_height, self.robot.data.root_pos_w[:, 2])
            integrated_yaw += self.robot.data.root_link_ang_vel_w[:, 2] * DT
            integrated_yaw_history[step] = integrated_yaw
            root_position_history[step] = self.robot.data.root_pos_w

            contact_force = torch.linalg.vector_norm(
                self.contacts.data.net_forces_w, dim=-1
            )
            wheel_contact = (
                contact_force[:, self.wheel_contact_body_ids]
                > args_cli.contact_force_threshold
            )
            wheel_contact_count += wheel_contact
            wheel_fz_sum += torch.abs(
                self.contacts.data.net_forces_w[:, self.wheel_contact_body_ids, 2]
            )
            gait_mask = self.is_gait.unsqueeze(1)
            stance_mask = gait_mask & stance
            swing_mask = gait_mask & ~stance
            stance_sample_count += stance_mask
            stance_contact_count += stance_mask & wheel_contact
            swing_sample_count += swing_mask
            swing_contact_count += swing_mask & wheel_contact
            wheel_center_rise = (
                self.robot.data.body_pos_w[:, self.wheel_robot_body_ids, 2]
                - start_wheel_center_height
            )
            swing_center_rise_sum += torch.where(
                swing_mask, wheel_center_rise, torch.zeros_like(wheel_center_rise)
            )
            max_swing_center_rise = torch.maximum(
                max_swing_center_rise,
                torch.where(
                    swing_mask,
                    wheel_center_rise,
                    torch.full_like(wheel_center_rise, -torch.inf),
                ),
            )
            min_simultaneous_wheel_contacts = torch.minimum(
                min_simultaneous_wheel_contacts,
                torch.sum(wheel_contact.to(torch.float32), dim=1),
            )
            if self.nonwheel_contact_ids:
                unintended_contact_count += torch.any(
                    contact_force[:, self.nonwheel_contact_ids]
                    > args_cli.contact_force_threshold,
                    dim=1,
                )
            max_leg_torque = torch.maximum(
                max_leg_torque,
                torch.max(
                    torch.abs(self.robot.data.applied_torque[:, self.leg_joint_ids]), dim=1
                ).values,
            )
            max_wheel_torque = torch.maximum(
                max_wheel_torque,
                torch.max(
                    torch.abs(self.robot.data.applied_torque[:, self.wheel_joint_ids]), dim=1
                ).values,
            )
            wheel_torque = torch.abs(
                self.robot.data.applied_torque[:, self.wheel_joint_ids]
            )
            wheel_saturation_count += torch.mean(
                (wheel_torque >= 0.99 * self.wheel_effort_limit_nm).to(torch.float32),
                dim=1,
            )
            leg_tracking_error = (
                self.robot.data.joint_pos[:, self.leg_joint_ids]
                - joint_target[:, self.leg_joint_ids]
            )
            leg_tracking_error_sq_sum += torch.mean(
                leg_tracking_error.square(), dim=1
            )
            wheel_velocity = self.robot.data.joint_vel[:, self.wheel_joint_ids]
            wheel_abs_velocity_sum += torch.mean(torch.abs(wheel_velocity), dim=1)
            wheel_abs_rotation += torch.mean(torch.abs(wheel_velocity), dim=1) * DT
            wheel_power = (
                self.robot.data.applied_torque[:, self.wheel_joint_ids]
                * wheel_velocity
            )
            wheel_positive_work += torch.sum(torch.clamp(wheel_power, min=0.0), dim=1) * DT
            wheel_braking_work += torch.sum(torch.clamp(-wheel_power, min=0.0), dim=1) * DT

        displacement_b = quat_apply_inverse(
            start_quat_w, self.robot.data.root_pos_w - start_pos_w
        )
        _, _, final_yaw = _relative_rpy(start_quat_w, self.robot.data.root_quat_w)
        planar_translation = torch.linalg.vector_norm(displacement_b[:, :2], dim=1)
        contact_ratio = wheel_contact_count / args_cli.drive_steps
        mean_wheel_fz = wheel_fz_sum / args_cli.drive_steps
        unintended_ratio = unintended_contact_count / args_cli.drive_steps
        wheel_saturation_ratio = wheel_saturation_count / args_cli.drive_steps
        stance_contact_ratio = stance_contact_count / torch.clamp(
            stance_sample_count, min=1.0
        )
        swing_contact_ratio = swing_contact_count / torch.clamp(
            swing_sample_count, min=1.0
        )
        mean_swing_center_rise = swing_center_rise_sum / torch.clamp(
            swing_sample_count, min=1.0
        )
        max_swing_center_rise = torch.where(
            torch.isfinite(max_swing_center_rise),
            max_swing_center_rise,
            torch.zeros_like(max_swing_center_rise),
        )
        base_height_drop = start_base_height - min_height
        leg_tracking_rmse = torch.sqrt(
            leg_tracking_error_sq_sum / args_cli.drive_steps
        )
        mean_abs_wheel_velocity = wheel_abs_velocity_sum / args_cli.drive_steps

        rows: list[dict[str, object]] = []
        for env_index, candidate in enumerate(CANDIDATES):
            yaw_deg = math.degrees(float(integrated_yaw[env_index]))
            relative_yaw_deg = math.degrees(float(final_yaw[env_index]))
            yaw_difference_rad = math.atan2(
                math.sin(float(final_yaw[env_index] - integrated_yaw[env_index])),
                math.cos(float(final_yaw[env_index] - integrated_yaw[env_index])),
            )
            yaw_estimator_disagreement_deg = abs(math.degrees(yaw_difference_rad))
            direction_progress_deg = candidate.direction * yaw_deg
            translation_m = float(planar_translation[env_index])
            max_roll_pitch_deg = math.degrees(float(max_roll_pitch[env_index]))
            cycle_yaw_deg: list[float] = []
            cycle_translation_m: list[float] = []
            if candidate.frequency_hz > 0.0:
                cycle_steps = round(1.0 / (candidate.frequency_hz * DT))
                previous_yaw_rad = 0.0
                previous_position = start_pos_w[env_index]
                for boundary_step in range(
                    cycle_steps - 1, args_cli.drive_steps, cycle_steps
                ):
                    boundary_yaw_rad = float(
                        integrated_yaw_history[boundary_step, env_index]
                    )
                    cycle_yaw_deg.append(
                        math.degrees(boundary_yaw_rad - previous_yaw_rad)
                    )
                    boundary_position = root_position_history[
                        boundary_step, env_index
                    ]
                    cycle_translation_m.append(
                        float(
                            torch.linalg.vector_norm(
                                boundary_position[:2] - previous_position[:2]
                            )
                        )
                    )
                    previous_yaw_rad = boundary_yaw_rad
                    previous_position = boundary_position
            direction_cycle_yaw_deg = [
                candidate.direction * value for value in cycle_yaw_deg
            ]
            direction_consistent_cycle_ratio = (
                sum(value > 0.0 for value in direction_cycle_yaw_deg)
                / len(direction_cycle_yaw_deg)
                if direction_cycle_yaw_deg
                else None
            )
            min_direction_cycle_yaw_deg = (
                min(direction_cycle_yaw_deg) if direction_cycle_yaw_deg else None
            )
            max_cycle_translation_m = (
                max(cycle_translation_m) if cycle_translation_m else None
            )
            stable = (
                float(base_height_drop[env_index]) <= 0.03
                and max_roll_pitch_deg <= 6.0
                and float(unintended_ratio[env_index]) == 0.0
            )
            required_contact_ratio = (
                0.85 if candidate.strategy.startswith("diagonal_") else 0.95
            )
            if candidate.direction > 0:
                outer_load = mean_wheel_fz[env_index, 1] + mean_wheel_fz[env_index, 3]
            else:
                outer_load = mean_wheel_fz[env_index, 0] + mean_wheel_fz[env_index, 2]
            total_load = torch.sum(mean_wheel_fz[env_index])
            outer_load_share = float(outer_load / torch.clamp(total_load, min=1.0e-6))
            success = (
                stable
                and direction_progress_deg >= 20.0
                and translation_m <= 0.20
                and float(torch.min(contact_ratio[env_index])) >= required_contact_ratio
            )
            is_feet_only_test = (
                candidate.strategy.startswith("diagonal_")
                or candidate.strategy.startswith("crawl_")
            ) and candidate.wheel_speed_radps == 0.0
            sustained_feet_turn = (
                is_feet_only_test
                and stable
                and len(direction_cycle_yaw_deg) >= 4
                and min_direction_cycle_yaw_deg is not None
                and min_direction_cycle_yaw_deg >= 2.0
                and direction_consistent_cycle_ratio == 1.0
                and max_cycle_translation_m is not None
                and max_cycle_translation_m <= 0.02
                and float(torch.min(stance_contact_ratio[env_index])) >= 0.95
                and float(torch.max(swing_contact_ratio[env_index])) <= 0.50
                and float(torch.min(max_swing_center_rise[env_index])) >= 0.02
                and float(wheel_saturation_ratio[env_index]) <= 0.01
                and yaw_estimator_disagreement_deg <= 1.0
            )
            score = (
                direction_progress_deg
                - 25.0 * translation_m
                - 2.0 * max(max_roll_pitch_deg - 10.0, 0.0)
                - 100.0 * float(unintended_ratio[env_index])
            )
            rows.append(
                {
                    **asdict(candidate),
                    "integrated_yaw_deg": yaw_deg,
                    "relative_quaternion_yaw_deg": relative_yaw_deg,
                    "yaw_estimator_disagreement_deg": yaw_estimator_disagreement_deg,
                    "direction_progress_deg": direction_progress_deg,
                    "cycle_yaw_deg": cycle_yaw_deg,
                    "direction_cycle_yaw_deg": direction_cycle_yaw_deg,
                    "direction_consistent_cycle_ratio": direction_consistent_cycle_ratio,
                    "cycle_translation_m": cycle_translation_m,
                    "max_cycle_translation_m": max_cycle_translation_m,
                    "translation_m": translation_m,
                    "forward_m": float(displacement_b[env_index, 0]),
                    "lateral_m": float(displacement_b[env_index, 1]),
                    "max_roll_pitch_deg": max_roll_pitch_deg,
                    "min_base_height_m": float(min_height[env_index]),
                    "base_height_drop_m": float(base_height_drop[env_index]),
                    "wheel_contact_ratio": [
                        float(value) for value in contact_ratio[env_index]
                    ],
                    "mean_wheel_contact_ratio": float(torch.mean(contact_ratio[env_index])),
                    "stance_wheel_contact_ratio": [
                        float(value) for value in stance_contact_ratio[env_index]
                    ],
                    "swing_wheel_contact_ratio": [
                        float(value) for value in swing_contact_ratio[env_index]
                    ],
                    "mean_swing_wheel_center_rise_m": [
                        float(value) for value in mean_swing_center_rise[env_index]
                    ],
                    "max_swing_wheel_center_rise_m": [
                        float(value) for value in max_swing_center_rise[env_index]
                    ],
                    "min_simultaneous_wheel_contacts": int(
                        min_simultaneous_wheel_contacts[env_index]
                    ),
                    "mean_wheel_vertical_force_n": [
                        float(value) for value in mean_wheel_fz[env_index]
                    ],
                    "outer_wheel_load_share": outer_load_share,
                    "unintended_contact_ratio": float(unintended_ratio[env_index]),
                    "max_leg_torque_nm": float(max_leg_torque[env_index]),
                    "max_wheel_torque_nm": float(max_wheel_torque[env_index]),
                    "wheel_torque_saturation_ratio": float(
                        wheel_saturation_ratio[env_index]
                    ),
                    "leg_joint_tracking_rmse_rad": float(
                        leg_tracking_rmse[env_index]
                    ),
                    "mean_abs_wheel_velocity_radps": float(
                        mean_abs_wheel_velocity[env_index]
                    ),
                    "mean_abs_wheel_rotation_rad": float(
                        wheel_abs_rotation[env_index]
                    ),
                    "wheel_positive_work_j": float(wheel_positive_work[env_index]),
                    "wheel_braking_work_j": float(wheel_braking_work[env_index]),
                    "stable": stable,
                    "success": success,
                    "sustained_feet_turn": sustained_feet_turn,
                    "score": score,
                }
            )
        return rows

    def _hold_gui(self, duration_s: float) -> None:
        if args_cli.headless or duration_s <= 0.0:
            return
        deadline = time.perf_counter() + duration_s
        while time.perf_counter() < deadline:
            self.sim.render()
            time.sleep(1.0 / 60.0)

    def hold_gui(self) -> None:
        self._hold_gui(args_cli.gui_hold_s)


def main() -> None:
    runner = LegWheelTurnSweep()
    rows = runner.run()
    ordered = sorted(rows, key=lambda row: float(row["score"]), reverse=True)

    print("\nTop leg-wheel turn candidates")
    print(
        "idx strategy dir hip stride freq lift wheel phase duty shift | "
        "yaw progress cycle_min cycle_sign trans rp drop swing_contact "
        "swing_rise wheel_sat sustained"
    )
    for row in ordered[: min(15, len(ordered))]:
        print(
            f"{row['index']:>3} {row['strategy']:<24} {row['direction']:+d} "
            f"{row['hip_bias_rad']:+.2f} {row['yaw_stride_rad']:+.2f} {row['frequency_hz']:.1f} "
            f"{row['lift_height_m']:.2f} {row['wheel_speed_radps']:.1f} "
            f"{row['phase_shift_cycles']:.1f} {row['duty_factor']:.2f} "
            f"{row['body_shift_m']:.2f} | "
            f"{row['integrated_yaw_deg']:+7.2f} {row['direction_progress_deg']:6.2f} "
            f"{min(row['direction_cycle_yaw_deg'], default=float('nan')):+6.2f} "
            f"{row['direction_consistent_cycle_ratio']!s:>4} "
            f"{row['translation_m']:.3f} {row['max_roll_pitch_deg']:.1f} "
            f"{row['base_height_drop_m']:.3f} "
            f"{max(row['swing_wheel_contact_ratio'], default=float('nan')):.2f} "
            f"{min(row['max_swing_wheel_center_rise_m'], default=float('nan')):.3f} "
            f"{row['wheel_torque_saturation_ratio']:.2f} "
            f"{row['sustained_feet_turn']}"
        )

    repository = Path(__file__).resolve().parents[2]
    report = {
        "experiment": "b2w_z1_diagnostic_leg_wheel_turn_sweep",
        "warning": "Diagnostic scripted gait; not a production or Unitree controller.",
        "git_head": _git_head(repository),
        "command": [sys.executable, *sys.argv],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(torch.cuda.current_device())
        if torch.cuda.is_available()
        else None,
        "asset_path": str(Path(B2W_Z1_USD).resolve()),
        "physics_dt_s": DT,
        "settle_steps": args_cli.settle_steps,
        "preload_ramp_steps": args_cli.preload_ramp_steps,
        "preload_hold_steps": args_cli.preload_hold_steps,
        "ramp_steps": args_cli.ramp_steps,
        "warmup_steps": args_cli.warmup_steps,
        "drive_steps": args_cli.drive_steps,
        "static_friction": args_cli.static_friction,
        "dynamic_friction": args_cli.dynamic_friction,
        "wheel_velocity_gain": args_cli.wheel_velocity_gain,
        "wheel_zero_speed_semantics": (
            "active velocity-servo brake/hold"
            if args_cli.wheel_velocity_gain > 0.0
            else "unactuated passive wheel"
        ),
        "legacy_success_criteria": {
            "direction_progress_deg_min": 20.0,
            "translation_m_max": 0.20,
            "max_roll_pitch_deg": 6.0,
            "base_height_drop_m_max": 0.03,
            "unintended_contact_ratio": 0.0,
            "min_wheel_contact_ratio_diagonal": 0.85,
            "min_wheel_contact_ratio_non_diagonal": 0.95,
        },
        "success_count": sum(bool(row["success"]) for row in rows),
        "sustained_feet_turn_count": sum(
            bool(row["sustained_feet_turn"]) for row in rows
        ),
        "sustained_feet_turn_criteria": {
            "complete_measured_cycles_min": 4,
            "direction_yaw_per_cycle_deg_min": 2.0,
            "direction_consistent_cycle_ratio": 1.0,
            "translation_per_cycle_m_max": 0.02,
            "max_roll_pitch_deg": 6.0,
            "base_height_drop_m_max": 0.03,
            "stance_wheel_contact_ratio_min": 0.95,
            "swing_wheel_contact_ratio_max": 0.50,
            "each_wheel_max_swing_center_rise_m_min": 0.02,
            "unintended_contact_ratio": 0.0,
            "wheel_torque_saturation_ratio_max": 0.01,
            "yaw_estimator_disagreement_deg_max": 1.0,
        },
        "best_candidate": ordered[0] if ordered else None,
        "results": rows,
    }
    if args_cli.report is not None:
        args_cli.report.parent.mkdir(parents=True, exist_ok=True)
        args_cli.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args_cli.report}")
    else:
        print(json.dumps(report, indent=2))

    runner.hold_gui()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
