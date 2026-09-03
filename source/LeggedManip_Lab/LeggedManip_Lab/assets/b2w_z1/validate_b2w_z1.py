#!/usr/bin/env python3
"""Load the B2-W + Z1 USD as an Isaac Lab articulation and smoke-test it."""

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=250, help="Number of physics steps to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import math  # noqa: E402

import torch  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from LeggedManip_Lab.assets.b2w_z1.b2w_z1_articulation_cfg import (  # noqa: E402
    B2W_Z1_ARM_HOME_JOINT_POS,
    B2W_Z1_ARM_STOW_JOINT_POS,
    B2W_Z1_CFG,
    B2W_Z1_TRAINING_CFG,
    B2W_Z1_USD,
)


EXPECTED_WHEEL_JOINTS = {f"{leg}_wheel_joint" for leg in ("FL", "FR", "RL", "RR")}
EXPECTED_MOUNT_POSITION = (0.211, 0.0, 0.110)
EXPECTED_TCP_POSITION = (0.145, 0.0, 0.0)


def _quaternion_wxyz(quaternion) -> tuple[float, float, float, float]:
    imaginary = quaternion.GetImaginary()
    return (
        float(quaternion.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )


def _assert_identity_quaternion(quaternion, label: str) -> None:
    actual = _quaternion_wxyz(quaternion)
    expected = (1.0, 0.0, 0.0, 0.0)
    if any(abs(value - target) > 1.0e-6 for value, target in zip(actual, expected)):
        raise AssertionError(f"Unexpected {label} rotation quaternion: {actual}")


def _relationship_targets(joint, relationship_name: str) -> list[str]:
    relationship = (
        joint.GetBody0Rel() if relationship_name == "body0" else joint.GetBody1Rel()
    )
    return [path.pathString for path in relationship.GetTargets()]


def validate_arm_presets() -> float:
    """Check the two reset presets and the limiting distal-arm lidar clearance."""
    for joint_name, expected in B2W_Z1_ARM_STOW_JOINT_POS.items():
        actual = B2W_Z1_CFG.init_state.joint_pos[joint_name]
        if not math.isclose(actual, expected, abs_tol=1.0e-9):
            raise AssertionError(f"Default stow mismatch for {joint_name}: {actual}")
    for joint_name, expected in B2W_Z1_ARM_HOME_JOINT_POS.items():
        actual = B2W_Z1_TRAINING_CFG.init_state.joint_pos[joint_name]
        if not math.isclose(actual, expected, abs_tol=1.0e-9):
            raise AssertionError(f"Training home mismatch for {joint_name}: {actual}")

    q2 = B2W_Z1_ARM_STOW_JOINT_POS["joint2"]
    q3 = B2W_Z1_ARM_STOW_JOINT_POS["joint3"]
    q4 = B2W_Z1_ARM_STOW_JOINT_POS["joint4"]
    if not math.isclose(q2 + q3 + q4, 0.0, abs_tol=1.0e-9):
        raise AssertionError("Stow pose must leave the wrist/gripper horizontal")

    # Exact Z1 chain dimensions from source/b2w_z1.urdf. The stator is the
    # lowest distal collision mesh in this pose (local min z = -36.35 mm).
    # The calculation is in the B2-W base frame and was cross-checked against
    # every transformed STL collision vertex.
    stator_origin_z = (
        0.110
        + 0.0585
        + 0.045
        + 0.350 * math.sin(q2)
        - 0.218 * math.sin(q2 + q3)
        + 0.057 * math.cos(q2 + q3)
    )
    stator_bottom_z = stator_origin_z - 0.03635
    lidar_collider_top_z = 0.17851 - 0.020 + 0.5 * 0.160
    clearance = stator_bottom_z - lidar_collider_top_z
    if clearance < 0.100:
        raise AssertionError(f"Stowed gripper-to-lidar clearance is only {clearance:.3f} m")
    return clearance


def validate_usd_structure() -> None:
    """Validate authored structure before creating the PhysX tensor view."""
    stage = Usd.Stage.Open(B2W_Z1_USD)
    if stage is None:
        raise RuntimeError(f"Could not open USD: {B2W_Z1_USD}")
    if stage.GetDefaultPrim().GetPath().pathString != "/b2w_z1":
        raise AssertionError(f"Unexpected default prim: {stage.GetDefaultPrim().GetPath()}")

    articulation_roots = [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if articulation_roots != ["/b2w_z1/base_link"]:
        raise AssertionError(f"Expected one articulation root; got {articulation_roots}")

    for joint_name in sorted(EXPECTED_WHEEL_JOINTS):
        joint = UsdPhysics.RevoluteJoint.Get(stage, f"/b2w_z1/joints/{joint_name}")
        if not joint:
            raise AssertionError(f"Missing revolute wheel joint: {joint_name}")
        if joint.GetAxisAttr().Get() != "Y":
            raise AssertionError(f"{joint_name} does not rotate around Y")

    mount = UsdPhysics.FixedJoint.Get(stage, "/b2w_z1/joints/z1_mount_joint")
    if not mount:
        raise AssertionError("Missing fixed z1_mount_joint")
    mount_position = mount.GetLocalPos0Attr().Get()
    if any(
        abs(float(actual) - expected) > 1.0e-6
        for actual, expected in zip(mount_position, EXPECTED_MOUNT_POSITION)
    ):
        raise AssertionError(f"Unexpected Z1 mount position: {mount_position}")
    _assert_identity_quaternion(mount.GetLocalRot0Attr().Get(), "Z1 mount")
    if _relationship_targets(mount, "body0") != ["/b2w_z1/base_link"]:
        raise AssertionError(f"Unexpected Z1 mount parent: {mount.GetBody0Rel().GetTargets()}")
    if _relationship_targets(mount, "body1") != ["/b2w_z1/link0"]:
        raise AssertionError(f"Unexpected Z1 mount child: {mount.GetBody1Rel().GetTargets()}")

    tcp = UsdPhysics.FixedJoint.Get(stage, "/b2w_z1/joints/tcp_fixed_joint")
    if not tcp:
        raise AssertionError("Missing nominal tcp_fixed_joint")
    tcp_position = tcp.GetLocalPos0Attr().Get()
    if any(
        abs(float(actual) - expected) > 1.0e-6
        for actual, expected in zip(tcp_position, EXPECTED_TCP_POSITION)
    ):
        raise AssertionError(f"Unexpected nominal TCP position: {tcp_position}")
    _assert_identity_quaternion(tcp.GetLocalRot0Attr().Get(), "nominal TCP")
    if _relationship_targets(tcp, "body0") != ["/b2w_z1/gripper_stator"]:
        raise AssertionError(f"Unexpected TCP parent: {tcp.GetBody0Rel().GetTargets()}")
    if _relationship_targets(tcp, "body1") != ["/b2w_z1/tcp_frame"]:
        raise AssertionError(f"Unexpected TCP child: {tcp.GetBody1Rel().GetTargets()}")

    usd_stow = {**B2W_Z1_ARM_STOW_JOINT_POS, "gripper_joint": -1.0}
    for joint_name, expected_radians in usd_stow.items():
        prim = stage.GetPrimAtPath(f"/b2w_z1/joints/{joint_name}")
        expected_degrees = math.degrees(expected_radians)
        for attribute_name in (
            "state:angular:physics:position",
            "drive:angular:physics:targetPosition",
        ):
            actual_degrees = prim.GetAttribute(attribute_name).Get()
            if actual_degrees is None or not math.isclose(
                float(actual_degrees), expected_degrees, abs_tol=1.0e-5
            ):
                raise AssertionError(
                    f"USD stow mismatch for {joint_name} {attribute_name}: {actual_degrees} deg"
                )


def main() -> None:
    validate_usd_structure()
    stow_clearance = validate_arm_presets()

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1500.0))
    robot = Articulation(B2W_Z1_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    if robot.num_joints != 23:
        raise AssertionError(f"Expected 23 movable joints; got {robot.num_joints}")
    wheel_ids, wheel_names = robot.find_joints(".*_wheel_joint")
    if set(wheel_names) != EXPECTED_WHEEL_JOINTS:
        raise AssertionError(f"Unexpected wheel joints: {wheel_names}")
    arm_ids, arm_names = robot.find_joints("joint[1-6]")
    expected_arm_stow = torch.tensor(
        [[B2W_Z1_ARM_STOW_JOINT_POS[name] for name in arm_names]],
        device=sim.device,
        dtype=robot.data.default_joint_pos.dtype,
    )
    if len(arm_names) != 6 or not torch.allclose(
        robot.data.default_joint_pos[:, arm_ids], expected_arm_stow
    ):
        raise AssertionError(f"Z1 does not have the expected photo-matched stow: {arm_names}")
    base_body_ids, base_body_names = robot.find_bodies("base_link")
    tcp_body_ids, tcp_body_names = robot.find_bodies("tcp_frame")
    if base_body_names != ["base_link"] or tcp_body_names != ["tcp_frame"]:
        raise AssertionError(
            f"Unexpected base/TCP bodies: base={base_body_names}, tcp={tcp_body_names}"
        )

    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()
    wheel_ids_tensor = torch.tensor(wheel_ids, device=sim.device)
    zero_wheel_velocity = torch.zeros((robot.num_instances, len(wheel_ids)), device=sim.device)

    for _ in range(args_cli.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.set_joint_velocity_target(zero_wheel_velocity, joint_ids=wheel_ids_tensor)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())

    root_position = robot.data.root_pos_w[0]
    root_quaternion = robot.data.root_quat_w[0]
    if not torch.isfinite(robot.data.joint_pos).all() or not torch.isfinite(root_position).all():
        raise AssertionError("Simulation produced non-finite articulation state")
    if float(root_position[2]) < 0.35:
        raise AssertionError(f"Robot collapsed during smoke test: base z={float(root_position[2]):.3f} m")
    # Isaac Lab quaternion ordering is (w, x, y, z). This checks that the
    # model remains broadly upright without requiring a trained balance policy.
    upright = 1.0 - 2.0 * float(root_quaternion[1] ** 2 + root_quaternion[2] ** 2)
    if not math.isfinite(upright) or upright < 0.7:
        raise AssertionError(f"Robot tipped during smoke test: upright={upright:.3f}")

    # The photo-matched fold returns the gripper toward the robot front.  Check
    # the semantic direction in simulation instead of relying only on an RPY
    # literal in the source URDF. Quaternion ordering is (w, x, y, z).
    def local_x_axis_world(quaternion: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quaternion.unbind()
        return torch.stack(
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y + w * z),
                2.0 * (x * z - w * y),
            )
        )

    base_x = local_x_axis_world(robot.data.body_quat_w[0, base_body_ids[0]])
    tcp_x = local_x_axis_world(robot.data.body_quat_w[0, tcp_body_ids[0]])
    forward_alignment = float(torch.dot(base_x, tcp_x))
    if not math.isfinite(forward_alignment) or forward_alignment < 0.90:
        raise AssertionError(
            f"Photo-matched TCP does not face robot front: dot={forward_alignment:.3f}"
        )

    print(f"PASS: one articulation, {robot.num_joints} movable joints, four wheels", flush=True)
    print("PASS: Z1 default is the photo-matched provisional stacked stow", flush=True)
    print("PASS: direct-open USD wrapper authors the same stow state", flush=True)
    print("PASS: separate TCP-training config uses the raised ready pose", flush=True)
    print(
        f"PASS: limiting distal-arm vertical clearance above lidar is {1000.0 * stow_clearance:.1f} mm",
        flush=True,
    )
    print("PASS: Z1 mounting plane is 0.035 m above the B2-W deck", flush=True)
    print(
        "PASS: mount/TCP parents, positions, rotations, and forward direction are consistent "
        f"(dot={forward_alignment:.3f})",
        flush=True,
    )
    print(
        f"PASS: {args_cli.steps} steps, base z={float(root_position[2]):.3f} m, upright={upright:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # Keep Kit's fast shutdown from hiding validation failures.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
