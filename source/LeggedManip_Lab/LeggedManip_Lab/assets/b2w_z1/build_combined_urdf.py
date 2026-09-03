#!/usr/bin/env python3
"""Build a B2-W + Z1 URDF from pinned Unitree robot descriptions.

The generated URDF keeps the official B2-W dynamics, adds the official Z1 arm
and gripper, renames the four misleading ``*_foot`` wheel links/joints, and
places the Z1 mounting plane 35 mm above the B2-W deck datum.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


UNITREE_ROS_COMMIT = "7d6075f7f58588b189b940130e3edab3c839b2df"
LEGS = ("FL", "FR", "RL", "RR")
WHEEL_LINK_RENAMES = {f"{leg}_foot": f"{leg}_wheel" for leg in LEGS}
WHEEL_JOINT_RENAMES = {f"{leg}_foot_joint": f"{leg}_wheel_joint" for leg in LEGS}
Z1_LINK_RENAMES = {f"link0{index}": f"link{index}" for index in range(7)}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unitree-ros-root",
        type=Path,
        required=True,
        help="Checkout of github.com/unitreerobotics/unitree_ros at the pinned commit.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Generated combined URDF path.")
    parser.add_argument(
        "--resolve-mesh-paths",
        action="store_true",
        help="Replace package:// mesh paths with absolute paths for Isaac Sim conversion.",
    )
    parser.add_argument("--mount-x", type=float, default=0.211)
    parser.add_argument("--mount-y", type=float, default=0.0)
    parser.add_argument("--deck-z", type=float, default=0.075)
    parser.add_argument("--adapter-height", type=float, default=0.035)
    parser.add_argument("--adapter-size-x", type=float, default=0.130)
    parser.add_argument("--adapter-size-y", type=float, default=0.120)
    return parser.parse_args()


def _validate_source_revision(unitree_root: Path) -> None:
    """Refuse to label output as official/pinned when the source is different."""
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(unitree_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("--unitree-ros-root must be a Git checkout of unitree_ros") from error
    if revision != UNITREE_ROS_COMMIT:
        raise ValueError(
            f"unitree_ros is at {revision}, but this generator requires {UNITREE_ROS_COMMIT}"
        )
    source_diff = subprocess.run(
        [
            "git",
            "-C",
            str(unitree_root),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "robots/b2w_description",
            "robots/z1_description",
        ],
        check=False,
    )
    if source_diff.returncode != 0:
        raise ValueError("Official B2-W/Z1 source files contain local modifications")


def _rename_b2w_wheels(robot: ET.Element) -> None:
    for link in robot.findall("link"):
        if link.get("name") in WHEEL_LINK_RENAMES:
            link.set("name", WHEEL_LINK_RENAMES[link.get("name")])
    for joint in robot.findall("joint"):
        if joint.get("name") in WHEEL_JOINT_RENAMES:
            joint.set("name", WHEEL_JOINT_RENAMES[joint.get("name")])
        for relation in (joint.find("parent"), joint.find("child")):
            if relation is not None and relation.get("link") in WHEEL_LINK_RENAMES:
                relation.set("link", WHEEL_LINK_RENAMES[relation.get("link")])


def _add_adapter_geometry(base_link: ET.Element, args: argparse.Namespace) -> None:
    center_z = args.deck_z + 0.5 * args.adapter_height
    size = f"{args.adapter_size_x:.6f} {args.adapter_size_y:.6f} {args.adapter_height:.6f}"
    origin = f"{args.mount_x:.6f} {args.mount_y:.6f} {center_z:.6f}"

    visual = ET.SubElement(base_link, "visual", {"name": "z1_adapter_visual"})
    ET.SubElement(visual, "origin", {"xyz": origin, "rpy": "0 0 0"})
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "box", {"size": size})
    material = ET.SubElement(visual, "material", {"name": "z1_adapter_aluminum"})
    ET.SubElement(material, "color", {"rgba": "0.48 0.50 0.52 1"})

    collision = ET.SubElement(base_link, "collision", {"name": "z1_adapter_collision"})
    ET.SubElement(collision, "origin", {"xyz": origin, "rpy": "0 0 0"})
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", {"size": size})


def _ensure_converter_visuals(robot: ET.Element) -> None:
    """Give frame-only links invisible visuals to avoid dangling converter references.

    Isaac Sim 5.1's URDF converter authors a reference to a visual prim even
    for links that have no visual element. A microscopic transparent sphere
    preserves the official fixed frames while making each reference valid.
    """
    for link in robot.findall("link"):
        if link.find("visual") is not None:
            continue
        visual = ET.SubElement(link, "visual", {"name": "converter_placeholder_visual"})
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "sphere", {"radius": "0.000001"})
        material = ET.SubElement(visual, "material", {"name": "converter_placeholder_transparent"})
        ET.SubElement(material, "color", {"rgba": "0 0 0 0"})


def _append_z1(robot: ET.Element, z1_robot: ET.Element, args: argparse.Namespace) -> None:
    ignored_tags = {"gazebo", "transmission"}
    for element in z1_robot:
        if element.tag in ignored_tags:
            continue
        if element.tag == "link" and element.get("name") == "world":
            continue
        if element.tag == "joint" and element.get("name") == "base_static_joint":
            continue

        element = copy.deepcopy(element)
        if element.tag == "link" and element.get("name") in Z1_LINK_RENAMES:
            element.set("name", Z1_LINK_RENAMES[element.get("name")])
        if element.tag == "joint":
            for relation in (element.find("parent"), element.find("child")):
                if relation is not None and relation.get("link") in Z1_LINK_RENAMES:
                    relation.set("link", Z1_LINK_RENAMES[relation.get("link")])
        robot.append(element)

    mount_joint = ET.Element("joint", {"name": "z1_mount_joint", "type": "fixed"})
    mount_z = args.deck_z + args.adapter_height
    ET.SubElement(
        mount_joint,
        "origin",
        {
            "xyz": f"{args.mount_x:.6f} {args.mount_y:.6f} {mount_z:.6f}",
            "rpy": "0 0 0",
        },
    )
    ET.SubElement(mount_joint, "parent", {"link": "base_link"})
    ET.SubElement(mount_joint, "child", {"link": "link0"})
    robot.append(mount_joint)


def _append_gripper(robot: ET.Element) -> None:
    stator_joint = ET.fromstring(
        """
        <joint name="gripper_stator_joint" type="fixed">
          <origin rpy="0 0 0" xyz="0.051 0 0"/>
          <parent link="link6"/>
          <child link="gripper_stator"/>
        </joint>
        """
    )
    stator_link = ET.fromstring(
        """
        <link name="gripper_stator">
          <visual>
            <geometry><mesh filename="package://z1_description/meshes/visual/z1_GripperStator.dae"/></geometry>
          </visual>
          <collision>
            <geometry><mesh filename="package://z1_description/meshes/collision/z1_GripperStator.STL"/></geometry>
          </collision>
          <inertial>
            <origin rpy="0 0 0" xyz="0.04764427 -0.00035819 -0.00249162"/>
            <mass value="0.52603655"/>
            <inertia ixx="0.00038683" ixy="-0.00000359" ixz="0.00007662"
                     iyy="0.00068614" iyz="0.00000209" izz="0.00066293"/>
          </inertial>
        </link>
        """
    )
    mover_joint = ET.fromstring(
        """
        <joint name="gripper_joint" type="revolute">
          <origin rpy="0 0 0" xyz="0.049 0 0"/>
          <parent link="gripper_stator"/>
          <child link="gripper_mover"/>
          <axis xyz="0 1 0"/>
          <dynamics damping="1.0" friction="1.0"/>
          <limit effort="30.0" velocity="3.1415" lower="-1.570796327" upper="0.0"/>
        </joint>
        """
    )
    mover_link = ET.fromstring(
        """
        <link name="gripper_mover">
          <visual>
            <geometry><mesh filename="package://z1_description/meshes/visual/z1_GripperMover.dae"/></geometry>
          </visual>
          <collision>
            <geometry><mesh filename="package://z1_description/meshes/collision/z1_GripperMover.STL"/></geometry>
          </collision>
          <inertial>
            <origin rpy="0 0 0" xyz="0.01320633 0.00476708 0.00380534"/>
            <mass value="0.27621302"/>
            <inertia ixx="0.00017716" ixy="0.00001683" ixz="-0.00001786"
                     iyy="0.00026787" iyz="0.00000262" izz="0.00035728"/>
          </inertial>
        </link>
        """
    )
    robot.extend((stator_joint, stator_link, mover_joint, mover_link))


def _resolve_mesh_paths(robot: ET.Element, unitree_root: Path) -> None:
    packages = {
        "package://b2w_description/": unitree_root / "robots" / "b2w_description",
        "package://z1_description/": unitree_root / "robots" / "z1_description",
    }
    for mesh in robot.iter("mesh"):
        filename = mesh.get("filename", "")
        for prefix, package_root in packages.items():
            if filename.startswith(prefix):
                mesh.set("filename", str((package_root / filename.removeprefix(prefix)).resolve()))
                break


def _validate(robot: ET.Element, args: argparse.Namespace) -> None:
    link_elements = robot.findall("link")
    joint_elements = robot.findall("joint")
    links = [element.get("name") for element in link_elements]
    joints = [element.get("name") for element in joint_elements]
    if len(links) != len(set(links)):
        raise ValueError("Duplicate link names in combined URDF")
    if len(joints) != len(set(joints)):
        raise ValueError("Duplicate joint names in combined URDF")
    movable_joints = [joint for joint in joint_elements if joint.get("type") != "fixed"]
    if len(movable_joints) != 23:
        raise ValueError(f"Expected 23 movable joints, got {len(movable_joints)}")

    expected_wheels = {f"{leg}_wheel_joint" for leg in LEGS}
    if not expected_wheels.issubset(joints):
        raise ValueError(f"Missing wheel joints: {sorted(expected_wheels.difference(joints))}")
    for wheel_name in expected_wheels:
        wheel_joint = next(joint for joint in joint_elements if joint.get("name") == wheel_name)
        if wheel_joint.get("type") != "continuous" or wheel_joint.find("axis").get("xyz") != "0 1 0":
            raise ValueError(f"Unexpected wheel joint type or axis: {wheel_name}")
        wheel_limit = wheel_joint.find("limit")
        if float(wheel_limit.get("effort")) != 20.0 or float(wheel_limit.get("velocity")) != 50.0:
            raise ValueError(f"Unexpected wheel limits: {wheel_name}")
        wheel_link_name = wheel_name.removesuffix("_joint")
        wheel_link = next(link for link in link_elements if link.get("name") == wheel_link_name)
        if abs(float(wheel_link.find("inertial/mass").get("value")) - 1.083) > 1.0e-9:
            raise ValueError(f"Unexpected wheel mass: {wheel_link_name}")

    expected_arm_joints = {f"joint{index}" for index in range(1, 7)} | {"gripper_joint"}
    if not expected_arm_joints.issubset(joints):
        raise ValueError(f"Missing Z1 joints: {sorted(expected_arm_joints.difference(joints))}")

    mount = next(joint for joint in robot.findall("joint") if joint.get("name") == "z1_mount_joint")
    actual_mount = tuple(float(value) for value in mount.find("origin").get("xyz").split())
    expected_mount = (args.mount_x, args.mount_y, args.deck_z + args.adapter_height)
    if any(abs(actual - expected) > 1.0e-9 for actual, expected in zip(actual_mount, expected_mount)):
        raise ValueError(f"Unexpected Z1 mount position: {actual_mount} != {expected_mount}")

    base_link = next(link for link in link_elements if link.get("name") == "base_link")
    adapter_visual = next(
        visual for visual in base_link.findall("visual") if visual.get("name") == "z1_adapter_visual"
    )
    adapter_size = tuple(float(value) for value in adapter_visual.find("geometry/box").get("size").split())
    expected_size = (args.adapter_size_x, args.adapter_size_y, args.adapter_height)
    if any(abs(actual - expected) > 1.0e-9 for actual, expected in zip(adapter_size, expected_size)):
        raise ValueError(f"Unexpected adapter size: {adapter_size} != {expected_size}")
    if not any(collision.get("name") == "z1_adapter_collision" for collision in base_link.findall("collision")):
        raise ValueError("Missing adapter collision geometry")


def main() -> None:
    args = _arguments()
    unitree_root = args.unitree_ros_root.resolve()
    _validate_source_revision(unitree_root)
    b2w_path = unitree_root / "robots" / "b2w_description" / "urdf" / "b2w_description.urdf"
    z1_path = unitree_root / "robots" / "z1_description" / "xacro" / "z1.urdf"
    if not b2w_path.is_file() or not z1_path.is_file():
        raise FileNotFoundError("Expected official B2-W and Z1 URDF files below --unitree-ros-root")

    robot = ET.parse(b2w_path).getroot()
    robot.set("name", "b2w_z1")
    robot.insert(
        0,
        ET.Comment(
            " Generated from unitreerobotics/unitree_ros commit "
            f"{UNITREE_ROS_COMMIT}; adapter height is user-measured hardware data. "
        ),
    )
    _rename_b2w_wheels(robot)
    base_link = next(link for link in robot.findall("link") if link.get("name") == "base_link")
    _add_adapter_geometry(base_link, args)
    _append_z1(robot, ET.parse(z1_path).getroot(), args)
    _append_gripper(robot)
    _ensure_converter_visuals(robot)
    if args.resolve_mesh_paths:
        _resolve_mesh_paths(robot, unitree_root)
    _validate(robot, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(args.output, encoding="utf-8", xml_declaration=True)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
