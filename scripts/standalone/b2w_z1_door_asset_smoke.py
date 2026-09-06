"""Headless dynamics smoke test for the articulated lever-handle door."""

import argparse
import json

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from LeggedManip_Lab.assets.door import DOOR_WITH_LEVER_CFG


def step(sim: SimulationContext, door: Articulation, count: int) -> None:
    for _ in range(count):
        door.write_data_to_sim()
        sim.step()
        door.update(sim.get_physics_dt())


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    sim.set_camera_view((2.8, -2.6, 1.8), (1.35, 0.0, 1.0))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    door = Articulation(DOOR_WITH_LEVER_CFG.replace(prim_path="/World/Door"))
    sim.reset()
    door.reset()
    step(sim, door, 20)

    door_id = door.find_joints("door_hinge")[0][0]
    handle_id = door.find_joints("handle_joint")[0][0]
    grasp_id = door.find_bodies("handle_grasp")[0][0]
    initial_grasp = door.data.body_pos_w[0, grasp_id].clone()

    effort = torch.zeros((1, door.num_joints), device=door.device)
    effort[:, handle_id] = 6.0
    door.set_joint_effort_target(effort)
    step(sim, door, args.steps)
    handle_after_torque = float(door.data.joint_pos[0, handle_id].item())

    effort.zero_()
    effort[:, door_id] = 35.0
    door.set_joint_effort_target(effort)
    step(sim, door, args.steps)
    door_after_torque = float(door.data.joint_pos[0, door_id].item())
    grasp_motion = float(torch.linalg.vector_norm(door.data.body_pos_w[0, grasp_id] - initial_grasp).item())

    report = {
        "joint_names": door.joint_names,
        "body_names": door.body_names,
        "handle_angle_after_6Nm_rad": handle_after_torque,
        "door_angle_after_35Nm_rad": door_after_torque,
        "handle_grasp_motion_m": grasp_motion,
        "finite": bool(
            torch.isfinite(door.data.joint_pos).all()
            and torch.isfinite(door.data.joint_vel).all()
            and torch.isfinite(door.data.body_pos_w).all()
        ),
    }
    print("B2W_Z1_DOOR_ASSET_SMOKE=" + json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["finite"]:
        raise RuntimeError("Door simulation produced non-finite state")
    if handle_after_torque < 0.20:
        raise RuntimeError("Lever handle did not rotate under test torque")
    if door_after_torque < 0.10:
        raise RuntimeError("Door panel did not rotate under test torque")


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
