"""Articulated inward-pull door with a downward-rotating lever handle."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


_ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
DOOR_WITH_LEVER_URDF = os.path.join(_ASSET_DIR, "door_with_lever.urdf")


DOOR_WITH_LEVER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=DOOR_WITH_LEVER_URDF,
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=False,
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.35, -0.45, 0.0),
        joint_pos={"door_hinge": 0.0, "handle_joint": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "door": ImplicitActuatorCfg(
            joint_names_expr=["door_hinge"],
            effort_limit_sim=180.0,
            velocity_limit_sim=1.5,
            stiffness=0.0,
            damping=1.0,
            friction=0.2,
        ),
        "handle": ImplicitActuatorCfg(
            joint_names_expr=["handle_joint"],
            effort_limit_sim=25.0,
            velocity_limit_sim=2.5,
            # A real lever is spring-loaded.  Without this return stiffness,
            # gravity rotates the offset handle to its lower stop and lets the
            # policy bypass the unlatching task without making contact.
            stiffness=12.0,
            damping=0.50,
            friction=0.35,
        ),
    },
)
