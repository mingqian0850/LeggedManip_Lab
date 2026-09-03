# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Isaac Lab articulation configuration for the wheeled B2-W + Z1 asset."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


_ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
B2W_Z1_GENERATED_USD = os.path.join(_ASSET_DIR, "b2w_z1.usd")
B2W_Z1_USD = os.path.join(_ASSET_DIR, "b2w_z1_stow.usda")

# Raised, forward-facing pose used while training TCP tracking.
B2W_Z1_ARM_HOME_JOINT_POS = {
    "joint1": 0.0,
    "joint2": 1.35,
    "joint3": -1.65,
    "joint4": 0.30,
    "joint5": 0.0,
    "joint6": 0.0,
}

# Hardware-like stacked Z-fold reconstructed from ``IMG_5219.jpg``. Unlike the
# arm's all-zero pose, this keeps the return link and gripper above the lidar.
# These values are a photo-matched starting point; replace them with encoder
# readback from the real robot when it is available.
B2W_Z1_ARM_STOW_JOINT_POS = {
    "joint1": 0.0,
    "joint2": 0.15,
    "joint3": -0.45,
    "joint4": 0.30,
    "joint5": 0.0,
    "joint6": 0.0,
}


B2W_Z1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=B2W_Z1_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # The wheel radius is larger than the fixed foot used by B2_Z1_CFG.
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            "FL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RL_hip_joint": 0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            ".*_wheel_joint": 0.0,
            # Safe hardware-like folded pose for display and generic resets.
            **B2W_Z1_ARM_STOW_JOINT_POS,
            "gripper_joint": -1.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            stiffness=250.0,
            damping=5.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        # Velocity targets should drive this group. The limits come from the
        # official B2-W description; damping is a provisional tracking gain.
        "wheels": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=20.0,
            effort_limit_sim=20.0,
            velocity_limit=50.0,
            velocity_limit_sim=50.0,
            stiffness=0.0,
            damping=1.2,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.0,
        ),
        "joint1": DelayedPDActuatorCfg(
            joint_names_expr=["joint1"],
            stiffness=50.0,
            damping=3.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "joint2": DelayedPDActuatorCfg(
            joint_names_expr=["joint2"],
            stiffness=50.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "joint3": DelayedPDActuatorCfg(
            joint_names_expr=["joint3"],
            stiffness=80.0,
            damping=3.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "joint4": DelayedPDActuatorCfg(
            joint_names_expr=["joint4"],
            stiffness=30.0,
            damping=3.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "joint5": DelayedPDActuatorCfg(
            joint_names_expr=["joint5"],
            stiffness=30.0,
            damping=2.5,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "joint6": DelayedPDActuatorCfg(
            joint_names_expr=["joint6"],
            stiffness=20.0,
            damping=1.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "gripper": DelayedPDActuatorCfg(
            joint_names_expr=["gripper_joint"],
            stiffness=20.0,
            damping=1.0,
            armature=0.001,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
    },
)


# TCP-training tasks should explicitly use this variant. It starts with the arm
# deployed, away from joint limits, while B2W_Z1_CFG retains the safe stow pose.
B2W_Z1_TRAINING_CFG = B2W_Z1_CFG.replace(
    init_state=B2W_Z1_CFG.init_state.replace(
        joint_pos={
            **B2W_Z1_CFG.init_state.joint_pos,
            **B2W_Z1_ARM_HOME_JOINT_POS,
        }
    )
)
