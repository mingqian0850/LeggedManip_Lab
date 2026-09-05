"""Unified 22-action B2-W + Z1 world-frame end-effector tracking task."""

import copy
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from LeggedManip_Lab.assets.b2w_z1.b2w_z1_articulation_cfg import B2W_Z1_TRAINING_CFG
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab import mdp


LEG_JOINTS = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
WHEEL_JOINTS = ["FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint"]
CONTROLLED_JOINTS = LEG_JOINTS + ARM_JOINTS + WHEEL_JOINTS

TCP_CFG = SceneEntityCfg("robot", body_names="tcp_frame")
ARM_CFG = SceneEntityCfg("robot", joint_names=ARM_JOINTS, preserve_order=True)
LEG_ARM_CFG = SceneEntityCfg("robot", joint_names=LEG_JOINTS + ARM_JOINTS, preserve_order=True)
CONTROLLED_CFG = SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True)
WHEEL_BODY_CFG = SceneEntityCfg(
    "robot",
    body_names=["FR_wheel", "FL_wheel", "RR_wheel", "RL_wheel"],
    preserve_order=True,
)
WHEEL_CONTACT_CFG = SceneEntityCfg(
    "contact_forces",
    body_names=["FR_wheel", "FL_wheel", "RR_wheel", "RL_wheel"],
    preserve_order=True,
)
UNDESIRED_CONTACT_CFG = SceneEntityCfg(
    "contact_forces",
    body_names=[
        "base_link",
        ".*_calf",
        "lidar_link",
        "link[0-6]",
        "gripper_.*",
    ],
)


def _make_training_asset_cfg():
    """Build an isolated stage-1 asset preset without mutating shared configs."""
    # Stage 1 learns a unified policy from a stable deterministic PD plant.
    # The RobotLab actuator preset is coupled to its active locomotion policy:
    # direct zero residuals repeatedly cross the low-base termination gate.
    # We therefore defer that harder actuator-transfer stage until nominal
    # tracking is learned, while still using the same B2-W + Z1 geometry.
    asset_cfg = copy.deepcopy(B2W_Z1_TRAINING_CFG)
    asset_cfg.spawn.articulation_props.enabled_self_collisions = True
    # Start with a deterministic plant.  Delay and dynamics randomization are
    # introduced only after the nominal task learns and passes the smoke gates.
    for actuator in asset_cfg.actuators.values():
        if hasattr(actuator, "min_delay"):
            actuator.min_delay = 0
            actuator.max_delay = 0
    return asset_cfg


@configclass
class B2WZ1EEWBCSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot = _make_training_asset_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class B2WZ1EEWBCCommandsCfg:
    tcp_pose = mdp.FKReachableWorldPoseCommandCfg(
        asset_name="robot",
        body_name="tcp_frame",
        resampling_time_range=(1.0e9, 1.0e9),
        # Stage 1 deliberately starts inside a broad arm capture basin.  The
        # ghost-root construction still provides a whole-body solution and the
        # terminal arm-home reward makes wheel motion useful.
        radius_range=(0.03, 0.12),
        bearing_range=(-math.pi, math.pi),
        yaw_range=(-0.15, 0.15),
        stationary_probability=0.15,
        settle_time_s=2.0,
        motion_time_s=2.5,
        recapture_during_settle=True,
        debug_vis=False,
    )


@configclass
class B2WZ1EEWBCActionsCfg:
    """One actor action vector: 12 leg positions + 6 arm positions + 4 wheel speeds."""

    leg_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.35,
        use_default_offset=True,
        preserve_order=True,
    )
    arm_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        scale={"joint[1-3]": 0.40, "joint[4-6]": 0.50},
        use_default_offset=True,
        preserve_order=True,
    )
    wheel_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINTS,
        scale=12.0,
        use_default_offset=True,
        preserve_order=True,
    )
    gripper_hold = mdp.FixedJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper_joint"],
    )


@configclass
class B2WZ1EEWBCPolicyObsCfg(ObsGroup):
    """Deployable observations; two frames provide velocity-free short memory."""

    root_linear_velocity = ObsTerm(func=mdp.root_linear_velocity_b, scale=2.0)
    root_angular_velocity = ObsTerm(func=mdp.root_angular_velocity_b, scale=0.25)
    projected_gravity = ObsTerm(func=mdp.root_projected_gravity_b)
    leg_arm_joint_position = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": LEG_ARM_CFG})
    controlled_joint_velocity = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": CONTROLLED_CFG, "joint_names": CONTROLLED_JOINTS},
        scale=0.05,
    )
    tcp_reference_position_error = ObsTerm(
        func=mdp.tcp_reference_position_error_b,
        params={"command_name": "tcp_pose", "asset_cfg": TCP_CFG},
    )
    tcp_reference_orientation_error = ObsTerm(
        func=mdp.tcp_reference_orientation_error_b,
        params={"command_name": "tcp_pose", "asset_cfg": TCP_CFG},
    )
    tcp_final_position_error = ObsTerm(
        func=mdp.tcp_final_position_error_b,
        params={"command_name": "tcp_pose", "asset_cfg": TCP_CFG},
    )
    tcp_final_orientation_error = ObsTerm(
        func=mdp.tcp_final_orientation_error_b,
        params={"command_name": "tcp_pose", "asset_cfg": TCP_CFG},
    )
    desired_tcp_twist = ObsTerm(
        func=mdp.desired_tcp_twist_b,
        params={"command_name": "tcp_pose"},
    )
    future_tcp_pose_errors = ObsTerm(
        func=mdp.future_tcp_pose_errors_b,
        params={
            "command_name": "tcp_pose",
            "offsets_s": (0.10, 0.25, 0.50),
            "asset_cfg": TCP_CFG,
        },
    )
    reference_progress = ObsTerm(
        func=mdp.reference_motion_progress,
        params={"command_name": "tcp_pose"},
    )
    previous_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 2
        self.flatten_history_dim = True


@configclass
class B2WZ1EEWBCPrivilegedObsCfg(ObsGroup):
    """Simulation-only critic signals; none are required by the deployed actor."""

    ghost_root_planar_error = ObsTerm(
        func=mdp.ghost_root_planar_error_b,
        params={"command_name": "tcp_pose"},
    )
    controlled_joint_effort = ObsTerm(func=mdp.joint_effort, params={"asset_cfg": CONTROLLED_CFG}, scale=0.02)
    wheel_contact_count = ObsTerm(
        func=mdp.wheel_contact_count,
        params={"sensor_cfg": WHEEL_CONTACT_CFG},
        scale=0.25,
    )
    tcp_goal_distance = ObsTerm(
        func=mdp.tcp_goal_distance,
        params={"command_name": "tcp_pose", "asset_cfg": TCP_CFG},
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class B2WZ1EEWBCObservationsCfg:
    policy: B2WZ1EEWBCPolicyObsCfg = B2WZ1EEWBCPolicyObsCfg()
    privileged: B2WZ1EEWBCPrivilegedObsCfg = B2WZ1EEWBCPrivilegedObsCfg()


@configclass
class B2WZ1EEWBCRewardsCfg:
    tcp_position_coarse = RewTerm(
        func=mdp.tcp_reference_position_tracking_exp,
        weight=2.0,
        params={"command_name": "tcp_pose", "std": 0.20, "asset_cfg": TCP_CFG},
    )
    tcp_position_fine = RewTerm(
        func=mdp.tcp_reference_position_tracking_exp,
        weight=5.0,
        params={"command_name": "tcp_pose", "std": 0.035, "asset_cfg": TCP_CFG},
    )
    tcp_orientation_coarse = RewTerm(
        func=mdp.tcp_reference_orientation_tracking_exp,
        weight=0.75,
        params={"command_name": "tcp_pose", "std": 0.35, "asset_cfg": TCP_CFG},
    )
    tcp_orientation_fine = RewTerm(
        func=mdp.tcp_reference_orientation_tracking_exp,
        weight=1.5,
        params={"command_name": "tcp_pose", "std": 0.10, "asset_cfg": TCP_CFG},
    )
    tcp_twist = RewTerm(
        func=mdp.tcp_twist_tracking_exp,
        weight=0.75,
        params={
            "command_name": "tcp_pose",
            "linear_std": 0.18,
            "angular_std": 0.6,
            "asset_cfg": TCP_CFG,
        },
    )
    final_success = RewTerm(
        func=mdp.final_tcp_success,
        weight=3.0,
        params={
            "command_name": "tcp_pose",
            "position_threshold": 0.03,
            "orientation_threshold": math.radians(6.0),
            "asset_cfg": TCP_CFG,
        },
    )
    terminal_arm_home = RewTerm(
        func=mdp.terminal_arm_home_deviation_l2,
        weight=-0.6,
        params={"command_name": "tcp_pose", "start_progress": 0.75, "asset_cfg": ARM_CFG},
    )
    arm_joint_margin = RewTerm(
        func=mdp.arm_joint_margin_barrier,
        weight=-0.5,
        params={"margin_threshold": 0.12, "asset_cfg": ARM_CFG},
    )
    base_height = RewTerm(func=mdp.base_height_error_l2, weight=-3.0, params={"target_height": 0.505})
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": LEG_ARM_CFG},
    )
    joint_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": CONTROLLED_CFG},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    leg_action = RewTerm(func=mdp.action_term_l2, weight=-0.002, params={"action_name": "leg_position"})
    arm_action = RewTerm(func=mdp.action_term_l2, weight=-0.001, params={"action_name": "arm_position"})
    wheel_action = RewTerm(
        func=mdp.action_term_l2,
        weight=-0.001,
        params={"action_name": "wheel_velocity"},
    )
    wheel_lateral_slip = RewTerm(
        func=mdp.wheel_lateral_slip_l2,
        weight=-0.10,
        params={"sensor_cfg": WHEEL_CONTACT_CFG, "asset_cfg": WHEEL_BODY_CFG},
    )
    settled_velocity = RewTerm(
        func=mdp.settled_velocity_l2,
        weight=-0.25,
        params={
            "command_name": "tcp_pose",
            "position_threshold": 0.05,
            "orientation_threshold": math.radians(10.0),
            "asset_cfg": TCP_CFG,
        },
    )
    termination = RewTerm(func=mdp.is_terminated, weight=-8.0)


@configclass
class B2WZ1EEWBCEventCfg:
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class B2WZ1EEWBCTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    low_base = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.45})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(35.0)})
    undesired_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": UNDESIRED_CONTACT_CFG, "threshold": 5.0},
    )


@configclass
class B2WZ1EEWBCCurriculumCfg:
    """Stage 1 is fixed; later stages widen commands and add randomization."""


@configclass
class B2WZ1EEWBCEnvCfg(ManagerBasedRLEnvCfg):
    scene: B2WZ1EEWBCSceneCfg = B2WZ1EEWBCSceneCfg(num_envs=1024, env_spacing=3.0)
    observations: B2WZ1EEWBCObservationsCfg = B2WZ1EEWBCObservationsCfg()
    actions: B2WZ1EEWBCActionsCfg = B2WZ1EEWBCActionsCfg()
    commands: B2WZ1EEWBCCommandsCfg = B2WZ1EEWBCCommandsCfg()
    rewards: B2WZ1EEWBCRewardsCfg = B2WZ1EEWBCRewardsCfg()
    terminations: B2WZ1EEWBCTerminationsCfg = B2WZ1EEWBCTerminationsCfg()
    events: B2WZ1EEWBCEventCfg = B2WZ1EEWBCEventCfg()
    curriculum: B2WZ1EEWBCCurriculumCfg = B2WZ1EEWBCCurriculumCfg()

    def __post_init__(self) -> None:
        self.sim.dt = 0.005
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class B2WZ1EEWBCEnvCfg_PLAY(B2WZ1EEWBCEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.commands.tcp_pose.debug_vis = True
