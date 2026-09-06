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
from LeggedManip_Lab.assets.door import DOOR_WITH_LEVER_CFG
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
LEG_CFG = SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True)
HIP_CFG = SceneEntityCfg("robot", joint_names=".*_hip_joint")
GRIPPER_STATOR_CFG = SceneEntityCfg("robot", body_names="gripper_stator")
GRIPPER_MOVER_CFG = SceneEntityCfg("robot", body_names="gripper_mover")
LIDAR_CFG = SceneEntityCfg("robot", body_names="lidar_link")
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
DOOR_GRASP_UNDESIRED_CONTACT_CFG = SceneEntityCfg(
    "contact_forces",
    body_names=[
        "base_link",
        ".*_calf",
        "lidar_link",
        "link[0-6]",
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
class B2WZ1LegFilter50ActionsCfg(B2WZ1EEWBCActionsCfg):
    """Shape-compatible leg-target smoothing for zero-shot and fine-tune tests."""

    leg_position = mdp.LowPassJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.35,
        use_default_offset=True,
        preserve_order=True,
        alpha=0.50,
    )


@configclass
class B2WZ1LegFilter35ActionsCfg(B2WZ1EEWBCActionsCfg):
    """Stronger diagnostic filter used to measure the tracking/smoothness trade-off."""

    leg_position = mdp.LowPassJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.35,
        use_default_offset=True,
        preserve_order=True,
        alpha=0.35,
    )


@configclass
class B2WZ1MirroredLegFilter35ActionsCfg(B2WZ1EEWBCActionsCfg):
    """Filtered leg action with a payload-aware soft mirror projection."""

    leg_position = mdp.MirroredLowPassJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.35,
        use_default_offset=True,
        preserve_order=True,
        alpha=0.35,
        asymmetric_residual_scale=0.25,
    )


@configclass
class B2WZ1AdaptiveMirroredLegFilter35ActionsCfg(B2WZ1EEWBCActionsCfg):
    """Soft mirror projection that releases the legs for extreme-low targets."""

    leg_position = mdp.MirroredLowPassJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.35,
        use_default_offset=True,
        preserve_order=True,
        alpha=0.35,
        asymmetric_residual_scale=0.25,
        command_name="tcp_pose",
        mirrored_above_height_m=-0.08,
        full_residual_below_height_m=-0.14,
        mirrored_below_planar_radius_m=0.20,
        full_residual_above_planar_radius_m=0.40,
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
        params={
            "command_name": "tcp_pose",
            "asset_cfg": TCP_CFG,
            "hide_during_settle": True,
        },
    )
    tcp_final_orientation_error = ObsTerm(
        func=mdp.tcp_final_orientation_error_b,
        params={
            "command_name": "tcp_pose",
            "asset_cfg": TCP_CFG,
            "hide_during_settle": True,
        },
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
    # A normalized one-sided barrier protects the 0.45 m termination boundary
    # without punishing useful upward or tilting whole-body motion.  The former
    # unnormalized L2 term remained tiny compared with the tracking rewards.
    base_height = RewTerm(
        func=mdp.base_height_safety_barrier,
        weight=-4.0,
        params={"safe_height": 0.49, "minimum_height": 0.45},
    )
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    spatial_pitch = None
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
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-50.0,
        params={"sensor_cfg": UNDESIRED_CONTACT_CFG, "threshold": 1.0},
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
    # RewardManager integrates weights with the 0.02 s control step.  A -200
    # weight therefore contributes -4 at failure, which is large enough to be
    # visible against an approximately 100-return successful episode.
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)


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


@configclass
class B2WZ1EEWBCStage2EnvCfg(B2WZ1EEWBCEnvCfg):
    """Stage 2: longer planar goals with explicit Stage 1 replay."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.tcp_pose.radius_range = (0.12, 0.20)
        self.commands.tcp_pose.short_radius_range = (0.03, 0.12)
        self.commands.tcp_pose.short_radius_probability = 0.30


@configclass
class B2WZ1EEWBCStage2EnvCfg_PLAY(B2WZ1EEWBCStage2EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.commands.tcp_pose.debug_vis = True


@configclass
class B2WZ1EEWBCStage3EnvCfg(B2WZ1EEWBCEnvCfg):
    """Stage 3: 30 cm planar goals with replay of the full earlier workspace."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.tcp_pose.radius_range = (0.20, 0.30)
        self.commands.tcp_pose.short_radius_range = (0.03, 0.20)
        self.commands.tcp_pose.short_radius_probability = 0.30


@configclass
class B2WZ1EEWBCStage3EnvCfg_PLAY(B2WZ1EEWBCStage3EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.commands.tcp_pose.debug_vis = True


@configclass
class B2WZ1EEWBCStage4EnvCfg(B2WZ1EEWBCEnvCfg):
    """Stage 4: 50 cm planar goals with replay of the full earlier workspace."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.tcp_pose.radius_range = (0.30, 0.50)
        self.commands.tcp_pose.short_radius_range = (0.03, 0.30)
        self.commands.tcp_pose.short_radius_probability = 0.30


@configclass
class B2WZ1EEWBCStage4EnvCfg_PLAY(B2WZ1EEWBCStage4EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.commands.tcp_pose.debug_vis = True


@configclass
class B2WZ1EEWBCStage5EnvCfg(B2WZ1EEWBCEnvCfg):
    """Stage 5: 70 cm planar goals that require coordinated base travel."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.tcp_pose.radius_range = (0.50, 0.70)
        self.commands.tcp_pose.short_radius_range = (0.03, 0.50)
        self.commands.tcp_pose.short_radius_probability = 0.30


@configclass
class B2WZ1EEWBCStage5EnvCfg_PLAY(B2WZ1EEWBCStage5EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.commands.tcp_pose.debug_vis = True


@configclass
class B2WZ1EEWBCStage6EnvCfg(B2WZ1EEWBCStage5EnvCfg):
    """Stage 6: introduce low TCP goals while replaying the 70 cm planar task."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.tcp_pose.height_offset_range = (-0.015, -0.005)
        self.commands.tcp_pose.pitch_range = (0.0, 0.0)
        self.commands.tcp_pose.spatial_probability = 0.20
        self.rewards.base_height.weight = -12.0
        self.rewards.flat_orientation.weight = -0.20


@configclass
class B2WZ1EEWBCStage6EnvCfg_PLAY(B2WZ1EEWBCStage6EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.commands.tcp_pose.debug_vis = True


@configclass
class B2WZ1EEWBCStage7EnvCfg(B2WZ1EEWBCStage6EnvCfg):
    """Stage 7: expand low TCP goals to four centimetres."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.tcp_pose.height_offset_range = (-0.04, -0.015)
        self.commands.tcp_pose.spatial_probability = 0.25


@configclass
class B2WZ1EEWBCStage7EnvCfg_PLAY(B2WZ1EEWBCStage7EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.commands.tcp_pose.debug_vis = True


@configclass
class B2WZ1EEWBCStage8EnvCfg(B2WZ1EEWBCStage7EnvCfg):
    """Stage 8: expand low TCP goals to eight centimetres."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.tcp_pose.height_offset_range = (-0.08, -0.04)
        self.commands.tcp_pose.spatial_probability = 0.30
        self.commands.tcp_pose.spatial_bearing_range = (-0.60, 0.60)
        self.rewards.spatial_pitch = RewTerm(
            func=mdp.spatial_base_pitch_tracking_exp,
            weight=1.0,
            params={
                "command_name": "tcp_pose",
                "maximum_height_offset": 0.08,
                "maximum_pitch": math.radians(10.0),
                "std": math.radians(5.0),
            },
        )


@configclass
class B2WZ1EEWBCStage8EnvCfg_PLAY(B2WZ1EEWBCStage8EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.commands.tcp_pose.debug_vis = True


@configclass
class B2WZ1EEWBCStage9EnvCfg(B2WZ1EEWBCStage8EnvCfg):
    """Stage 9: extend front-sector low TCP goals to twelve centimetres."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.tcp_pose.height_offset_range = (-0.12, -0.08)
        self.commands.tcp_pose.spatial_probability = 0.35
        self.rewards.spatial_pitch.params["maximum_height_offset"] = 0.12
        self.rewards.spatial_pitch.params["maximum_pitch"] = math.radians(14.0)


@configclass
class B2WZ1EEWBCStage9EnvCfg_PLAY(B2WZ1EEWBCStage9EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.commands.tcp_pose.debug_vis = True


@configclass
class B2WZ1EEWBCStage10SafetyRewardsCfg(B2WZ1EEWBCRewardsCfg):
    """Posture and calibrated clearance terms for the extreme-low curriculum."""

    settling_leg_posture = RewTerm(
        func=mdp.settling_joint_default_deviation_l2,
        weight=-8.0,
        params={"command_name": "tcp_pose", "asset_cfg": LEG_CFG},
    )
    hip_posture = RewTerm(
        func=mdp.joint_default_deviation_l2,
        weight=-1.0,
        params={"asset_cfg": HIP_CFG},
    )
    leg_left_right_symmetry = RewTerm(
        func=mdp.leg_left_right_symmetry_l2,
        weight=-0.5,
        params={"asset_cfg": LEG_CFG},
    )
    root_roll = RewTerm(func=mdp.root_roll_l2, weight=-20.0)
    root_roll_rate = RewTerm(func=mdp.root_roll_rate_l2, weight=-1.0)
    gripper_stator_lidar_clearance = RewTerm(
        func=mdp.body_pair_center_distance_barrier,
        weight=-20.0,
        params={
            "first_asset_cfg": GRIPPER_STATOR_CFG,
            "second_asset_cfg": LIDAR_CFG,
            "safe_distance": 0.24,
            "collision_distance": 0.17,
        },
    )
    gripper_mover_lidar_clearance = RewTerm(
        func=mdp.body_pair_center_distance_barrier,
        weight=-10.0,
        params={
            "first_asset_cfg": GRIPPER_MOVER_CFG,
            "second_asset_cfg": LIDAR_CFG,
            "safe_distance": 0.20,
            "collision_distance": 0.13,
        },
    )


@configclass
class B2WZ1EEWBCStage10EnvCfg(B2WZ1EEWBCStage9EnvCfg):
    """Stage 10: extend front-sector low TCP goals to eighteen centimetres."""

    rewards: B2WZ1EEWBCStage10SafetyRewardsCfg = B2WZ1EEWBCStage10SafetyRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.tcp_pose.height_offset_range = (-0.18, -0.12)
        self.commands.tcp_pose.spatial_probability = 0.40
        self.rewards.spatial_pitch.params["maximum_height_offset"] = 0.18
        self.rewards.spatial_pitch.params["maximum_pitch"] = math.radians(18.0)


@configclass
class B2WZ1EEWBCStage10EnvCfg_PLAY(B2WZ1EEWBCStage10EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.commands.tcp_pose.debug_vis = True


@configclass
class B2WZ1DynamicTrackingCommandsCfg:
    """Continuous trajectories used to measure latency, jitter, and posture."""

    tcp_pose = mdp.PeriodicWorldPoseCommandCfg(
        asset_name="robot",
        body_name="tcp_frame",
        resampling_time_range=(1.0e9, 1.0e9),
        trajectory_types=(
            "hold",
            "line_x",
            "line_y",
            "circle_xy",
            "figure8_xy",
            "vertical",
            "yaw_scan",
            "six_d",
        ),
        frequency_range_hz=(0.10, 0.30),
        center_offset_b=(0.25, 0.0, 0.0),
        line_amplitude_m=0.10,
        circle_radius_m=0.08,
        figure8_amplitude_m=(0.12, 0.06),
        vertical_amplitude_m=0.08,
        orientation_amplitude_rpy=(math.radians(7.0), math.radians(7.0), math.radians(15.0)),
        stationary_probability=0.0,
        settle_time_s=2.0,
        motion_time_s=2.5,
        ramp_time_s=2.5,
        recapture_during_settle=True,
        debug_vis=False,
    )


@configclass
class B2WZ1DynamicTrainingCommandsCfg:
    """Mixed dynamic and wide-workspace replay distribution."""

    tcp_pose = mdp.PeriodicWorldPoseCommandCfg(
        asset_name="robot",
        body_name="tcp_frame",
        resampling_time_range=(1.0e9, 1.0e9),
        trajectory_types=(
            "waypoint_planar",
            "waypoint_low",
            "hold",
            "line_x",
            "line_y",
            "circle_xy",
            "figure8_xy",
            "vertical",
            "yaw_scan",
            "six_d",
        ),
        trajectory_weights=(0.35, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05),
        frequency_range_hz=(0.08, 0.35),
        center_offset_b=(0.25, 0.0, 0.0),
        radius_range=(0.30, 0.70),
        short_radius_range=(0.03, 0.30),
        short_radius_probability=0.30,
        bearing_range=(-math.pi, math.pi),
        spatial_bearing_range=(-0.60, 0.60),
        height_offset_range=(-0.18, -0.005),
        yaw_range=(-0.25, 0.25),
        line_amplitude_m=0.10,
        circle_radius_m=0.08,
        figure8_amplitude_m=(0.12, 0.06),
        vertical_amplitude_m=0.08,
        orientation_amplitude_rpy=(math.radians(7.0), math.radians(7.0), math.radians(15.0)),
        stationary_probability=0.0,
        settle_time_s=2.0,
        motion_time_s=2.5,
        ramp_time_s=2.5,
        recapture_during_settle=True,
        debug_vis=False,
    )


@configclass
class B2WZ1WorkspaceGridCommandsCfg:
    """Parallel Stage-20 targets with three deterministic magnitude variants."""

    tcp_pose = mdp.WorkspaceSweepWorldPoseCommandCfg(
        asset_name="robot",
        body_name="tcp_frame",
        resampling_time_range=(1.0e9, 1.0e9),
        settle_time_s=2.0,
        motion_time_s=2.5,
        transition_time_s=2.5,
        hold_time_s=2.5,
        cycle_targets=False,
        recapture_during_settle=True,
        debug_vis=False,
    )


@configclass
class B2WZ1WorkspaceSweepCommandsCfg:
    """Sequential Stage-20 target sequence used for the review recording."""

    tcp_pose = mdp.WorkspaceSweepWorldPoseCommandCfg(
        asset_name="robot",
        body_name="tcp_frame",
        resampling_time_range=(1.0e9, 1.0e9),
        settle_time_s=2.0,
        motion_time_s=2.5,
        transition_time_s=2.5,
        hold_time_s=1.5,
        cycle_targets=True,
        recapture_during_settle=True,
        debug_vis=False,
    )


WIDE_WORKSPACE_TARGET_NAMES = (
    "home",
    "front_1m",
    "rear_0p8m",
    "high_0p25m",
    "low_0p25m",
    "diagonal_right",
    "large_6d_pose",
    "diagonal_left",
    "left_0p75m",
    "right_0p75m",
    "return_home",
)
WIDE_WORKSPACE_TARGET_OFFSETS_B = (
    (0.00, 0.00, 0.00),
    (1.00, 0.00, 0.00),
    (-0.80, 0.00, 0.00),
    (0.35, 0.00, 0.25),
    (0.45, 0.00, -0.25),
    (0.70, -0.70, 0.00),
    (0.45, 0.20, 0.12),
    (0.70, 0.70, 0.00),
    (0.15, 0.75, 0.00),
    (0.15, -0.75, 0.00),
    (0.00, 0.00, 0.00),
)
WIDE_WORKSPACE_TARGET_RPY = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, math.radians(-20.0), 0.0),
    (0.0, math.radians(20.0), 0.0),
    (0.0, 0.0, math.radians(-20.0)),
    (math.radians(25.0), math.radians(-20.0), math.radians(45.0)),
    (0.0, 0.0, math.radians(20.0)),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)

# The review sequence inserts explicit home waypoints between unrelated hard
# targets.  This prevents an unsafe cross-workspace chord from invalidating all
# later video labels while still leaving the known right-side failure last.
WIDE_SWEEP_TARGET_NAMES = (
    "home",
    "front_1m",
    "rear_0p8m",
    "high_0p25m",
    "low_0p25m",
    "diagonal_right",
    "mid_home_1",
    "large_6d_pose",
    "mid_home_2",
    "diagonal_left",
    "left_0p75m",
    "mid_home_3",
    "right_0p75m",
    "return_home",
)
WIDE_SWEEP_TARGET_OFFSETS_B = (
    (0.00, 0.00, 0.00),
    (1.00, 0.00, 0.00),
    (-0.80, 0.00, 0.00),
    (0.35, 0.00, 0.25),
    (0.45, 0.00, -0.25),
    (0.70, -0.70, 0.00),
    (0.00, 0.00, 0.00),
    (0.45, 0.20, 0.12),
    (0.00, 0.00, 0.00),
    (0.70, 0.70, 0.00),
    (0.15, 0.75, 0.00),
    (0.00, 0.00, 0.00),
    (0.15, -0.75, 0.00),
    (0.00, 0.00, 0.00),
)
WIDE_SWEEP_TARGET_RPY = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, math.radians(-20.0), 0.0),
    (0.0, math.radians(20.0), 0.0),
    (0.0, 0.0, math.radians(-20.0)),
    (0.0, 0.0, 0.0),
    (math.radians(25.0), math.radians(-20.0), math.radians(45.0)),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, math.radians(20.0)),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)


@configclass
class B2WZ1WideWorkspaceGridCommandsCfg:
    """Stage-20b stress grid extending well beyond the 70 cm training radius."""

    tcp_pose = mdp.WorkspaceSweepWorldPoseCommandCfg(
        asset_name="robot",
        body_name="tcp_frame",
        resampling_time_range=(1.0e9, 1.0e9),
        settle_time_s=2.0,
        motion_time_s=4.0,
        ramp_time_s=4.0,
        transition_time_s=4.0,
        hold_time_s=3.0,
        cycle_targets=False,
        target_names=WIDE_WORKSPACE_TARGET_NAMES,
        target_offsets_b=WIDE_WORKSPACE_TARGET_OFFSETS_B,
        target_rpy=WIDE_WORKSPACE_TARGET_RPY,
        target_scales=(0.75, 1.0, 1.25),
        recapture_during_settle=True,
        debug_vis=False,
    )


@configclass
class B2WZ1WideWorkspaceSweepCommandsCfg:
    """Single-robot wide sweep for visibly auditing whole-body locomotion."""

    tcp_pose = mdp.WorkspaceSweepWorldPoseCommandCfg(
        asset_name="robot",
        body_name="tcp_frame",
        resampling_time_range=(1.0e9, 1.0e9),
        settle_time_s=2.0,
        motion_time_s=4.0,
        ramp_time_s=4.0,
        transition_time_s=4.0,
        hold_time_s=1.5,
        cycle_targets=True,
        target_names=WIDE_SWEEP_TARGET_NAMES,
        target_offsets_b=WIDE_SWEEP_TARGET_OFFSETS_B,
        target_rpy=WIDE_SWEEP_TARGET_RPY,
        target_scales=(1.0,),
        recapture_during_settle=True,
        debug_vis=False,
    )


@configclass
class B2WZ1DynamicTrackingRewardsCfg(B2WZ1EEWBCRewardsCfg):
    """Tracking rewards plus explicit natural-posture objectives."""

    spatial_pitch = RewTerm(
        func=mdp.spatial_base_pitch_tracking_exp,
        weight=1.0,
        params={
            "command_name": "tcp_pose",
            "maximum_height_offset": 0.18,
            "maximum_pitch": math.radians(18.0),
            "std": math.radians(5.0),
        },
    )
    settling_leg_posture = RewTerm(
        func=mdp.settling_joint_default_deviation_l2,
        weight=-8.0,
        params={"command_name": "tcp_pose", "asset_cfg": LEG_CFG},
    )
    hip_posture = RewTerm(
        func=mdp.joint_default_deviation_l2,
        weight=-1.0,
        params={"asset_cfg": HIP_CFG},
    )
    leg_left_right_symmetry = RewTerm(
        func=mdp.leg_left_right_symmetry_l2,
        weight=-0.5,
        params={"asset_cfg": LEG_CFG},
    )
    root_roll = RewTerm(func=mdp.root_roll_l2, weight=-20.0)
    root_roll_rate = RewTerm(func=mdp.root_roll_rate_l2, weight=-1.0)
    gripper_stator_lidar_clearance = RewTerm(
        func=mdp.body_pair_center_distance_barrier,
        weight=-20.0,
        params={
            "first_asset_cfg": GRIPPER_STATOR_CFG,
            "second_asset_cfg": LIDAR_CFG,
            "safe_distance": 0.24,
            "collision_distance": 0.17,
        },
    )
    gripper_mover_lidar_clearance = RewTerm(
        func=mdp.body_pair_center_distance_barrier,
        weight=-10.0,
        params={
            "first_asset_cfg": GRIPPER_MOVER_CFG,
            "second_asset_cfg": LIDAR_CFG,
            "safe_distance": 0.20,
            "collision_distance": 0.13,
        },
    )


@configclass
class B2WZ1DynamicTrackingSmoothRewardsCfg(B2WZ1DynamicTrackingRewardsCfg):
    """Natural-posture rewards with a conservative leg acceleration cost.

    The coefficient matches Isaac Lab's standard locomotion regularizer.  It is
    intentionally small: the low-pass action term provides the main smoothing,
    while this term only discourages the residual high-frequency compensation
    observed after adapting the policy to the filtered actuator target.
    """

    leg_joint_acceleration = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": LEG_CFG},
    )


@configclass
class B2WZ1DynamicTrackingStanceRewardsCfg(B2WZ1DynamicTrackingRewardsCfg):
    """Stronger geometric stance prior for the continuously grounded B2W."""

    # Hip ab/adduction is the most visually obvious source of a splayed stance.
    # Pitching and lowering remain available through the thigh/calf joints.
    hip_posture = RewTerm(
        func=mdp.joint_default_deviation_l2,
        weight=-2.0,
        params={"asset_cfg": HIP_CFG},
    )
    # This is a soft mirror prior, not a hard constraint: asymmetric arm loads
    # can still produce the small leg differences needed for balance.
    leg_left_right_symmetry = RewTerm(
        func=mdp.leg_left_right_symmetry_l2,
        weight=-2.0,
        params={"asset_cfg": LEG_CFG},
    )


@configclass
class B2WZ1DynamicTrackingBalancedStanceRewardsCfg(B2WZ1DynamicTrackingRewardsCfg):
    """Moderate stance prior that leaves room for asymmetric payload support."""

    hip_posture = RewTerm(
        func=mdp.joint_default_deviation_l2,
        weight=-1.25,
        params={"asset_cfg": HIP_CFG},
    )
    leg_left_right_symmetry = RewTerm(
        func=mdp.leg_left_right_symmetry_l2,
        weight=-1.0,
        params={"asset_cfg": LEG_CFG},
    )


@configclass
class B2WZ1DynamicTrackingEnvCfg(B2WZ1EEWBCStage6EnvCfg):
    """Training distribution for continuous general-purpose 6D EE tracking."""

    commands: B2WZ1DynamicTrainingCommandsCfg = B2WZ1DynamicTrainingCommandsCfg()
    rewards: B2WZ1DynamicTrackingRewardsCfg = B2WZ1DynamicTrackingRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 24.0
        # A moving target must not be punished by the point-to-point
        # "stop once close" regularizer.  Keep a weak home bias so the actor
        # may still move the arm instead of delegating every task to the base.
        self.rewards.settled_velocity = None
        self.rewards.terminal_arm_home.weight = -0.15
        self.rewards.final_success.weight = 0.5


@configclass
class B2WZ1DynamicTrackingEnvCfg_PLAY(B2WZ1DynamicTrackingEnvCfg):
    commands: B2WZ1DynamicTrackingCommandsCfg = B2WZ1DynamicTrackingCommandsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.commands.tcp_pose.debug_vis = True


@configclass
class B2WZ1DynamicTrackingSlowEnvCfg_PLAY(B2WZ1DynamicTrackingEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.tcp_pose.frequency_range_hz = (0.10, 0.10)


@configclass
class B2WZ1DynamicTrackingMediumEnvCfg_PLAY(B2WZ1DynamicTrackingEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.tcp_pose.frequency_range_hz = (0.20, 0.20)


@configclass
class B2WZ1DynamicTrackingFilter50EnvCfg(B2WZ1DynamicTrackingEnvCfg):
    """Mixed training distribution with a 0.50 leg-target low-pass filter."""

    actions: B2WZ1LegFilter50ActionsCfg = B2WZ1LegFilter50ActionsCfg()


@configclass
class B2WZ1DynamicTrackingFilter35EnvCfg(B2WZ1DynamicTrackingEnvCfg):
    """Mixed training distribution with the selected 0.35 leg-target filter."""

    actions: B2WZ1LegFilter35ActionsCfg = B2WZ1LegFilter35ActionsCfg()


@configclass
class B2WZ1DynamicTrackingSmoothFilter35EnvCfg(B2WZ1DynamicTrackingFilter35EnvCfg):
    """Stage-18b fine-tuning task: filtered leg targets plus acceleration cost."""

    rewards: B2WZ1DynamicTrackingSmoothRewardsCfg = B2WZ1DynamicTrackingSmoothRewardsCfg()


@configclass
class B2WZ1DynamicTrackingStanceFilter35EnvCfg(B2WZ1DynamicTrackingFilter35EnvCfg):
    """Stage-18c fine-tuning task with a stronger natural-stance prior."""

    rewards: B2WZ1DynamicTrackingStanceRewardsCfg = B2WZ1DynamicTrackingStanceRewardsCfg()


@configclass
class B2WZ1DynamicTrackingBalancedStanceFilter35EnvCfg(B2WZ1DynamicTrackingFilter35EnvCfg):
    """Stage-18d short fine-tune with a payload-aware soft stance prior."""

    rewards: B2WZ1DynamicTrackingBalancedStanceRewardsCfg = B2WZ1DynamicTrackingBalancedStanceRewardsCfg()


@configclass
class B2WZ1DynamicTrackingMirroredFilter35EnvCfg(B2WZ1DynamicTrackingEnvCfg):
    """Stage-19 task with a structured, softly mirrored leg action space."""

    actions: B2WZ1MirroredLegFilter35ActionsCfg = B2WZ1MirroredLegFilter35ActionsCfg()


@configclass
class B2WZ1DynamicTrackingAdaptiveMirroredFilter35EnvCfg(B2WZ1DynamicTrackingEnvCfg):
    """Stage-19b task with height-adaptive structured leg actions."""

    actions: B2WZ1AdaptiveMirroredLegFilter35ActionsCfg = B2WZ1AdaptiveMirroredLegFilter35ActionsCfg()


@configclass
class B2WZ1WorkspaceGridAdaptiveMirroredFilter35EnvCfg_PLAY(B2WZ1DynamicTrackingEnvCfg):
    """Stage-20 batch audit: one independently evaluated workspace target per environment."""

    commands: B2WZ1WorkspaceGridCommandsCfg = B2WZ1WorkspaceGridCommandsCfg()
    actions: B2WZ1AdaptiveMirroredLegFilter35ActionsCfg = B2WZ1AdaptiveMirroredLegFilter35ActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 39
        self.scene.env_spacing = 3.0
        self.episode_length_s = 9.0
        self.commands.tcp_pose.debug_vis = True


@configclass
class B2WZ1WorkspaceSweepAdaptiveMirroredFilter35EnvCfg_PLAY(B2WZ1DynamicTrackingEnvCfg):
    """Stage-20 single-robot review sequence covering difficult EE poses."""

    commands: B2WZ1WorkspaceSweepCommandsCfg = B2WZ1WorkspaceSweepCommandsCfg()
    actions: B2WZ1AdaptiveMirroredLegFilter35ActionsCfg = B2WZ1AdaptiveMirroredLegFilter35ActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        target_count = len(self.commands.tcp_pose.target_names)
        segment_time = self.commands.tcp_pose.transition_time_s + self.commands.tcp_pose.hold_time_s
        self.episode_length_s = self.commands.tcp_pose.settle_time_s + target_count * segment_time + 2.0
        self.commands.tcp_pose.debug_vis = True
        self.viewer.eye = (2.8, 2.8, 1.9)
        self.viewer.lookat = (0.10, 0.0, 0.55)


@configclass
class B2WZ1WideWorkspaceGridAdaptiveMirroredFilter35EnvCfg_PLAY(B2WZ1DynamicTrackingEnvCfg):
    """Stage-20b parallel extrapolation audit with 0.75/1.0/1.25 target scales."""

    commands: B2WZ1WideWorkspaceGridCommandsCfg = B2WZ1WideWorkspaceGridCommandsCfg()
    actions: B2WZ1AdaptiveMirroredLegFilter35ActionsCfg = B2WZ1AdaptiveMirroredLegFilter35ActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 33
        self.scene.env_spacing = 4.0
        self.episode_length_s = 12.0
        self.commands.tcp_pose.debug_vis = True


@configclass
class B2WZ1WideWorkspaceSweepAdaptiveMirroredFilter35EnvCfg_PLAY(B2WZ1DynamicTrackingEnvCfg):
    """Stage-20b wide single-robot review sequence."""

    commands: B2WZ1WideWorkspaceSweepCommandsCfg = B2WZ1WideWorkspaceSweepCommandsCfg()
    actions: B2WZ1AdaptiveMirroredLegFilter35ActionsCfg = B2WZ1AdaptiveMirroredLegFilter35ActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        target_count = len(self.commands.tcp_pose.target_names)
        segment_time = self.commands.tcp_pose.transition_time_s + self.commands.tcp_pose.hold_time_s
        self.episode_length_s = self.commands.tcp_pose.settle_time_s + target_count * segment_time + 2.0
        self.commands.tcp_pose.debug_vis = True
        self.viewer.eye = (4.4, 4.4, 2.8)
        self.viewer.lookat = (0.05, 0.0, 0.55)


@configclass
class B2WZ1DynamicTrackingFilter50MediumEnvCfg_PLAY(B2WZ1DynamicTrackingMediumEnvCfg_PLAY):
    actions: B2WZ1LegFilter50ActionsCfg = B2WZ1LegFilter50ActionsCfg()


@configclass
class B2WZ1DynamicTrackingFilter35MediumEnvCfg_PLAY(B2WZ1DynamicTrackingMediumEnvCfg_PLAY):
    actions: B2WZ1LegFilter35ActionsCfg = B2WZ1LegFilter35ActionsCfg()


@configclass
class B2WZ1DynamicTrackingMirroredFilter35MediumEnvCfg_PLAY(B2WZ1DynamicTrackingMediumEnvCfg_PLAY):
    actions: B2WZ1MirroredLegFilter35ActionsCfg = B2WZ1MirroredLegFilter35ActionsCfg()


@configclass
class B2WZ1DynamicTrackingAdaptiveMirroredFilter35MediumEnvCfg_PLAY(B2WZ1DynamicTrackingMediumEnvCfg_PLAY):
    actions: B2WZ1AdaptiveMirroredLegFilter35ActionsCfg = B2WZ1AdaptiveMirroredLegFilter35ActionsCfg()


@configclass
class B2WZ1DynamicTrackingFilter35SlowEnvCfg_PLAY(B2WZ1DynamicTrackingSlowEnvCfg_PLAY):
    actions: B2WZ1LegFilter35ActionsCfg = B2WZ1LegFilter35ActionsCfg()


@configclass
class B2WZ1EEWBCStage5Filter35EnvCfg_PLAY(B2WZ1EEWBCStage5EnvCfg_PLAY):
    actions: B2WZ1LegFilter35ActionsCfg = B2WZ1LegFilter35ActionsCfg()


@configclass
class B2WZ1EEWBCStage6Filter35EnvCfg_PLAY(B2WZ1EEWBCStage6EnvCfg_PLAY):
    actions: B2WZ1LegFilter35ActionsCfg = B2WZ1LegFilter35ActionsCfg()


@configclass
class B2WZ1EEWBCStage10Filter35EnvCfg_PLAY(B2WZ1EEWBCStage10EnvCfg_PLAY):
    actions: B2WZ1LegFilter35ActionsCfg = B2WZ1LegFilter35ActionsCfg()


@configclass
class B2WZ1EEWBCStage10MirroredFilter35EnvCfg_PLAY(B2WZ1EEWBCStage10EnvCfg_PLAY):
    actions: B2WZ1MirroredLegFilter35ActionsCfg = B2WZ1MirroredLegFilter35ActionsCfg()


@configclass
class B2WZ1EEWBCStage10AdaptiveMirroredFilter35EnvCfg_PLAY(B2WZ1EEWBCStage10EnvCfg_PLAY):
    actions: B2WZ1AdaptiveMirroredLegFilter35ActionsCfg = B2WZ1AdaptiveMirroredLegFilter35ActionsCfg()


@configclass
class B2WZ1DynamicTrackingFastEnvCfg_PLAY(B2WZ1DynamicTrackingEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.tcp_pose.frequency_range_hz = (0.35, 0.35)


@configclass
class B2WZ1DynamicTrackingFilter35FastEnvCfg_PLAY(B2WZ1DynamicTrackingFastEnvCfg_PLAY):
    actions: B2WZ1LegFilter35ActionsCfg = B2WZ1LegFilter35ActionsCfg()


@configclass
class B2WZ1DynamicTrackingSixDEnvCfg_PLAY(B2WZ1DynamicTrackingMediumEnvCfg_PLAY):
    """Single-environment friendly demo with a visible continuous 6D target."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.tcp_pose.trajectory_types = ("six_d",)
        self.commands.tcp_pose.trajectory_weights = None
        # The default Isaac Lab camera is intended for large multi-environment
        # scenes and makes a single B2W almost invisible in recorded demos.
        self.viewer.eye = (2.6, 2.6, 1.8)
        self.viewer.lookat = (0.10, 0.0, 0.55)


@configclass
class B2WZ1DynamicTrackingFilter35SixDEnvCfg_PLAY(B2WZ1DynamicTrackingSixDEnvCfg_PLAY):
    """Six-dimensional demonstration using the selected filtered leg action."""

    actions: B2WZ1LegFilter35ActionsCfg = B2WZ1LegFilter35ActionsCfg()


@configclass
class B2WZ1DynamicTrackingMirroredFilter35SixDEnvCfg_PLAY(B2WZ1DynamicTrackingSixDEnvCfg_PLAY):
    """Six-dimensional demo using the structured mirrored leg action."""

    actions: B2WZ1MirroredLegFilter35ActionsCfg = B2WZ1MirroredLegFilter35ActionsCfg()


@configclass
class B2WZ1DynamicTrackingAdaptiveMirroredFilter35SixDEnvCfg_PLAY(B2WZ1DynamicTrackingSixDEnvCfg_PLAY):
    """Six-dimensional demo using reach-aware structured leg actions."""

    actions: B2WZ1AdaptiveMirroredLegFilter35ActionsCfg = B2WZ1AdaptiveMirroredLegFilter35ActionsCfg()


@configclass
class B2WZ1DynamicTrackingWaypointEnvCfg_PLAY(B2WZ1DynamicTrackingEnvCfg_PLAY):
    """Demo a target beyond arm-only reach so whole-body base motion is visible."""

    commands: B2WZ1DynamicTrainingCommandsCfg = B2WZ1DynamicTrainingCommandsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.tcp_pose.trajectory_types = ("waypoint_planar",)
        self.commands.tcp_pose.trajectory_weights = None
        self.commands.tcp_pose.radius_range = (0.55, 0.65)
        self.commands.tcp_pose.short_radius_probability = 0.0
        self.commands.tcp_pose.bearing_range = (-0.35, 0.35)


@configclass
class B2WZ1DoorAlignSceneCfg(B2WZ1EEWBCSceneCfg):
    """Free-standing articulated door added to every replicated scene."""

    door = DOOR_WITH_LEVER_CFG.replace(prim_path="{ENV_REGEX_NS}/Door")


@configclass
class B2WZ1DoorAlignCommandsCfg:
    """Stage 11: a collision-free target in front of the closed handle."""

    tcp_pose = mdp.DoorHandlePoseCommandCfg(
        asset_name="robot",
        body_name="tcp_frame",
        door_asset_name="door",
        handle_body_name="handle_grasp",
        resampling_time_range=(1.0e9, 1.0e9),
        approach_offset_handle=(-0.10, 0.0, 0.0),
        position_jitter_range=((-0.015, 0.015), (-0.025, 0.025), (-0.02, 0.02)),
        preserve_start_orientation=True,
        stationary_probability=0.0,
        settle_time_s=2.0,
        motion_time_s=3.5,
        recapture_during_settle=True,
        debug_vis=False,
    )


@configclass
class B2WZ1DoorAlignEnvCfg(B2WZ1EEWBCEnvCfg):
    """Stage 11: reuse the unified WBC to reach a door-handle pre-grasp pose."""

    scene: B2WZ1DoorAlignSceneCfg = B2WZ1DoorAlignSceneCfg(num_envs=1024, env_spacing=3.0)
    commands: B2WZ1DoorAlignCommandsCfg = B2WZ1DoorAlignCommandsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 12.0


@configclass
class B2WZ1DoorAlignEnvCfg_PLAY(B2WZ1DoorAlignEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0


@configclass
class B2WZ1DoorNearContactEnvCfg(B2WZ1DoorAlignEnvCfg):
    """Stage 12: approach to 3 cm while holding the gripper fully open."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.tcp_pose.approach_offset_handle = (-0.03, 0.0, 0.0)
        self.commands.tcp_pose.position_jitter_range = (
            (-0.005, 0.005),
            (-0.005, 0.005),
            (-0.005, 0.005),
        )
        # The Z1 URDF limit is [-pi/2, 0].  Negative rotates the movable jaw
        # away from the fixed jaw; zero closes it.
        self.actions.gripper_hold.target_position = -1.45


@configclass
class B2WZ1DoorNearContactEnvCfg_PLAY(B2WZ1DoorNearContactEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0


@configclass
class B2WZ1DoorGraspEnvCfg(B2WZ1DoorNearContactEnvCfg):
    """Stage 13: enter the grasp center, then close the Z1 gripper."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # The nominal TCP lies near the front face of the gripper.  Placing it
        # at the handle center creates a permanently infeasible 2 cm
        # penetration command; this contact-consistent offset still lets the
        # fingers wrap around the handle without the WBC pushing into the door.
        self.commands.tcp_pose.approach_offset_handle = (-0.02, 0.0, 0.0)
        self.commands.tcp_pose.position_jitter_range = (
            (0.0, 0.0),
            (-0.002, 0.002),
            (-0.002, 0.002),
        )
        self.actions.gripper_hold = mdp.PhasedJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper_joint"],
            command_name="tcp_pose",
            open_position=-1.45,
            # The 50 mm handle blocks the mover near -0.43 rad.  A zero-radian
            # target produced roughly 100 N contact and destabilized the base;
            # retain a small preload instead of commanding hard closure.
            closed_position=-0.35,
            close_start_progress=0.90,
            capture_distance=0.04,
            close_duration_s=1.5,
        )
        # Contact at the gripper is the objective in this phase.  Contacts on
        # the arm, lidar, base, and calves remain illegal.
        self.rewards.undesired_contacts.params["sensor_cfg"] = DOOR_GRASP_UNDESIRED_CONTACT_CFG
        self.terminations.undesired_contact.params["sensor_cfg"] = DOOR_GRASP_UNDESIRED_CONTACT_CFG


@configclass
class B2WZ1DoorGraspEnvCfg_PLAY(B2WZ1DoorGraspEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0


@configclass
class B2WZ1DoorTurnCommandsCfg:
    """Approach, close the gripper, then push the lever downward."""

    tcp_pose = mdp.DoorHandleTurnCommandCfg(
        asset_name="robot",
        body_name="tcp_frame",
        door_asset_name="door",
        handle_body_name="handle_grasp",
        resampling_time_range=(1.0e9, 1.0e9),
        approach_offset_handle=(-0.02, 0.0, 0.0),
        position_jitter_range=((0.0, 0.0), (-0.002, 0.002), (-0.002, 0.002)),
        preserve_start_orientation=True,
        stationary_probability=0.0,
        settle_time_s=2.0,
        motion_time_s=3.5,
        recapture_during_settle=True,
        turn_start_s=7.0,
        turn_translation_w=(0.0, 0.015, -0.075),
        debug_vis=False,
    )


@configclass
class B2WZ1DoorTurnEnvCfg(B2WZ1DoorGraspEnvCfg):
    """Stage 14: test handle rotation using the existing low-target WBC."""

    commands: B2WZ1DoorTurnCommandsCfg = B2WZ1DoorTurnCommandsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 16.0


@configclass
class B2WZ1DoorTurnEnvCfg_PLAY(B2WZ1DoorTurnEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0


@configclass
class B2WZ1DoorPullCommandsCfg:
    tcp_pose = mdp.DoorHandlePullCommandCfg(
        asset_name="robot",
        body_name="tcp_frame",
        door_asset_name="door",
        handle_body_name="handle_grasp",
        resampling_time_range=(1.0e9, 1.0e9),
        approach_offset_handle=(-0.02, 0.0, 0.0),
        position_jitter_range=((0.0, 0.0), (-0.002, 0.002), (-0.002, 0.002)),
        preserve_start_orientation=True,
        stationary_probability=0.0,
        settle_time_s=2.0,
        motion_time_s=3.5,
        recapture_during_settle=True,
        turn_start_s=7.0,
        turn_translation_w=(0.0, 0.015, -0.075),
        pull_start_s=11.0,
        pull_translation_w=(-0.22, -0.13, 0.0),
        debug_vis=False,
    )


@configclass
class B2WZ1DoorPullActionsCfg(B2WZ1EEWBCActionsCfg):
    door_latch = mdp.DoorLatchActionCfg(
        asset_name="door",
        door_joint_name="door_hinge",
        handle_joint_name="handle_joint",
        release_angle=0.30,
        lock_stiffness=500.0,
        lock_damping=20.0,
        maximum_lock_effort=120.0,
    )
    compliant_grasp = mdp.CompliantGraspActionCfg(
        asset_name="robot",
        door_asset_name="door",
        tcp_body_name="tcp_frame",
        handle_body_name="handle_grasp",
        robot_force_body_name="gripper_stator",
        door_force_body_name="door_panel",
        gripper_action_name="gripper_hold",
        activation_closure_progress=0.95,
        activation_distance=0.06,
        stiffness=800.0,
        damping=40.0,
        maximum_force=80.0,
        break_distance=0.20,
    )


@configclass
class B2WZ1DoorPullEnvCfg(B2WZ1DoorTurnEnvCfg):
    """Stage 15: release the latch and pull the door toward the robot."""

    commands: B2WZ1DoorPullCommandsCfg = B2WZ1DoorPullCommandsCfg()
    actions: B2WZ1DoorPullActionsCfg = B2WZ1DoorPullActionsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.episode_length_s = 22.0


@configclass
class B2WZ1DoorPullEnvCfg_PLAY(B2WZ1DoorPullEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
