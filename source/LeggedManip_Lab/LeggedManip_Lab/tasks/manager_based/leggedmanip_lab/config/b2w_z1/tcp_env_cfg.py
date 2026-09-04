"""Nominal flat-ground B2W + Z1 world-frame TCP training task."""

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

from LeggedManip_Lab.assets.b2w_z1.b2w_z1_articulation_cfg import B2W_Z1_ROBOT_LAB_TCP_CFG
from LeggedManip_Lab.controllers import ROBOT_LAB_B2W_POLICY_PATH
from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab import mdp

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
TCP_CFG = SceneEntityCfg("robot", body_names="tcp_frame")
ARM_CFG = SceneEntityCfg("robot", joint_names=ARM_JOINTS, preserve_order=True)


@configclass
class B2WZ1TCPSceneCfg(InteractiveSceneCfg):
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
    robot = B2W_Z1_ROBOT_LAB_TCP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class B2WZ1TCPCommandsCfg:
    tcp_pose = mdp.FKReachableWorldPoseCommandCfg(
        asset_name="robot",
        body_name="tcp_frame",
        # One immutable target per episode. These bounds describe root
        # relocation, not a hand-authored Cartesian arm workspace.
        resampling_time_range=(1.0e9, 1.0e9),
        radius_range=(0.05, 0.20),
        bearing_range=(-math.pi, math.pi),
        yaw_range=(-0.25, 0.25),
        stationary_probability=0.1,
        settle_time_s=1.0,
        motion_time_s=2.0,
        debug_vis=False,
    )


@configclass
class B2WZ1TCPActionsCfg:
    coordinator = mdp.B2WZ1TCPActionCfg(
        asset_name="robot",
        checkpoint_path=str(ROBOT_LAB_B2W_POLICY_PATH),
        command_name="tcp_pose",
        body_name="tcp_frame",
        low_level_decimation=4,
        max_linear_velocity=0.5,
        max_yaw_velocity=0.8,
        forward_command_bias=0.05,
        max_arm_joint_speed=2.5,
        ik_gain=0.25,
        debug_vis=False,
    )


@configclass
class B2WZ1TCPObservationGroupCfg(ObsGroup):
    """Deployable 57-D coordinator state with no privileged simulator data."""

    tcp_position_error = ObsTerm(
        func=mdp.tcp_reference_position_error_b,
        params={"command_name": "tcp_pose", "asset_cfg": TCP_CFG},
    )
    tcp_orientation_error = ObsTerm(
        func=mdp.tcp_reference_orientation_error_b,
        params={"command_name": "tcp_pose", "asset_cfg": TCP_CFG},
    )
    final_tcp_position_error = ObsTerm(
        func=mdp.tcp_final_position_error_b,
        params={"command_name": "tcp_pose", "asset_cfg": TCP_CFG},
    )
    final_tcp_orientation_error = ObsTerm(
        func=mdp.tcp_final_orientation_error_b,
        params={"command_name": "tcp_pose", "asset_cfg": TCP_CFG},
    )
    desired_tcp_twist = ObsTerm(
        func=mdp.desired_tcp_twist_b,
        params={"command_name": "tcp_pose"},
    )
    arm_joint_position = ObsTerm(func=mdp.normalized_arm_joint_position, params={"asset_cfg": ARM_CFG})
    arm_joint_velocity = ObsTerm(
        func=mdp.scaled_arm_joint_velocity,
        params={"asset_cfg": ARM_CFG, "scale": 0.2},
    )
    root_linear_velocity = ObsTerm(func=mdp.root_linear_velocity_b)
    root_angular_velocity = ObsTerm(func=mdp.root_angular_velocity_b, scale=0.25)
    projected_gravity = ObsTerm(func=mdp.root_projected_gravity_b)
    previous_action = ObsTerm(func=mdp.last_action)
    frozen_policy_action = ObsTerm(
        func=mdp.frozen_policy_last_action,
        params={"action_name": "coordinator"},
        scale=0.1,
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class B2WZ1TCPObservationsCfg:
    policy: B2WZ1TCPObservationGroupCfg = B2WZ1TCPObservationGroupCfg()
    critic: B2WZ1TCPObservationGroupCfg = B2WZ1TCPObservationGroupCfg()


@configclass
class B2WZ1TCPRewardsCfg:
    # Coarse and fine kernels give both a learnable capture basin and a precise
    # terminal objective.
    tcp_position_coarse = RewTerm(
        func=mdp.tcp_reference_position_tracking_exp,
        weight=1.5,
        params={"command_name": "tcp_pose", "std": 0.20, "asset_cfg": TCP_CFG},
    )
    tcp_position_fine = RewTerm(
        func=mdp.tcp_reference_position_tracking_exp,
        weight=4.0,
        params={"command_name": "tcp_pose", "std": 0.03, "asset_cfg": TCP_CFG},
    )
    tcp_orientation_coarse = RewTerm(
        func=mdp.tcp_reference_orientation_tracking_exp,
        weight=0.5,
        params={"command_name": "tcp_pose", "std": 0.35, "asset_cfg": TCP_CFG},
    )
    tcp_orientation_fine = RewTerm(
        func=mdp.tcp_reference_orientation_tracking_exp,
        weight=1.0,
        params={"command_name": "tcp_pose", "std": 0.08, "asset_cfg": TCP_CFG},
    )
    tcp_twist = RewTerm(
        func=mdp.tcp_twist_tracking_exp,
        weight=0.5,
        params={
            "command_name": "tcp_pose",
            "linear_std": 0.15,
            "angular_std": 0.5,
            "asset_cfg": TCP_CFG,
        },
    )
    final_success = RewTerm(
        func=mdp.final_tcp_success,
        weight=3.0,
        params={
            "command_name": "tcp_pose",
            "position_threshold": 0.025,
            "orientation_threshold": math.radians(5.0),
            "asset_cfg": TCP_CFG,
        },
    )
    # TCP tracking alone has infinitely many arm/base solutions. This terminal
    # redundancy objective makes the base recover the arm toward home, but does
    # not expose or reward the sampled ghost-root pose itself.
    terminal_arm_home = RewTerm(
        func=mdp.terminal_arm_home_deviation_l2,
        weight=-0.75,
        params={"command_name": "tcp_pose", "start_progress": 0.8, "asset_cfg": ARM_CFG},
    )
    arm_joint_margin = RewTerm(
        func=mdp.arm_joint_margin_barrier,
        weight=-0.5,
        params={"margin_threshold": 0.15, "asset_cfg": ARM_CFG},
    )
    arm_joint_velocity = RewTerm(
        func=mdp.arm_joint_velocity_l2,
        weight=-1.0e-4,
        params={"asset_cfg": ARM_CFG},
    )
    coordinator_action = RewTerm(
        func=mdp.coordinator_action_l2,
        weight=-0.02,
        params={"action_name": "coordinator"},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    settled_velocity = RewTerm(
        func=mdp.settled_velocity_l2,
        weight=-0.2,
        params={
            "command_name": "tcp_pose",
            "position_threshold": 0.04,
            "orientation_threshold": math.radians(8.0),
            "asset_cfg": TCP_CFG,
        },
    )
    base_height = RewTerm(
        func=mdp.base_height_error_l2,
        weight=-2.0,
        params={"target_height": 0.615},
    )
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    termination = RewTerm(func=mdp.is_terminated, weight=-5.0)


@configclass
class B2WZ1TCPEventCfg:
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class B2WZ1TCPTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    low_base = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.45})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(35.0)})


@configclass
class B2WZ1TCPCurriculumCfg:
    """Nominal phase intentionally has no terrain or dynamics curriculum."""


@configclass
class B2WZ1TCPEnvCfg(ManagerBasedRLEnvCfg):
    scene: B2WZ1TCPSceneCfg = B2WZ1TCPSceneCfg(num_envs=1024, env_spacing=3.0)
    observations: B2WZ1TCPObservationsCfg = B2WZ1TCPObservationsCfg()
    actions: B2WZ1TCPActionsCfg = B2WZ1TCPActionsCfg()
    commands: B2WZ1TCPCommandsCfg = B2WZ1TCPCommandsCfg()
    rewards: B2WZ1TCPRewardsCfg = B2WZ1TCPRewardsCfg()
    terminations: B2WZ1TCPTerminationsCfg = B2WZ1TCPTerminationsCfg()
    events: B2WZ1TCPEventCfg = B2WZ1TCPEventCfg()
    curriculum: B2WZ1TCPCurriculumCfg = B2WZ1TCPCurriculumCfg()

    def __post_init__(self) -> None:
        self.sim.dt = 0.005
        # MVP coordinator and frozen policy run at 50 Hz; Z1 DIK and physics run
        # at 200 Hz. A slower coordinator is a later controlled ablation.
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class B2WZ1TCPEnvCfg_PLAY(B2WZ1TCPEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
