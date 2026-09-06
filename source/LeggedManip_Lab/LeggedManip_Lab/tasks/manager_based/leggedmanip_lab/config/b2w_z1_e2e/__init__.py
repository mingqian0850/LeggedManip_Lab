import gymnasium as gym

from . import agents


gym.register(
    id="B2W-Z1-EE-WBC-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage3EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage4EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage4-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage4EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage5-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage5EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage6-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage6EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage6-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage6EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage7-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage7EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage7-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage7EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage8-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage8EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage8-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage8EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage9-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage9EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage9-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage9EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage10-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage10EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage10-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage10EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPosturePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Slow-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingSlowEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Medium-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingMediumEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Filter50-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingFilter50EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPosturePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Filter35-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingFilter35EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPosturePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Smooth-Filter35-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingSmoothFilter35EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPosturePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Stance-Filter35-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingStanceFilter35EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPosturePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Balanced-Stance-Filter35-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingBalancedStanceFilter35EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPosturePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Mirrored-Filter35-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingMirroredFilter35EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPosturePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Adaptive-Mirrored-Filter35-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingAdaptiveMirroredFilter35EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPosturePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Filter50-Medium-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingFilter50MediumEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Filter35-Medium-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingFilter35MediumEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Mirrored-Filter35-Medium-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingMirroredFilter35MediumEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Adaptive-Mirrored-Filter35-Medium-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingAdaptiveMirroredFilter35MediumEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Filter35-Slow-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingFilter35SlowEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Filter35-Fast-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingFilter35FastEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage5-Filter35-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage5Filter35EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage6-Filter35-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage6Filter35EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage10-Filter35-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage10Filter35EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage10-Mirrored-Filter35-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage10MirroredFilter35EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Stage10-Adaptive-Mirrored-Filter35-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCStage10AdaptiveMirroredFilter35EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Fast-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingFastEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-SixD-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingSixDEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Filter35-SixD-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingFilter35SixDEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Mirrored-Filter35-SixD-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingMirroredFilter35SixDEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Adaptive-Mirrored-Filter35-SixD-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingAdaptiveMirroredFilter35SixDEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Dynamic-Waypoint-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DynamicTrackingWaypointEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Door-Align-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DoorAlignEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Door-Align-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DoorAlignEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Door-Near-Contact-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DoorNearContactEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Door-Near-Contact-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DoorNearContactEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Door-Grasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DoorGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Door-Grasp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DoorGraspEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Door-Turn-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DoorTurnEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Door-Turn-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DoorTurnEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Door-Pull-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DoorPullEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-EE-WBC-Door-Pull-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1DoorPullEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCFineTunePPORunnerCfg",
    },
)
