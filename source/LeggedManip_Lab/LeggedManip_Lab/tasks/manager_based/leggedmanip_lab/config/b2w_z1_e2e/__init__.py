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
    id="B2W-Z1-EE-WBC-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)
