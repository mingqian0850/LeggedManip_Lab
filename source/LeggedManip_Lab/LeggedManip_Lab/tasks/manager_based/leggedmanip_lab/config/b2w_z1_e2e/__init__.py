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
    id="B2W-Z1-EE-WBC-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ee_wbc_env_cfg:B2WZ1EEWBCEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1EEWBCPPORunnerCfg",
    },
)
