import gymnasium as gym

from . import agents

gym.register(
    id="B2W-Z1-TCP",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tcp_env_cfg:B2WZ1TCPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1TCPPPORunnerCfg",
    },
)

gym.register(
    id="B2W-Z1-TCP-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tcp_env_cfg:B2WZ1TCPEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:B2WZ1TCPPPORunnerCfg",
    },
)
