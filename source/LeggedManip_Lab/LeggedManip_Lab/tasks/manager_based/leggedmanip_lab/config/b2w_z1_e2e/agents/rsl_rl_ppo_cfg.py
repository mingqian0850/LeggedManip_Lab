from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class B2WZ1EEWBCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage-1 single-critic PPO; multi-critic is a later controlled ablation."""

    obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
    num_steps_per_env = 32
    check_for_nan = True
    max_iterations = 3000
    # Stabilization runs are selected by deterministic batch evaluation rather
    # than by taking the final checkpoint, so retain intermediate policies.
    save_interval = 25
    experiment_name = "b2w_z1_e2e_ee_wbc"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.45,
            std_type="log",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class B2WZ1EEWBCFineTunePPORunnerCfg(B2WZ1EEWBCPPORunnerCfg):
    """Conservative PPO updates for workspace/posture curriculum expansion."""

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.10,
        entropy_coef=0.001,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.7,
    )


@configclass
class B2WZ1EEWBCPosturePPORunnerCfg(B2WZ1EEWBCPPORunnerCfg):
    """Very conservative updates for retaining workspace while refining posture."""

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.08,
        entropy_coef=0.0005,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=0.5,
    )


@configclass
class B2WZ1EEWBCStage21PPORunnerCfg(B2WZ1EEWBCPPORunnerCfg):
    """Small updates for wide-range heading curriculum without precision collapse."""

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.06,
        entropy_coef=0.0005,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=2.5e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.0025,
        max_grad_norm=0.5,
    )
