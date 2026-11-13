
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from isaaclab_dreamwaq_rl.rsl_rl.rl_cfg import RslRlFTNetActorCfg


@configclass
class DreamWaQPPOActorCfg(RslRlFTNetActorCfg):
    class_name: str = "isaaclab_dreamwaq_rl.rsl_rl.models.ftnet:FTNetActor"

    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    priv_encoder_hidden_dims: tuple[int, ...] = (128, 64)

    hist_encoder_output_channels: tuple[int, ...] = (128, 64)
    hist_encoder_kernel_sizes: tuple[int, ...] = (3, 3)
    hist_encoder_strides: tuple[int, ...] = (1, 1)
    hist_encoder_dilations: tuple[int, ...] = (1, 1)
    hist_encoder_padding: str = "none"
    hist_encoder_norm: str = "none"
    hist_encoder_max_pool: bool = False
    hist_encoder_global_pool: str = "none"
    hist_encoder_flatten: bool = True

    encoder_out_dim: int = 16
    cenet_checkpoint: str = "logs/cenet_pretrained.pt"
    activation: str = "elu"
    obs_normalization: bool = False

    distribution_cfg = RslRlFTNetActorCfg.GaussianDistributionCfg()
    distribution_cfg.init_std = 1.0
    distribution_cfg.std_type = "scalar"


@configclass
class DreamWaQPPOCriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    obs_normalization: bool = False


@configclass
class DreamWaQAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for DreamWaQ PPO."""

    class_name: str = (
        "isaaclab_dreamwaq_rl.rsl_rl.dreamwaq_ppo:DreamWaQPPO"
    )
    cenet_loss_coef: float = 1.0
    cenet_beta: float = 1.0

@configclass
class DreamWaQPPOCfg(RslRlOnPolicyRunnerCfg):
    seed: int = 42
    device: str = "cuda:0"

    num_steps_per_env: int = 24
    max_iterations: int = 10000
    save_interval: int = 100

    experiment_name: str = "dreamwaq"
    run_name: str = ""

    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    logger: str = "tensorboard"

    empirical_normalization: bool = False

    obs_groups = {
        "actor": ["actor"],
        "critic": ["critic"],
    }

    actor = DreamWaQPPOActorCfg()
    
    critic = DreamWaQPPOCriticCfg()

    DreamWaQAlgorithmCfg(
        class_name="isaaclab_dreamwaq_rl.rsl_rl.dreamwaq_ppo:DreamWaQPPO",
        clip_param=0.2,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        cenet_loss_coef=1.0,
        cenet_beta=1.0,
        max_grad_norm=1.0,
    )
