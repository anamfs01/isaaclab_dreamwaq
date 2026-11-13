# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
RSL-RL model configurations for the DreamWaQ project.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass


#########################
# Model configurations #
#########################


@configclass
class RslRlFTNetActorCfg:
    """Configuration for the DreamWaQ FTNetActor model."""

    class_name: str = "FTNetActor"

    # DreamWaQ actor:
    #
    # 64 -> 512 -> 256 -> 128 -> 12
    #
    # 64 = 45-D proprioception
    #    + 3-D estimated body velocity
    #    + 16-D latent context
    actor_hidden_dims: list[int] = [512, 256, 128]

    # Kept for compatibility with the custom FTNet interface.
    priv_encoder_hidden_dims: list[int] = [128, 64]

    # These fields are kept because the RSL-RL configuration
    # interface expects them.
    #
    # DreamWaQ history processing itself is performed by CENet.
    hist_encoder_output_channels: int | tuple[int] | list[int] = [128, 64]

    hist_encoder_kernel_sizes: int | tuple[int] | list[int] = [3, 3]

    hist_encoder_strides: int | tuple[int] | list[int] = [1, 1]

    hist_encoder_dilations: int | tuple[int] | list[int] = [1, 1]

    hist_encoder_padding: Literal[
        "none",
        "zeros",
        "reflect",
        "replicate",
        "circular",
    ] = "none"

    hist_encoder_norm: (
        Literal["none", "batch", "layer"]
        | tuple[str]
        | list[str]
    ) = "none"

    hist_encoder_max_pool: bool | tuple[bool] | list[bool] = False

    hist_encoder_global_pool: Literal[
        "none",
        "max",
        "avg",
    ] = "none"

    hist_encoder_flatten: bool = True

    # DreamWaQ latent context dimension.
    encoder_out_dim: int = 16

    activation: str = "elu"

    obs_normalization: bool = False

    distribution_cfg: DistributionCfg | None = None

    @configclass
    class DistributionCfg:
        """Configuration for the actor output distribution."""

        class_name: str = MISSING

    @configclass
    class GaussianDistributionCfg(DistributionCfg):
        """Configuration for the Gaussian action distribution."""

        class_name: str = "GaussianDistribution"

        # Initial standard deviation of the action distribution.
        init_std: float = 1.0

        # RSL-RL 5.x distribution parameterization.
        std_type: Literal["scalar", "log"] = "scalar"

    @configclass
    class HeteroscedasticGaussianDistributionCfg(
        GaussianDistributionCfg
    ):
        """Configuration for the heteroscedastic Gaussian distribution."""

        class_name: str = "HeteroscedasticGaussianDistribution"

        output_channels: tuple[int] | list[int] = [64, 64]


@configclass
class RslRlDreamFLEXActorCfg:
    """Configuration for the DreamFLEXActor model."""

    class_name: str = "DreamFLEXActor"

    actor_hidden_dims: list[int] = MISSING

    hist_encoder_hidden_dims: (
        int | tuple[int] | list[int]
    ) = MISSING

    latent_decoder_hidden_dims: (
        int | tuple[int] | list[int]
    ) = MISSING

    fault_decoder_hidden_dims: (
        int | tuple[int] | list[int]
    ) = MISSING

    latent_dim: int = MISSING

    activation: str = MISSING

    obs_normalization: bool = False

    distribution_cfg: DistributionCfg | None = None

    @configclass
    class DistributionCfg:
        """Configuration for the output distribution."""

        class_name: str = MISSING

    @configclass
    class GaussianDistributionCfg(DistributionCfg):
        """Configuration for the Gaussian output distribution."""

        class_name: str = "GaussianDistribution"

        init_std: float = MISSING

        std_type: Literal["scalar", "log"] = "scalar"

    @configclass
    class HeteroscedasticGaussianDistributionCfg(
        GaussianDistributionCfg
    ):
        """Configuration for the heteroscedastic Gaussian output distribution."""

        class_name: str = "HeteroscedasticGaussianDistribution"

        output_channels: tuple[int] | list[int] = MISSING


@configclass
class RslRlPINNActorCfg:
    """Configuration for the PINNActor model."""

    class_name: str = "PINNActor"

    actor_hidden_dims: list[int] = MISSING

    hist_encoder_hidden_dims: (
        int | tuple[int] | list[int]
    ) = MISSING

    latent_dim: int = MISSING

    activation: str = MISSING

    obs_normalization: bool = False

    distribution_cfg: DistributionCfg | None = None

    @configclass
    class DistributionCfg:
        """Configuration for the output distribution."""

        class_name: str = MISSING

    @configclass
    class GaussianDistributionCfg(DistributionCfg):
        """Configuration for the Gaussian output distribution."""

        class_name: str = "GaussianDistribution"

        init_std: float = MISSING

        std_type: Literal["scalar", "log"] = "scalar"

    @configclass
    class HeteroscedasticGaussianDistributionCfg(
        GaussianDistributionCfg
    ):
        """Configuration for the heteroscedastic Gaussian output distribution."""

        class_name: str = "HeteroscedasticGaussianDistribution"

        output_channels: tuple[int] | list[int] = MISSING


@configclass
class RslRlGCNActorCfg:
    """Configuration for the GCNActor model."""

    class_name: str = "GCNActor"

    actor_hidden_dims: list[int] = MISSING

    activation: str = MISSING

    obs_normalization: bool = False

    distribution_cfg: DistributionCfg | None = None

    @configclass
    class DistributionCfg:
        """Configuration for the output distribution."""

        class_name: str = MISSING

    @configclass
    class GaussianDistributionCfg(DistributionCfg):
        """Configuration for the Gaussian output distribution."""

        class_name: str = "GaussianDistribution"

        init_std: float = MISSING

        std_type: Literal["scalar", "log"] = "scalar"

    @configclass
    class HeteroscedasticGaussianDistributionCfg(
        GaussianDistributionCfg
    ):
        """Configuration for the heteroscedastic Gaussian output distribution."""

        class_name: str = "HeteroscedasticGaussianDistribution"

        output_channels: tuple[int] | list[int] = MISSING