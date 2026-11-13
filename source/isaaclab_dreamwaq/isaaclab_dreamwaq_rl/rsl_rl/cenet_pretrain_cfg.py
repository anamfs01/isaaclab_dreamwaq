# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for DreamWaQ CENet pretraining.
"""

from isaaclab.utils import configclass


@configclass
class CENetPretrainCfg:
    """Training configuration for the DreamWaQ Context-Aided Estimator Network."""

    # ---------------------------------------------------------------
    # Observation dimensions
    # ---------------------------------------------------------------

    history_length: int = 5
    proprioception_dim: int = 45

    # 5 × 45 = 225
    history_dim: int = 225

    # ---------------------------------------------------------------
    # CENet dimensions
    # ---------------------------------------------------------------

    encoder_hidden_dims: tuple[int, ...] = (128, 64)

    velocity_dim: int = 3

    latent_dim: int = 16

    decoder_hidden_dims: tuple[int, ...] = (64, 128)

    # Reconstruct the next 45-D proprioceptive observation
    reconstruction_dim: int = 45

    # ---------------------------------------------------------------
    # β-VAE loss
    # ---------------------------------------------------------------

    beta: float = 1.0

    # ---------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------

    learning_rate: float = 1.0e-3

    weight_decay: float = 0.0

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    batch_size: int = 256

    num_epochs: int = 100

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    dataset_size: int = 100_000

    num_workers: int = 0

    # ---------------------------------------------------------------
    # Saving
    # ---------------------------------------------------------------

    save_interval: int = 10

    checkpoint_name: str = "cenet_pretrained.pt"