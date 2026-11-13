# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
DreamWaQ Context-Aided Estimator Network (CENet).

DreamWaQ CENet takes five historical proprioceptive observations
and jointly estimates:

    1. Body linear velocity
    2. A 16-D latent context vector
    3. The next proprioceptive observation

The shared encoder follows the architecture described in the
DreamWaQ paper:

    225 -> 128 -> 64

The latent representation has 16 dimensions.
"""

from __future__ import annotations

import torch
from torch import nn


class CENet(nn.Module):
    """
    DreamWaQ Context-Aided Estimator Network.

    Input:
        5 × 45 = 225-D temporal proprioceptive observation.

    Outputs:
        body velocity: 3-D
        latent context: 16-D
        reconstructed next observation: 45-D
    """

    def __init__(
        self,
        input_dim: int = 225,
        latent_dim: int = 16,
        velocity_dim: int = 3,
        observation_dim: int = 45,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.velocity_dim = velocity_dim
        self.observation_dim = observation_dim

        # ---------------------------------------------------------------
        # Shared encoder
        #
        # DreamWaQ:
        #
        #       225
        #        ↓
        #       128
        #        ↓
        #        64
        # ---------------------------------------------------------------

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),

            nn.Linear(128, 64),
            nn.ELU(),
        )

        # ---------------------------------------------------------------
        # Body velocity estimation head
        #
        # 64 -> 3
        # ---------------------------------------------------------------

        self.velocity_head = nn.Sequential(
            nn.Linear(64, velocity_dim),
        )

        # ---------------------------------------------------------------
        # β-VAE latent heads
        #
        # 64 -> 16 mean
        # 64 -> 16 log variance
        # ---------------------------------------------------------------

        self.mu_head = nn.Linear(
            64,
            latent_dim,
        )

        self.logvar_head = nn.Linear(
            64,
            latent_dim,
        )

        # ---------------------------------------------------------------
        # Decoder
        #
        # Reconstruct the NEXT proprioceptive observation.
        #
        #       16
        #        ↓
        #       64
        #        ↓
        #       128
        #        ↓
        #       45
        #
        # The DreamWaQ paper describes this as reconstruction of
        # o_(t+1), rather than reconstruction of the input history.
        # ---------------------------------------------------------------

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ELU(),

            nn.Linear(64, 128),
            nn.ELU(),

            nn.Linear(128, observation_dim),
        )

    def encode(
        self,
        history: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Encode the temporal proprioceptive history.

        Args:
            history:
                Tensor with shape [batch, 225].

        Returns:
            features:
                Shared 64-D encoder representation.

            mu:
                16-D latent mean.

            logvar:
                16-D latent log variance.
        """

        features = self.encoder(history)

        mu = self.mu_head(features)

        logvar = self.logvar_head(features)

        return features, mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reparameterization trick for the β-VAE.
        """

        std = torch.exp(
            0.5 * logvar
        )

        epsilon = torch.randn_like(std)

        return mu + epsilon * std

    def decode(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct the next proprioceptive observation.
        """

        return self.decoder(latent)

    def forward(
        self,
        history: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Run the complete CENet.

        Args:
            history:
                [batch, 225]

        Returns:
            velocity:
                [batch, 3]

            latent:
                [batch, 16]

            mu:
                [batch, 16]

            logvar:
                [batch, 16]

            reconstruction:
                [batch, 45]
        """

        features, mu, logvar = self.encode(history)

        # Body velocity estimation.
        velocity = self.velocity_head(features)

        # Sample latent context.
        latent = self.reparameterize(
            mu,
            logvar,
        )

        # Reconstruct next observation.
        reconstruction = self.decode(latent)

        return {
            "velocity": velocity,
            "latent": latent,
            "mu": mu,
            "logvar": logvar,
            "reconstruction": reconstruction,
        }

    def compute_loss(
        self,
        next_observation: torch.Tensor,
        velocity_target: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        beta: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the DreamWaQ CENet hybrid loss.

        L_CE = L_est + L_VAE

        L_est:
            MSE between estimated and true body velocity.

        L_VAE:
            Reconstruction MSE + beta * KL divergence.
        """

        # ---------------------------------------------------------------
        # Body velocity estimation loss
        # ---------------------------------------------------------------

        velocity_loss = torch.mean(
            (
                outputs["velocity"]
                - velocity_target
            ) ** 2
        )

        # ---------------------------------------------------------------
        # Reconstruction loss
        # ---------------------------------------------------------------

        reconstruction_loss = torch.mean(
            (
                outputs["reconstruction"]
                - next_observation
            ) ** 2
        )

        # ---------------------------------------------------------------
        # KL divergence
        #
        # q(z | o_H) against N(0, I)
        # ---------------------------------------------------------------

        mu = outputs["mu"]

        logvar = outputs["logvar"]

        kl_loss = -0.5 * torch.mean(
    torch.sum(
        1.0
        + logvar
        - mu.pow(2)
        - logvar.exp(),
        dim=1,
    )
)

        # ---------------------------------------------------------------
        # β-VAE loss
        # ---------------------------------------------------------------

        vae_loss = (
            reconstruction_loss
            + beta * kl_loss
        )

        # ---------------------------------------------------------------
        # Complete CENet loss
        # ---------------------------------------------------------------

        total_loss = (
            velocity_loss
            + vae_loss
        )

        return {
            "total_loss": total_loss,
            "velocity_loss": velocity_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "vae_loss": vae_loss,
        }