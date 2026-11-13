# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
DreamWaQ FTNet actor model for RSL-RL 5.0.1.

DreamWaQ actor input:

    45-D proprioception
        +
     3-D estimated body velocity
        +
    16-D latent context
        =
    64-D actor input

Actor network:

    64 -> 512 -> 256 -> 128 -> 12
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories

from .cenet import CENet


class FTNetActor(nn.Module):
    """
    DreamWaQ actor model compatible with RSL-RL 5.0.1.

    The model receives the current proprioceptive observation and
    five-frame proprioceptive history.

    The history is processed by CENet to obtain:

        - estimated body velocity: 3-D
        - latent terrain/context representation: 16-D

    These are combined with the current 45-D proprioception and
    passed through the DreamWaQ policy network.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        actor_hidden_dims: tuple[int, ...] | list[int] = (
            512,
            256,
            128,
        ),
        priv_encoder_hidden_dims: tuple[int, ...] | list[int] = (
            128,
            64,
        ),
        hist_encoder_output_channels: (
            int | tuple[int, ...] | list[int]
        ) = (128, 64),
        hist_encoder_kernel_sizes: (
            int | tuple[int, ...] | list[int]
        ) = (3, 3),
        hist_encoder_strides: (
            int | tuple[int, ...] | list[int]
        ) = (1, 1),
        hist_encoder_dilations: (
            int | tuple[int, ...] | list[int]
        ) = (1, 1),
        hist_encoder_padding: str = "none",
        hist_encoder_norm: (
            str | tuple[str, ...] | list[str]
        ) = "none",
        hist_encoder_max_pool: (
            bool | tuple[bool, ...] | list[bool]
        ) = False,
        hist_encoder_global_pool: str = "none",
        hist_encoder_flatten: bool = True,
        encoder_out_dim: int = 16,
        activation: str = "elu",
        obs_normalization=False,
distribution_cfg=None,
cenet_checkpoint="cenet_pretrained.pt",
**kwargs: Any,
    ) -> None:
        super().__init__()

        # ---------------------------------------------------------------
        # Keep the arguments in the signature because they are part of
        # the existing DreamWaQ/RSL-RL configuration interface.
        # CENet performs the actual history encoding.
        # ---------------------------------------------------------------

        del priv_encoder_hidden_dims
        del hist_encoder_output_channels
        del hist_encoder_kernel_sizes
        del hist_encoder_strides
        del hist_encoder_dilations
        del hist_encoder_padding
        del hist_encoder_norm
        del hist_encoder_max_pool
        del hist_encoder_global_pool
        del hist_encoder_flatten
        del kwargs

        self.obs_groups, self.obs_dim = self._get_obs_dim(
    obs,
    obs_groups,
    obs_set,
)

        self.output_dim = output_dim

        # ---------------------------------------------------------------
        # DreamWaQ dimensions
        # ---------------------------------------------------------------

        self.proprio_dim = 45

        self.history_dim = 225

        self.velocity_dim = 3

        self.latent_dim = encoder_out_dim

        self.actor_input_dim = (
            self.proprio_dim
            + self.velocity_dim
            + self.latent_dim
        )

        # Should be:
        #
        # 45 + 3 + 16 = 64
        #
        if self.actor_input_dim != 64:
            raise ValueError(
                "DreamWaQ actor input must be 64-D. "
                f"Got {self.actor_input_dim}-D."
            )

        # ---------------------------------------------------------------
        # Observation normalization
        # ---------------------------------------------------------------

        self.obs_normalization = obs_normalization

        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(
                self.obs_dim
            )
        else:
            self.obs_normalizer = nn.Identity()

        # CENet
        self.cenet = CENet(
            input_dim=self.history_dim,
            latent_dim=self.latent_dim,
            velocity_dim=self.velocity_dim,
            observation_dim=self.proprio_dim,
)

        # Load pretrained CENet checkpoint
        checkpoint_path = Path(cenet_checkpoint).expanduser()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"CENet checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "The CENet checkpoint does not contain 'model_state_dict'."
            )

        self.cenet.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(f"Loaded pretrained CENet from: {checkpoint_path}")

        
        # ---------------------------------------------------------------
                # Action distribution
        # RSL-RL 5.0.1 expects the model itself to create and manage
        # the output distribution.

        if distribution_cfg is not None:
            distribution_cfg = dict(distribution_cfg)

            dist_class = resolve_callable(
                distribution_cfg.pop("class_name")
            )

            distribution = dist_class(
                output_dim,
                **distribution_cfg,
            )

            self.distribution = distribution
            mlp_output_dim = distribution.input_dim

        else:
            self.distribution = None
            mlp_output_dim = output_dim

        # DreamWaQ policy network

        self.actor = MLP(
            self.actor_input_dim,
            mlp_output_dim,
            actor_hidden_dims,
            activation,
        )

        # Initialize the output distribution's MLP weights.
        if self.distribution is not None:
            self.distribution.init_mlp_weights(
                self.actor
            )

    # ===================================================================
    # Observation handling
    # ===================================================================

    @staticmethod
    def _get_obs_dim(
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
) -> tuple[list[str], int]:
        """
        Resolve the observation groups used by RSL-RL.

        In the current DreamWaQ environment the observation group
        is named 'policy'. RSL-RL resolves the missing 'actor'
        observation set to this group.
        """

        active_obs_groups = obs_groups[obs_set]

        obs_dim = 0

        for obs_group in active_obs_groups:

            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    "FTNetActor only supports 1-D observation "
                    f"groups. Got shape {obs[obs_group].shape} "
                    f"for '{obs_group}'."
                )

            obs_dim += obs[obs_group].shape[-1]

        return active_obs_groups, obs_dim

    def _get_policy_observation(
                    self,
                    obs: TensorDict,
                ) -> torch.Tensor:
                    """
                    Concatenate the observation groups selected by RSL-RL.

                    The current environment normally gives us:

                        policy = [45-D proprioception | 225-D history]

                    Therefore the resulting tensor is 270-D.
                    """

                    obs_list = [
                        obs[group]
                        for group in self.obs_groups
                    ]

                    return torch.cat(
                        obs_list,
                        dim=-1,
                    )

    def _split_observation(
                self,
                policy_obs: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                """
                Split the 270-D policy observation.

                Layout:

                    0:45     -> current proprioception
                    45:270   -> five-frame history
                """

                expected_dim = (
                    self.proprio_dim
                    + self.history_dim
                )

                if policy_obs.shape[-1] != expected_dim:
                    raise ValueError(
                        "DreamWaQ expected the policy observation to "
                        f"contain {expected_dim} values "
                        f"(45 proprio + 225 history), but received "
                        f"{policy_obs.shape[-1]}."
                    )

                proprio = policy_obs[
                    ...,
                    : self.proprio_dim,
                ]

                history = policy_obs[
                    ...,
                    self.proprio_dim :,
                ]

                return proprio, history

            # ===================================================================
            # DreamWaQ latent computation
            # ===================================================================

    def get_latent(
                self,
                obs: TensorDict,
                masks: torch.Tensor | None = None,
                hidden_state: HiddenState = None,
            ) -> torch.Tensor:
                """
                Build the DreamWaQ actor latent.

                Steps:

                    policy observation
                            ↓
                    45-D proprio + 225-D history
                            ↓
                        CENet
                        ↙      ↘
                    3-D       16-D
                velocity    latent
                        ↘      ↙
                    concatenate
                            ↓
                        64-D
                """

                del hidden_state

                if masks is not None:
                    obs = unpad_trajectories(
                        obs,
                        masks,
                    )

                policy_obs = self._get_policy_observation(
                    obs
                )

                # Normalize only if explicitly requested.
                policy_obs = self.obs_normalizer(
                    policy_obs
                )

                proprio, history = self._split_observation(
                    policy_obs
                )

                # ---------------------------------------------------------------
                # CENet expects:
                #
                # [batch, 225]
                #
                # The observation history is already flattened by the
                # observation manager.
                # ---------------------------------------------------------------

                if history.ndim != 2:
                    raise ValueError(
                        "DreamWaQ history must be a 2-D tensor "
                        "with shape [batch, 225]. "
                        f"Received {history.shape}."
                    )

                # ---------------------------------------------------------------
                # CENet
                # ---------------------------------------------------------------

                cenet_output = self.cenet(
                    history
                )

                estimated_velocity = cenet_output["velocity"]
                latent_context = cenet_output["mu"]

                # ---------------------------------------------------------------
                # Combine:
                #
                # 45-D proprioception
                # + 3-D estimated velocity
                # + 16-D context
                # = 64-D
                #
                # We intentionally do NOT detach these tensors here.
                #
                # This allows the actor loss to propagate through CENet.
                # The separate CENet hybrid loss will be integrated later.
                # ---------------------------------------------------------------

                actor_latent = torch.cat(
                    (
                        proprio,
                        estimated_velocity,
                        latent_context,
                    ),
                    dim=-1,
                )

                if actor_latent.shape[-1] != 64:
                    raise RuntimeError(
                        "DreamWaQ actor latent must be 64-D. "
                        f"Received {actor_latent.shape[-1]}-D."
                    )

                return actor_latent

            # ===================================================================
            # Forward
            # ===================================================================

    def forward(
                self,
                obs: TensorDict,
                masks: torch.Tensor | None = None,
                hidden_state: HiddenState = None,
                stochastic_output: bool = False,
            ) -> torch.Tensor:
                """
                Forward pass through the DreamWaQ actor.
                """

                latent = self.get_latent(
                    obs,
                    masks,
                    hidden_state,
                )

                # ---------------------------------------------------------------
                # Actor MLP
                # ---------------------------------------------------------------

                mlp_output = self.actor(
                    latent
                )

                # ---------------------------------------------------------------
                # Action distribution
                # ---------------------------------------------------------------

                if self.distribution is not None:

                    if stochastic_output:
                        self.distribution.update(
                            mlp_output
                        )

                        return self.distribution.sample()

                    return self.distribution.deterministic_output(
                        mlp_output
                    )

                return mlp_output

            # ===================================================================
            # Distribution interface required by RSL-RL
            # ===================================================================

    @property
    def output_mean(self) -> torch.Tensor:
                """Return the current action-distribution mean."""

                if self.distribution is None:
                    raise RuntimeError(
                        "FTNetActor has no action distribution."
                    )

                return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
                """Return the current action-distribution standard deviation."""

                if self.distribution is None:
                    raise RuntimeError(
                        "FTNetActor has no action distribution."
                    )

                return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
                """Return the current action-distribution entropy."""

                if self.distribution is None:
                    raise RuntimeError(
                        "FTNetActor has no action distribution."
                    )

                return self.distribution.entropy

    @property
    def output_distribution_params(
                self,
            ) -> tuple[torch.Tensor, ...]:
                """Return the raw distribution parameters."""

                if self.distribution is None:
                    raise RuntimeError(
                        "FTNetActor has no action distribution."
                    )

                return self.distribution.params

    def get_output_log_prob(
                self,
                outputs: torch.Tensor,
            ) -> torch.Tensor:
                """Compute log probability of sampled actions."""

                if self.distribution is None:
                    raise RuntimeError(
                        "FTNetActor has no action distribution."
                    )

                return self.distribution.log_prob(
                    outputs
                )

    def get_kl_divergence(
                self,
                old_params: tuple[torch.Tensor, ...],
                new_params: tuple[torch.Tensor, ...],
            ) -> torch.Tensor:
                """Compute KL divergence between two distributions."""

                if self.distribution is None:
                    raise RuntimeError(
                        "FTNetActor has no action distribution."
                    )

                return self.distribution.kl_divergence(
                    old_params,
                    new_params,
                )

            # ===================================================================
            # Recurrent-model compatibility
            # ===================================================================

    def reset(
                self,
                dones: torch.Tensor | None = None,
                hidden_state: HiddenState = None,
            ) -> None:
                """Reset model state. FTNet is non-recurrent."""

                del dones
                del hidden_state

    def get_hidden_state(self) -> HiddenState:
                """Return hidden state. FTNet is non-recurrent."""

                return None

    def detach_hidden_state(
                self,
                dones: torch.Tensor | None = None,
            ) -> None:
                """No-op because FTNet is non-recurrent."""

                del dones

            # ===================================================================
            # Observation normalization
            # ===================================================================

    def update_normalization(
        self,
        obs: TensorDict,
    ) -> None:
        """Update observation-normalization statistics."""

        if not self.obs_normalization:
            return

        policy_obs = self._get_policy_observation(obs)

        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer.update(policy_obs)

                # ===================================================================
                # CENet loss
                # ===================================================================

    def compute_cenet_loss(
        self,
        history: torch.Tensor,
        next_observation: torch.Tensor,
        velocity_target: torch.Tensor,
        beta: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute the DreamWaQ CENet hybrid loss."""

        outputs = self.cenet(history)

        return self.cenet.compute_loss(
            next_observation=next_observation,
            velocity_target=velocity_target,
            outputs=outputs,
            beta=beta,
        )