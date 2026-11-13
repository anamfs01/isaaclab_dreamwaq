
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""DreamWaQ PPO algorithm with CENet auxiliary learning."""

from __future__ import annotations

import torch
import torch.nn as nn

from tensordict import TensorDict
from rsl_rl.storage.rollout_storage import RolloutStorage
from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_callable, resolve_obs_groups






class DreamWaQRolloutStorage(RolloutStorage):
    """Rollout storage extended with CENet training targets."""

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs,
        actions_shape,
        device="cpu",
    ):
        super().__init__(
            training_type=training_type,
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs=obs,
            actions_shape=actions_shape,
            device=device,
        )

        if training_type != "rl":
            raise ValueError("DreamWaQ storage supports RL training only.")

        self.next_proprio = torch.zeros(
            num_transitions_per_env, num_envs, 45, device=device
        )

        self.velocity_target = torch.zeros(
            num_transitions_per_env, num_envs, 3, device=device
        )

        self.cenet_valid = torch.zeros(
            num_transitions_per_env, num_envs,
            dtype=torch.bool,
            device=device,
        )

    def add_transition(
        self,
        transition,
        next_proprio,
        velocity_target,
        valid,
    ):
        """Store a PPO transition and its CENet targets."""

        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow!")

        self.next_proprio[self.step].copy_(next_proprio)
        self.velocity_target[self.step].copy_(velocity_target)
        self.cenet_valid[self.step].copy_(valid.reshape(-1).bool())

        super().add_transition(transition)


    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """Generate PPO mini-batches including CENet targets."""

        if self.training_type != "rl":
            raise ValueError("This generator is only for RL training.")

        batch_size = self.num_envs * self.num_transitions_per_env

        if num_mini_batches <= 0:
            raise ValueError("num_mini_batches must be greater than zero.")

        if batch_size % num_mini_batches != 0:
            raise ValueError(
                "Rollout batch size must be divisible by num_mini_batches."
            )

        mini_batch_size = batch_size // num_mini_batches

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)

        old_distribution_params = tuple(
            p.flatten(0, 1) for p in self.distribution_params
        )

        next_proprio = self.next_proprio.flatten(0, 1)
        velocity_target = self.velocity_target.flatten(0, 1)
        cenet_valid = self.cenet_valid.flatten(0, 1)

        for _ in range(num_epochs):
            # Shuffle samples independently for every epoch.
            indices = torch.randperm(batch_size, device=self.device)

            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                batch = RolloutStorage.Batch(
                    observations=observations[batch_idx],
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(
                        p[batch_idx] for p in old_distribution_params
                    ),
                )

                batch.next_proprio = next_proprio[batch_idx]
                batch.velocity_target = velocity_target[batch_idx]
                batch.cenet_valid = cenet_valid[batch_idx]

                yield batch


    def clear(self):
        """Reset storage for the next rollout."""
        super().clear()

class DreamWaQPPO(PPO):
    """PPO with an additional CENet hybrid learning objective."""

    def __init__(
        self,
        actor,
        critic,
        storage,
        cenet_loss_coef: float = 1.0,
        cenet_beta: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            actor,
            critic,
            storage,
            **kwargs,
        )

        self.cenet_loss_coef = cenet_loss_coef
        self.cenet_beta = cenet_beta

    
    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> None:
        """Store PPO transitions and aligned CENet targets."""

        # The transition contains the observations used to select
        # the action, before the environment step.
        current_obs = self.transition.observations
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if current_obs is None:
            raise RuntimeError(
                "Current observations are missing from the PPO transition."
            )

        # Capture the current true body velocity from privileged observations.
        velocity_target = current_obs["critic"][:, -3:].clone()

        # Capture the next proprioception from the returned observation.
        next_proprio = obs["actor"][:, :45].clone()

        # Identify transitions that did not terminate or truncate.
        valid = ~dones.bool().reshape(-1)

        # Preserve the timeout reward correction used by PPO.
        self.transition.rewards = rewards.clone()

        if "time_outs" in extras:
            self.transition.rewards += (
                self.gamma
                * torch.squeeze(
                    self.transition.values
                    * extras["time_outs"].unsqueeze(1).to(self.device),
                    1,
                )
            )

        self.transition.dones = dones

        # Store the PPO transition and its matching CENet targets.
        self.storage.add_transition(
            self.transition,
            next_proprio,
            velocity_target,
            valid,
)

        # Clear the temporary transition.
        self.transition.clear()

        # Reset recurrent states if the model supports them.
        self.actor.reset(dones)
        self.critic.reset(dones)


    def update(self) -> dict[str, float]:
        """Optimize PPO and CENet losses together."""

        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise NotImplementedError(
                "DreamWaQPPO currently supports feed-forward models only."
            )

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_cenet_loss = 0.0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
        )

        for batch in generator:

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (
                        batch.advantages - batch.advantages.mean()
                    ) / (batch.advantages.std() + 1e-8)

            # Recalculate actor distribution and critic values.
            self.actor(
                batch.observations,
                stochastic_output=True,
            )

            actions_log_prob = self.actor.get_output_log_prob(
                batch.actions
            )

            values = self.critic(batch.observations)

            entropy = self.actor.output_entropy

            # Adaptive KL learning-rate adjustment.
            if self.desired_kl is not None and self.schedule == "adaptive":

                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params,
                        self.actor.output_distribution_params,
                    )

                    kl_mean = kl.mean()

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(
                        1e-5,
                        self.learning_rate / 1.5,
                    )

                elif (
                    kl_mean < self.desired_kl / 2.0
                    and kl_mean > 0.0
                ):
                    self.learning_rate = min(
                        1e-2,
                        self.learning_rate * 1.5,
                    )

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

            # PPO surrogate loss.
            ratio = torch.exp(
    actions_log_prob.reshape(-1)
    - batch.old_actions_log_prob.reshape(-1)
)

            surrogate = (
                -torch.squeeze(batch.advantages) * ratio
            )

            surrogate_clipped = (
                -torch.squeeze(batch.advantages)
                * torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                )
            )

            surrogate_loss = torch.max(
                surrogate,
                surrogate_clipped,
            ).mean()

            # Critic value loss.
            if self.use_clipped_value_loss:

                value_clipped = (
                    batch.values
                    + (values - batch.values).clamp(
                        -self.clip_param,
                        self.clip_param,
                    )
                )

                value_losses = (values - batch.returns).pow(2)

                value_losses_clipped = (
                    value_clipped - batch.returns
                ).pow(2)

                value_loss = torch.max(
                    value_losses,
                    value_losses_clipped,
                ).mean()

            else:
                value_loss = (
                    batch.returns - values
                ).pow(2).mean()

            # Combined PPO objective.
            ppo_loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy.mean()
            )

            # CENet auxiliary hybrid objective.
            valid = batch.cenet_valid

            if valid.any():

                actor_observation = batch.observations["actor"]

                # Actor observation layout:
                # [45-D current proprioception | 225-D history]
                history = actor_observation[:, 45:270]

                cenet_losses = self.actor.compute_cenet_loss(
                    history=history[valid],
                    next_observation=batch.next_proprio[valid],
                    velocity_target=batch.velocity_target[valid],
                    beta=self.cenet_beta,
                )

                cenet_loss = cenet_losses["total_loss"]

            else:
                cenet_loss = ppo_loss.new_zeros(())

            # Optimize PPO and CENet together.
            total_loss = (
                ppo_loss
                + self.cenet_loss_coef * cenet_loss
            )

            self.optimizer.zero_grad()
            total_loss.backward()

            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.max_grad_norm,
            )

            nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                self.max_grad_norm,
            )

            if self.is_multi_gpu:
                self.reduce_parameters()

            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_cenet_loss += cenet_loss.item()

        num_updates = (
            self.num_learning_epochs * self.num_mini_batches
        )

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_cenet_loss /= num_updates

        self.storage.clear()

        return {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "cenet": mean_cenet_loss,
        }

    
    @staticmethod
    def construct_algorithm(
        obs: TensorDict,
        env,
        cfg: dict,
        device: str,
    ):
        """Construct the DreamWaQ PPO algorithm."""

        # Read configurations without modifying the runner's config.
        algorithm_cfg = dict(cfg["algorithm"])
        actor_cfg = dict(cfg["actor"])
        critic_cfg = dict(cfg["critic"])

        # Remove class names before passing constructor arguments.
        algorithm_cfg.pop("class_name", None)

        actor_class = resolve_callable(
            actor_cfg.pop("class_name")
        )
        critic_class = resolve_callable(
            critic_cfg.pop("class_name")
        )

        # This first implementation supports feed-forward PPO only.
        if algorithm_cfg.get("rnd_cfg") is not None:
            raise NotImplementedError("RND is not supported yet.")

        if algorithm_cfg.get("symmetry_cfg") is not None:
            raise NotImplementedError("Symmetry is not supported yet.")

        # Resolve observation groups.
        obs_groups = resolve_obs_groups(
            obs,
            cfg["obs_groups"],
            ["actor", "critic"],
        )

        # Build actor.
        actor = actor_class(
            obs,
            obs_groups,
            "actor",
            env.num_actions,
            **actor_cfg,
        ).to(device)

        # Build critic.
        critic = critic_class(
            obs,
            obs_groups,
            "critic",
            1,
            **critic_cfg,
        ).to(device)

        # Build custom rollout storage.
        storage = DreamWaQRolloutStorage(
            training_type="rl",
            num_envs=env.num_envs,
            num_transitions_per_env=cfg["num_steps_per_env"],
            obs=obs,
            actions_shape=[env.num_actions],
            device=device,
        )

        # Build custom PPO.
        algorithm = DreamWaQPPO(
            actor=actor,
            critic=critic,
            storage=storage,
            device=device,
            multi_gpu_cfg=cfg.get("multi_gpu"),
            **algorithm_cfg,
        )

        return algorithm

