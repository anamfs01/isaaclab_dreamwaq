
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Collect training data for the DreamWaQ CENet.

Collected data:
    history          : [N, 225]
    next_observation : [N, 45]
    velocity         : [N, 3]
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher


# ---------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Collect DreamWaQ CENet training data."
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=256,
    help="Number of parallel environments.",
)

parser.add_argument(
    "--num_steps",
    type=int,
    default=400,
    help="Number of simulation steps.",
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()


# ---------------------------------------------------------
# Launch Isaac Sim
# ---------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ---------------------------------------------------------
# Imports after launching Isaac Sim
# ---------------------------------------------------------

import torch
import gymnasium as gym

# Register DreamWaQ Gym environment
import isaaclab_dreamwaq_tasks  # noqa: F401

from isaaclab_dreamwaq_tasks.tasks.manager_based.isaaclab_dreamwaq.isaaclab_dreamwaq_env_cfg import (
    IsaaclabDreamwaqEnvCfg,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ENV_NAME = "Template-Isaaclab-Dreamwaq-v0"

SAVE_PATH = "logs/cenet_dataset.pt"


# ---------------------------------------------------------
# Main collection function
# ---------------------------------------------------------

def main():

    env = None

    try:
        print("Creating DreamWaQ environment...")

        # Create environment configuration
        env_cfg = IsaaclabDreamwaqEnvCfg()

        # Set number of parallel environments
        env_cfg.scene.num_envs = args_cli.num_envs

        # Create environment
        env = gym.make(
            ENV_NAME,
            cfg=env_cfg,
        )

        env = env.unwrapped

        print("Environment created.")
        print(f"Number of environments: {env.num_envs}")

        # -------------------------------------------------
        # Reset environment
        # -------------------------------------------------

        obs, _ = env.reset()

        # -------------------------------------------------
        # Dataset storage
        # -------------------------------------------------

        histories = []
        next_observations = []
        velocities = []

        print("Starting CENet data collection...")

        # -------------------------------------------------
        # Collection loop
        # -------------------------------------------------

        # Initialize action buffer
        action = torch.zeros(
            env.num_envs,
            env.action_manager.total_action_dim,
            device=env.device,
        )

        for step in range(args_cli.num_steps):

            # Current actor observation
            actor_obs = obs["actor"]

            # Extract 225-D history
            history = actor_obs[:, 45:270].reshape(
                env.num_envs,
                225,
            )

            # True body linear velocity
            velocity = (
                env.scene["robot"].data.root_lin_vel_b.clone()
            )

            # Update random action every 20 steps
            if step % 20 == 0:
                action.uniform_(-0.3, 0.3)

            # Step environment
            next_obs, _, terminated, truncated, _ = env.step(action)

            # Extract next proprioceptive observation
            next_actor_obs = next_obs["actor"]
            next_proprioception = next_actor_obs[:, :45]

            # Identify completed episodes
            done = terminated | truncated

            # Keep only transitions that did not terminate
            valid = ~done

            if valid.any():

                histories.append(
                    history[valid].detach().cpu()
                )

                next_observations.append(
                    next_proprioception[valid].detach().cpu()
                )

                velocities.append(
                    velocity[valid].detach().cpu()
                )

            # Update current observation
            obs = next_obs

            # Progress
            if (step + 1) % 100 == 0:
                print(
                    f"Collected simulation step "
                    f"{step + 1}/{args_cli.num_steps}"
                )

        # -------------------------------------------------
        # Combine collected data
        # -------------------------------------------------

        if not histories:
            raise RuntimeError(
                "No valid samples were collected."
            )

        histories = torch.cat(histories, dim=0)

        next_observations = torch.cat(
            next_observations,
            dim=0,
        )

        velocities = torch.cat(
            velocities,
            dim=0,
        )

        # -------------------------------------------------
        # Save dataset
        # -------------------------------------------------

        os.makedirs(
            os.path.dirname(SAVE_PATH) or ".",
            exist_ok=True,
        )

        dataset = {
            "history": histories,
            "next_observation": next_observations,
            "velocity": velocities,
        }

        torch.save(
            dataset,
            SAVE_PATH,
        )

        print()
        print("======================================")
        print("CENet dataset collection completed.")
        print("======================================")
        print(f"History shape: {histories.shape}")
        print(
            f"Next observation shape: "
            f"{next_observations.shape}"
        )
        print(f"Velocity shape: {velocities.shape}")
        print(f"Saved to: {SAVE_PATH}")

    finally:

        if env is not None:
            env.close()

        simulation_app.close()


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
