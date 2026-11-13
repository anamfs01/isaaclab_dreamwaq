# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
DreamWaQ reward terms.

These reward functions are designed for quadrupedal locomotion and follow
the reward structure used by DreamWaQ:

    - Linear velocity tracking
    - Angular velocity tracking
    - Vertical velocity penalty
    - Base angular velocity penalty
    - Base orientation penalty
    - Joint acceleration penalty
    - Joint power penalty
    - Body height penalty
    - Foot clearance penalty
    - Action-rate penalty
    - Power-distribution penalty
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import RayCaster

def _get_terrain_height(sensor: RayCaster) -> torch.Tensor:
    """Calculate average terrain height using valid ray hits."""
    ray_hits = sensor.data.ray_hits_w
    hit_heights = ray_hits[..., 2]

    valid_hits = torch.isfinite(hit_heights)
    valid_count = valid_hits.sum(dim=1)

    terrain_height = torch.where(
        valid_hits,
        hit_heights,
        torch.zeros_like(hit_heights),
    ).sum(dim=1) / valid_count.clamp(min=1)

    return terrain_height

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Basic locomotion rewards
# -----------------------------------------------------------------------------


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Reward tracking of commanded linear velocity in the XY plane.

    DreamWaQ:
        exp(-4 * ||v_cmd_xy - v_xy||^2)
    """

    asset: Articulation = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)

    velocity_error = command[:, :2] - asset.data.root_lin_vel_b[:, :2]

    return torch.exp(
        -torch.sum(torch.square(velocity_error), dim=1) / (std**2)
    )


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Reward tracking of commanded yaw angular velocity.

    DreamWaQ:
        exp(-4 * (w_cmd - w_z)^2)
    """

    asset: Articulation = env.scene[asset_cfg.name]

    command = env.command_manager.get_command(command_name)

    velocity_error = command[:, 2] - asset.data.root_ang_vel_b[:, 2]

    return torch.exp(
        -torch.square(velocity_error) / (std**2)
    )


# -----------------------------------------------------------------------------
# Stability penalties
# -----------------------------------------------------------------------------


def lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Penalize vertical base velocity.
    """

    asset: Articulation = env.scene[asset_cfg.name]

    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Penalize roll and pitch angular velocity.
    """

    asset: Articulation = env.scene[asset_cfg.name]

    return torch.sum(
        torch.square(asset.data.root_ang_vel_b[:, :2]),
        dim=1,
    )


def orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Penalize deviation from an upright orientation.

    The projected gravity vector should ideally be:
        [0, 0, -1]

    Therefore only the X/Y components are penalized.
    """

    asset: Articulation = env.scene[asset_cfg.name]

    projected_gravity = asset.data.projected_gravity_b

    return torch.sum(
        torch.square(projected_gravity[:, :2]),
        dim=1,
    )


# -----------------------------------------------------------------------------
# Joint penalties
# -----------------------------------------------------------------------------


def joint_acceleration_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Penalize joint acceleration.

    DreamWaQ uses:
        ||q_ddot||^2
    """

    asset: Articulation = env.scene[asset_cfg.name]

    joint_acc = asset.data.joint_acc[:, asset_cfg.joint_ids]

    return torch.sum(
        torch.square(joint_acc),
        dim=1,
    )


def joint_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Penalize joint mechanical power.

    DreamWaQ:
        sum |tau_i| * |qdot_i|
    """

    asset: Articulation = env.scene[asset_cfg.name]

    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    # Applied torque is preferred when available.
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]

    power = torch.abs(torque) * torch.abs(joint_vel)

    return torch.sum(power, dim=1)


# -----------------------------------------------------------------------------
# Body height
# -----------------------------------------------------------------------------


def body_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Penalize deviation of the robot body height from a desired height.

    Terrain height is estimated from the height scanner.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # Estimate terrain height from the scanner.
    terrain_height = _get_terrain_height(sensor)

    body_height = asset.data.root_pos_w[:, 2] - terrain_height

    return torch.square(body_height - target_height)


# -----------------------------------------------------------------------------
# Foot clearance
# -----------------------------------------------------------------------------


def foot_clearance(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Penalize incorrect foot clearance during motion.

    The penalty is weighted by horizontal foot velocity, encouraging the
    robot to lift its feet sufficiently while they are moving.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    foot_pos = asset.data.body_pos_w[
        :,
        asset_cfg.body_ids,
        :,
    ]

    foot_vel = asset.data.body_lin_vel_w[
        :,
        asset_cfg.body_ids,
        :,
    ]

    terrain_height = _get_terrain_height(sensor)

    foot_height = (
        foot_pos[..., 2]
        - terrain_height.unsqueeze(1)
    )

    horizontal_velocity = torch.linalg.norm(
        foot_vel[..., :2],
        dim=-1,
    )

    clearance_error = torch.square(
        foot_height - target_height
    )

    return torch.sum(
        clearance_error * horizontal_velocity,
        dim=1,
    )


# -----------------------------------------------------------------------------
# Action smoothness
# -----------------------------------------------------------------------------


def action_rate_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Penalize changes between consecutive actions.

    DreamWaQ:
        ||a_t - a_(t-1)||^2
    """

    action = env.action_manager.action
    previous_action = env.action_manager.prev_action

    return torch.sum(
        torch.square(action - previous_action),
        dim=1,
    )
class SmoothnessL2(ManagerTermBase):
    """Penalize second-order changes in consecutive actions."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.previous_action = torch.zeros(
            env.num_envs,
            env.action_manager.total_action_dim,
            device=env.device,
        )

        self.previous_previous_action = torch.zeros_like(
            self.previous_action
        )

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action = env.action_manager.action

        second_difference = (
            action
            - 2.0 * self.previous_action
            + self.previous_previous_action
        )

        penalty = torch.sum(
            torch.square(second_difference),
            dim=-1,
        )

        self.previous_previous_action.copy_(self.previous_action)
        self.previous_action.copy_(action)

        return penalty

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)

        self.previous_action[env_ids] = 0.0
        self.previous_previous_action[env_ids] = 0.0

# -----------------------------------------------------------------------------
# Power distribution
# -----------------------------------------------------------------------------


def power_distribution(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Penalize unequal distribution of motor power.

    DreamWaQ introduces this term to avoid concentrating excessive
    mechanical power on only a subset of motors.
    """

    asset: Articulation = env.scene[asset_cfg.name]

    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]

    joint_power = torch.abs(torque) * torch.abs(joint_vel)

    power_variance = torch.var(
        joint_power,
        dim=1,
        unbiased=False,
    )

    return power_variance


# -----------------------------------------------------------------------------
# Contact / termination helpers
# -----------------------------------------------------------------------------


def feet_contact_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Penalize excessive contact forces on selected bodies.
    """

    sensor = env.scene.sensors[sensor_cfg.name]

    force_norm = torch.linalg.norm(
        sensor.data.net_forces_w[:, sensor_cfg.body_ids],
        dim=-1,
    )

    violations = torch.clamp(
        force_norm - threshold,
        min=0.0,
    )

    return torch.sum(
        violations,
        dim=1,
    )


def is_alive(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Constant reward while the robot is alive.
    """

    return (~env.termination_manager.terminated).float()


def is_terminated(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """
    Penalty when the episode terminates for a reason other than timeout.
    """

    return env.termination_manager.terminated.float()

def root_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate episodes when the robot base falls below a minimum height."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height