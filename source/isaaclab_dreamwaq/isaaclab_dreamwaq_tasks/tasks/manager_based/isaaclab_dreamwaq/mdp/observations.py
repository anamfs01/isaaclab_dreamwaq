# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Custom observation terms used by the DreamWaQ environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _safe_obs_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf values before sending observations to the policy."""
    return torch.nan_to_num(
        tensor,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


class CustomProprioceptiveObservations(ManagerTermBase):
    """
    45-D proprioceptive observation used by DreamWaQ.

    Observation layout:
        12 - joint position error
        12 - joint velocity
         3 - base angular velocity
         3 - projected gravity
         3 - commanded base velocity
        12 - previous action

    Total:
        45 dimensions
    """

    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRLEnv,
        num_action: int = 12,
    ):
        super().__init__(cfg, env)

        self.num_action = num_action

        self.asset: Articulation = env.scene[
            cfg.params["asset_cfg"].name
        ]

        self.asset_cfg = cfg.params["asset_cfg"]
        self.sensor_cfg = cfg.params["sensor_cfg"]

        self._prev_action = torch.zeros(
            self.num_envs,
            self.num_action,
            device=self.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the stored previous actions."""

        if env_ids is None:
            self._prev_action.zero_()
        else:
            self._prev_action[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:

        commands = env.command_manager.get_command(
            "base_velocity"
        )

        actions = env.action_manager.get_term(
            "joint_pos"
        ).raw_actions

        observations = torch.cat(
            (
                # 12 - joint position error
                self.asset.data.joint_pos
                - self.asset.data.default_joint_pos,

                # 12 - joint velocity
                self.asset.data.joint_vel,

                # 3 - base angular velocity
                self.asset.data.root_ang_vel_b,

                # 3 - projected gravity
                self.asset.data.projected_gravity_b,

                # 3 - commanded base velocity
                commands,

                # 12 - previous action
                self._prev_action,
            ),
            dim=-1,
        )

        # Store the current action for the next observation.
        self._prev_action = actions.clone()

        return _safe_obs_tensor(observations)


class CustomPrivilegedObservations(ManagerTermBase):
    """
    Privileged observation used by the DreamWaQ critic.

    Layout:
        45 - proprioception
        N  - terrain height measurements
         3 - true base linear velocity

    For the intended DreamWaQ terrain scanner:
        45 + 187 + 3 = 235 dimensions.
    """

    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRLEnv,
        num_action: int = 12,
    ):
        super().__init__(cfg, env)

        self.num_action = num_action

        self.asset: Articulation = env.scene[
            cfg.params["asset_cfg"].name
        ]

        self.ray_sensor: RayCaster = env.scene.sensors[
            "height_scanner"
        ]

        self.asset_cfg = cfg.params["asset_cfg"]
        self.sensor_cfg = cfg.params["sensor_cfg"]

        self._prev_action = torch.zeros(
            self.num_envs,
            self.num_action,
            device=self.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the stored previous actions."""

        if env_ids is None:
            self._prev_action.zero_()
        else:
            self._prev_action[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:

        commands = env.command_manager.get_command(
            "base_velocity"
        )

        actions = env.action_manager.get_term(
            "joint_pos"
        ).raw_actions

        # ---------------------------------------------------------
        # Proprioceptive part: 45 dimensions
        # ---------------------------------------------------------

        proprio = torch.cat(
            (
                # 12 - joint position error
                self.asset.data.joint_pos
                - self.asset.data.default_joint_pos,

                # 12 - joint velocity
                self.asset.data.joint_vel,

                # 3 - base angular velocity
                self.asset.data.root_ang_vel_b,

                # 3 - projected gravity
                self.asset.data.projected_gravity_b,

                # 3 - commanded base velocity
                commands,

                # 12 - previous action
                self._prev_action,
            ),
            dim=-1,
        )

        # ---------------------------------------------------------
        # Terrain information
        # ---------------------------------------------------------

        heights = self._get_heights()

        # ---------------------------------------------------------
        # Privileged observation
        # ---------------------------------------------------------

        privileged = torch.cat(
            (
                # 45 - proprioception
                proprio,

                # terrain height scan
                heights,

                # 3 - true base linear velocity
                self.asset.data.root_lin_vel_b,
            ),
            dim=-1,
        )

        # Store current action for the next timestep.
        self._prev_action = actions.clone()

        return _safe_obs_tensor(privileged)

    def _get_heights(self) -> torch.Tensor:
        """Return clipped terrain height measurements."""

        heights = torch.clip(
            self.ray_sensor.data.pos_w[:, 2].unsqueeze(1)
            - self.ray_sensor.data.ray_hits_w[..., 2]
            - 0.3,
            -1.0,
            1.0,
        )

        return _safe_obs_tensor(heights)


class CustomProprioceptionHistory(ManagerTermBase):
    """
    Maintains the last five frames of DreamWaQ proprioception.

    Each frame contains 45 dimensions:

        12 - joint position error
        12 - joint velocity
         3 - base angular velocity
         3 - projected gravity
         3 - commanded base velocity
        12 - previous action

    History:

        5 frames × 45 dimensions = 225 dimensions

    This history is used as the input to the DreamWaQ
    adaptation/history encoder.
    """

    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRLEnv,
        history_length: int = 5,
        num_prop: int = 45,
        num_action: int = 12,
    ):
        super().__init__(cfg, env)

        self.history_length = history_length
        self.num_prop = num_prop
        self.num_action = num_action

        self.asset: Articulation = env.scene[
            cfg.params["asset_cfg"].name
        ]

        self.asset_cfg = cfg.params["asset_cfg"]
        self.sensor_cfg = cfg.params["sensor_cfg"]

        # ---------------------------------------------------------
        # History buffer
        #
        # Shape:
        # [num_envs, 5, 45]
        # ---------------------------------------------------------

        self._obs_history_buffer = torch.zeros(
            self.num_envs,
            self.history_length,
            self.num_prop,
            device=self.device,
        )

        # Previous action used as part of the 45-D observation.
        self._prev_action = torch.zeros(
            self.num_envs,
            self.num_action,
            device=self.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset history and previous actions."""

        if env_ids is None:
            self._obs_history_buffer.zero_()
            self._prev_action.zero_()
        else:
            self._obs_history_buffer[env_ids] = 0.0
            self._prev_action[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        num_hist: int = 5,
    ) -> torch.Tensor:

        commands = env.command_manager.get_command(
            "base_velocity"
        )

        actions = env.action_manager.get_term(
            "joint_pos"
        ).raw_actions

        # ---------------------------------------------------------
        # Build current 45-D proprioceptive observation
        # ---------------------------------------------------------

        obs = torch.cat(
            (
                # 12 - joint position error
                self.asset.data.joint_pos
                - self.asset.data.default_joint_pos,

                # 12 - joint velocity
                self.asset.data.joint_vel,

                # 3 - base angular velocity
                self.asset.data.root_ang_vel_b,

                # 3 - projected gravity
                self.asset.data.projected_gravity_b,

                # 3 - commanded base velocity
                commands,

                # 12 - previous action
                self._prev_action,
            ),
            dim=-1,
        )

        # ---------------------------------------------------------
        # Update history
        # ---------------------------------------------------------

        first_step = env.episode_length_buf <= 1

        new_history = torch.cat(
            (
                self._obs_history_buffer[:, 1:],
                obs.unsqueeze(1),
            ),
            dim=1,
        )

        # At the beginning of an episode, initialize every
        # history frame with the current observation instead of
        # leaving old/zero history.
        self._obs_history_buffer = torch.where(
            first_step[:, None, None],
            obs.unsqueeze(1).expand(
                -1,
                self.history_length,
                -1,
            ),
            new_history,
        )

        # Store current action for the next timestep.
        self._prev_action = actions.clone()

        # ---------------------------------------------------------
        # Return requested number of history frames
        # ---------------------------------------------------------

        history = self._obs_history_buffer[:, -num_hist:]

        return _safe_obs_tensor(
            history.reshape(self.num_envs, -1)
        )