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
from isaaclab.sensors import ContactSensor, RayCaster

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
    Standard proprioceptive observations used by DreamWaQ.
    """

    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRLEnv,
        num_action: int = 12,
    ):
        super().__init__(cfg, env)

        self.num_action = num_action

        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors["contact_forces"]

        self.asset_cfg = cfg.params["asset_cfg"]
        self.sensor_cfg = cfg.params["sensor_cfg"]

        self._prev_action = torch.zeros(
            self.num_envs,
            self.num_action,
            device=self.device,
        )

        self.foot_ids, _ = self.contact_sensor.find_bodies(
            ".*_foot",
            preserve_order=True,
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        self._prev_action[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:

        forces_w = self.contact_sensor.data.net_forces_w
        foot_forces = forces_w[:, self.foot_ids, :]

        foot_contact = (
            foot_forces.norm(dim=-1) > 1.0
        ).float()

        commands = env.command_manager.get_command("base_velocity")

        actions = env.action_manager.get_term(
            "joint_pos"
        ).raw_actions

        observations = torch.cat(
            (
                self.asset.data.joint_pos
                - self.asset.data.default_joint_pos,

                self.asset.data.joint_vel,

                self.asset.data.root_ang_vel_b,

                self.asset.data.projected_gravity_b,

                commands,

                self._prev_action,

                foot_contact,
            ),
            dim=-1,
        )

        self._prev_action = actions

        return _safe_obs_tensor(observations)
    
class CustomPrivilegedObservations(ManagerTermBase):
    def __init__(
          self,
          cfg: ObservationTermCfg,
          env: ManagerBasedRLEnv,
          num_action: int = 12,
    ):
        super().__init__(cfg, env)

        self.num_action = num_action

        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors["contact_forces"]
        self.ray_sensor: RayCaster = env.scene.sensors["height_scanner"]

        self.asset_cfg = cfg.params["asset_cfg"]
        self.sensor_cfg = cfg.params["sensor_cfg"]

        self._prev_action = torch.zeros(
            self.num_envs,
            self.num_action,
            device=self.device,
        )

        self.foot_ids, _ = self.contact_sensor.find_bodies(
            ".*_foot",
            preserve_order=True,
        )

    def reset(self, env_ids=None):
        self._prev_action[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:

        forces_w = self.contact_sensor.data.net_forces_w
        foot_forces = forces_w[:, self.foot_ids, :]
        foot_contact = (foot_forces.norm(dim=-1) > 1.0).float()

        commands = env.command_manager.get_command("base_velocity")
        actions = env.action_manager.get_term("joint_pos").raw_actions

        proprio = torch.cat(
            (
                self.asset.data.joint_pos - self.asset.data.default_joint_pos,
                self.asset.data.joint_vel,
                self.asset.data.root_ang_vel_b,
                self.asset.data.projected_gravity_b,
                commands,
                self._prev_action,
                foot_contact,
            ),
            dim=-1,
        )

        heights = self._get_heights()

        privileged = torch.cat(
            (
                proprio,
                heights,
                self.asset.data.root_lin_vel_b,
            ),
            dim=-1,
        )

        self._prev_action = actions

        return _safe_obs_tensor(privileged)

    def _get_heights(self):

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
    Maintains a rolling history of proprioceptive observations.
    Used by DreamWaQ's adaptation module.
    """

    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedRLEnv,
        history_length: int = 30,
        num_prop: int = 49,
        num_action: int = 12,
    ):
        super().__init__(cfg, env)

        self.history_length = history_length
        self.num_prop = num_prop
        self.num_action = num_action

        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.contact_sensor: ContactSensor = env.scene.sensors["contact_forces"]

        self.asset_cfg = cfg.params["asset_cfg"]
        self.sensor_cfg = cfg.params["sensor_cfg"]

        self._obs_history_buffer = torch.zeros(
            self.num_envs,
            self.history_length,
            self.num_prop,
            device=self.device,
        )

        self._prev_action = torch.zeros(
            self.num_envs,
            self.num_action,
            device=self.device,
        )

        self.foot_ids, _ = self.contact_sensor.find_bodies(
            ".*_foot",
            preserve_order=True,
        )

    def reset(self, env_ids: Sequence[int] | None = None):
        self._obs_history_buffer[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        num_hist: int = 5,
    ) -> torch.Tensor:

        forces_w = self.contact_sensor.data.net_forces_w
        foot_forces = forces_w[:, self.foot_ids, :]
        foot_contact = (foot_forces.norm(dim=-1) > 1.0).float()

        commands = env.command_manager.get_command("base_velocity")
        actions = env.action_manager.get_term("joint_pos").raw_actions

        obs = torch.cat(
            (
                self.asset.data.joint_pos - self.asset.data.default_joint_pos,
                self.asset.data.joint_vel,
                self.asset.data.root_ang_vel_b,
                self.asset.data.projected_gravity_b,
                commands,
                self._prev_action,
                foot_contact,
            ),
            dim=-1,
        )

        self._obs_history_buffer = torch.where(
            (env.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs] * self.history_length, dim=1),
            torch.cat(
                (
                    self._obs_history_buffer[:, 1:],
                    obs.unsqueeze(1),
                ),
                dim=1,
            ),
        )

        self._prev_action = actions

        return _safe_obs_tensor(
            self._obs_history_buffer[:, -num_hist:]
        )