"""Action helpers that do not add trainable dimensions to unified EE-WBC."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class FixedJointPositionAction(ActionTerm):
    """Hold selected joints at their reset default without consuming actor output."""

    cfg: FixedJointPositionActionCfg

    def __init__(self, cfg: FixedJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = self._asset
        self.joint_ids, self.joint_names = self.robot.find_joints(cfg.joint_names, preserve_order=True)
        if len(self.joint_ids) == 0:
            raise ValueError(f"No joints matched fixed hold patterns: {cfg.joint_names}")
        self.targets = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
        self._raw_actions = torch.empty((self.num_envs, 0), device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"Expected zero-width hold action, received {tuple(actions.shape)}")

    def apply_actions(self) -> None:
        self.robot.set_joint_position_target(self.targets, joint_ids=self.joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.targets[env_ids] = self.robot.data.default_joint_pos[env_ids][:, self.joint_ids]


@configclass
class FixedJointPositionActionCfg(ActionTermCfg):
    class_type: type = FixedJointPositionAction
    joint_names: list[str] = MISSING
