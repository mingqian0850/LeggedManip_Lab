# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Adapter for the public ``robot_lab`` Unitree B2-W locomotion policy.

The upstream deployment contract is pinned in ``third_party/rl_sar_b2w``.
This module deliberately contains no Isaac Lab environment logic: callers
provide tensors in the explicit joint order below and receive leg-position
and wheel-velocity targets in that same order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

ROBOT_LAB_B2W_POLICY_SHA256 = "38155076408e8eccb22690c6c5be14bd1dcb9149245ca5e493308a9f6ff93b34"
ROBOT_LAB_B2W_OBSERVATION_SIZE = 57
ROBOT_LAB_B2W_ACTION_SIZE = 16
ROBOT_LAB_B2W_LEG_ACTION_SIZE = 12
ROBOT_LAB_B2W_POLICY_PATH = (
    Path(__file__).resolve().parents[4] / "third_party" / "rl_sar_b2w" / "artifacts" / "policy.pt"
)

ROBOT_LAB_B2W_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "FR_wheel_joint",
    "FL_wheel_joint",
    "RR_wheel_joint",
    "RL_wheel_joint",
)

_DEFAULT_JOINT_POSITION = (
    0.0,
    0.8,
    -1.5,
    0.0,
    0.8,
    -1.5,
    0.0,
    0.8,
    -1.5,
    0.0,
    0.8,
    -1.5,
    0.0,
    0.0,
    0.0,
    0.0,
)
_ACTION_SCALE = (
    0.125,
    0.25,
    0.25,
    0.125,
    0.25,
    0.25,
    0.125,
    0.25,
    0.25,
    0.125,
    0.25,
    0.25,
    5.0,
    5.0,
    5.0,
    5.0,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RobotLabB2WAction:
    """Decoded output of the upstream policy."""

    raw: torch.Tensor
    leg_position_targets: torch.Tensor
    wheel_velocity_targets: torch.Tensor


class RobotLabB2WPolicy:
    """Run and decode the pinned 57-observation to 16-action policy."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        num_envs: int,
        device: str | torch.device,
        runtime_checks: bool = True,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")

        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"B2-W checkpoint not found: {self.checkpoint}. Run third_party/rl_sar_b2w/fetch.sh first."
            )
        actual_sha256 = _sha256(self.checkpoint)
        if actual_sha256 != ROBOT_LAB_B2W_POLICY_SHA256:
            raise ValueError(
                f"B2-W checkpoint SHA-256 mismatch: expected {ROBOT_LAB_B2W_POLICY_SHA256}, got {actual_sha256}"
            )

        self.device = torch.device(device)
        self.num_envs = num_envs
        self.runtime_checks = runtime_checks
        self.actor = torch.jit.load(str(self.checkpoint), map_location=self.device).eval()
        self.default_joint_position = torch.tensor(
            _DEFAULT_JOINT_POSITION, device=self.device, dtype=torch.float32
        ).unsqueeze(0)
        self.action_scale = torch.tensor(_ACTION_SCALE, device=self.device, dtype=torch.float32).unsqueeze(0)
        self.last_action = torch.zeros((num_envs, ROBOT_LAB_B2W_ACTION_SIZE), device=self.device, dtype=torch.float32)

        with torch.inference_mode():
            probe = self.actor(
                torch.zeros(
                    (1, ROBOT_LAB_B2W_OBSERVATION_SIZE),
                    device=self.device,
                    dtype=torch.float32,
                )
            )
        if not isinstance(probe, torch.Tensor) or probe.shape != (1, ROBOT_LAB_B2W_ACTION_SIZE):
            raise ValueError(f"unexpected policy output for a 57-D probe: {type(probe)=}, {probe.shape=}")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear recurrent action history for all or selected environments."""
        if env_ids is None:
            self.last_action.zero_()
        else:
            self.last_action[env_ids] = 0.0

    def _check_input(self, name: str, value: torch.Tensor, width: int) -> torch.Tensor:
        if value.shape != (self.num_envs, width):
            raise ValueError(f"{name} must have shape {(self.num_envs, width)}, got {tuple(value.shape)}")
        return value.to(device=self.device, dtype=torch.float32)

    def observation(
        self,
        *,
        base_angular_velocity_b: torch.Tensor,
        projected_gravity_b: torch.Tensor,
        velocity_command_b: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Build the exact noise-free actor observation used during playback."""
        base_angular_velocity_b = self._check_input("base_angular_velocity_b", base_angular_velocity_b, 3)
        projected_gravity_b = self._check_input("projected_gravity_b", projected_gravity_b, 3)
        velocity_command_b = self._check_input("velocity_command_b", velocity_command_b, 3)
        joint_position = self._check_input("joint_position", joint_position, ROBOT_LAB_B2W_ACTION_SIZE)
        joint_velocity = self._check_input("joint_velocity", joint_velocity, ROBOT_LAB_B2W_ACTION_SIZE)

        joint_position_relative = joint_position - self.default_joint_position
        joint_position_relative = joint_position_relative.clone()
        joint_position_relative[:, ROBOT_LAB_B2W_LEG_ACTION_SIZE:] = 0.0
        observation = torch.cat(
            (
                0.25 * base_angular_velocity_b,
                projected_gravity_b,
                velocity_command_b,
                joint_position_relative,
                0.05 * joint_velocity,
                self.last_action,
            ),
            dim=-1,
        )
        observation = torch.clamp(observation, -100.0, 100.0)
        if observation.shape != (self.num_envs, ROBOT_LAB_B2W_OBSERVATION_SIZE):
            raise AssertionError(f"internal observation shape error: {tuple(observation.shape)}")
        return observation

    def act(
        self,
        *,
        base_angular_velocity_b: torch.Tensor,
        projected_gravity_b: torch.Tensor,
        velocity_command_b: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
    ) -> RobotLabB2WAction:
        """Evaluate the actor and convert normalized output to actuator targets."""
        observation = self.observation(
            base_angular_velocity_b=base_angular_velocity_b,
            projected_gravity_b=projected_gravity_b,
            velocity_command_b=velocity_command_b,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
        )
        with torch.inference_mode():
            raw_action = self.actor(observation)
        if raw_action.shape != (self.num_envs, ROBOT_LAB_B2W_ACTION_SIZE):
            raise ValueError(f"unexpected policy output shape: {tuple(raw_action.shape)}")
        if self.runtime_checks and not torch.all(torch.isfinite(raw_action)):
            raise FloatingPointError("policy produced a non-finite action")

        raw_action = torch.clamp(raw_action, -100.0, 100.0)
        self.last_action.copy_(raw_action)
        scaled_action = raw_action * self.action_scale
        leg_position_targets = (
            self.default_joint_position[:, :ROBOT_LAB_B2W_LEG_ACTION_SIZE]
            + scaled_action[:, :ROBOT_LAB_B2W_LEG_ACTION_SIZE]
        )
        wheel_velocity_targets = scaled_action[:, ROBOT_LAB_B2W_LEG_ACTION_SIZE:]
        return RobotLabB2WAction(
            raw=raw_action,
            leg_position_targets=leg_position_targets,
            wheel_velocity_targets=wheel_velocity_targets,
        )
