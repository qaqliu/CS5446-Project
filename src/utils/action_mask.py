"""Action masks for Flatland grid railway environments."""

from typing import Any

import torch
from torch import Tensor
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState


def get_action_mask(
    env: RailEnv,
    current_infos: dict[str, dict[int, Any]],
) -> dict[int, Tensor]:
    """Return one CPU boolean mask of shape (5,) per train.

    env and current_infos must describe the same environment step, after reset.
    Block invalid left/forward/right requests, including requests Flatland
    would silently replace with another movement action. Forward remains valid
    on curves and dead ends according to Flatland's movement semantics.

    Decision trains retain DO_NOTHING and STOP_MOVING. Waiting, malfunctioning,
    completed, and non-decision trains receive only DO_NOTHING. Track occupancy
    and deadlock avoidance are not checked. Observation features are not needed.
    """
    masks: dict[int, Tensor] = {}
    for agent in env.agents:
        agent_id = agent.handle
        mask = torch.zeros(5, dtype=torch.bool)
        mask[RailEnvActions.DO_NOTHING.value] = True
        masks[agent_id] = mask

        if (
            not current_infos["action_required"][agent_id]
            or current_infos["state"][agent_id] not in (
                TrainState.READY_TO_DEPART, TrainState.MOVING, TrainState.STOPPED
            )
            or current_infos["malfunction"][agent_id] > 0
        ):
            continue

        mask[RailEnvActions.STOP_MOVING.value] = True
        configuration = agent.current_configuration
        if configuration is None:
            configuration = agent.initial_configuration

        for action in (
            RailEnvActions.MOVE_LEFT,
            RailEnvActions.MOVE_FORWARD,
            RailEnvActions.MOVE_RIGHT,
        ):
            # Flatland's internal helper also exposes silent action corrections;
            # apply_action_independent alone would accept those corrections.
            cell_valid, _, transition_valid, _, action_valid = (
                env.rail._check_action_on_agent(action, configuration)
            )
            mask[action.value] = bool(cell_valid and transition_valid and action_valid)
    return masks
