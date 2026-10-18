"""Common abstract interface for value-based agents."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn

ActionMasks = dict[int, Tensor]
ActionValues = dict[int, Tensor]
Actions = dict[int, int]
Observations = dict[int, Any]
Infos = dict[str, dict[int, Any]]
AdditionalInfos = dict[str, Any]


class ValueAgent(nn.Module, ABC):
    """Compute action values and select actions with epsilon-greedy exploration.

    Subclasses configure their models and implement Q-value computation.
    Environment interaction, epsilon schedules, targets, and losses are
    maintained externally. Results are keyed by agent ID.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = None

    @abstractmethod
    def _get_q_values(
        self,
        observations: Observations,
        current_infos: Infos,
        addtional_infos: AdditionalInfos | None = None,
    ) -> ActionValues:
        """Return five floating-point Q values per agent, preserving gradients."""
        raise NotImplementedError

    def _get_action(
        self,
        q_values: ActionValues,
        action_mask: ActionMasks,
        epsilon: float = 0.0,
        deterministic: bool = False,
    ) -> Actions:
        """Select independently per agent among allowed actions.

        With probability epsilon, sample uniformly from allowed actions;
        otherwise select the first action with the maximum allowed Q value.
        Deterministic selection disables exploration. Masks must be boolean
        tensors of shape (5,), matching the Q values.
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("Epsilon must be between 0 and 1.")
        if q_values.keys() != action_mask.keys():
            raise ValueError("Q values and masks must use the same agent IDs.")

        actions: Actions = {}
        for agent_id, agent_values in q_values.items():
            agent_mask = action_mask[agent_id].to(device=agent_values.device)
            if not agent_mask.any().item():
                raise ValueError(f"Agent {agent_id} has no valid action.")
            if (
                not deterministic
                and epsilon > 0.0
                and torch.rand((), device=agent_values.device).item() < epsilon
            ):
                action = torch.multinomial(agent_mask.float(), 1)
            else:
                action = agent_values.masked_fill(~agent_mask, -torch.inf).argmax()
            actions[agent_id] = int(action.item())
        return actions

    def forward(
        self,
        observations: Observations,
        current_infos: Infos,
        addtional_infos: AdditionalInfos | None = None,
        action_mask: ActionMasks | None = None,
        epsilon: float = 0.0,
        deterministic: bool = False,
    ) -> tuple[ActionMasks, ActionValues, Actions]:
        """Return action masks, original Q values, and selected actions.

        When no mask is supplied, create an all-True mask matching Q values.
        Additional information is forwarded unchanged. Returned Q values
        retain their gradients and are not modified by action masking.
        """
        q_values = self._get_q_values(
            observations, current_infos, addtional_infos=addtional_infos
        )
        if action_mask is None:
            action_mask = {
                agent_id: torch.ones_like(agent_values, dtype=torch.bool)
                for agent_id, agent_values in q_values.items()
            }
        actions = self._get_action(
            q_values, action_mask, epsilon=epsilon, deterministic=deterministic
        )
        return action_mask, q_values, actions


class IndependentValueAgent(ValueAgent):
    """Use a separate model for each agent.

    Each model receives observations, current_infos, and the
    keyword arguments addtional_infos and agent_id. It returns five Q values
    for that agent and decides which information to read from the inputs.
    """

    def __init__(self, models: dict[int, nn.Module]) -> None:
        super().__init__()
        # ModuleDict registers all models for parameter and checkpoint management.
        self.model = nn.ModuleDict(
            {str(agent_id): model for agent_id, model in models.items()}
        )

    def _get_q_values(
        self,
        observations: Observations,
        current_infos: Infos,
        addtional_infos: AdditionalInfos | None = None,
    ) -> ActionValues:
        return {
            int(agent_id): model(
                observations,
                current_infos,
                addtional_infos=addtional_infos,
                agent_id=int(agent_id),
            )
            for agent_id, model in self.model.items()
        }


class ShareNetworkValueAgent(ValueAgent):
    """Use one shared model to compute Q values separately for each agent.

    The model receives observations, current_infos, and the
    keyword arguments addtional_infos and agent_id. It returns a five-entry
    Q values tensor for the specified agent.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def _get_q_values(
        self,
        observations: Observations,
        current_infos: Infos,
        addtional_infos: AdditionalInfos | None = None,
    ) -> ActionValues:
        return {
            agent_id: self.model(
                observations,
                current_infos,
                addtional_infos=addtional_infos,
                agent_id=agent_id,
            )
            for agent_id in observations
        }


class CentralizedValueAgent(ValueAgent):
    """Use one model to compute Q values jointly for all agents.

    The model receives observations, current_infos, and the
    keyword argument addtional_infos. It returns all agents' Q values in a single
    call. The base agent selects actions with masked epsilon-greedy exploration.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def _get_q_values(
        self,
        observations: Observations,
        current_infos: Infos,
        addtional_infos: AdditionalInfos | None = None,
    ) -> ActionValues:
        return self.model(
            observations, current_infos, addtional_infos=addtional_infos
        )
