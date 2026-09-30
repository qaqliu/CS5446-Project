"""Common abstract interface for state-value critics."""

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn


Observations = dict[int, Any]
Infos = dict[str, dict[int, Any]]
AdditionalInfos = dict[str, Any]
AgentValues = dict[int, Tensor]
CriticValues = AgentValues | Tensor


class Critic(nn.Module, ABC):
    """Estimate state values from the supplied inputs.

    Subclasses configure independent, shared, or centralized models. Per-agent
    values are keyed by agent ID, with a tensor of shape (1,) for each agent.
    A centralized critic may instead return one tensor of shape (1,) for the
    team value. Return values preserve gradients for training.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = None

    @abstractmethod
    def _get_values(
        self,
        observations: Observations,
        current_infos: Infos,
        additional_infos: AdditionalInfos | None = None,
    ) -> CriticValues:
        """Return state-value estimates using the configured models."""
        raise NotImplementedError

    def forward(
        self,
        observations: Observations,
        current_infos: Infos,
        additional_infos: AdditionalInfos | None = None,
    ) -> CriticValues:
        """Forward the supplied inputs to the value computation.

        The caller maintains additional information and historical inputs.
        Advantage estimation, value targets, and losses are handled separately.
        """
        return self._get_values(
            observations, current_infos, additional_infos=additional_infos
        )


class IndependentCritic(Critic):
    """Use a separate state-value model for each agent.

    Each model receives observations, current_infos, and the keyword arguments
    additional_infos and agent_id. It returns a value tensor of shape (1,).
    """

    def __init__(self, models: dict[int, nn.Module]) -> None:
        super().__init__()
        self.model = nn.ModuleDict(
            {str(agent_id): model for agent_id, model in models.items()}
        )

    def _get_values(
        self,
        observations: Observations,
        current_infos: Infos,
        additional_infos: AdditionalInfos | None = None,
    ) -> AgentValues:
        return {
            int(agent_id): model(
                observations,
                current_infos,
                additional_infos=additional_infos,
                agent_id=int(agent_id),
            )
            for agent_id, model in self.model.items()
        }


class ShareNetworkCritic(Critic):
    """Use one shared model to estimate each agent's state value separately.

    The model receives observations, current_infos, and the keyword arguments
    additional_infos and agent_id. It returns a value tensor of shape (1,).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def _get_values(
        self,
        observations: Observations,
        current_infos: Infos,
        additional_infos: AdditionalInfos | None = None,
    ) -> AgentValues:
        return {
            agent_id: self.model(
                observations,
                current_infos,
                additional_infos=additional_infos,
                agent_id=agent_id,
            )
            for agent_id in observations
        }


class CentralizedCritic(Critic):
    """Estimate values for all agents or one team value in a single model call.

    The model receives observations, current_infos, and the keyword arguments
    additional_infos and team_value.
    Its output must match team_value: a per-agent value dictionary when
    False, or one team-value tensor of shape (1,) when True.
    """

    def __init__(self, model: nn.Module, team_value: bool = False) -> None:
        super().__init__()
        self.model = model
        self.team_value = team_value

    def _get_values(
        self,
        observations: Observations,
        current_infos: Infos,
        additional_infos: AdditionalInfos | None = None,
    ) -> CriticValues:
        values = self.model(
            observations,
            current_infos,
            additional_infos=additional_infos,
            team_value=self.team_value,
        )
        if self.team_value:
            if not isinstance(values, Tensor) or values.shape != (1,):
                raise ValueError(
                    "A team-value model must return a tensor of shape (1,)."
                )
        elif not isinstance(values, dict):
            raise ValueError("A per-agent value model must return a dictionary.")
        return values
