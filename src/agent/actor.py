"""Common abstract interface for actors."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn
from torchrl.modules import MaskedCategorical

ActionMasks = dict[int, Tensor]
ActionLogits = dict[int, Tensor]
Actions = dict[int, int]
Observations = dict[int, Any]
Infos = dict[str, dict[int, Any]]
AdditionalInfos = dict[str, Any]


class Actor(nn.Module, ABC):
    """Read environment information and select actions.

    Subclasses implement the decision process and configure independent,
    shared, or centralized models. Results are keyed by agent ID. An external
    collector reads, resets, and steps the environment.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = None

    @abstractmethod
    def _get_logits(
        self,
        observations: Observations,
        current_infos: Infos,
        addtional_infos: AdditionalInfos | None = None,
    ) -> ActionLogits:
        """Return five action logits per agent, preserving gradients for training."""
        raise NotImplementedError

    def get_logits_batch(
        self, observations: list[Observations], current_infos: list[Infos],
    ) -> ActionLogits:
        """Return [time, action] logits per agent without selecting new actions.

        Default models retain their full observation/info dictionaries and their
        original single-state calls. Models can opt in to batched local inputs
        by implementing forward_batch, as the demo MLP does.
        """
        outputs = [
            self._get_logits(obs, infos)
            for obs, infos in zip(observations, current_infos)
        ]
        return {i: torch.stack([out[i] for out in outputs]) for i in outputs[0]}

    def _get_action(
        self,
        logits: ActionLogits,
        action_mask: ActionMasks,
        deterministic: bool = False,
    ) -> Actions:
        """Apply action masks and select actions.

        Sample from the action distribution unless deterministic is True,
        in which case select the most probable valid action. Return a mapping
        from agent IDs to Flatland action IDs.
        """
        if logits.keys() != action_mask.keys():
            raise ValueError("Logits and masks must use the same agent IDs.")

        actions: Actions = {}
        for agent_id, agent_logits in logits.items():
            agent_mask = action_mask[agent_id].to(device=agent_logits.device)
            if not agent_mask.any().item():
                raise ValueError(f"Agent {agent_id} has no valid action.")
            distribution = MaskedCategorical(logits=agent_logits, mask=agent_mask)
            action = distribution.mode if deterministic else distribution.sample()
            actions[agent_id] = int(action.item())
        return actions

    def forward(
        self,
        observations: Observations,
        current_infos: Infos,
        addtional_infos: AdditionalInfos | None = None,
        action_mask: ActionMasks | None = None,
        deterministic: bool = False,
    ) -> tuple[ActionMasks, ActionLogits, Actions]:
        """Return action masks, logits, and actions.

        When no mask is supplied, create an all-True mask matching the logits.
        Optional additional information is forwarded unchanged to the model.
        Masks, logits, and actions must use the same agent IDs. The caller
        performs environment interaction and maintains additional information.
        """
        logits = self._get_logits(
            observations, current_infos, addtional_infos=addtional_infos
        )
        if action_mask is None:
            action_mask = {
                agent_id: torch.ones_like(agent_logits, dtype=torch.bool)
                for agent_id, agent_logits in logits.items()
            }
        actions = self._get_action(logits, action_mask, deterministic=deterministic)
        return action_mask, logits, actions


class IndependentActor(Actor):
    """Use a separate model for each agent.

    Each model receives observations, current_infos, and the
    keyword arguments addtional_infos and agent_id. It returns five logits
    for that agent and decides which information to read from the inputs.
    """

    def __init__(self, models: dict[int, nn.Module]) -> None:
        super().__init__()
        # ModuleDict registers all models for parameter and checkpoint management.
        self.model = nn.ModuleDict(
            {str(agent_id): model for agent_id, model in models.items()}
        )

    def _get_logits(
        self,
        observations: Observations,
        current_infos: Infos,
        addtional_infos: AdditionalInfos | None = None,
    ) -> ActionLogits:
        return {
            int(agent_id): model(
                observations,
                current_infos,
                addtional_infos=addtional_infos,
                agent_id=int(agent_id),
            )
            for agent_id, model in self.model.items()
        }


    def get_logits_batch(
        self, observations: list[Observations], current_infos: list[Infos],
    ) -> ActionLogits:
        if not all(callable(getattr(model, "forward_batch", None)) for model in self.model.values()):
            return super().get_logits_batch(observations, current_infos)
        return {
            int(i): model.forward_batch([obs[int(i)] for obs in observations])
            for i, model in self.model.items()
        }


class ShareNetworkActor(Actor):
    """Use one shared model to compute logits separately for each agent.

    The model receives observations, current_infos, and the
    keyword arguments addtional_infos and agent_id. It returns a five-entry
    logits tensor for the specified agent.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def _get_logits(
        self,
        observations: Observations,
        current_infos: Infos,
        addtional_infos: AdditionalInfos | None = None,
    ) -> ActionLogits:
        return {
            agent_id: self.model(
                observations,
                current_infos,
                addtional_infos=addtional_infos,
                agent_id=agent_id,
            )
            for agent_id in observations
        }


    def get_logits_batch(
        self, observations: list[Observations], current_infos: list[Infos],
    ) -> ActionLogits:
        if not callable(getattr(self.model, "forward_batch", None)):
            return super().get_logits_batch(observations, current_infos)
        agent_ids = list(observations[0])
        logits = self.model.forward_batch([
            obs[i] for obs in observations for i in agent_ids
        ]).reshape(len(observations), len(agent_ids), -1)
        return {i: logits[:, col] for col, i in enumerate(agent_ids)}


class CentralizedActor(Actor):
    """Use one model to compute logits jointly for all agents.

    The model receives observations, current_infos, and the
    keyword argument addtional_infos. It returns all agents' logits in a single
    call. The base actor selects actions from the masked distributions.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def _get_logits(
        self,
        observations: Observations,
        current_infos: Infos,
        addtional_infos: AdditionalInfos | None = None,
    ) -> ActionLogits:
        return self.model(
            observations, current_infos, addtional_infos=addtional_infos
        )
