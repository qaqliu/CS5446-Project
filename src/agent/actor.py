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
        addtional_infos: list[AdditionalInfos | None] | None = None,
    ) -> ActionLogits:
        """Return [time, action] logits per agent without selecting new actions.

        Default models retain their full observation/info dictionaries and their
        original single-state calls. Models can opt in to batched local inputs
        by implementing forward_batch, as the demo MLP does.
        """
        extras = addtional_infos if addtional_infos is not None else [None] * len(observations)
        outputs = [
            self._get_logits(obs, infos, addtional_infos=extra)
            for obs, infos, extra in zip(observations, current_infos, extras)
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
        *,
        attention_mask: Tensor | None = None,
    ) -> tuple[ActionMasks, ActionLogits, Actions]:
        """Return action masks, logits, and actions.

        When no mask is supplied, create an all-True mask matching the logits.
        Optional additional information is forwarded to the model. Communication
        actors also accept attention_mask: True means communication is allowed.
        Masks, logits, and actions must use the same agent IDs. The caller
        performs environment interaction and maintains additional information.
        """
        if attention_mask is not None:
            addtional_infos = dict(addtional_infos or {}, attention_mask=attention_mask)
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
        addtional_infos: list[AdditionalInfos | None] | None = None,
    ) -> ActionLogits:
        if addtional_infos is not None:
            return super().get_logits_batch(observations, current_infos, addtional_infos)
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
        addtional_infos: list[AdditionalInfos | None] | None = None,
    ) -> ActionLogits:
        if addtional_infos is not None:
            return super().get_logits_batch(observations, current_infos, addtional_infos)
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


class CommunicationIndependentNetworkActor(Actor):
    """One communication model per train, each reading all observations.

    Model interface: forward(observations, attention_mask, agent_id=None).
    With agent_id set, return that train's [5] logits. With None, the model
    contract is to return a dict of [5] logits keyed by all observation IDs.
    This wrapper calls each train's own model with its agent_id explicitly.
    attention_mask is [N, N], follows list(observations), and defaults to all
    True (including self). True means allowed; models using nn.MultiheadAttention
    must invert it for that API's boolean attn_mask. Models handle input devices.
    """

    def __init__(self, models: dict[int, nn.Module]) -> None:
        super().__init__()
        self.model = nn.ModuleDict({str(i): model for i, model in models.items()})

    def _communication_attention_mask(self, observations, addtional_infos):
        """Rows/columns follow list(observations); True permits attending to a train."""
        mask = (addtional_infos or {}).get("attention_mask")
        n = len(observations)
        if mask is None:
            return torch.ones(n, n, dtype=torch.bool)
        if mask.shape != (n, n) or mask.dtype != torch.bool:
            raise ValueError("attention_mask must be a boolean [num_agents, num_agents] tensor.")
        return mask

    def _get_logits(
        self,
        observations: Observations,
        current_infos: Infos,
        addtional_infos: AdditionalInfos | None = None,
    ) -> ActionLogits:
        attention_mask = self._communication_attention_mask(observations, addtional_infos)
        return {
            int(i): model(observations, attention_mask, agent_id=int(i))
            for i, model in self.model.items()
        }


    def get_logits_batch(
        self, observations: list[Observations], current_infos: list[Infos],
        addtional_infos: list[AdditionalInfos | None] | None = None,
    ) -> ActionLogits:
        """Batch time independently within each train's communication network."""
        if (
            not all(callable(getattr(model, "forward_batch", None)) for model in self.model.values())
            or any(list(obs) != list(observations[0]) for obs in observations)
        ):
            return super().get_logits_batch(observations, current_infos, addtional_infos)
        extras = addtional_infos if addtional_infos is not None else [None] * len(observations)
        masks = torch.stack([
            self._communication_attention_mask(obs, extra)
            for obs, extra in zip(observations, extras)
        ])
        return {
            int(i): model.forward_batch(observations, masks, agent_id=int(i))
            for i, model in self.model.items()
        }

class CommunicationSharedNetworkActor(Actor):
    """One shared communication model computes all trains' logits in one call.

    Same model/mask contract as CommunicationIndependentNetworkActor. Calling
    the model with agent_id=None returns {agent_id: [5] logits} for every train.
    get_logits_batch retains whole-agent observations at each time step, so PPO
    updates can backpropagate through communication without flattening trains.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def _communication_attention_mask(self, observations, addtional_infos):
        """Rows/columns follow list(observations); True permits attending to a train."""
        mask = (addtional_infos or {}).get("attention_mask")
        n = len(observations)
        if mask is None:
            return torch.ones(n, n, dtype=torch.bool)
        if mask.shape != (n, n) or mask.dtype != torch.bool:
            raise ValueError("attention_mask must be a boolean [num_agents, num_agents] tensor.")
        return mask

    def _get_logits(
        self,
        observations: Observations,
        current_infos: Infos,
        addtional_infos: AdditionalInfos | None = None,
    ) -> ActionLogits:
        attention_mask = self._communication_attention_mask(observations, addtional_infos)
        return self.model(observations, attention_mask, agent_id=None)


    def get_logits_batch(
        self, observations: list[Observations], current_infos: list[Infos],
        addtional_infos: list[AdditionalInfos | None] | None = None,
    ) -> ActionLogits:
        """One forward for all time steps; each keeps its own communication mask."""
        if (
            not callable(getattr(self.model, "forward_batch", None))
            or any(list(obs) != list(observations[0]) for obs in observations)
        ):
            return super().get_logits_batch(observations, current_infos, addtional_infos)
        extras = addtional_infos if addtional_infos is not None else [None] * len(observations)
        masks = torch.stack([
            self._communication_attention_mask(obs, extra)
            for obs, extra in zip(observations, extras)
        ])
        return self.model.forward_batch(observations, masks, agent_id=None)
