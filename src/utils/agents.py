"""Build policy agents from experiment configuration."""

from src.agent import (
    Actor, CentralizedActor, IndependentActor, ShareNetworkActor,
    CommunicationIndependentNetworkActor, CommunicationSharedNetworkActor,
    Critic, CentralizedCritic, IndependentCritic, ShareNetworkCritic,
)
from src.model import build_model


def build_agents(config: dict) -> tuple[Actor, Critic]:
    """Build actor and critic wrappers with separately instantiated models.

    Independent agents get one model per train; shared agents reuse one model.
    Centralized agents require models implementing the joint-input interface.
    """
    actor_config = config["actor"]
    if actor_config["type"] == "independent":
        actor = IndependentActor({
            agent_id: build_model(**actor_config["model"])
            for agent_id in range(config["number_of_agents"])
        })
    elif actor_config["type"] == "shared":
        actor = ShareNetworkActor(build_model(**actor_config["model"]))
    elif actor_config["type"] == "communication_independent":
        actor = CommunicationIndependentNetworkActor({
            agent_id: build_model(**actor_config["model"])
            for agent_id in range(config["number_of_agents"])
        })
    elif actor_config["type"] == "communication_shared":
        actor = CommunicationSharedNetworkActor(build_model(**actor_config["model"]))
    elif actor_config["type"] == "centralized":
        actor = CentralizedActor(build_model(**actor_config["model"]))
    else:
        raise ValueError(f"Unknown actor type: {actor_config['type']!r}")

    critic_config = config["critic"]
    if critic_config["type"] == "independent":
        critic = IndependentCritic({
            agent_id: build_model(**critic_config["model"])
            for agent_id in range(config["number_of_agents"])
        })
    elif critic_config["type"] == "shared":
        critic = ShareNetworkCritic(build_model(**critic_config["model"]))
    elif critic_config["type"] == "centralized":
        critic = CentralizedCritic(
            build_model(**critic_config["model"]),
            team_value=critic_config.get("team_value", False),
        )
    else:
        raise ValueError(f"Unknown critic type: {critic_config['type']!r}")
    return actor, critic
