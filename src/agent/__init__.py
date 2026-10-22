"""Actor and critic interfaces for railway agents."""

from .actor import (
    Actor, CentralizedActor, IndependentActor, ShareNetworkActor,
    CommunicationIndependentNetworkActor, CommunicationSharedNetworkActor,
)
from .critic import Critic, CentralizedCritic, IndependentCritic, ShareNetworkCritic

__all__ = [
    "Actor",
    "IndependentActor",
    "ShareNetworkActor",
    "CentralizedActor",
    "CommunicationIndependentNetworkActor",
    "CommunicationSharedNetworkActor",
    "Critic",
    "IndependentCritic",
    "ShareNetworkCritic",
    "CentralizedCritic",
]
