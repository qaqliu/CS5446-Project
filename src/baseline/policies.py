"""Non-learning reference policies.

Every policy is a callable (env, observations, infos, masks, decisions) ->
actions, so learned policies (src.eval.runner.ActorPolicy) and baselines
share the same evaluation loop.
"""

from typing import Protocol

import numpy as np
import torch
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState


Actions = dict[int, int]
ActionMasks = dict[int, torch.Tensor]
DecisionFlags = dict[int, bool]
MOVING_ACTIONS = (
    RailEnvActions.MOVE_LEFT,
    RailEnvActions.MOVE_FORWARD,
    RailEnvActions.MOVE_RIGHT,
)


class Policy(Protocol):
    def __call__(
        self,
        env: RailEnv,
        observations: dict,
        infos: dict,
        masks: ActionMasks,
        decisions: DecisionFlags,
    ) -> Actions: ...


class RandomPolicy:
    """Sample uniformly among the valid actions of each deciding train."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, env, observations, infos, masks, decisions) -> Actions:
        actions: Actions = {}
        for handle, mask in masks.items():
            if decisions[handle]:
                valid = torch.nonzero(mask).flatten().tolist()
                actions[handle] = int(self.rng.choice(valid))
            else:
                actions[handle] = int(torch.nonzero(mask)[0])
        return actions


class ShortestPathPolicy:
    """Greedily follow each train's own shortest path, ignoring other trains.

    Departs as soon as possible and never stops voluntarily, so it measures
    what uncoordinated selfish routing achieves; congestion and deadlocks grow
    with train density.
    """

    def __call__(self, env, observations, infos, masks, decisions) -> Actions:
        distances = env.distance_map.get()
        actions: Actions = {}
        for agent in env.agents:
            handle = agent.handle
            mask = masks[handle]
            if not decisions[handle]:
                actions[handle] = int(torch.nonzero(mask)[0])
                continue
            configuration = (
                agent.initial_configuration
                if infos["state"][handle] == TrainState.READY_TO_DEPART
                else agent.current_configuration
            )
            best_action, best_distance = None, np.inf
            # Forward first so it wins ties (it is the only accelerating move).
            for action in (RailEnvActions.MOVE_FORWARD, *MOVING_ACTIONS):
                if not mask[action.value]:
                    continue
                _, ((row, column), direction), *_ = (
                    env.rail._check_action_on_agent(action, configuration)
                )
                distance = distances[handle, row, column, direction]
                if distance < best_distance:
                    best_action, best_distance = action, distance
            if best_action is None:
                best_action = int(torch.nonzero(mask)[0])
            actions[handle] = int(getattr(best_action, "value", best_action))
        return actions


class DepartureCap:
    """Wrap any policy with a rule-based departure schedule.

    The winning NeurIPS 2020 RL teams all limited how many trains are on the
    map at once. Trains ready to depart are considered in order of their
    latest arrival; one is released only if fewer than max_active trains are
    on the map, its origin is free, and no train on the map faces it on the
    first `lookahead` cells of its shortest path. All other actions (routing,
    stopping) come from the wrapped policy.
    """

    def __init__(self, policy, max_active: int = 4, lookahead: int = 20) -> None:
        self.policy = policy
        self.max_active = max_active
        self.lookahead = lookahead

    def __call__(self, env, observations, infos, masks, decisions) -> Actions:
        actions = self.policy(env, observations, infos, masks, decisions)
        on_map = {
            a.current_configuration[0]: a.current_configuration[1]
            for a in env.agents
            if a.state.is_on_map_state() and a.current_configuration is not None
        }
        active = len(on_map)
        for agent in sorted(env.agents, key=lambda a: a.latest_arrival):
            handle = agent.handle
            if infos["state"][handle] != TrainState.READY_TO_DEPART or actions[handle] in (
                RailEnvActions.DO_NOTHING.value, RailEnvActions.STOP_MOVING.value,
            ):
                continue
            path = agent.get_shortest_path(env.distance_map) or []
            origin = agent.initial_configuration
            blocked = origin[0] in on_map or any(
                waypoint.position in on_map
                and on_map[waypoint.position] == (waypoint.direction + 2) % 4
                for waypoint in path[1:self.lookahead + 1]
            )
            if active >= self.max_active or blocked:
                actions[handle] = RailEnvActions.DO_NOTHING.value
            else:
                active += 1
                on_map[origin[0]] = origin[1]
        return actions


class DepartureGatedShortestPathPolicy(DepartureCap):
    """Shortest-path routing with the rule-based departure schedule."""

    def __init__(self, max_active: int = 4, lookahead: int = 20) -> None:
        super().__init__(ShortestPathPolicy(), max_active, lookahead)
