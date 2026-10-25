"""Prioritized planning with space-time reservations (non-learning reference).

Follows the planning core of the NeurIPS 2020 Flatland winner (prioritized
planning; Silver 2005, as used by Li et al. 2021): trains are planned one at
a time in priority order, each taking its earliest-arrival path through
(cell, direction, time) that avoids every cell-time already reserved by
higher-priority trains, including waiting at the station or on track.

Conservative conflict rules, so plans never rely on Flatland's same-step
close-following resolution:
- vertex: a cell holds at most one train per time step;
- swap: two trains never exchange cells in one step;
- follow: a train only enters a cell that was free in the previous step.

During execution a train whose actual state differs from its plan (for
example, departure readiness off by a step) is re-planned from its current
state against the remaining reservations. Without malfunctions and with unit
speeds, plans otherwise execute exactly.
"""

import heapq
import itertools
import math

import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState

OFF = None  # Position of a train still waiting at its station, off the map.


class PrioritizedPlanningPolicy:
    """Plan all trains at the first call of an episode, then follow the plans.

    priority: "departure" orders by earliest departure then shortest-path
    length; "distance" orders by shortest-path length only. max_expansions
    bounds each search; a train with no path found stays at its station.
    """

    def __init__(self, priority: str = "departure", max_expansions: int = 200_000) -> None:
        self.priority = priority
        self.max_expansions = max_expansions
        self._episode = None

    # -- reservations -------------------------------------------------------
    def _reset(self, env) -> None:
        self._episode = id(env), env._elapsed_steps
        self.occupied: dict[tuple, int] = {}     # (cell, t) -> handle
        self.moves: dict[tuple, int] = {}        # (from_cell, to_cell, t) -> handle
        self.plans: dict[int, dict[int, tuple]] = {}
        self.t_max = env._max_episode_steps
        self.distances = env.distance_map.get()
        self.planned_failures: set[int] = set()

    def _free(self, cell, t, handle) -> bool:
        return self.occupied.get((cell, t), handle) == handle

    def _can_move(self, src, dst, t, handle) -> bool:
        """src at time t -> dst at time t + 1 (src is OFF when entering the map)."""
        if not self._free(dst, t + 1, handle) or not self._free(dst, t, handle):
            return False
        if src is OFF:
            return True
        return self.moves.get((dst, src, t), handle) == handle

    def _release(self, handle) -> None:
        for t, configuration in self.plans.pop(handle, {}).items():
            if configuration is not OFF:
                self.occupied.pop((configuration[0], t), None)
        self.moves = {k: v for k, v in self.moves.items() if v != handle}

    def _reserve(self, handle, plan: dict[int, tuple]) -> None:
        self.plans[handle] = plan
        times = sorted(plan)
        for t in times:
            if plan[t] is not OFF:
                self.occupied[(plan[t][0], t)] = handle
        for t in times[:-1]:
            if plan[t] is not OFF and plan.get(t + 1) is not OFF:
                self.moves[(plan[t][0], plan[t + 1][0], t)] = handle

    # -- search --------------------------------------------------------------
    def _heuristic(self, handle, configuration) -> float:
        if configuration is OFF:
            return 0.0
        (row, column), direction = configuration
        value = self.distances[handle, row, column, direction]
        return value if math.isfinite(value) else 1e6

    def _plan(self, env, agent, start, t0) -> dict[int, tuple] | None:
        """Earliest-arrival A* over (configuration, time) from start at time t0."""
        handle = agent.handle
        targets = agent.targets
        initial = agent.initial_configuration
        earliest = agent.earliest_departure
        counter = itertools.count()
        frontier = [(t0 + self._heuristic(handle, start if start is not OFF else initial), next(counter), t0, start)]
        parents = {(start, t0): None}
        expansions = 0
        while frontier and expansions < self.max_expansions:
            _, _, t, configuration = heapq.heappop(frontier)
            expansions += 1
            if configuration is not OFF and configuration in targets:
                plan, node = {}, (configuration, t)
                while node is not None:
                    plan[node[1]] = node[0]
                    node = parents[node]
                return plan
            if t >= self.t_max:
                continue
            successors = []
            if configuration is OFF:
                successors.append(OFF)                       # keep waiting off the map
                if t >= earliest and self._can_move(OFF, initial[0], t, handle):
                    successors.append(initial)               # enter at the origin
            else:
                position, direction = configuration
                if self._free(position, t + 1, handle):
                    successors.append(configuration)         # stop in this cell
                for new_direction, allowed in enumerate(env.rail.get_transitions(configuration)):
                    if allowed:
                        new_position = get_new_position(position, new_direction)
                        if self._can_move(position, new_position, t, handle):
                            successors.append((new_position, new_direction))
            for successor in successors:
                node = (successor, t + 1)
                if node in parents:
                    continue
                parents[node] = (configuration, t)
                estimate = t + 1 + self._heuristic(handle, successor if successor is not OFF else initial)
                heapq.heappush(frontier, (estimate, next(counter), t + 1, successor))
        return None

    def _order(self, env) -> list:
        def length(agent):
            (row, column), direction = agent.initial_configuration
            value = self.distances[agent.handle, row, column, direction]
            return value if math.isfinite(value) else 1e6
        if self.priority == "distance":
            return sorted(env.agents, key=length)
        return sorted(env.agents, key=lambda a: (a.earliest_departure, length(a)))

    def _plan_and_reserve(self, env, agent, start, t0) -> None:
        plan = self._plan(env, agent, start, t0)
        if plan is None:
            # Stay at the station (or where the train is) for the rest of the episode.
            plan = {t: start for t in range(t0, self.t_max + 1)}
            self.planned_failures.add(agent.handle)
        self._reserve(agent.handle, plan)

    # -- execution ------------------------------------------------------------
    @staticmethod
    def _actual(agent):
        if agent.state == TrainState.DONE:
            return "done"
        if agent.state.is_off_map_state():
            return OFF
        return agent.current_configuration

    @staticmethod
    def _action(env, configuration, target) -> int:
        if target is configuration or target == configuration:
            return (RailEnvActions.DO_NOTHING if configuration is OFF else RailEnvActions.STOP_MOVING).value
        if configuration is OFF:
            return RailEnvActions.MOVE_FORWARD.value
        _, direction = configuration
        exits = env.rail.get_transitions(configuration)
        if sum(exits) == 1:
            return RailEnvActions.MOVE_FORWARD.value
        new_direction = target[1]
        if new_direction == direction:
            return RailEnvActions.MOVE_FORWARD.value
        if new_direction == (direction - 1) % 4:
            return RailEnvActions.MOVE_LEFT.value
        return RailEnvActions.MOVE_RIGHT.value

    def __call__(self, env, observations, infos, masks, decisions) -> dict[int, int]:
        t = env._elapsed_steps
        if self._episode is None or self._episode[0] != id(env) or t == 0 and self._episode[1] != 0 or t < self._episode[1]:
            self._reset(env)
            for agent in self._order(env):
                self._plan_and_reserve(env, agent, self._actual(agent), t)
        self._episode = id(env), t
        actions = {}
        for agent in self._order(env):
            handle = agent.handle
            actual = self._actual(agent)
            if actual == "done":
                actions[handle] = RailEnvActions.DO_NOTHING.value
                continue
            plan = self.plans.get(handle, {})
            if plan.get(t, "missing") != actual or (t + 1) not in plan:
                self._release(handle)
                self._plan_and_reserve(env, agent, actual, t)
                plan = self.plans[handle]
            action = self._action(env, actual, plan.get(t + 1, actual))
            mask = masks[handle]
            if not mask[action]:
                action = int(np.flatnonzero(mask.numpy())[0])
            actions[handle] = action
        return actions
