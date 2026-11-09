"""Traffic-aware reward variants for the project's Flatland environment.

Both rewards use the global state only while training. The policy observation
and five-action interface are unchanged. ``RewardRailEnv`` calls ``reset`` and
``after_step`` after all trains have advanced, so forecasts use a consistent
occupancy snapshot. Forecasts are deliberately short and heuristic; they are
not a collision-proof scheduler or a full time-dependent route planner.
"""

from __future__ import annotations

import math
from collections import defaultdict

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.step_utils.states import TrainState

from .registry import register
from .rewards import ChampionRewards


def _configuration(agent, *, ready_only: bool):
    """Use a running train's position, or its origin before departure."""
    if agent.state == TrainState.DONE:
        return None
    if agent.current_configuration is not None:
        return agent.current_configuration
    if ready_only and agent.state != TrainState.READY_TO_DEPART:
        return None
    return agent.initial_configuration


def _marker(agent):
    """Count movement within a cell for non-unit-speed scenarios too."""
    return (agent.current_configuration, agent.speed_counter.distance)


def _distance(grid, handle, configuration):
    if configuration is None:
        return math.inf
    (row, column), direction = configuration
    if not (0 <= row < grid.shape[1] and 0 <= column < grid.shape[2]):
        return math.inf
    return float(grid[handle, row, column, direction])


def _next_states(env, grid, handle, configuration):
    """Return legal rail successors whose target is reachable."""
    if configuration is None:
        return
    position, _ = configuration
    for direction, allowed in enumerate(env.rail.get_transitions(configuration)):
        if not allowed:
            continue
        next_position = get_new_position(position, direction)
        next_config = (next_position, direction)
        next_distance = _distance(grid, handle, next_config)
        if math.isfinite(next_distance):
            yield next_config, next_distance


def _greedy_path(env, grid, agent, lookahead):
    """Project a train's next cells along its directional shortest rail path."""
    current = _configuration(agent, ready_only=True)
    if current is None:
        return []
    current_distance = _distance(grid, agent.handle, current)
    if not math.isfinite(current_distance):
        return []
    # An off-map train can enter its origin no earlier than the next step.
    first_step = 0 if agent.current_configuration is not None else 1
    path = [(current[0], first_step)]
    for step in range(first_step + 1, lookahead + 1):
        candidates = [
            (distance, configuration)
            for configuration, distance in _next_states(env, grid, agent.handle, current)
            if distance < current_distance
        ]
        if not candidates:
            break
        current_distance, current = min(candidates)
        path.append((current[0], step))
    return path


def _forecasts(env, grid, lookahead):
    """Index bounded shortest-path projections by cell for both rewards."""
    by_cell = defaultdict(list)
    for agent in env.agents:
        for cell, step in _greedy_path(env, grid, agent, lookahead):
            by_cell[cell].append((agent.handle, step))
    return by_cell


def _risk_and_peers(by_cell, handles, conflict_window):
    """Measure near-time path overlap, including opposing single-track use."""
    handles = tuple(handles)
    pair_scores = defaultdict(float)
    peers = {handle: set() for handle in handles}
    hotspots = set()
    for cell, visits in by_cell.items():
        for i, (first, first_step) in enumerate(visits):
            for second, second_step in visits[i + 1 :]:
                if first == second or abs(first_step - second_step) > conflict_window:
                    continue
                score = 1.0 / (1.0 + max(first_step, second_step))
                pair = (min(first, second), max(first, second))
                pair_scores[pair] = max(pair_scores[pair], score)
                peers[first].add(second)
                peers[second].add(first)
                hotspots.add(cell)
    risk = {handle: 0.0 for handle in handles}
    for (first, second), score in pair_scores.items():
        risk[first] += score
        risk[second] += score
    return risk, peers, hotspots


@register(name="anticipatory_right_of_way")
class AnticipatoryRightOfWayRewards(ChampionRewards):
    """Reward early conflict avoidance and useful clearance of shared track.

    Per unfinished train, the Champion progress/arrival/deadlock terms receive
    ``time_cost``. A train that moves receives ``risk_coef`` times the clipped
    decrease in its short-horizon path-overlap risk. Moving toward the target
    out of a previously contested cell earns ``clearance_bonus``. A train that
    stays put while a previously conflicting peer advances and the conflict
    disappears receives ``yield_bonus`` once per peer per episode, capped by
    ``yield_clip`` per step. Idle states cannot repeatedly collect a bonus.

    The risk forecast follows each train's current shortest rail route. It can
    signal an approaching junction or opposing use of a single-track segment,
    but it does not know another policy's actual future actions.
    """

    def __init__(self, reward_config: dict | None):
        super().__init__(reward_config)
        cfg = reward_config or {}
        self.lookahead = int(cfg.get("lookahead", 3))
        self.conflict_window = int(cfg.get("conflict_window", 1))
        self.risk_coef = float(cfg.get("risk_coef", 0.10))
        self.risk_clip = float(cfg.get("risk_clip", 2.0))
        self.yield_bonus = float(cfg.get("yield_bonus", 0.05))
        self.yield_clip = int(cfg.get("yield_clip", 2))
        self.clearance_bonus = float(cfg.get("clearance_bonus", 0.05))
        self.time_cost = float(cfg.get("time_cost", -0.01))
        if (
            self.lookahead < 1 or self.conflict_window < 0 or self.risk_clip < 0
            or self.yield_clip < 0 or self.risk_coef < 0 or self.yield_bonus < 0
            or self.clearance_bonus < 0 or self.time_cost > 0
        ):
            raise ValueError("lookahead must be positive, bonuses nonnegative, and time_cost nonpositive")

    def _traffic(self, env):
        grid = env.distance_map.get()
        forecast = _forecasts(env, grid, self.lookahead)
        return _risk_and_peers(
            forecast, (agent.handle for agent in env.agents), self.conflict_window
        )

    def reset(self, env):
        super().reset(env)
        self.prev_risk, self.prev_peers, self.prev_hotspots = self._traffic(env)
        self.prev_markers = {a.handle: _marker(a) for a in env.agents}
        self.yielded_pairs = set()

    def after_step(self, env, rewards):
        old_distances = dict(self.prev_dist)
        shaped = super().after_step(env, rewards)
        risk, peers, hotspots = self._traffic(env)
        markers = {a.handle: _marker(a) for a in env.agents}
        for agent in env.agents:
            handle = agent.handle
            if agent.state == TrainState.DONE:
                continue  # Champion pays arrival exactly once.
            shaped[handle] += self.time_cost
            moved = markers[handle] != self.prev_markers[handle]
            if moved:
                risk_delta = self.prev_risk[handle] - risk[handle]
                risk_delta = max(-self.risk_clip, min(self.risk_clip, risk_delta))
                shaped[handle] += self.risk_coef * risk_delta
                old_cell = self.prev_markers[handle][0]
                new_cell = agent.current_configuration
                if (
                    old_cell is not None
                    and new_cell is not None
                    and old_cell[0] in self.prev_hotspots
                    and new_cell[0] != old_cell[0]
                    and new_cell[0] not in hotspots
                    and self.prev_dist[handle] < old_distances[handle]
                ):
                    shaped[handle] += self.clearance_bonus
            elif agent.state == TrainState.READY_TO_DEPART or agent.state.is_on_map_state():
                released = self.prev_peers[handle] - peers[handle]
                qualifying = [
                    peer for peer in released
                    if (handle, peer) not in self.yielded_pairs
                    and markers[peer] != self.prev_markers[peer]
                ]
                for peer in qualifying[: self.yield_clip]:
                    self.yielded_pairs.add((handle, peer))
                shaped[handle] += self.yield_bonus * min(len(qualifying), self.yield_clip)
            shaped[handle] = float(shaped[handle])
        self.prev_risk, self.prev_peers, self.prev_hotspots = risk, peers, hotspots
        self.prev_markers = markers
        return shaped


@register(name="congestion_aware_journey_time")
class CongestionAwareJourneyTimeRewards(ChampionRewards):
    """Reward movement that lowers a short-horizon traffic-aware ETA.

    ETA enumerates rail choices up to ``lookahead`` cells, then adds the exact
    directional shortest-path distance for the unsearched tail. Each traversed
    cell costs one step (adjusted for train speed), plus decaying penalties for
    current occupancy and competing trains' projected demand. Thus a longer
    rail route can have lower ETA than a blocked shortest route. A moving train
    receives ``eta_coef * clip(previous_eta - current_eta, ±eta_clip)``;
    stationary trains receive no positive ETA shaping. Champion's arrival and
    deadlock events and ``time_cost`` continue to favor timely completion.
    """

    def __init__(self, reward_config: dict | None):
        super().__init__(reward_config)
        cfg = reward_config or {}
        self.lookahead = int(cfg.get("lookahead", 3))
        self.conflict_window = int(cfg.get("conflict_window", 1))
        self.occupancy_penalty = float(cfg.get("occupancy_penalty", 2.0))
        self.flow_penalty = float(cfg.get("flow_penalty", 1.0))
        self.max_flow = int(cfg.get("max_flow", 2))
        self.eta_coef = float(cfg.get("eta_coef", 0.05))
        self.eta_clip = float(cfg.get("eta_clip", 3.0))
        self.time_cost = float(cfg.get("time_cost", -0.01))
        if (
            self.lookahead < 1 or self.conflict_window < 0 or self.max_flow < 0
            or self.occupancy_penalty < 0 or self.flow_penalty < 0 or self.eta_clip < 0
            or self.eta_coef < 0 or self.time_cost > 0
        ):
            raise ValueError("lookahead must be positive, shaping nonnegative, and time_cost nonpositive")

    def _eta(self, env, grid, agent, forecast):
        if agent.state == TrainState.DONE:
            return 0.0
        origin = _configuration(agent, ready_only=False)
        baseline = _distance(grid, agent.handle, origin)
        if not math.isfinite(baseline):
            return 1000.0
        speed = float(agent.speed_counter.max_speed)
        step_time = 1.0 / speed if speed > 0 else 1.0

        def search(configuration, depth, seen):
            remaining = _distance(grid, agent.handle, configuration)
            if remaining <= 0 or depth >= self.lookahead:
                return remaining * step_time
            best = math.inf
            for successor, _ in _next_states(env, grid, agent.handle, configuration):
                if successor in seen:
                    continue
                cell = successor[0]
                occupant = int(env.agent_positions[cell[0], cell[1]])
                occupied = occupant >= 0 and occupant != agent.handle
                flow = sum(
                    other != agent.handle and abs(visit_step - depth - 1) <= self.conflict_window
                    for other, visit_step in forecast.get(cell, ())
                )
                congestion = (
                    self.occupancy_penalty * occupied
                    + self.flow_penalty * min(flow, self.max_flow)
                ) / (depth + 1)
                candidate = step_time + congestion + search(successor, depth + 1, seen | {successor})
                best = min(best, candidate)
            return best if math.isfinite(best) else remaining * step_time

        return search(origin, 0, {origin})

    def _etas(self, env):
        grid = env.distance_map.get()
        forecast = _forecasts(env, grid, self.lookahead)
        return {a.handle: self._eta(env, grid, a, forecast) for a in env.agents}

    def reset(self, env):
        super().reset(env)
        self.prev_eta = self._etas(env)
        self.prev_markers = {a.handle: _marker(a) for a in env.agents}

    def after_step(self, env, rewards):
        shaped = super().after_step(env, rewards)
        eta = self._etas(env)
        markers = {a.handle: _marker(a) for a in env.agents}
        for agent in env.agents:
            handle = agent.handle
            if markers[handle] != self.prev_markers[handle]:
                improvement = self.prev_eta[handle] - eta[handle]
                improvement = max(-self.eta_clip, min(self.eta_clip, improvement))
                shaped[handle] += self.eta_coef * improvement
            if agent.state != TrainState.DONE:
                shaped[handle] += self.time_cost
            shaped[handle] = float(shaped[handle])
        self.prev_eta, self.prev_markers = eta, markers
        return shaped
