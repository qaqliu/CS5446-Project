"""Episode evaluation metrics: completion, travel time, delay, deadlock, fairness."""

from dataclasses import dataclass, field

import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState


def find_deadlocked_trains(env: RailEnv) -> set[int]:
    """Return trains that can never move again given current occupancy.

    A train is a deadlock candidate when every track exit from its current
    configuration leads to a cell occupied by another train. Candidates whose
    blockers can still move are released repeatedly; the remaining fixed point
    is a set of trains blocking each other in cycles (e.g. head-on on a single
    track), plus trains queued behind such a cycle.
    """
    occupancy = env.agent_positions
    next_cells: dict[int, list[tuple[int, int]]] = {}
    for agent in env.agents:
        if agent.state == TrainState.DONE or agent.current_configuration is None:
            continue
        position, direction = agent.current_configuration
        exits = env.rail.get_transitions(agent.current_configuration)
        next_cells[agent.handle] = [
            get_new_position(position, new_direction)
            for new_direction, allowed in enumerate(exits)
            if allowed
        ]

    def blockers(handle: int) -> list[int] | None:
        """Occupying trains for every exit, or None when some exit is free."""
        result = []
        for row, column in next_cells[handle]:
            occupant = int(occupancy[row, column])
            if occupant < 0 or occupant == handle:
                return None
            result.append(occupant)
        return result

    candidates = {
        handle for handle, cells in next_cells.items()
        if cells and blockers(handle) is not None
    }
    changed = True
    while changed:
        changed = False
        for handle in list(candidates):
            if any(other not in candidates for other in blockers(handle)):
                candidates.remove(handle)
                changed = True
    return candidates


@dataclass
class TrainRecord:
    handle: int
    max_speed: float
    earliest_departure: int
    latest_arrival: int
    departure_step: int | None = None
    arrival_step: int | None = None
    deadlock_step: int | None = None
    last_progress_step: int = 0
    last_progress: tuple | None = None
    episode_return: float = 0.0
    # Lateness against latest arrival; unfinished trains use the optimistic
    # projection of finishing now along the shortest path.
    delay: float = 0.0

    @property
    def completed(self) -> bool:
        return self.arrival_step is not None

    @property
    def travel_time(self) -> int | None:
        if self.departure_step is None or self.arrival_step is None:
            return None
        return self.arrival_step - self.departure_step


@dataclass
class EpisodeTracker:
    """Record per-train progress. Call update after every env.step.

    A train is stuck when it ends the episode on the map, unfinished, without
    having moved for stuck_threshold steps. Unlike a deadlock, a stuck train
    may still have a free exit that the policy refuses to take.
    """

    env: RailEnv
    stuck_threshold: int = 10
    records: dict[int, TrainRecord] = field(init=False)
    steps: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.records = {
            agent.handle: TrainRecord(
                handle=agent.handle,
                max_speed=float(agent.speed_counter.max_speed),
                earliest_departure=int(agent.earliest_departure),
                latest_arrival=int(agent.latest_arrival),
            )
            for agent in self.env.agents
        }

    def update(self, rewards: dict[int, float]) -> None:
        self.steps = self.env._elapsed_steps
        for agent in self.env.agents:
            record = self.records[agent.handle]
            record.episode_return += float(rewards[agent.handle])
            if record.departure_step is None and agent.state.is_on_map_state():
                record.departure_step = self.steps
            if record.arrival_step is None and agent.arrival_time is not None:
                record.arrival_step = int(agent.arrival_time)
            progress = (
                agent.current_configuration,
                agent.speed_counter.distance,
            )
            if progress != record.last_progress:
                record.last_progress = progress
                record.last_progress_step = self.steps
        for handle in find_deadlocked_trains(self.env):
            if self.records[handle].deadlock_step is None:
                self.records[handle].deadlock_step = self.steps

    def summary(self) -> dict[str, float]:
        distance_map = self.env.distance_map
        for agent in self.env.agents:
            record = self.records[agent.handle]
            if record.completed:
                record.delay = max(0, record.arrival_step - record.latest_arrival)
            else:
                record.delay = max(
                    0, -agent.get_current_delay(self.steps, distance_map)
                )
        records = list(self.records.values())
        stuck = [
            r for r in records
            if not r.completed
            and r.departure_step is not None
            and self.steps - r.last_progress_step >= self.stuck_threshold
        ]
        completed = [r for r in records if r.completed]
        delays = np.array([r.delay for r in records], dtype=float)
        n = len(records)
        return {
            "num_trains": n,
            "episode_steps": self.steps,
            "max_episode_steps": self.env._max_episode_steps,
            "all_arrived": len(completed) == n,
            "completion_rate": len(completed) / max(n, 1),
            # Low departure with low deadlock means trains avoid conflicts by waiting.
            "departure_rate": sum(r.departure_step is not None for r in records)
            / max(n, 1),
            "on_time_rate": sum(r.delay == 0 for r in completed) / max(n, 1),
            "deadlock_rate": sum(r.deadlock_step is not None for r in records)
            / max(n, 1),
            "stuck_rate": len(stuck) / max(n, 1),
            "mean_travel_time": _mean([r.travel_time for r in completed]),
            "mean_arrival_delay": _mean([r.delay for r in completed]),
            "mean_delay_all": float(delays.mean()) if n else float("nan"),
            "delay_gini": gini(delays),
            "mean_return": _mean([r.episode_return for r in records]),
            # NeurIPS 2020 Flatland score (Laurent et al. 2021, Sec. 2.3): every
            # train pays -1 per step until it arrives (or until t_max), so
            # 1 - mean(arrival step or t_max) / t_max. Rewards fast arrival, not
            # just arrival; holding trains back costs score.
            "competition_score": _competition_score(records, self.env._max_episode_steps),
            "makespan": max((r.arrival_step for r in completed), default=float("nan")),
        }


def _competition_score(records: list, t_max: int | None) -> float:
    if not records or not t_max:
        return float("nan")
    arrivals = [r.arrival_step if r.completed else t_max for r in records]
    return float(1.0 - np.mean(arrivals) / t_max)


def gini(values: np.ndarray) -> float:
    """Gini coefficient of non-negative values: 0 is perfectly equal delay."""
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float((2 * ranks - n - 1) @ values / (n * values.sum()))


def _mean(values: list) -> float:
    return float(np.mean(values)) if values else float("nan")
