"""Cooperative and timetable-aware rewards for Flatland rail dispatching.

Both rewards retain the champion's distance progress, one-time deadlock
penalty, and one-time arrival bonus. They run after the complete environment
step through RewardRailEnv, when every train's new position is available.
"""

from collections import defaultdict, deque

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.step_utils.states import TrainState

from .registry import register
from .rewards import ChampionRewards


_ACTIONABLE = {TrainState.READY_TO_DEPART, TrainState.MOVING, TrainState.STOPPED}


def _position(agent):
    """Return a train's occupied cell, excluding off-map and finished trains."""
    if agent.state == TrainState.DONE or not agent.state.is_on_map_state():
        return None
    configuration = agent.current_configuration
    return configuration[0] if configuration is not None else None


@register(name="unlocking_other_trains")
class UnlockingOtherTrainsRewards(ChampionRewards):
    """Pay a train once per peer when vacating a cell that blocks that peer.

    A peer is blocked when all of its legal next cells are occupied (or, before
    departure, its origin is occupied). When the blocker leaves a legal next
    cell free, it receives ``unlock_bonus`` for that peer. Peers queued behind
    it receive geometrically discounted credit up to ``max_chain_depth``.
    Credit is capped per step and each blocker/beneficiary pair can be paid
    only once per episode, preventing repeated move-away/move-back farming.
    An actionable, unfinished train also pays ``step_cost`` each step.
    """

    def __init__(self, reward_config: dict | None):
        super().__init__(reward_config)
        config = reward_config or {}
        self.step_cost = float(config.get("step_cost", -0.02))
        self.unlock_bonus = float(config.get("unlock_bonus", 0.5))
        self.chain_discount = float(config.get("chain_discount", 0.5))
        self.max_unlock_bonus = float(config.get("max_unlock_bonus", 1.0))
        self.max_chain_depth = int(config.get("max_chain_depth", 3))
        if self.step_cost > 0:
            raise ValueError("step_cost must be nonpositive")
        if self.unlock_bonus < 0 or not 0 <= self.chain_discount <= 1:
            raise ValueError("unlock_bonus must be nonnegative and chain_discount in [0, 1]")
        if self.max_unlock_bonus < 0 or self.max_chain_depth < 1:
            raise ValueError("max_unlock_bonus must be nonnegative and max_chain_depth >= 1")

    def reset(self, env):
        super().reset(env)
        self.prev_positions = {agent.handle: _position(agent) for agent in env.agents}
        self.prev_configurations = {
            agent.handle: agent.current_configuration for agent in env.agents
        }
        self.prev_states = {agent.handle: agent.state for agent in env.agents}
        self.credited_pairs = set()

    def _blocked_dependencies(self, env):
        """Map each occupied blocker to trains it currently holds in a queue."""
        occupant = {
            position: handle for handle, position in self.prev_positions.items()
            if position is not None
        }
        dependents = defaultdict(set)
        blocked_cells = {}
        for agent in env.agents:
            handle = agent.handle
            state = self.prev_states[handle]
            if state == TrainState.DONE:
                continue
            configuration = self.prev_configurations[handle]
            if configuration is None:
                if state != TrainState.READY_TO_DEPART:
                    continue
                exits = (agent.initial_configuration[0],)
            else:
                position = configuration[0]
                transitions = env.rail.get_transitions(configuration)
                exits = tuple(
                    get_new_position(position, direction)
                    for direction, allowed in enumerate(transitions) if allowed
                )
            if not exits or any(cell not in occupant for cell in exits):
                continue
            blocked_cells[handle] = exits
            for cell in exits:
                blocker = occupant[cell]
                if blocker != handle:
                    dependents[blocker].add(handle)
        return dependents, blocked_cells

    def _unlock_credit(self, env, current_positions):
        dependents, blocked_cells = self._blocked_dependencies(env)
        occupant_now = {
            position: handle for handle, position in current_positions.items()
            if position is not None
        }
        credits = defaultdict(float)
        for blocker, old_cell in self.prev_positions.items():
            if old_cell is None or current_positions[blocker] == old_cell:
                continue
            direct = [
                peer for peer in dependents[blocker]
                if old_cell in blocked_cells[peer]
                and occupant_now.get(old_cell) in (None, peer)
            ]
            if not direct:
                continue
            queue = deque((peer, 1) for peer in direct)
            visited = {blocker}
            credit = 0.0
            while queue:
                peer, depth = queue.popleft()
                if peer in visited:
                    continue
                visited.add(peer)
                pair = (blocker, peer)
                if pair not in self.credited_pairs:
                    credit += self.unlock_bonus * self.chain_discount ** (depth - 1)
                    self.credited_pairs.add(pair)
                if depth < self.max_chain_depth:
                    queue.extend((other, depth + 1) for other in dependents[peer])
            credits[blocker] = min(credit, self.max_unlock_bonus)
        return credits

    def after_step(self, env, rewards):
        shaped = super().after_step(env, rewards)
        current_positions = {agent.handle: _position(agent) for agent in env.agents}
        credits = self._unlock_credit(env, current_positions)
        for agent in env.agents:
            handle = agent.handle
            if agent.state in _ACTIONABLE:
                shaped[handle] += self.step_cost
            shaped[handle] += credits.get(handle, 0.0)
        self.prev_positions = current_positions
        self.prev_configurations = {
            agent.handle: agent.current_configuration for agent in env.agents
        }
        self.prev_states = {agent.handle: agent.state for agent in env.agents}
        return shaped


@register(name="deadline_slack_rescue")
class DeadlineSlackRescueRewards(ChampionRewards):
    """Give extra progress credit to trains at risk of missing their deadline.

    ``slack = latest_arrival - elapsed_steps - remaining_ETA``. Before each
    step, risk is clipped to ``[0, 1]`` from ``(slack_window-slack) /
    slack_window``. An actionable train gets

    ``risk * (urgent_progress_coef * clipped_ETA_gain +
              slack_loss_penalty * clipped_slack_loss)``.

    ETA uses direction-aware rail distance and train speed, including partial
    progress within the current cell. Unchanged position and speed progress
    yield no positive term; delay loses slack and earns a nonpositive term.
    The champion arrival/deadlock terms remain, plus ``step_cost`` for each
    actionable, unfinished step.
    """

    def __init__(self, reward_config: dict | None):
        super().__init__(reward_config)
        config = reward_config or {}
        self.step_cost = float(config.get("step_cost", -0.02))
        self.slack_window = float(config.get("slack_window", 10.0))
        self.urgent_progress_coef = float(config.get("urgent_progress_coef", 0.2))
        self.slack_loss_penalty = float(config.get("slack_loss_penalty", -0.1))
        self.max_urgent_progress = float(config.get("max_urgent_progress", 1.0))
        self.max_slack_loss = float(config.get("max_slack_loss", 1.0))
        if self.step_cost > 0:
            raise ValueError("step_cost must be nonpositive")
        if self.slack_window <= 0 or self.urgent_progress_coef < 0:
            raise ValueError("slack_window must be positive and urgent_progress_coef >= 0")
        if self.slack_loss_penalty > 0 or self.max_urgent_progress < 0 or self.max_slack_loss < 0:
            raise ValueError("slack_loss_penalty must be <= 0 and shaping caps >= 0")

    @staticmethod
    def _eta(env, agent):
        if agent.state == TrainState.DONE:
            return 0.0
        distance = ChampionRewards._distance(env, agent)
        speed_counter = agent.speed_counter
        speed = float(speed_counter.max_speed)
        if speed <= 0:
            return 1000.0
        fraction = float(speed_counter.distance) if agent.current_configuration else 0.0
        return max(0.0, (distance - fraction) / speed)

    def reset(self, env):
        super().reset(env)
        self.prev_eta = {agent.handle: self._eta(env, agent) for agent in env.agents}
        self.prev_states = {agent.handle: agent.state for agent in env.agents}
        self.prev_elapsed = int(env._elapsed_steps)

    def after_step(self, env, rewards):
        shaped = super().after_step(env, rewards)
        elapsed = int(env._elapsed_steps)
        delta_steps = max(0, elapsed - self.prev_elapsed)
        for agent in env.agents:
            handle = agent.handle
            eta = self._eta(env, agent)
            previous_eta = self.prev_eta[handle]
            if self.prev_states[handle] in _ACTIONABLE:
                previous_slack = float(agent.latest_arrival) - self.prev_elapsed - previous_eta
                current_slack = float(agent.latest_arrival) - elapsed - eta
                risk = min(1.0, max(0.0, (self.slack_window - previous_slack) / self.slack_window))
                eta_gain = min(self.max_urgent_progress, max(0.0, previous_eta - eta))
                slack_loss = min(self.max_slack_loss, max(0.0, previous_slack - current_slack))
                shaped[handle] += risk * (
                    self.urgent_progress_coef * eta_gain
                    + self.slack_loss_penalty * slack_loss
                )
            if agent.state in _ACTIONABLE and delta_steps:
                shaped[handle] += self.step_cost * min(delta_steps, 1)
            self.prev_eta[handle] = eta
            self.prev_states[handle] = agent.state
        self.prev_elapsed = elapsed
        return shaped
