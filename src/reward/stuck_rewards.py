"""Champion reward plus a per-step penalty for trains stuck on the map.

The champion reward punishes a train once when the deadlock detector flags
it, but a train queued behind a jam or blocked without a formal deadlock
costs nothing. This variant adds stuck_penalty for every step a departed,
unfinished train stays in the same cell without advancing within it.
"""

from flatland.envs.step_utils.states import TrainState

from .registry import register
from .rewards import ChampionRewards


@register(name="champion_stuck")
class ChampionStuckRewards(ChampionRewards):
    """Champion shaping plus stuck_penalty (default -0.01) per stuck step."""

    def __init__(self, reward_config: dict | None):
        super().__init__(reward_config)
        self.stuck_penalty = (reward_config or {}).get("stuck_penalty", -0.01)

    @staticmethod
    def _progress(agent):
        return agent.current_configuration, agent.speed_counter.distance

    def reset(self, env):
        super().reset(env)
        self.prev_progress = {a.handle: self._progress(a) for a in env.agents}

    def after_step(self, env, rewards):
        shaped = super().after_step(env, rewards)
        for agent in env.agents:
            progress = self._progress(agent)
            on_map = agent.state.is_on_map_state() and agent.state != TrainState.DONE
            if on_map and progress == self.prev_progress[agent.handle]:
                shaped[agent.handle] += self.stuck_penalty
            self.prev_progress[agent.handle] = progress
        return shaped
