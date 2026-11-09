"""Single-component controls for the Champion reward ablation."""

from flatland.envs.rewards import Rewards

from .deadlock import DeadlockChecker
from .registry import register
from .rewards import ChampionRewards


class _AfterStepFloatRewards(Rewards[float]):
    """Let RewardRailEnv apply the complete reward after all trains move."""

    def step_reward(self, agent, agent_transition_data, distance_map, elapsed_steps):
        return 0.0

    def end_of_episode_reward(self, agent, distance_map, elapsed_steps):
        return 0.0

    def cumulate(self, *rewards):
        return sum(rewards)

    def empty(self):
        return 0.0


@register(name="shortest_rail_distance")
class ShortestRailDistanceRewards(_AfterStepFloatRewards):
    """Only Champion's direction-aware shortest-rail-distance progress term.

    r_i(t) = progress_coef * (d_i(t-1) - d_i(t)). There is no arrival
    bonus, deadlock penalty, or time cost. Champion's distance helper also
    maps unreachable distances to 1000 and DONE to zero.
    """

    def __init__(self, reward_config: dict | None):
        self.progress_coef = (reward_config or {}).get("progress_coef", 0.01)

    def reset(self, env):
        self.prev_dist = {
            agent.handle: ChampionRewards._distance(env, agent)
            for agent in env.agents
        }

    def after_step(self, env, rewards):
        shaped = {}
        for agent in env.agents:
            handle = agent.handle
            distance = ChampionRewards._distance(env, agent)
            shaped[handle] = float(self.progress_coef * (self.prev_dist[handle] - distance))
            self.prev_dist[handle] = distance
        return shaped


@register(name="deadlock_only")
class DeadlockOnlyRewards(_AfterStepFloatRewards):
    """Only Champion's one-time dependency-based deadlock penalty.

    r_i(t) = deadlock_penalty if the checker first flags train i this
    episode, otherwise zero. No arrival, progress, or time term is added.
    """

    def __init__(self, reward_config: dict | None):
        self.deadlock_penalty = (reward_config or {}).get("deadlock_penalty", -5.0)
        self.deadlock_checker = DeadlockChecker()

    def reset(self, env):
        self.deadlock_checker.reset(env)
        self.already_punished = set()

    def after_step(self, env, rewards):
        self.deadlock_checker.update_deadlocks()
        shaped = {}
        for agent in env.agents:
            handle = agent.handle
            if self.deadlock_checker.is_deadlocked(handle) and handle not in self.already_punished:
                shaped[handle] = float(self.deadlock_penalty)
                self.already_punished.add(handle)
            else:
                shaped[handle] = 0.0
        return shaped
