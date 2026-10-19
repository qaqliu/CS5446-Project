"""Registered reward implementations."""

import math

from flatland.envs.rewards import DefaultRewards as FlatlandDefaultRewards, Rewards
from flatland.envs.step_utils.states import TrainState

from .deadlock import DeadlockChecker
from .registry import register

@register(name="default")
class DefaultRewards(FlatlandDefaultRewards):
    """Adapt Flatland's default reward to the common reward-config interface."""

    def __init__(self, reward_config: dict | None):
        super().__init__(**(reward_config or {}))


@register(name="champion")
class ChampionRewards(Rewards[float]):
    """Distance progress + one-time arrival bonus and deadlock penalty.

    Replaces default scoring. Reset and after_step are called by RewardRailEnv
    so global occupancy is read only after every train has been updated.
    Defaults follow the 2020 RL winner's published reward formula.

    Champion reward sources

    The NeurIPS 2020 RL winner JBR_HSE reports `0.01 * distance_progress - 5 * deadlocked + 10 * finished` in [Section 4.2 of the competition paper](https://proceedings.mlr.press/v133/laurent21a/laurent21a.pdf).

    The [original competition repository](https://github.com/jbr-ai-labs/NeurIPS2020-Flatland-Competition-Solution) contains reward configurations but omits the imported `env.rewards` implementations. This adaptation uses the same team's later MIT-licensed [MAMBA repository](https://github.com/jbr-ai-labs/mamba/tree/2c97258f71bf1c421c40ce14fd2f7cc3fe7fe19f/env/flatland), commit `2c97258f71bf1c421c40ce14fd2f7cc3fe7fe19f`, specifically:

    - `rewards/NearRewardShaper.py`: direction-aware shortest-distance progress, infinity mapped to 1000.
    - `rewards/DeadlockPunishment.py`: punish each train at most once per episode.
    - `rewards/FinishRewardShaper.py`: reward each train at most once per episode.
    - `DeadlockChecker.py`: active-agent dependency-based deadlock heuristic.
    - `Flatland.py`: clear original environment rewards before applying shaping.

    This is an adaptation of published shaping components, not a claim to reproduce the entire champion system. We use the paper's weights, migrate Flatland 2.x fields to 4.3, and evaluate all components after the complete environment step. We omit unused far-deadlock code and custom observation graph mutations. Unlike the upstream wrapper, we do not terminate individual agents upon detected deadlock or change actions/observations. The dependency detector is an occupancy heuristic, not a general proof of deadlock under all Flatland dynamics. Intermediate-stop-specific rewards are not included.

    `ChampionRewards` requires the reset/after-step hooks provided by the project's `RewardRailEnv` in `src/utils/environment.py`; passing it to an unmodified upstream `RailEnv` alone will not invoke shaping.

    MIT License

    Copyright (c) 2022 jbr-ai-labs

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(self, reward_config: dict | None):
        reward_config = reward_config or {}
        self.progress_coef = reward_config.get("progress_coef", 0.01)
        self.deadlock_penalty = reward_config.get("deadlock_penalty", -5.0)
        self.finish_bonus = reward_config.get("finish_bonus", 10.0)
        self.deadlock_checker = DeadlockChecker()

    @staticmethod
    def _distance(env, agent):
        if agent.state == TrainState.DONE:
            return 0.0
        configuration = agent.current_configuration
        if configuration is None:
            configuration = agent.initial_configuration
        (row, column), direction = configuration
        distance = float(env.distance_map.get()[agent.handle, row, column, direction])
        return distance if math.isfinite(distance) else 1000.0

    def reset(self, env):
        self.deadlock_checker.reset(env)
        self.prev_dist = {a.handle: self._distance(env, a) for a in env.agents}
        self.already_punished = set()
        self.already_rewarded = set()

    def after_step(self, env, rewards):
        self.deadlock_checker.update_deadlocks()
        shaped = {}
        for agent in env.agents:
            handle = agent.handle
            distance = self._distance(env, agent)
            reward = self.progress_coef * (self.prev_dist[handle] - distance)
            self.prev_dist[handle] = distance
            if self.deadlock_checker.is_deadlocked(handle) and handle not in self.already_punished:
                reward += self.deadlock_penalty
                self.already_punished.add(handle)
            if agent.state == TrainState.DONE and handle not in self.already_rewarded:
                reward += self.finish_bonus
                self.already_rewarded.add(handle)
            shaped[handle] = float(reward)
        return shaped

    def step_reward(self, agent, agent_transition_data, distance_map, elapsed_steps):
        # Global shaping runs after the whole step, not inside the per-agent loop.
        return 0.0

    def end_of_episode_reward(self, agent, distance_map, elapsed_steps):
        return 0.0

    def cumulate(self, *rewards):
        return sum(rewards)

    def empty(self):
        return 0.0
