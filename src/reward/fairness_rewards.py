"""Team reward that prioritizes the train facing the worst service delay."""

import math

from flatland.envs.rewards import Rewards
from flatland.envs.step_utils.states import TrainState

from .deadlock import DeadlockChecker
from .registry import register
from .rewards import ChampionRewards


@register(name="leave_no_train_behind")
class LeaveNoTrainBehindRewards(Rewards[float]):
    """Give every train the same bounded, worst-train-sensitive team reward.

    At each step, q_i is the larger of normalized consecutive no-progress
    waiting and normalized *projected* final-arrival lateness for unfinished
    train i. Normalization is x / (x + scale), so 0 <= q_i <= 1 and long
    delays still change the signal. Q = max_i q_i, or zero if all are done.

    Each train receives

        [finish_bonus * new_arrivals
         + deadlock_penalty * new_deadlocks
         + step_cost * unfinished_count
         + fairness_coef * (previous_Q - Q)] / number_of_trains.

    Events are counted once per train. The fairness term has magnitude at
    most fairness_coef / number_of_trains, and the negative time cost makes
    a wait-then-move cycle unprofitable in undiscounted reward. This is a
    training reward, not the official Flatland timetable score.
    """

    def __init__(self, reward_config: dict | None):
        config = reward_config or {}
        self.finish_bonus = float(config.get("finish_bonus", 10.0))
        self.deadlock_penalty = float(config.get("deadlock_penalty", -5.0))
        self.step_cost = float(config.get("step_cost", -0.01))
        self.fairness_coef = float(config.get("fairness_coef", 1.0))
        self.wait_scale = float(config.get("wait_scale", 5.0))
        self.lateness_scale = float(config.get("lateness_scale", 10.0))
        if self.wait_scale <= 0 or self.lateness_scale <= 0:
            raise ValueError("wait_scale and lateness_scale must be positive")
        if self.finish_bonus < 0 or self.deadlock_penalty > 0 or self.step_cost >= 0:
            raise ValueError("finish_bonus must be nonnegative and penalties negative")
        if self.fairness_coef < 0:
            raise ValueError("fairness_coef must be nonnegative")
        self.deadlock_checker = DeadlockChecker()

    @staticmethod
    def _progress(agent):
        return agent.current_configuration, agent.speed_counter.distance

    @staticmethod
    def _normalized(amount: float, scale: float) -> float:
        if math.isinf(amount):
            return 1.0
        return amount / (amount + scale)

    def _pressure(self, env) -> float:
        now = env._elapsed_steps
        pressures = []
        for agent in env.agents:
            if agent.state == TrainState.DONE:
                continue
            distance = ChampionRewards._distance(env, agent)
            max_speed = float(agent.speed_counter.max_speed)
            remaining_time = distance / max_speed if max_speed > 0 else math.inf
            earliest = max(now, agent.earliest_departure)
            projected_lateness = max(0.0, earliest + remaining_time - agent.latest_arrival)
            wait_pressure = self._normalized(self.wait_steps[agent.handle], self.wait_scale)
            delay_pressure = self._normalized(projected_lateness, self.lateness_scale)
            pressures.append(max(wait_pressure, delay_pressure))
        return max(pressures, default=0.0)

    def reset(self, env):
        self.deadlock_checker.reset(env)
        self.already_rewarded = {
            agent.handle for agent in env.agents if agent.state == TrainState.DONE
        }
        self.already_punished = set()
        self.prev_progress = {agent.handle: self._progress(agent) for agent in env.agents}
        self.wait_steps = {agent.handle: 0 for agent in env.agents}
        self.prev_pressure = self._pressure(env)

    def after_step(self, env, rewards):
        self.deadlock_checker.update_deadlocks()
        now = env._elapsed_steps
        new_arrivals = 0
        new_deadlocks = 0
        unfinished = 0
        for agent in env.agents:
            handle = agent.handle
            progress = self._progress(agent)
            if agent.state == TrainState.DONE:
                self.wait_steps[handle] = 0
                if handle not in self.already_rewarded:
                    new_arrivals += 1
                    self.already_rewarded.add(handle)
            else:
                unfinished += 1
                eligible = now >= agent.earliest_departure
                self.wait_steps[handle] = (
                    self.wait_steps[handle] + 1
                    if eligible and progress == self.prev_progress[handle]
                    else 0
                )
                if (
                    self.deadlock_checker.is_deadlocked(handle)
                    and handle not in self.already_punished
                ):
                    new_deadlocks += 1
                    self.already_punished.add(handle)
            self.prev_progress[handle] = progress

        pressure = self._pressure(env)
        n = len(env.agents)
        team_reward = (
            self.finish_bonus * new_arrivals
            + self.deadlock_penalty * new_deadlocks
            + self.step_cost * unfinished
            + self.fairness_coef * (self.prev_pressure - pressure)
        ) / max(n, 1)
        self.prev_pressure = pressure
        return {agent.handle: float(team_reward) for agent in env.agents}

    def step_reward(self, agent, agent_transition_data, distance_map, elapsed_steps):
        return 0.0

    def end_of_episode_reward(self, agent, distance_map, elapsed_steps):
        return 0.0

    def cumulate(self, *rewards):
        return sum(rewards)

    def empty(self):
        return 0.0
