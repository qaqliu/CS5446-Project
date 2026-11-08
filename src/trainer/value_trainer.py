"""Shared replay-free PQN for local Flatland observations.

PQN uses fixed Q(lambda) targets from each fresh synchronous trajectory,
following Algorithm 1 of https://arxiv.org/abs/2407.04811. There is no target
network or historical replay in PQN. Independent worlds are stepped on CPU;
network inference and updates are batched in PyTorch.
"""

from copy import deepcopy
from pathlib import Path

import torch
from flatland.envs.rail_env import RailEnv
from torch.nn import functional as F

from src.agent import ShareNetworkValueAgent
from src.utils.action_mask import get_action_mask
from src.utils.environment import build_env, reset_env
from src.utils.exploration import ExplorationStats, epsilon_probabilities

from .registry import register
from .value_targets import masked_max, q_lambda_returns


class ValueTrainer:
    """Common synchronous collector and Q regression; budgets count env.step."""

    def __init__(
        self, env: RailEnv, value_agent: ShareNetworkValueAgent, config: dict
    ) -> None:
        self.agent, self.config = value_agent, config
        self.device = torch.device(config["trainer"]["device"])
        self.agent.to(self.device)
        self.model = self.agent.model
        self.num_envs = config["trainer"].get("num_envs", 1)
        self.total_env_steps = config["total_env_steps"]
        self.rollout_steps = config["rollout_steps"]  # Per world for PQN.
        self.batch_size = config["batch_size"]
        self.update_epochs = config["update_epochs"]
        self.gamma = config["gamma"]
        self.max_grad_norm = config["max_grad_norm"]
        self.epsilon_start = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay_steps = config["epsilon_decay_steps"]
        for name in (
            "num_envs",
            "total_env_steps",
            "rollout_steps",
            "batch_size",
            "update_epochs",
            "epsilon_decay_steps",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.total_env_steps % self.num_envs:
            raise ValueError("total_env_steps must be divisible by num_envs.")
        if not (0 <= self.epsilon_end <= self.epsilon_start <= 1):
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1.")
        if not 0 <= self.gamma <= 1 or self.max_grad_norm <= 0:
            raise ValueError("Invalid gamma or max_grad_norm.")
        if config["observation"]["type"] != "flattened_normalized_tree":
            raise ValueError("The value trainers require flattened tree observations.")
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=config["q_lr"],
            weight_decay=config.get("weight_decay", 0.0),
        )
        # Each world receives its own strided episode stream in random-map runs.
        if config.get("train_map_seeds") is not None:
            env.scenario_stream_count = self.num_envs
        self.envs = [env] + [
            build_env(config, stream_index=i, stream_count=self.num_envs)
            for i in range(1, self.num_envs)
        ]
        self.states = [None] * self.num_envs
        self.env_steps = self.agent_transitions = self.gradient_updates = 0

    def epsilon(self) -> float:
        progress = min(self.env_steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    @torch.no_grad()
    def _collect_step(self, exploration: ExplorationStats) -> dict[str, torch.Tensor]:
        """Advance each world once; keep arrived-agent padding out of learning."""
        features, masks, valid = [], [], []
        agent_ids = list(range(self.config["number_of_agents"]))
        for index, env in enumerate(self.envs):
            state = self.states[index]
            if state is None or state["episode_done"]:
                observations, infos = reset_env(env, self.config)
                state = dict(
                    observations=observations,
                    infos=infos,
                    dones=torch.zeros(len(agent_ids), dtype=torch.bool),
                    episode_done=False,
                )
                self.states[index] = state
            features.append(
                self.model.encode([state["observations"][i] for i in agent_ids])
            )
            current_mask = get_action_mask(env, state["infos"])
            masks.append(torch.stack([current_mask[i] for i in agent_ids]))
            valid.append(~state["dones"])
        features, masks, valid = (
            torch.stack(features),
            torch.stack(masks),
            torch.stack(valid),
        )
        was_training = self.agent.training
        self.agent.eval()
        q_values = self.model.forward_features(features).cpu()
        self.agent.train(was_training)
        probabilities = epsilon_probabilities(q_values, masks, self.epsilon())
        actions = torch.multinomial(probabilities.reshape(-1, 5), 1).reshape(
            valid.shape
        )
        exploration.update(
            probabilities,
            masks,
            actions,
            valid,
            greedy_actions=q_values.masked_fill(~masks, -torch.inf).argmax(-1),
        )
        rewards, dones, next_features, next_masks = [], [], [], []
        for index, env in enumerate(self.envs):
            obs, reward, done, infos = env.step(
                dict(zip(agent_ids, actions[index].tolist()))
            )
            terminal = torch.tensor(
                [bool(done[i] or done["__all__"]) for i in agent_ids]
            )
            self.states[index] = dict(
                observations=obs,
                infos=infos,
                dones=terminal,
                episode_done=bool(done["__all__"]),
            )
            rewards.append(
                torch.tensor([reward[i] for i in agent_ids], dtype=torch.float32)
            )
            dones.append(terminal)
            next_features.append(self.model.encode([obs[i] for i in agent_ids]))
            next_mask = get_action_mask(env, infos)
            next_masks.append(torch.stack([next_mask[i] for i in agent_ids]))
        self.env_steps += self.num_envs
        self.agent_transitions += int(valid.sum())
        return dict(
            observations=features,
            actions=actions,
            rewards=torch.stack(rewards).masked_fill(~valid, 0.0),
            dones=torch.stack(dones),
            valid=valid,
            next_observations=torch.stack(next_features),
            next_masks=torch.stack(next_masks),
        )

    def _regress(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[float, float]:
        self.agent.train()
        predictions = (
            self.model.forward_features(observations)
            .gather(
                -1,
                actions.to(self.device).unsqueeze(-1),
            )
            .squeeze(-1)
        )
        targets = targets.detach().to(predictions)
        loss = F.mse_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Q loss.")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.gradient_updates += 1
        return float(loss.detach()), float(
            (predictions.detach() - targets).abs().mean()
        )

    def train_steps(self, num_steps: int) -> dict:
        if num_steps <= 0 or num_steps % self.num_envs:
            raise ValueError(
                "Training chunk must be positive and divisible by num_envs."
            )
        if self.env_steps + num_steps > self.total_env_steps:
            raise ValueError("Training chunk would exceed the interaction budget.")
        exploration = ExplorationStats()
        losses = self._train_steps(num_steps, exploration)
        return {
            "env_steps": self.env_steps,
            "agent_transitions": self.agent_transitions,
            "gradient_updates": self.gradient_updates,
            "epsilon": self.epsilon(),
            "q_loss": sum(x[0] * x[2] for x in losses) / sum(x[2] for x in losses)
            if losses
            else None,
            "absolute_td_error": sum(x[1] * x[2] for x in losses)
            / sum(x[2] for x in losses)
            if losses
            else None,
            **exploration.summary(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = deepcopy(self.config)
        config.get("wandb", {}).pop("api_key", None)
        torch.save(
            dict(
                value_agent=self.agent.state_dict(),
                config=config,
                env_steps=self.env_steps,
                total_env_steps=self.env_steps,
                gradient_updates=self.gradient_updates,
            ),
            path,
        )

    def train(self) -> None:
        """Train without validation; scripts.train_value supplies evaluation."""
        interval = self.config.get("log_interval_steps", 4096)
        if interval <= 0 or interval % self.num_envs:
            raise ValueError(
                "log_interval_steps must be positive and divisible by num_envs."
            )
        while self.env_steps < self.total_env_steps:
            metrics = self.train_steps(
                min(interval, self.total_env_steps - self.env_steps)
            )
            print(metrics, flush=True)
        self.save_checkpoint(Path(self.config["save_dir"]) / "final.pt")


@register(name="pqn")
class PQNTrainer(ValueTrainer):
    """Regress fixed Q(lambda) returns from each fresh synchronous trajectory."""

    def __init__(
        self, env: RailEnv, value_agent: ShareNetworkValueAgent, config: dict
    ) -> None:
        super().__init__(env, value_agent, config)
        self.trace_lambda = config["trace_lambda"]
        if not 0 <= self.trace_lambda <= 1:
            raise ValueError("trace_lambda must be in [0, 1].")

    def _train_steps(
        self, num_steps: int, exploration: ExplorationStats
    ) -> list[tuple[float, float, int]]:
        losses = []
        for start in range(0, num_steps // self.num_envs, self.rollout_steps):
            length = min(self.rollout_steps, num_steps // self.num_envs - start)
            transitions = [self._collect_step(exploration) for _ in range(length)]
            rollout = {
                key: torch.stack([step[key] for step in transitions])
                for key in transitions[0]
            }
            with torch.no_grad():
                next_values = masked_max(
                    self.model.forward_features(rollout["next_observations"]),
                    rollout["next_masks"].to(self.device),
                    rollout["dones"].to(self.device),
                ).cpu()
                targets = q_lambda_returns(
                    rollout["rewards"],
                    rollout["dones"],
                    rollout["valid"],
                    next_values,
                    self.gamma,
                    self.trace_lambda,
                )
            valid = rollout["valid"]
            observations, actions, targets = (
                rollout["observations"][valid],
                rollout["actions"][valid],
                targets[valid],
            )
            # Targets are computed once with collection weights, then held fixed.
            for _ in range(self.update_epochs):
                indices = torch.randperm(len(actions))
                for offset in range(0, len(indices), self.batch_size):
                    batch = indices[offset : offset + self.batch_size]
                    loss, error = self._regress(
                        observations[batch], actions[batch], targets[batch]
                    )
                    losses.append((loss, error, len(batch)))
        return losses
