"""Exact interaction budgets around episode-based PPO updates.

Intermediate checkpoint boundaries retain episodes without forcing an update.
Weights-only checkpoints support evaluation, not exact optimizer/env resumption.
"""

from copy import deepcopy
from pathlib import Path

import torch
from flatland.envs.step_utils.states import TrainState

from src.utils.agents import build_agents
from src.utils.environment import build_env
from src.utils.exploration import ExplorationStats

from .policy_trainer import PPODemoTrainer, PPOTrainer


def join_fragments(fragments: list[dict]) -> dict:
    """Join frozen-weight fragments without adding artificial terminal steps."""
    result = {}
    for key, first in fragments[0].items():
        values = [fragment[key] for fragment in fragments]
        if key == "agent_ids":
            if any(value != first for value in values):
                raise ValueError("Agent order changed within a single episode.")
            result[key] = first
        elif isinstance(first, list):
            result[key] = [item for value in values for item in value]
        else:
            result[key] = torch.cat(values, dim=0)
    return result


class ExactPPO:
    """Bound collection while preserving upstream episode-level update cadence."""

    def __init__(self, config: dict) -> None:
        self.actor, self.critic = build_agents(config)
        cls = PPOTrainer if config["trainer"]["type"] == "ppo" else PPODemoTrainer
        self.trainer = cls(build_env(config), self.actor, self.critic, config)
        self.config = self.trainer.config
        self.env_steps = self.agent_transitions = self.gradient_updates = 0
        self.optimizer_steps = self.completed_rollouts = 0
        self.fragments = []
        optimizers = (
            [self.trainer.optimizer]
            if cls is PPOTrainer
            else [self.trainer.actor_optimizer, self.trainer.critic_optimizer]
        )
        for optimizer in optimizers:
            optimizer.register_step_post_hook(self._count_optimizer)

    def _count_optimizer(self, optimizer, args, kwargs) -> None:
        self.optimizer_steps += 1

    def train_steps(self, steps: int) -> dict:
        if steps <= 0 or self.env_steps + steps > self.config["total_env_steps"]:
            raise ValueError("Training chunk exceeds the positive remaining budget.")
        remaining, stats, losses = steps, ExplorationStats(), []

        def record(module, args, kwargs, output):
            masks, logits, actions = output
            observations, infos = args[:2]
            handles = list(observations)
            legal = torch.stack([masks[h] for h in handles]).cpu()
            values = torch.stack([logits[h].detach().cpu() for h in handles])
            probabilities = values.masked_fill(~legal, -torch.inf).softmax(-1)
            valid = torch.tensor(
                [infos["state"][h] != TrainState.DONE for h in handles]
            )
            stats.update(
                probabilities, legal, torch.tensor([actions[h] for h in handles]), valid
            )

        hook = self.actor.register_forward_hook(record, with_kwargs=True)
        try:
            while remaining:
                trainer = self.trainer
                trainer.entropy_coef = trainer._get_entropy_coef(
                    self.completed_rollouts + 1
                )
                if isinstance(trainer, PPOTrainer):
                    rollout = trainer.collect_rollout(max_steps=remaining)
                else:
                    trainer.rollout_steps = min(remaining, self.config["rollout_steps"])
                    rollout = trainer.collect_rollout()
                consumed = len(rollout["actions"])
                self.env_steps += consumed
                self.agent_transitions += int(rollout["valid_mask"].sum())
                remaining -= consumed
                self.fragments.append(rollout)
                final = self.env_steps == self.config["total_env_steps"]
                if (
                    not isinstance(trainer, PPOTrainer)
                    or trainer._episode_done
                    or final
                ):
                    rollout = trainer.compute_advantages(join_fragments(self.fragments))
                    self.fragments = []
                    for _ in range(trainer.update_epochs):
                        metrics = trainer.update_one_epoch(rollout)
                        losses.append(metrics)
                        self.gradient_updates += metrics["num_updates"]
                        if metrics["early_stop"]:
                            break
                    if isinstance(trainer, PPOTrainer):
                        trainer._step_schedulers()
                    else:
                        trainer.actor_scheduler.step()
                        trainer.critic_scheduler.step()
                    self.completed_rollouts += 1
        finally:
            hook.remove()
        summary = dict(
            env_steps=self.env_steps,
            agent_transitions=self.agent_transitions,
            gradient_updates=self.gradient_updates,
            optimizer_steps=self.optimizer_steps,
            completed_rollouts=self.completed_rollouts,
            pending_episode_steps=sum(len(x["actions"]) for x in self.fragments),
            **stats.summary(),
        )
        for key in ("actor_loss", "value_loss", "entropy", "approx_kl"):
            weight = "critic_samples" if key == "value_loss" else "actor_samples"
            count = sum(item[weight] for item in losses)
            summary[key] = (
                sum(item[key] * item[weight] for item in losses) / count
                if count
                else None
            )
        return summary

    def save_checkpoint(self, path: Path) -> None:
        """Save evaluation weights and counters, without exact-resume state."""
        config = deepcopy(self.config)
        config.get("wandb", {}).pop("api_key", None)
        torch.save(
            dict(
                actor=self.actor.state_dict(),
                critic=self.critic.state_dict(),
                config=config,
                env_steps=self.env_steps,
                total_env_steps=self.env_steps,
                gradient_updates=self.gradient_updates,
                optimizer_steps=self.optimizer_steps,
                pending_episode_steps=sum(len(x["actions"]) for x in self.fragments),
            ),
            path,
        )
