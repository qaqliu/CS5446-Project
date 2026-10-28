# Checkpoints

Names match `config/protocol/<method>.yaml` and the results report. Put this folder at the repo root as `checkpoints/` (git-ignored).

- `best.pt`: checkpoint selected on validation maps; all reported numbers use it.
- `final.pt`: last checkpoint; use it as `init_from` to continue training.
- `evaluation.csv`: every validation/test episode behind the reported numbers.
- Load a policy in code: `src.eval.runner.load_actor_checkpoint("checkpoints/<method>/seed42/best.pt")`.

Held-out completion at each train count (sampled policy, mean over 3 seeds). Hybrids reuse the base method's `best.pt` with a departure cap chosen per size on validation.

| Method | Folder | Recipe | Seeds | Selected epoch per seed | Env steps (selected / trained) | 5 trains | 10 trains | 20 trains | 50 trains | 100 trains |
|---|---|---|---|---|---|---|---|---|---|---|
| Shared PPO, random maps, 3M | `shared_ppo_random_maps/` | `config/protocol/shared_ppo_random_maps.yaml` | 3 | 2700, 3000, 1800 | 2.6M / 3.1M | 71% | 51% | 25% | 9% | – |
| Shared PPO, random maps, 10M | `shared_ppo_random_maps_10M/` | `config/protocol/shared_ppo_random_maps_10M.yaml` | 3 | 5000, 5000, 4000 | 4.8M / 10.2M | 83% | 64% | 38% | 14% | 7% |
| Shared PPO, 15M (10M + 5M with LR decay) | `shared_ppo_15M_lr_decay/` | `config/protocol/shared_ppo_15M_lr_decay.yaml` | 3 | 2450, 4900, 4900 | 14.4M / 15.3M | 90% | 78% | 45% | 15% | – |
| Shared PPO, 15M (10M + 5M curriculum at 10 trains) | `shared_ppo_15M_curriculum10/` | `config/protocol/shared_ppo_15M_curriculum10.yaml` | 3 | 2940, 3920, 4410 | 14.1M / 15.3M | 90% | 80% | 47% | 17% | – |
| Shared PPO, 13M (10M + 3M curriculum at 20 trains) | `shared_ppo_13M_curriculum20/` | `config/protocol/shared_ppo_13M_curriculum20.yaml` | 3 | 2930, 879, 2930 | 12.5M / 13.2M | 89% | 76% | 42% | 16% | – |
| Shared PPO, one fixed map, 3M | `shared_ppo_fixed_map/` | `config/protocol/shared_ppo_fixed_map.yaml` | 3 | 600, 900, 300 | 0.6M / 3.1M | 37% | 27% | 11% | 5% | – |
| Shared PPO, 20-map pool, 3M | `shared_ppo_pool20/` | `config/protocol/shared_ppo_pool20.yaml` | 3 | 2100, 3000, 1800 | 2.4M / 3.1M | 57% | 40% | 21% | 8% | – |
| Shared PPO, champion reward, 3M | `shared_ppo_random_maps_champion_reward/` | `config/protocol/shared_ppo_random_maps_champion_reward.yaml` | 3 | 3000, 3000, 2400 | 2.9M / 3.1M | 74% | 61% | 31% | 11% | – |
| Shared PPO + train ID, 3M | `shared_id_ppo_random_maps/` | `config/protocol/shared_id_ppo_random_maps.yaml` | 3 | 3000, 2100, 2700 | 2.7M / 3.1M | 76% | – | – | – | – |
| Independent PPO, 3M | `independent_ppo_random_maps/` | `config/protocol/independent_ppo_random_maps.yaml` | 3 | 2100, 3000, 2100 | 2.5M / 3.1M | 39% | – | – | – | – |
| Communication + grouped entropy | `comm_ppo_random_maps/` | `config/protocol/comm_ppo_random_maps.yaml` | 3 | 7500, 6000, 7500 | – | 66% | 42% | 19% | 8% | – |
| Communication, plain entropy | `comm_ppo_random_maps_no_grouped_entropy/` | `config/protocol/comm_ppo_random_maps_no_grouped_entropy.yaml` | 3 | 4500, 5250, 6750 | – | 64% | 42% | 17% | 7% | – |
| Communication, multi-size (5/10/20) | `comm_ppo_multi_size/` | `config/protocol/comm_ppo_multi_size.yaml` | 3 | 5250, 7500, 6750 | 2.9M / 3.3M | 69% | 47% | 25% | 10% | – |
| Shared PPO, 15M (10M + 5M curriculum at 10 trains) + departure cap | `shared_ppo_15M_curriculum10_with_cap/` | `config/protocol/shared_ppo_15M_curriculum10.yaml` + cap | 3 | 2940, 3920, 4410 | same as base | 92% | 87% | 80% | 48% | 28% |
| Shared PPO, 15M (10M + 5M with LR decay) + departure cap | `shared_ppo_15M_lr_decay_with_cap/` | `config/protocol/shared_ppo_15M_lr_decay.yaml` + cap | 3 | 2450, 4900, 4900 | same as base | 92% | 84% | 76% | 43% | 27% |
| Shared PPO, random maps, 10M + departure cap | `shared_ppo_random_maps_10M_with_cap/` | `config/protocol/shared_ppo_random_maps_10M.yaml` + cap | 3 | 5000, 5000, 4000 | same as base | 89% | 74% | 73% | 33% | 21% |

## Notes

- `shared_ppo_15M_lr_decay`: stopped at epoch 2450 (machine ran out of memory) and resumed from that checkpoint with the LR/entropy schedules continued; checkpoints after it are renumbered to follow on.
- `shared_ppo_15M_curriculum10`: stopped at epoch 1960 (machine ran out of memory) and resumed from that checkpoint with the LR/entropy schedules continued; checkpoints after it are renumbered to follow on.
- `shared_ppo_13M_curriculum20`: stopped at epoch 586 (machine ran out of memory) and resumed from that checkpoint with the LR/entropy schedules continued; checkpoints after it are renumbered to follow on.

## Departure caps chosen on validation

- `shared_ppo_random_maps_10M_with_cap` seed 42: caps {5: 2, 10: 2, 20: 4, 50: 8, 100: 8}
- `shared_ppo_random_maps_10M_with_cap` seed 43: caps {5: 2, 10: 2, 20: 2, 50: 4, 100: 8}
- `shared_ppo_random_maps_10M_with_cap` seed 44: caps {5: 8, 10: 4, 20: 4, 50: 4, 100: 8}
- `shared_ppo_15M_curriculum10_with_cap` seed 42: caps {5: 8, 10: 4, 20: 4, 50: 8, 100: 8}
- `shared_ppo_15M_curriculum10_with_cap` seed 43: caps {5: 8, 10: 2, 20: 4, 50: 4, 100: 4}
- `shared_ppo_15M_curriculum10_with_cap` seed 44: caps {5: 8, 10: 4, 20: 4, 50: 4, 100: 8}
- `shared_ppo_15M_lr_decay_with_cap` seed 42: caps {5: 8, 10: 2, 20: 4, 50: 8, 100: 8}
- `shared_ppo_15M_lr_decay_with_cap` seed 43: caps {5: 8, 10: 4, 20: 4, 50: 4, 100: 4}
- `shared_ppo_15M_lr_decay_with_cap` seed 44: caps {5: 8, 10: 4, 20: 4, 50: 8, 100: 8}
