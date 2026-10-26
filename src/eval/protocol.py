"""The shared evaluation protocol: maps, seeds, sizes and task settings.

Follows the NeurIPS 2020 Flatland evaluation generator and score
(Laurent et al., 2021, Sec. 2.3 and App. B) and the train/validation/test
protocol of the organizers' baselines (Mohanty et al., 2020, Sec. 4.1),
restricted to at most 100 trains, without malfunctions, in Flatland 4.3.

Every method is scored on the same held-out maps. Training is free (any map
distribution or strategy) as long as it never uses validation or test seeds.
"""

import math
from copy import deepcopy

# Held-out map seeds. Validation selects checkpoints and tunes settings;
# test maps produce the reported numbers and are never used for decisions.
VALIDATION_SEEDS = tuple(range(2000, 2020))
TEST_SEEDS = tuple(range(1000, 1020))
RESERVED_SEEDS = frozenset(VALIDATION_SEEDS) | frozenset(TEST_SEEDS)

# Test sizes (number of trains); the map grows with the competition formula.
TEST_SIZES = (5, 10, 20, 50)
OPTIONAL_TEST_SIZES = (100,)

# Sampled-policy episodes per test map (headline number) plus one
# deterministic episode reported separately.
SAMPLED_EPISODES = 3


def competition_map(trains: int) -> dict:
    """Map parameters of the competition evaluation generator for n trains.

    cities = n // 10 + 2, square size = ceil(sqrt(6 (ceil(4/2) + 3)^2 cities)) + 7,
    at most 2 rails between cities and 4 rails (2 pairs) inside a city,
    random city placement, episode limit t_max = 8 (w + h + n / cities).
    5 trains -> 25x25, 10 -> 29x29, 20 -> 32x32, 50 -> 40x40, 100 -> 50x50.
    """
    cities = trains // 10 + 2
    size = math.ceil(math.sqrt(6 * (math.ceil(4 / 2) + 3) ** 2 * cities)) + 7
    return dict(
        width=size, height=size, max_num_cities=cities, grid_mode=False,
        max_rails_between_cities=2, max_rail_pairs_in_city=2,
        max_episode_steps=int(8 * (2 * size + trains / cities)),
    )


def protocol_config(policy_config: dict, trains: int, map_seed: int) -> dict:
    """Environment config for one evaluation episode.

    Keeps what the policy's networks need (observation settings, models) and
    enforces the task: competition map for `trains`, unit speed for every
    train, origin-to-destination tasks without scheduled intermediate
    stations, no malfunctions, competition episode limit, fixed map seed, and a
    fixed reward (training rewards do not affect evaluation metrics).
    Trains may still stop and wait anywhere.
    """
    config = deepcopy(policy_config)
    config.update(competition_map(trains))
    config.update(
        number_of_agents=trains, map_seed=map_seed, random_map=False,
        speed_ratio_map={1.0: 1.0},
        intermediate_stops={"enabled": False, "line_length": 2},
    )
    # Metrics never read rewards; a fixed reward keeps evaluation independent of
    # each method's training reward (and supplies the team DeadlockChecker).
    config["reward"] = {"type": "champion", "config": None}
    config.pop("map_pool", None)
    config.pop("training_maps", None)
    config["trainer"] = dict(config.get("trainer", {}), device="cpu")
    config["wandb"] = {"enabled": False}
    return config


def training_sizes(policy_config: dict) -> tuple[int, ...]:
    """Train counts a policy was trained on; checkpoint selection uses only these."""
    sizes = (policy_config.get("training_maps") or {}).get("sizes")
    return tuple(sizes) if sizes else (int(policy_config["number_of_agents"]),)


def check_training_config(config: dict) -> list[str]:
    """Problems that would leak held-out maps into training (empty if none)."""
    problems = []
    maps = config.get("training_maps") or {}
    seeds = set(maps.get("pool") or [])
    if not config.get("random_map", False):
        seeds.add(config.get("map_seed"))
    leaked = sorted(s for s in seeds if s in RESERVED_SEEDS)
    if leaked:
        problems.append(
            f"training uses validation/test map seeds {leaked}; pick seeds outside "
            f"{VALIDATION_SEEDS[0]}-{VALIDATION_SEEDS[-1]} and {TEST_SEEDS[0]}-{TEST_SEEDS[-1]}"
        )
    return problems
