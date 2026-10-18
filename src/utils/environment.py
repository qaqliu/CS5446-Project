"""Build and reset Flatland environments from experiment configuration."""

from functools import partial

from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rewards import DefaultRewards
from flatland.envs.timetable_generators import timetable_generator


def _make_timetable(agents, distance_map, agents_hints, np_random, *, max_episode_steps):
    timetable = timetable_generator(agents, distance_map, agents_hints, np_random)
    if max_episode_steps is not None:
        timetable = timetable._replace(max_episode_steps=max_episode_steps)
    return timetable


def build_env(config: dict) -> RailEnv:
    """Construct the configured environment; call reset_env before using it."""
    obs_config = config["observation"]
    obs_type = obs_config["type"]
    if obs_type == "global":
        obs_builder = GlobalObsForRailEnv()
    elif obs_type in ("tree", "flattened_normalized_tree"):
        # predictor: null, or {max_depth: 20} to enable shortest-path prediction.
        predictor_config = obs_config.get("predictor")
        predictor = (
            ShortestPathPredictorForRailEnv(**predictor_config)
            if predictor_config is not None else None
        )
        kwargs = {"max_depth": obs_config["max_depth"], "predictor": predictor}
        if obs_type == "tree":
            obs_builder = TreeObsForRailEnv(**kwargs)
        else:
            from flatland.ml.observations.flatten_tree_observation_for_rail_env import (
                FlattenedNormalizedTreeObsForRailEnv,
            )
            obs_builder = FlattenedNormalizedTreeObsForRailEnv(
                observation_radius=obs_config["observation_radius"], **kwargs
            )
    else:
        raise ValueError(f"Unknown observation type: {obs_type!r}")

    if config["reward"]["type"] != "default":
        raise ValueError("Only the default reward is implemented.")
    stops = config["intermediate_stops"]
    line_length = stops["line_length"] if stops["enabled"] else 2
    if stops["enabled"] and line_length < 3:
        raise ValueError("Intermediate stops require line_length >= 3.")
    max_steps = config["max_episode_steps"]
    if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
        raise ValueError("max_episode_steps must be null or a positive integer.")

    return RailEnv(
        width=config["width"],
        height=config["height"],
        number_of_agents=config["number_of_agents"],
        rail_generator=sparse_rail_generator(
            max_num_cities=config["max_num_cities"],
            grid_mode=config["grid_mode"],
            max_rails_between_cities=config["max_rails_between_cities"],
            max_rail_pairs_in_city=config["max_rail_pairs_in_city"],
        ),
        line_generator=sparse_line_generator(
            speed_ratio_map=config["speed_ratio_map"], line_length=line_length
        ),
        obs_builder_object=obs_builder,
        rewards=DefaultRewards(),
        timetable_generator=partial(_make_timetable, max_episode_steps=max_steps),
        random_seed=config["map_seed"],
        remove_agents_at_target=True,
    )


def reset_env(env: RailEnv, config: dict) -> tuple[dict, dict]:
    """Reset using a seeded first map, then retain or regenerate the scenario.

    Fixed-map mode retains both rails and train tasks/timetables. Random-map
    mode advances the seeded RNG to generate new scenarios on later resets.
    Use this helper for subsequent episodes so random_map remains effective.
    """
    first_reset = env.rail is None
    regenerate = first_reset or config["random_map"]
    return env.reset(
        regenerate_rail=regenerate,
        regenerate_schedule=regenerate,
        random_seed=config["map_seed"] if first_reset else None,
    )
