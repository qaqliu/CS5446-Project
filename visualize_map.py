"""Render the configured map and initial train positions without a checkpoint."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
from flatland.utils.rendertools import RenderTool

from train_policy import load_config
from src.utils.environment import build_env, reset_env


def visualize_map(config: dict) -> Path:
    # Use the same seeded map/task generation as training.
    env = build_env(config)
    reset_env(env, config)
    output_dir = Path(config["visualize_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "map_preview.png"
    renderer = RenderTool(env, gl="PILSVG", screen_width=1200, screen_height=1200)
    try:
        renderer.render_env(
            show=False, show_observations=False, show_predictions=False,
            show_inactive_agents=True, show_rowcols=True,
        )
        imageio.imwrite(output_path, renderer.get_image())
    finally:
        renderer.close_window()
    print(f"Map: {output_path} | size={env.width}x{env.height} | "
          f"agents={len(env.agents)} | map_seed={config['map_seed']}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parent / "config/policy/single_map/ppo_demo_no_intermediate_stops.yaml",
    )
    args = parser.parse_args()
    visualize_map(load_config(args.config))


if __name__ == "__main__":
    main()
