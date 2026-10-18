"""Render one deterministic policy episode to an MP4 video."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import torch
from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import RenderTool

from train_policy import load_config
from src.utils.action_mask import get_action_mask
from src.utils.agents import build_agents
from src.utils.environment import build_env, reset_env
from src.utils.seed import set_random_seed


def find_checkpoint(save_dir: Path, total_epochs: int) -> Path:
    """Prefer the requested epoch, otherwise use the largest numeric filename."""
    requested = save_dir / f"{total_epochs}.pt"
    if requested.is_file():
        return requested
    candidates = [p for p in save_dir.glob("*.pt") if p.is_file() and p.stem.isdecimal()]
    if not candidates:
        raise FileNotFoundError(f"No epoch checkpoints found in {save_dir}")
    return max(candidates, key=lambda p: int(p.stem))


@torch.no_grad()
def visualize(config: dict, fps: int = 7) -> Path:
    if fps <= 0:
        raise ValueError("fps must be positive.")
    checkpoint_path = find_checkpoint(Path(config["save_dir"]).expanduser(), config["total_epochs"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    epoch = int(checkpoint_path.stem)
    set_random_seed(config["trainer"]["training_seed"])
    # Recover the trained architecture; use the current config for the scenario.
    actor, _ = build_agents(checkpoint["config"])
    actor.load_state_dict(checkpoint["actor"])
    actor.to(config["trainer"]["device"]).eval()
    env = build_env(config)
    observations, infos = reset_env(env, config)
    renderer = RenderTool(env, gl="PILSVG", screen_width=1000, screen_height=1000)
    visualize_dir = Path(config["visualize_dir"]).expanduser()
    visualize_dir.mkdir(parents=True, exist_ok=True)
    video_path = visualize_dir / f"{epoch}.mp4"
    steps = 0
    reward = 0.0
    arrivals = {}

    def frame():
        renderer.render_env(show=False, show_observations=False,
                            show_predictions=False, show_inactive_agents=True, step=steps)
        return renderer.get_image()[:, :, :3]

    print(f"Checkpoint: {checkpoint_path}")
    try:
        # Stream frames directly to the encoder, including initial and terminal states.
        with imageio.get_writer(video_path, fps=fps, codec="libx264",
                                pixelformat="yuv420p", macro_block_size=2) as writer:
            writer.append_data(frame())
            while True:
                _, _, actions = actor(
                    observations, infos, action_mask=get_action_mask(env, infos),
                    deterministic=True,
                )
                observations, rewards, dones, infos = env.step(actions)
                steps += 1
                reward += sum(rewards.values())
                for agent in env.agents:
                    if agent.state == TrainState.DONE:
                        arrivals.setdefault(agent.handle, steps)
                writer.append_data(frame())
                if dones["__all__"]:
                    break
    finally:
        renderer.close_window()
    print(f"Video: {video_path} | steps={steps} | "
          f"arrived={len(arrivals)}/{len(env.agents)} | "
          f"reward={reward / len(env.agents):.4f} | arrival_steps={arrivals}")
    return video_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parent / "config/policy/single_map/ppo_demo_no_intermediate_stops.yaml",
    )
    parser.add_argument("--fps", type=int, default=7, help="Video frames per second (default: 7).")
    args = parser.parse_args()
    visualize(load_config(args.config), fps=args.fps)


if __name__ == "__main__":
    main()
