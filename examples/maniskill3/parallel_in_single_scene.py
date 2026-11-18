"""Small ManiSkill PickCube demo; run with the separate ManiSkill Python env."""

from __future__ import annotations

import argparse

import gymnasium as gym
import mani_skill.envs  # noqa: F401 - registers PickCube-v1 with Gymnasium


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=100, help="0 runs until Ctrl+C")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render", action="store_true", help="show ManiSkill's interactive viewer")
    args = parser.parse_args()
    num_envs = args.num_envs if args.num_envs is not None else (1 if args.render else 4)
    if num_envs < 1 or args.steps < 0:
        parser.error("--num-envs must be positive and --steps must be >= 0")

    options = dict(
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        num_envs=num_envs,
        parallel_in_single_scene=not args.render,
    )
    if args.render:
        options["render_mode"] = "human"
    env = gym.make("PickCube-v1", **options)
    try:
        env.reset(seed=args.seed)
        step = 0
        while args.steps == 0 or step < args.steps:
            env.step(env.action_space.sample())
            if args.render:
                env.render()
            step += 1
        print(f"PickCube-v1: {num_envs} environments, {step} steps completed", flush=True)
    except KeyboardInterrupt:
        print("Stopped by user", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
