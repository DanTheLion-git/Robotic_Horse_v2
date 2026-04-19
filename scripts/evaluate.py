#!/usr/bin/env python3
"""
evaluate.py — Visualize a trained Highland Cow walking policy

Usage:
    python scripts/evaluate.py                                    # Use best model
    python scripts/evaluate.py --model checkpoints/cow_walk_final.zip  # Specific model
    python scripts/evaluate.py --no-render                        # Headless metrics
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.cow_walk_env import HighlandCowWalkEnv, CONTROL_DT

import mujoco
import mujoco.viewer


def load_model(model_path, checkpoint_dir):
    """Load PPO model and VecNormalize stats."""
    model = PPO.load(model_path)

    vecnorm_path = os.path.join(checkpoint_dir, "vecnormalize.pkl")
    vecnorm = None
    if os.path.exists(vecnorm_path):
        vecnorm = VecNormalize.load(
            vecnorm_path,
            DummyVecEnv([lambda: HighlandCowWalkEnv(randomize=False)]))
        vecnorm.training = False
        vecnorm.norm_reward = False
        print(f"Loaded VecNormalize from {vecnorm_path}")
    return model, vecnorm


def evaluate_headless(model, vecnorm, n_episodes=10, target_speed=1.0):
    """Run evaluation without rendering."""
    env = HighlandCowWalkEnv(target_speed=target_speed, randomize=False)

    all_rewards, all_lengths, all_speeds, all_heights = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        if vecnorm is not None:
            obs = vecnorm.normalize_obs(obs)

        total_reward = 0
        speeds, heights = [], []
        step = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if vecnorm is not None:
                obs = vecnorm.normalize_obs(obs)

            total_reward += reward
            speeds.append(info.get("forward_vel", 0))
            heights.append(info.get("height", 0))
            step += 1

            if terminated or truncated:
                break

        all_rewards.append(total_reward)
        all_lengths.append(step)
        all_speeds.append(np.mean(speeds))
        all_heights.append(np.mean(heights))

        print(f"  Episode {ep+1}/{n_episodes}: "
              f"reward={total_reward:.1f}, steps={step}, "
              f"avg_speed={np.mean(speeds):.2f} m/s, "
              f"avg_height={np.mean(heights):.2f} m")

    env.close()

    print("\n" + "=" * 50)
    print("  Evaluation Summary")
    print("=" * 50)
    print(f"  Mean reward:  {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"  Mean length:  {np.mean(all_lengths):.0f} steps")
    print(f"  Mean speed:   {np.mean(all_speeds):.2f} m/s (target: {target_speed})")
    print(f"  Mean height:  {np.mean(all_heights):.2f} m")
    print("=" * 50)


def evaluate_visual(model, vecnorm, target_speed=1.0):
    """Run with MuJoCo viewer for visual evaluation."""
    env = HighlandCowWalkEnv(target_speed=target_speed, randomize=False)
    obs, _ = env.reset()
    if vecnorm is not None:
        obs = vecnorm.normalize_obs(obs)

    m = env.model
    d = env.data

    print("\nLaunching MuJoCo viewer...")
    print("  Press ESC to quit")
    print("  The cow should start walking forward")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        step = 0
        total_reward = 0

        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if vecnorm is not None:
                obs = vecnorm.normalize_obs(obs)
            total_reward += reward
            step += 1

            viewer.sync()

            if step % 100 == 0:
                print(f"  Step {step}: "
                      f"speed={info.get('forward_vel', 0):.2f} m/s, "
                      f"height={info.get('height', 0):.2f} m, "
                      f"reward={total_reward:.1f}")

            if terminated or truncated:
                print(f"\n  Episode ended at step {step} "
                      f"({'terminated' if terminated else 'truncated'})")
                print(f"  Total reward: {total_reward:.1f}")
                obs, _ = env.reset()
                if vecnorm is not None:
                    obs = vecnorm.normalize_obs(obs)
                step = 0
                total_reward = 0

            time.sleep(CONTROL_DT)

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained walking policy")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model .zip (default: best model)")
    parser.add_argument("--no-render", action="store_true",
                        help="Run headless evaluation only")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (headless mode)")
    parser.add_argument("--target-speed", type=float, default=1.0,
                        help="Target speed in m/s")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir = os.path.join(project_dir, "checkpoints")

    # Find model
    if args.model:
        model_path = args.model
    else:
        best_path = os.path.join(checkpoint_dir, "best", "best_model.zip")
        final_path = os.path.join(checkpoint_dir, "cow_walk_final.zip")
        if os.path.exists(best_path):
            model_path = best_path
        elif os.path.exists(final_path):
            model_path = final_path
        else:
            print("No trained model found! Run train.py first.")
            sys.exit(1)

    print(f"Loading model: {model_path}")
    model, vecnorm = load_model(model_path, checkpoint_dir)

    if args.no_render:
        evaluate_headless(model, vecnorm, args.episodes, args.target_speed)
    else:
        evaluate_visual(model, vecnorm, args.target_speed)


if __name__ == "__main__":
    main()
