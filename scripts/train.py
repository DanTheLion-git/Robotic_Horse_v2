#!/usr/bin/env python3
"""
train.py — Train Highland Cow walking policy using PPO (Stable-Baselines3)

Usage:
    python scripts/train.py                     # Train for 2M steps
    python scripts/train.py --steps 5000000     # Train longer
    python scripts/train.py --resume            # Resume from latest checkpoint

Outputs:
    checkpoints/cow_walk_*.zip   — model checkpoints every 100k steps
    checkpoints/cow_walk_final.zip — final trained model
    runs/                        — TensorBoard logs

Monitor training:
    tensorboard --logdir runs/
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList
)
from stable_baselines3.common.utils import set_random_seed

# Import and register the environment
from envs.cow_walk_env import HighlandCowWalkEnv
import gymnasium as gym


def make_env(rank, seed=0, target_speed=1.0):
    """Create a single environment instance."""
    def _init():
        env = HighlandCowWalkEnv(target_speed=target_speed, randomize=True)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed + rank)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train Highland Cow walking gait")
    parser.add_argument("--steps", type=int, default=2_000_000,
                        help="Total training timesteps (default: 2M)")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--target-speed", type=float, default=1.0,
                        help="Target forward speed in m/s (default: 1.0)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cpu', 'cuda'")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir = os.path.join(project_dir, "checkpoints")
    log_dir = os.path.join(project_dir, "runs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print("  Highland Cow — PPO Gait Training")
    print("=" * 60)
    print(f"  Steps:          {args.steps:,}")
    print(f"  Parallel envs:  {args.n_envs}")
    print(f"  Target speed:   {args.target_speed} m/s")
    print(f"  Device:         {args.device}")
    print(f"  Checkpoints:    {checkpoint_dir}")
    print(f"  Logs:           {log_dir}")
    print("=" * 60)

    # Create parallel training environments
    train_envs = SubprocVecEnv(
        [make_env(i, args.seed, args.target_speed) for i in range(args.n_envs)]
    )
    train_envs = VecMonitor(train_envs)
    train_envs = VecNormalize(train_envs, norm_obs=True, norm_reward=True,
                              clip_obs=10.0, clip_reward=10.0)

    # Create evaluation environment (single, unnormalized for true metrics)
    eval_env = SubprocVecEnv([make_env(100, args.seed, args.target_speed)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    # PPO hyperparameters tuned for quadruped locomotion
    # Based on Rudin et al. (2022) and Legged Gym configs
    ppo_kwargs = dict(
        policy="MlpPolicy",
        env=train_envs,
        learning_rate=3e-4,
        n_steps=4096,           # more steps per update for stable learning
        batch_size=256,
        n_epochs=5,             # fewer epochs to prevent overfitting
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.005,         # less entropy for more exploitation
        vf_coef=0.5,
        max_grad_norm=1.0,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
            log_std_init=-1.5,   # smaller initial std (conservative PD targets)
        ),
        verbose=1,
        tensorboard_log=log_dir,
        device=args.device,
        seed=args.seed,
    )

    if args.resume:
        # Find latest checkpoint
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir)
                              if f.startswith("cow_walk_") and f.endswith(".zip")])
        if checkpoints:
            latest = os.path.join(checkpoint_dir, checkpoints[-1])
            print(f"\nResuming from: {latest}")
            model = PPO.load(latest, env=train_envs, **{k: v for k, v in ppo_kwargs.items()
                                                         if k not in ("policy", "env")})
        else:
            print("\nNo checkpoint found, starting fresh")
            model = PPO(**ppo_kwargs)
    else:
        model = PPO(**ppo_kwargs)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path=checkpoint_dir,
        name_prefix="cow_walk",
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(checkpoint_dir, "best"),
        log_path=log_dir,
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )
    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # Train!
    print("\nStarting training...\n")
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(checkpoint_dir, "cow_walk_final")
    model.save(final_path)
    train_envs.save(os.path.join(checkpoint_dir, "vecnormalize.pkl"))
    print(f"\nFinal model saved: {final_path}.zip")
    print(f"VecNormalize stats: {os.path.join(checkpoint_dir, 'vecnormalize.pkl')}")

    train_envs.close()
    eval_env.close()


if __name__ == "__main__":
    main()
