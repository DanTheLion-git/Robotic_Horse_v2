#!/usr/bin/env python3
"""
train.py — Train Highland Cow walking policy using PPO (Stable-Baselines3)

Supports curriculum learning: starts with target_speed=0 (standing),
gradually increases to walking speed as the cow learns balance.

Usage:
    python scripts/train.py                     # Full curriculum (stand → walk)
    python scripts/train.py --steps 5000000     # Train longer
    python scripts/train.py --resume            # Resume from latest checkpoint
    python scripts/train.py --no-curriculum     # Skip standing phase

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.utils import set_random_seed

from envs.cow_walk_env import HighlandCowWalkEnv
import gymnasium as gym


def make_env(rank, seed=0, target_speed=1.0, randomize=True):
    """Create a single environment instance."""
    def _init():
        env = HighlandCowWalkEnv(target_speed=target_speed, randomize=randomize)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed + rank)
    return _init


class CurriculumCallback(BaseCallback):
    """Gradually increase target_speed across training.

    Phase 1 (0 to ramp_start): target_speed = 0 (learn to stand)
    Phase 2 (ramp_start to ramp_end): linearly ramp to final_speed
    Phase 3 (ramp_end to end): target_speed = final_speed
    """

    def __init__(self, final_speed=1.0, ramp_start_frac=0.15, ramp_end_frac=0.5,
                 verbose=0):
        super().__init__(verbose)
        self.final_speed = final_speed
        self.ramp_start_frac = ramp_start_frac
        self.ramp_end_frac = ramp_end_frac
        self._last_speed = -1.0

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.model._total_timesteps
        if progress < self.ramp_start_frac:
            speed = 0.0
        elif progress < self.ramp_end_frac:
            ramp_progress = ((progress - self.ramp_start_frac)
                             / (self.ramp_end_frac - self.ramp_start_frac))
            speed = self.final_speed * ramp_progress
        else:
            speed = self.final_speed

        # Update all sub-environments
        if abs(speed - self._last_speed) > 0.01:
            env = self.training_env
            # Access underlying envs through VecNormalize → VecMonitor → SubprocVecEnv
            base_env = env
            while hasattr(base_env, 'venv'):
                base_env = base_env.venv
            for i in range(base_env.num_envs):
                base_env.env_method("set_target_speed", speed, indices=[i])
            self._last_speed = speed
            if self.verbose:
                print(f"\n  [Curriculum] step {self.num_timesteps:,} "
                      f"({progress:.0%}): target_speed={speed:.2f} m/s")
        return True


def main():
    parser = argparse.ArgumentParser(description="Train Highland Cow walking gait")
    parser.add_argument("--steps", type=int, default=3_000_000,
                        help="Total training timesteps (default: 3M)")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--target-speed", type=float, default=1.0,
                        help="Final target forward speed in m/s (default: 1.0)")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Skip curriculum, train at target-speed from start")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cpu', 'cuda'")
    args = parser.parse_args()

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint_dir = os.path.join(project_dir, "checkpoints")
    log_dir = os.path.join(project_dir, "runs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Start with target_speed=0 for curriculum, or full speed if --no-curriculum
    initial_speed = args.target_speed if args.no_curriculum else 0.0

    print("=" * 60)
    print("  Highland Cow — PPO Gait Training (v3)")
    print("=" * 60)
    print(f"  Steps:          {args.steps:,}")
    print(f"  Parallel envs:  {args.n_envs}")
    print(f"  Target speed:   {args.target_speed} m/s")
    print(f"  Curriculum:     {'OFF' if args.no_curriculum else 'ON (stand → walk)'}")
    print(f"  Device:         {args.device}")
    print(f"  Checkpoints:    {checkpoint_dir}")
    print(f"  Logs:           {log_dir}")
    print("=" * 60)

    # Create parallel training environments
    train_envs = SubprocVecEnv(
        [make_env(i, args.seed, initial_speed) for i in range(args.n_envs)]
    )
    train_envs = VecMonitor(train_envs)
    train_envs = VecNormalize(train_envs, norm_obs=True, norm_reward=True,
                              clip_obs=10.0, clip_reward=10.0)

    # Eval environment (single, unnormalized rewards for true metrics)
    eval_env = SubprocVecEnv([make_env(100, args.seed, args.target_speed,
                                       randomize=False)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    ppo_kwargs = dict(
        policy="MlpPolicy",
        env=train_envs,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=1.0,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
            log_std_init=-1.5,
        ),
        verbose=1,
        tensorboard_log=log_dir,
        device=args.device,
        seed=args.seed,
    )

    if args.resume:
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
    callback_list = [checkpoint_cb, eval_cb]

    if not args.no_curriculum:
        curriculum_cb = CurriculumCallback(
            final_speed=args.target_speed,
            ramp_start_frac=0.15,   # first 15% = standing only
            ramp_end_frac=0.50,     # ramp speed from 15% to 50%
            verbose=1,
        )
        callback_list.append(curriculum_cb)

    callbacks = CallbackList(callback_list)

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
