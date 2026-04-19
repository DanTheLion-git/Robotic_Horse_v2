#!/usr/bin/env python3
"""
evaluate.py — Visualize a trained Highland Cow walking policy

Usage:
    python scripts/evaluate.py                                    # Use best model
    python scripts/evaluate.py --model checkpoints/cow_walk_final.zip  # Specific model
    python scripts/evaluate.py --no-render                        # Headless metrics

Keyboard Controls (visual mode):
    W / S  — increase / decrease forward speed
    A / D  — turn left / right
    SPACE  — stop (zero command)
    Q      — quit
"""

import argparse
import os
import sys
import time
import threading
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


def evaluate_headless(model, vecnorm, n_episodes=10):
    """Run evaluation without rendering."""
    env = HighlandCowWalkEnv(randomize=False, cmd_vx=1.0, cmd_yaw=0.0)

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
    print(f"  Mean speed:   {np.mean(all_speeds):.2f} m/s")
    print(f"  Mean height:  {np.mean(all_heights):.2f} m")
    print("=" * 50)


class KeyboardController:
    """Thread-safe keyboard input for WASD control."""

    def __init__(self):
        self.cmd_vx = 0.0
        self.cmd_yaw = 0.0
        self.running = True
        self._lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self._input_loop, daemon=True)
        t.start()

    def _input_loop(self):
        """Read single keystrokes (cross-platform)."""
        try:
            import tty
            import termios
            import select

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                while self.running:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        ch = sys.stdin.read(1).lower()
                        self._handle_key(ch)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except (ImportError, termios.error):
            # Windows fallback
            try:
                import msvcrt
                while self.running:
                    if msvcrt.kbhit():
                        ch = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                        self._handle_key(ch)
                    else:
                        time.sleep(0.05)
            except ImportError:
                print("  [Warning] No keyboard input available")

    def _handle_key(self, ch):
        with self._lock:
            if ch == 'w':
                self.cmd_vx = min(self.cmd_vx + 0.2, 2.0)
            elif ch == 's':
                self.cmd_vx = max(self.cmd_vx - 0.2, -0.5)
            elif ch == 'a':
                self.cmd_yaw = min(self.cmd_yaw + 0.2, 1.0)
            elif ch == 'd':
                self.cmd_yaw = max(self.cmd_yaw - 0.2, -1.0)
            elif ch == ' ':
                self.cmd_vx = 0.0
                self.cmd_yaw = 0.0
            elif ch == 'q':
                self.running = False

    def get_command(self):
        with self._lock:
            return self.cmd_vx, self.cmd_yaw


def evaluate_visual(model, vecnorm):
    """Run with MuJoCo viewer + WASD keyboard control."""
    env = HighlandCowWalkEnv(randomize=False, cmd_vx=0.0, cmd_yaw=0.0)
    obs, _ = env.reset()
    if vecnorm is not None:
        obs = vecnorm.normalize_obs(obs)

    m = env.model
    d = env.data

    kb = KeyboardController()
    kb.start()

    print("\nLaunching MuJoCo viewer...")
    print("  W/S — forward/backward speed")
    print("  A/D — turn left/right")
    print("  SPACE — stop")
    print("  Q — quit")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        step = 0
        total_reward = 0

        while viewer.is_running() and kb.running:
            cmd_vx, cmd_yaw = kb.get_command()
            env.set_command(cmd_vx, cmd_yaw)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if vecnorm is not None:
                obs = vecnorm.normalize_obs(obs)
            total_reward += reward
            step += 1

            viewer.sync()

            if step % 100 == 0:
                print(f"  Step {step}: "
                      f"cmd=[{cmd_vx:.1f}, {cmd_yaw:.1f}], "
                      f"vel={info.get('forward_vel', 0):.2f} m/s, "
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

    kb.running = False
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained walking policy")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model .zip (default: best model)")
    parser.add_argument("--no-render", action="store_true",
                        help="Run headless evaluation only")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (headless mode)")
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
        evaluate_headless(model, vecnorm, args.episodes)
    else:
        evaluate_visual(model, vecnorm)


if __name__ == "__main__":
    main()
