# Robotic Horse v2 — MuJoCo + Reinforcement Learning

Train a Highland Cow quadruped to walk using reinforcement learning in MuJoCo.

## Why v2?

v1 (Gazebo/ROS2) used position-controlled joints with hand-coded gait trajectories.
This approach is fundamentally limited — real quadruped robots (MIT Cheetah, ANYmal, Unitree)
all use **torque control + learned policies** via RL. This project trains a walking policy
from scratch using PPO in MuJoCo, which provides much better physics simulation for contact-rich
locomotion.

## Project Structure

```
Robotic_Horse_v2/
├── models/
│   └── highland_cow.xml      # MuJoCo MJCF model (13 actuated DOF)
├── envs/
│   ├── __init__.py
│   └── cow_walk_env.py        # Gymnasium environment
├── scripts/
│   ├── train.py               # PPO training (Stable-Baselines3)
│   └── evaluate.py            # Visualization & metrics
├── checkpoints/               # Saved models (auto-created)
├── runs/                      # TensorBoard logs (auto-created)
├── reference/                 # Reference images
├── requirements.txt
└── README.md
```

## Setup

### Requirements
- Python 3.9+ (3.10 or 3.11 recommended)
- Works on Linux, Windows, macOS
- GPU optional (2-3× speedup with NVIDIA CUDA)
- 8-16 GB RAM recommended for 8 parallel environments

### Install

```bash
cd Robotic_Horse_v2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

## Training

### Basic training (2M steps, ~1-2 hours on CPU)

```bash
python scripts/train.py
```

### Options

```bash
python scripts/train.py --steps 5000000      # Train longer
python scripts/train.py --n-envs 4           # Fewer parallel envs (less RAM)
python scripts/train.py --device cuda        # Force GPU
python scripts/train.py --target-speed 1.5   # Train for faster walking
python scripts/train.py --resume             # Continue from checkpoint
```

### Monitor training

```bash
tensorboard --logdir runs/
```

Key metrics to watch:
- `ep_rew_mean`: should steadily increase
- `ep_len_mean`: should increase toward 2000 (full episodes)
- Forward velocity should approach target (1.0 m/s)

## Evaluation

### Visual (MuJoCo viewer)

```bash
python scripts/evaluate.py
```

### Headless metrics

```bash
python scripts/evaluate.py --no-render --episodes 20
```

### Specific model

```bash
python scripts/evaluate.py --model checkpoints/cow_walk_500000_steps.zip
```

## The Robot

### Dimensions (150cm Highland Cow)
- Body: ~107 kg, split into rear (48kg) and front (59kg)
- Front legs: thigh 38cm, shank 34cm, cannon 18cm
- Rear legs: thigh 50cm, shank 45cm, cannon 22cm
- 13 actuated joints: spine_yaw + 4×(hip_yaw, thigh_pitch, knee_pitch)
- 4 passive cannon joints (coupled via equality constraints)

### Passive Mechanisms
- **Front cannon**: parallelogram linkage keeps metacarpal near-vertical
- **Rear cannon**: reciprocal apparatus couples stifle↔hock (bovine anatomy)

### Reward Function
The RL agent learns to maximize:
- Forward velocity (tracking target speed)
- Staying upright (alive bonus)

While minimizing:
- Energy use (torque × velocity)
- Body tilt (orientation penalty)
- Jerky motion (action rate penalty)
- Sideways drift (lateral velocity penalty)

## Next Steps

After stable walking is achieved:
1. Add terrain randomization (slopes, bumps, curbs)
2. Train trot gait (target: 2.0 m/s)
3. Add carriage as trailing body, train pulling
4. Export policy for Gazebo/ROS2 deployment
5. Use force data to finalize motor selection & BOM
