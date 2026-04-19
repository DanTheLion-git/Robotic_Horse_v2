# Robotic Horse v2 — Progress Summary #1

**Date:** April 19, 2026
**Project:** Highland Cow Quadruped Robot — MuJoCo RL Pipeline
**Repository:** [DanTheLion-git/Robotic_Horse_v2](https://github.com/DanTheLion-git/Robotic_Horse_v2)

---

## Background

This project is the second iteration of a robotic Highland Cow quadruped designed to pull a vis-à-vis carriage. The first version (Robotic_Horse v1) used Gazebo with ROS2 and hand-coded position-based gait control. While it could move, the walking was unnatural and unstable — the cow would buckle at the knees, tip forward during trotting, and fail to turn the carriage properly. After extensive debugging of forces, joint configurations, and gait parameters, it became clear that hand-tuning a stable quadruped gait in Gazebo was impractical.

The decision was made to start fresh with **MuJoCo** (a faster, more accurate physics simulator) and **Reinforcement Learning** (letting the robot learn to walk on its own instead of hand-coding every joint trajectory).

---

## What Was Built

### 1. Initial Pipeline Setup (Commit `91c8c63`)

The first version of the MuJoCo pipeline included:

- **`models/highland_cow.xml`** — MJCF model converted from the v1 URDF
  - 13 actuated DOF: 1 spine yaw + 4 legs × 3 joints (hip roll, hip pitch, knee)
  - 4 passive cannon bones with equality constraints coupling them to knees
  - Raw torque actuators (motor elements with direct force control)
  - Highland Cow proportions: 150cm withers height, ~115kg total mass
  - Body barrel: 1.10m × 0.44m × 0.44m (80kg)
  - Front legs: 0.45m upper + 0.45m lower segments (6.5kg each)
  - Rear legs: 0.48m upper + 0.48m lower segments (7.5kg each)

- **`envs/cow_walk_env.py`** — Gymnasium environment
  - 54-dimensional observation space (quaternion, velocities, joint states, contacts)
  - 13-dimensional action space (direct torques)
  - Basic reward: forward velocity - energy penalty - fall penalty
  - 50 settling steps at reset

- **`scripts/train.py`** — PPO training with Stable-Baselines3
  - 8 parallel vectorized environments
  - TensorBoard logging
  - Periodic checkpoint saving

- **`scripts/evaluate.py`** — Visual evaluation with MuJoCo viewer + headless metrics mode

- **`requirements.txt`** — Dependencies: mujoco ≥3.1.0, gymnasium ≥0.29.0, stable-baselines3, torch, tensorboard

- **`README.md`** — Full setup, training, and evaluation guide

### 2. Critical Bug Discovery

When running `evaluate.py`, the cow collapsed onto its chest within a single timestep. Investigation revealed **three critical bugs**:

| Bug | Cause | Effect |
|-----|-------|--------|
| **Front knee range violation** | Joint range was [-1.80, -0.10] but initial qpos was 0.0 | 0 is outside the valid range — MuJoCo applies massive constraint forces to snap the joint back, catapulting the cow |
| **Rear knee range violation** | Joint range was [0.10, 1.80] but initial qpos was 0.0 | Same explosive snap in the opposite direction |
| **No standing keyframe** | All joints initialized to 0.0 (legs perfectly straight) | Unstable configuration even without the range violations |

Additionally, **raw torque control** made RL training nearly impossible — the policy would need to simultaneously learn gravity compensation, balance, AND locomotion from scratch. Modern RL quadrupeds universally use PD position control.

### 3. Complete Rebuild (Commit `8dca455`)

Based on research into RL quadruped best practices (Rudin et al. 2022, Legged Gym, ANYmal, Mini Cheetah), the entire model and environment were rebuilt from scratch.

#### MJCF Model Changes

| Aspect | Before (v1) | After (v2) |
|--------|-------------|------------|
| **DOF** | 13 (spine + 4×3 + 4 passive) | 12 (4×3: hip_roll, hip_pitch, knee) |
| **Actuators** | Raw torque motors | PD position actuators (`<general>` with `biastype="affine"`) |
| **Initial pose** | All joints at 0.0 (invalid!) | Standing keyframe with computed angles |
| **Spine** | Articulated yaw joint | Single rigid torso (deferred) |
| **Cannon bones** | 4 passive joints with equality constraints | Removed (simplification) |
| **Collisions** | All geoms collide | Only feet collide with ground |
| **Armature** | Low (0.01) | Higher (0.05–0.08) for stability |

**PD Controller Details:**
```
force = kp × (ctrl - qpos) - kd × qvel
```
- Hip roll: kp=80, kd=4, force limit ±120 Nm
- Hip pitch: kp=200, kd=10, force limit ±350 Nm
- Knee: kp=200, kd=10, force limit ±400 Nm

**Standing Keyframe Angles:**
- Front legs: roll=±0.05, pitch=0.0, knee=-0.55
- Rear legs: roll=±0.05, pitch=-0.10, knee=-0.60
- Body spawns at z=1.02m

#### Environment Changes

| Aspect | Before (v1) | After (v2) |
|--------|-------------|------------|
| **Action meaning** | Raw torque values | Offset from standing pose (action=0 → stand still) |
| **Orientation obs** | Raw quaternion (4D, double-cover) | Projected gravity vector (3D) |
| **Velocity obs** | World frame | Body frame (rotation-invariant) |
| **Joint pos obs** | Absolute angles | Offset from default standing pose |
| **Domain randomization** | None | Mass ±15%, friction 0.5–2.0× |
| **Reset settling** | 50 steps | 200 steps |
| **Observation dims** | 54 | 48 |
| **Action dims** | 13 | 12 |

**Action Scale** (maps [-1, 1] to joint angle offset):
- Hip roll: ±0.2 rad
- Hip pitch: ±0.4 rad
- Knee: ±0.4 rad

**Reward Function:**
- Forward velocity tracking (toward target speed)
- Height maintenance bonus (penalizes deviation from 1.0m)
- Upright bonus (reward for gravity vector pointing down)
- Energy penalty (minimize torque × velocity)
- Action smoothness penalty (minimize action changes between steps)
- Lateral velocity penalty
- Angular velocity penalty
- Termination on fall (projected gravity z-component > -0.5)

#### Training Script Updates
- 3-layer policy network: [256, 128, 64] (was [256, 256])
- 4096 steps per update (was 2048)
- 5 optimization epochs (was 10)
- Lower entropy coefficient: 0.005 (was 0.01)
- Conservative initial std: -1.5 (was -1.0)

---

## Current State

- ✅ MuJoCo MJCF model with PD position control and standing keyframe
- ✅ Gymnasium RL environment with modern observation/action design
- ✅ PPO training script with proper hyperparameters
- ✅ Evaluation script with visual and headless modes
- ✅ All changes committed and pushed to GitHub
- ⏳ Training has not been run yet (requires Linux machine with Python 3.10+)
- ⏳ Old v1 checkpoints are incompatible and should be deleted before training

---

## Next Steps

1. **Run initial training** (2M steps, ~1–2 hours on CPU) and verify the cow can stand and walk
2. **Tune if needed** — adjust PD gains, reward weights, action scale, or add curriculum learning (start with target_speed=0)
3. **Add terrain randomization** once flat-ground walking is stable
4. **Reintroduce spine articulation** (13 DOF) for turning
5. **Add cannon bones** with passive coupling for more realistic gait
6. **Attach carriage** as a trailing body in the simulation
7. **Export trained policy** for deployment in Gazebo/ROS2 (real hardware path)
8. **Use force/torque data** to finalize motor selection and bill of materials

---

## Hardware Requirements for Training

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Linux (Ubuntu 22.04+) | Same |
| **Python** | 3.10+ | 3.11 |
| **RAM** | 8 GB | 16 GB |
| **CPU** | 4 cores | 8+ cores (for parallel envs) |
| **GPU** | Not required (CPU training) | NVIDIA GPU speeds up policy inference |
| **Storage** | 2 GB | 5 GB (for checkpoints + TensorBoard logs) |

---

## Key Lessons Learned

1. **Always validate joint ranges against initial poses** — MuJoCo silently snaps joints into range with explosive force
2. **PD position control is essential for RL quadrupeds** — raw torque control makes the learning problem orders of magnitude harder
3. **Standing keyframes matter** — the policy should start from a stable configuration so it can learn incrementally
4. **Projected gravity > quaternions** for RL observations — avoids the double-cover problem and is more intuitive for the network
5. **Action-as-offset-from-standing** is the standard approach — action=0 means "hold your current pose," making the initial policy (random near zero) approximately stable
6. **Hand-coded gaits have a ceiling** — the Gazebo v1 approach showed that position-trajectory control can't adapt to perturbations, terrain, or dynamic loads the way a learned policy can
