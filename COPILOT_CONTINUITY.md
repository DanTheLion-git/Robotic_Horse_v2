# Robotic Highland Cow — Project Continuity Document
## For Copilot: Read this file to resume working on this project

---

## Project Overview

Daniel van Leeuwen is building a **robotic Highland Cow** (150cm withers, ~115kg) that pulls a vis-à-vis carriage for his wedding/events business (The Lions Alliance). The robot uses **MuJoCo + Reinforcement Learning** to learn natural quadruped locomotion. This is the v2 approach — v1 used Gazebo/ROS2 with hand-coded gaits but failed due to instability.

**Repository:** `https://github.com/DanTheLion-git/Robotic_Horse_v2`  
**Local path:** `C:\Users\DanielvanLeeuwen\Documents\Daniel\research\Robotic_Horse_v2\`  
**Training runs on:** Daniel's Ubuntu Linux laptop (MuJoCo not installed on Windows)

---

## Current State (v5, commit `ef4a818`)

### What exists and works:
- **Simplified MJCF model** (`models/highland_cow.xml`) — v5, 13 DOF
  - Split body: rear torso + front body connected by spine_yaw joint
  - Front legs: nearly straight (bovine forelimb, hp=+0.15, kn=-0.18, upper=0.44m, lower=0.42m)
  - Rear legs: Z-shaped backward (bovine hindlimb, hp=-0.25, kn=+0.50, upper=0.48m, lower=0.45m)
  - Simplified body (2 barrel capsules + head capsule), decorative geoms stripped for fast sim
  - The cow **stands stably** ✓

- **Hand-coded gait controller** (`scripts/test_walk.py`)
  - Sinusoidal lateral-sequence walk (FL→RR→FR→RL phase pattern)
  - Supports walk and trot gaits, adjustable speed
  - **Has NOT been tested yet** — Daniel needs to run it on Linux
  - Run with: `python scripts/test_walk.py` (or `--gait trot`, `--stand`, `--speed 0.5`)

- **RL Environment** (`envs/cow_walk_env.py`) — v5, command-conditioned
  - 13 actuators, 53-dim observation (includes [cmd_vx, cmd_yaw])
  - **Reference motion reward** added: sinusoidal gait prior guides RL toward walking pattern
  - Commands randomized per episode during training (vx: -0.5 to 2.0, yaw: ±1.0)
  - 20% episodes are "stand still" to maintain balance ability

- **Training script** (`scripts/train.py`) — PPO, 3M steps, 8 parallel envs, [256,128,64] network
- **Evaluate script** (`scripts/evaluate.py`) — WASD keyboard controls for steering

### What does NOT work yet:
- **The cow doesn't walk under RL control** — it stands but doesn't move when given velocity commands
- The hand-coded test_walk.py hasn't been validated yet (may need tuning)
- No trained model exists for v5 (previous checkpoints incompatible due to obs/action dim changes)

---

## Critical Technical Details

### MuJoCo Joint Convention (this caused MANY bugs in prior iterations)
- Hinge joint around Y axis: `Ry(θ)` transforms child frame
- **Positive hip_pitch** → upper leg swings BACKWARD
- **Negative hip_pitch** → upper leg swings FORWARD
- **Positive knee** → lower leg bends BACKWARD relative to upper
- **Negative knee** → lower leg bends FORWARD relative to upper

### PD Actuators
```
force = kp × (ctrl - qpos) - kd × qvel
gainprm = "kp 0 0", biasprm = "0 -kp -kd"
```
- Front hip_pitch: kp=200, kd=8, ±350 Nm
- Front knee: kp=150, kd=6, ±300 Nm
- Rear hip_pitch: kp=250, kd=10, ±400 Nm
- Rear knee: kp=250, kd=10, ±400 Nm
- Spine yaw: kp=150, kd=8, ±200 Nm

### Joint Order (13 DOF)
```
0: spine_yaw
1-3: FL (hip_roll, hip_pitch, knee)
4-6: FR (hip_roll, hip_pitch, knee)
7-9: RL (hip_roll, hip_pitch, knee)
10-12: RR (hip_roll, hip_pitch, knee)
```

### Default Standing Angles
```python
[0.0,                          # spine_yaw
 0.0,  0.15, -0.18,           # FL
 0.0,  0.15, -0.18,           # FR
 0.0, -0.25,  0.50,           # RL
 0.0, -0.25,  0.50]           # RR
```

### Gait Reference Motion (in env reward)
- Lateral-sequence walk: FL=0.0, FR=0.5, RL=0.75, RR=0.25 phase offsets
- Hip pitch oscillates ±0.15 rad (front) / ±0.18 rad (rear) around default
- Knee lifts +0.25 (front) / +0.20 (rear) during swing (positive = backward bend = shorter leg = foot clearance)
- Frequency scales with |cmd_vx|: 0.8 Hz at slow, up to 2.0 Hz at fast

### Mass Distribution (~114 kg)
- Rear torso: 42 kg
- Front body: 48 kg (includes head/neck mass)
- Front legs: 2 × 5.5 kg = 11 kg
- Rear legs: 2 × 6.5 kg = 13 kg

---

## How to Resume Development

### Step 1: Validate the hand-coded gait
Tell Daniel to run on his Linux laptop:
```bash
cd ~/Robotic_Horse_v2 && git pull
python scripts/test_walk.py --stand      # Does it stay standing?
python scripts/test_walk.py              # Does it walk forward?
python scripts/test_walk.py --gait trot  # Does it trot?
```
If test_walk.py fails (cow falls), the problem is in the MJCF model or gait parameters, NOT RL. Fix those first before training.

### Step 2: Train RL
```bash
rm -rf checkpoints/ runs/
python scripts/train.py --steps 3000000
tensorboard --logdir runs/   # Monitor in another terminal
```

### Step 3: Evaluate
```bash
python scripts/evaluate.py   # WASD to steer
```

---

## Known Issues & Likely Next Problems

1. **If test_walk.py makes the cow fall:** The gait amplitudes or phase offsets need tuning. Try reducing `FRONT_HP_AMP` and `REAR_HP_AMP` in test_walk.py (and matching values in cow_walk_env.py). Start with `--speed 0.3` for tiny steps.

2. **If RL still doesn't learn to walk after 3M steps:** 
   - Check TensorBoard for reward trends — if reward plateaus at standing-only value (~3.0-3.5), the reference motion weight may need increasing (change `r_reference` weight from 1.0 to 2.0 in `_compute_reward`)
   - The velocity tracking reward may have too tight a gaussian — try changing the exponent from `-4.0` to `-2.0` in the vx_error exp
   - Try training with `--no-curriculum` flag
   - Consider reducing ACTION_SCALE so the RL can't make huge jerky motions

3. **If the cow walks but unnaturally:**
   - Increase `p_action_rate` weight (currently 0.008) for smoother motions
   - Add gait symmetry reward: penalize difference between left/right leg phases
   - The reference motion should help here — increase its weight

4. **Spine yaw not being used for turning:**
   - Check that `cmd_yaw` reward actually triggers spine movement
   - May need a separate reward term specifically for spine_yaw tracking

---

## Remaining TODOs (from prior sessions)

| ID | Task | Notes |
|----|-------|-------|
| passive-springs | Add fetlock spring compliance | For real robot: elastic energy storage |
| force-analysis | Run force analysis simulation | Record torques from trained walking to size motors |
| bill-of-materials | Generate bill of materials | Map torques → real QDD motors → shopping list |
| detailed-3d-model | Create detailed 3D model | After RL works: add cow shell visuals back, export for 3D printing |
| sim-force-validation | Run sim force validation | Was for Gazebo v1 — replace with MuJoCo force analysis |

---

## Project History (chronological)

1. **v1 (Gazebo/ROS2)** — Hand-coded position gaits. Worked somewhat but unstable: buckling knees, forward tipping, couldn't turn carriage. Repo: `DanTheLion-git/Robotic_Horse`
2. **v2 initial** — Converted URDF to MuJoCo, set up RL pipeline with PPO
3. **v3** — Rebuilt with Go1-pattern symmetric Z-shaped legs. Cow stands stably. Commit `329da5b`
4. **v4** — Anatomically correct bovine legs (different front/rear), split body with spine yaw, 13 DOF. Commit `df2dc33`
5. **v5 (current)** — Simplified body, hand-coded gait controller for testing, reference motion reward in RL. Commit `ef4a818`

---

## Environment & Credentials
- **GitHub:** DanTheLion-git
- **Pi SSH:** superwortel@lionsalliance-pi
- **Dev machine:** Ubuntu laptop (for MuJoCo/RL), Windows PC (for file management/coding)
- **Other projects:** `theLionsAlliance.com/` (business website), `lionsalliance-pi/` (Docker apps on Pi)

---

## File Map

```
Robotic_Horse_v2/
├── models/
│   └── highland_cow.xml          # MuJoCo MJCF (v5, simplified bovine)
├── envs/
│   ├── __init__.py
│   └── cow_walk_env.py           # Gymnasium RL env (v5, 13 DOF, ref motion)
├── scripts/
│   ├── train.py                  # PPO training (SB3, curriculum)
│   ├── evaluate.py               # Visual eval with WASD steering
│   └── test_walk.py              # Hand-coded sinusoidal gait (no RL)
├── checkpoints/                  # Trained model saves (empty until training)
├── runs/                         # TensorBoard logs
├── reference/                    # (empty)
├── Progress_Summary_1.md         # Initial MuJoCo setup documentation
├── Progress_Summary_2.md         # Bovine anatomy rebuild documentation
├── README.md
├── requirements.txt              # mujoco, gymnasium, stable-baselines3, torch
└── .gitignore
```
