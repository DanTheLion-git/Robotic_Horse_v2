# Robotic Horse v2 — Progress Summary #2

**Date:** July 2025  
**Project:** Highland Cow Quadruped Robot — MuJoCo RL Pipeline  
**Repository:** [DanTheLion-git/Robotic_Horse_v2](https://github.com/DanTheLion-git/Robotic_Horse_v2)

---

## What Changed Since Summary #1

### Problem
The v3 model (from Summary #1) used identical Z-shaped legs on all four limbs, based on the Unitree Go1/A1 pattern. While this stood stably and provided a solid RL training foundation, the cow looked and moved like a generic quadruped robot rather than a bovine. The user requested an anatomically correct rebuild.

### Key Decisions

1. **Split body architecture (13 DOF)** — The torso is now split into:
   - **Rear torso** (root body with freejoint): houses abdomen, rump, rear legs
   - **Front body** (child via `spine_yaw` hinge joint): houses chest, shoulders, neck, head, front legs
   - The spine yaw joint (±0.35 rad / ±20°) enables turning by differential body articulation

2. **Different front/rear leg geometry** — Real cows have fundamentally different limb configurations:
   - **Front legs**: Nearly straight. Upper leg goes slightly backward (hp=+0.15 rad), lower leg comes slightly forward (kn=-0.18 rad). The forearm hangs nearly vertical (1.7° from plumb). This mimics the bovine scapula→humerus→radius chain.
   - **Rear legs**: Classic backward-Z. Upper leg (femur) goes forward (hp=-0.25 rad), lower leg (tibia/metatarsus) bends backward (kn=+0.50 rad). The hock angle creates 14.3° from vertical. Feet land directly under the hip.

3. **Proper bovine proportions** — Based on real Highland Cow anatomy scaled to 150cm withers:
   - Front upper leg: 0.44m, lower: 0.42m (lighter: 3.0kg + 2.0kg)
   - Rear upper leg: 0.48m, lower: 0.45m (heavier: 3.5kg + 2.5kg)
   - Rear legs slightly longer than front (bovine rump is higher than shoulders)
   - Deep chest, withers hump, barrel shape, neck curve, horns — all capsules/ellipsoids

4. **Command-conditioned RL policy** — The robot receives velocity commands [cmd_vx, cmd_yaw] as part of its observation. During training, commands are randomized per episode (-0.5 to 2.0 m/s forward, ±1.0 rad/s yaw). 20% of episodes are "stand still" commands. This single policy handles walk, trot, reverse, and turning.

---

## Files Modified

| File | Changes |
|------|---------|
| `models/highland_cow.xml` | Complete rewrite: split body, 13 DOF, bovine anatomy, new PD gains |
| `envs/cow_walk_env.py` | v5: 13 actuators, 53-dim observation, spine_yaw in joint list |
| `scripts/evaluate.py` | Added WASD keyboard controls for steering during evaluation |
| `scripts/train.py` | Updated banner to v5 |

---

## Technical Details

### Joint Configuration (Standing Pose)

| Joint | Default Angle | Range | kp | kd | Force Limit |
|-------|:---:|:---:|:---:|:---:|:---:|
| spine_yaw | 0.0 | ±0.35 | 150 | 8 | ±200 Nm |
| front hip_roll | 0.0 | ±0.4 | 100 | 5 | ±150 |
| front hip_pitch | +0.15 | -0.5 to 0.8 | 200 | 8 | ±350 |
| front knee | -0.18 | -0.8 to 0.5 | 150 | 6 | ±300 |
| rear hip_roll | 0.0 | ±0.4 | 100 | 5 | ±150 |
| rear hip_pitch | -0.25 | -0.8 to 0.4 | 250 | 10 | ±400 |
| rear knee | +0.50 | -0.3 to 1.5 | 250 | 10 | ±400 |

### Mass Distribution (~114 kg total)
- Rear torso: 42 kg
- Front body (chest + neck + head): 48 kg
- Front legs: 2 × 5.5 kg = 11 kg
- Rear legs: 2 × 6.5 kg = 13 kg

### Standing Height Verification
- Body center: z = 1.05m
- Front hips: z = 0.87m → feet at z ≈ 0.065m (settles to ground)
- Rear hips: z = 0.95m → feet at z ≈ 0.099m (settles to ground)

---

## How to Train (on Linux laptop)

```bash
cd ~/Robotic_Horse_v2
git pull origin main

# Clean old training data
rm -rf checkpoints/ runs/

# Install/update deps
pip install -r requirements.txt

# Train (3M steps, ~2-4 hours on CPU, faster with GPU)
python scripts/train.py --steps 3000000

# Monitor
tensorboard --logdir runs/

# Evaluate with WASD steering
python scripts/evaluate.py
```

### Keyboard Controls (evaluate.py)
| Key | Action |
|-----|--------|
| W | Increase forward speed (+0.2 m/s) |
| S | Decrease forward speed (-0.2 m/s) |
| A | Turn left (+0.2 rad/s) |
| D | Turn right (-0.2 rad/s) |
| SPACE | Stop (zero all commands) |
| Q | Quit |

---

## What's Next

1. **Train the v4 model** — The new anatomy needs fresh training from scratch. Previous checkpoints won't work (different obs/action dims).

2. **Tune rewards if needed** — The reward function may need adjustment for the asymmetric leg geometry. Watch for:
   - Does the cow stand stably? (height reward should keep it up)
   - Does it learn to walk forward? (velocity tracking reward)
   - Does spine yaw enable turning? (yaw rate tracking reward)

3. **Add terrain randomization** — Once flat-ground walking works, add slopes, bumps, and varying friction to build robust policies.

4. **Carriage attachment** — Add a carriage model connected via shaft to test pulling behavior.

5. **Motor sizing** — Record joint torques during trained walking to size real QDD motors.

---

## Git History
- `329da5b` — v3: Z-shaped legs, capsule body, curriculum (Summary #1)
- `df2dc33` — v4: Anatomically correct bovine with spine yaw (this summary)
