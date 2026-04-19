#!/usr/bin/env python3
"""
test_walk.py — Hand-coded sinusoidal gait controller for Highland Cow

Applies a lateral-sequence walk pattern directly to the MuJoCo model
WITHOUT any RL. This verifies the model CAN physically walk and provides
a visual reference for what the gait should look like.

Usage:
    python scripts/test_walk.py                  # Walk at default speed
    python scripts/test_walk.py --speed 0.5      # Slower walk
    python scripts/test_walk.py --gait trot      # Trot gait pattern
    python scripts/test_walk.py --stand          # Just stand (test balance)

Controls:
    Q / ESC — quit

Gait patterns:
  walk:  lateral sequence (FL→RR→FR→RL), ~1.2 Hz
  trot:  diagonal pairs (FL+RR, FR+RL), ~1.8 Hz
"""

import argparse
import math
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models", "highland_cow.xml")

# Joint order in the actuator array (must match MJCF actuator order)
# 0: spine_yaw
# 1-3: FL (roll, pitch, knee)
# 4-6: FR (roll, pitch, knee)
# 7-9: RL (roll, pitch, knee)
# 10-12: RR (roll, pitch, knee)

# Default standing angles
DEFAULT = np.array([
    0.0,                          # spine_yaw
    0.0,  0.15, -0.18,           # FL
    0.0,  0.15, -0.18,           # FR
    0.0, -0.25,  0.50,           # RL
    0.0, -0.25,  0.50,           # RR
])

# Gait phase offsets (fraction of cycle)
GAITS = {
    "walk": {
        "phases": {"fl": 0.00, "fr": 0.50, "rl": 0.75, "rr": 0.25},
        "freq": 1.2,
    },
    "trot": {
        "phases": {"fl": 0.00, "fr": 0.50, "rl": 0.50, "rr": 0.00},
        "freq": 1.8,
    },
}

# Leg config: (ctrl_offset, is_front)
LEGS = {
    "fl": (1, True),
    "fr": (4, True),
    "rl": (7, False),
    "rr": (10, False),
}

# Gait amplitudes
FRONT_HP_AMP = 0.15    # hip pitch swing amplitude (rad)
FRONT_KN_LIFT = 0.25   # knee lift during swing (rad)
REAR_HP_AMP = 0.18
REAR_KN_LIFT = 0.20
HIP_ROLL_AMP = 0.03    # lateral weight shift
SPINE_AMP = 0.05       # body undulation


def compute_gait(t, gait_name="walk", speed_scale=1.0):
    """Compute target joint angles for all 13 actuators at time t.

    Args:
        t: simulation time in seconds
        gait_name: "walk" or "trot"
        speed_scale: multiplier for step frequency and amplitude

    Returns:
        np.array of 13 target joint angles
    """
    gait = GAITS[gait_name]
    freq = gait["freq"] * speed_scale
    phases = gait["phases"]

    ctrl = DEFAULT.copy()

    for leg_name, (offset, is_front) in LEGS.items():
        phase = phases[leg_name]
        phi = 2.0 * math.pi * (freq * t + phase)

        hp_amp = FRONT_HP_AMP if is_front else REAR_HP_AMP
        kn_lift = FRONT_KN_LIFT if is_front else REAR_KN_LIFT

        # Hip roll: lateral weight shift (cosine for smooth oscillation)
        ctrl[offset + 0] = HIP_ROLL_AMP * math.cos(phi)

        # Hip pitch: swing forward/backward around default
        # Negative sin = swing forward (foot moves ahead)
        # Positive sin = push backward (stance phase)
        ctrl[offset + 1] = DEFAULT[offset + 1] - hp_amp * math.sin(phi) * speed_scale

        # Knee: lift foot during swing phase (first half of cycle)
        # max(0, sin) creates a pulse only during the swing portion
        swing = max(0.0, math.sin(phi))
        ctrl[offset + 2] = DEFAULT[offset + 2] + kn_lift * swing * speed_scale

    # Spine yaw: slight body undulation synchronized with diagonal legs
    spine_phase = 2.0 * math.pi * freq * t
    ctrl[0] = SPINE_AMP * math.sin(spine_phase) * speed_scale

    return ctrl


def main():
    parser = argparse.ArgumentParser(description="Test hand-coded walking gait")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speed multiplier (0.5=slow, 1.0=normal, 1.5=fast)")
    parser.add_argument("--gait", type=str, default="walk",
                        choices=list(GAITS.keys()),
                        help="Gait pattern")
    parser.add_argument("--stand", action="store_true",
                        help="Just stand still (test balance)")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Simulation duration in seconds")
    args = parser.parse_args()

    # Load model
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)

    # Reset to standing keyframe
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "standing")
    mujoco.mj_resetDataKeyframe(m, d, key_id)

    # Let the model settle for 1 second
    print("Settling model...")
    for _ in range(500):
        mujoco.mj_step(m, d)

    torso_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    settled_height = d.xpos[torso_id][2]
    print(f"  Settled height: {settled_height:.3f}m")

    if args.stand:
        print("\nStanding test — cow should remain balanced")
        print("Press ESC in viewer to quit")
    else:
        print(f"\nGait: {args.gait} @ speed={args.speed:.1f}x")
        print(f"  Frequency: {GAITS[args.gait]['freq'] * args.speed:.1f} Hz")
        print("Press ESC in viewer to quit")

    # Launch viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        sim_time = 0.0
        step = 0
        dt = m.opt.timestep
        ctrl_dt = 0.02
        ctrl_steps = int(ctrl_dt / dt)

        while viewer.is_running() and sim_time < args.duration:
            # Update control targets every ctrl_dt
            if step % ctrl_steps == 0:
                if args.stand:
                    d.ctrl[:13] = DEFAULT
                else:
                    d.ctrl[:13] = compute_gait(sim_time, args.gait, args.speed)

            mujoco.mj_step(m, d)
            sim_time += dt
            step += 1

            # Sync viewer at ~50 Hz
            if step % ctrl_steps == 0:
                viewer.sync()

                # Print status every 2 seconds
                if step % (ctrl_steps * 100) == 0:
                    pos = d.xpos[torso_id]
                    # Compute forward velocity from body
                    cvel = d.cvel[torso_id]
                    fwd_vel = cvel[3]  # linear x velocity
                    print(f"  t={sim_time:.1f}s: "
                          f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                          f"height={pos[2]:.3f}m, fwd_vel={fwd_vel:.2f}m/s")

                # Real-time pacing
                time.sleep(ctrl_dt)

    print("\nDone!")


if __name__ == "__main__":
    main()
