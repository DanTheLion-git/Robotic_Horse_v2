"""
Highland Cow Walking Environment v5 — Bovine Anatomy + Command-Conditioned
===========================================================================
13 DOF anatomically correct bovine model with split body (spine yaw joint).

The robot receives a velocity command [cmd_vx, cmd_yaw] as part of its
observation. During training, commands are randomly sampled each episode.
During evaluation, commands come from keyboard input (WASD).

Observation (53 dims):
  - velocity command [vx, yaw]        (2)
  - projected gravity in body frame   (3)
  - body angular velocity             (3)
  - body linear velocity (x, y)       (2)
  - joint positions (offset from default) (13)
  - joint velocities                  (13)
  - previous action                   (13)
  - foot contact flags                (4)

Action (13 dims):
  - Position offsets from standing pose, normalized to [-1, 1]
  - Order: spine_yaw, FL(roll,pitch,knee), FR(...), RL(...), RR(...)
"""

import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models", "highland_cow.xml")

N_ACTUATORS = 13

JOINT_NAMES = [
    "spine_yaw",
    "fl_hip_roll", "fl_hip_pitch", "fl_knee",
    "fr_hip_roll", "fr_hip_pitch", "fr_knee",
    "rl_hip_roll", "rl_hip_pitch", "rl_knee",
    "rr_hip_roll", "rr_hip_pitch", "rr_knee",
]

FOOT_GEOM_NAMES = [
    "fl_foot_geom", "fr_foot_geom", "rl_foot_geom", "rr_foot_geom",
]

# Default standing pose — bovine anatomy (must match keyframe in MJCF)
# Front legs: nearly straight (hp=0.15, kn=-0.18)
# Rear legs: Z-shaped backward (hp=-0.25, kn=0.50)
DEFAULT_ANGLES = np.array([
    0.0,                          # spine_yaw
    0.0,  0.15, -0.18,           # FL
    0.0,  0.15, -0.18,           # FR
    0.0, -0.25,  0.50,           # RL
    0.0, -0.25,  0.50,           # RR
], dtype=np.float32)

ACTION_SCALE = np.array([
    0.30,                         # spine_yaw
    0.20, 0.40, 0.35,            # FL
    0.20, 0.40, 0.35,            # FR
    0.20, 0.40, 0.50,            # RL
    0.20, 0.40, 0.50,            # RR
], dtype=np.float32)

# Episode parameters
MAX_EPISODE_STEPS = 2000   # 2000 × 0.02s = 40 seconds
N_SUBSTEPS = 10            # physics steps per control step
CONTROL_DT = 0.02          # 50 Hz control

TARGET_HEIGHT = 1.00

# ── Reference gait parameters (sinusoidal walking prior) ──
# These define what a "good walk" looks like, giving RL a strong initial signal.
# Phase offsets for lateral-sequence walk
GAIT_PHASES = {
    "fl": 0.00, "fr": 0.50, "rl": 0.75, "rr": 0.25,
}
# Leg config: (ctrl_offset, is_front)
GAIT_LEGS = [
    ("fl", 1, True), ("fr", 4, True),
    ("rl", 7, False), ("rr", 10, False),
]
FRONT_HP_AMP = 0.15    # front hip pitch swing
FRONT_KN_LIFT = 0.25   # front knee lift during swing
REAR_HP_AMP = 0.18     # rear hip pitch swing
REAR_KN_LIFT = 0.20    # rear knee lift during swing
HIP_ROLL_AMP = 0.03    # lateral sway
SPINE_GAIT_AMP = 0.05  # body undulation

# Command ranges for training randomization
CMD_VX_RANGE = (-0.5, 2.0)    # m/s: reverse to trot
CMD_YAW_RANGE = (-1.0, 1.0)   # rad/s: turn left/right


class HighlandCowWalkEnv(gym.Env):
    """Command-conditioned MuJoCo environment for Highland Cow locomotion."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, target_speed=1.0, randomize=True,
                 cmd_vx=None, cmd_yaw=None):
        super().__init__()

        self.render_mode = render_mode
        self.randomize = randomize

        # Command state — if None, will be randomized each episode
        self._fixed_cmd_vx = cmd_vx
        self._fixed_cmd_yaw = cmd_yaw
        self.cmd_vx = cmd_vx if cmd_vx is not None else 0.0
        self.cmd_yaw = cmd_yaw if cmd_yaw is not None else 0.0

        # Legacy compat
        self.target_speed = target_speed

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        # Cache IDs
        self._torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._joint_ids = [mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        self._joint_qpos_idxs = [self.model.jnt_qposadr[jid]
                                 for jid in self._joint_ids]
        self._joint_qvel_idxs = [self.model.jnt_dofadr[jid]
                                 for jid in self._joint_ids]
        self._foot_geom_ids = [mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOM_NAMES]
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Joint limits for clamping targets
        self._joint_lower = np.array([
            self.model.jnt_range[jid, 0] for jid in self._joint_ids],
            dtype=np.float32)
        self._joint_upper = np.array([
            self.model.jnt_range[jid, 1] for jid in self._joint_ids],
            dtype=np.float32)

        # Store originals for domain randomization
        self._default_body_mass = self.model.body_mass.copy()
        self._default_geom_friction = self.model.geom_friction.copy()

        # Action space: 13 joint offsets
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_ACTUATORS,), dtype=np.float32)

        # Obs: command(2) + gravity(3) + angvel(3) + linvel(2) + joints(13+13)
        #    + prev_action(13) + contacts(4) = 53
        obs_dim = 2 + 3 + 3 + 2 + N_ACTUATORS * 3 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._prev_action = np.zeros(N_ACTUATORS, dtype=np.float32)
        self._step_count = 0

    def set_command(self, vx, yaw):
        """Set velocity command (used by keyboard control in evaluate)."""
        self.cmd_vx = float(vx)
        self.cmd_yaw = float(yaw)

    def set_target_speed(self, speed):
        """Legacy: set forward speed command."""
        self.cmd_vx = float(speed)

    def _get_projected_gravity(self):
        rot_mat = self.data.xmat[self._torso_id].reshape(3, 3)
        gravity_world = np.array([0.0, 0.0, -1.0])
        return (rot_mat.T @ gravity_world).astype(np.float32)

    def _get_body_velocity(self):
        cvel = self.data.cvel[self._torso_id]
        angvel_world = cvel[0:3]
        linvel_world = cvel[3:6]
        rot_mat = self.data.xmat[self._torso_id].reshape(3, 3)
        angvel_body = (rot_mat.T @ angvel_world).astype(np.float32)
        linvel_body = (rot_mat.T @ linvel_world).astype(np.float32)
        return linvel_body, angvel_body

    def _get_joint_state(self):
        qpos = np.array([self.data.qpos[idx] for idx in self._joint_qpos_idxs],
                        dtype=np.float32)
        qvel = np.array([self.data.qvel[idx] for idx in self._joint_qvel_idxs],
                        dtype=np.float32)
        pos_offset = qpos - DEFAULT_ANGLES
        qvel = np.clip(qvel, -15.0, 15.0)
        return pos_offset, qvel

    def _get_foot_contacts(self):
        contacts = np.zeros(4, dtype=np.float32)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for j, fid in enumerate(self._foot_geom_ids):
                if ((c.geom1 == fid and c.geom2 == self._floor_geom_id) or
                    (c.geom2 == fid and c.geom1 == self._floor_geom_id)):
                    contacts[j] = 1.0
        return contacts

    def _get_reference_joints(self):
        """Compute reference joint angles from sinusoidal gait prior.

        Gives the RL agent a strong hint about what walking looks like.
        The reference frequency scales with the commanded speed.
        When cmd_vx ≈ 0, reference is just the standing pose.
        """
        speed = abs(self.cmd_vx)
        if speed < 0.1:
            return DEFAULT_ANGLES.copy()

        # Scale frequency with speed: slow walk 0.8 Hz, fast trot 2.0 Hz
        freq = 0.8 + 0.6 * speed
        speed_scale = min(speed / 1.0, 1.5)  # clamp amplitude scaling
        direction = 1.0 if self.cmd_vx > 0 else -1.0

        t = self._step_count * CONTROL_DT
        ref = DEFAULT_ANGLES.copy()

        # Spine follows yaw command slightly
        spine_phase = 2.0 * math.pi * freq * t
        ref[0] = SPINE_GAIT_AMP * math.sin(spine_phase) * speed_scale
        if abs(self.cmd_yaw) > 0.1:
            ref[0] += self.cmd_yaw * 0.10

        for leg_name, offset, is_front in GAIT_LEGS:
            phase = GAIT_PHASES[leg_name]
            phi = 2.0 * math.pi * (freq * t + phase)

            hp_amp = FRONT_HP_AMP if is_front else REAR_HP_AMP
            kn_lift = FRONT_KN_LIFT if is_front else REAR_KN_LIFT

            ref[offset + 0] = HIP_ROLL_AMP * math.cos(phi)
            ref[offset + 1] = DEFAULT_ANGLES[offset + 1] - direction * hp_amp * math.sin(phi) * speed_scale
            ref[offset + 2] = DEFAULT_ANGLES[offset + 2] + kn_lift * max(0.0, math.sin(phi)) * speed_scale

        return ref

    def _get_obs(self):
        proj_gravity = self._get_projected_gravity()
        linvel_body, angvel_body = self._get_body_velocity()
        joint_pos_offset, joint_vel = self._get_joint_state()
        contacts = self._get_foot_contacts()

        # Normalize commands for the observation
        cmd_obs = np.array([
            self.cmd_vx / 2.0,     # roughly [-0.25, 1.0]
            self.cmd_yaw / 1.0,    # [-1.0, 1.0]
        ], dtype=np.float32)

        obs = np.concatenate([
            cmd_obs,                        # 2 - what should I do?
            proj_gravity,                   # 3 - which way is up?
            angvel_body,                    # 3 - am I spinning?
            linvel_body[:2],                # 2 - forward + lateral speed
            joint_pos_offset,               # 13 - how far from standing?
            joint_vel,                      # 13 - how fast are joints moving?
            self._prev_action,              # 13 - what did I do last step?
            contacts,                       # 4 - which feet are on ground?
        ])
        return obs

    def _compute_reward(self, action):
        """Command-tracking + reference motion reward function.

        Rewards:
          - Track cmd_vx with forward velocity
          - Track cmd_yaw with yaw rate
          - Match reference gait pattern (motion prior)
          - Stay upright and at correct height
          - Keep feet on ground
        Penalties:
          - Energy, action smoothness, lateral drift
        """
        linvel_body, angvel_body = self._get_body_velocity()
        forward_vel = linvel_body[0]
        lateral_vel = linvel_body[1]
        yaw_rate = angvel_body[2]

        body_height = self.data.xpos[self._torso_id][2]
        proj_grav = self._get_projected_gravity()
        joint_pos_offset, joint_vel = self._get_joint_state()

        # ── REWARDS ──

        # 1. Forward velocity tracking (main task objective)
        vx_error = forward_vel - self.cmd_vx
        r_velocity = 2.0 * math.exp(-4.0 * vx_error**2)

        # 2. Yaw rate tracking
        yaw_error = yaw_rate - self.cmd_yaw
        r_yaw = 0.8 * math.exp(-4.0 * yaw_error**2)

        # 3. Reference motion tracking (gait prior — guides early learning)
        ref_joints = self._get_reference_joints()
        actual_joints = DEFAULT_ANGLES + joint_pos_offset
        ref_error = np.sum((actual_joints - ref_joints)**2)
        r_reference = 1.0 * math.exp(-2.0 * ref_error)

        # 4. Height maintenance
        height_error = body_height - TARGET_HEIGHT
        r_alive = 1.0 * math.exp(-8.0 * height_error**2)

        # 5. Upright bonus
        r_upright = 0.5 * max(0.0, -proj_grav[2])

        # 6. Foot contact (at least 2 feet should be on ground)
        contacts = self._get_foot_contacts()
        r_contact = 0.1 * np.sum(contacts)

        # ── PENALTIES ──

        # 7. Energy
        torques = self.data.actuator_force[:N_ACTUATORS]
        energy = np.sum(np.abs(torques * joint_vel))
        p_energy = 0.0002 * energy

        # 8. Action rate (smoothness)
        action_diff = action - self._prev_action
        p_action_rate = 0.008 * np.sum(action_diff**2)

        # 9. Action magnitude
        p_action_mag = 0.002 * np.sum(action**2)

        # 10. Lateral velocity
        lat_threshold = 0.1 * abs(self.cmd_yaw)
        lat_excess = max(0.0, abs(lateral_vel) - lat_threshold)
        p_lateral = 0.2 * lat_excess**2

        # 11. Roll/pitch angular velocity
        p_angvel_rp = 0.03 * (angvel_body[0]**2 + angvel_body[1]**2)

        # 12. Joint velocity smoothness
        p_joint_vel = 0.0003 * np.sum(joint_vel**2)

        reward = (r_velocity + r_yaw + r_reference + r_alive + r_upright + r_contact
                  - p_energy - p_action_rate - p_action_mag
                  - p_lateral - p_angvel_rp - p_joint_vel)

        return float(reward)

    def _is_terminated(self):
        body_pos = self.data.xpos[self._torso_id]
        proj_grav = self._get_projected_gravity()
        if body_pos[2] < 0.50:
            return True
        if proj_grav[2] > -0.3:
            return True
        return False

    def _is_truncated(self):
        return self._step_count >= MAX_EPISODE_STEPS

    def _apply_domain_randomization(self):
        if not self.randomize or self.np_random is None:
            return

        mass_scale = self.np_random.uniform(0.85, 1.15)
        self.model.body_mass[:] = self._default_body_mass * mass_scale

        friction_scale = self.np_random.uniform(0.5, 2.0)
        for fid in self._foot_geom_ids:
            self.model.geom_friction[fid] = (
                self._default_geom_friction[fid] * friction_scale)

    def _sample_command(self):
        """Sample a random velocity command for this episode."""
        if self._fixed_cmd_vx is not None:
            self.cmd_vx = self._fixed_cmd_vx
        elif self.np_random is not None:
            self.cmd_vx = float(self.np_random.uniform(*CMD_VX_RANGE))
        else:
            self.cmd_vx = 0.0

        if self._fixed_cmd_yaw is not None:
            self.cmd_yaw = self._fixed_cmd_yaw
        elif self.np_random is not None:
            self.cmd_yaw = float(self.np_random.uniform(*CMD_YAW_RANGE))
        else:
            self.cmd_yaw = 0.0

        # 20% chance of "stand still" command (helps learn standing)
        if self.np_random is not None and self.np_random.random() < 0.20:
            self.cmd_vx = 0.0
            self.cmd_yaw = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "standing")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        self._apply_domain_randomization()
        self._sample_command()

        if self.np_random is not None:
            for i, idx in enumerate(self._joint_qpos_idxs):
                noise = self.np_random.uniform(-0.03, 0.03)
                self.data.qpos[idx] += noise
                self.data.qpos[idx] = np.clip(
                    self.data.qpos[idx],
                    self._joint_lower[i], self._joint_upper[i])
            for idx in self._joint_qvel_idxs:
                self.data.qvel[idx] = self.np_random.uniform(-0.05, 0.05)

        self.data.ctrl[:N_ACTUATORS] = DEFAULT_ANGLES

        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

        self._prev_action = np.zeros(N_ACTUATORS, dtype=np.float32)
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        target = DEFAULT_ANGLES + action * ACTION_SCALE
        target = np.clip(target, self._joint_lower, self._joint_upper)
        self.data.ctrl[:N_ACTUATORS] = target

        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        self._prev_action = action.copy()

        linvel_body, angvel_body = self._get_body_velocity()
        info = {
            "forward_vel": float(linvel_body[0]),
            "yaw_rate": float(angvel_body[2]),
            "height": float(self.data.xpos[self._torso_id][2]),
            "cmd_vx": self.cmd_vx,
            "cmd_yaw": self.cmd_yaw,
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            img = renderer.render()
            renderer.close()
            return img

    def close(self):
        pass


gym.register(
    id="HighlandCowWalk-v0",
    entry_point="envs.cow_walk_env:HighlandCowWalkEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)
