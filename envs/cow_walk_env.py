"""
Highland Cow Walking Environment v3 — Gymnasium + MuJoCo
========================================================
Based on proven Unitree Go1/A1 Z-shaped leg topology.

Key design (based on MuJoCo Menagerie quadruped models):
  1. Z-shaped legs: hip_pitch forward, knee bends back → feet under hips
  2. ALL 4 legs identical configuration (proven RL pattern)
  3. PD position actuators with action = offset from standing pose
  4. Projected gravity observation (no quaternion double-cover)
  5. Curriculum learning support: target_speed ramps from 0 (standing) to walking

Observation (48 dims):
  - projected gravity in body frame  (3)
  - body angular velocity            (3)
  - body linear velocity (x, y)      (2)
  - joint positions (offset from default) (12)
  - joint velocities                  (12)
  - previous action                   (12)
  - foot contact flags                (4)

Action (12 dims):
  - Position offsets from standing pose, normalized to [-1, 1]
  - Mapped to: target = default_angles + action * action_scale
  - PD controller in MJCF tracks the target position
"""

import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models", "highland_cow.xml")

N_ACTUATORS = 12

# Joint names in MJCF order (must match actuator order)
JOINT_NAMES = [
    "fl_hip_roll", "fl_hip_pitch", "fl_knee",
    "fr_hip_roll", "fr_hip_pitch", "fr_knee",
    "rl_hip_roll", "rl_hip_pitch", "rl_knee",
    "rr_hip_roll", "rr_hip_pitch", "rr_knee",
]

FOOT_GEOM_NAMES = [
    "fl_foot_geom", "fr_foot_geom", "rl_foot_geom", "rr_foot_geom",
]

# Default standing pose — Z-shaped legs (must match keyframe in MJCF)
# All 4 legs identical: hip_roll=0, hip_pitch=0.40, knee=-0.80
DEFAULT_ANGLES = np.array([
    0.0,  0.40,  -0.80,   # FL: roll, pitch, knee
    0.0,  0.40,  -0.80,   # FR
    0.0,  0.40,  -0.80,   # RL
    0.0,  0.40,  -0.80,   # RR
], dtype=np.float32)

# How far from default each joint can deviate (per action unit)
ACTION_SCALE = np.array([
    0.2, 0.5, 0.5,   # FL
    0.2, 0.5, 0.5,   # FR
    0.2, 0.5, 0.5,   # RL
    0.2, 0.5, 0.5,   # RR
], dtype=np.float32)

# Episode parameters
MAX_EPISODE_STEPS = 2000   # 2000 × 0.02s = 40 seconds
N_SUBSTEPS = 10            # physics steps per control step (0.002s × 10 = 0.02s)
CONTROL_DT = 0.02          # 50 Hz control

# Target body height (center of torso when standing, after settling)
TARGET_HEIGHT = 0.98


class HighlandCowWalkEnv(gym.Env):
    """MuJoCo environment for Highland Cow RL locomotion training."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, target_speed=1.0, randomize=True):
        super().__init__()

        self.target_speed = target_speed
        self.render_mode = render_mode
        self.randomize = randomize

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

        # Store original model values for domain randomization
        self._default_body_mass = self.model.body_mass.copy()
        self._default_geom_friction = self.model.geom_friction.copy()

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_ACTUATORS,), dtype=np.float32)

        # Obs: gravity(3) + angvel(3) + linvel(2) + joint_pos(12)
        #    + joint_vel(12) + prev_action(12) + contacts(4) = 48
        obs_dim = 3 + 3 + 2 + 12 + 12 + 12 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._prev_action = np.zeros(N_ACTUATORS, dtype=np.float32)
        self._step_count = 0

    def _get_projected_gravity(self):
        """Get gravity vector projected into body frame (3D)."""
        rot_mat = self.data.xmat[self._torso_id].reshape(3, 3)
        gravity_world = np.array([0.0, 0.0, -1.0])
        return (rot_mat.T @ gravity_world).astype(np.float32)

    def _get_body_velocity(self):
        """Get body linear and angular velocity in body frame."""
        # cvel is [angular(3), linear(3)] in world frame
        cvel = self.data.cvel[self._torso_id]
        angvel_world = cvel[0:3]
        linvel_world = cvel[3:6]

        # Rotate to body frame
        rot_mat = self.data.xmat[self._torso_id].reshape(3, 3)
        angvel_body = (rot_mat.T @ angvel_world).astype(np.float32)
        linvel_body = (rot_mat.T @ linvel_world).astype(np.float32)

        return linvel_body, angvel_body

    def _get_joint_state(self):
        """Get joint positions (offset from default) and velocities."""
        qpos = np.array([self.data.qpos[idx] for idx in self._joint_qpos_idxs],
                        dtype=np.float32)
        qvel = np.array([self.data.qvel[idx] for idx in self._joint_qvel_idxs],
                        dtype=np.float32)

        # Position as offset from default standing pose
        pos_offset = qpos - DEFAULT_ANGLES

        # Clip velocities for numerical stability
        qvel = np.clip(qvel, -15.0, 15.0)

        return pos_offset, qvel

    def _get_foot_contacts(self):
        """Get binary foot contact flags."""
        contacts = np.zeros(4, dtype=np.float32)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for j, fid in enumerate(self._foot_geom_ids):
                if ((c.geom1 == fid and c.geom2 == self._floor_geom_id) or
                    (c.geom2 == fid and c.geom1 == self._floor_geom_id)):
                    contacts[j] = 1.0
        return contacts

    def _get_obs(self):
        """Build observation vector."""
        proj_gravity = self._get_projected_gravity()            # 3
        linvel_body, angvel_body = self._get_body_velocity()    # 3+3
        joint_pos_offset, joint_vel = self._get_joint_state()   # 12+12
        contacts = self._get_foot_contacts()                    # 4

        obs = np.concatenate([
            proj_gravity,                   # 3 - which way is up?
            angvel_body,                    # 3 - am I spinning?
            linvel_body[:2],                # 2 - forward + lateral speed
            joint_pos_offset,               # 12 - how far from standing?
            joint_vel,                      # 12 - how fast are joints moving?
            self._prev_action,              # 12 - what did I do last step?
            contacts,                       # 4 - which feet are on ground?
        ])
        return obs

    def _compute_reward(self, action):
        """Reward function with curriculum support.

        When target_speed=0: maximizes standing stability reward.
        When target_speed>0: adds forward velocity tracking.
        """
        linvel_body, angvel_body = self._get_body_velocity()
        forward_vel = linvel_body[0]
        lateral_vel = linvel_body[1]

        body_height = self.data.xpos[self._torso_id][2]
        proj_grav = self._get_projected_gravity()
        joint_pos_offset, joint_vel = self._get_joint_state()

        # ── REWARDS ──

        # 1. Forward velocity tracking (gaussian around target)
        vel_error = forward_vel - self.target_speed
        r_velocity = 1.5 * math.exp(-4.0 * vel_error**2)

        # 2. Alive / height bonus (strongest reward — stay upright!)
        height_error = body_height - TARGET_HEIGHT
        r_alive = 1.0 * math.exp(-8.0 * height_error**2)

        # 3. Upright bonus (projected gravity z should be -1 when level)
        r_upright = 0.5 * max(0.0, -proj_grav[2])  # 0.5 when level, 0 when tilted

        # 4. Foot contact regularity (reward having feet on ground)
        contacts = self._get_foot_contacts()
        r_contact = 0.1 * np.sum(contacts)  # reward each grounded foot

        # ── PENALTIES ──

        # 5. Energy penalty
        torques = self.data.actuator_force[:N_ACTUATORS]
        energy = np.sum(np.abs(torques * joint_vel))
        p_energy = 0.0003 * energy

        # 6. Action rate penalty (smooth motion)
        action_diff = action - self._prev_action
        p_action_rate = 0.01 * np.sum(action_diff**2)

        # 7. Action magnitude penalty (prefer staying near standing pose)
        p_action_mag = 0.003 * np.sum(action**2)

        # 8. Lateral velocity penalty
        p_lateral = 0.3 * lateral_vel**2

        # 9. Angular velocity penalty (no spinning)
        p_angvel = 0.03 * np.sum(angvel_body**2)

        # 10. Joint velocity penalty (smooth joints)
        p_joint_vel = 0.0005 * np.sum(joint_vel**2)

        reward = (r_velocity + r_alive + r_upright + r_contact
                  - p_energy - p_action_rate - p_action_mag
                  - p_lateral - p_angvel - p_joint_vel)

        return float(reward)

    def _is_terminated(self):
        """Check if episode should end (fallen)."""
        body_pos = self.data.xpos[self._torso_id]
        proj_grav = self._get_projected_gravity()

        # Body too low (collapsed)
        if body_pos[2] < 0.50:
            return True

        # Body too tilted (proj_gravity z close to 0 or positive = sideways/flipped)
        if proj_grav[2] > -0.3:
            return True

        return False

    def _is_truncated(self):
        return self._step_count >= MAX_EPISODE_STEPS

    def _apply_domain_randomization(self):
        """Randomize physics parameters for robust policy."""
        if not self.randomize or self.np_random is None:
            return

        # Mass randomization (±15%)
        mass_scale = self.np_random.uniform(0.85, 1.15)
        self.model.body_mass[:] = self._default_body_mass * mass_scale

        # Foot friction randomization (0.5-2.0×)
        friction_scale = self.np_random.uniform(0.5, 2.0)
        for fid in self._foot_geom_ids:
            self.model.geom_friction[fid] = (
                self._default_geom_friction[fid] * friction_scale)

    def set_target_speed(self, speed):
        """Set target forward speed (used by curriculum callback)."""
        self.target_speed = speed

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset to keyframe (standing pose)
        mujoco.mj_resetData(self.model, self.data)
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "standing")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # Domain randomization
        self._apply_domain_randomization()

        # Add small noise to initial joint positions (not freejoint)
        if self.np_random is not None:
            for i, idx in enumerate(self._joint_qpos_idxs):
                noise = self.np_random.uniform(-0.03, 0.03)
                self.data.qpos[idx] += noise
                # Clamp to joint limits
                self.data.qpos[idx] = np.clip(
                    self.data.qpos[idx],
                    self._joint_lower[i], self._joint_upper[i])

            # Small velocity noise
            for idx in self._joint_qvel_idxs:
                self.data.qvel[idx] = self.np_random.uniform(-0.05, 0.05)

        # Set initial ctrl to standing pose (PD will maintain it)
        self.data.ctrl[:N_ACTUATORS] = DEFAULT_ANGLES

        # Settle physics (let the cow find stable contact)
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

        self._prev_action = np.zeros(N_ACTUATORS, dtype=np.float32)
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Convert action to target joint positions
        # action=0 → target=default (standing), action=±1 → ±action_scale offset
        target = DEFAULT_ANGLES + action * ACTION_SCALE

        # Clamp to joint limits
        target = np.clip(target, self._joint_lower, self._joint_upper)

        # Send target positions to PD actuators
        self.data.ctrl[:N_ACTUATORS] = target

        # Step physics
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        self._prev_action = action.copy()

        # Diagnostic info
        linvel_body, _ = self._get_body_velocity()
        info = {
            "forward_vel": float(linvel_body[0]),
            "height": float(self.data.xpos[self._torso_id][2]),
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


# Register environment
gym.register(
    id="HighlandCowWalk-v0",
    entry_point="envs.cow_walk_env:HighlandCowWalkEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)
