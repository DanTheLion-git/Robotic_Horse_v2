"""
Highland Cow Walking Environment — Gymnasium + MuJoCo

Observation (52 dims):
  - body height                     (1)
  - body orientation (quaternion)   (4)
  - body linear velocity            (3)
  - body angular velocity           (3)
  - joint positions (13 actuated)   (13)
  - joint velocities (13 actuated)  (13)
  - previous action                 (13)
  - foot contact forces (4 feet)    (4) → binarized as contact flags
  Total: 1 + 4 + 3 + 3 + 13 + 13 + 13 + 4 = 54 (but we'll use a flat obs)

Action (13 dims):
  - Torque commands for: spine + 4×(hip, thigh, knee)
  - Normalized to [-1, 1], scaled by gear ratios in MJCF

Reward:
  - Forward velocity tracking (target: 1.0 m/s walk)
  - Alive bonus (upright)
  - Energy penalty (minimize torque × velocity)
  - Orientation penalty (stay level)
  - Smoothness penalty (minimize action rate of change)
  - Foot clearance reward (lift feet during swing)
"""

import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

# Path to the MJCF model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "models", "highland_cow.xml")

# Actuated joint names (must match MJCF <actuator> order)
ACTUATOR_NAMES = [
    "spine_yaw_act",
    "fl_hip_act", "fl_thigh_act", "fl_knee_act",
    "fr_hip_act", "fr_thigh_act", "fr_knee_act",
    "rl_hip_act", "rl_thigh_act", "rl_knee_act",
    "rr_hip_act", "rr_thigh_act", "rr_knee_act",
]
N_ACTUATORS = len(ACTUATOR_NAMES)

# Actuated joint names (for reading positions/velocities)
JOINT_NAMES = [
    "spine_yaw",
    "fl_hip_yaw", "fl_thigh_pitch", "fl_knee_pitch",
    "fr_hip_yaw", "fr_thigh_pitch", "fr_knee_pitch",
    "rl_hip_yaw", "rl_thigh_pitch", "rl_knee_pitch",
    "rr_hip_yaw", "rr_thigh_pitch", "rr_knee_pitch",
]

FOOT_SITE_NAMES = [
    "fl_foot_site", "fr_foot_site", "rl_foot_site", "rr_foot_site",
]

# Reward weights
REWARD_FORWARD_VEL   = 2.0    # reward for moving forward
REWARD_ALIVE         = 0.5    # bonus for staying upright
PENALTY_ENERGY       = 0.005  # penalty per unit of energy (torque × velocity)
PENALTY_ORIENTATION  = 1.0    # penalty for body tilt
PENALTY_ACTION_RATE  = 0.02   # penalty for jerky actions
PENALTY_LATERAL_VEL  = 0.5    # penalty for sideways drift
REWARD_FOOT_CLEARANCE = 0.1   # reward for lifting swing feet

# Episode parameters
TARGET_SPEED     = 1.0   # m/s desired forward speed
MAX_EPISODE_STEPS = 2000  # 2000 × 0.01s = 20 seconds
CONTROL_DT       = 0.01  # 10ms control period (5 physics steps at dt=0.002)
N_SUBSTEPS       = 5     # physics steps per control step


class HighlandCowWalkEnv(gym.Env):
    """MuJoCo environment for training Highland Cow walking gait via RL."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, target_speed=TARGET_SPEED):
        super().__init__()

        self.target_speed = target_speed
        self.render_mode = render_mode

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        # Override timestep to match our control frequency
        self.model.opt.timestep = 0.002

        # Cache joint/actuator/site IDs
        self._joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                           for n in JOINT_NAMES]
        self._joint_qpos_idxs = [self.model.jnt_qposadr[jid] for jid in self._joint_ids]
        self._joint_qvel_idxs = [self.model.jnt_dofadr[jid] for jid in self._joint_ids]
        self._foot_site_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
                               for n in FOOT_SITE_NAMES]

        # Base body ID (for reading position/orientation)
        self._base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_ACTUATORS,), dtype=np.float32)

        # Observation: height(1) + quat(4) + linvel(3) + angvel(3)
        #            + joint_pos(13) + joint_vel(13) + prev_action(13) + contacts(4)
        obs_dim = 1 + 4 + 3 + 3 + 13 + 13 + 13 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._prev_action = np.zeros(N_ACTUATORS, dtype=np.float32)
        self._step_count = 0

        # Renderer
        self._renderer = None
        if render_mode == "human":
            self._init_renderer()

    def _init_renderer(self):
        """Initialize MuJoCo viewer for rendering."""
        try:
            import mujoco.viewer
            self._renderer = mujoco.viewer
        except ImportError:
            print("Warning: mujoco.viewer not available, rendering disabled")

    def _get_obs(self):
        """Build observation vector."""
        # Body height (z position of base_link)
        body_pos = self.data.xpos[self._base_body_id]
        height = np.array([body_pos[2]], dtype=np.float32)

        # Body orientation (quaternion)
        body_quat = self.data.xquat[self._base_body_id].astype(np.float32)

        # Body velocities (in world frame)
        # Use cvel (6D spatial velocity: [angular(3), linear(3)])
        body_vel = self.data.cvel[self._base_body_id]
        linvel = body_vel[3:6].astype(np.float32)
        angvel = body_vel[0:3].astype(np.float32)

        # Joint positions and velocities (actuated joints only)
        joint_pos = np.array([self.data.qpos[idx] for idx in self._joint_qpos_idxs],
                             dtype=np.float32)
        joint_vel = np.array([self.data.qvel[idx] for idx in self._joint_qvel_idxs],
                             dtype=np.float32)
        # Clip velocities for stability
        joint_vel = np.clip(joint_vel, -20.0, 20.0)

        # Foot contact forces (touch sensors → binary contact flags)
        contacts = np.zeros(4, dtype=np.float32)
        for i, sid in enumerate(self._foot_site_ids):
            # Check if foot geom is in contact with ground
            for j in range(self.data.ncon):
                c = self.data.contact[j]
                foot_geom_name = FOOT_SITE_NAMES[i].replace("_site", "_geom")
                foot_geom_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_geom_name)
                floor_geom_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
                if (c.geom1 == foot_geom_id and c.geom2 == floor_geom_id) or \
                   (c.geom2 == foot_geom_id and c.geom1 == floor_geom_id):
                    contacts[i] = 1.0
                    break

        obs = np.concatenate([
            height,           # 1
            body_quat,        # 4
            linvel,           # 3
            angvel,           # 3
            joint_pos,        # 13
            joint_vel,        # 13
            self._prev_action,# 13
            contacts,         # 4
        ])
        return obs

    def _compute_reward(self, action):
        """Compute reward for the current state and action."""
        # Forward velocity (x-direction in world frame)
        body_vel = self.data.cvel[self._base_body_id]
        forward_vel = body_vel[3]  # linear x velocity
        lateral_vel = body_vel[4]  # linear y velocity

        # 1. Forward velocity reward (gaussian around target speed)
        vel_error = forward_vel - self.target_speed
        reward_vel = REWARD_FORWARD_VEL * math.exp(-2.0 * vel_error**2)

        # 2. Alive bonus
        reward_alive = REWARD_ALIVE

        # 3. Energy penalty (torque × joint velocity)
        joint_vel = np.array([self.data.qvel[idx] for idx in self._joint_qvel_idxs])
        gear = np.array([self.model.actuator_gear[i, 0] for i in range(N_ACTUATORS)])
        torques = action * gear
        energy = np.sum(np.abs(torques * joint_vel))
        penalty_energy = PENALTY_ENERGY * energy

        # 4. Orientation penalty (body should stay level)
        body_quat = self.data.xquat[self._base_body_id]
        # Deviation from upright: quat [1,0,0,0] = upright
        # Use 1 - w² as a simple tilt metric
        tilt = 1.0 - body_quat[0]**2
        penalty_orient = PENALTY_ORIENTATION * tilt

        # 5. Action rate penalty (smooth motion)
        action_diff = action - self._prev_action
        penalty_rate = PENALTY_ACTION_RATE * np.sum(action_diff**2)

        # 6. Lateral velocity penalty
        penalty_lateral = PENALTY_LATERAL_VEL * lateral_vel**2

        reward = (reward_vel + reward_alive
                  - penalty_energy - penalty_orient
                  - penalty_rate - penalty_lateral)

        return float(reward)

    def _is_terminated(self):
        """Check if episode should end (fallen over)."""
        body_pos = self.data.xpos[self._base_body_id]
        body_quat = self.data.xquat[self._base_body_id]

        # Fallen: body too low or tilted too much
        height = body_pos[2]
        upright = body_quat[0]  # w component of quaternion

        if height < 0.45:  # body center below 45cm = collapsed
            return True
        if abs(upright) < 0.5:  # tilted more than ~60 degrees
            return True
        return False

    def _is_truncated(self):
        """Check if episode exceeded max steps."""
        return self._step_count >= MAX_EPISODE_STEPS

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Set initial pose: standing with slightly bent legs
        # The model starts at z=1.20, legs should settle to natural stance
        # Add small random perturbation for robustness
        if self.np_random is not None:
            noise = self.np_random.uniform(-0.02, 0.02, size=self.model.nq)
            noise[:7] = 0  # don't perturb free joint (pos + quat)
            self.data.qpos[:] += noise

        # Step physics a few times to settle contacts
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self._prev_action = np.zeros(N_ACTUATORS, dtype=np.float32)
        self._step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Apply action as control signal
        self.data.ctrl[:N_ACTUATORS] = action

        # Step physics N_SUBSTEPS times
        for _ in range(N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        self._prev_action = action.copy()

        info = {
            "forward_vel": float(self.data.cvel[self._base_body_id][3]),
            "height": float(self.data.xpos[self._base_body_id][2]),
            "step": self._step_count,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human" and self._renderer is not None:
            # Handled externally via viewer.launch_passive
            pass
        elif self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            img = renderer.render()
            renderer.close()
            return img

    def close(self):
        pass


# Register the environment with Gymnasium
gym.register(
    id="HighlandCowWalk-v0",
    entry_point="envs.cow_walk_env:HighlandCowWalkEnv",
    max_episode_steps=MAX_EPISODE_STEPS,
)
