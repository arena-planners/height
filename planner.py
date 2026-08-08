"""HEIGHT (selfAttn_merge_SRNN_lidar) wrapper for the arena_planners bridge.

The released turtlebot checkpoint uses a DISCRETE action space (gym.spaces.Discrete(9)),
so the policy head is Categorical and act() returns an action INDEX, not [v, omega]. The
index maps to a (delta_v, delta_w) increment via action_convert; the planner accumulates
those increments into a desiredVelocity = [v, w] state (clipped to the robot limits) exactly
as the upstream env does in CrowdSim3DTB.step. That accumulated [v, omega] is what we command.

Observation construction reproduces CrowdSim3DTbObs-v0.generate_ob:
  robot_node      (1, 5)  = [px, py, gx, gy, theta]                         world frame
  spatial_edges   (N, 4)  = [rel_px, rel_py, hvx, hvy]                      robot frame, pad 15
  detected_human_num (1,) = visible human count, clamped >= 1
  point_clouds    (1, R)  = lidar ranges in metres, clamped to sensor_range, robot-relative,
                            index 0 = robot heading (theta), increasing CCW, R = 360/angular_res
  obstacle_vertices / obstacle_num / temporal_edges are in the obs dict but NOT read by the
  lidar network (see selfAttn_srnn_merge_lidar.forward); supplied as zeros to satisfy any
  obs-space sanity check inside Policy.__init__.

GRU hidden state is a single key 'rnn' of shape (1, nenv, human_node_rnn_size). Both the GRU
state and the desiredVelocity accumulator are cleared in on_reset.
"""

from __future__ import annotations

import glob
import math
import os
import pathlib
import sys

import numpy as np
import torch
from arena_planners.sdk import load_manifest, main_loop

# Vendored upstream code lives flat under this directory; expose it on sys.path so the
# upstream's `from training.networks...` / `from crowd_nav...` absolute imports resolve.
sys.path.insert(0, os.path.dirname(__file__))

_MODEL_DIR = pathlib.Path(__file__).parent / "model"

_GOAL_MAX_DIST: float = 4.0
_PAD_VALUE: float = 15.0
def _pick_device() -> torch.device:
    """CUDA only if it can actually launch a kernel; this torch build predates newer GPUs."""
    if torch.cuda.is_available():
        try:
            (torch.zeros(1, device="cuda") + 1).cpu()
            return torch.device("cuda")
        except RuntimeError:
            pass
    return torch.device("cpu")


_DEVICE = _pick_device()

# Discrete action table from CrowdSim3DTB.set_action_space (config.env.action_space == 'discrete').
# index -> [delta_v (m/s), delta_w (rad/s)].
_ACTION_CONVERT: dict[int, tuple[float, float]] = {
    0: (0.05, 0.1), 1: (0.05, 0.0), 2: (0.05, -0.1),
    3: (0.0, 0.1), 4: (0.0, 0.0), 5: (0.0, -0.1),
    6: (-0.05, 0.1), 7: (-0.05, 0.0), 8: (-0.05, -0.1),
}

_actor_critic: torch.nn.Module | None = None
_config: object | None = None
_rnn_hxs: dict[str, torch.Tensor] | None = None
_desired_velocity: list[float] | None = None  # [v, w] accumulator, mirrors env desiredVelocity
_max_human_num: int = 11
_ray_num: int = 180
_sensor_range: float = 25.0
_v_min: float = -0.5
_v_max: float = 0.5
_w_min: float = -1.0
_w_max: float = 1.0
_rnn_size: int = 128


def _build_policy() -> torch.nn.Module:
    import gym
    from crowd_nav.configs.config import Config
    from training.networks.model import Policy

    global _config, _max_human_num, _ray_num, _sensor_range
    global _v_min, _v_max, _w_min, _w_max, _rnn_size

    config = Config()
    # Force single-process CPU/GPU inference; the trained weights are independent of these.
    config.training.num_processes = 1
    config.ppo.num_mini_batch = 1
    config.training.cuda = _DEVICE.type == "cuda"
    _config = config

    _max_human_num = (
        config.sim.human_num
        + config.sim.human_num_range
        + config.sim.static_human_num
        + config.sim.static_human_range
    )
    _ray_num = int(360.0 / config.lidar.angular_res)
    _sensor_range = float(config.lidar.sensor_range)
    _v_min, _v_max = float(config.robot.v_min), float(config.robot.v_max)
    _w_min, _w_max = float(config.robot.w_min), float(config.robot.w_max)
    _rnn_size = int(config.SRNN.human_node_rnn_size)

    obs_space = gym.spaces.Dict({
        "robot_node": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5), dtype=np.float32),
        "temporal_edges": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32),
        "spatial_edges": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(max(1, _max_human_num), 4), dtype=np.float32),
        "detected_human_num": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        "obstacle_vertices": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12, 8), dtype=np.float32),
        "obstacle_num": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        "point_clouds": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, _ray_num), dtype=np.float32),
    })
    # Released turtlebot checkpoint = discrete 9-action space (Categorical head).
    action_space = gym.spaces.Discrete(len(_ACTION_CONVERT))

    actor_critic = Policy(obs_space.spaces, action_space, base="selfAttn_merge_srnn_lidar", base_kwargs=config)

    weights = _resolve_weights()
    state = torch.load(str(weights), map_location=_DEVICE)
    if isinstance(state, list):
        state = state[0]
    actor_critic.load_state_dict(state, strict=True)
    actor_critic.to(_DEVICE)
    actor_critic.eval()
    return actor_critic


def _resolve_weights() -> pathlib.Path:
    fixed = _MODEL_DIR / "height_policy.pt"
    if fixed.exists():
        return fixed
    matches = sorted(glob.glob(str(_MODEL_DIR / "*.pt")))
    if not matches:
        raise FileNotFoundError(f"no checkpoint found in {_MODEL_DIR}")
    return pathlib.Path(matches[0])


def _init_state() -> None:
    global _rnn_hxs, _desired_velocity
    # selfAttn_merge_SRNN_lidar.forward indexes rnn_hxs['rnn']; single key, (1, nenv, rnn_size).
    _rnn_hxs = {"rnn": torch.zeros(1, 1, _rnn_size, device=_DEVICE)}
    _desired_velocity = [0.0, 0.0]


def _world_to_robot(x: float, y: float, theta: float) -> tuple[float, float]:
    # Mirrors CrowdSim3DTbObs.world_to_robot: rot_angle = -(theta - pi/2).
    rot = -(theta - math.pi / 2.0)
    c, s = math.cos(rot), math.sin(rot)
    return c * x - s * y, s * x + c * y


def _resample_scan(ranges: np.ndarray) -> np.ndarray:
    """Resample an arbitrary-length range array to _ray_num beams over a full circle.

    HEIGHT's lidar is robot-relative: ray 0 points along the robot heading, increasing CCW,
    angular_res degrees apart over 360 degrees. The arena LaserScan is also robot-relative
    (laser frame), so a uniform full-circle resample preserves the angular mapping when the
    arena scan is full-circle. Partial-FOV scans are padded with sensor_range outside the FOV
    by the caller before this resample is reached.
    """
    arr = np.asarray(ranges, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, _sensor_range)
    arr = np.clip(arr, 0.0, _sensor_range)
    if len(arr) == _ray_num:
        return arr
    xp = np.linspace(0.0, 1.0, len(arr), dtype=np.float32)
    x = np.linspace(0.0, 1.0, _ray_num, dtype=np.float32)
    return np.interp(x, xp, arr).astype(np.float32)


def step(features: dict) -> list[float]:
    global _actor_critic, _rnn_hxs, _desired_velocity

    if _actor_critic is None:
        _actor_critic = _build_policy()
    if _rnn_hxs is None or _desired_velocity is None:
        _init_state()

    robot_pose = features.get("robot_pose")
    robot_state = features.get("robot_state")
    if robot_pose is None or robot_state is None:
        return [0.0, 0.0]
    px, py, theta = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    vx, vy = float(robot_state[2]), float(robot_state[3])

    goal_pose = features.get("goal_pose")
    target: tuple[float, float] | None = None
    if goal_pose is not None:
        target = (float(goal_pose[0]), float(goal_pose[1]))
    if target is None:
        return [0.0, 0.0]
    gx, gy = target

    # The 'absolute' checkpoint trained with px, py, gx, gy in a +/-4 m box around the origin, so
    # recentre on the robot and clamp the goal offset to stay in-distribution (a far goal spins it).
    gdx, gdy = gx - px, gy - py
    gdist = math.hypot(gdx, gdy)
    if gdist > _GOAL_MAX_DIST:
        gdx, gdy = gdx * _GOAL_MAX_DIST / gdist, gdy * _GOAL_MAX_DIST / gdist
    robot_node = np.array([[0.0, 0.0, gdx, gdy, theta]], dtype=np.float32)
    temporal_edges = np.array([[vx, vy]], dtype=np.float32)

    # spatial_edges: (max_human_num, 4) = [rel_px, rel_py, hvx, hvy] robot frame, pad 15, sorted by dist.
    spatial = np.full((max(1, _max_human_num), 4), _PAD_VALUE, dtype=np.float32)
    peds = features.get("pedestrians")
    if peds is None:
        peds = []
    visible_count = 0
    for i, ped in enumerate(peds[:_max_human_num]):
        rel_x, rel_y = _world_to_robot(float(ped[1]) - px, float(ped[2]) - py, theta)
        hvx, hvy = _world_to_robot(float(ped[3]), float(ped[4]), theta)
        spatial[i] = (rel_x, rel_y, hvx, hvy)
        visible_count += 1
    order = np.argsort(np.linalg.norm(spatial[:, :2], axis=1))
    spatial = spatial[order]
    detected_n = max(visible_count, 1)

    laser_raw = features.get("laser_scan")
    if laser_raw is not None and len(laser_raw) > 0:
        scan = _resample_scan(np.asarray(laser_raw, dtype=np.float32))
    else:
        scan = np.full(_ray_num, _sensor_range, dtype=np.float32)
    point_clouds = scan[np.newaxis, :]

    obs = {
        "robot_node": torch.from_numpy(robot_node).to(_DEVICE).unsqueeze(0),
        "temporal_edges": torch.from_numpy(temporal_edges).to(_DEVICE).unsqueeze(0),
        "spatial_edges": torch.from_numpy(spatial).to(_DEVICE).unsqueeze(0),
        "detected_human_num": torch.tensor([[float(detected_n)]], device=_DEVICE),
        "obstacle_vertices": torch.zeros(1, 12, 8, device=_DEVICE),
        "obstacle_num": torch.zeros(1, 1, device=_DEVICE),
        "point_clouds": torch.from_numpy(point_clouds).to(_DEVICE).unsqueeze(0),
    }
    masks = torch.ones(1, 1, device=_DEVICE)

    with torch.no_grad():
        _, action, _, new_hxs = _actor_critic.act(obs, _rnn_hxs, masks, deterministic=True)
    _rnn_hxs = new_hxs

    idx = int(action.squeeze().cpu().item())
    delta_v, delta_w = _ACTION_CONVERT[idx]
    _desired_velocity[0] = float(np.clip(_desired_velocity[0] + delta_v, _v_min, _v_max))
    _desired_velocity[1] = float(np.clip(_desired_velocity[1] + delta_w, _w_min, _w_max))
    return [_desired_velocity[0], _desired_velocity[1]]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    global _rnn_hxs, _desired_velocity
    _rnn_hxs = None
    _desired_velocity = None


if __name__ == "__main__":
    manifest = load_manifest(pathlib.Path(__file__).parent / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
