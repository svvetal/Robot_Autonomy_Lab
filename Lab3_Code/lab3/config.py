"""Lab 3 simulation/planning parameters (extends the Lab 2 settings)."""
from dataclasses import dataclass
import numpy as np

HOME = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
BLOCK_HALF = 0.021          # real lab cubes are 4.2 cm
TABLE_TOP = 0.0001
BLOCK_Z = TABLE_TOP + BLOCK_HALF + 0.002          # resting cube-centre height

# Gripper: finger joint value = distance of each pad's inner face from centre.
GRIP_OPEN_WIDE = 0.040     # Lab 2 value, used when nothing is nearby
GRIP_OPEN = 0.030          # pre-grasp opening inside the grid (1 cm clearance per side)
GRIP_CLOSE = 0.0215         # per finger, 4.2 cm cube
TOOL_SITE_OFFSET = 0.103

# Top-down grasp, fingers closing along world Y (yaw 0) ...
TOP_R = np.diag([1., -1., -1.])


def top_grasp_R(yaw: float) -> np.ndarray:
    """Top-down tool orientation rotated by `yaw` about world Z.

    yaw=0      -> fingers close along world Y  (symbolic axis 'y')
    yaw=pi/2   -> fingers close along world X  (symbolic axis 'x')
    """
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return Rz @ TOP_R


GRASP_YAW = {'y': 0.0, 'x': np.pi / 2}


@dataclass(frozen=True)
class Settings:
    seed: int = 13
    hover_z: float = 0.10          # pre-grasp / pre-place height above cube centre
    lift_z: float = 0.12           # transport height above cube centre
    placement_gap: float = 0.003
    safe_joint_margin: float = 0.035
    collision_margin: float = 0.004
    rrt_step: float = 0.17
    rrt_edge_resolution: float = 0.010
    rrt_iterations: int = 3000
    rrt_goal_bias: float = 0.18
    rrt_shortcuts: int = 120
    ik_trials: int = 14
    ik_position_tolerance: float = 0.006
    ik_orientation_tolerance: float = 0.10
    joint_speed: float = 0.65
    segment_min_seconds: float = 0.6
    settle_seconds: float = 0.35
    command_hold_seconds: float = 0.45
    video_fps: int = 25
    render_width: int = 960
    render_height: int = 720
    max_symbolic_expansions: int = 50000
