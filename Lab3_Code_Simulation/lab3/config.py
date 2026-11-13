from dataclasses import dataclass
import numpy as np

# Home joints in radians.
HOME = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
BLOCK_HALF = 0.020
# Each gripper joint is the half-gap in meters.
GRIP_OPEN = 0.040
GRIP_CLOSE = 0.0205
TOOL_SITE_OFFSET = 0.103
TOP_R = np.diag([1., -1., -1.])

TABLE_TOP = 0.0001

BLOCK_START_CLEARANCE = 0.0005
BLOCK_CENTER_Z = TABLE_TOP + BLOCK_HALF + BLOCK_START_CLEARANCE

# Central grid coordinates in world-frame meters.
GRID_CENTER_X = 0.460
GRID_CENTER_Y = 0.000
GRID_PITCH = 0.050
GRID_ROWS = 3
GRID_COLS = 3
GRID_CELLS = tuple(f'G{r}{c}' for r in range(GRID_ROWS) for c in range(GRID_COLS))

# Symmetric block staging rows; adjust these to move the initial cubes.
STAGING_X = (0.310, 0.410, 0.510, 0.610)
STAGING_RED_Y = 0.180
STAGING_BLUE_Y = -0.180
BLOCK_NAMES = tuple(f'{color}_{i}' for color in ('red', 'blue') for i in range(4))
BLOCK_COLORS = {name: name.split('_')[0] for name in BLOCK_NAMES}
BLOCK_RGBA = {
    'red': '0.92 0.12 0.07 1',
    'blue': '0.02 0.43 0.80 1',
}

def grid_position(row: int, col: int, *, z: float = BLOCK_CENTER_Z) -> np.ndarray:

    if not (0 <= row < GRID_ROWS and 0 <= col < GRID_COLS):
        raise ValueError(f'Invalid 3x3 grid cell: ({row}, {col})')
    return np.array([GRID_CENTER_X + (1 - row) * GRID_PITCH,
                     GRID_CENTER_Y + (col - 1) * GRID_PITCH,
                     z], dtype=float)

def staging_position(block_name: str, *, z: float = BLOCK_CENTER_Z) -> np.ndarray:

    if block_name not in BLOCK_NAMES:
        raise ValueError(f'Unknown block {block_name!r}')
    color, index = block_name.split('_')
    y = STAGING_RED_Y if color == 'red' else STAGING_BLUE_Y
    return np.array([STAGING_X[int(index)], y, z], dtype=float)

def initial_block_positions() -> dict[str, np.ndarray]:

    return {name: staging_position(name) for name in BLOCK_NAMES}

@dataclass(frozen=True)
class Settings:
    seed: int = 13

    safe_joint_margin: float = 0.035
    collision_margin: float = 0.004
    rrt_step: float = 0.17
    rrt_edge_resolution: float = 0.010
    # Larger IK/RRT budgets improve difficult searches but increase planning time.
    rrt_iterations: int = 2200
    rrt_goal_bias: float = 0.18
    rrt_shortcuts: int = 35
    ik_trials: int = 6
    ik_position_tolerance: float = 0.008
    ik_orientation_tolerance: float = 0.14
    joint_speed: float = 1.0
    segment_min_seconds: float = 0.5
    placement_gap: float = 0.003
    hover_z: float = 0.14
    lift_z: float = 0.17
