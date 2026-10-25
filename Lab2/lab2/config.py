"""Configuration and task constants. Distances in metres, angles in radians."""
from dataclasses import dataclass
import numpy as np

# From SimpleScript.py. Do NOT adjust any shelf or table geometry.
HOME = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
BLOCK_HALF = 0.020
GRIP_OPEN = 0.040
GRIP_CLOSE = 0.0205
TOOL_SITE_OFFSET = 0.103   # fingertip workspace, defined on hand body in scene.py
TABLE_TOP = 0.0001
TABLE_BLOCK_POS = np.array([0.590, 0.0, TABLE_TOP + BLOCK_HALF + 0.002])
# Original shelf slab top z = center z + half-height; gap leaves small air clearance.
LEFT_TOP = 0.1475
RIGHT_TOPS = (0.1475, 0.3475, 0.5475)

@dataclass(frozen=True)
class Settings:
    seed: int = 13
    trials: int = 3
    # Robot-frame world X is the HORIZONTAL direction along the shelf.
    # Place a cube near the front edge (world Y), but across a visibly
    # wider usable interior length on both shelves (world X). These ranges
    # are candidate regions, not a guarantee of MuJoCo collision feasibility.
    left_x: tuple[float, float] = (0.430, 0.560)
    left_y: tuple[float, float] = (0.311, 0.329)
    right_x: tuple[float, float] = (0.430, 0.560)
    right_y: tuple[float, float] = (-0.329, -0.311)
    clearance_z: float = 0.18
    hover_z: float = 0.14
    lift_z: float = 0.17
    placement_gap: float = 0.003
    safe_joint_margin: float = 0.035
    collision_margin: float = 0.004
    rrt_step: float = 0.17
    rrt_edge_resolution: float = 0.055
    rrt_iterations: int = 3000
    rrt_goal_bias: float = 0.18
    rrt_shortcuts: int = 120
    ik_trials: int = 14
    ik_position_tolerance: float = 0.008
    ik_orientation_tolerance: float = 0.14
    joint_speed: float = 0.65  # rad/s (conservative; quintic peak 1.875x average)
    segment_min_seconds: float = 0.8
    settle_seconds: float = 0.35
    command_hold_seconds: float = 0.45
    video_fps: int = 25
    render_width: int = 960
    render_height: int = 720

# Tool +Z points into the cube. In top grasp the tool approaches from above.
TOP_R = np.diag([1., -1., -1.])
# Finger opening along world X, tool +Z along world +Y. Side approach from y below block.
SIDE_POS_Y_R = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
# Tool +Z along -Y, allowing approach to opposite shelf from central aisle.
SIDE_NEG_Y_R = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])


# The per-trial number changes the independent random stream while keeping the
# base seed under CLI control. Separate sampling from the IK/RRT random stream.
TRIAL_SEED_STRIDE = 7919
PLANNER_SEED_OFFSET = 104729


def trial_seed(base_seed: int, trial: int) -> int:
    if trial < 0:
        raise ValueError('trial index must be nonnegative')
    return int(base_seed) + TRIAL_SEED_STRIDE * trial


def _stratified_uniform(rng, span, stratum: int, strata_count: int):
    """Random horizontal coordinate within one of N nonoverlapping bins.

    For 3 demonstrations this spreads placements along the shelf instead of
    accidentally clustering all three in one region. For 1 trial it is an
    ordinary uniform sample across the full interval.
    """
    lower, upper = map(float, span)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError('invalid shelf X range')
    if not 0 <= stratum < strata_count:
        raise ValueError('invalid horizontal placement bin')
    width = (upper - lower) / strata_count
    return rng.uniform(lower + width*stratum, lower + width*(stratum+1))


def shelf_targets(rng: np.random.Generator, cfg: Settings, trial: int):
    """Seeded, horizontally STRATIFIED targets (world X) for both shelves.

    The single left shelf uses forward bins; right shelves use reverse bins.
    World Y stays near the front edge for valid side regrasp clearance; height
    on the right cycles bottom/middle/top. Shelves themselves are unchanged.
    """
    if not 0 <= trial < cfg.trials:
        raise ValueError('trial out of range')
    left = np.array([
        _stratified_uniform(rng, cfg.left_x, trial, cfg.trials),
        rng.uniform(*cfg.left_y),
        LEFT_TOP + BLOCK_HALF + cfg.placement_gap,
    ])
    right = np.array([
        _stratified_uniform(rng, cfg.right_x, cfg.trials - trial - 1, cfg.trials),
        rng.uniform(*cfg.right_y),
        RIGHT_TOPS[trial % 3] + BLOCK_HALF + cfg.placement_gap,
    ])
    return left, right, trial % 3


def targets_for_trial(cfg: Settings, trial: int):
    """Precompute a trial's placement independent of RRT/IK sampling."""
    return shelf_targets(np.random.default_rng(trial_seed(cfg.seed, trial)), cfg, trial)
