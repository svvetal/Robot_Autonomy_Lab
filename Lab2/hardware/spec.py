"""Explicitly locked stage ordering from lab2.task.ManipulationTask.run_one.

The same simulation task generates the plan; this list is an extra validation barrier.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

STAGES = (
    '01_table_pregrasp_top', '02_table_grasp_top', '03_close_gripper_attach',
    '04_lift_cube_from_table', '05_left_shelf_preplacement',
    '06_place_left_from_above', '07_release_on_left_shelf',
    '08_retract_after_left_place', '09_side_grasp_approach',
    '10_side_grasp', '11_close_gripper_regrasp',
    '12_side_grasp_lift_clear_shelf', '13_side_grasp_retreat',
    '14_right_shelf_side_approach', '15_place_on_right_shelf',
    '16_release_on_right_shelf', '17_right_shelf_retract',
    '18_return_home',
)
GRIP_ACTIONS = {
    '03_close_gripper_attach': 'close',
    '07_release_on_left_shelf': 'open',
    '11_close_gripper_regrasp': 'close',
    '16_release_on_right_shelf': 'open',
}
SCHEMA_VERSION = 1
# Standalone constants for old ROS/Franka Python 3.8 environments.
# Mirrored directly from lab2/config.py; planning still imports and uses original code.
HOME = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
TRIAL_SEED_STRIDE = 7919
# MuJoCo Panda joint ranges with Lab2's 0.035 rad safety margin.
JOINT_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -.0175, -2.8973]) + .035
JOINT_HI = np.array([2.8973, 1.7628, 2.8973, -.0698, 2.8973, 3.7525, 2.8973]) - .035


def sampled_targets(seed, trials, i):
    # Exact translation of lab2.config.targets_for_trial for on-robot validation.
    rng = np.random.default_rng(int(seed) + TRIAL_SEED_STRIDE * i)
    left_low = .430 + (.560-.430)*i / trials
    left_high = .430 + (.560-.430)*(i+1) / trials
    other = trials-i-1
    right_low = .430 + (.560-.430)*other / trials
    right_high = .430 + (.560-.430)*(other+1) / trials
    left = [float(rng.uniform(left_low, left_high)),
            float(rng.uniform(.311, .329)), .1475+.020+.003]
    right = [float(rng.uniform(right_low, right_high)),
             float(rng.uniform(-.329, -.311)),
             (.1475, .3475, .5475)[i % 3]+.020+.003]
    return left, right, i % 3


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _vec7(value, label):
    x = np.asarray(value, dtype=float)
    if x.shape != (7,) or not np.isfinite(x).all():
        raise ValueError(f'{label} must contain seven finite joint angles')
    return x


def validate_plan(doc):
    """Structural/continuity verification; no substitute for collision checking."""
    if doc.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported plan schema_version')
    count = doc.get('trials')
    seed = doc.get('seed')
    if not isinstance(count, int) or count < 1 or not isinstance(seed, int):
        raise ValueError('Invalid trial count or seed')
    records = doc.get('runs')
    if not isinstance(records, list) or len(records) != count:
        raise ValueError('Expected one run per trial')
    home_tool_pose = np.asarray(doc.get('home_tool_pose'), dtype=float)
    if home_tool_pose.shape != (4, 4) or not np.isfinite(home_tool_pose).all():
        raise ValueError('Missing 4x4 home tool pose')
    if not np.allclose(doc.get('home'), HOME, atol=1e-7):
        raise ValueError('Expected original MuJoCo HOME joint configuration')
    for i, run in enumerate(records):
        if run.get('trial') != i + 1 or run.get('seed') != seed + TRIAL_SEED_STRIDE * i:
            raise ValueError(f'Trial numbering or seed mismatch at {i+1}')
        left, right, idx = sampled_targets(seed, count, i)
        if (not np.allclose(run.get('left_xyz'), left, atol=1e-7)
                or not np.allclose(run.get('right_xyz'), right, atol=1e-7)
                or run.get('right_shelf') != idx + 1):
            raise ValueError(f'Shelf coordinates no longer match the sim sampler in trial {i+1}')
        items = run.get('stages', [])
        if [s.get('name') for s in items] != list(STAGES):
            raise ValueError(f'18-stage sequence mismatch in trial {i+1}')
        prev = HOME.copy()
        held = False
        for stage in items:
            name = stage['name']
            expected_action = GRIP_ACTIONS.get(name)
            if expected_action:
                if stage.get('type') != 'grip' or stage.get('action') != expected_action:
                    raise ValueError(f'Invalid gripper operation at {name}')
                if held == (expected_action == 'close'):
                    raise ValueError(f'Invalid attachment state at {name}')
                held = expected_action == 'close'
                continue
            if stage.get('type') != 'move' or stage.get('carrying') is not held:
                raise ValueError(f'Invalid move/carry state at {name}')
            path = stage.get('path')
            times = stage.get('durations')
            if not isinstance(path, list) or len(path) < 2 or len(times) != len(path) - 1:
                raise ValueError(f'Empty/incorrect path at {name}')
            tool_pose = np.asarray(stage.get('goal_tool_pose'), float)
            if tool_pose.shape != (4, 4) or not np.isfinite(tool_pose).all():
                raise ValueError(f'Missing goal tool pose: {name}')
            joints = [_vec7(q, name) for q in path]
            if any(np.any(q < JOINT_LO - 1e-6) or np.any(q > JOINT_HI + 1e-6)
                   for q in joints):
                raise ValueError(f'Joint waypoint outside bounded original sim limits: {name}')
            if np.max(np.abs(joints[0]-prev)) > 0.001:
                raise ValueError(f'Path discontinuity at {name}')
            if not all(np.isfinite(t) and 0.5 <= t <= 180 for t in times):
                raise ValueError(f'Invalid duration at {name}')
            prev = joints[-1]
        if held or np.max(np.abs(prev - HOME)) > 0.001:
            raise ValueError(f'Trial {i+1} did not release block and return HOME')
    return True
