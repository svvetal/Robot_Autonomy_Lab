"""Motion plan format and validation."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .heights import KEYS, heights_fingerprint

STAGES = (
    '01_table_pregrasp_top', '02_table_grasp_top', '03_close_gripper_attach',
    '04_lift_cube_from_table', '05_left_shelf_preplacement',
    '06_place_left_from_above', '07_release_on_left_shelf',
    '08_retract_after_left_place', '09_side_grasp_approach',
    '10_side_grasp', '11_close_gripper_regrasp',
    '12_side_grasp_lift_clear_shelf', '13_side_grasp_retreat',
    '14_right_shelf_side_approach', '15_place_on_right_shelf',
    '16_release_on_right_shelf', '17_right_shelf_retract', '18_return_home',
)
GRIP_ACTIONS = {
    '03_close_gripper_attach': 'close',
    '07_release_on_left_shelf': 'open',
    '11_close_gripper_regrasp': 'close',
    '16_release_on_right_shelf': 'open',
}
SCHEMA_VERSION = 3
# Planning, capture and validation use this joint-space sampling interval.
COLLISION_RESOLUTION_RAD = 0.010
HOME = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
TRIAL_SEED_STRIDE = 7919
JOINT_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -.0175, -2.8973]) + .035
JOINT_HI = np.array([2.8973, 1.7628, 2.8973, -.0698, 2.8973, 3.7525, 2.8973]) - .035

def sampled_targets(seed, trials, i, heights_mm):
    """Generate the same placement targets as the simulator."""
    rng = np.random.default_rng(int(seed) + TRIAL_SEED_STRIDE * i)
    left_low = .430 + (.560-.430)*i / trials
    left_high = .430 + (.560-.430)*(i+1) / trials
    other = trials - i - 1
    right_low = .430 + (.560-.430)*other / trials
    right_high = .430 + (.560-.430)*(other+1) / trials
    left = [float(rng.uniform(left_low, left_high)),
            float(rng.uniform(.311, .329)), heights_mm['left_bottom_mm']/1000. + .023]
    right_key = ('right_bottom_mm', 'right_middle_mm', 'right_top_mm')[i % 3]
    right = [float(rng.uniform(right_low, right_high)),
             float(rng.uniform(-.329, -.311)), heights_mm[right_key]/1000. + .023]
    return left, right, i % 3

def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def _vec7(value, label):
    q = np.asarray(value, float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError('{}: require seven finite joint angles'.format(label))
    return q

def validate_plan(doc):
    """Verify plan structure, waypoints and target consistency."""
    if not isinstance(doc, dict) or doc.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Unsupported plan schema (old trajectories may cross shelf geometry); regenerate using hardware_main.py plan')
    if doc.get('collision_resolution_rad') != COLLISION_RESOLUTION_RAD:
        raise ValueError('Plan does not attest to fine-grained collision edge checks; regenerate')
    n, seed = doc.get('trials'), doc.get('seed')
    if type(n) is not int or n < 1 or type(seed) is not int:
        raise ValueError('Invalid trial count / seed')
    h = doc.get('shelf_heights_mm')
    if not isinstance(h, dict) or set(h) != set(KEYS):
        raise ValueError('Missing 4 calibrated shelf heights in hardware plan')
    if any(isinstance(h[k], bool) or not isinstance(h[k], (float, int)) or
           not np.isfinite(h[k]) or not (60 <= h[k] <= 900) for k in KEYS):
        raise ValueError('Invalid calibrated height units/values')
    if not (h['right_middle_mm'] - h['right_bottom_mm'] > 55 and
            h['right_top_mm'] - h['right_middle_mm'] > 55):
        raise ValueError('Invalid right shelf ordering')
    if doc.get('shelf_heights_sha256') != heights_fingerprint(h):
        raise ValueError('Shelf height fingerprint mismatch: plan corrupt or stale')
    for key in ('source_scene_sha256', 'runtime_scene_sha256'):
        value = doc.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Missing scene provenance: {}'.format(key))
    if not np.allclose(_vec7(doc.get('home'), 'HOME'), HOME, atol=1e-7):
        raise ValueError('Unexpected HOME joint configuration')
    home_pose = np.asarray(doc.get('home_tool_pose'), float)
    if home_pose.shape != (4, 4) or not np.isfinite(home_pose).all():
        raise ValueError('Missing 4x4 simulated home tool transform')
    runs = doc.get('runs')
    if not isinstance(runs, list) or len(runs) != n:
        raise ValueError('Expected one run per trial')
    for i, run in enumerate(runs):
        if run.get('trial') != i+1 or run.get('seed') != seed + TRIAL_SEED_STRIDE * i:
            raise ValueError('Wrong trial number or seed')
        l, r, shelf = sampled_targets(seed, n, i, h)
        if (not np.allclose(run.get('left_xyz'), l, atol=1e-7) or
            not np.allclose(run.get('right_xyz'), r, atol=1e-7) or
            run.get('right_shelf') != shelf + 1):
            raise ValueError('Shelf target mismatch at trial {}'.format(i+1))
        items = run.get('stages')
        if not isinstance(items, list) or [s.get('name') for s in items] != list(STAGES):
            raise ValueError('Wrong 18-stage task sequence')
        last = HOME.copy()
        held = False
        for st in items:
            name = st['name']
            if name in GRIP_ACTIONS:
                op = GRIP_ACTIONS[name]
                if st.get('type') != 'grip' or st.get('action') != op:
                    raise ValueError('Bad gripper stage '+name)
                if held == (op == 'close'):
                    raise ValueError('Incorrect cube attachment state '+name)
                held = op == 'close'
                continue
            if st.get('type') != 'move' or st.get('carrying') is not held:
                raise ValueError('Bad stage move/carry state '+name)
            path, durations = st.get('path'), st.get('durations')
            if not isinstance(path, list) or len(path) < 2 or not isinstance(durations, list) or len(durations) != len(path)-1:
                raise ValueError('Incorrect path or durations at '+name)
            pose = np.asarray(st.get('goal_tool_pose'), float)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError('Invalid tool pose at '+name)
            qs = [_vec7(q, name) for q in path]
            if any(np.any(q < JOINT_LO-1e-6) or np.any(q > JOINT_HI+1e-6) for q in qs):
                raise ValueError('Joint limit outside safe bounds at '+name)
            if np.max(np.abs(qs[0]-last)) > .001:
                raise ValueError('Stage path discontinuity at '+name)
            if not all(isinstance(t,(float,int)) and np.isfinite(t) and 0.5 <= t <= 180 for t in durations):
                raise ValueError('Invalid duration at '+name)
            last = qs[-1]
        if held or np.max(np.abs(last - HOME)) > .001:
            raise ValueError('Trial must release cube and finish at HOME')
    return True
