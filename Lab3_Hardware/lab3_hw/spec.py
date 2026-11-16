"""Hardware plan format, constants and structural validation (numpy only, Python 3.8).

A plan is produced offline by `hardware_main.py plan` (MuJoCo) and executed on
the robot PC by `hardware_main.py execute` (frankapy). Layout:

{
  "schema": SCHEMA, "seed": 13, "created_utc": ...,
  "sim_sources_sha256": ..., "layout": {...}, "cells": {cell: [x, y, z]},
  "home": [7], "home_tool_pose": 4x4, "gripper": {...},
  "initial_occupancy": {cell: block|null}, "block_colour": {block: colour},
  "patterns": [
    {"name": "pickaxe", "status": "done" | "infeasible",
     "symbolic_actions": [...], "start_occupancy": {...}, "end_occupancy": {...},
     "events": [
        {"type": "move", "name", "carrying", "smooth", "path", "durations", "goal_tool_pose"},
        {"type": "grip", "action": "close" | "open", "block", "cell", "xyz"}, ...]}]
}
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .trajectory import max_joint_speed

SCHEMA = 'lab3-hardware-plan/1'
COLLISION_RESOLUTION_RAD = 0.010
HOME = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8])
# frankapy FrankaArm.reset_joints() pose (FC.HOME_JOINTS) -- NOT the plan's HOME.
FRANKAPY_RESET = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
HOME_MOVE_SECONDS = 6.0
JOINT_LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]) + 0.035
JOINT_HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]) - 0.035
# Waypoints mode plays each straight descent/lift as a few position-controlled goto_joints
# chords. capture.py stores per move the fewest chords (1 = one smooth Lab 2 style motion)
# whose tool path stays within MAX_STRAIGHT_DRIFT_M of the straight line; validate.py
# re-checks that and collision-checks the chords actually executed.
MAX_STRAIGHT_DRIFT_M = 0.005
MAX_SEGMENTS = 3
FAILSAFE_SEGMENTS = 3


def segment_indices(n_points, segments):
    """Indices of the path points used as chord ends for `segments` chords."""
    segments = max(1, min(int(segments), n_points - 1))
    return sorted(set(int(round(x)) for x in np.linspace(0, n_points - 1, segments + 1)))


MAX_JOINT_SPEED = 1.0          # rad/s at time_scale 1 (Franka limits are 2.1-2.6)

# frankapy gripper widths are the TOTAL finger separation (sim stores per finger).
PREGRASP_WIDTH = 0.060         # = 2 x GRIP_OPEN (0.030). NEVER open fully in the grid:
#                                an 8 cm opening hits neighbouring cubes.
CUBE_WIDTH = 0.042            # measured lab cube edge
GRASP_WIDTH_RANGE = (0.032, 0.048)


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(payload, indent=1, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def load_plan(path):
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_plan(doc)
    return doc


def _vec7(v, label):
    q = np.asarray(v, float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError('{}: need seven finite joint angles'.format(label))
    return q


def _pose(v, label):
    H = np.asarray(v, float)
    if H.shape != (4, 4) or not np.isfinite(H).all():
        raise ValueError('{}: need a finite 4x4 pose'.format(label))
    return H


def validate_plan(doc):
    """Structural + kinematic sanity of a plan (no MuJoCo needed)."""
    if not isinstance(doc, dict) or doc.get('schema') != SCHEMA:
        raise ValueError('Unsupported plan schema; regenerate with hardware_main.py plan')
    if not np.allclose(_vec7(doc.get('home'), 'home'), HOME, atol=1e-7):
        raise ValueError('Unexpected HOME configuration in plan')
    _pose(doc.get('home_tool_pose'), 'home_tool_pose')
    cells = doc.get('cells')
    if not isinstance(cells, dict) or len(cells) != 17:
        raise ValueError('Plan must list the 17 cell positions (9 grid + 8 storage)')
    occ = doc.get('initial_occupancy')
    colour = doc.get('block_colour')
    if not isinstance(occ, dict) or set(occ) != set(cells) or not isinstance(colour, dict):
        raise ValueError('Bad initial occupancy / colours')
    patterns = doc.get('patterns')
    if not isinstance(patterns, list) or not patterns:
        raise ValueError('Plan has no patterns')

    last, held, occ = HOME.copy(), None, dict(occ)
    home_between = doc.get('home_between', True)
    if not isinstance(home_between, bool):
        raise ValueError('home_between must be true/false')
    feasible = [p.get('name') for p in patterns if p.get('status') != 'infeasible']
    for pat in patterns:
        name = pat.get('name')
        if pat.get('start_occupancy') != occ:
            raise ValueError('{}: start occupancy does not follow from previous pattern'.format(name))
        if pat.get('status') == 'infeasible':
            if pat.get('events'):
                raise ValueError('{}: infeasible pattern must have no events'.format(name))
            if pat.get('end_occupancy') != occ:
                raise ValueError('{}: infeasible pattern must not change the table'.format(name))
            continue
        if pat.get('status') != 'done':
            raise ValueError('{}: only fully planned patterns can go to hardware'.format(name))
        for k, ev in enumerate(pat.get('events', [])):
            label = '{} event {} ({})'.format(name, k, ev.get('name', ev.get('action')))
            if ev.get('type') == 'grip':
                if ev.get('action') == 'close':
                    if held is not None or occ.get(ev.get('cell')) != ev.get('block'):
                        raise ValueError(label + ': close without that cube in the cell')
                    held, occ[ev['cell']] = ev['block'], None
                elif ev.get('action') == 'open':
                    if held != ev.get('block') or occ.get(ev.get('cell')) is not None:
                        raise ValueError(label + ': open into an occupied cell / wrong cube')
                    if not np.allclose(ev.get('xyz'), cells[ev['cell']], atol=1e-6):
                        raise ValueError(label + ': release point is not the cell centre')
                    occ[ev['cell']], held = held, None
                else:
                    raise ValueError(label + ': unknown grip action')
                continue
            if ev.get('type') != 'move':
                raise ValueError(label + ': unknown event type')
            if bool(ev.get('carrying')) != (held is not None):
                raise ValueError(label + ': carrying flag inconsistent with gripper state')
            qs = [_vec7(q, label) for q in ev.get('path', [])]
            durs = ev.get('durations')
            if len(qs) < 2 or not isinstance(durs, list):
                raise ValueError(label + ': path too short')
            if 'segments' in ev and (not ev.get('smooth') or type(ev['segments']) is not int
                                     or not 1 <= ev['segments'] <= MAX_SEGMENTS):
                raise ValueError(label + ': invalid straight-move segment count')
            if len(durs) != (1 if ev.get('smooth') else len(qs) - 1):
                raise ValueError(label + ': durations do not match path')
            if not all(isinstance(t, (int, float)) and 0.1 <= t <= 120 for t in durs):
                raise ValueError(label + ': implausible duration')
            if any(np.any(q < JOINT_LO - 1e-6) or np.any(q > JOINT_HI + 1e-6) for q in qs):
                raise ValueError(label + ': joint outside safety limits')
            if np.max(np.abs(qs[0] - last)) > 1e-3:
                raise ValueError(label + ': path does not start where the last one ended')
            if max_joint_speed(ev) > MAX_JOINT_SPEED:
                raise ValueError(label + ': joint speed above {} rad/s'.format(MAX_JOINT_SPEED))
            _pose(ev.get('goal_tool_pose'), label)
            last = qs[-1]
        if held is not None:
            raise ValueError('{}: pattern ends holding a cube'.format(name))
        if (home_between or name == feasible[-1]) and np.max(np.abs(last - HOME)) > 1e-3:
            raise ValueError('{}: pattern must end at HOME'.format(name))
        if pat.get('end_occupancy') != occ:
            raise ValueError('{}: events do not produce the recorded end occupancy'.format(name))
    return True


def render_table(occupancy, colour):
    """ASCII view of the table as seen from BEHIND the robot (robot at the bottom)."""
    sym = lambda b: '.' if b is None else ('R' if colour[b] == 'red' else 'B')
    lines = ['   far side (+x)']
    for i in range(4):
        red = occupancy.get('sr{}'.format(i))
        blue = occupancy.get('sb{}'.format(i))
        mid = ''.join(sym(occupancy.get('g{}{}'.format(i, c))) for c in range(3)) if i < 3 else '   '
        lines.append(' {}   {}   {}'.format(sym(red), mid, sym(blue)))
    lines.append('  left(+y)  right(-y)   [robot]')
    return '\n'.join(lines)
