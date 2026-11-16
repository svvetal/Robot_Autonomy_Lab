"""Supervised execution of a validated Lab 3 plan on the Franka (Python 3.8).

Mirrors the simulation event by event: every planned move is replayed with the
same joint path and time profile (optionally slowed by --time-scale), every
grip is a real gripper command, and joint/tool-frame agreement with the
simulation is checked after every move. Any deviation stops the robot.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .robot import HardwareError, pose_mismatch
from .spec import HOME, FAILSAFE_SEGMENTS, segment_indices, validate_plan, render_table
from .trajectory import sample_move


class SafetyLimits(object):
    start_joint_rad = 0.05      # robot must be this close to a move's first waypoint
    goal_joint_rad = 0.03       # ... and to its last waypoint afterwards
    settle_joint_rad = 0.008    # above this, a 1 s goto_joints removes residual error
    pose_translation_m = 0.010  # sim vs robot tool frame
    pose_angle_rad = 0.10
    min_time_scale = 1.0        # never faster than the simulation
    stream_dt = 0.02            # 50 Hz, as in frankapy's dynamic-joint example
    waypoint_min_s = 1.0
    # waypoints mode: goto_joints chords per straight descent/lift. None -> use each move's
    # validated 'segments' (1 = single smooth motion when drift <= 5 mm); an int forces that
    # count everywhere (3 = fail-safe: worst drift 1.15 mm, 2 pauses).
    straight_segments = None
    # frankapy goto_joints can return (joint_motion_generator_acceleration_discontinuity)
    # before a long move has physically finished; re-read for up to this long first.
    finish_wait_s = 2.0
    cartesian_joint_rad = 0.10  # goto_pose picks its own elbow posture: check the TOOL, allow joint drift


def joint_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def require_close(actual, expected, tol, label):
    err = joint_error(actual, expected)
    if not np.isfinite(err) or err > tol:
        raise HardwareError('{}: joint error {:.4f} rad > {:.4f}; STOP'.format(label, err, tol))
    return err


def wait_until_close(robot, target, tol, max_wait_s, poll_s=0.1):
    """Re-read the joints until within `tol` of `target` or `max_wait_s` passes.
    Nothing is commanded meanwhile; the caller still enforces `tol` afterwards."""
    err = joint_error(robot.joints(), target)
    waited = 0.0
    while err > tol and waited < max_wait_s:
        time.sleep(poll_s)
        waited += poll_s
        err = joint_error(robot.joints(), target)
    if waited > 0:
        print('    (move reported done early; waited {:.1f} s, joint error now {:.4f} rad)'.format(
            waited, err))
    return err


def check_tool(robot, expected_H, limits, label):
    pos, ang = pose_mismatch(np.asarray(expected_H, float), robot.tool_pose())
    if pos > limits.pose_translation_m or ang > limits.pose_angle_rad:
        raise HardwareError('{}: sim/robot tool frame mismatch {:.4f} m / {:.3f} rad; STOP'
                            .format(label, pos, ang))
    return pos, ang


def prompt_operator(msg):
    if input(msg + ' Type YES to continue; anything else aborts: ').strip() != 'YES':
        raise HardwareError('Operator cancelled; arm remains where it is')


def log_event(path, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(utc=datetime.now(timezone.utc).isoformat(), **fields)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec, allow_nan=False, default=float) + '\n')


def play_move(robot, ev, limits, time_scale, motion):
    path = [np.asarray(q, float) for q in ev['path']]
    if motion == 'cartesian' and ev.get('smooth'):
        return _play_cartesian(robot, ev, path, limits, time_scale)
    require_close(robot.joints(), path[0], limits.start_joint_rad, ev['name'] + ' start')
    if motion == 'stream':
        _, qs = sample_move(ev, limits.stream_dt, time_scale)
        robot.stream(qs, limits.stream_dt)
    else:                                       # Lab 2 style, stops at waypoints
        if ev.get('smooth'):                    # straight descent/lift: few evenly spaced segments
            n = limits.straight_segments or ev.get('segments', FAILSAFE_SEGMENTS)
            pts = [path[i] for i in segment_indices(len(path), n)]
        else:
            pts = path
        total = sum(ev['durations']) * time_scale
        steps = list(zip(pts, pts[1:]))
        lengths = [np.max(np.abs(b - a)) for a, b in steps]
        for (a, b), L in zip(steps, lengths):
            if L < 1e-9:
                continue
            robot.move_joints(b, max(limits.waypoint_min_s, total * L / max(sum(lengths), 1e-9)))
    # Wait until the arm has really ARRIVED (settle tolerance), not just come within the
    # abort limit: goto_joints can return while the skill is still executing on the robot,
    # and a settle command sent then would overlap the running skill.
    err = wait_until_close(robot, path[-1], limits.settle_joint_rad, limits.finish_wait_s)
    if limits.settle_joint_rad < err <= limits.goal_joint_rad:
        robot.move_joints(path[-1], 1.0)
        wait_until_close(robot, path[-1], limits.settle_joint_rad, limits.finish_wait_s)
    require_close(robot.joints(), path[-1], limits.goal_joint_rad, ev['name'] + ' goal')
    return check_tool(robot, ev['goal_tool_pose'], limits, ev['name'])


def _play_cartesian(robot, ev, path, limits, time_scale):
    """Straight descents/lifts as ONE smooth Cartesian move (frankapy goto_pose).

    Target = planned end position, CURRENT tool orientation (no twist from the small
    sim/robot frame difference). The Cartesian controller chooses its own redundant
    posture, so joints are checked loosely and the tool pose strictly. Joints are
    re-aligned to the plan only when the move ends in the air (lift / retreat),
    never next to the cubes.
    """
    require_close(robot.joints(), path[0], limits.cartesian_joint_rad, ev['name'] + ' start')
    H0 = robot.tool_pose()
    H = H0.copy()
    H[:3, 3] = np.asarray(ev['goal_tool_pose'], float)[:3, 3]
    robot.move_tool_linear(H, sum(ev['durations']) * time_scale, path=path)
    going_up = H[2, 3] > H0[2, 3]
    err = joint_error(robot.joints(), path[-1])
    if going_up and limits.settle_joint_rad < err <= limits.cartesian_joint_rad:
        robot.move_joints(path[-1], 2.0 if err > limits.goal_joint_rad else 1.0)
    tol = limits.goal_joint_rad if going_up else limits.cartesian_joint_rad
    require_close(robot.joints(), path[-1], tol, ev['name'] + ' goal')
    return check_tool(robot, ev['goal_tool_pose'], limits, ev['name'])


def select_patterns(doc, start=None, only=None):
    names = [p['name'] for p in doc['patterns']]
    for n in (start, only):
        if n is not None and n not in names:
            raise ValueError('Unknown pattern {!r}; plan has {}'.format(n, names))
    if only is not None:
        return [names.index(only)]
    first = names.index(start) if start else 0
    return list(range(first, len(names)))


def execute(doc, robot, start=None, only=None, auto_stages=False, time_scale=1.5,
            motion='stream', confirm=prompt_operator, log_path='outputs/hardware_run_log.jsonl',
            limits=None, out=print):
    validate_plan(doc)
    limits = limits or SafetyLimits()
    if time_scale < limits.min_time_scale:
        raise ValueError('time_scale must be >= {}'.format(limits.min_time_scale))
    idx = select_patterns(doc, start, only)
    colour = doc['block_colour']
    first_move = next((e for i in idx for e in doc['patterns'][i]['events'] if e['type'] == 'move'), None)
    if first_move is not None and np.max(np.abs(np.asarray(first_move['path'][0]) - HOME)) > 1e-3:
        raise HardwareError('This plan does not return to HOME between patterns, so {} does not '
                            'start at HOME. Run the whole sequence, or use the plan that returns '
                            'HOME (outputs/hardware_plan.json) for --start/--only.'.format(
                                doc['patterns'][idx[0]]['name']))

    out('Supervised Lab 3 execution. A physical E-STOP attendant is REQUIRED.')
    require_close(robot.joints(), HOME, limits.start_joint_rad, 'Initial HOME')
    check_tool(robot, doc['home_tool_pose'], limits, 'Initial HOME tool frame')
    first = doc['patterns'][idx[0]]
    out('Arrange the cubes EXACTLY like this (view from behind the robot):')
    out(render_table(first['start_occupancy'], colour))
    for cell, b in sorted(first['start_occupancy'].items()):
        if b is not None:
            out('  {:5s} {:4s} centre at x={:.3f} y={:+.3f} m'.format(
                cell, colour[b], doc['cells'][cell][0], doc['cells'][cell][1]))
    confirm('Cubes placed as shown, workspace clear, E-STOP attendant ready?')
    robot.open_to_pregrasp()
    log_event(log_path, status='session_started', patterns=[doc['patterns'][i]['name'] for i in idx],
              time_scale=time_scale, motion=motion)

    try:
        for i in idx:
            pat = doc['patterns'][i]
            if pat['status'] == 'infeasible':
                out('=== {}: no plan exists (infeasible) -> SKIPPED'.format(pat['name']))
                log_event(log_path, pattern=pat['name'], status='skipped_infeasible')
                continue
            out('=== {}: {} symbolic actions, {} robot events'.format(
                pat['name'], len(pat['symbolic_actions']), len(pat['events'])))
            if not auto_stages:
                confirm('Start pattern {}?'.format(pat['name']))
            for k, ev in enumerate(pat['events']):
                label = '{} [{}/{}]'.format(pat['name'], k + 1, len(pat['events']))
                if ev['type'] == 'grip':
                    if ev['action'] == 'close':
                        out('{} close on {} at {}'.format(label, ev['block'], ev['cell']))
                        w = robot.grasp_cube()
                        if not auto_stages:
                            confirm('Cube {} held firmly between the fingers (width {:.3f} m)?'
                                    .format(ev['block'], w))
                    else:
                        out('{} release {} at {}'.format(label, ev['block'], ev['cell']))
                        w = robot.open_to_pregrasp()
                    log_event(log_path, pattern=pat['name'], event=k, grip=ev['action'],
                              block=ev['block'], cell=ev['cell'], width_m=w)
                    continue
                if not auto_stages:
                    confirm('{} next: {} ({} waypoints).'.format(label, ev['name'], len(ev['path'])))
                out('{} {} ({:.1f} s)'.format(label, ev['name'], sum(ev['durations']) * time_scale))
                pos, ang = play_move(robot, ev, limits, time_scale, motion)
                log_event(log_path, pattern=pat['name'], event=k, move=ev['name'],
                          tool_err_m=pos, tool_err_rad=ang)
            out('=== {} DONE. Expected table:'.format(pat['name']))
            out(render_table(pat['end_occupancy'], colour))
            log_event(log_path, pattern=pat['name'], status='completed')
            if not auto_stages:
                confirm('Table matches the expected {} layout?'.format(pat['name']))
    except BaseException as exc:
        robot.stop()
        log_event(log_path, status='ABORTED', error=str(exc))
        out('ABORTED: {}. No automatic recovery: the arm stays where it is. '
            'Use guide mode / reset_joints under supervision.'.format(exc))
        raise
    require_close(robot.joints(), HOME, limits.start_joint_rad, 'Final HOME')
    log_event(log_path, status='session_completed')
    out('All selected patterns completed.')
