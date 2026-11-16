#!/usr/bin/env python3
"""Diagnostic: move to the PRE-GRASP (hover) pose of the first pick only, then stop.

Plays only the first planned move of a pattern (HOME -> hover above the first
cube), with the same validated joint path, streaming and safety checks as
`hardware_main.py execute`. No descent, no grasp. At the hover pose it prints
and logs the tool pose so you can measure, with a ruler, the gap between the
fingertip bottoms and the table. Optionally it then retraces the same move back
to HOME.

  python3 pregrasp_test.py --ack-safety                 # pickaxe, time scale 2.0
  python3 pregrasp_test.py --ack-safety --pattern box --time-scale 3.0
  python3 pregrasp_test.py --ack-safety --dry-run sim --viewer   # rehearse in MuJoCo

Simulation reference at the hover pose (table top at z = 0):
  tool point z = 0.135 m (table_top_z 0.012), fingertip bottoms ~0.113 m above the table.
At the grasp pose the fingertips should be ~0.013 m above the table.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hardware_main import PLAN, VALID, OUT, verify_attestation   # noqa: E402

SIM_FINGER_BOTTOM_AT_HOVER = 0.113   # table_top_z=0.012 plan; was 0.112 (measured 0.100 before)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--pattern', default='pickaxe')
    p.add_argument('--ack-safety', action='store_true', help='E-STOP attendant present')
    p.add_argument('--time-scale', type=float, default=2.0)
    p.add_argument('--motion', choices=('stream', 'waypoints', 'cartesian'), default='stream')
    p.add_argument('--plan', type=Path, default=PLAN)
    p.add_argument('--validation', type=Path, default=VALID)
    p.add_argument('--log', type=Path, default=OUT / 'hardware_run_log.jsonl')
    p.add_argument('--straight-segments', type=int, choices=(1, 2, 3), default=None,
                   help='waypoints mode: force this many chords per straight move '
                        '(default: per-move value from the plan; 3 = fail-safe)')
    p.add_argument('--dry-run', choices=('sim',))
    p.add_argument('--viewer', action='store_true')
    p.add_argument('--home-slowdown', type=float, default=2.5,
                   help='extra slow-down for the hover -> HOME return (>=1, default 2.5)')
    p.add_argument('--recover', action='store_true',
                   help='arm stopped at grasp depth: open, retrace up to hover, then HOME')
    p.add_argument('--grasp', action='store_true',
                   help='after the hover: descend, close on the cube, open, retrace up (no lift)')
    args = p.parse_args(argv)
    if not args.ack_safety:
        p.error('Motion locked: pass --ack-safety (E-STOP attendant present)')

    from lab3_hw.spec import load_plan, validate_plan, HOME
    from lab3_hw.execute import (SafetyLimits, require_close, check_tool, prompt_operator,
                                 log_event, play_move, select_patterns)
    doc = load_plan(args.plan)
    validate_plan(doc)
    verify_attestation(args.validation, args.plan, doc)
    limits = SafetyLimits()
    limits.straight_segments = args.straight_segments
    if args.time_scale < limits.min_time_scale:
        p.error('--time-scale must be >= {}'.format(limits.min_time_scale))

    pat = doc['patterns'][select_patterns(doc, only=args.pattern)[0]]
    if pat['status'] == 'infeasible' or not pat['events']:
        raise SystemExit('Pattern {} has no motions'.format(args.pattern))
    ev = pat['events'][0]
    grip = next(e for e in pat['events'] if e['type'] == 'grip')
    if ev['type'] != 'move' or ev['carrying'] or not ev['name'].endswith('_approach'):
        raise SystemExit('First event of {} is not a pick approach: {}'.format(
            args.pattern, ev.get('name')))
    if np.max(np.abs(np.asarray(ev['path'][0]) - np.asarray(HOME))) > 1e-6:
        raise SystemExit('Approach does not start at HOME; refusing')
    cube = doc['cells'][grip['cell']]
    hover = np.asarray(ev['goal_tool_pose'], float)[:3, 3]

    if args.dry_run:
        from lab3_hw.simrobot import SimRobot
        robot = SimRobot(seed=doc['seed'], viewer=args.viewer, realtime=args.viewer)
        robot.arrange(pat['start_occupancy'], doc['cells'])
    else:
        from lab3_hw.robot import FrankaRobot
        robot = FrankaRobot()

    try:
        if args.recover:
            down = pat['events'][1]
            require_close(robot.joints(), down['path'][-1],
                          limits.cartesian_joint_rad if args.motion == 'cartesian' else limits.start_joint_rad,
                          'Recover: arm is not at the {} grasp-depth pose'.format(grip['cell']))
            print('RECOVER: arm is at grasp depth above {}. Steps: open 6 cm -> retrace up -> HOME.'
                  .format(grip['cell']))
            prompt_operator('Open the gripper to 6 cm?')
            robot.open_to_pregrasp()
            log_event(args.log, status='recover_opened', width_m=robot.gripper_width())
            prompt_operator('Retrace the descent back up to the hover ({:.1f} s)?'.format(
                sum(down['durations']) * args.time_scale))
            up = dict(down, name=down['name'] + '_reverse', path=list(reversed(down['path'])),
                      goal_tool_pose=ev['goal_tool_pose'])
            play_move(robot, up, limits, args.time_scale, args.motion)
            log_event(args.log, status='back_at_hover')
        else:
            _hover(robot, doc, ev, grip, cube, hover, limits, args, prompt_operator, log_event,
                   play_move, require_close, check_tool, HOME, pat)
        # Same validated joint path, one smooth segment, but slower: the hover -> HOME
        # return tripped a joint/cartesian reflex at normal speed on 2026-11-16.
        slow = max(1.0, args.home_slowdown)
        back = dict(ev, name=ev['name'] + '_reverse', path=list(reversed(ev['path'])),
                    durations=[d * slow for d in reversed(ev['durations'])],
                    goal_tool_pose=doc['home_tool_pose'])
        prompt_operator('\nRetrace the same path back to HOME ({:.1f} s)?'.format(
            sum(back['durations']) * args.time_scale))
        play_move(robot, back, limits, args.time_scale, args.motion)
        require_close(robot.joints(), HOME, limits.start_joint_rad, 'Final HOME')
        log_event(args.log, status='pregrasp_test_completed')
        print('Back at HOME.')
    except BaseException as exc:
        robot.stop()
        log_event(args.log, status='ABORTED', error=str(exc))
        print('ABORTED: {}. The arm stays where it is.'.format(exc))
        raise
    finally:
        if args.dry_run:
            robot.close()
    return 0


def _hover(robot, doc, ev, grip, cube, hover, limits, args, prompt_operator, log_event,
           play_move, require_close, check_tool, HOME, pat):
    if True:
        print('PRE-GRASP TEST: {} -> hover above {} ({}) at cube centre x={:.3f} y={:+.3f} z={:.4f}'
              .format(ev['name'], grip['block'], grip['cell'], *cube))
        print('Runs: HOME -> hover' + (' -> descend -> close -> open -> back up -> HOME.'
              if args.grasp else ' -> HOME. No descent, no grasp.'))
        require_close(robot.joints(), HOME, limits.start_joint_rad, 'Initial HOME')
        check_tool(robot, doc['home_tool_pose'], limits, 'Initial HOME tool frame')
        prompt_operator('Workspace clear, E-STOP attendant ready, move {:.1f} s to the hover pose?'
                        .format(sum(ev['durations']) * args.time_scale))
        robot.open_to_pregrasp()
        log_event(args.log, status='pregrasp_test_started', pattern=args.pattern,
                  move=ev['name'], time_scale=args.time_scale, motion=args.motion)
        pos, ang = play_move(robot, ev, limits, args.time_scale, args.motion)
        H = robot.tool_pose()
        q = robot.joints()
        print('\nAt pre-grasp pose. Tool error vs sim: {:.4f} m / {:.4f} rad'.format(pos, ang))
        print('  robot tool (franka_tool) xyz = {:.4f} {:+.4f} {:.4f}'.format(*H[:3, 3]))
        print('  planned tool xyz             = {:.4f} {:+.4f} {:.4f}'.format(*hover))
        print('  tool z axis (should be ~0 0 -1) = {}'.format(np.round(H[:3, 2], 3).tolist()))
        print('  gripper width = {:.4f} m'.format(robot.gripper_width()))
        print('\nMEASURE NOW with a ruler:')
        print('  1. table surface -> bottom of the fingertips   (sim: {:.0f} mm)'.format(
            SIM_FINGER_BOTTOM_AT_HOVER * 1000))
        print('  2. is the gripper centred over the cube in x and y? estimate the offset in mm')
        print('  3. are the fingers parallel to the cube faces?')
        log_event(args.log, status='pregrasp_reached', pattern=args.pattern, cell=grip['cell'],
                  tool_err_m=pos, tool_err_rad=ang, robot_tool_xyz=H[:3, 3].tolist(),
                  planned_tool_xyz=hover.tolist(), planned_cube_xyz=list(cube),
                  tool_z_axis=H[:3, 2].tolist(), joints=q.tolist())

        if args.grasp:
            down = pat['events'][1]
            if down['type'] != 'move' or not down['name'].endswith('_descend') or \
                    pat['events'][2] is not grip:
                raise SystemExit('Unexpected event order after the approach; refusing')
            nseg = (args.straight_segments or down.get('segments', 3)) if args.motion == 'waypoints' else None
            prompt_operator('\nDescend {:.1f} s to grasp depth{} and close on {}?'.format(
                sum(down['durations']) * args.time_scale,
                '' if nseg is None else ' in {} segment(s), plan drift {} mm'.format(
                    nseg, down.get('chord_drift_mm', '?')), grip['block']))
            pos, ang = play_move(robot, down, limits, args.time_scale, args.motion)
            Hd = robot.tool_pose()
            print('At grasp depth: tool z = {:.4f} (planned {:.4f}), error {:.4f} m'.format(
                Hd[2, 3], down['goal_tool_pose'][2][3], pos))
            log_event(args.log, status='grasp_depth_reached', tool_err_m=pos, tool_err_rad=ang,
                      robot_tool_xyz=Hd[:3, 3].tolist(), joints=robot.joints().tolist())
            try:
                w = robot.grasp_cube()
                print('GRASP OK: width {:.4f} m'.format(w))
                log_event(args.log, status='grasp_ok', width_m=w)
            except Exception as exc:          # keep going: open and retreat, don't stay down
                w = robot.gripper_width()
                print('GRASP FAILED: {} (width now {:.4f} m)'.format(exc, w))
                log_event(args.log, status='grasp_failed', error=str(exc), width_m=w)
            prompt_operator('Open to 6 cm and retrace the descent back up to the hover?')
            robot.open_to_pregrasp()
            up = dict(down, name=down['name'] + '_reverse', path=list(reversed(down['path'])),
                      goal_tool_pose=ev['goal_tool_pose'])
            play_move(robot, up, limits, args.time_scale, args.motion)
            log_event(args.log, status='back_at_hover')



if __name__ == '__main__':
    raise SystemExit(main())
