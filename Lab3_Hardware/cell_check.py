#!/usr/bin/env python3
"""Robot PC: visit every table cell -- hover, descend, grasp, open, back to hover.

Checks that the taped cube positions match where the robot really goes. Uses
outputs/cell_check_plan.json (built and collision-validated by make_cell_check.py
on the MuJoCo PC) and the same motion code and safety checks as `execute`:
position-controlled goto_joints, the plan's smooth single/2-segment descents,
joint + tool-frame checks after every move.

Table setup: ONE cube only. Before each cell the script asks you to put it on
that cell's mark. Descents were validated with nothing else on the table.

  python3 -u cell_check.py --ack-safety                       # all 17 cells, time-scale 2.0
  python3 -u cell_check.py --ack-safety --cells g11 sb3 sr0   # subset (keeps the safe order)
  python3 -u cell_check.py --ack-safety --no-grasp            # descend + open only, no closing
  python3 -u cell_check.py --ack-safety --straight-segments 3 # fail-safe descents
  python3 -u cell_check.py --ack-safety --dry-run sim         # rehearse in MuJoCo (PC)

A good cell: grasp width ~0.042 m and the cube does NOT visibly shift when the
fingers close. If it shifts, note the direction: the tape mark is off that way.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs'
CHECK = OUT / 'cell_check_plan.json'
PLAN = OUT / 'hardware_plan.json'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--ack-safety', action='store_true', help='E-STOP attendant present')
    p.add_argument('--cells', nargs='+', help='subset of cells (default: all 17)')
    p.add_argument('--time-scale', type=float, default=2.0)
    p.add_argument('--no-grasp', action='store_true', help='descend and open only, never close')
    p.add_argument('--straight-segments', type=int, choices=(1, 2, 3), default=None)
    p.add_argument('--check', type=Path, default=CHECK)
    p.add_argument('--log', type=Path, default=OUT / 'hardware_run_log.jsonl')
    p.add_argument('--dry-run', choices=('sim',))
    p.add_argument('--viewer', action='store_true')
    args = p.parse_args(argv)
    if not args.ack_safety:
        p.error('Motion locked: pass --ack-safety (E-STOP attendant present)')

    from lab3_hw.spec import HOME, sha256_file
    from lab3_hw.execute import (SafetyLimits, require_close, check_tool, prompt_operator,
                                 log_event, play_move, joint_error)
    doc = json.loads(Path(args.check).read_text(encoding='utf-8'))
    if doc.get('schema') != 'lab3-cell-check-1' or \
            doc.get('validation', {}).get('result') != 'cell_check_collision_validation_passed':
        raise SystemExit('{} is not a validated cell-check plan: run make_cell_check.py'.format(args.check))
    if doc['plan_sha256'] != sha256_file(PLAN):
        raise SystemExit('cell_check_plan.json was built from a different hardware_plan.json: rebuild it')
    if np.max(np.abs(np.asarray(doc['home']) - HOME)) > 1e-9:
        raise SystemExit('HOME differs from the cell-check plan')
    limits = SafetyLimits()
    limits.straight_segments = args.straight_segments
    if args.time_scale < limits.min_time_scale:
        p.error('--time-scale must be >= 1')

    entries = {e['cell']: e for e in doc['cells']}
    order = [c for c in doc['order'] if args.cells is None or c in args.cells]
    unknown = sorted(set(args.cells or []) - set(entries))
    if unknown or not order:
        p.error('unknown cells {}; valid: {}'.format(unknown, doc['order']))

    def descent_ev(e):
        return dict(type='move', name='check_{}_descend'.format(e['cell']), smooth=True,
                    segments=e['segments'], path=e['path'], durations=e['durations'],
                    goal_tool_pose=e['grasp_tool_pose'])

    def ascent_ev(e):
        return dict(type='move', name='check_{}_ascend'.format(e['cell']), smooth=True,
                    segments=e['segments'], path=list(reversed(e['path'])),
                    durations=e['durations'], goal_tool_pose=e['hover_tool_pose'])

    def joint_move(name, a, b, goal_pose):
        """Validated transit if it exists in the plan, else via HOME (also validated)."""
        for t in doc['transits']:
            if np.allclose(t['path'][0], a, atol=1e-9) and np.allclose(t['path'][-1], b, atol=1e-9):
                return dict(type='move', name=name, smooth=False, path=t['path'],
                            durations=t['durations'], goal_tool_pose=goal_pose)
        return None

    def go(robot, a_name, a_q, b_name, b_q, goal_pose):
        ev = joint_move('{}->{}'.format(a_name, b_name), a_q, b_q, goal_pose)
        if ev is None:   # subset run: chain through HOME with validated HOME legs
            home = list(map(float, HOME))
            for x_name, x, y_name, y, pose in ((a_name, a_q, 'HOME', home, doc['home_tool_pose']),
                                               ('HOME', home, b_name, b_q, goal_pose)):
                if x_name == y_name:
                    continue
                leg = joint_move('{}->{}'.format(x_name, y_name), x, y, pose)
                if leg is None:
                    raise SystemExit('No validated move {} -> {}'.format(x_name, y_name))
                play_move(robot, leg, limits, args.time_scale, 'waypoints')
            return
        play_move(robot, ev, limits, args.time_scale, 'waypoints')

    if args.dry_run:
        from lab3_hw.simrobot import SimRobot
        robot = SimRobot(seed=doc['seed'], viewer=args.viewer, realtime=args.viewer)
    else:
        from lab3_hw.robot import FrankaRobot
        robot = FrankaRobot()

    def place_single_cube(cell):          # dry run only: one cube on the cell, others far away
        from lab3.geometry import make_transform
        names = sorted(robot.sim.blocks)
        for k, b in enumerate(names):
            xyz = entries[cell]['xyz'] if k == 0 else [2.0 + 0.1 * k, 2.0, 0.5]
            robot.sim.set_block(robot.sim.data, b, make_transform(np.asarray(xyz, float), np.eye(3)))
        robot.sim.mj.mj_forward(robot.sim.model, robot.sim.data)

    results = []
    try:
        print('CELL CHECK: {} cells in this order: {}'.format(len(order), ' '.join(order)))
        print('Table: ONE cube only; you move it to each cell when asked. Gripper opens to 6 cm.')
        require_close(robot.joints(), HOME, limits.start_joint_rad, 'Initial HOME')
        check_tool(robot, doc['home_tool_pose'], limits, 'Initial HOME tool frame')
        prompt_operator('Workspace clear, E-STOP attendant ready, ONE cube on the table?')
        robot.open_to_pregrasp()
        log_event(args.log, status='cell_check_started', cells=order, time_scale=args.time_scale,
                  grasp=not args.no_grasp)
        here_name, here_q = 'HOME', list(map(float, HOME))
        for i, cell in enumerate(order):
            e = entries[cell]
            x, y, z = e['xyz']
            print('\n[{}/{}] {}  cube centre x={:.3f} y={:+.3f}  (descent from {})'.format(
                i + 1, len(order), cell, x, y, e['source']))
            if args.dry_run:
                place_single_cube(cell)
            prompt_operator('Put the cube on the {} mark (nothing else on the table). Move to its hover?'
                            .format(cell))
            go(robot, here_name, here_q, cell, e['hover_q'], e['hover_tool_pose'])
            here_name, here_q = cell, e['hover_q']
            n = args.straight_segments or e['segments']
            prompt_operator('At hover above {}. Descend ({} segment(s))?'.format(cell, n))
            pos, ang = play_move(robot, descent_ev(e), limits, args.time_scale, 'waypoints')
            H = robot.tool_pose()
            rec = dict(cell=cell, planned_cube_xyz=e['xyz'], tool_xyz=H[:3, 3].tolist(),
                       tool_err_m=pos, width_m=None, grasp='skipped')
            if not args.no_grasp:
                try:
                    rec['width_m'] = robot.grasp_cube()
                    rec['grasp'] = 'ok'
                except Exception as exc:          # report, then open and go up anyway
                    rec['width_m'] = robot.gripper_width()
                    rec['grasp'] = 'FAILED: {}'.format(exc)
                print('  grasp {}: width {:.4f} m. Did the cube SHIFT when the fingers closed? '
                      'Note the direction.'.format(rec['grasp'], rec['width_m']))
            log_event(args.log, status='cell_check_cell', **rec)
            results.append(rec)
            prompt_operator('Open to 6 cm and go back up to the hover?')
            robot.open_to_pregrasp()
            play_move(robot, ascent_ev(e), limits, args.time_scale, 'waypoints')
        prompt_operator('\nAll cells done. Return to HOME?')
        go(robot, here_name, here_q, 'HOME', list(map(float, HOME)), doc['home_tool_pose'])
        require_close(robot.joints(), HOME, limits.start_joint_rad, 'Final HOME')
        log_event(args.log, status='cell_check_completed')
    except BaseException as exc:
        robot.stop()
        log_event(args.log, status='ABORTED', error=str(exc))
        print('ABORTED: {}. The arm stays where it is.'.format(exc))
        raise
    finally:
        if results:
            print('\n cell  grasp     width    tool error')
            for r in results:
                print(' {:4s}  {:8s}  {}  {:.1f} mm'.format(
                    r['cell'], r['grasp'][:8],
                    '  -    ' if r['width_m'] is None else '{:.4f}'.format(r['width_m']),
                    r['tool_err_m'] * 1e3))
        if args.dry_run:
            robot.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
