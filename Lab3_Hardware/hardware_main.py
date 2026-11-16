#!/usr/bin/env python3
"""Lab 3 on the real Franka: plan in MuJoCo, validate, then execute with frankapy.

Offline (any PC with MuJoCo + Lab3_Code; NO robot):
  python3 hardware_main.py plan                 # capture the simulation's exact motions
  python3 hardware_main.py validate             # independent collision replay -> attestation
  python3 hardware_main.py inspect              # what will run
  python3 hardware_main.py layout               # where to tape cells / place cubes
  python3 hardware_main.py execute --dry-run sim --viewer --auto-stages --ack-safety
                                                # full hardware code path against MuJoCo

Robot PC (docker with frankapy; NO MuJoCo needed):
  python3 hardware_main.py measure --guide 30   # calibration helper (hand guiding)
  python3 hardware_main.py home --ack-safety    # frankapy reset pose -> plan HOME
  python3 hardware_main.py check-hardware       # read-only HOME / tool-frame check
  python3 hardware_main.py execute --ack-safety [--only pickaxe | --start box]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs'
PLAN = OUT / 'hardware_plan.json'
VALID = OUT / 'hardware_plan_validation.json'


def attestation(plan_path, doc, checked):
    from lab3_hw.spec import sha256_file
    return dict(schema=1, plan_sha256=sha256_file(plan_path),
                sim_sources_sha256=doc['sim_sources_sha256'], verified_samples=int(checked),
                result='offline_mujoco_collision_validation_passed',
                warning='Not proof of real-world alignment or physical safety')


def verify_attestation(record_path, plan_path, doc):
    from lab3_hw.spec import sha256_file
    try:
        rec = json.loads(Path(record_path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise SystemExit('Missing validation record {} -- run "validate" first ({})'.format(
            record_path, exc))
    if (rec.get('plan_sha256') != sha256_file(plan_path) or
            rec.get('sim_sources_sha256') != doc['sim_sources_sha256'] or
            rec.get('result') != 'offline_mujoco_collision_validation_passed'):
        raise SystemExit('Validation record does not match this plan: re-run "validate"')
    return rec


def _limits(straight_segments):
    from lab3_hw.execute import SafetyLimits
    lim = SafetyLimits()
    lim.straight_segments = straight_segments
    return lim


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('plan', help='capture the simulation motions into a plan (MuJoCo)')
    a.add_argument('--seed', type=int, default=13)
    a.add_argument('--patterns', nargs='+', default=None)
    a.add_argument('--out', type=Path, default=PLAN)
    a.add_argument('--no-home-between', action='store_true',
                   help='no return to HOME between patterns (HOME only at the end)')
    a = sub.add_parser('validate', help='collision-replay a plan in MuJoCo, write attestation')
    a.add_argument('--plan', type=Path, default=PLAN)
    a.add_argument('--out', type=Path, default=VALID)
    for name, hlp in (('inspect', 'print the plan'), ('layout', 'print cube/cell coordinates')):
        a = sub.add_parser(name, help=hlp)
        a.add_argument('--plan', type=Path, default=PLAN)
    a = sub.add_parser('measure', help='robot: read tool pose (optionally after guide mode)')
    a.add_argument('--guide', type=float, default=0.0, help='seconds of hand-guiding first')
    a.add_argument('--cell', default='g11', help='cell whose cube the fingers surround')
    a.add_argument('--plan', type=Path, default=PLAN)
    a = sub.add_parser('home', help='robot: move from frankapy reset pose to the plan HOME')
    a.add_argument('--plan', type=Path, default=PLAN)
    a.add_argument('--validation', type=Path, default=VALID)
    a.add_argument('--ack-safety', action='store_true')
    a = sub.add_parser('check-hardware', help='robot: read-only HOME and tool-frame check')
    a.add_argument('--plan', type=Path, default=PLAN)
    a.add_argument('--validation', type=Path, default=VALID)
    a = sub.add_parser('execute', help='run the validated plan on the robot')
    a.add_argument('--plan', type=Path, default=PLAN)
    a.add_argument('--validation', type=Path, default=VALID)
    g = a.add_mutually_exclusive_group()
    g.add_argument('--only', help='run just this pattern (cubes arranged as its start)')
    g.add_argument('--start', help='resume from this pattern onwards')
    a.add_argument('--ack-safety', action='store_true', help='E-STOP attendant present')
    a.add_argument('--auto-stages', action='store_true', help='no per-move prompts')
    a.add_argument('--time-scale', type=float, default=1.5,
                   help='slow-down vs simulation (>=1, default 1.5; 1.0 = sim speed)')
    a.add_argument('--motion', choices=('stream', 'waypoints', 'cartesian'), default='stream',
                   help='cartesian: straight descents/lifts via goto_pose, other moves via goto_joints')
    a.add_argument('--straight-segments', type=int, choices=(1, 2, 3), default=None,
                   help='waypoints mode: force this many chords per straight move '
                        '(default: per-move value from the plan; 3 = fail-safe)')
    a.add_argument('--dry-run', choices=('sim',), help='execute against MuJoCo, not the robot')
    a.add_argument('--viewer', action='store_true', help='dry run: show the MuJoCo viewer')
    a.add_argument('--log', type=Path, default=OUT / 'hardware_run_log.jsonl')
    args = p.parse_args(argv)

    from lab3_hw.spec import atomic_json, load_plan, render_table

    if args.cmd == 'plan':
        from lab3_hw.simlink import use_sim
        use_sim()
        from lab3.determinism import ensure_deterministic_hashing
        ensure_deterministic_hashing()    # same symbolic plans as run_lab3.py
        from lab3_hw.capture import build_plan
        doc = build_plan(args.seed, args.patterns, home_between=not args.no_home_between)
        atomic_json(args.out, doc)
        print('Saved {} ({} patterns). Next: hardware_main.py validate'.format(
            args.out, len(doc['patterns'])))
        return 0

    doc = load_plan(args.plan)

    if args.cmd == 'validate':
        from lab3_hw.validate import validate_collisions
        n = validate_collisions(doc)
        atomic_json(args.out, attestation(args.plan, doc, n))
        print('PASS: {} joint states collision-checked; wrote {}'.format(n, args.out))
        return 0

    if args.cmd == 'inspect':
        print('Plan: seed {}, created {}'.format(doc['seed'], doc['created_utc']))
        print('Layout:', json.dumps(doc['layout']))
        for pat in doc['patterns']:
            moves = [e for e in pat['events'] if e['type'] == 'move']
            t = sum(sum(e['durations']) for e in moves)
            print('\n== {} [{}]  {} moves, {:.0f} s at sim speed'.format(
                pat['name'], pat['status'], len(moves), t))
            for s in pat.get('symbolic_actions', []):
                print('     ', s)
            print(render_table(pat['end_occupancy'], doc['block_colour']))
        return 0

    if args.cmd == 'layout':
        print('Cube-centre coordinates in the robot base frame (metres):')
        for c, (x, y, z) in sorted(doc['cells'].items()):
            b = doc['initial_occupancy'][c]
            print('  {:4s} x={:.3f}  y={:+.3f}  z={:.4f}   start: {}'.format(
                c, x, y, z, '{} ({})'.format(b, doc['block_colour'][b]) if b else 'empty'))
        print(render_table(doc['initial_occupancy'], doc['block_colour']))
        return 0

    if args.cmd == 'measure':
        from lab3_hw.robot import FrankaRobot
        robot = FrankaRobot()
        if args.guide > 0:
            print('Guide mode for {:.0f} s: put the OPEN fingers around the cube at {} '
                  'at grasp depth, centred, fingers along the grid lines.'.format(args.guide, args.cell))
            robot.guide(args.guide)
        H = robot.tool_pose()
        x, y, z = H[:3, 3]
        ex, ey, ez = doc['cells'][args.cell]
        print('Tool (franka_tool) at x={:.4f} y={:+.4f} z={:.4f}'.format(x, y, z))
        print('Plan expects the {} cube centre at x={:.4f} y={:+.4f} z={:.4f}'.format(args.cell, ex, ey, ez))
        print('Offset (measured - plan): dx={:+.1f} mm dy={:+.1f} mm dz={:+.1f} mm'.format(
            (x - ex) * 1e3, (y - ey) * 1e3, (z - ez) * 1e3))
        if args.cell == 'g11':
            gx, gy = doc['layout']['grid_center_xy']
            print('If the offset is > 3 mm, put this in Lab3_Code/table_layout.json and re-plan:')
            print(json.dumps({'grid_center_xy': [round(gx + x - ex, 4), round(gy + y - ey, 4)],
                              'table_top_z': round(doc['layout']['table_top_z'] + z - ez, 4)}))
        return 0

    if args.cmd == 'home':
        if not args.ack_safety:
            p.error('Motion locked: pass --ack-safety (E-STOP attendant present)')
        verify_attestation(args.validation, args.plan, doc)
        from lab3_hw.robot import FrankaRobot
        from lab3_hw.execute import prompt_operator, joint_error
        from lab3_hw.spec import HOME, FRANKAPY_RESET, HOME_MOVE_SECONDS
        robot = FrankaRobot()
        q = robot.joints()
        if joint_error(q, HOME) < 0.01:
            print('Already at the plan HOME.')
            return 0
        if joint_error(q, FRANKAPY_RESET) > 0.10:
            raise SystemExit('Arm is neither at frankapy reset pose nor HOME (max error {:.3f} rad). '
                             'Run reset_joints.py first.'.format(joint_error(q, FRANKAPY_RESET)))
        prompt_operator('Move reset pose -> plan HOME in {:.0f} s (validated offline)?'.format(
            HOME_MOVE_SECONDS))
        robot.move_joints(FRANKAPY_RESET if joint_error(q, FRANKAPY_RESET) > 0.01 else q, 2.0)
        robot.move_joints(HOME, HOME_MOVE_SECONDS)
        print('At HOME (max joint error {:.4f} rad).'.format(joint_error(robot.joints(), HOME)))
        return 0

    if args.cmd == 'check-hardware':
        verify_attestation(args.validation, args.plan, doc)
        from lab3_hw.robot import FrankaRobot
        from lab3_hw.execute import SafetyLimits, require_close, check_tool
        from lab3_hw.spec import HOME
        robot = FrankaRobot()
        print('Joints:', robot.joints().round(4).tolist())
        print('Tool xyz:', robot.tool_pose()[:3, 3].round(4).tolist(),
              ' sim HOME tool xyz:', [round(r[3], 4) for r in doc['home_tool_pose'][:3]])
        print('Gripper width: {:.4f} m'.format(robot.gripper_width()))
        require_close(robot.joints(), HOME, SafetyLimits.start_joint_rad, 'HOME')
        pos, ang = check_tool(robot, doc['home_tool_pose'], SafetyLimits(), 'HOME tool frame')
        print('READ-ONLY CHECK PASSED (tool error {:.4f} m, {:.4f} rad)'.format(pos, ang))
        return 0

    if args.cmd == 'execute':
        if not args.ack_safety:
            p.error('Motion locked: pass --ack-safety (E-STOP attendant present)')
        verify_attestation(args.validation, args.plan, doc)
        from lab3_hw.execute import execute
        if args.dry_run:
            from lab3_hw.simrobot import SimRobot
            from lab3_hw.execute import select_patterns
            robot = SimRobot(seed=doc['seed'], viewer=args.viewer, realtime=args.viewer)
            first = doc['patterns'][select_patterns(doc, args.start, args.only)[0]]
            robot.arrange(first['start_occupancy'], doc['cells'])
        else:
            from lab3_hw.robot import FrankaRobot
            robot = FrankaRobot()
        try:
            execute(doc, robot, start=args.start, only=args.only, auto_stages=args.auto_stages,
                    time_scale=args.time_scale, motion=args.motion, log_path=args.log,
                    limits=_limits(args.straight_segments))
        finally:
            if args.dry_run:
                robot.close()
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
