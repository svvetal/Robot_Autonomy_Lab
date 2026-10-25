#!/usr/bin/env python3
"""Hardware deployment entry point. `plan`, `validate`, `inspect` never touch robot."""
import argparse
import json
from pathlib import Path


def read_plan(path):
    from hardware.spec import validate_plan
    doc = json.loads(Path(path).read_text())
    validate_plan(doc)
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description='CMU Lab2 Franka physical execution')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan', help='Run original MuJoCo task & export identical 18-stage RRT plan')
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--out', type=Path, default=Path('outputs/hardware_plan_seed13.json'))
    p = sub.add_parser('sample', help='Preview original deterministic shelf XYZ (NO MuJoCo, NO robot)')
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--out', type=Path, default=Path('outputs/hardware_target_samples.json'))
    p = sub.add_parser('inspect', help='Print planned locations and all 18 stages (NO robot)')
    p.add_argument('--plan', type=Path, default=Path('outputs/hardware_plan_seed13.json'))
    p = sub.add_parser('validate', help='Recheck each saved edge against MuJoCo scene (NO robot)')
    p.add_argument('--plan', type=Path, default=Path('outputs/hardware_plan_seed13.json'))
    p = sub.add_parser('execute', help='Send commands to physical Franka; operator approval required')
    p.add_argument('--plan', type=Path, default=Path('outputs/hardware_plan_seed13.json'))
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--trial', type=int, help='Execute one trial; defaults to all')
    selection.add_argument('--all', action='store_true', help='Explicitly select all trials')
    p.add_argument('--ack-safety', action='store_true',
                   help='Confirm partner, attended e-stop and calibrated shelf setup')
    p.add_argument('--auto-stages', action='store_true',
                   help='No per-stage approval; still requires pre-trial and grasp checks')
    p.add_argument('--log', type=Path, default=Path('outputs/hardware_run_log.jsonl'))
    args = parser.parse_args(argv)
    if args.command == 'plan':
        from hardware.plan import build_plan
        from hardware.spec import atomic_json
        print('Planning the original 18 stages in MuJoCo. No physical robot motion.')
        data = build_plan(seed=args.seed, trials=args.trials)
        atomic_json(args.out, data)
        print(f'Saved {args.trials} collision-checked trials to {args.out}')
        return 0
    if args.command == 'sample':
        from hardware.spec import sampled_targets, atomic_json, TRIAL_SEED_STRIDE
        if args.trials < 1:
            parser.error('--trials must be positive')
        samples = []
        for i in range(args.trials):
            left, right, idx = sampled_targets(args.seed, args.trials, i)
            samples.append({'trial': i+1, 'seed': args.seed + i*TRIAL_SEED_STRIDE,
                            'table_xyz': [0.590, 0., .0221],
                            'left_xyz': left, 'right_xyz': right, 'right_shelf': idx+1})
        atomic_json(args.out, {'seed': args.seed, 'trials': args.trials,
                               'note': 'OFFLINE SAMPLING ONLY, NOT PHYSICALLY VERIFIED',
                               'samples': samples})
        print(json.dumps(samples, indent=2))
        print('Saved sample targets to', args.out)
        return 0
    data = read_plan(args.plan)
    if args.command == 'inspect':
        for run in data['runs']:
            print(f'TRIAL {run["trial"]} seed={run["seed"]}: '
                  f'left xyz={run["left_xyz"]} | '
                  f'right xyz={run["right_xyz"]} | right level={run["right_shelf"]}')
            for stage in run['stages']:
                if stage['type'] == 'move':
                    print(f'  {stage["name"]}: {len(stage["path"])-1} joint segments, '
                          f'{sum(stage["durations"]):.2f}s planned, '
                          f'goal cube xyz={stage.get("goal_xyz")}')
                else:
                    print(f'  {stage["name"]}: {stage["action"]} gripper')
        return 0
    if args.command == 'validate':
        from hardware.validate import validate_collisions
        checks = validate_collisions(data)
        print(f'PASS: collision-checked {checks} sampled joint states against unchanged MuJoCo scene.')
        print('NOTE: this does not verify the physical robot, block grasp or calibration.')
        return 0
    if args.command == 'execute':
        if not args.ack_safety:
            parser.error('Physical movement locked: add --ack-safety only with partner/e-stop operator')
        from hardware.robot import FrankaRobot
        from hardware.execute import execute
        robot = FrankaRobot()
        which = [args.trial] if args.trial is not None else None
        execute(data, robot, selected_trials=which,
                auto_stages=args.auto_stages, log_path=args.log)
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
