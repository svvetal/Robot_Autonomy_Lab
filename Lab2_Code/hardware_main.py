#!/usr/bin/env python3
"""Command-line interface for Franka planning and execution."""
import argparse
import json
from pathlib import Path

def read_plan(path):
    from hardware.spec import validate_plan
    doc=json.loads(Path(path).read_text(encoding='utf-8'))
    validate_plan(doc)
    return doc

def attest_record(plan_path,doc,checked):
    from hardware.spec import sha256_file
    return {
        'schema_version':1,
        'plan_sha256':sha256_file(plan_path),
        'shelf_heights_sha256':doc['shelf_heights_sha256'],
        'runtime_scene_sha256':doc['runtime_scene_sha256'],
        'verified_samples':int(checked),
        'result':'offline_mujoco_collision_validation_passed',
        'warning':'Not proof of real-world shelf alignment or physical safety',
    }

def verify_attestation(record_path,plan_path,doc):
    from hardware.spec import sha256_file
    try: record=json.loads(Path(record_path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('Missing/invalid offline validation record; run hardware_main.py validate --out: {}'.format(exc)) from exc
    if (record.get('schema_version') != 1 or
        record.get('plan_sha256') != sha256_file(plan_path) or
        record.get('shelf_heights_sha256') != doc['shelf_heights_sha256'] or
        record.get('runtime_scene_sha256') != doc['runtime_scene_sha256'] or
        record.get('result') != 'offline_mujoco_collision_validation_passed' or
        type(record.get('verified_samples')) is not int or record['verified_samples'] < 1):
        raise ValueError('Validation record does not match current plan/scene; regenerate and revalidate')
    return record

def main(argv=None):
    p=argparse.ArgumentParser(description='Calibrated Lab2 measured-Z Franka workflow')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('sample',help='Sample cube target XYZ with measured heights (NO robot/MuJoCo)')
    a.add_argument('--seed',type=int,default=13)
    a.add_argument('--trials',type=int,default=3)
    a.add_argument('--out',type=Path,default=Path('outputs/hardware_target_samples.json'))
    a=sub.add_parser('plan',help='Generate calibrated 18-stage motion paths using MuJoCo (NO robot)')
    a.add_argument('--seed',type=int,default=13)
    a.add_argument('--trials',type=int,default=3)
    a.add_argument('--out',type=Path,default=Path('outputs/hardware_plan_measured.json'))
    a=sub.add_parser('inspect',help='Show measured Z and plan stages (NO robot)')
    a.add_argument('--plan',type=Path,default=Path('outputs/hardware_plan_measured.json'))
    a=sub.add_parser('validate',help='Collision-check entire calibrated plan in MuJoCo (NO robot)')
    a.add_argument('--plan',type=Path,default=Path('outputs/hardware_plan_measured.json'))
    a.add_argument('--out',type=Path,default=Path('outputs/hardware_plan_validation.json'))
    a=sub.add_parser('check-hardware',help='Read-only real Franka HOME/tool comparison; NO motion')
    a.add_argument('--plan',type=Path,default=Path('outputs/hardware_plan_measured.json'))
    a.add_argument('--validation',type=Path,default=Path('outputs/hardware_plan_validation.json'))
    a=sub.add_parser('execute',help='Execute only a verified matching plan using frankapy')
    a.add_argument('--plan',type=Path,default=Path('outputs/hardware_plan_measured.json'))
    a.add_argument('--validation',type=Path,default=Path('outputs/hardware_plan_validation.json'))
    selection=a.add_mutually_exclusive_group()
    selection.add_argument('--trial',type=int,help='Execute only the selected trial')
    selection.add_argument('--all',action='store_true',help='Execute all trials, staged setup between trials')
    a.add_argument('--ack-safety',action='store_true',help='Operator explicitly accepts supervised motion')
    a.add_argument('--auto-stages',action='store_true',help='Skip stage prompts ONLY after staged approval')
    a.add_argument('--log',type=Path,default=Path('outputs/hardware_run_log.jsonl'))
    args=p.parse_args(argv)
    from hardware.heights import load_measured_heights,heights_fingerprint,assert_matching_heights
    from hardware.spec import sampled_targets,atomic_json,TRIAL_SEED_STRIDE
    if args.command=='sample':
        if args.trials < 1: p.error('--trials must be positive')
        h=load_measured_heights()
        samples=[]
        for i in range(args.trials):
            left,right,shelf=sampled_targets(args.seed,args.trials,i,h)
            samples.append(dict(trial=i+1,seed=args.seed+i*TRIAL_SEED_STRIDE,
                                table_xyz=[.59,0.,.0221],left_xyz=left,
                                right_xyz=right,right_shelf=shelf+1))
        atomic_json(args.out,dict(seed=args.seed,trials=args.trials,
                                  shelf_heights_mm=h,shelf_heights_sha256=heights_fingerprint(h),
                                  note='NO ROBOT; offline target positions only',samples=samples))
        print(json.dumps(samples,indent=2));print('Saved samples to',args.out)
        return 0
    if args.command=='plan':
        from hardware.plan import build_plan
        doc=build_plan(args.seed,args.trials)
        atomic_json(args.out,doc)
        print('Saved measured-height plan to {}; {} trial(s)'.format(args.out,args.trials))
        print('Shelf tops (mm):',doc['shelf_heights_mm'])
        print('IMPORTANT: run validate to produce hardware_plan_validation.json before hardware execution')
        return 0
    doc=read_plan(args.plan)
    assert_matching_heights(doc)
    if args.command=='inspect':
        print('SHELF TOP HEIGHTS (mm relative to robot base):',doc['shelf_heights_mm'])
        print('Height calibration SHA256:',doc['shelf_heights_sha256'])
        for run in doc['runs']:
            print('TRIAL {}: left XYZ={}, right XYZ={}, right shelf {}'.format(
                run['trial'],run['left_xyz'],run['right_xyz'],run['right_shelf']))
            for stage in run['stages']:
                if stage['type']=='move':
                    print('  {}: {} joint segments'.format(stage['name'],len(stage['path'])-1))
                else: print('  {}: gripper {}'.format(stage['name'],stage['action']))
        return 0
    if args.command=='validate':
        from hardware.validate import validate_collisions
        checked=validate_collisions(doc)
        atomic_json(args.out,attest_record(args.plan,doc,checked))
        print('PASS: {} collision-sampled joint states; wrote {}'.format(checked,args.out))
        print('CAUTION: offline validation cannot prove physical clearance or grasp success.')
        return 0
    if args.command=='check-hardware':

        verify_attestation(args.validation,args.plan,doc)
        from hardware.robot import FrankaRobot
        from hardware.execute import SafetyLimits, require_close, check_tool
        robot=FrankaRobot()
        q=robot.joints();H=robot.tool_pose()
        print('Actual seven joint angles (rad):',q.tolist())
        print('Actual frankapy tool XYZ (m):',H[:3,3].tolist())
        print('Simulated home tool XYZ (m):', [row[3] for row in doc['home_tool_pose'][:3]])
        print('Measured shelf tops (mm):',doc['shelf_heights_mm'])
        require_close(q,doc['home'],SafetyLimits.start_joint_rad,'Initial HOME')
        pos,angle=check_tool(robot,doc['home_tool_pose'],SafetyLimits(),'HOME tool frame')
        print('READ-ONLY CHECK PASSED (position error {:.4f} m; rotation error {:.4f} rad).'.format(pos,angle))
        print('This is NOT proof of physical shelf alignment or safe path clearance.')
        return 0
    if args.command=='execute':
        if not args.ack_safety:
            p.error('Physical movement locked: supervised E-STOP operator must provide --ack-safety')

        record=verify_attestation(args.validation,args.plan,doc)
        print('Validated plan SHA256:',record['plan_sha256'])
        print('Measured shelf tops (mm):',doc['shelf_heights_mm'])
        print('No Cartesian frame transform is automatically calibrated: verify robot base and tool!')
        from hardware.robot import FrankaRobot
        from hardware.execute import execute
        robot=FrankaRobot()
        selected=[args.trial] if args.trial is not None else None
        execute(doc,robot,selected_trials=selected,auto_stages=args.auto_stages,log_path=args.log)
        return 0
    return 1

if __name__=='__main__':
    raise SystemExit(main())
