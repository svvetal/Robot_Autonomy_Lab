#!/usr/bin/env python3
"""MuJoCo-independent structure/sample/FK screening.

Does NOT validate physics, collisions, grips, or a completed Lab2 execution.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from lab2.config import Settings, TABLE_BLOCK_POS, targets_for_trial, TOP_R, SIDE_POS_Y_R, SIDE_NEG_Y_R
from lab2.scene import prepare_scene, scene_shelf_geometry
from lab2.offline_fk import XmlKinematics


def run(seed=13, trials=3, full=True):
    robot = XmlKinematics()
    cfg = Settings(seed=seed, trials=trials)
    samples = []
    for i in range(trials):
        # Same exact targets as the one-session CLI.
        left, right, shelf = targets_for_trial(cfg, i)
        cases = [('table_top', TABLE_BLOCK_POS, TOP_R),
                 ('left_top', left, TOP_R),
                 ('left_side', left, SIDE_POS_Y_R),
                 ('right_side', right, SIDE_NEG_Y_R)]
        if full:
            cases.extend([('table_hover', TABLE_BLOCK_POS+[0,0,cfg.hover_z], TOP_R),
                          ('left_hover', left+[0,0,cfg.hover_z], TOP_R),
                          ('side_approach', left+[0,-.13,0], SIDE_POS_Y_R),
                          ('right_approach', right+[0,+.14,0], SIDE_NEG_Y_R)])
        checks = []
        for k, (name, pos, R) in enumerate(cases):
            feasible, err = robot.pose_reachable(pos, R, seed=seed+i*101+k, attempts=10)
            checks.append({'target': name, 'reachable_in_xml_FK': feasible,
                           'fk_error_position_m_and_orientation_rad': err})
        samples.append({'trial': i+1, 'left_xyz': left.round(5).tolist(),
                        'right_xyz': right.round(5).tolist(), 'right_shelf': shelf+1,
                        'kinematic_targets': checks})
    original = scene_shelf_geometry()
    generated = prepare_scene()
    assert original == scene_shelf_geometry(generated), 'Shelf geometry was changed'
    return {'type': 'static_offline_preflight_NOT_a_MuJoCo_execution',
            'unaltered_obstacle_count': len(original),
            'trial_count': trials,
            'seed': seed,
            'trials': samples,
            'all_tested_poses_kinematically_reachable': all(c['reachable_in_xml_FK']
                                    for s in samples for c in s['kinematic_targets'])}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--quick', action='store_true', help='only 4 key grasp/placement poses per trial')
    p.add_argument('--output', type=Path, default=Path('outputs/offline_preflight.json'))
    args = p.parse_args()
    report = run(args.seed, args.trials, not args.quick)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'report': str(args.output),
                      'all_sampled_poses_reachable': report['all_tested_poses_kinematically_reachable'],
                      'note': report['type']}, indent=2))
    if not report['all_tested_poses_kinematically_reachable']:
        raise SystemExit(2)
