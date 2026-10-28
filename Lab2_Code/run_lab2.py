#!/usr/bin/env python3
"""Run the multi-trial MuJoCo manipulation task."""
import argparse
import json
import os
from pathlib import Path

def build_placements(cfg):
    """Generate placements for the requested trials."""
    from lab2.config import trial_seed, targets_for_trial
    return [
        {'trial': i + 1, 'seed': trial_seed(cfg.seed, i),
         'left': left.tolist(), 'right': right.tolist(), 'right_shelf': shelf}
        for i in range(cfg.trials)
        for left, right, shelf in [targets_for_trial(cfg, i)]
    ]

def execute_session(sim, cfg, player, placements, out):
    """Run all trials in the same MuJoCo session."""
    from lab2.config import PLANNER_SEED_OFFSET
    from lab2.task import ManipulationTask
    import numpy as np
    results = []
    for i, spec in enumerate(placements):

        if i:
            player.hold(0.75)
            sim.prepare_next_trial()
            player.frame()
            player.hold(0.65)
        sim.rng = np.random.default_rng(spec['seed'] + PLANNER_SEED_OFFSET)
        targets = (np.asarray(spec['left'], float),
                   np.asarray(spec['right'], float),
                   spec['right_shelf'])
        print(f"Trial {i+1}/{cfg.trials} | placement_seed={spec['seed']} "
              f"| left_x={spec['left'][0]:.3f}m right_x={spec['right'][0]:.3f}m "
              f"| right_shelf={spec['right_shelf']+1}", flush=True)
        try:
            result = ManipulationTask(sim, cfg, player).run_one(i, targets=targets)
            result['transition_before'] = ('initial_scene' if i == 0 else
                                           'scripted_cube_respawn_ONLY_no_scene_reset')
            results.append(result)
            print(f"  Completed {len(result['stages'])} stages; "
                  f"simulation clock={sim.data.time:.1f}s", flush=True)
        except Exception as ex:
            results.append({'trial': i + 1, 'status': 'FAILED',
                            'reason': str(ex), 'seed': spec['seed']})
            (out / 'results.json').write_text(json.dumps(results, indent=2))
            raise
        (out / 'results.json').write_text(json.dumps(results, indent=2))
    return results

def main():
    p = argparse.ArgumentParser(description='Lab 2: uninterrupted 3-trial shelf demonstration')
    p.add_argument('--trials', type=int, default=3, help='3 for lab demonstration')
    p.add_argument('--seed', type=int, default=13, help='base random seed for target positions and planner')
    p.add_argument('--headless', action='store_true', help='do not open MuJoCo viewer')
    p.add_argument('--record', action='store_true', help='one continuous MP4 of ALL trials')
    p.add_argument('--control', choices=('kinematic', 'torque'), default='kinematic')
    p.add_argument('--out', type=Path, default=Path('outputs'))
    p.add_argument('--check-scene', action='store_true',
                   help='generate calibrated scene and show actual shelf top heights + targets; no MuJoCo import')
    p.add_argument('--export-targets', type=Path, default=None,
                   help='write calibrated XYZ positions for hardware integration; DOES NOT move robot')
    args = p.parse_args()
    if args.trials < 1:
        p.error('--trials must be positive')
    from lab2.config import Settings
    from lab2.scene import scene_shelf_geometry, prepare_scene
    from lab2.shelf_heights import load_heights, heights_as_mm
    cfg = Settings(seed=args.seed, trials=args.trials)
    placements = build_placements(cfg)
    heights = load_heights()

    runtime_xml = prepare_scene()
    scene = scene_shelf_geometry(runtime_xml)
    top_geometry = {
        body: round(scene[body][0][2] + scene[body][1][2], 6)
        for body in heights
    }
    for body in heights:
        if abs(top_geometry[body] - heights[body]) > 1e-6:
            raise RuntimeError(f'Shelf {body}: rendered top and target height disagree')
    report = {
        'height_input': 'shelf_heights.json (millimetres, measured in robot-base frame)',
        'shelf_top_heights_mm': heights_as_mm(heights),
        'shelf_top_heights_m_in_runtime_xml': top_geometry,
        'runtime_scene': str(runtime_xml),
        'shelf_parts': len(scene),
        'cube_half_height_m': 0.020,
        'placement_gap_m': cfg.placement_gap,
        'target_frame': 'MuJoCo world; assumed aligned with the physical Panda base (link0)',
        'seed': args.seed,
        'placements_cube_center_xyz_m': placements,
        'hardware_ready': False,
        'warning': ('Position targets only; use hardware_main.py plan + validate to build '
                    'frankapy trajectories. Actual physical clearance and frame alignment remain unverified.')
    }
    if args.export_targets is not None:
        args.export_targets.parent.mkdir(parents=True, exist_ok=True)
        args.export_targets.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Wrote position targets (NO hardware motion): {args.export_targets}')
    if args.check_scene:
        print(json.dumps(report, indent=2))
        return
    if args.export_targets is not None:

        return
    if args.headless and args.record and 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl' if os.name == 'posix' else 'glfw'
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit('MuJoCo not installed. Run: python -m pip install -r requirements.txt') from exc
    from lab2.simulator import LabSim
    from lab2.playback import Player
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'placements.json').write_text(json.dumps(placements, indent=2))

    sim = LabSim(cfg, control=args.control)
    movie = args.out / 'all_trials_continuous.mp4' if args.record else None
    print(f'Single MuJoCo session | {args.trials} trial(s) | '
          f'seed={args.seed} | control={args.control} | recording={movie}', flush=True)
    with Player(sim, cfg, visualize=not args.headless, record_path=movie) as player:
        results = execute_session(sim, cfg, player, placements, args.out)
        player.hold(1.0)
    if movie:
        for r in results:
            r['video'] = str(movie)
    (args.out / 'results.json').write_text(json.dumps(results, indent=2))
    print(f'Finished {len(results)} trials IN ONE SESSION. Log: {args.out / "results.json"}')
    if movie:
        print(f'Continuous video: {movie}')

if __name__ == '__main__':
    main()
