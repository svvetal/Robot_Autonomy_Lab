#!/usr/bin/env python3
"""Lab 2: all trials run in ONE live MuJoCo simulator and ONE viewer."""
import argparse
import json
import os
from pathlib import Path


def build_placements(cfg):
    """Deterministic trial targets, independent of IK/RRT sample counts."""
    from lab2.config import trial_seed, targets_for_trial
    return [
        {'trial': i + 1, 'seed': trial_seed(cfg.seed, i),
         'left': left.tolist(), 'right': right.tolist(), 'right_shelf': shelf}
        for i in range(cfg.trials)
        for left, right, shelf in [targets_for_trial(cfg, i)]
    ]


def execute_session(sim, cfg, player, placements, out):
    """One scene, one continuous clock, one player and no MjData reset."""
    from lab2.config import PLANNER_SEED_OFFSET
    from lab2.task import ManipulationTask
    import numpy as np
    results = []
    for i, spec in enumerate(placements):
        # Next trial: don't destroy or recreate MuJoCo or the viewer. Only
        # the single block is returned to the table as an explicit scripted
        # setup action (one-block assignment environment).
        if i:
            player.hold(0.75)  # make each trial boundary visible in recording
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
                   help='print trial targets + static geometry; no MuJoCo import')
    args = p.parse_args()
    if args.trials < 1:
        p.error('--trials must be positive')
    from lab2.config import Settings
    from lab2.scene import scene_shelf_geometry
    cfg = Settings(seed=args.seed, trials=args.trials)
    placements = build_placements(cfg)
    if args.check_scene:
        print(json.dumps({'shelf_parts': len(scene_shelf_geometry()),
                          'seed': args.seed, 'placements': placements}, indent=2))
        return
    if args.headless and args.record and 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl' if os.name == 'posix' else 'glfw'
    try:
        import mujoco  # noqa: F401
    except ImportError as exc:
        raise SystemExit('MuJoCo not installed. Run: python -m pip install -r requirements.txt') from exc
    from lab2.simulator import LabSim
    from lab2.playback import Player
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'placements.json').write_text(json.dumps(placements, indent=2))
    # Only ONE MuJoCo instance, ONE Player/viewer, ONE output video. No reset
    # inside the for loop. The simulation clock increases across all trials.
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
