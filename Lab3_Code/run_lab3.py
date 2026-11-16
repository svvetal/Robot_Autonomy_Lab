#!/usr/bin/env python3
"""Lab 3: plan and execute the block patterns in one MuJoCo session.

  python3 run_lab3.py                         # viewer, all four patterns
  python3 run_lab3.py --headless --record     # outputs/lab3_all_patterns.mp4
  python3 run_lab3.py --patterns box bow --control torque
  python3 run_lab3.py --plan-only             # symbolic plans only, no MuJoCo
"""
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path


def preplan(patterns, cfg, log=print):
    """Chain symbolic plans through all patterns before the viewer opens."""
    import time
    from lab3.layout import ALL_CELLS, INITIAL_BLOCKS
    from lab3.symbolic.domain import make_planner
    from lab3.symbolic.problem import init_from_occupancy, goal_for_pattern, apply_plan
    occ = {c: None for c in ALL_CELLS}
    for _, col, cell in INITIAL_BLOCKS:
        occ[cell] = col
    planner, plans = make_planner(), {}
    for name in patterns:
        t = time.time()
        plan = planner.plan(init_from_occupancy(occ), goal_for_pattern(name),
                            max_expansions=cfg.max_symbolic_expansions)
        plans[name] = (dict(occ), plan)
        log(f'{name:8s}: ' + ('INFEASIBLE -> will skip' if plan is None else
                              f'{len(plan)} actions') + f'  [{time.time() - t:.1f}s]')
        if plan is not None:
            for a in plan:
                log(f'           {a.label()}')
            occ = apply_plan(occ, plan)
    return plans


def execute_patterns(sim, cfg, player, plans, patterns, on_result=None, log=print,
                     home_between=True):
    """Run the patterns in one session. Used by BOTH the simulator (run_lab3)
    and the hardware-plan capture (Lab3_Hardware), so the motions are identical.

    home_between=False: no return to HOME between patterns -- the next pattern's first
    approach is planned (IK + RRT) from wherever the previous one ended; the arm goes
    HOME once, after the last pattern that has a plan."""
    from lab3.executor import TaskExecutor, PatternResult
    ex = TaskExecutor(sim, cfg, player, log=log)
    results = []
    player.hold(1.0)
    feasible = [n for n in patterns if plans[n][1] is not None]
    last_feasible = feasible[-1] if feasible else None
    for name in patterns:
        if hasattr(player, 'begin_pattern'):
            player.begin_pattern(name, ex.world)
        expected_occ, plan = plans[name]
        # Use the pre-computed plan only if the world is where we expected it.
        use = plan if ex.world.occupancy() == expected_occ else None
        if use is None and plan is not None:
            log(f'  [{name}] world differs from pre-plan -> replanning')
        if plan is None:
            r = PatternResult(name, 'infeasible', reason='no symbolic plan; pattern skipped')
            r.final_grid = ex.world.ascii()
            log(f'--- {name}: no plan exists -> SKIPPED')
        else:
            log(f'--- {name} ---')
            r = ex.run_pattern(name, plan=use)
            if home_between or name == last_feasible:
                ex.go_home()
            player.hold(1.5)                        # pause on the finished pattern
        log(f'--- {name}: {r.status.upper()} (t={sim.data.time:.1f}s)\n{r.final_grid}')
        if hasattr(player, 'end_pattern'):
            player.end_pattern(name, r, ex.world)
        results.append(r)
        if on_result is not None:
            on_result(results)
    player.hold(1.0)
    return results


def main():
    from lab3.determinism import ensure_deterministic_hashing
    ensure_deterministic_hashing()        # identical symbolic plans every run
    from lab3.layout import PATTERN_ORDER, PATTERNS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--patterns', nargs='+', default=PATTERN_ORDER, choices=list(PATTERNS))
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--control', choices=('kinematic', 'torque'), default='kinematic')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--record', action='store_true', help='write one continuous MP4')
    p.add_argument('--plan-only', action='store_true', help='print symbolic plans and exit')
    p.add_argument('--no-home-between', action='store_true',
                   help='go straight on to the next pattern (HOME only at the end)')
    p.add_argument('--out', type=Path, default=Path('outputs'))
    args = p.parse_args()

    from lab3.config import Settings
    cfg = Settings(seed=args.seed)
    print('=== Symbolic planning (HW3 A*) ===', flush=True)
    plans = preplan(args.patterns, cfg)
    if args.plan_only:
        return

    if args.headless and args.record and 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'
    from lab3.simulator import LabSim
    from lab3.playback import Player

    args.out.mkdir(parents=True, exist_ok=True)
    sim = LabSim(cfg, control=args.control)
    movie = args.out / 'lab3_all_patterns.mp4' if args.record else None
    print(f'=== Execution | control={args.control} | recording={movie} ===', flush=True)

    def save(results):
        (args.out / 'results.json').write_text(
            json.dumps([asdict(r) for r in results], indent=2, default=float))

    with Player(sim, cfg, visualize=not args.headless, record_path=movie) as player:
        execute_patterns(sim, cfg, player, plans, args.patterns, on_result=save,
                         log=lambda m: print(m, flush=True),
                         home_between=not args.no_home_between)
    print(f'Done. Log: {args.out / "results.json"}' + (f' | video: {movie}' if movie else ''))


if __name__ == '__main__':
    main()
