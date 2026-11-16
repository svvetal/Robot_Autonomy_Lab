"""Build the hardware plan by running the EXACT simulation task (MuJoCo, no robot).

The capture uses Lab3_Code's run_lab3.execute_patterns -- the same function the
simulator runs -- with the same seed, so the symbolic plans, grasp axes,
RRT-Connect paths and straight-line descents are identical to the simulation
you watched. Instead of rendering, CapturePlayer records every move and grip.
"""
from datetime import datetime, timezone

import numpy as np

from .simlink import use_sim, sim_sources_sha256
from .spec import (SCHEMA, HOME, CUBE_WIDTH, MAX_SEGMENTS, MAX_STRAIGHT_DRIFT_M, validate_plan,
                   segment_indices)


def chord_drift(fk, path, segments, samples=41):
    """Max distance (m) of the tool from the straight line start->end when the path is
    played as `segments` straight joint-space chords."""
    Q = np.asarray(path, float)
    P0, P1 = fk(Q[0])[:3, 3], fk(Q[-1])[:3, 3]
    u = (P1 - P0) / max(np.linalg.norm(P1 - P0), 1e-12)
    idx = segment_indices(len(Q), segments)
    worst = 0.0
    for a, b in zip(idx, idx[1:]):
        for s in np.linspace(0.0, 1.0, samples):
            d = fk(Q[a] + s * (Q[b] - Q[a]))[:3, 3] - P0
            worst = max(worst, float(np.linalg.norm(d - np.dot(d, u) * u)))
    return worst


def choose_segments(fk, path):
    for n in range(1, MAX_SEGMENTS + 1):
        d = chord_drift(fk, path, n)
        if d <= MAX_STRAIGHT_DRIFT_M:
            return n, d
    raise RuntimeError('straight move drifts {:.1f} mm even with {} chords'.format(d * 1e3, n))


class CapturePlayer(object):
    def __init__(self, sim):
        self.sim = sim
        self.patterns = []
        self.current = None

    # Player interface used by lab3 ------------------------------------------
    def frame(self):
        pass

    def hold(self, seconds):
        pass                              # kinematic: nothing moves while holding

    def move(self, motion):
        sim = self.sim
        if self.current is None:
            raise RuntimeError('move outside a pattern')
        if np.max(np.abs(sim.arm_q() - np.asarray(motion.path[0]))) > 1e-3:
            raise RuntimeError('Trajectory discontinuity at ' + motion.name)
        carrying = sim.held is not None
        extra = {}
        if getattr(motion, 'smooth', False):
            n, d = choose_segments(sim.fk, motion.path)
            extra = dict(segments=n, chord_drift_mm=round(d * 1e3, 3))
        sim.advance(np.asarray(motion.path[-1], float), np.zeros(7), 0.002)
        self.current['events'].append(dict(**extra, **dict(
            type='move', name=motion.name, carrying=carrying,
            smooth=bool(getattr(motion, 'smooth', False)),
            path=[np.asarray(q, float).tolist() for q in motion.path],
            durations=[float(t) for t in motion.durations],
            goal_tool_pose=sim.tool_pose().tolist())))

    def on_grip(self, step, plan, block):
        from lab3.layout import cell_xyz
        ev = dict(type='grip', action='close' if step.close else 'open',
                  block=block, cell=plan.cell)
        if not step.close:
            ev['xyz'] = cell_xyz(plan.cell).tolist()
        self.current['events'].append(ev)

    # Pattern bookkeeping ------------------------------------------------------
    def begin_pattern(self, name, world):
        self.current = dict(name=name, events=[], start_occupancy=dict(world.at))
        self.patterns.append(self.current)

    def end_pattern(self, name, result, world):
        self.current.update(status=result.status, symbolic_actions=result.symbolic_actions,
                            executed=result.executed, end_occupancy=dict(world.at),
                            final_grid=result.final_grid)
        self.current = None


def build_plan(seed=13, patterns=None, log=print, home_between=True):
    use_sim()
    import os
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise RuntimeError('plan capture must run with PYTHONHASHSEED=0 (hardware_main.py '
                           'sets it) so the symbolic plans match run_lab3.py')
    import run_lab3
    from lab3.config import Settings
    from lab3.layout import ALL_CELLS, INITIAL_BLOCKS, PATTERN_ORDER, cell_xyz, layout_summary
    from lab3.simulator import LabSim

    patterns = list(patterns or PATTERN_ORDER)
    cfg = Settings(seed=int(seed))
    log('Symbolic planning (HW3 A*) ...')
    plans = run_lab3.preplan(patterns, cfg, log=log)
    sim = LabSim(cfg, control='kinematic')
    home_tool_pose = sim.tool_pose().tolist()
    player = CapturePlayer(sim)
    log('Motion planning (IK + RRT-Connect), same seed as the simulator ...')
    results = run_lab3.execute_patterns(sim, cfg, player, plans, patterns, log=log,
                                        home_between=home_between)
    bad = [r.pattern for r in results if r.status not in ('done', 'infeasible')]
    if bad:
        raise RuntimeError('Simulation failed for {}; not producing a hardware plan'.format(bad))
    if np.max(np.abs(sim.arm_q() - HOME)) > 1e-3:
        raise RuntimeError('Simulation did not finish at HOME')

    initial = {c: None for c in ALL_CELLS}
    for name, _col, cell in INITIAL_BLOCKS:
        initial[cell] = name
    doc = dict(
        home_between=bool(home_between),
        schema=SCHEMA, seed=int(seed), created_utc=datetime.now(timezone.utc).isoformat(),
        source='Lab3_Code run_lab3.execute_patterns (kinematic MuJoCo), seed {}'.format(seed),
        sim_sources_sha256=sim_sources_sha256(), layout=layout_summary(),
        cells={c: cell_xyz(c).tolist() for c in ALL_CELLS},
        home=HOME.tolist(), home_tool_pose=home_tool_pose,
        gripper=dict(pregrasp_width_m=2 * 0.030, cube_width_m=CUBE_WIDTH),
        settings=dict(hover_z=cfg.hover_z, placement_gap=cfg.placement_gap,
                      joint_speed=cfg.joint_speed),
        initial_occupancy=initial,
        block_colour={name: col for name, col, _ in INITIAL_BLOCKS},
        patterns=player.patterns,
    )
    validate_plan(doc)
    return doc
