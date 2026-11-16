"""Independent MuJoCo replay of a saved plan (no robot).

Re-checks every joint configuration along every saved path at 0.01 rad
resolution against the table, shelves and all cubes (with the held cube),
re-executes every grip, and confirms each pattern ends with the cubes exactly
where the plan says. Also rejects plans made with different code or layout.
"""
import numpy as np

from .simlink import use_sim, sim_sources_sha256
from .spec import (validate_plan, COLLISION_RESOLUTION_RAD, FRANKAPY_RESET, HOME,
                   FAILSAFE_SEGMENTS, MAX_STRAIGHT_DRIFT_M, segment_indices)


def validate_collisions(doc, log=print):
    validate_plan(doc)
    use_sim()
    from lab3.config import Settings
    from lab3.layout import layout_summary, cell_xyz
    from lab3.rrt import interpolate_edge
    from lab3.simulator import LabSim

    if doc['sim_sources_sha256'] != sim_sources_sha256():
        raise ValueError('Lab3_Code (code, table_layout.json or shelf heights) changed since '
                         'this plan was made: run hardware_main.py plan again')
    if doc['layout'] != layout_summary():
        raise ValueError('Table layout differs from the plan: re-plan')
    for c, xyz in doc['cells'].items():
        if not np.allclose(xyz, cell_xyz(c), atol=1e-9):
            raise ValueError('Cell {} moved since planning: re-plan'.format(c))

    sim = LabSim(Settings(seed=doc['seed']), control='kinematic')
    checked = 0
    # The `home` command: straight joint move frankapy reset pose <-> plan HOME.
    for q in interpolate_edge(FRANKAPY_RESET, HOME, COLLISION_RESOLUTION_RAD):
        issues = sim.collision_reasons(q)
        if issues:
            raise RuntimeError('reset->HOME move collides: {}'.format(issues[:2]))
        checked += 1
    log('  reset->HOME move OK')
    for pat in doc['patterns']:
        for ev in pat['events']:
            if ev['type'] == 'grip':
                if ev['action'] == 'close':
                    sim.attach(ev['block'])
                else:
                    sim.release(np.asarray(ev['xyz'], float))
                continue
            if np.max(np.abs(sim.arm_q() - ev['path'][0])) > 1e-3:
                raise RuntimeError('{} / {}: discontinuity'.format(pat['name'], ev['name']))
            for a, b in zip(ev['path'], ev['path'][1:]):
                for q in interpolate_edge(a, b, COLLISION_RESOLUTION_RAD):
                    issues = sim.collision_reasons(q)
                    if issues:
                        raise RuntimeError('Collision in {} / {}: {}'.format(
                            pat['name'], ev['name'], issues[:2]))
                    checked += 1
            if ev.get('smooth'):
                # What waypoints mode really executes: straight joint chords between a few
                # path points. Check the stored count AND the fail-safe count.
                from .capture import chord_drift
                for n in sorted({ev.get('segments', FAILSAFE_SEGMENTS), FAILSAFE_SEGMENTS}):
                    d = chord_drift(sim.fk, ev['path'], n)
                    if d > MAX_STRAIGHT_DRIFT_M + 1e-6:
                        raise RuntimeError('{} / {}: {} chord(s) drift {:.2f} mm > {:.1f} mm'.format(
                            pat['name'], ev['name'], n, d * 1e3, MAX_STRAIGHT_DRIFT_M * 1e3))
                    idx = segment_indices(len(ev['path']), n)
                    for i, j in zip(idx, idx[1:]):
                        for q in interpolate_edge(ev['path'][i], ev['path'][j],
                                                  COLLISION_RESOLUTION_RAD):
                            issues = sim.collision_reasons(q)
                            if issues:
                                raise RuntimeError('Chord collision in {} / {} ({} chords): {}'
                                                   .format(pat['name'], ev['name'], n, issues[:2]))
                            checked += 1
            sim.advance(np.asarray(ev['path'][-1], float), np.zeros(7), 0.002)
            H = sim.tool_pose()
            if np.linalg.norm(H[:3, 3] - np.asarray(ev['goal_tool_pose'])[:3, 3]) > 1e-4:
                raise RuntimeError('{} / {}: stored tool pose disagrees with FK'.format(
                    pat['name'], ev['name']))
        for cell, block in pat['end_occupancy'].items():
            if block is not None:
                err = np.linalg.norm(sim.block_pose(block)[:2, 3] - cell_xyz(cell)[:2])
                if err > 1e-3:
                    raise RuntimeError('{}: {} ends {:.1f} mm from {}'.format(
                        pat['name'], block, err * 1000, cell))
        log('  {:8s} {:10s} OK ({} events)'.format(pat['name'], pat['status'], len(pat['events'])))
    return checked
