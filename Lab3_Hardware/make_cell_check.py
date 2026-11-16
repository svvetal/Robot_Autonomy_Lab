#!/usr/bin/env python3
"""Build + validate outputs/cell_check_plan.json (MuJoCo PC) for cell_check.py (robot PC).

For every table cell (9 grid + 8 storage) it reuses the main plan's validated
descent into that cell (a pick descent if there is one, else a place descent),
and chains the cells with direct joint moves between their hover poses:

  HOME -> hover(c1) -> descend/ascend c1 -> hover(c2) -> ... -> hover(cN) -> HOME

Collision checks (0.01 rad, same checker as validate):
  * hover-to-hover and HOME moves: with a cube on ALL 17 cells (conservative);
    a move that fails is replaced by hover -> HOME -> hover.
  * each descent, dense path AND the joint chords actually executed: open
    fingers (6 cm) and ONE cube on the target cell only.

  python3 make_cell_check.py            # after `hardware_main.py plan` + `validate`
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from hardware_main import PLAN, VALID, OUT, verify_attestation   # noqa: E402

CHECK = OUT / 'cell_check_plan.json'
ORDER = ['sr0', 'sr1', 'sr2', 'sr3',
         'g00', 'g01', 'g02', 'g12', 'g11', 'g10', 'g20', 'g21', 'g22',
         'sb3', 'sb2', 'sb1', 'sb0']
TRANSIT_SPEED = 0.35          # rad/s peak (min-jerk), well below the plan's 0.65
TRANSIT_MIN_S = 3.0


def transit_duration(a, b):
    return max(TRANSIT_MIN_S, 1.875 * float(np.max(np.abs(np.subtract(b, a)))) / TRANSIT_SPEED)


def main():
    from lab3_hw.spec import load_plan, sha256_file, atomic_json, HOME, COLLISION_RESOLUTION_RAD, \
        segment_indices, MAX_STRAIGHT_DRIFT_M
    from lab3_hw.simlink import use_sim, sim_sources_sha256
    doc = load_plan(PLAN)
    verify_attestation(VALID, PLAN, doc)
    use_sim()
    from lab3.config import Settings
    from lab3.rrt import interpolate_edge
    from lab3.simulator import LabSim
    from lab3_hw.capture import chord_drift
    if doc['sim_sources_sha256'] != sim_sources_sha256():
        raise SystemExit('Lab3_Code changed since the plan was made: re-plan first')

    # ---- one validated descent per cell, pick preferred ----------------------------
    found = {}
    for pat in doc['patterns']:
        for ev in pat['events']:
            if ev.get('smooth') and ev['name'].endswith('_descend'):
                kind, cell = ev['name'].split('_')[:2]
                if cell not in found or (found[cell][0] == 'place' and kind == 'pick'):
                    found[cell] = (kind, pat['name'], ev)
    missing = [c for c in ORDER if c not in found]
    if missing or set(ORDER) != set(doc['cells']):
        raise SystemExit('No planned descent for cells {}'.format(missing))

    cells_xyz = doc['cells']
    allcubes = [('c_' + c, 'blue', c) for c in ORDER]
    sim_all = LabSim(Settings(seed=doc['seed']), blocks=allcubes)

    def edge_free(sim, a, b, label):
        n = 0
        for q in interpolate_edge(a, b, COLLISION_RESOLUTION_RAD):
            issues = sim.collision_reasons(np.asarray(q, float))
            if issues:
                return False, '{}: {}'.format(label, issues[:2])
            n += 1
        return True, n

    checked = 0
    entries = []
    for cell in ORDER:
        kind, pat_name, ev = found[cell]
        path = [np.asarray(q, float) for q in ev['path']]
        sim1 = LabSim(Settings(seed=doc['seed']), blocks=[('cube', 'blue', cell)])
        segs = ev.get('segments', 3)
        for n in sorted({segs, 3}):
            if chord_drift(sim1.fk, path, n) > MAX_STRAIGHT_DRIFT_M + 1e-6:
                raise SystemExit('{}: {} chord(s) drift above limit'.format(cell, n))
        idx = sorted(set(segment_indices(len(path), segs)) | set(segment_indices(len(path), 3)))
        for a, b in list(zip(path, path[1:])) + [(path[i], path[j]) for n in sorted({segs, 3})
                                                   for i, j in zip(segment_indices(len(path), n),
                                                                   segment_indices(len(path), n)[1:])]:
            ok, res = edge_free(sim1, a, b, cell + ' descent')
            if not ok:
                raise SystemExit('Descent collides with open fingers: ' + res)
            checked += res
        top, bottom = sim1.fk(path[0]), sim1.fk(path[-1])
        entries.append(dict(cell=cell, xyz=cells_xyz[cell], source='{} {} ({})'.format(
            pat_name, ev['name'], kind), carrying_in_plan=bool(ev['carrying']),
            hover_q=path[0].tolist(), path=[q.tolist() for q in path],
            durations=list(ev['durations']), segments=segs,
            chord_drift_mm=ev.get('chord_drift_mm'),
            hover_tool_pose=top.tolist(), grasp_tool_pose=bottom.tolist()))
        print('  {:4s} descent from {:24s} {} segment(s), bottom tool z {:.4f}'.format(
            cell, ev['name'] + ' (' + pat_name + ')', segs, bottom[2, 3]))

    # ---- transits at hover height, all 17 cells occupied ---------------------------
    stops = [('HOME', np.asarray(HOME, float))] + [(e['cell'], np.asarray(e['hover_q'])) for e in entries] \
        + [('HOME', np.asarray(HOME, float))]
    transits = []
    for (na, a), (nb, b) in zip(stops, stops[1:]):
        ok, res = edge_free(sim_all, a, b, '{}->{}'.format(na, nb))
        if ok:
            checked += res
            transits.append(dict(name='{}->{}'.format(na, nb), path=[a.tolist(), b.tolist()],
                                 durations=[transit_duration(a, b)]))
            continue
        legs = [(a, np.asarray(HOME, float)), (np.asarray(HOME, float), b)]
        path, durs = [a.tolist()], []
        for x, y in legs:
            ok2, res2 = edge_free(sim_all, x, y, '{}->HOME->{}'.format(na, nb))
            if not ok2:
                raise SystemExit('No collision-free transit: ' + res2)
            checked += res2
            path.append(y.tolist())
            durs.append(transit_duration(x, y))
        transits.append(dict(name='{}->HOME->{}'.format(na, nb), path=path, durations=durs))
        print('  transit {}->{} goes via HOME'.format(na, nb))

    # HOME <-> every hover, so a --cells subset can always chain through HOME.
    home = np.asarray(HOME, float)
    have = {t['name'] for t in transits}
    for e in entries:
        h = np.asarray(e['hover_q'], float)
        for na, a, nb, b in (('HOME', home, e['cell'], h), (e['cell'], h, 'HOME', home)):
            if '{}->{}'.format(na, nb) in have:
                continue
            ok, res = edge_free(sim_all, a, b, '{}->{}'.format(na, nb))
            if not ok:
                raise SystemExit('HOME leg collides: ' + res)
            checked += res
            transits.append(dict(name='{}->{}'.format(na, nb), path=[a.tolist(), b.tolist()],
                                 durations=[transit_duration(a, b)]))

    out = dict(schema='lab3-cell-check-1', plan_sha256=sha256_file(PLAN), seed=doc['seed'],
               layout=doc['layout'], home=list(map(float, HOME)), home_tool_pose=doc['home_tool_pose'],
               gripper=doc['gripper'], order=ORDER, cells=entries, transits=transits,
               validation=dict(result='cell_check_collision_validation_passed',
                               joint_states_checked=int(checked),
                               conditions='transits: cubes on all 17 cells; descents: open '
                                          'fingers, one cube on the target cell only'))
    atomic_json(CHECK, out)
    total = sum(sum(t['durations']) for t in transits) + 2 * sum(sum(e['durations']) for e in entries)
    print('PASS: {} joint states collision-checked. Wrote {} ({} cells, ~{:.0f} s of motion at '
          'time-scale 1).'.format(checked, CHECK, len(entries), total))


if __name__ == '__main__':
    main()
