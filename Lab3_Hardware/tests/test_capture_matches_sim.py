"""The saved hardware plan is exactly what run_lab3.py executes in simulation.

Runs Lab3_Code's simulation (real Player, no viewer) with the plan's seed and
compares every move path and grip against outputs/hardware_plan.json.
Run:  python3 -m tests.test_capture_matches_sim     (~2 min)
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab3_hw.simlink import use_sim                       # noqa: E402
from lab3_hw.spec import load_plan                        # noqa: E402

PLAN = Path(__file__).resolve().parents[1] / 'outputs' / 'hardware_plan.json'


def main():
    use_sim()
    from lab3.determinism import ensure_deterministic_hashing
    ensure_deterministic_hashing()
    import run_lab3
    from lab3.config import Settings
    from lab3.playback import Player
    from lab3.simulator import LabSim
    doc = load_plan(PLAN)
    names = [p['name'] for p in doc['patterns']]
    cfg = Settings(seed=doc['seed'])
    plans = run_lab3.preplan(names, cfg, log=lambda *a: None)
    sim = LabSim(cfg)
    seen = []

    class Recording(Player):
        def move(self, motion):
            seen.append(('move', [np.asarray(q) for q in motion.path]))
            Player.move(self, motion)

        def on_grip(self, step, plan, block):
            seen.append(('grip', 'close' if step.close else 'open', block, plan.cell))

    with Recording(sim, cfg) as player:
        run_lab3.execute_patterns(sim, cfg, player, plans, names, log=lambda *a: None)
    saved = []
    for p in doc['patterns']:
        for ev in p['events']:
            if ev['type'] == 'move':
                saved.append(('move', [np.asarray(q) for q in ev['path']]))
            else:
                saved.append(('grip', ev['action'], ev['block'], ev['cell']))
    assert len(seen) == len(saved), (len(seen), len(saved))
    for a, b in zip(seen, saved):
        assert a[0] == b[0]
        if a[0] == 'grip':
            assert a == b, (a, b)
        else:
            assert len(a[1]) == len(b[1]) and all(np.allclose(x, y, atol=1e-12)
                                                  for x, y in zip(a[1], b[1]))
    print('OK: {} events of the simulation run are identical to the hardware plan'.format(len(seen)))


if __name__ == '__main__':
    main()
