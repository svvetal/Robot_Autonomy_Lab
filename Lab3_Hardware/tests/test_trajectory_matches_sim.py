"""The q(t) streamed to the robot equals what the simulator plays.

Plays every move of the saved plan through lab3.playback.Player (the code that
made your simulation video), records each commanded configuration and its
time, and compares with lab3_hw.trajectory.q_at at the same time.
Run:  python3 -m tests.test_trajectory_matches_sim
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab3_hw.simlink import use_sim                       # noqa: E402
from lab3_hw.spec import load_plan                        # noqa: E402
from lab3_hw.trajectory import q_at                       # noqa: E402

PLAN = Path(__file__).resolve().parents[1] / 'outputs' / 'hardware_plan.json'


def main():
    use_sim()
    from lab3.config import Settings
    from lab3.planning import Move
    from lab3.playback import Player
    from lab3.simulator import LabSim
    doc = load_plan(PLAN)
    sim = LabSim(Settings())
    player = Player(sim, sim.cfg)
    player.hold = lambda s: None                         # only the move itself
    worst, n_moves, n_samples = 0.0, 0, 0
    for pat in doc['patterns']:
        for ev in pat['events']:
            if ev['type'] != 'move':
                continue
            rec = []
            orig = sim.advance
            sim.set_arm(sim.data, ev['path'][0])
            sim.advance = lambda q, qd, dt: rec.append(np.array(q, float))
            player.move(Move(ev['name'], [np.array(q) for q in ev['path']], ev['durations'],
                             smooth=ev['smooth']))
            sim.advance = orig
            # Reconstruct the time of every sample exactly as Player does.
            times = []
            dt = float(sim.model.opt.timestep)
            if ev['smooth']:
                T = ev['durations'][0]
                n = max(2, int(np.ceil(T / dt)))
                times = [i / n * T for i in range(1, n + 1)]
            else:
                t0 = 0.0
                for d in ev['durations']:
                    n = max(2, int(np.ceil(d / dt)))
                    times += [t0 + k / n * d for k in range(1, n + 1)]
                    t0 += d
            assert len(times) == len(rec), ev['name']
            err = max(np.max(np.abs(q_at(ev, t) - q)) for t, q in zip(times, rec))
            worst = max(worst, err)
            n_moves += 1
            n_samples += len(rec)
    print('{} moves, {} samples: max |hardware q(t) - simulator q(t)| = {:.2e} rad'.format(
        n_moves, n_samples, worst))
    assert worst < 1e-9
    print('OK: hardware trajectories are identical to the simulator playback')


if __name__ == '__main__':
    main()
