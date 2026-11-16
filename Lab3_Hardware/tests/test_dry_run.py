"""Run the HARDWARE executor end to end against MuJoCo (SimRobot).

Exercises the exact code path used on the robot (plan + attestation loading,
HOME / tool-frame checks, streaming, gripper logic, per-move joint and pose
checks), then verifies every cube ends in the right cell.
Run:  python3 -m tests.test_dry_run
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab3_hw.execute import execute                       # noqa: E402
from lab3_hw.simrobot import SimRobot                     # noqa: E402
from lab3_hw.spec import load_plan                        # noqa: E402

PLAN = ROOT / 'outputs' / 'hardware_plan.json'


def check_final(robot, pat, cells):
    worst = 0.0
    for cell, block in pat['end_occupancy'].items():
        if block is not None:
            err = np.linalg.norm(robot.sim.block_pose(block)[:2, 3] - np.asarray(cells[cell])[:2])
            worst = max(worst, err)
    return worst


def main():
    doc = load_plan(PLAN)
    log = ROOT / 'outputs' / 'dry_run_log.jsonl'
    quiet = lambda *a: None
    for motion, kw in (('stream', {}), ('waypoints', {'only': 'box'})):
        robot = SimRobot(seed=doc['seed'])
        names = [p['name'] for p in doc['patterns']]
        first = doc['patterns'][names.index(kw['only'])] if kw else doc['patterns'][0]
        robot.arrange(first['start_occupancy'], doc['cells'])
        execute(doc, robot, auto_stages=True, time_scale=1.0, motion=motion,
                confirm=lambda m: None, log_path=log, out=quiet, **kw)
        last = doc['patterns'][names.index(kw['only'])] if kw else doc['patterns'][-1]
        err = check_final(robot, last, doc['cells'])
        grasps = sum(1 for c in robot.commands if c[0] == 'grasp')
        print('{:9s}: {} grasps, cubes end within {:.2f} mm of their cells'.format(
            motion, grasps, err * 1000))
        assert err < 0.002
    print('OK: hardware executor reproduces the patterns in MuJoCo')


if __name__ == '__main__':
    main()
