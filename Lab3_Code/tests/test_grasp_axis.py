"""Motion-level check of the symbolic finger rule.
Cubes at g10 and g12 flank g11 along world Y: an axis-'y' grasp of g11 must
collide, an axis-'x' grasp must be collision-free.
Run from Lab3_Code:  python3 -m tests.test_grasp_axis"""
import numpy as np
from scipy.optimize import least_squares
from lab3.config import Settings, GRASP_YAW, top_grasp_R
from lab3.geometry import make_transform, pose_error
from lab3.ik import PoseIK
from lab3.layout import cell_xyz
from lab3.rrt import PlanFailure
from lab3.simulator import LabSim

def main():
    blocks = [('red0', 'red', 'g11'), ('blue0', 'blue', 'g10'), ('blue1', 'blue', 'g12')]
    sim = LabSim(Settings(), blocks=blocks)
    ik = PoseIK(sim, sim.cfg)
    p = cell_xyz('g11')
    for axis, expect_free in (('y', False), ('x', True)):
        R = top_grasp_R(GRASP_YAW[axis])
        # Reach the pose ignoring collisions, then ask the collision checker.
        q = ik.solve(make_transform(p + [0, 0, 0.1], R), sim.arm_q())

        target = make_transform(p, R)
        sol = least_squares(lambda x: pose_error(target, sim.fk(x)) * [1, 1, 1, .3, .3, .3],
                            q, bounds=(sim.limits[:, 0], sim.limits[:, 1]))
        reasons = sim.collision_reasons(sol.x)
        print(f'axis {axis}: pose err {np.linalg.norm(pose_error(target, sim.fk(sol.x))[:3])*1000:.1f} mm,'
              f' collisions: {reasons[:2] or "none"}')
        assert (not reasons) == expect_free, axis
        try:
            ik.solve(target, sol.x)
            got = True
        except PlanFailure:
            got = False
        assert got == expect_free, f'collision-aware IK axis {axis}'
    print('OK: finger-collision rule confirmed at motion level')

if __name__ == '__main__':
    main()
