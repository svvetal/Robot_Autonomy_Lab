"""Executor outcomes and trajectories must preserve the planner's contract."""
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest

import numpy as np

from lab3.config import HOME, Settings, initial_block_positions
from lab3.pick_place import PickPlaceSkill
from lab3.task_executor import TaskExecutor


class ExecutorContractTests(unittest.TestCase):
    def test_locked_goal_is_certified_and_skipped_without_motion(self):
        sim = SimpleNamespace(
            control="kinematic", attachment=None,
            block_positions=initial_block_positions, arm_q=lambda: HOME.copy(),
            data=SimpleNamespace(time=0.0),
        )
        executor = TaskExecutor(sim, Settings())
        with redirect_stdout(StringIO()):
            reports = executor.run(("axe",))
        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertEqual(report["status"], "skipped_top_grasp_infeasible")
        self.assertEqual(report["infeasibility_certificate"]["blocked_cells"],
                         ["G00", "G01", "G10", "G11"])
        self.assertEqual(report["moves_completed"], 0)
        self.assertEqual(report["reset"]["status"], "achieved")
        self.assertEqual(sim.data.time, 0.0)
        self.assertIsNone(sim.attachment)

    def test_time_parameterization_stays_on_rrt_edges_and_bounds_speed(self):
        skill = PickPlaceSkill.__new__(PickPlaceSkill)
        skill.cfg = Settings()
        a = np.zeros(7)
        b = np.array([.4, -.3, .2, -.1, 0., .05, -.4])
        c = np.array([-.2, .1, .15, .2, -.3, 0., -.1])
        samples = skill._smooth_motion_samples((a, b, c), .002)
        for q, velocity in samples:
            on_edge = False
            for start, end in ((a, b), (b, c)):
                delta = end - start
                u = float((q - start) @ delta / (delta @ delta))
                if -1e-10 <= u <= 1. + 1e-10 and np.allclose(
                    q, start + u * delta, atol=1e-10, rtol=0.,
                ):
                    on_edge = True
                    break
            self.assertTrue(on_edge)
            self.assertLessEqual(float(np.abs(velocity).max()),
                                 skill.cfg.joint_speed + 1e-8)
        np.testing.assert_allclose(samples[-1][0], c)
        np.testing.assert_allclose(samples[-1][1], np.zeros(7), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
