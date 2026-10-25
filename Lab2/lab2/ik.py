"""Pose IK with seeded bounded search and reject-on-collision semantics."""
import numpy as np
from scipy.optimize import least_squares
from .geometry import pose_error
from .rrt import PlanFailure
from .config import HOME


class PoseIK:
    def __init__(self, sim, cfg):
        self.sim, self.cfg = sim, cfg

    def solve(self, target, nearby, carrying=False):
        sim, cfg = self.sim, self.cfg
        lo, hi = sim.limits[:, 0], sim.limits[:, 1]
        nearby = np.clip(np.asarray(nearby, float), lo + 1.e-5, hi - 1.e-5)
        seeds = [nearby, np.clip(HOME, lo, hi), (lo + hi)/2]
        # Unlike a fixed waypoint table, these starts sample diverse IK branches.
        seeds.extend(sim.rng.uniform(lo, hi) for _ in range(cfg.ik_trials))
        best_q, best_cost = None, np.inf
        for seed in seeds:
            def residual(q):
                H = sim.evaluate(q, carrying=False)
                e = pose_error(target, H)
                return np.r_[e[:3], 0.22*e[3:]]
            sol = least_squares(residual, np.clip(seed, lo+1.e-5, hi-1.e-5),
                                bounds=(lo, hi), max_nfev=105,
                                ftol=1.e-5, xtol=1.e-5, gtol=1.e-5)
            err = residual(sol.x)
            pos_err, orient_err = np.linalg.norm(err[:3]), np.linalg.norm(err[3:]) / 0.22
            if pos_err > cfg.ik_position_tolerance or orient_err > cfg.ik_orientation_tolerance:
                continue
            if not sim.collision_free(sol.x, carrying=carrying):
                continue
            score = np.linalg.norm(sol.x - nearby) + pos_err*100 + orient_err*5
            if score < best_cost:
                best_q, best_cost = sol.x.copy(), score
                # Nearby solution is preferable because short paths have fewer obstacles.
                if score < 1.7:
                    break
        if best_q is None:
            p = target[:3, 3]
            raise PlanFailure(f'No collision-free IK found for tool at {np.round(p, 3).tolist()}')
        return best_q
