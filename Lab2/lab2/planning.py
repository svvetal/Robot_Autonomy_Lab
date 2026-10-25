"""High-level joint motion planner, lazy RRT, and recordable action graph."""
from dataclasses import dataclass
import numpy as np
from .rrt import JointRRT, PlanFailure
from .trajectory import duration_for_edge


@dataclass
class Move:
    name: str
    path: list
    durations: list


@dataclass
class Grip:
    name: str
    close: bool
    block_target: object = None


class JointPlanner:
    def __init__(self, sim, cfg):
        self.sim, self.cfg = sim, cfg
        self.cache = {}

    def plan(self, start, goal, carrying):
        sim, cfg = self.sim, self.cfg
        planner = JointRRT(lambda q: sim.collision_free(q, carrying), sim.limits,
                           sim.rng, step=cfg.rrt_step,
                           resolution=cfg.rrt_edge_resolution,
                           max_iter=cfg.rrt_iterations, goal_bias=cfg.rrt_goal_bias)
        try:
            path = planner.plan(start, goal)
        except PlanFailure as exc:
            # MuJoCo contact geom diagnostics make follow-on issues actionable.
            endpoint = 'start' if 'start is in collision' in str(exc) else (
                'goal' if 'goal is in collision' in str(exc) else None)
            if endpoint and hasattr(sim, 'collision_reasons'):
                q = start if endpoint == 'start' else goal
                details = sim.collision_reasons(q, carrying=carrying)
                raise PlanFailure(f'{exc} (carrying={carrying}); contacts: '
                                  f'{"; ".join(details) or "no contact found"}') from exc
            raise
        return planner.shortcut(path, cfg.rrt_shortcuts)

    def build_move(self, name, start, goal, carrying):
        path = self.plan(start, goal, carrying)
        durations = [duration_for_edge(a, b, self.cfg.joint_speed,
                                       self.cfg.segment_min_seconds)
                     for a, b in zip(path, path[1:])]
        return Move(name=name, path=path, durations=durations)
