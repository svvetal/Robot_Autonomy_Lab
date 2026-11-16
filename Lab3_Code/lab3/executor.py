"""Task-level executor: HW3 symbolic plans -> motion primitives in MuJoCo.

For each pattern:
  1. occupancy (world model) -> A* symbolic plan (HW3 planner)
  2. no plan  -> pattern is infeasible: log and skip
  3. for each pick/place action: plan the primitive with the planner's axis;
     if the motion fails, try the other axis when it is symbolically legal;
     if a place still fails, park the cube in a free storage slot and replan.
"""
import time
from dataclasses import dataclass, field

from .config import HOME
from .layout import STORAGE, finger_neighbours
from .primitives import Primitives
from .rrt import PlanFailure
from .symbolic.domain import make_planner
from .symbolic.problem import init_from_occupancy, goal_for_pattern
from .world import WorldModel

MAX_REPLANS = 3


@dataclass
class PatternResult:
    pattern: str
    status: str                                   # done | infeasible | failed
    symbolic_actions: list = field(default_factory=list)
    executed: list = field(default_factory=list)
    replans: int = 0
    plan_seconds: float = 0.0
    sim_time_end: float = 0.0
    final_grid: str = ''
    drift: list = field(default_factory=list)
    reason: str = ''


class TaskExecutor:
    def __init__(self, sim, cfg, player, world=None, log=print):
        self.sim, self.cfg, self.player = sim, cfg, player
        self.world = world or WorldModel(sim.initial_blocks)
        self.prims = Primitives(sim, cfg)
        self.symbolic = make_planner()
        self.log = log

    # ------------------------------------------------------------ symbolic
    def symbolic_plan(self, pattern):
        t = time.time()
        plan = self.symbolic.plan(init_from_occupancy(self.world.occupancy()),
                                  goal_for_pattern(pattern),
                                  max_expansions=self.cfg.max_symbolic_expansions)
        return plan, time.time() - t

    def _axis_options(self, cell, preferred):
        """Preferred axis first, then the other one if its finger cells are empty."""
        occ = self.world.occupancy()
        legal = [ax for ax, (a, b) in finger_neighbours(cell).items()
                 if occ.get(a) is None and occ.get(b) is None]
        return [preferred] + [ax for ax in legal if ax != preferred]

    # ------------------------------------------------------------ actions
    def _do_pick(self, cell, colour, axis, result):
        block = self.world.at[cell]
        if block is None or self.world.colour[block] != colour:
            raise RuntimeError(f'world model has no {colour} cube at {cell}')
        last = None
        for ax in self._axis_options(cell, axis):
            try:
                plan = self.prims.plan_pick(block, cell, ax)
            except PlanFailure as exc:
                last = exc
                self.log(f'      pick {cell} axis {ax} failed: {exc}')
                continue
            self.prims.execute(plan, self.player)
            self.world.pick(cell, colour)
            result.executed.append(f'pick({block}, {cell}, axis={ax})')
            return
        raise PlanFailure(f'pick {colour}@{cell} impossible on every axis: {last}')

    def _try_place(self, cell, axis, result):
        last = None
        for ax in self._axis_options(cell, axis):
            try:
                plan = self.prims.plan_place(cell, ax)
            except PlanFailure as exc:
                last = exc
                self.log(f'      place {cell} axis {ax} failed: {exc}')
                continue
            self.prims.execute(plan, self.player)
            block = self.world.place(cell)
            result.executed.append(f'place({block}, {cell}, axis={ax})')
            return True
        self.log(f'      place at {cell} impossible: {last}')
        return False

    def _park(self, result):
        """Put the held cube in a free slot of ITS OWN colour's storage column (recovery)."""
        for cell in STORAGE[self.world.colour[self.world.held]]:
            if self.world.at[cell] is None and self._try_place(cell, 'y', result):
                return
        raise PlanFailure('could not park held cube anywhere')

    # ------------------------------------------------------------ pattern
    def run_pattern(self, pattern, plan=None):
        res = PatternResult(pattern, 'failed')
        if plan is None:
            plan, res.plan_seconds = self.symbolic_plan(pattern)
        for attempt in range(MAX_REPLANS + 1):
            if plan is None:
                res.status, res.reason = 'infeasible', 'no symbolic plan; pattern skipped'
                self.log(f'  [{pattern}] no plan found -> SKIPPING')
                break
            if attempt == 0:
                res.symbolic_actions = [a.label() for a in plan]
            self.log(f'  [{pattern}] executing {len(plan)} symbolic actions')
            try:
                replan = False
                for i, act in enumerate(plan, 1):
                    cell, colour, axis = act.args
                    self.log(f'    {i:02d}. {act.label()}')
                    if act.name == 'pick':
                        self._do_pick(cell, colour, axis, res)
                    elif not self._try_place(cell, axis, res):
                        self._park(res)
                        replan = True
                        break
                if not replan:
                    res.status = 'done'
                    break
            except PlanFailure as exc:
                if self.sim.held is not None:
                    self._park(res)
                res.reason = str(exc)
            res.replans += 1
            self.log(f'  [{pattern}] replanning from current state ({res.replans})')
            plan, dt = self.symbolic_plan(pattern)
            res.plan_seconds += dt
        res.sim_time_end = float(self.sim.data.time)
        res.final_grid = self.world.ascii()
        res.drift = [list(d) for d in self.world.check_against_sim(self.sim)]
        if res.status == 'done' and not self.world.matches(pattern):
            res.status, res.reason = 'failed', 'world model does not match pattern'
        return res

    def go_home(self):
        self.player.move(self.prims.plan_home(HOME))
