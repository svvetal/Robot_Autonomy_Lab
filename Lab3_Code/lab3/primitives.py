"""Pick / place motion primitives built on the Lab 2 IK + RRT-Connect stack.

Every primitive is planned COMPLETELY (all joint waypoints, collision checked)
before anything moves, so a failure leaves the robot untouched and the executor
can try the other grasp axis or replan.

  pick(block, axis):  RRT -> hover  | straight descent | close | straight lift
  place(cell, axis):  RRT -> hover  | straight descent | open  | straight retreat

`axis` ('x' or 'y') is the world axis the fingers close along, chosen by the
symbolic planner so the open fingers never sweep into an occupied cell.
"""
from dataclasses import dataclass, field
import numpy as np

from .config import GRASP_YAW, top_grasp_R
from .geometry import make_transform, transform_inv, pose_error
from .ik import PoseIK
from .layout import cell_xyz
from .planning import JointPlanner, Move, Grip, chain_move
from .rrt import PlanFailure, interpolate_edge
from .simulator import yaw_of

CART_STEP = 0.002        # m per differential-IK step on straight-line moves
MAX_JOINT_JUMP = 0.05    # rad per 2 mm step; more means we are near a singularity
DLS_DAMPING = 0.02       # damped least squares lambda
NULL_GAIN = 0.5          # posture hold in the Jacobian null space
MAX_DQ = 0.05            # rad, clamp per DLS iteration
POS_TOL = 2e-4           # m
ROT_TOL = 2e-3           # rad


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _wrap_quarter(a):
    """Wrap to [-pi/4, pi/4): cube orientation modulo its 90-degree symmetry."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


@dataclass
class PrimitivePlan:
    kind: str                      # 'pick' | 'place'
    block: str
    cell: str
    axis: str
    yaw: float
    steps: list = field(default_factory=list)   # Move | Grip


class Primitives:
    def __init__(self, sim, cfg):
        self.sim, self.cfg = sim, cfg
        self.ik = PoseIK(sim, cfg)
        self.planner = JointPlanner(sim, cfg)

    # ----------------------------------------------------------- helpers
    def _edges_free(self, path, carrying):
        res = self.cfg.rrt_edge_resolution
        return all(self.sim.collision_free(q, carrying)
                   for a, b in zip(path, path[1:]) for q in interpolate_edge(a, b, res))

    def _jacobian(self, q):
        """6x7 world-frame tool Jacobian [linear; angular] and tool pose at q."""
        sim, mj = self.sim, self.sim.mj
        s = sim.scratch
        sim.set_arm(s, q)
        mj.mj_kinematics(sim.model, s)
        mj.mj_comPos(sim.model, s)
        jp = np.zeros((3, sim.model.nv))
        jr = np.zeros((3, sim.model.nv))
        mj.mj_jacSite(sim.model, s, jp, jr, sim.tool_id)
        return np.vstack([jp[:, sim.vadr], jr[:, sim.vadr]]), sim.tool_pose(s)

    def _dls_step(self, q, target, q_rest):
        """Damped-least-squares step toward `target`; the redundant DoF pulls
        the arm toward `q_rest` so the posture cannot wander (no elbow drift)."""
        J, H = self._jacobian(q)
        e = pose_error(target, H)
        Jp = J.T @ np.linalg.inv(J @ J.T + DLS_DAMPING ** 2 * np.eye(6))
        # Exact projector: a damped one leaks the posture term into the tool motion.
        N = np.eye(7) - np.linalg.pinv(J, rcond=1e-4) @ J
        null = N @ (-NULL_GAIN * (q - q_rest))
        dq = Jp @ e + null
        dq *= min(1.0, MAX_DQ / max(np.max(np.abs(dq)), 1e-12))
        lo, hi = self.sim.limits[:, 0], self.sim.limits[:, 1]
        return np.clip(q + dq, lo, hi), e

    def _refine(self, q, target, iters=200):
        """Polish a coarse IK solution to sub-millimetre accuracy, same posture."""
        q = np.asarray(q, float)
        q_rest = q.copy()
        for _ in range(iters):
            q, e = self._dls_step(q, target, q_rest)
            if np.linalg.norm(e[:3]) < POS_TOL and np.linalg.norm(e[3:]) < ROT_TOL:
                return q
        raise PlanFailure('IK refinement did not converge')

    def _straight(self, q0, p0, p1, R, carrying, offset=np.zeros(3)):
        """Straight-line tool motion p0 -> p1 at fixed orientation R.

        Continuous differential IK in CART_STEP increments, with null-space
        posture hold toward q0: consecutive configurations differ by a small,
        monotone step, so the 'other' joints don't jitter.
        """
        q_rest = np.asarray(q0, float)
        q = q_rest.copy()
        path = [q.copy()]
        n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / CART_STEP)))
        for s in np.linspace(0, 1, n + 1)[1:]:
            target = make_transform(p0 + s * (p1 - p0) - offset, R)
            for _ in range(20):
                q, e = self._dls_step(q, target, q_rest)
                if np.linalg.norm(e[:3]) < POS_TOL and np.linalg.norm(e[3:]) < ROT_TOL:
                    break
            else:
                raise PlanFailure('straight-line move: differential IK did not converge')
            if np.max(np.abs(q - path[-1])) > MAX_JOINT_JUMP:
                raise PlanFailure('straight-line move: joint jump (near singularity)')
            path.append(q.copy())
        if not self._edges_free(path, carrying):
            raise PlanFailure('straight-line move collides')
        return path

    def _yaw_candidates(self, base_yaw):
        """Tool yaws equivalent for a symmetric parallel gripper (yaw, yaw +/- pi)."""
        return sorted({round(_wrap(base_yaw + k * np.pi), 9) for k in (-1, 0, 1)})

    def _best_hover(self, q_now, hover_xyz, yaws, carrying, offset_fn):
        """IK for each candidate yaw; return candidates sorted by joint distance."""
        sols = []
        for yaw in yaws:
            R = top_grasp_R(yaw)
            try:
                target = make_transform(hover_xyz - offset_fn(R), R)
                q = self._refine(self.ik.solve(target, q_now, carrying=carrying), target)
                if not self.sim.collision_free(q, carrying):
                    continue
                sols.append((float(np.linalg.norm(q - q_now)), yaw, R, q))
            except PlanFailure:
                continue
        if not sols:
            raise PlanFailure(f'no collision-free hover IK at {np.round(hover_xyz, 3).tolist()}')
        return sorted(sols, key=lambda s: s[0])

    # ----------------------------------------------------------- pick
    def plan_pick(self, block, cell, axis):
        sim, cfg = self.sim, self.cfg
        if sim.held is not None:
            raise PlanFailure(f'cannot pick {block}: already holding {sim.held}')
        q_now = sim.arm_q()
        H_block = sim.block_pose(block)
        p = H_block[:3, 3].copy()                       # use the cube's REAL pose
        base_yaw = GRASP_YAW[axis] + _wrap_quarter(yaw_of(H_block[:3, :3]))
        hover = p + [0, 0, cfg.hover_z]
        last = None
        for _, yaw, R, q_hover in self._best_hover(q_now, hover, self._yaw_candidates(base_yaw),
                                                   False, lambda R: np.zeros(3)):
            try:
                to_hover = self.planner.build_move(f'pick_{cell}_approach', q_now, q_hover, False)
                down = self._straight(q_hover, hover, p, R, carrying=False)
                H_tool = sim.fk(down[-1])
                attachment = transform_inv(H_tool) @ H_block
                with sim.hypothetical_grasp(block, attachment):
                    up = list(reversed(down))
                    if not self._edges_free(up, carrying=True):
                        raise PlanFailure('lift with cube collides')
                return PrimitivePlan('pick', block, cell, axis, yaw, [
                    to_hover,
                    chain_move(f'pick_{cell}_descend', down, cfg),
                    Grip(f'pick_{cell}_close', close=True, block_target=block),
                    chain_move(f'pick_{cell}_lift', up, cfg),
                ])
            except PlanFailure as exc:
                last = exc
        raise PlanFailure(f'pick {block}@{cell} axis={axis}: {last}')

    # ----------------------------------------------------------- place
    def plan_place(self, cell, axis):
        sim, cfg = self.sim, self.cfg
        if sim.held is None:
            raise PlanFailure('cannot place: nothing held')
        q_now = sim.arm_q()
        att = sim.attachment
        p = cell_xyz(cell) + [0, 0, cfg.placement_gap]
        # Cube yaw relative to the tool: choose tool yaw so the cube lands grid-aligned.
        rel = _wrap_quarter(yaw_of(att[:3, :3]))
        base_yaw = GRASP_YAW[axis] - rel
        hover = p + [0, 0, cfg.hover_z]
        offset_fn = lambda R: R @ att[:3, 3]           # tool->cube offset in world
        last = None
        for _, yaw, R, q_hover in self._best_hover(q_now, hover, self._yaw_candidates(base_yaw),
                                                   True, offset_fn):
            try:
                transport = self.planner.build_move(f'place_{cell}_transport', q_now, q_hover, True)
                down = self._straight(q_hover, hover, p, R, carrying=True, offset=offset_fn(R))
                # Retreat retraces the descent with open fingers. The symbolic rule
                # guarantees both finger-side cells are empty, so opening is safe.
                up = list(reversed(down))
                return PrimitivePlan('place', sim.held, cell, axis, yaw, [
                    transport,
                    chain_move(f'place_{cell}_descend', down, cfg),
                    Grip(f'place_{cell}_open', close=False, block_target=cell_xyz(cell)),
                    chain_move(f'place_{cell}_retreat', up, cfg),
                ])
            except PlanFailure as exc:
                last = exc
        raise PlanFailure(f'place {sim.held}@{cell} axis={axis}: {last}')

    def plan_home(self, q_home):
        return self.planner.build_move('return_home', self.sim.arm_q(), q_home,
                                       self.sim.held is not None)

    # ----------------------------------------------------------- execute
    def execute(self, plan: PrimitivePlan, player):
        sim = self.sim
        for step in plan.steps:
            if isinstance(step, Move):
                player.move(step)
            elif isinstance(step, Grip):
                if step.close:
                    sim.attach(step.block_target)
                else:
                    held = sim.held
                    sim.release(step.block_target)
                if hasattr(player, 'on_grip'):        # hardware plan capture hook
                    player.on_grip(step, plan, plan.block if step.close else held)
                player.hold(self.cfg.command_hold_seconds)
