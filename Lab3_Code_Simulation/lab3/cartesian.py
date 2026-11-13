from __future__ import annotations

import numpy as np

from .geometry import pose_error, rotation_residual
from .rrt import PlanFailure, interpolate_edge

class CartesianPathError(PlanFailure):
    pass

class VerticalCartesianPlanner:
    def __init__(
        self, sim, cfg, *, spacing_m=0.005,
        position_tolerance_m=0.0015, orientation_tolerance_rad=0.035,
        max_step_joint_change_rad=0.18, damping=0.02,
    ):
        self.sim = sim
        self.cfg = cfg
        self.spacing = float(spacing_m)
        self.pos_tol = float(position_tolerance_m)
        self.rot_tol = float(orientation_tolerance_rad)
        self.max_q_step = float(max_step_joint_change_rad)
        self.damping = float(damping)
        if min(self.spacing, self.pos_tol, self.rot_tol,
               self.max_q_step, self.damping) <= 0:
            raise ValueError("Cartesian planner parameters must be positive")

    def _solve_local(self, target, q_seed, *, carrying, target_contact,
                     table_support, step_bound=None):

        q_seed = np.asarray(q_seed, dtype=float)
        q = q_seed.copy()
        sim = self.sim
        lo, hi = sim.limits[:, 0], sim.limits[:, 1]
        if q.shape != (7,) or not np.isfinite(q).all():
            raise CartesianPathError("Invalid IK starting joints")
        if np.any(q < lo) or np.any(q > hi):
            raise CartesianPathError("IK starting joints outside safety limits")
        bound = self.max_q_step if step_bound is None else float(step_bound)
        jacp = np.zeros((3, sim.model.nv), dtype=float)
        jacr = np.zeros((3, sim.model.nv), dtype=float)
        for _ in range(65):
            current = sim.evaluate(q, carrying=carrying)
            err = pose_error(target, current)
            if (np.linalg.norm(err[:3]) <= self.pos_tol and
                    np.linalg.norm(err[3:]) <= self.rot_tol):
                break

            sim.mj.mj_jacSite(sim.model, sim.scratch, jacp, jacr, sim.tool_id)
            J = np.vstack((jacp[:, sim.vadr], 0.25 * jacr[:, sim.vadr]))
            weighted_error = np.r_[err[:3], 0.25 * err[3:]]
            try:
                dq = J.T @ np.linalg.solve(
                    J @ J.T + self.damping**2 * np.eye(6), weighted_error
                )
            except np.linalg.LinAlgError as exc:
                raise CartesianPathError("Jacobian IK failed") from exc
            if not np.isfinite(dq).all():
                raise CartesianPathError("Non-finite Cartesian IK step")
            biggest = float(np.max(np.abs(dq)))
            if biggest > 0.075:
                dq *= 0.075 / biggest

            old_error = float(np.linalg.norm(weighted_error))
            accepted = False
            for alpha in (1.0, 0.5, 0.25, 0.125):
                q_try = np.clip(q + alpha * dq, lo + 1e-6, hi - 1e-6)
                err_try = pose_error(target, sim.evaluate(q_try, carrying=carrying))
                new_error = float(np.linalg.norm(np.r_[err_try[:3], 0.25 * err_try[3:]]))
                if new_error < old_error - 1e-8:
                    q = q_try
                    accepted = True
                    break
            if not accepted:
                break
        else:
            pass

        actual = sim.evaluate(q, carrying=carrying)
        err = pose_error(target, actual)
        if (np.linalg.norm(err[:3]) > self.pos_tol or
                np.linalg.norm(err[3:]) > self.rot_tol):
            raise CartesianPathError(
                f"Local Cartesian IK did not converge: "
                f"xyz={np.linalg.norm(err[:3])*1000:.1f} mm, "
                f"rotation={np.degrees(np.linalg.norm(err[3:])):.1f} deg"
            )
        if float(np.max(np.abs(q - q_seed))) > bound:
            raise CartesianPathError(
                "Large joint change required for small Cartesian step "
                "(rejecting IK branch switch)"
            )
        if not sim.collision_free(
            q, carrying=carrying,
            allow_gripper_contact_with=target_contact,
            allow_table_support_with=table_support,
        ):
            reasons = sim.collision_reasons(
                q, carrying=carrying,
                allow_gripper_contact_with=target_contact,
                allow_table_support_with=table_support,
            )
            raise CartesianPathError("Cartesian waypoint collision: " + "; ".join(reasons))
        return q

    def refine_start(self, target, q_seed, *, carrying=False,
                     target_contact=None, table_support=None):

        return self._solve_local(
            np.asarray(target, dtype=float), q_seed, carrying=carrying,
            target_contact=target_contact, table_support=table_support,
            step_bound=0.60,
        )

    def plan_linear(
        self, start_q, start_pose, end_pose, *, carrying=False,
        target_contact=None, table_support=None,
    ) -> tuple[np.ndarray, ...]:

        a = np.asarray(start_pose, float)
        b = np.asarray(end_pose, float)
        if a.shape != (4, 4) or b.shape != (4, 4):
            raise ValueError("Start/end must be 4x4 TCP transforms")
        if np.linalg.norm(rotation_residual(a[:3, :3], b[:3, :3])) > 1e-6:
            raise CartesianPathError("Linear approach requires fixed tool orientation")
        start_q = np.asarray(start_q, float)
        actual_start = self.sim.evaluate(start_q, carrying=carrying)
        initial_error = pose_error(a, actual_start)
        if (np.linalg.norm(initial_error[:3]) > self.pos_tol or
                np.linalg.norm(initial_error[3:]) > self.rot_tol):
            raise CartesianPathError("Start IK solution differs from target TCP")
        length = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
        n = max(2, int(np.ceil(length / self.spacing)))
        qs = [start_q.copy()]
        previous = start_q.copy()
        for i in range(1, n + 1):
            t = i / n
            target = a.copy()
            target[:3, 3] = (1. - t) * a[:3, 3] + t * b[:3, 3]
            q = self._solve_local(
                target, previous, carrying=carrying,
                target_contact=target_contact, table_support=table_support,
            )
            for q_mid in interpolate_edge(previous, q, self.cfg.rrt_edge_resolution):
                if not self.sim.collision_free(
                    q_mid, carrying=carrying,
                    allow_gripper_contact_with=target_contact,
                    allow_table_support_with=table_support,
                ):
                    raise CartesianPathError(
                        f"Collision between Cartesian points {i-1}/{n} and {i}/{n}")
            qs.append(q.copy())
            previous = q
        return tuple(qs)

    def plan_vertical(
        self, start_q, start_pose, end_pose, *, carrying=False,
        target_contact=None, table_support=None,
    ) -> tuple[np.ndarray, ...]:

        a = np.asarray(start_pose, float)
        b = np.asarray(end_pose, float)
        if a.shape != (4, 4) or b.shape != (4, 4):
            raise ValueError("Start/end must be 4x4 TCP transforms")
        if np.linalg.norm(a[:2, 3] - b[:2, 3]) > 1e-7:
            raise CartesianPathError("Vertical path requires fixed XY")
        return self.plan_linear(
            start_q, a, b, carrying=carrying,
            target_contact=target_contact, table_support=table_support,
        )