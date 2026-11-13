from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator

from .config import (
    BLOCK_HALF,
    BLOCK_NAMES,
    GRIP_CLOSE,
    GRIP_OPEN,
    GRID_CELLS,
    STAGING_X,
    STAGING_RED_Y,
    STAGING_BLUE_Y,
    Settings,
    grid_position,
)
from .geometry import make_transform, transform_inv, rotation_residual
from .simulator import ScriptedAttachment
from .access import top_access_options_xyz
from .grasp_planner import GraspCandidate, GraspPlanner
from .cartesian import VerticalCartesianPlanner, CartesianPathError
from .ik import PoseIK
from .planning import JointPlanner, Move
from .rrt import PlanFailure, interpolate_edge
from .trajectory import duration_for_edge, quintic

class PickPlaceError(RuntimeError):
    pass

@dataclass(frozen=True)
class PickPlaceResult:

    block_name: str
    destination: str
    selected_grasp_yaw_deg: float
    source_xyz_m: tuple[float, float, float]
    final_xyz_m: tuple[float, float, float]
    position_error_m: float
    simulation_time_s: float
    phases: tuple[str, ...]
    attachment_method: str = "scripted_kinematic_debug_only"
    placement_rotation_deg: float = 0.0
    selected_grasp_mode: str = "top"
    grasp_quality_min_singular: float = 0.0
    grasp_quality_volume: float = 0.0
    force_closure_approx: bool = False
    placement_approach: str = "vertical"
    grasp_height_offset_m: float = 0.0

def location_xyz(location: str | Sequence[float]) -> np.ndarray:

    if not isinstance(location, str):
        result = np.asarray(location, dtype=float)
        if result.shape != (3,) or not np.isfinite(result).all():
            raise ValueError("Destination xyz must contain three finite values")
        return result.copy()

    loc = location.upper().strip()
    if loc in GRID_CELLS:
        return grid_position(int(loc[1]), int(loc[2]))
    if len(loc) == 2 and loc[0] in ("R", "B") and loc[1] in "0123":
        i = int(loc[1])
        y = STAGING_RED_Y if loc[0] == "R" else STAGING_BLUE_Y
        return np.array([STAGING_X[i], y, grid_position(1, 1)[2]], dtype=float)
    raise ValueError(
        f"Unknown destination {location!r}; use G00..G22, R0..R3, B0..B3, "
        "or a world-frame xyz vector"
    )

class PickPlaceSkill:

    def __init__(
        self,
        sim,
        cfg: Settings,
        *,
        step_callback: Callable[[str, object], None] | None = None,
        callback_stride: int = 20,
        placement_xy_tolerance_m: float = 0.006,
        placement_z_tolerance_m: float = 0.008,
        placement_steps: int = 5,
        grasp_modes: Sequence[str] = ("top",),
    ) -> None:
        if sim.control != "kinematic":
            raise ValueError(
                "This first Lab 3 pick/place skill supports only "
                "Lab3Sim(control='kinematic'); torque/physics grasps are not yet implemented"
            )
        if callback_stride < 1 or placement_steps < 2:
            raise ValueError("callback_stride >= 1 and placement_steps >= 2 required")
        if placement_xy_tolerance_m <= 0 or placement_z_tolerance_m <= 0:
            raise ValueError("Placement tolerances must be positive")
        self.sim = sim
        self.cfg = cfg

        self.grasp_modes = tuple(grasp_modes)
        self.grasps = GraspPlanner(sim, cfg, grasp_modes=self.grasp_modes)
        self.cartesian = VerticalCartesianPlanner(sim, cfg)
        self.ik = PoseIK(sim, cfg)
        self.joint_planner = JointPlanner(sim, cfg)
        self.step_callback = step_callback
        self.callback_stride = callback_stride
        self.xy_tol = float(placement_xy_tolerance_m)
        self.z_tol = float(placement_z_tolerance_m)
        self.placement_steps = int(placement_steps)
        self.phases: list[str] = []
        self._ticks = 0

    def _phase(self, text: str) -> None:
        self.phases.append(text)
        if self.step_callback is not None:
            self.step_callback(text, self.sim)

    def _smooth_motion_samples(
        self, waypoints: Sequence[np.ndarray], dt: float
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Time-parameterize each checked RRT edge without bending its path.

        A multi-knot cubic spline can leave collision-free line segments. A
        minimum-jerk scalar profile stays on each segment, has zero velocity
        and acceleration at the knots, and obeys the joint-speed bound.
        """
        path = [np.asarray(q, dtype=float).copy() for q in waypoints]
        if not path or any(q.shape != (7,) or not np.isfinite(q).all() for q in path):
            raise PickPlaceError('Invalid joint-space waypoints')
        if not np.isfinite(dt) or dt <= 0:
            raise PickPlaceError('Timestep must be positive')
        path = [q for i, q in enumerate(path)
                if i == 0 or not np.allclose(q, path[i - 1], atol=1e-8, rtol=0)]
        if len(path) == 1:
            return [(path[0], np.zeros(7))]
        samples = []
        for start, end in zip(path, path[1:]):
            duration = duration_for_edge(
                start, end, self.cfg.joint_speed, self.cfg.segment_min_seconds,
            )
            steps = max(1, int(np.ceil(duration / dt)))
            # An integer number of simulation steps makes the duration used
            # for the speed calculation agree with the actual execution time.
            duration = steps * dt
            for k in range(1, steps + 1):
                q, qdot, _ = quintic(start, end, k * dt, duration)
                samples.append((q, qdot))
        return samples

    def _collision_checked_motion(
        self,
        waypoints: Sequence[np.ndarray],
        stage: str,
        *,
        carrying: bool = False,
        allow_gripper_contact_with: str | None = None,
    ) -> None:

        path = [np.asarray(q, float) for q in waypoints]
        if len(path) < 2:
            raise PickPlaceError(f'{stage}: missing joint path')
        if not np.allclose(path[0], self.sim.arm_q(), atol=1e-4):
            raise PickPlaceError(f'{stage}: robot is not at planned path start')

        dt = float(self.sim.model.opt.timestep)
        if not np.isfinite(dt) or dt <= 0:
            raise PickPlaceError('MuJoCo timestep must be positive')
        samples = self._smooth_motion_samples(path, dt)
        check_stride = max(1, int(
            self.cfg.rrt_edge_resolution / max(self.cfg.joint_speed * dt, 1e-9)
        ))
        self._phase(stage)
        for i, (q_des, qd_des) in enumerate(samples):
            if i % check_stride == 0 or i == len(samples) - 1:
                if not self.sim.collision_free(
                    q_des,
                    carrying=carrying,
                    allow_gripper_contact_with=allow_gripper_contact_with,
                ):
                    reasons = self.sim.collision_reasons(
                        q_des,
                        carrying=carrying,
                        allow_gripper_contact_with=allow_gripper_contact_with,
                    )
                    raise PickPlaceError(
                        f'{stage}: smoothed path collides: ' + '; '.join(reasons)
                    )
            self.sim.advance(q_des, qd_des, dt)
            self._ticks += 1
            if (self.step_callback is not None
                    and self._ticks % self.callback_stride == 0):
                self.step_callback(stage, self.sim)

    def _vertical_spline(self, qs: Sequence[np.ndarray]):

        path = np.asarray(qs, dtype=float)
        if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2:
            raise PickPlaceError('Invalid vertical IK path')
        if not np.isfinite(path).all():
            raise PickPlaceError('Non-finite Cartesian IK path')
        s = np.linspace(0.0, 1.0, len(path))
        return PchipInterpolator(s, path, axis=0)

    def _vertical_path_is_free(
        self, qs: Sequence[np.ndarray], *, stage: str,
        carrying: bool, target_contact: str | None = None,
        table_support: str | None = None,
    ) -> None:

        spline = self._vertical_spline(qs)
        begin = self.sim.evaluate(np.asarray(qs[0]), carrying=carrying)
        end = self.sim.evaluate(np.asarray(qs[-1]), carrying=carrying)

        if np.linalg.norm(rotation_residual(begin[:3, :3], end[:3, :3])) > 0.060:
            raise PickPlaceError(f'{stage}: TCP orientation changes during linear motion')

        for t in np.linspace(0., 1., max(24, 4 * len(qs))):
            q = np.asarray(spline(t), float)
            if not self.sim.collision_free(
                q, carrying=carrying,
                allow_gripper_contact_with=target_contact,
                allow_table_support_with=table_support,
            ):
                reasons = self.sim.collision_reasons(
                    q, carrying=carrying,
                    allow_gripper_contact_with=target_contact,
                    allow_table_support_with=table_support,
                )
                raise PickPlaceError(f'{stage}: Cartesian line collision: ' + '; '.join(reasons))
            H = self.sim.evaluate(q, carrying=carrying)
            expected = (1-t)*begin[:3, 3] + t*end[:3, 3]
            linear_error = float(np.linalg.norm(H[:3, 3] - expected))
            rot_error = np.linalg.norm(rotation_residual(begin[:3, :3], H[:3, :3]))
            if linear_error > 0.004 or rot_error > 0.060:
                raise PickPlaceError(
                    f'{stage}: TCP deviates from desired Cartesian line '
                    f'(translation={linear_error*1000:.1f}mm, '
                    f'rotation={np.degrees(rot_error):.1f}deg)'
                )

    def _vertical_motion(
        self, qs: Sequence[np.ndarray], stage: str, *,
        carrying: bool, target_contact: str | None = None,
        table_support: str | None = None,
    ) -> None:

        path = [np.asarray(q, float) for q in qs]
        if not np.allclose(path[0], self.sim.arm_q(), atol=1e-4):
            raise PickPlaceError(f'{stage}: current joints differ from planned path start')
        self._vertical_path_is_free(
            path, stage=stage, carrying=carrying,
            target_contact=target_contact, table_support=table_support,
        )
        spline = self._vertical_spline(path)
        H_start = self.sim.evaluate(path[0], carrying=carrying)
        H_end = self.sim.evaluate(path[-1], carrying=carrying)
        travel_distance = float(np.linalg.norm(H_end[:3, 3] - H_start[:3, 3]))

        max_slope = float(np.max(np.abs(spline.derivative()(np.linspace(0, 1, 161)))))
        duration = max(
            0.6,
            1.875 * travel_distance / 0.085,
            1.875 * max_slope / float(self.cfg.joint_speed),
        ) * 1.05
        dt = float(self.sim.model.opt.timestep)
        n = max(2, int(np.ceil(duration / dt)))
        self._phase(stage)
        for k in range(1, n + 1):
            u = k / n
            progress = 10*u**3 - 15*u**4 + 6*u**5
            speed = (30*u**2 - 60*u**3 + 30*u**4) / duration
            q = np.asarray(spline(progress), float)
            qdot = np.asarray(spline.derivative()(progress) * speed, float)
            if k % max(1, int(0.02 / dt)) == 0 or k == n:
                if not self.sim.collision_free(
                    q, carrying=carrying,
                    allow_gripper_contact_with=target_contact,
                    allow_table_support_with=table_support,
                ):
                    reasons = self.sim.collision_reasons(
                        q, carrying=carrying,
                        allow_gripper_contact_with=target_contact,
                        allow_table_support_with=table_support,
                    )
                    raise PickPlaceError(f'{stage}: execution collision: ' + '; '.join(reasons))
            self.sim.advance(q, qdot, dt)
            self._ticks += 1
            if (self.step_callback is not None and
                    self._ticks % self.callback_stride == 0):
                self.step_callback(stage, self.sim)

    def _animate_gripper(
        self, target: float, stage: str, *, target_block: str,
        duration: float = 0.30,
    ) -> None:

        q = self.sim.arm_q()
        dt = float(self.sim.model.opt.timestep)
        steps = max(2, int(np.ceil(duration / dt)))
        start = float(self.sim.gripper_command)
        self._phase(stage)
        for i in range(1, steps + 1):
            u = i / steps
            smoothstep = 10 * u**3 - 15 * u**4 + 6 * u**5
            self.sim.gripper_command = start + (float(target) - start) * smoothstep
            if not self.sim.collision_free(
                q, carrying=False, allow_gripper_contact_with=target_block
            ):
                raise PickPlaceError(f'{stage}: finger sweep hits an obstacle')
            self.sim.advance(q, np.zeros(7), dt)
            self._ticks += 1
            if (self.step_callback is not None
                    and self._ticks % self.callback_stride == 0):
                self.step_callback(stage, self.sim)

    def _joint_move(
        self, start_q: np.ndarray, end_q: np.ndarray, stage: str, *, carrying: bool
    ) -> Move:
        try:
            move = self.joint_planner.build_move(
                stage, start_q, end_q, carrying=carrying,
            )
            # Validate exactly the samples execution will check before any
            # phase starts. This also rejects an inaccurate cached plan.
            dt = float(self.sim.model.opt.timestep)
            samples = self._smooth_motion_samples(move.path, dt)
            stride = max(1, int(
                self.cfg.rrt_edge_resolution / max(self.cfg.joint_speed * dt, 1e-9)
            ))
            for i, (q, _) in enumerate(samples):
                if i % stride == 0 or i == len(samples) - 1:
                    if not self.sim.collision_free(q, carrying=carrying):
                        raise PlanFailure('Time-parameterized joint path collision: ' +
                                          '; '.join(self.sim.collision_reasons(
                                              q, carrying=carrying,
                                          )))
            return move
        except PlanFailure as exc:
            raise PickPlaceError(f"{stage}: RRT-Connect could not find a path: {exc}") from exc

    def _closed_gripper_is_safe(self, q: np.ndarray, target: str) -> bool:

        old = self.sim.gripper_command
        try:
            self.sim.gripper_command = GRIP_CLOSE
            return self.sim.collision_free(
                q, carrying=False, allow_gripper_contact_with=target
            )
        finally:
            self.sim.gripper_command = old

    def _gripper_sweep_is_free(
        self, q: np.ndarray, start: float, end: float, *,
        target_block: str, carrying: bool = False,
    ) -> bool:
        """Check the entire jaw sweep, not just its endpoint."""
        old = self.sim.gripper_command
        try:
            steps = max(2, int(np.ceil(abs(end - start) / 0.0005)))
            for opening in np.linspace(start, end, steps + 1):
                self.sim.gripper_command = float(opening)
                if not self.sim.collision_free(
                    q, carrying=carrying,
                    allow_gripper_contact_with=target_block,
                    allow_table_support_with=target_block if carrying else None,
                ):
                    return False
            return True
        finally:
            self.sim.gripper_command = old

    def _release_retreat_is_free(
        self, retreat_path: Sequence[np.ndarray], block_name: str, opening: float,
    ) -> bool:
        """Preflight the released-object mode at the *destination* pose.

        Planning normally leaves data at the source. A released cube must stay
        at the predicted destination while the empty fingers retreat around it.
        All temporary state is restored before the transfer is considered.
        """
        attachment = self.sim.attachment
        if attachment is None or attachment.block_name != block_name:
            raise PickPlaceError('Release preflight requires the carried cube')
        original_pose = self.sim.block_pose(block_name)
        original_grip = self.sim.gripper_command
        try:
            self.sim.evaluate(retreat_path[0], carrying=True)
            placed_pose = self.sim.block_pose(block_name, self.sim.scratch)
            self.sim.set_block_pose(block_name, placed_pose)
            self.sim.attachment = None
            self.sim.gripper_command = float(opening)
            self._vertical_path_is_free(
                retreat_path, stage='preflight_release_retreat', carrying=False,
                target_contact=block_name,
            )
            return self._gripper_sweep_is_free(
                retreat_path[-1], opening, GRIP_OPEN, target_block=block_name,
            )
        except PickPlaceError:
            return False
        finally:
            self.sim.set_block_pose(block_name, original_pose)
            self.sim.attachment = attachment
            self.sim.gripper_command = original_grip

    def _select_release_opening(
        self, q: np.ndarray, block_name: str, *,
        retreat_path: Sequence[np.ndarray] | None = None,
    ) -> float | None:
        old = self.sim.gripper_command
        # A 43 mm gap clears a 40 mm cube without opening all the way into
        # neighbouring cells. Open fully only after the vertical retreat.
        for opening in (0.0215, 0.022, 0.023, 0.024, 0.0255, 0.028, GRIP_OPEN):
            if not self._gripper_sweep_is_free(
                q, old, opening, target_block=block_name, carrying=True,
            ):
                continue
            if retreat_path is not None and not self._release_retreat_is_free(
                retreat_path, block_name, opening,
            ):
                continue
            return opening
        return None

    def _placement_clearance_score(self, block_name, xyz, jaw):
        positions = self.sim.block_positions()
        jaw = np.asarray(jaw[:2], float)
        jaw /= max(1e-9, np.linalg.norm(jaw))
        penalty = 0.0
        for name, center in positions.items():
            if name == block_name or abs(float(center[2] - xyz[2])) > .045:
                continue
            for side in (-1., 1.):
                for aperture in (.021, .023, .025):
                    finger = np.asarray(xyz[:2]) + side * aperture * jaw
                    dist = np.maximum(np.abs(finger - center[:2]) - BLOCK_HALF, 0.)
                    penalty += max(0., .007 - float(np.linalg.norm(dist))) / .007
        return penalty

    def _choose_complete_transfer(
        self, block_name: str, goal_xyz: np.ndarray,
        source_rotation: np.ndarray,
    ):
        failures = []
        attempts_by_mode = {mode: 0 for mode in self.grasp_modes}
        rejections_by_mode: dict[str, str] = {}
        strategies = (
            ("top", (0.0,), (45.,)),
            ("top", (0.007,), (45.,)),
            ("top", (0.011,), (45.,)),
            ("tilted", (0.0,), (45.,)),
            ("tilted", (0.007,), (45., 60., 30.)),
            ("side", (0.007,), (45.,)),
            ("side", (0.0,), (45.,)),
        )
        print(f"[grasp] {block_name}: checking {self.grasp_modes}", flush=True)
        local_place_options = {
            (option.yaw_deg % 360.0, option.height_offset_m)
            for option in top_access_options_xyz(
                goal_xyz,
                {name: xyz for name, xyz in self.sim.block_positions().items()
                 if name != block_name}, phase="place",
            )
        }
        checked = set()
        fallback_notified = False
        for mode, heights, pitches in strategies:
            if mode not in self.grasp_modes:
                continue
            if mode != "top" and "top" in self.grasp_modes and not fallback_notified:
                print("[grasp] All top-grasp transfers failed; trying alternative approaches", flush=True)
                fallback_notified = True
            print(f"[grasp] checking {mode}, height={heights[0]*1000:.0f} mm"
                  + (f", tilt={pitches}" if mode == 'tilted' else ''),
                  flush=True)
            original_grip = self.sim.gripper_command
            try:
                # The open approach must fit the same corridor as the closed
                # fingers. An 80 mm gap cannot be used between adjacent cubes.
                self.sim.gripper_command = 0.024
                candidates = self.grasps.find_grasps(
                    block_name, start_q=self.sim.arm_q(), modes=(mode,),
                    height_offsets_m=heights, tilt_pitches_deg=pitches,
                )
            finally:
                self.sim.gripper_command = original_grip
            failures.extend(self.grasps.last_rejections)
            if not candidates:
                reason = (self.grasps.last_rejections[0]
                          if self.grasps.last_rejections else 'no reachable pickup')
                rejections_by_mode.setdefault(mode, reason)
                print(f"[grasp] {mode}: no feasible pickup; {reason[:220]}", flush=True)
                continue
            combinations = []
            for grasp in candidates:
                original_grip = self.sim.gripper_command
                try:
                    self.sim.gripper_command = grasp.approach_opening_m
                    self._vertical_path_is_free(
                        grasp.approach_qs, stage='preflight_pickup_approach',
                        carrying=False, target_contact=block_name,
                    )
                except PickPlaceError as exc:
                    failures.append(str(exc))
                    rejections_by_mode.setdefault(mode, str(exc))
                    continue
                finally:
                    self.sim.gripper_command = original_grip
                for turn in (0., 90., -90., 180.):
                    if grasp.mode == "top" and (
                        (grasp.yaw_deg + turn) % 360.0, grasp.contact_height_offset_m,
                    ) not in local_place_options:
                        failures.append(
                            f"top destination {grasp.yaw_deg+turn:g}°: "
                            "original finger geometry has no free release/retreat"
                        )
                        continue
                    a = np.deg2rad(turn)
                    ca, sa = float(np.cos(a)), float(np.sin(a))
                    Rz = np.array([[ca, -sa, 0.], [sa, ca, 0.], [0., 0., 1.]])
                    jaw = Rz @ grasp.grasp_pose[:3, 1]
                    risk = self._placement_clearance_score(block_name, goal_xyz, jaw)
                    combinations.append(((risk, grasp.score, abs(turn)), grasp,
                                         turn, Rz @ source_rotation))
            combinations.sort(key=lambda item: item[0])
            unreachable_pickups = set()
            for _, grasp, turn, destination_rotation in combinations:
                key = (grasp.mode, grasp.yaw_deg, grasp.tilt_from_vertical_deg,
                       grasp.contact_height_offset_m)
                if key in unreachable_pickups:
                    continue
                attempts_by_mode[mode] += 1
                if not self._gripper_sweep_is_free(
                    grasp.grasp_q, grasp.approach_opening_m, GRIP_CLOSE,
                    target_block=block_name,
                ):
                    unreachable_pickups.add(key)
                    rejections_by_mode.setdefault(mode, 'Closed fingers hit an obstacle')
                    continue
                try:
                    transfer, place_path, lift_path, placement_kind, release_opening = self._preplan_transfer(
                        block_name, grasp, goal_xyz, destination_rotation,
                    )
                    original_grip = self.sim.gripper_command
                    try:
                        self.sim.gripper_command = grasp.approach_opening_m
                        inbound = self._joint_move(
                            self.sim.arm_q(), grasp.pregrasp_q,
                            'travel_to_pregrasp', carrying=False,
                        )
                    finally:
                        self.sim.gripper_command = original_grip
                    if not self._gripper_sweep_is_free(
                        self.sim.arm_q(), original_grip, grasp.approach_opening_m,
                        target_block=block_name,
                    ):
                        raise PickPlaceError('Cannot prepare pickup aperture at current pose')
                    print(f"[grasp] SELECTED {grasp.mode} {grasp.yaw_deg:g}°; "
                          f"pickup height={grasp.contact_height_offset_m*1000:.0f} mm; "
                          f"lift={'reverse-top-approach' if mode == 'top' else 'vertical'}; "
                          f"placement={placement_kind}", flush=True)
                    return (grasp, inbound, lift_path, transfer, place_path,
                            turn, placement_kind, release_opening)
                except (PickPlaceError, PlanFailure) as exc:
                    identity = (mode, grasp.yaw_deg,
                                grasp.tilt_from_vertical_deg,
                                grasp.contact_height_offset_m, turn)
                    if identity in checked:
                        continue
                    checked.add(identity)
                    message = str(exc)
                    rejections_by_mode.setdefault(mode, message)
                    failures.append(
                        f'{grasp.mode} {grasp.yaw_deg:g}° '
                        f'pitch={grasp.tilt_from_vertical_deg:g}° '
                        f'offset={grasp.contact_height_offset_m*1000:g}mm '
                        f'/ turn={turn:g}°: {message[:190]}'
                    )
                    if 'start is in collision' in message:
                        unreachable_pickups.add(key)
            reason = rejections_by_mode.get(mode, 'no complete feasible transfer')
            print(f"[grasp] {mode}: {len(candidates)} pickup candidates, "
                  f"{attempts_by_mode[mode]} transfer checks so far; "
                  f"first rejection: {reason[:220]}", flush=True)
        blocker_geoms = sorted(set(re.findall(
            r"\b(?:red|blue)_[0-3]_geom\b", "; ".join(failures)
        )))
        reason_text = '; '.join(
            f"{mode}: {reason[:150]}" for mode, reason in rejections_by_mode.items()
        )
        raise PickPlaceError(
            f'No collision-free grasp/transfer for {block_name}. '
            f'Modes attempted: {attempts_by_mode}. '
            f'Examples: {reason_text}'
            + ('; blocked_by=' + ','.join(blocker_geoms) if blocker_geoms else '')
        )

    def _placement_approach_offsets(self, block_name: str, goal_xyz: np.ndarray):
        repulsion = np.zeros(2)
        for name, pos in self.sim.block_positions().items():
            if name == block_name or abs(float(pos[2] - goal_xyz[2])) > 0.05:
                continue
            delta = goal_xyz[:2] - pos[:2]
            distance = float(np.linalg.norm(delta))
            if 0.035 < distance < 0.11:
                repulsion += delta / (distance**2)
        choices = [np.zeros(2)]
        norm = float(np.linalg.norm(repulsion))
        if norm > 1e-9:
            away = repulsion / norm
            choices.extend([0.028 * away, 0.048 * away])
        unique = []
        for offset in choices:
            if not any(np.allclose(offset, seen) for seen in unique):
                unique.append(offset)
        return unique

    def _plan_vertical_placement(
        self, block_name: str, goal_xyz: np.ndarray, object_rotation: np.ndarray,
        *, start_q: np.ndarray | None = None,
        lateral_offset_xy: np.ndarray | None = None,
    ) -> tuple[Move, list[np.ndarray], float]:

        attachment = self.sim.attachment
        if attachment is None or attachment.block_name != block_name:
            raise PickPlaceError("Placement requires the requested attached block")
        tool_from_object = transform_inv(attachment.tool_to_block)

        place_tool = make_transform(goal_xyz, object_rotation) @ tool_from_object
        lateral = np.zeros(2) if lateral_offset_xy is None else np.asarray(lateral_offset_xy, float)
        pre_tool = place_tool.copy()
        pre_tool[:2, 3] += lateral
        pre_tool[2, 3] += self.cfg.hover_z
        start_q = self.sim.arm_q() if start_q is None else np.asarray(start_q, float)

        try:
            pre_q_coarse = self.ik.solve(pre_tool, start_q, carrying=True)
            pre_q = self.cartesian.refine_start(
                pre_tool, pre_q_coarse, carrying=True,
            )
            approach = list(self.cartesian.plan_linear(
                pre_q, pre_tool, place_tool,
                carrying=True, table_support=block_name,
            ))
            self._vertical_path_is_free(
                approach, stage='preflight_place_approach', carrying=True,
                table_support=block_name,
            )
            release_opening = self._select_release_opening(
                approach[-1], block_name, retreat_path=list(reversed(approach)),
            )
            if release_opening is None:
                raise PickPlaceError('No collision-free finger release opening')
            travel = self._joint_move(
                start_q, pre_q, 'carried_transfer_to_preplacement', carrying=True
            )
        except (PlanFailure, CartesianPathError) as exc:
            raise PickPlaceError(f'Destination vertical placement planning failed: {exc}') from exc
        return travel, approach, release_opening

    def _preplan_transfer(
        self, block_name: str, grasp: GraspCandidate,
        goal_xyz: np.ndarray, object_rotation: np.ndarray,
    ) -> tuple[Move, list[np.ndarray], list[np.ndarray], str, float]:

        if self.sim.attachment is not None:
            raise PickPlaceError('Cannot preplan while already carrying a cube')
        original_grip = self.sim.gripper_command
        block_H = self.sim.block_pose(block_name)
        try:
            self.sim.gripper_command = GRIP_CLOSE
            predicted_tool_H = self.sim.evaluate(grasp.grasp_q)
            predicted_attachment = ScriptedAttachment(
                block_name=block_name,
                tool_to_block=transform_inv(predicted_tool_H) @ block_H,
            )
            self.sim.attachment = predicted_attachment

            if grasp.mode == "top":
                lift_path = list(reversed(grasp.approach_qs))
            else:
                lift_tool = grasp.grasp_pose.copy()
                lift_tool[2, 3] += self.cfg.hover_z
                lift_path = list(self.cartesian.plan_vertical(
                    grasp.grasp_q, grasp.grasp_pose, lift_tool,
                    carrying=True, table_support=block_name,
                ))
            self._vertical_path_is_free(
                lift_path, stage='preflight_vertical_lift', carrying=True,
                table_support=block_name,
            )
            errors = []
            for offset in self._placement_approach_offsets(block_name, goal_xyz):
                try:
                    travel, place_approach, release_opening = self._plan_vertical_placement(
                        block_name, goal_xyz, object_rotation,
                        start_q=lift_path[-1], lateral_offset_xy=offset,
                    )
                    mode = 'vertical' if np.linalg.norm(offset) < 1e-8 else 'diagonal'
                    return travel, place_approach, lift_path, mode, release_opening
                except (PickPlaceError, PlanFailure) as exc:
                    errors.append(str(exc))
            raise PickPlaceError('All placement approaches failed: ' +
                                 '; '.join(errors[:2]))
        finally:
            self.sim.attachment = None
            self.sim.gripper_command = original_grip

    def _destination_is_available(self, block_name: str, goal_xyz: np.ndarray) -> None:

        for name, p in self.sim.block_positions().items():
            if name == block_name:
                continue
            if (
                np.linalg.norm(p[:2] - goal_xyz[:2]) < 2.0 * BLOCK_HALF + self.cfg.collision_margin
                and abs(float(p[2] - goal_xyz[2])) < 2.0 * BLOCK_HALF
            ):
                raise PickPlaceError(f"Destination is occupied or too close to {name}")

    def move(self, block_name: str, destination: str | Sequence[float]) -> PickPlaceResult:

        if block_name not in BLOCK_NAMES:
            raise ValueError(f"Unknown block: {block_name}")
        if self.sim.attachment is not None:
            raise PickPlaceError("Another block is already attached; reset/resolve it first")
        self.phases.clear()
        self._ticks = 0
        goal_xyz = location_xyz(destination)
        source_H = self.sim.block_pose(block_name)
        source_xyz = source_H[:3, 3].copy()
        source_rotation = source_H[:3, :3].copy()
        self._destination_is_available(block_name, goal_xyz)
        if np.linalg.norm(source_xyz[:2] - goal_xyz[:2]) < self.xy_tol:
            raise PickPlaceError("Block is already at/near the requested destination")

        (grasp, inbound, lift_path, to_pre, place_approach,
         placement_turn, placement_kind, release_opening) = (
            self._choose_complete_transfer(
                block_name, goal_xyz, source_rotation,
            )
        )

        self._animate_gripper(
            grasp.approach_opening_m, 'prepare_pickup_aperture', target_block=block_name,
        )
        self._collision_checked_motion(
            inbound.path, 'rrt_to_pregrasp', carrying=False
        )
        self._vertical_motion(
            grasp.approach_qs,
            'vertical_pickup_approach' if grasp.mode == 'top' else 'linear_pickup_approach',
            carrying=False, target_contact=block_name,
        )

        if not self._closed_gripper_is_safe(self.sim.arm_q(), block_name):
            raise PickPlaceError('Closed gripper would hit a neighboring cube')
        self._animate_gripper(
            GRIP_CLOSE, 'close_gripper', target_block=block_name
        )
        self.sim.attach(block_name)
        if not self.sim.collision_free(
            self.sim.arm_q(), carrying=True,
            allow_table_support_with=block_name,
        ):
            reasons = self.sim.collision_reasons(
                self.sim.arm_q(), carrying=True,
                allow_table_support_with=block_name,
            )
            raise PickPlaceError('Collision immediately after pickup: ' + '; '.join(reasons))

        self._vertical_motion(
            lift_path, 'vertical_lift',
            carrying=True, table_support=block_name,
        )
        self._collision_checked_motion(
            to_pre.path, 'rrt_transfer_with_cube', carrying=True
        )
        self._vertical_motion(
            place_approach, placement_kind + '_place_approach',
            carrying=True, table_support=block_name,
        )

        observed = self.sim.block_pose(block_name)[:3, 3]
        xy_error = float(np.linalg.norm(observed[:2] - goal_xyz[:2]))
        z_error = float(abs(observed[2] - goal_xyz[2]))
        if xy_error > self.xy_tol or z_error > self.z_tol:
            raise PickPlaceError(
                f'Placement error before release: xy={xy_error:.4f} m, '
                f'z={z_error:.4f} m; cube remains attached'
            )
        # Execute the exact aperture whose complete retreat was preflighted.
        # Re-selecting here from endpoint clearance alone can choose a smaller
        # opening whose fingers collide further along the retreat.
        self._animate_gripper(
            release_opening, 'open_gripper', target_block=block_name
        )
        self.sim.release()
        self._vertical_motion(
            list(reversed(place_approach)), 'retreat_after_release',
            carrying=False, target_contact=block_name,
        )

        self._animate_gripper(GRIP_OPEN, 'open_above_blocks', target_block=block_name)
        final_xyz = self.sim.block_pose(block_name)[:3, 3].copy()
        final_error = float(np.linalg.norm(final_xyz - goal_xyz))
        return PickPlaceResult(
            block_name=block_name,
            destination=str(destination),
            selected_grasp_yaw_deg=grasp.yaw_deg,
            source_xyz_m=tuple(float(v) for v in source_xyz),
            final_xyz_m=tuple(float(v) for v in final_xyz),
            position_error_m=final_error,
            simulation_time_s=float(self.sim.data.time),
            phases=tuple(self.phases),
            placement_rotation_deg=float(placement_turn),
            selected_grasp_mode=grasp.mode,
            grasp_quality_min_singular=grasp.quality_min_singular,
            grasp_quality_volume=grasp.quality_volume,
            force_closure_approx=grasp.force_closure_approx,
            placement_approach=placement_kind,
            grasp_height_offset_m=grasp.contact_height_offset_m,
        )

def main() -> None:

    parser = argparse.ArgumentParser(description="Execute one Lab 3 MuJoCo transfer")
    parser.add_argument("--block", default="red_0", choices=BLOCK_NAMES)
    parser.add_argument("--to", default="G11", dest="destination",
                        help="G00..G22, R0..R3 or B0..B3")
    mode_options = parser.add_mutually_exclusive_group()
    mode_options.add_argument("--top-only", action="store_true", help="Use top grasps only")
    mode_options.add_argument("--side-only", action="store_true", help="Diagnose side grasps only")
    mode_options.add_argument("--tilted-only", action="store_true", help="Diagnose tilted grasps only")
    parser.add_argument("--view", action="store_true",
                        help="Watch the live generated motion in a MuJoCo viewer")
    parser.add_argument("--inspect-grasps", action="store_true",
                        help="Print all grasp yaw scores and 0-degree rejection reasons, then exit")
    args = parser.parse_args()

    from .simulator import Lab3Sim

    sim = Lab3Sim(Settings(), control="kinematic")
    modes = (("top",) if args.top_only else ("side",) if args.side_only else
             ("tilted",) if args.tilted_only else ("top",))
    if args.inspect_grasps:
        selector = GraspPlanner(sim, sim.cfg, grasp_modes=modes)
        grasps = selector.find_grasps(args.block)
        print('Feasible grasps (lower score is better):')
        for candidate in grasps:
            print(f'  {candidate.mode:4s} angle={candidate.yaw_deg:5.1f}°, '
                  f'pitch={candidate.tilt_from_vertical_deg:.0f}°, '
                  f'score={candidate.score:.3f}, '
                  f'sigma_min={candidate.quality_min_singular:.3f}, '
                  f'G_volume={candidate.quality_volume:.3f}, '
                  f'closure_proxy={candidate.force_closure_approx}, '
                  f'neighbour_risks={candidate.nearby_finger_risks}')
        print('Rejected grasps:')
        for reason in selector.last_rejections:
            print(' ', reason)
        return
    handle = None
    if args.view:
        from mujoco import viewer
        handle = viewer.launch_passive(sim.model, sim.data)
        handle.cam.distance = 2.3
        handle.cam.azimuth = 140
        handle.cam.elevation = -30

    def report(phase: str, _: object) -> None:

        if phase != report.previous:
            print("  ->", phase, flush=True)
            report.previous = phase
        if handle is not None:
            if not handle.is_running():
                raise PickPlaceError("MuJoCo viewer was closed during transfer")
            handle.sync()

    report.previous = ""
    skill = PickPlaceSkill(
        sim, sim.cfg, step_callback=report,
        grasp_modes=modes,
    )
    try:
        result = skill.move(args.block, args.destination)
        print("Transfer complete (scripted attachment, kinematic simulator only)")
        print(f"Block: {result.block_name}")
        print(f"Destination: {result.destination}")
        print(f"Grasp mode: {result.selected_grasp_mode}, "
              f"orientation/approach angle: {result.selected_grasp_yaw_deg:.0f} deg")
        print(f"Grasp matrix quality: sigma_min={result.grasp_quality_min_singular:.4f}, "
              f"sqrt(det(GG^T))={result.grasp_quality_volume:.4f}, "
              f"sampled_closure={result.force_closure_approx}")
        print(f"Observed final xyz: {np.round(result.final_xyz_m, 4).tolist()}")
        print(f"Final position error: {result.position_error_m:.4f} m")
        print(f"MuJoCo simulation time: {result.simulation_time_s:.2f} s")
        if handle is not None:
            print("Close the viewer window to exit.")
            while handle.is_running():
                handle.sync()
                import time
                time.sleep(0.04)
    except (PickPlaceError, PlanFailure) as exc:
        print(f"BLOCK TRANSFER STOPPED: {exc}")
        print(f"Current attachment: {sim.attachment}")
        raise SystemExit(1) from exc
    finally:
        if handle is not None:
            handle.close()

if __name__ == "__main__":
    main()