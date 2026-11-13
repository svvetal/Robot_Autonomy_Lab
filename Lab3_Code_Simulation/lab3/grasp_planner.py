from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from .config import BLOCK_HALF, BLOCK_NAMES, TOP_R, Settings
from .access import top_access_options_xyz
from .geometry import make_transform
from .ik import PoseIK
from .cartesian import VerticalCartesianPlanner
from .grasp_quality import GraspQuality, evaluate_cube_grasp
from .rrt import PlanFailure

DEFAULT_TOP_GRASP_YAWS_DEG = (0., 90., 180., 270.)
DEFAULT_SIDE_AZIMUTHS_DEG = (0., 45., 90., 135., 180., 225., 270., 315.)

class GraspPlanningFailure(PlanFailure):
    pass

@dataclass(frozen=True)
class GraspCandidate:
    block_name: str
    yaw_deg: float
    pregrasp_pose: np.ndarray
    grasp_pose: np.ndarray
    pregrasp_q: np.ndarray
    grasp_q: np.ndarray
    approach_qs: tuple[np.ndarray, ...]
    score: float
    nearby_finger_risks: int
    face_alignment_error: float = 0.0
    mode: str = "top"
    approach_direction_world: tuple[float, float, float] = (0., 0., -1.)
    quality_min_singular: float = 0.0
    quality_volume: float = 0.0
    force_closure_approx: bool = False
    grasp_quality: float = 0.0
    contact_height_offset_m: float = 0.0
    tilt_from_vertical_deg: float = 0.0
    approach_opening_m: float = 0.024

@lru_cache(maxsize=128)
def _quality(cube_R_tuple, jaw_tuple, friction):
    return evaluate_cube_grasp(
        np.array(cube_R_tuple).reshape(3, 3), np.array(jaw_tuple),
        half_width_m=BLOCK_HALF, friction_coefficient=friction,
    )

class GraspPlanner:
    def __init__(
        self, sim, cfg: Settings, *,
        yaw_degrees: Sequence[float] = DEFAULT_TOP_GRASP_YAWS_DEG,
        approach_steps: int = 4,
        face_alignment_weight: float = 1.2,
        friction_coefficient: float = 0.6,
        grasp_modes: Sequence[str] = ("top",),
        side_azimuths: Sequence[float] = DEFAULT_SIDE_AZIMUTHS_DEG,
    ):
        if not yaw_degrees or approach_steps < 2:
            raise ValueError("Need at least one yaw and two approach steps")
        self.sim, self.cfg = sim, cfg
        self.yaw_degrees = tuple(float(a) for a in yaw_degrees)
        self.approach_steps = int(approach_steps)
        self.face_alignment_weight = float(face_alignment_weight)
        self.friction_coefficient = float(friction_coefficient)
        self.grasp_modes = tuple(grasp_modes)
        self.side_azimuths = tuple(float(x) for x in side_azimuths)
        if not self.grasp_modes or set(self.grasp_modes) - {"top", "tilted", "side"}:
            raise ValueError("Invalid grasp modes")
        self.ik = PoseIK(sim, cfg)
        self.cartesian = VerticalCartesianPlanner(sim, cfg)
        self.last_rejections: list[str] = []

    @staticmethod
    def top_rotation(yaw_deg: float) -> np.ndarray:
        a = np.deg2rad(yaw_deg)
        c, s = float(np.cos(a)), float(np.sin(a))
        return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ TOP_R

    @staticmethod
    def tilted_rotation(azimuth_deg: float, pitch_from_vertical_deg: float = 45.) -> np.ndarray:
        a = np.deg2rad(azimuth_deg)
        t = np.deg2rad(pitch_from_vertical_deg)
        approach = np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), -np.cos(t)])
        jaw = np.array([-np.sin(a), np.cos(a), 0.])
        return np.column_stack((np.cross(jaw, approach), jaw, approach))

    @staticmethod
    def side_rotation(azimuth_deg: float) -> np.ndarray:
        return GraspPlanner.tilted_rotation(azimuth_deg, 90.)

    def _finger_neighbour_risk(self, block_name: str, R: np.ndarray) -> int:
        positions = self.sim.block_positions()
        center = positions[block_name]
        jaw = R[:2, 1]
        risk = 0
        for name, xyz in positions.items():
            if name == block_name or abs(float(xyz[2] - center[2])) > .05:
                continue
            d = xyz[:2] - center[:2]
            along = abs(float(d @ jaw))
            across = abs(float(d[0] * jaw[1] - d[1] * jaw[0]))
            if 0.015 < along < 0.085 and across < 0.029:
                risk += 1
        return risk

    def find_grasps(
        self, block_name: str, *, start_q=None, modes: Sequence[str] | None = None,
        height_offsets_m: Sequence[float] = (0.,),
        tilt_pitches_deg: Sequence[float] = (45.,),
    ) -> list[GraspCandidate]:
        self.last_rejections.clear()
        if block_name not in BLOCK_NAMES:
            raise ValueError(f"Unknown block: {block_name}")
        if self.sim.attachment is not None:
            raise GraspPlanningFailure("A cube is already attached")
        cube_H = self.sim.block_pose(block_name)
        center, cube_R = cube_H[:3, 3].copy(), cube_H[:3, :3].copy()
        if cube_R[2, 2] < .92:
            raise GraspPlanningFailure("Cube is not upright")
        q0 = self.sim.arm_q() if start_q is None else np.asarray(start_q, float)
        if q0.shape != (7,) or not np.isfinite(q0).all():
            raise ValueError("start_q must have 7 finite joints")
        selected = tuple(modes) if modes is not None else self.grasp_modes
        if not selected or set(selected) - set(self.grasp_modes):
            raise ValueError("Requested grasp mode is not enabled")
        offsets = tuple(float(x) for x in height_offsets_m)
        pitches = tuple(float(x) for x in tilt_pitches_deg)
        if not offsets or any(not (0. <= x <= 0.011) for x in offsets):
            raise ValueError("Grasp offsets must be within [0, 0.011] meters")
        if "tilted" in selected and (not pitches or any(not (15. <= p <= 80.) for p in pitches)):
            raise ValueError("Tilt pitches must be between 15 and 80 degrees")
        local_top_options = {
            (option.yaw_deg % 360.0, option.height_offset_m)
            for option in top_access_options_xyz(
                center,
                {name: xyz for name, xyz in self.sim.block_positions().items()
                 if name != block_name},
                phase="pick",
            )
        } if "top" in selected else set()
        options = []
        if "top" in selected:
            options.extend(("top", yaw, 0., self.top_rotation(yaw)) for yaw in self.yaw_degrees)
        if "tilted" in selected:
            options.extend(("tilted", a, pitch, self.tilted_rotation(a, pitch))
                           for pitch in pitches for a in self.side_azimuths)
        if "side" in selected:
            options.extend(("side", a, 90., self.side_rotation(a)) for a in self.side_azimuths)
        ranked = []
        for mode, angle, pitch, R in options:
            quality = _quality(
                tuple(np.round(cube_R, 5).ravel()),
                tuple(np.round(R[:, 1], 5)), self.friction_coefficient,
            )
            risk = self._finger_neighbour_risk(block_name, R)
            ranked.append((risk + 2. * (1. - quality.contact_alignment) - .2 * quality.score,
                           mode, angle, pitch, R, quality, risk))
        ranked.sort(key=lambda row: row[0])
        valid: list[GraspCandidate] = []
        for _, mode, angle, pitch, R, quality, risk in ranked:
            if quality.contact_alignment < .68:
                self.last_rejections.append(f"{mode} {angle:g}°: oblique contacts")
                continue
            direction = R[:, 2]
            hover = (self.cfg.hover_z if mode == "top" else
                     min(self.cfg.hover_z, .085 if mode == "tilted" else .060))
            for height_offset in offsets:
                if mode == "top" and (angle % 360.0, height_offset) not in local_top_options:
                    self.last_rejections.append(
                        f"top {angle:g}° offset={height_offset*1000:g}mm: "
                        "original finger geometry has no free vertical approach"
                    )
                    continue
                grasp_center = center + np.array([0., 0., height_offset])
                pregrasp = make_transform(grasp_center - hover * direction, R)
                grasp = make_transform(grasp_center, R)
                try:
                    coarse = self.ik.solve(pregrasp, q0, carrying=False)
                    q_pre = self.cartesian.refine_start(pregrasp, coarse)
                    approach = self.cartesian.plan_linear(
                        q_pre, pregrasp, grasp, carrying=False, target_contact=block_name,
                    )
                    motion = sum(float(np.linalg.norm(b - a)) for a, b in
                                 zip(approach, approach[1:]))
                    score = (
                        .3 * float(np.linalg.norm(q_pre - q0)) + .15 * motion + .7 * risk
                        + self.face_alignment_weight * (1. - quality.contact_alignment)
                        + (1.0 if not quality.force_closure_approx else 0.) - quality.score
                        + 0.1 * (height_offset / 0.01)
                    )
                    valid.append(GraspCandidate(
                        block_name=block_name, yaw_deg=angle,
                        pregrasp_pose=pregrasp, grasp_pose=grasp,
                        pregrasp_q=q_pre.copy(), grasp_q=approach[-1].copy(),
                        approach_qs=tuple(q.copy() for q in approach),
                        score=float(score), nearby_finger_risks=risk,
                        face_alignment_error=1. - quality.contact_alignment,
                        mode=mode, approach_direction_world=tuple(float(v) for v in direction),
                        quality_min_singular=quality.min_singular,
                        quality_volume=quality.volume,
                        force_closure_approx=quality.force_closure_approx,
                        grasp_quality=quality.score,
                        contact_height_offset_m=height_offset,
                        tilt_from_vertical_deg=pitch,
                        approach_opening_m=float(self.sim.gripper_command),
                    ))
                except (PlanFailure, ValueError) as exc:
                    self.last_rejections.append(f"{mode} {angle:g}° pitch={pitch:g}° offset={height_offset*1000:g}mm: {exc}")
        return sorted(valid, key=lambda c: c.score)

    def best_grasp(self, block_name: str, *, start_q=None) -> GraspCandidate:
        options = self.find_grasps(block_name, start_q=start_q)
        if not options:
            raise GraspPlanningFailure(
                f"No feasible grasp for {block_name}: " +
                "; ".join(self.last_rejections[:4])
            )
        return options[0]
