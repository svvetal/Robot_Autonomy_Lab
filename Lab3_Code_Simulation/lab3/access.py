"""Fast local preconditions for the assignment's vertical, face-aligned top grasps.

The mesh is the original Panda ``finger_0.obj`` convex collision hull, plus
the five original fingertip collision boxes.  A horizontal projection of the
part of each hull that sweeps through block height gives an exact 2-D test
against upright, axis-aligned cubes.  Projecting the aperture sweep together
with the vertical sweep is conservative: it also checks intermediate openings
at intermediate heights.  It never makes the fingers thinner or moves cells.

This is a local gripper test, not an arm reachability certificate.  MuJoCo/IK
still validates each complete transfer before execution.  A failed result is
relative to the finite, face-aligned top-grasp library below; it is not a
proof that every possible robot manipulation is impossible.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial import ConvexHull

from .config import (
    BLOCK_HALF, GRID_CELLS, GRIP_CLOSE, STAGING_X, STAGING_RED_Y,
    STAGING_BLUE_Y, TOOL_SITE_OFFSET, grid_position,
)

LOCATIONS = tuple(GRID_CELLS) + tuple(f"R{i}" for i in range(4)) + tuple(
    f"B{i}" for i in range(4)
)
_INDEX = {name: index for index, name in enumerate(LOCATIONS)}
_XY = tuple(
    tuple(grid_position(int(name[1]), int(name[2]))[:2])
    if name in GRID_CELLS else
    (STAGING_X[int(name[1])], STAGING_RED_Y if name[0] == "R" else STAGING_BLUE_Y)
    for name in LOCATIONS
)
TOP_YAWS_DEG = (0.0, 90.0, 180.0, 270.0)
TOP_HEIGHT_OFFSETS_M = (0.0, 0.007, 0.011)
PICKUP_OPENING_M = 0.024
RELEASE_OPENING_M = 0.0215
# Numerical tolerance only; existing collision geometry is used unchanged.
_EPS = 1e-9
_FINGER_BASE_Z = 0.0584
_PADS = (
    ((0.0085, 0.004, 0.0085), (0.0, 0.0055, 0.0445)),
    ((0.003, 0.002, 0.003), (0.0055, 0.002, 0.0500)),
    ((0.003, 0.002, 0.003), (-0.0055, 0.002, 0.0500)),
    ((0.003, 0.002, 0.0035), (0.0055, 0.002, 0.0395)),
    ((0.003, 0.002, 0.0035), (-0.0055, 0.002, 0.0395)),
)


@dataclass(frozen=True)
class AccessOption:
    yaw_deg: float
    height_offset_m: float
    approach_opening_m: float = PICKUP_OPENING_M
    release_opening_m: float = RELEASE_OPENING_M


def occupancy_mask(occupied_locations: Iterable[str] | int) -> int:
    """Encode occupied named cells; color and block identity do not matter."""
    if isinstance(occupied_locations, int):
        if occupied_locations < 0 or occupied_locations >> len(LOCATIONS):
            raise ValueError("Invalid location occupancy mask")
        return occupied_locations
    mask = 0
    for location in occupied_locations:
        if location not in _INDEX:
            raise ValueError(f"Unknown location {location!r}")
        mask |= 1 << _INDEX[location]
    return mask


@lru_cache(maxsize=1)
def _mesh() -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    path = Path(__file__).resolve().parents[1] / "franka_emika_panda/assets/finger_0.obj"
    vertices = np.array([
        [float(value) for value in line.split()[1:4]]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("v ")
    ])
    hull = ConvexHull(vertices)
    edges = tuple(sorted({
        tuple(sorted((int(triangle[i]), int(triangle[j]))))
        for triangle in hull.simplices for i, j in ((0, 1), (1, 2), (2, 0))
    }))
    return vertices, edges


@lru_cache(maxsize=32)
def _local_swept_sections(height_offset_m: float) -> tuple[np.ndarray, ...]:
    """Project original finger pieces that can enter the block-height slab.

    In top orientation local +z points down. The finger base is 44.6 mm
    above TCP; raising TCP increases the clipping plane by the same amount.
    Approach from above can only encounter points below the block top, so
    clipping the convex hull at that plane and projecting is exact.
    """
    vertices, edges = _mesh()
    cut_z = TOOL_SITE_OFFSET - _FINGER_BASE_Z + height_offset_m - BLOCK_HALF
    points = list(vertices[vertices[:, 2] >= cut_z - _EPS])
    for i, j in edges:
        a, b = vertices[i], vertices[j]
        if (a[2] - cut_z) * (b[2] - cut_z) < 0.0:
            points.append(a + (b - a) * ((cut_z - a[2]) / (b[2] - a[2])))
    result = []
    if len(points) >= 3:
        xy = np.asarray(points)[:, :2]
        result.append(xy[ConvexHull(xy).vertices])
    for halfsize, center in _PADS:
        sx, sy, sz = halfsize
        x, y, z = center
        if z + sz >= cut_z - _EPS:
            result.append(np.array([
                [x - sx, y - sy], [x + sx, y - sy],
                [x + sx, y + sy], [x - sx, y + sy],
            ]))
    return tuple(result)


@lru_cache(maxsize=128)
def _swept_polygons(yaw: float, height: float, phase: str) -> tuple[np.ndarray, ...]:
    upper_opening = PICKUP_OPENING_M if phase == "pick" else RELEASE_OPENING_M
    a = np.deg2rad(yaw)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    result = []
    for polygon in _local_swept_sections(height):
        # Minkowski sum with the aperture interval checks the entire opening
        # or closing sweep. Right finger has the original 180-degree rotation.
        swept = np.concatenate([
            polygon + np.array([0.0, opening])
            for opening in (GRIP_CLOSE, upper_opening)
        ])
        swept = swept[ConvexHull(swept).vertices]
        for side in (-1.0, 1.0):
            xy = side * swept.copy()
            xy[:, 1] *= -1.0  # TOP_R's horizontal part is diag(+1, -1).
            result.append(xy @ rotation.T)
    return tuple(result)


def _overlaps_square(polygon: np.ndarray, center: tuple[float, float]) -> bool:
    """Separating-axis test; exact for each convex polygon and block square."""
    edges = np.roll(polygon, -1, axis=0) - polygon
    axes = np.concatenate((np.eye(2), np.column_stack((-edges[:, 1], edges[:, 0]))))
    c = np.asarray(center)
    for axis in axes:
        if float(axis @ axis) < 1e-18:
            continue
        projected = polygon @ axis
        middle = float(c @ axis)
        radius = BLOCK_HALF * float(np.abs(axis).sum())
        if projected.max() <= middle - radius + _EPS:
            return False
        if projected.min() >= middle + radius - _EPS:
            return False
    return True


@lru_cache(maxsize=2048)
def _blocker_masks(index: int, phase: str) -> tuple[tuple[AccessOption, int], ...]:
    center = np.asarray(_XY[index])
    result = []
    # 0 and 180 have identical local clearance, as do 90 and 270. Return
    # both wrist orientations because the whole-arm IK may distinguish them.
    for height, yaw in product(TOP_HEIGHT_OFFSETS_M, TOP_YAWS_DEG):
        polygons = _swept_polygons(yaw % 180.0, height, phase)
        blocked = 0
        for j, xy in enumerate(_XY):
            if j == index:
                continue
            relative = np.asarray(xy) - center
            if np.linalg.norm(relative) > 0.13:
                continue
            if any(_overlaps_square(polygon, tuple(relative)) for polygon in polygons):
                blocked |= 1 << j
        result.append((AccessOption(yaw, height), blocked))
    return tuple(result)


@lru_cache(maxsize=100_000)
def _options(index: int, mask: int, phase: str) -> tuple[AccessOption, ...]:
    return tuple(option for option, blockers in _blocker_masks(index, phase)
                 if not mask & blockers)


def top_access_options(
    location: str, occupied_locations: Iterable[str] | int, *, phase: str = "pick",
) -> tuple[AccessOption, ...]:
    """Collision-free local candidates, including the complete aperture sweep.

    The target cell may be included in ``occupied_locations``. For a transfer,
    remove the source before testing the destination with ``phase='place'``.
    """
    if location not in _INDEX:
        raise ValueError(f"Unknown location {location!r}")
    if phase not in ("pick", "place"):
        raise ValueError("phase must be 'pick' or 'place'")
    return _options(_INDEX[location], occupancy_mask(occupied_locations), phase)


def can_access(location: str, occupied_locations: Iterable[str] | int, *, phase: str = "pick") -> bool:
    return bool(top_access_options(location, occupied_locations, phase=phase))


def top_access_options_xyz(
    center_xyz: Sequence[float],
    obstacle_centers: Iterable[Sequence[float]] | Mapping[str, Sequence[float]],
    *, phase: str = "pick",
) -> tuple[AccessOption, ...]:
    """Evaluate measured centers without snapping them to symbolic locations.

    ``obstacle_centers`` must exclude the manipulated cube. Obstacles remain
    upright axis-aligned cubes; unlike the symbolic cache, their centers may
    differ slightly in height. All dimensions are meters.
    """
    if phase not in ("pick", "place"):
        raise ValueError("phase must be 'pick' or 'place'")
    center = np.asarray(center_xyz, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center_xyz must contain three finite values")
    values = obstacle_centers.values() if isinstance(obstacle_centers, Mapping) else obstacle_centers
    obstacles = [np.asarray(xyz, dtype=float) for xyz in values]
    if any(xyz.shape != (3,) or not np.isfinite(xyz).all() for xyz in obstacles):
        raise ValueError("Obstacle centers must contain three finite values")
    nearby = [xyz - center for xyz in obstacles
              if np.linalg.norm(xyz[:2] - center[:2]) <= 0.13]
    result = []
    for height, yaw in product(TOP_HEIGHT_OFFSETS_M, TOP_YAWS_DEG):
        for relative in nearby:
            # Different block heights change the local clipping plane.
            effective_height = round(height - float(relative[2]), 9)
            polygons = _swept_polygons(yaw % 180.0, effective_height, phase)
            if any(_overlaps_square(polygon, tuple(relative[:2])) for polygon in polygons):
                break
        else:
            result.append(AccessOption(yaw, height))
    return tuple(result)


@lru_cache(maxsize=100_000)
def _peel_result(mask: int) -> tuple[tuple[str, ...], int]:
    remaining = mask
    order = []
    while remaining:
        for i, location in enumerate(LOCATIONS):
            if remaining & (1 << i) and _options(i, remaining, "pick"):
                remaining &= ~(1 << i)
                order.append(location)
                break
        else:
            return tuple(order), remaining
    return tuple(order), 0


def peel_order(occupied_locations: Iterable[str] | int) -> tuple[str, ...] | None:
    """Return a removal certificate, or None for a mutually blocked cluster.

    Greedy removal is complete for this local model: removing an obstacle
    cannot invalidate any other grasp. A block need not be immediately
    accessible if another block can first be removed, as in the box pattern.
    """
    order, blocked = _peel_result(occupancy_mask(occupied_locations))
    return order if not blocked else None


def blocked_core(occupied_locations: Iterable[str] | int) -> tuple[str, ...]:
    """Cells left mutually blocked after all locally removable cubes peel away."""
    _, remaining = _peel_result(occupancy_mask(occupied_locations))
    return tuple(location for i, location in enumerate(LOCATIONS)
                 if remaining & (1 << i))


def is_recoverable(occupied_locations: Iterable[str] | int) -> bool:
    return peel_order(occupied_locations) is not None


def all_accessible(occupied_locations: Iterable[str] | int) -> bool:
    """Diagnostic only; this is stronger than the assignment requires."""
    mask = occupancy_mask(occupied_locations)
    return all(_options(i, mask, "pick") for i in range(len(LOCATIONS))
               if mask & (1 << i))
