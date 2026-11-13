from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

@dataclass(frozen=True)
class GraspQuality:
    min_singular: float
    volume: float
    force_closure_approx: bool
    contact_alignment: float
    score: float
    grasp_matrix: np.ndarray

def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("Degenerate contact direction")
    return v / n

def _orthonormal_tangents(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([0., 0., 1.])
    if abs(float(np.dot(n, reference))) > 0.9:
        reference = np.array([1., 0., 0.])
    t1 = _unit(np.cross(n, reference))
    return t1, _unit(np.cross(n, t1))

def _positive_span(G: np.ndarray, *, threshold: float = 1e-7) -> bool:

    if np.linalg.matrix_rank(G, tol=1e-7) < 6:
        return False
    m = G.shape[1]
    Aeq = np.zeros((7, m + 1))
    Aeq[:6, :m] = G
    Aeq[6, :m] = 1.0
    beq = np.array([0., 0., 0., 0., 0., 0., 1.])
    Aub = np.hstack([-np.eye(m), np.ones((m, 1))])
    cost = np.zeros(m + 1)
    cost[-1] = -1.0
    result = linprog(cost, A_ub=Aub, b_ub=np.zeros(m),
                     A_eq=Aeq, b_eq=beq,
                     bounds=[(0., None)] * (m + 1), method="highs")
    return bool(result.success and result.x[-1] > threshold)

def evaluate_cube_grasp(
    cube_rotation: np.ndarray,
    jaw_direction_world: np.ndarray,
    *,
    half_width_m: float = 0.020,
    friction_coefficient: float = 0.6,
    torsional_radius_m: float = 0.004,
    cone_edges: int = 8,
) -> GraspQuality:

    R = np.asarray(cube_rotation, dtype=float)
    if R.shape != (3, 3):
        raise ValueError("cube_rotation must be 3x3")
    jaw = _unit(jaw_direction_world)
    axes = (R[:, 0], R[:, 1])
    axis = _unit(max(axes, key=lambda a: abs(float(np.dot(jaw, a)))))
    alignment = abs(float(np.dot(jaw, axis)))
    if not (0 < friction_coefficient and half_width_m > 0 and
            torsional_radius_m >= 0 and cone_edges >= 4):
        raise ValueError("Invalid friction/contact geometry")

    rays = []
    for side in (-1.0, 1.0):
        p = side * half_width_m * axis
        n = -side * axis
        t1, t2 = _orthonormal_tangents(n)
        for k in range(cone_edges):
            theta = 2 * np.pi * k / cone_edges
            f = _unit(n + friction_coefficient * (
                np.cos(theta) * t1 + np.sin(theta) * t2))
            for twist_sign in (-1.0, 1.0):
                torque = (np.cross(p, f)
                          + twist_sign * torsional_radius_m * n)
                rays.append(np.r_[f, torque / half_width_m])
    G = np.stack(rays, axis=1)
    sv = np.linalg.svd(G, compute_uv=False)
    smallest = float(sv[-1])
    logdet = float(np.linalg.slogdet(G @ G.T)[1])
    volume = float(np.exp(0.5 * logdet)) if np.isfinite(logdet) else 0.0
    closure = _positive_span(G)

    quality_score = alignment * (
        0.7 * np.log1p(smallest) + 0.3 * np.log1p(volume))
    return GraspQuality(
        min_singular=smallest,
        volume=volume,
        force_closure_approx=closure,
        contact_alignment=alignment,
        score=float(quality_score),
        grasp_matrix=G,
    )