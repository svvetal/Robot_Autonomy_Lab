"""Numerical pose helpers independent of MuJoCo (unit-testable)."""
import numpy as np
from scipy.spatial.transform import Rotation


def make_transform(position, rotation):
    H = np.eye(4)
    H[:3, :3] = np.asarray(rotation, dtype=float)
    H[:3, 3] = np.asarray(position, dtype=float)
    return H


def transform_inv(H):
    T = np.eye(4)
    T[:3, :3] = H[:3, :3].T
    T[:3, 3] = -H[:3, :3].T @ H[:3, 3]
    return T


def xyzw_to_wxyz(q):
    return np.r_[q[3], q[:3]]


def wxyz_to_xyzw(q):
    return np.r_[q[1:], q[0]]


def matrix_to_mjquat(R):
    return xyzw_to_wxyz(Rotation.from_matrix(R).as_quat())


def quat_to_matrix(quat):
    return Rotation.from_quat(wxyz_to_xyzw(np.asarray(quat))).as_matrix()


def rotation_residual(target, current):
    """World-frame axis-angle rotation taking *current* toward *target*."""
    return Rotation.from_matrix(target @ current.T).as_rotvec()


def pose_error(goal, current):
    return np.r_[goal[:3, 3] - current[:3, 3],
                 rotation_residual(goal[:3, :3], current[:3, :3])]
