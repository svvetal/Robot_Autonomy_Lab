"""Endpoint-rest, continuous-velocity-and-acceleration quintic joint trajectories."""
import numpy as np


def quintic(q0, q1, t, duration):
    if duration <= 0:
        raise ValueError('duration must be > 0')
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    u = np.clip(t / duration, 0.0, 1.0)
    s = 10*u**3 - 15*u**4 + 6*u**5
    sd = (30*u**2 - 60*u**3 + 30*u**4) / duration
    sdd = (60*u - 180*u**2 + 120*u**3) / duration**2
    dq = q1-q0
    return q0 + s*dq, sd*dq, sdd*dq


def duration_for_edge(a, b, max_joint_speed=0.65, minimum=0.8):
    """Peak derivative of quintic s is 1.875; clamp actual peak joint speed."""
    return max(minimum, 1.875 * float(np.max(np.abs(np.asarray(b)-np.asarray(a)))) /
               max(max_joint_speed, 1e-8))


def path_validity(path, edge_checker):
    path = [np.asarray(q) for q in path]
    return len(path) >= 1 and all(edge_checker(a, b) for a, b in zip(path, path[1:]))
