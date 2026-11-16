"""Time parameterisation of planned moves -- IDENTICAL to the simulator playback.

lab3/playback.py plays two kinds of Move:
  * RRT moves: one quintic (min-jerk) profile per joint-space segment,
  * straight-line moves (smooth=True): one quintic profile along the whole
    joint polyline, parameterised by Euclidean arc length.
This module re-implements exactly those formulas in plain numpy (Python 3.8,
no MuJoCo) so the real robot is streamed the same q(t) the simulation played.
tests/test_trajectory_matches_sim.py checks the two agree.
"""
import numpy as np


def _quintic_s(u):
    u = np.clip(u, 0.0, 1.0)
    return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5


def total_duration(move, time_scale=1.0):
    return float(sum(move['durations'])) * float(time_scale)


def q_at(move, t, time_scale=1.0):
    """Joint configuration of `move` at time t (seconds, after time scaling)."""
    path = np.asarray(move['path'], float)
    durs = np.asarray(move['durations'], float) * float(time_scale)
    if t >= float(np.sum(durs)) - 1e-12:
        return path[-1].copy()                 # exactly the goal, as the simulator does
    if move.get('smooth', False):
        T = float(durs[0])
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum = np.r_[0.0, np.cumsum(seg)]
        arc = _quintic_s(t / T) * cum[-1]
        k = min(int(np.searchsorted(cum, arc, side='right')) - 1, len(seg) - 1)
        k = max(k, 0)
        u = 0.0 if seg[k] < 1e-12 else (arc - cum[k]) / seg[k]
        return path[k] + np.clip(u, 0.0, 1.0) * (path[k + 1] - path[k])
    t0 = 0.0
    for i, d in enumerate(durs):
        if t <= t0 + d or i == len(durs) - 1:
            if (t - t0) / d >= 1.0:
                return path[i + 1].copy()       # exact segment end, as the simulator
            return path[i] + _quintic_s((t - t0) / d) * (path[i + 1] - path[i])
        t0 += d
    return path[-1]


def sample_move(move, dt, time_scale=1.0):
    """Samples q(0), q(dt), ..., q(T) (last sample exactly the goal)."""
    T = total_duration(move, time_scale)
    n = max(2, int(np.ceil(T / dt)) + 1)
    ts = np.linspace(0.0, T, n)
    return ts, np.array([q_at(move, t, time_scale) for t in ts])


def max_joint_speed(move, time_scale=1.0, dt=0.005):
    ts, qs = sample_move(move, dt, time_scale)
    return float(np.max(np.abs(np.diff(qs, axis=0)) / np.diff(ts)[:, None]))
