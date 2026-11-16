"""Bidirectional RRT-Connect joint-space planner."""
from dataclasses import dataclass
import numpy as np

class PlanFailure(RuntimeError):
    pass

@dataclass
class _Node:
    q: np.ndarray
    parent: int

def interpolate_edge(a, b, resolution):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(1, int(np.ceil(np.max(np.abs(b-a)) / resolution)))
    for s in np.linspace(0., 1., n + 1):
        yield a + (b-a)*s

class JointRRT:
    def __init__(self, valid, limits, rng, step=0.17, resolution=0.055,
                 max_iter=3000, goal_bias=0.18):
        self.valid = valid
        self.limits = np.asarray(limits, float)
        self.rng = rng
        self.step, self.resolution = step, resolution
        self.max_iter, self.goal_bias = max_iter, goal_bias

    def edge_free(self, a, b):
        return all(self.valid(q) for q in interpolate_edge(a, b, self.resolution))

    def _extend(self, tree, sample):
        nearest_idx = min(range(len(tree)), key=lambda k: np.linalg.norm(tree[k].q - sample))
        old = tree[nearest_idx].q
        delta = sample-old
        dist = np.linalg.norm(delta)
        if dist < 1.e-9:
            return nearest_idx, True
        new = old + delta*min(1.0, self.step/dist)
        if not self.edge_free(old, new):
            return None, False
        tree.append(_Node(new, nearest_idx))
        return len(tree)-1, dist <= self.step

    def _connect(self, tree, goal):
        for _ in range(1000):
            idx, reached = self._extend(tree, goal)
            if idx is None or reached:
                return idx, reached
        return None, False

    @staticmethod
    def _path(tree, idx):
        out = []
        while idx >= 0:
            n = tree[idx]
            out.append(n.q.copy())
            idx = n.parent
        return out[::-1]

    def plan(self, start, goal):
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        if not self.valid(start):
            raise PlanFailure('RRT start is in collision')
        if not self.valid(goal):
            raise PlanFailure('RRT goal is in collision')
        if self.edge_free(start, goal):
            return [start.copy(), goal.copy()]
        trees = [[_Node(start.copy(), -1)], [_Node(goal.copy(), -1)]]
        for it in range(self.max_iter):
            primary = it % 2
            a, b = trees[primary], trees[1-primary]
            sample = goal if primary == 0 else start
            if self.rng.random() > self.goal_bias:
                sample = self.rng.uniform(self.limits[:, 0], self.limits[:, 1])
            idx, _ = self._extend(a, sample)
            if idx is None:
                continue
            jdx, reached = self._connect(b, a[idx].q)
            if reached and jdx is not None:
                pa = self._path(a, idx)
                pb = self._path(b, jdx)
                if primary == 0:
                    return pa + pb[-2::-1]
                return pb + pa[-2::-1]
        raise PlanFailure(f'RRT-Connect failed after {self.max_iter} iterations')

    def shortcut(self, path, attempts=120):
        result = [np.asarray(p, float).copy() for p in path]
        for _ in range(attempts):
            if len(result) <= 2:
                break
            i, j = sorted(self.rng.choice(len(result), size=2, replace=False))
            if j > i+1 and self.edge_free(result[i], result[j]):
                result[i+1:j] = []
        return result
