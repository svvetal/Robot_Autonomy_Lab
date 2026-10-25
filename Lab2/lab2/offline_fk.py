"""Independent analytic FK from starter MJCF for *reachability preflight* only.

This deliberately does not assert dynamic feasibility, collision-freedom, or simulation
success; MuJoCo remains the authoritative kinematics/collision engine at runtime.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from .config import HOME, TOOL_SITE_OFFSET
from .geometry import make_transform, pose_error


class XmlKinematics:
    def __init__(self, mjcf=None):
        if mjcf is None:
            mjcf = Path(__file__).resolve().parents[1] / 'franka_emika_panda' / 'panda_torque_table.xml'
        root = ET.parse(mjcf).getroot()
        link = next(b for b in root.iter('body') if b.get('name') == 'link0')
        self.links = []
        while True:
            self.links.append(link)
            if link.get('name') == 'hand':
                break
            child = link.find('body')
            if child is None:
                raise ValueError('Franka kinematic chain broken')
            link = child
        # Values from robot model, not hardcoded measured positions.
        self.limits = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628],
                                [-2.8973, 2.8973], [-3.0718, -0.0698],
                                [-2.8973, 2.8973], [-0.0175, 3.7525],
                                [-2.8973, 2.8973]])

    def forward(self, q):
        H = np.eye(4)
        idx = 0
        for body in self.links:
            pos = np.fromstring(body.get('pos', '0 0 0'), sep=' ')
            quat = np.fromstring(body.get('quat', '1 0 0 0'), sep=' ')
            rot = Rotation.from_quat(np.r_[quat[1:], quat[0]]).as_matrix()
            H = H @ make_transform(pos, rot)
            j = body.find('joint')
            if j is not None and j.get('name', '').startswith('joint'):
                # All Panda joints in supplied class default rotate about Z.
                H = H @ make_transform((0, 0, 0),
                                       Rotation.from_rotvec([0, 0, q[idx]]).as_matrix())
                idx += 1
        H = H @ make_transform((0, 0, TOOL_SITE_OFFSET), np.eye(3))
        return H

    def pose_reachable(self, xyz, rotation, seed=13, attempts=14):
        target = make_transform(xyz, rotation)
        rng = np.random.default_rng(seed)
        lo, hi = self.limits[:, 0]+0.035, self.limits[:, 1]-0.035
        def fun(q):
            e = pose_error(target, self.forward(q))
            return np.r_[e[:3], 0.22*e[3:]]
        best = (np.inf, None)
        for k in range(attempts):
            q0 = np.clip(HOME, lo, hi) if k == 0 else rng.uniform(lo, hi)
            sol = least_squares(fun, q0, bounds=(lo, hi), max_nfev=100)
            err = fun(sol.x)
            score = (float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:])/0.22))
            if np.linalg.norm(err) < best[0]:
                best = (float(np.linalg.norm(err)), score)
            if score[0] < 0.008 and score[1] < 0.14:
                return True, score
        return False, best[1]
