"""A MuJoCo stand-in for FrankaRobot, for dry runs of the hardware executor.

It implements the same methods execute.py calls (joints, tool_pose, stream,
move_joints, grasp_cube, open_to_pregrasp, ...), so the full hardware code
path -- plan loading, safety checks, streaming, gripper logic -- can be
exercised without a robot. Grasping attaches the cube between the fingers
(nothing is told which cube to take), so a wrong plan shows up as a miss.
"""
import time

import numpy as np

from .robot import HardwareError
from .simlink import use_sim
from .spec import PREGRASP_WIDTH, CUBE_WIDTH


class SimRobot(object):
    def __init__(self, seed=13, viewer=False, control='kinematic', realtime=False):
        use_sim()
        from lab3.config import Settings
        from lab3.simulator import LabSim
        self.sim = LabSim(Settings(seed=seed), control=control)
        self.realtime = realtime
        self.viewer = None
        if viewer:
            from mujoco import viewer as mjv
            self.viewer = mjv.launch_passive(self.sim.model, self.sim.data)
            from lab3.playback import Player
            Player._aim(self.viewer.cam)
        self.width = PREGRASP_WIDTH
        self.commands = []

    def _sync(self, dt):
        if self.viewer is not None:
            self.viewer.sync()
        if self.realtime:
            time.sleep(dt)

    def joints(self):
        return self.sim.arm_q()

    def tool_pose(self):
        self.sim.mj.mj_forward(self.sim.model, self.sim.data)
        return self.sim.tool_pose()

    def gripper_width(self):
        return self.width

    def open_to_pregrasp(self):
        if self.sim.held is not None:
            name = self.sim.held
            H = self.sim.tool_pose() @ self.sim.attachment
            from lab3.layout import BLOCK_Z
            self.sim.release(np.r_[H[:2, 3], BLOCK_Z])
            self.commands.append(('release', name))
        self.width = PREGRASP_WIDTH
        return self.width

    def grasp_cube(self, force=10.0):
        tool = self.tool_pose()[:3, 3]
        best = min(self.sim.blocks, key=lambda b: np.linalg.norm(
            self.sim.block_pose(b)[:3, 3] - tool))
        if np.linalg.norm(self.sim.block_pose(best)[:3, 3] - tool) > 0.01:
            self.width = 0.0
            raise HardwareError('Grasp width 0.000 m: no cube between the fingers')
        self.sim.attach(best)
        self.commands.append(('grasp', best))
        self.width = CUBE_WIDTH
        return self.width

    def stream(self, qs, dt):
        qs = np.asarray(qs, float)
        for a, b in zip(qs, qs[1:]):
            self.sim.advance(b, (b - a) / dt, dt)
            self._sync(dt)
        self.commands.append(('stream', len(qs)))

    def move_tool_linear(self, H, duration, path=None):
        if path is None:
            raise HardwareError('SimRobot.move_tool_linear needs the planned joint path')
        for q in path[1:]:
            self.sim.advance(np.asarray(q, float), np.zeros(7), 0.02)
            self._sync(0.02)
        self.commands.append(('goto_pose', duration))

    def move_joints(self, q, duration):
        q0, q = self.sim.arm_q(), np.asarray(q, float)
        n = max(2, int(duration / 0.02))
        for k in range(1, n + 1):
            u = k / n
            s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            self.sim.advance(q0 + s * (q - q0), np.zeros(7), 0.02)
            self._sync(0.02)
        self.commands.append(('goto_joints', duration))

    def arrange(self, occupancy, cells):
        """Place the cubes as the operator would before a --start/--only run."""
        from lab3.geometry import make_transform
        for cell, block in occupancy.items():
            if block is not None:
                self.sim.set_block(self.sim.data, block,
                                   make_transform(np.asarray(cells[cell], float), np.eye(3)))
        self.sim.mj.mj_forward(self.sim.model, self.sim.data)

    def guide(self, seconds):
        pass

    def stop(self):
        pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
