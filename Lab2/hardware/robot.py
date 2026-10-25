"""Small, replaceable frankapy adapter: the only module that commands hardware."""
import numpy as np


class HardwareError(RuntimeError):
    pass


class FrankaRobot:
    def __init__(self, arm=None):
        if arm is None:
            # Lazy import: planning/testing never needs Franka ROS packages.
            from frankapy import FrankaArm
            arm = FrankaArm()
        self.arm = arm

    def joints(self):
        q = np.asarray(self.arm.get_joints(), dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise HardwareError('Invalid joint sensor values')
        return q

    def tool_pose(self):
        pose = self.arm.get_pose()
        p = np.asarray(pose.translation, float)
        R = np.asarray(pose.rotation, float)
        if p.shape != (3,) or R.shape != (3, 3):
            raise HardwareError('Unexpected frankapy tool pose shape')
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = p
        return H

    def gripper_width(self):
        w = float(self.arm.get_gripper_width())
        if not np.isfinite(w) or not 0 <= w <= .085:
            raise HardwareError(f'Unexpected gripper width reading: {w}')
        return w

    def open_gripper(self):
        self.arm.open_gripper(block=True)
        width = self.gripper_width()
        if width < .055:
            raise HardwareError(f'Gripper did not open enough ({width:.3f} m)')
        return width

    def grasp_block(self, cube_width=.040, force=10.0):
        # Use grasp mode and a force limit instead of blind full close.
        self.arm.goto_gripper(width=cube_width, grasp=True, speed=.02,
                              force=force, epsilon_inner=.006,
                              epsilon_outer=.006, block=True,
                              ignore_errors=False)
        width = self.gripper_width()
        if not .032 <= width <= .048:
            raise HardwareError(f'Unexpected grasp width {width:.3f}m, cube expected 0.040m')
        # Width alone cannot confirm a secure grasp; operator inspection required.
        return width

    def move_joints(self, q, duration):
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise HardwareError('Invalid commanded configuration')
        self.arm.goto_joints(q, duration=float(duration), use_impedance=False,
                             block=True, ignore_errors=False,
                             ignore_virtual_walls=False)

    def stop(self):
        try:
            self.arm.stop_skill()
        except Exception:
            pass


def pose_mismatch(expected, actual):
    distance = float(np.linalg.norm(expected[:3, 3]-actual[:3, 3]))
    C = expected[:3, :3] @ actual[:3, :3].T
    angle = float(np.arccos(np.clip((np.trace(C)-1.)/2., -1., 1.)))
    return distance, angle
