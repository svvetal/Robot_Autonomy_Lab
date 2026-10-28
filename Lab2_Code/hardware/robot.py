"""FrankaPy joint, pose and gripper interface."""
import numpy as np

class HardwareError(RuntimeError):
    pass

class FrankaRobot:
    def __init__(self, arm=None):
        if arm is None:
            from frankapy import FrankaArm
            arm = FrankaArm()
        self.arm = arm

    def joints(self):
        q = np.asarray(self.arm.get_joints(),float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise HardwareError('Invalid joint sensor values')
        return q

    def tool_pose(self):
        pose = self.arm.get_pose()
        p = np.asarray(pose.translation,float)
        R = np.asarray(pose.rotation,float)
        if p.shape != (3,) or R.shape != (3,3) or not np.isfinite(p).all() or not np.isfinite(R).all():
            raise HardwareError('Invalid frankapy pose')
        H = np.eye(4); H[:3,:3]=R; H[:3,3]=p
        return H

    def gripper_width(self):
        w=float(self.arm.get_gripper_width())
        if not np.isfinite(w) or not 0 <= w <= .085:
            raise HardwareError('Invalid gripper width')
        return w

    def open_gripper(self):
        self.arm.open_gripper(block=True)
        width=self.gripper_width()
        if width < .055:
            raise HardwareError('Gripper failed to open: {:.3f} m'.format(width))
        return width

    def grasp_block(self,cube_width=.040,force=10.):
        self.arm.goto_gripper(width=cube_width,grasp=True,speed=.02,
                              force=force,epsilon_inner=.006,epsilon_outer=.006,
                              block=True,ignore_errors=False)
        width=self.gripper_width()
        if not .032 <= width <= .048:
            raise HardwareError('Unexpected 40 mm block grasp width: {:.3f} m'.format(width))

        # Gripper width alone does not verify a stable grasp.
        return width

    def move_joints(self,q,duration):
        q=np.asarray(q,float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise HardwareError('Invalid joint command')
        self.arm.goto_joints(q,duration=float(duration),use_impedance=False,
                             block=True,ignore_errors=False,ignore_virtual_walls=False)

    def stop(self):
        try: self.arm.stop_skill()
        except Exception: pass

def pose_mismatch(expected,actual):
    distance=float(np.linalg.norm(expected[:3,3]-actual[:3,3]))
    C=expected[:3,:3] @ actual[:3,:3].T
    angle=float(np.arccos(np.clip((np.trace(C)-1.)/2.,-1.,1.)))
    return distance,angle
