"""FrankaPy interface (robot PC, Python 3.8, no MuJoCo).

Two ways to play a planned move:
  * stream()     : dense q(t) streamed to frankapy's dynamic joint skill
                   (frankapy/examples/run_dynamic_joints.py pattern). Reproduces
                   the simulator's continuous motion, including the jitter-free
                   straight-line descents.
  * move_joints(): one goto_joints per waypoint (Lab 2 behaviour). Stops at
                   every waypoint; use only if dynamic skills misbehave.
"""
import time

import numpy as np

from .spec import PREGRASP_WIDTH, CUBE_WIDTH, GRASP_WIDTH_RANGE


class HardwareError(RuntimeError):
    pass


class FrankaRobot:
    ARRIVED_RAD = 0.008     # same as SafetyLimits.settle_joint_rad

    def __init__(self, arm=None):
        if arm is None:
            from frankapy import FrankaArm
            arm = FrankaArm()            # also initialises the ROS node
        self.arm = arm
        self._pub = None

    # ------------------------------------------------------------ sensing
    def joints(self):
        q = np.asarray(self.arm.get_joints(), float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise HardwareError('Invalid joint sensor values')
        return q

    def tool_pose(self):
        pose = self.arm.get_pose()
        H = np.eye(4)
        H[:3, :3] = np.asarray(pose.rotation, float)
        H[:3, 3] = np.asarray(pose.translation, float)
        if not np.isfinite(H).all():
            raise HardwareError('Invalid frankapy pose')
        return H

    def gripper_width(self):
        w = float(self.arm.get_gripper_width())
        if not np.isfinite(w) or not 0.0 <= w <= 0.085:
            raise HardwareError('Invalid gripper width')
        return w

    # ------------------------------------------------------------ gripper
    def open_to_pregrasp(self):
        """Open to 6 cm only -- a full 8 cm opening hits neighbouring cubes."""
        self.arm.goto_gripper(PREGRASP_WIDTH, grasp=False, speed=0.04, block=True)
        w = self.gripper_width()
        if abs(w - PREGRASP_WIDTH) > 0.006:
            raise HardwareError('Gripper did not open to {:.3f} m (got {:.3f})'.format(
                PREGRASP_WIDTH, w))
        return w

    def grasp_cube(self, force=10.0):
        self.arm.goto_gripper(CUBE_WIDTH, grasp=True, speed=0.02, force=force,
                              epsilon_inner=0.008, epsilon_outer=0.008, block=True,
                              ignore_errors=False)
        w = self.gripper_width()
        lo, hi = GRASP_WIDTH_RANGE
        if not lo <= w <= hi:
            raise HardwareError('Grasp width {:.3f} m outside {}-{} m: cube missed?'.format(w, lo, hi))
        return w

    # ------------------------------------------------------------ arm
    def move_joints(self, q, duration):
        q = np.asarray(q, float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise HardwareError('Invalid joint command')
        t0 = time.time()
        self.arm.goto_joints(q.tolist(), duration=float(duration), use_impedance=False,
                             block=True, ignore_errors=False, ignore_virtual_walls=False)
        # On this lab PC, goto_joints can return long before the robot finishes: the action
        # server ends the skill on a franka_interface 'not ready' flicker while the robot keeps
        # executing it (seen 2026-11-16: returned after 0.8 s of a 9 s move; stop_skill() then
        # could not stop it). Never return -- and so never send the next command -- until the
        # arm has actually arrived, or the full commanded duration + 2 s has passed.
        deadline = t0 + float(duration) + 2.0
        while time.time() < deadline:
            if float(np.max(np.abs(self.joints() - q))) < self.ARRIVED_RAD:
                break
            time.sleep(0.05)

    def move_tool_linear(self, H, duration, path=None):
        """Straight-line tool motion to pose H with ONE min-jerk profile (frankapy goto_pose).

        Cartesian IMPEDANCE controller (frankapy's default): the min-jerk target moves on the
        straight line and the tool is pulled toward it, so any lag is ALONG the line.
        (use_impedance=False, libfranka's internal Cartesian controller, was rejected on
        2026-11-16 with cartesian_motion_generator_joint_acceleration_discontinuity.)
        `path` is only used by SimRobot.
        """
        from autolab_core import RigidTransform
        H = np.asarray(H, float)
        if H.shape != (4, 4) or not np.isfinite(H).all():
            raise HardwareError('Invalid Cartesian target')
        T = RigidTransform(rotation=H[:3, :3], translation=H[:3, 3],
                           from_frame='franka_tool', to_frame='world')
        self.arm.goto_pose(T, duration=float(duration), use_impedance=True, block=True,
                           ignore_errors=False, ignore_virtual_walls=False)

    def stream(self, qs, dt):
        """Stream joint targets qs[0..N] at period dt with the dynamic joint skill."""
        import rospy
        from frankapy import FrankaConstants as FC, SensorDataMessageType
        from frankapy.proto import JointPositionSensorMessage, ShouldTerminateSensorMessage
        from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
        from franka_interface_msgs.msg import SensorDataGroup

        qs = np.asarray(qs, float)
        if qs.ndim != 2 or qs.shape[1] != 7 or len(qs) < 3 or not np.isfinite(qs).all():
            raise HardwareError('Invalid streamed trajectory')
        if self._pub is None:
            self._pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup,
                                        queue_size=1000)
            rospy.sleep(0.5)                       # let the publisher connect
        T = dt * (len(qs) - 1)
        rate = rospy.Rate(1.0 / dt)
        # Start the dynamic skill at the second sample; keep it alive past the end.
        # block=False: a blocking dynamic skill would wait for the stream it needs.
        self.arm.goto_joints(qs[1].tolist(), duration=T, dynamic=True, buffer_time=T + 10.0,
                             block=False, ignore_virtual_walls=False)
        t0 = rospy.Time.now().to_time()
        for i in range(2, len(qs)):
            msg = JointPositionSensorMessage(id=i, timestamp=rospy.Time.now().to_time() - t0,
                                             joints=qs[i].tolist())
            self._pub.publish(make_sensor_group_msg(
                trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                    msg, SensorDataMessageType.JOINT_POSITION)))
            rate.sleep()
        # Hold the last target briefly so the controller settles, then terminate.
        rospy.sleep(0.3)
        term = ShouldTerminateSensorMessage(timestamp=rospy.Time.now().to_time() - t0,
                                            should_terminate=True)
        self._pub.publish(make_sensor_group_msg(
            termination_handler_sensor_msg=sensor_proto2ros_msg(
                term, SensorDataMessageType.SHOULD_TERMINATE)))
        if hasattr(self.arm, 'wait_for_skill'):
            self.arm.wait_for_skill()
        else:
            time.sleep(0.5)

    def guide(self, seconds):
        """Hand-guiding (gravity compensation) for calibration -- see Lab 1."""
        self.arm.run_guide_mode(float(seconds), block=True)

    def stop(self):
        try:
            self.arm.stop_skill()
        except Exception:
            pass


def pose_mismatch(expected, actual):
    distance = float(np.linalg.norm(np.asarray(expected)[:3, 3] - np.asarray(actual)[:3, 3]))
    C = np.asarray(expected)[:3, :3] @ np.asarray(actual)[:3, :3].T
    angle = float(np.arccos(np.clip((np.trace(C) - 1.0) / 2.0, -1.0, 1.0)))
    return distance, angle
