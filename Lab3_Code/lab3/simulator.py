"""MuJoCo state management and collision queries for N named cubes.

Refactor of the Lab 2 LabSim:
  * every cube is a named free body (red0..3, blue0..3);
  * one cube at a time can be `held` (kinematically attached to the tool);
  * collision checks cover robot<->world, robot<->every non-held cube,
    and held cube<->world/other cubes;
  * the planning copy (`scratch`) always shows the fingers at the width they
    will really have: GRIP_OPEN (pre-grasp) when empty, GRIP_CLOSE when holding.
"""
from contextlib import contextmanager
import numpy as np

from .config import HOME, Settings, GRIP_OPEN, GRIP_CLOSE
from .layout import BLOCK_Z
from .geometry import make_transform, matrix_to_mjquat, quat_to_matrix, transform_inv
from .layout import INITIAL_BLOCKS, cell_xyz
from .scene import prepare_scene, TOOL_SITE

ROBOT_PREFIXES = ('link', 'hand', 'left_finger', 'right_finger')
# Penetrations smaller than this are numerical contact noise, not collisions.
CONTACT_TOL = 0.0005
# A held cube may graze its support (table) at grasp/release by up to this much.
SUPPORT_TOL = 0.004
KP = np.array([120., 120., 100., 90., 60., 40., 30.])
KD = np.array([10., 10., 9., 8., 6., 5., 4.])
TAU_LIM = np.array([87., 87., 87., 87., 12., 12., 12.])


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def upright(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class LabSim:
    def __init__(self, cfg: Settings, *, control='kinematic', blocks=INITIAL_BLOCKS):
        import mujoco as mj
        self.mj, self.cfg, self.control = mj, cfg, control
        self.initial_blocks = list(blocks)
        self.model = mj.MjModel.from_xml_path(str(prepare_scene(blocks=blocks)))
        self.data = mj.MjData(self.model)
        self.scratch = mj.MjData(self.model)
        m = self.model

        jids = [self._id(mj.mjtObj.mjOBJ_JOINT, f'joint{k}') for k in range(1, 8)]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jids], int)
        self.vadr = np.array([m.jnt_dofadr[j] for j in jids], int)
        self.limits = np.asarray(m.jnt_range[jids], float).copy()
        self.limits[:, 0] += cfg.safe_joint_margin
        self.limits[:, 1] -= cfg.safe_joint_margin
        self.tool_id = self._id(mj.mjtObj.mjOBJ_SITE, TOOL_SITE)
        self.arm_act = np.array([self._id(mj.mjtObj.mjOBJ_ACTUATOR, f'actuator{k}')
                                 for k in range(1, 8)])
        self.gripper_act = self._id(mj.mjtObj.mjOBJ_ACTUATOR, 'actuator8')
        fj = [self._id(mj.mjtObj.mjOBJ_JOINT, f'finger_joint{k}') for k in (1, 2)]
        self.finger_qadr = [int(m.jnt_qposadr[j]) for j in fj]
        self.finger_vadr = [int(m.jnt_dofadr[j]) for j in fj]

        # Cubes: name -> (qpos addr, dof addr, geom id)
        self.blocks = {}
        for name, colour, _cell in self.initial_blocks:
            bid = self._id(mj.mjtObj.mjOBJ_BODY, name)
            jnt = m.body_jntadr[bid]
            if jnt < 0 or m.jnt_type[jnt] != mj.mjtJoint.mjJNT_FREE:
                raise ValueError(f'{name} must have a free joint')
            self.blocks[name] = (int(m.jnt_qposadr[jnt]), int(m.jnt_dofadr[jnt]),
                                 int(m.body_geomadr[bid]))
        self.colour = {name: colour for name, colour, _ in self.initial_blocks}
        self.geom_block = {g: name for name, (_, _, g) in self.blocks.items()}

        self.robot_geoms, self.world_geoms = set(), set()
        for g in range(m.ngeom):
            body = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ''
            if body.startswith(ROBOT_PREFIXES) and body != 'link0':
                self.robot_geoms.add(g)
            elif g not in self.geom_block:
                self.world_geoms.add(g)          # table, shelves, ground
        self.table_geom = int(m.body_geomadr[self._id(mj.mjtObj.mjOBJ_BODY, 'TablePlane')])

        self.rng = np.random.default_rng(cfg.seed)
        self.held = None          # name of the attached cube
        self.attachment = None    # tool -> cube transform
        self.reset()

    # ------------------------------------------------------------------ utils
    def _id(self, obj_type, name):
        k = self.mj.mj_name2id(self.model, obj_type, name)
        if k < 0:
            raise RuntimeError(f'Missing MuJoCo object {obj_type}: {name}')
        return int(k)

    def _gname(self, g):
        mj = self.mj
        return (mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, g) or
                mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY,
                              int(self.model.geom_bodyid[g])) or f'geom#{g}')

    @property
    def grip_width(self):
        return GRIP_CLOSE if self.held is not None else GRIP_OPEN

    def arm_q(self, d=None):
        return np.asarray((self.data if d is None else d).qpos[self.qadr], float).copy()

    def set_arm(self, d, q, qd=None):
        d.qpos[self.qadr] = np.asarray(q, float)
        d.qvel[self.vadr] = 0 if qd is None else qd

    def set_fingers(self, d, width):
        d.qpos[self.finger_qadr] = width
        d.qvel[self.finger_vadr] = 0

    def set_block(self, d, name, H):
        qa, va, _ = self.blocks[name]
        d.qpos[qa:qa + 3] = H[:3, 3]
        d.qpos[qa + 3:qa + 7] = matrix_to_mjquat(H[:3, :3])
        d.qvel[va:va + 6] = 0

    def block_pose(self, name, d=None):
        d = self.data if d is None else d
        qa = self.blocks[name][0]
        return make_transform(d.qpos[qa:qa + 3].copy(), quat_to_matrix(d.qpos[qa + 3:qa + 7]))

    def tool_pose(self, d=None):
        d = self.data if d is None else d
        return make_transform(d.site_xpos[self.tool_id].copy(),
                              d.site_xmat[self.tool_id].reshape(3, 3).copy())

    # ------------------------------------------------------------ planning
    def fk(self, q):
        """Tool pose at q - kinematics only (no collision detection), used by IK."""
        self.set_arm(self.scratch, q)
        self.mj.mj_kinematics(self.model, self.scratch)
        return self.tool_pose(self.scratch)

    def evaluate(self, q, carrying=False):
        """Pose the planning copy at q (cubes as in the live sim) and return the tool pose."""
        s = self.scratch
        self.set_arm(s, q)
        self.set_fingers(s, self.grip_width)
        for name, (qa, va, _) in self.blocks.items():
            if name != self.held:
                s.qpos[qa:qa + 7] = self.data.qpos[qa:qa + 7]
        self.mj.mj_forward(self.model, s)
        if carrying and self.held is not None:
            self.set_block(s, self.held, self.tool_pose(s) @ self.attachment)
            self.mj.mj_forward(self.model, s)
        return self.tool_pose(s)

    def collision_reasons(self, q, carrying=False, max_reports=8):
        q = np.asarray(q, float)
        if q.shape != (7,) or not np.isfinite(q).all():
            return ['nonfinite or incorrectly shaped joint configuration']
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            return ['joint configuration outside safety limits']
        # The held cube always moves with the tool, whatever the caller passed.
        carrying = self.held is not None
        self.evaluate(q, carrying=carrying)
        held_geom = self.blocks[self.held][2] if carrying else None
        issues = []
        s = self.scratch
        for con in s.contact[:s.ncon]:
            pen = -float(con.dist)
            if pen <= CONTACT_TOL:
                continue
            a, b = int(con.geom1), int(con.geom2)
            ra, rb = a in self.robot_geoms, b in self.robot_geoms
            kind = None
            if ra != rb:
                other = b if ra else a
                if other == held_geom:
                    continue                          # fingers grip the held cube
                if other in self.world_geoms or other in self.geom_block:
                    kind = 'robot/' + ('world' if other in self.world_geoms else 'cube')
            elif held_geom in (a, b):
                other = b if a == held_geom else a
                if other == self.table_geom and pen <= SUPPORT_TOL:
                    continue                          # resting on table at grasp/release
                if other in self.world_geoms or other in self.geom_block:
                    kind = 'held-cube/' + ('world' if other in self.world_geoms else 'cube')
            if kind is None:
                continue
            issues.append(f'{self._gname(a)} <-> {self._gname(b)}: '
                          f'penetration={pen:.4f}m ({kind})')
            if len(issues) >= max_reports:
                break
        return issues

    def collision_free(self, q, carrying=False):
        return not self.collision_reasons(q, carrying=carrying, max_reports=1)

    @contextmanager
    def hypothetical_grasp(self, name, attachment):
        """Plan as if `name` were already held with `attachment` (no live change)."""
        saved = (self.held, self.attachment)
        self.held, self.attachment = name, attachment
        try:
            yield
        finally:
            self.held, self.attachment = saved

    @contextmanager
    def hypothetical_release(self):
        saved = (self.held, self.attachment)
        self.held, self.attachment = None, None
        try:
            yield
        finally:
            self.held, self.attachment = saved

    # ------------------------------------------------------------ live state
    def reset(self):
        mj = self.mj
        for d in (self.data, self.scratch):
            mj.mj_resetData(self.model, d)
            self.set_arm(d, HOME)
            self.set_fingers(d, GRIP_OPEN)
            for name, _colour, cell in self.initial_blocks:
                self.set_block(d, name, make_transform(cell_xyz(cell), np.eye(3)))
        self.held = self.attachment = None
        self.data.ctrl[self.gripper_act] = GRIP_OPEN
        mj.mj_forward(self.model, self.data)
        mj.mj_forward(self.model, self.scratch)

    def attach(self, name):
        if self.held is not None:
            raise RuntimeError(f'Cannot attach {name}: already holding {self.held}')
        self.mj.mj_forward(self.model, self.data)
        tool, block = self.tool_pose(), self.block_pose(name)
        dist = np.linalg.norm(tool[:3, 3] - block[:3, 3])
        if dist > 0.02:
            raise RuntimeError(f'Grasp of {name} not reached; tool/cube offset {dist:.3f} m')
        self.held, self.attachment = name, transform_inv(tool) @ block

    def release(self, target_xyz=None):
        if self.held is None:
            raise RuntimeError('Cannot release: nothing held')
        self.mj.mj_forward(self.model, self.data)
        H = self.tool_pose() @ self.attachment
        H[:3, :3] = upright(yaw_of(H[:3, :3]))        # cubes land flat
        if target_xyz is not None and self.control == 'kinematic':
            H[:3, 3] = target_xyz
            H[2, 3] = BLOCK_Z                          # no physics: put it on the table
        self.set_block(self.data, self.held, H)
        self.held = self.attachment = None
        self.mj.mj_forward(self.model, self.data)

    def advance(self, q_des, qd_des, dt):
        """One control step (kinematic playback or PD + gravity compensation)."""
        mj, d = self.mj, self.data
        width = self.grip_width
        if self.control == 'kinematic':
            self.set_arm(d, q_des, qd_des)
            self.set_fingers(d, width)
            d.ctrl[self.gripper_act] = width
            if self.held is not None:
                mj.mj_forward(self.model, d)
                self.set_block(d, self.held, self.tool_pose() @ self.attachment)
            mj.mj_forward(self.model, d)
            d.time += dt
        elif self.control == 'torque':
            mj.mj_forward(self.model, d)
            q, qd = self.arm_q(), d.qvel[self.vadr]
            tau = KP * (q_des - q) + KD * (qd_des - qd) + d.qfrc_bias[self.vadr]
            d.ctrl[self.arm_act] = np.clip(tau, -TAU_LIM, TAU_LIM)
            d.ctrl[self.gripper_act] = width
            mj.mj_step(self.model, d)
            if self.held is not None:
                self.set_block(d, self.held, self.tool_pose() @ self.attachment)
                mj.mj_forward(self.model, d)
        else:
            raise ValueError(f'Unknown control mode {self.control!r}')
        if not np.isfinite(d.qpos).all():
            raise RuntimeError('Non-finite simulation state; stopping')
