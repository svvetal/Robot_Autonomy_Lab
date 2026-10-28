"""MuJoCo state management and collision queries."""
import numpy as np
from .config import HOME, Settings, GRIP_OPEN, GRIP_CLOSE
from .geometry import make_transform, matrix_to_mjquat, quat_to_matrix
from .scene import prepare_scene
from .collisions import supported_pickup_contact

class LabSim:
    def __init__(self, cfg: Settings, *, control='kinematic'):
        import mujoco as mj
        self.mj = mj
        self.cfg = cfg
        self.control = control
        xml = prepare_scene()
        self.model = mj.MjModel.from_xml_path(str(xml))
        self.data = mj.MjData(self.model)
        self.scratch = mj.MjData(self.model)
        self.jids = np.array([self._id(mj.mjtObj.mjOBJ_JOINT, f'joint{k}')
                              for k in range(1, 8)], int)
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.jids], int)
        self.vadr = np.array([self.model.jnt_dofadr[j] for j in self.jids], int)
        self.limits = np.asarray(self.model.jnt_range[self.jids], float).copy()
        self.limits[:, 0] += cfg.safe_joint_margin
        self.limits[:, 1] -= cfg.safe_joint_margin
        self.tool_id = self._id(mj.mjtObj.mjOBJ_SITE, 'lab2_tool')
        self.block_id = self._id(mj.mjtObj.mjOBJ_BODY, 'Block')
        freeid = self.model.body_jntadr[self.block_id]
        if freeid < 0 or self.model.jnt_type[freeid] != mj.mjtJoint.mjJNT_FREE:
            raise ValueError('Block must have a free joint')
        self.block_qadr = int(self.model.jnt_qposadr[freeid])
        self.block_vadr = int(self.model.jnt_dofadr[freeid])
        self.arm_act = np.array([self._id(mj.mjtObj.mjOBJ_ACTUATOR, f'actuator{k}')
                                 for k in range(1, 8)])
        self.gripper_act = self._id(mj.mjtObj.mjOBJ_ACTUATOR, 'actuator8')
        self.finger_jids = [self._id(mj.mjtObj.mjOBJ_JOINT, f'finger_joint{k}')
                            for k in (1, 2)]
        self.finger_qadr = [int(self.model.jnt_qposadr[j]) for j in self.finger_jids]
        self.finger_vadr = [int(self.model.jnt_dofadr[j]) for j in self.finger_jids]
        self.robot_geoms = set()
        self.world_geoms = set()
        self.block_geoms = set()
        for g in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[g])
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body) or ''
            if name.startswith(('link', 'hand', 'left_finger', 'right_finger')):
                if name != 'link0':
                    self.robot_geoms.add(g)
            elif name == 'Block':
                self.block_geoms.add(g)
            elif name == 'world' or name.endswith('Shelf') or 'Shelf' in name or name == 'TablePlane':
                self.world_geoms.add(g)
        if not self.robot_geoms or not self.world_geoms or not self.block_geoms:
            raise RuntimeError('Expected robot, shelf, and block collision geoms not found')
        self.rng = np.random.default_rng(cfg.seed)
        self.attachment = None
        self.gripper_command = GRIP_OPEN
        self._saved_block_masks = None
        self.reset()

    def _id(self, obj_type, name):
        k = self.mj.mj_name2id(self.model, obj_type, name)
        if k < 0:
            raise RuntimeError(f'Missing MuJoCo {obj_type}: {name}')
        return int(k)

    def arm_q(self, d=None):
        return np.asarray((self.data if d is None else d).qpos[self.qadr], float).copy()

    def set_arm(self, d, q, qd=None):
        d.qpos[self.qadr] = np.asarray(q, float)
        d.qvel[self.vadr] = 0 if qd is None else qd

    def set_block(self, d, H):
        a = self.block_qadr
        d.qpos[a:a+3] = H[:3, 3]
        d.qpos[a+3:a+7] = matrix_to_mjquat(H[:3, :3])
        d.qvel[self.block_vadr:self.block_vadr+6] = 0

    def block_pose(self, d=None):
        d = self.data if d is None else d
        a = self.block_qadr
        return make_transform(d.qpos[a:a+3].copy(), quat_to_matrix(d.qpos[a+3:a+7]))

    def tool_pose(self, d=None):
        d = self.data if d is None else d
        return make_transform(d.site_xpos[self.tool_id].copy(),
                              d.site_xmat[self.tool_id].reshape(3, 3).copy())

    def evaluate(self, q, carrying=False):
        """Compute the simulated tool pose at the given joints."""
        self.set_arm(self.scratch, q)

        self.scratch.qpos[self.finger_qadr] = self.gripper_command
        self.scratch.qvel[self.finger_vadr] = 0
        self.mj.mj_forward(self.model, self.scratch)
        if carrying:
            if self.attachment is None:
                raise RuntimeError('Collision requested with block but no attachment')
            self.set_block(self.scratch, self.tool_pose(self.scratch) @ self.attachment)
            self.mj.mj_forward(self.model, self.scratch)
        return self.tool_pose(self.scratch)

    def collision_reasons(self, q, carrying=False, max_reports=8):
        """Return collisions involving the robot or carried cube."""
        mj = self.mj
        q = np.asarray(q, float)
        if q.shape != (7,) or not np.isfinite(q).all():
            return ['nonfinite or incorrectly shaped joint configuration']
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            return ['joint configuration outside safety limits']
        self.evaluate(q, carrying=carrying)
        issues = []
        cube_center = self.block_pose(self.scratch)[:3, 3]

        for con in self.scratch.contact[:self.scratch.ncon]:
            a, b = int(con.geom1), int(con.geom2)
            robot_world = ((a in self.robot_geoms and b in self.world_geoms) or
                           (b in self.robot_geoms and a in self.world_geoms))
            cube_world = carrying and ((a in self.block_geoms and b in self.world_geoms) or
                                       (b in self.block_geoms and a in self.world_geoms))
            if con.dist > 0 or not (robot_world or cube_world):
                continue
            if cube_world:
                world_geom = b if a in self.block_geoms else a
                cube_geom = a if a in self.block_geoms else b
                world_body = int(self.model.geom_bodyid[world_geom])
                body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, world_body) or ''
                if supported_pickup_contact(
                        shelf_body=body_name,
                        shelf_center=self.scratch.geom_xpos[world_geom],
                        shelf_half_size=self.model.geom_size[world_geom, :3],
                        block_center=cube_center,
                        block_half_size=self.model.geom_size[cube_geom, :3],
                        contact_distance=float(con.dist)):
                    continue
            def geo_name(g):
                geom = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, g)
                body = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY,
                                     int(self.model.geom_bodyid[g]))
                return geom or body or f'geom#{g}'
            issues.append(f'{geo_name(a)} <-> {geo_name(b)}: penetration={-float(con.dist):.5f}m'
                          f' ({"robot/world" if robot_world else "carried-block/world"})')
            if len(issues) >= max_reports:
                break
        return issues

    def collision_free(self, q, carrying=False):
        return not self.collision_reasons(q, carrying=carrying, max_reports=1)

    def reset(self):
        self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_resetData(self.model, self.scratch)
        self.attachment = None
        self.set_arm(self.data, HOME)
        self.set_arm(self.scratch, HOME)
        from .config import TABLE_BLOCK_POS
        self.set_block(self.data, make_transform(TABLE_BLOCK_POS, np.eye(3)))
        self.set_block(self.scratch, make_transform(TABLE_BLOCK_POS, np.eye(3)))
        self.gripper_command = GRIP_OPEN
        self.data.ctrl[self.gripper_act] = GRIP_OPEN
        self.data.qpos[self.finger_qadr] = GRIP_OPEN
        self.scratch.qpos[self.finger_qadr] = GRIP_OPEN
        self.mj.mj_forward(self.model, self.data)
        self.mj.mj_forward(self.model, self.scratch)

    # Only the cube is reset; the MuJoCo session continues across trials.
    def prepare_next_trial(self):
        """Return the cube to the table between trials."""
        from .config import TABLE_BLOCK_POS
        if self.attachment is not None:
            raise RuntimeError('Cannot respawn a block that is attached')
        home_error = np.linalg.norm(self.arm_q() - HOME)
        if home_error > 0.13:
            raise RuntimeError(f'Arm must return home before next trial (error={home_error:.3f} rad)')
        elapsed = float(self.data.time)
        self.set_block(self.data, make_transform(TABLE_BLOCK_POS, np.eye(3)))
        self.set_block(self.scratch, make_transform(TABLE_BLOCK_POS, np.eye(3)))
        self.gripper_command = GRIP_OPEN
        self.data.qpos[self.finger_qadr] = GRIP_OPEN
        self.data.qvel[self.finger_vadr] = 0
        self.scratch.qpos[self.finger_qadr] = GRIP_OPEN
        self.scratch.qvel[self.finger_vadr] = 0
        self.data.ctrl[self.gripper_act] = GRIP_OPEN
        self.mj.mj_forward(self.model, self.data)
        self.mj.mj_forward(self.model, self.scratch)
        assert abs(self.data.time - elapsed) < 1e-8, 'Trial setup changed simulation clock'

    def attach(self):
        if self.attachment is not None:
            raise RuntimeError('Cannot attach: cube already held')
        self.mj.mj_forward(self.model, self.data)
        tool = self.tool_pose()
        block = self.block_pose()
        from .geometry import transform_inv
        d = np.linalg.norm(tool[:3, 3] - block[:3, 3])
        if d > 0.045:
            raise RuntimeError(f'Grasp location not reached; block/tool offset {d:.3f} m')
        self.attachment = transform_inv(tool) @ block
        self.gripper_command = GRIP_CLOSE

    def release(self, target=None):
        if self.attachment is None:
            raise RuntimeError('Cannot release: no attached cube')

        self.mj.mj_forward(self.model, self.data)
        actual = self.tool_pose() @ self.attachment
        if target is not None and self.control == 'kinematic':
            actual[:3, 3] = np.asarray(target, float)

            actual[2, 3] -= self.cfg.placement_gap
        self.set_block(self.data, actual)
        self.attachment = None
        self.gripper_command = GRIP_OPEN

    def advance(self, q_des, qd_des, dt):
        """Advance a single kinematic or torque-control step."""
        mj = self.mj
        if self.control == 'kinematic':
            self.set_arm(self.data, q_des, qd_des)
            self.data.ctrl[self.gripper_act] = self.gripper_command
            self.data.qpos[self.finger_qadr] = self.gripper_command
            self.data.qvel[self.finger_vadr] = 0
            if self.attachment is not None:
                mj.mj_forward(self.model, self.data)
                self.set_block(self.data, self.tool_pose() @ self.attachment)
            mj.mj_forward(self.model, self.data)
            self.data.time += dt
        elif self.control == 'torque':
            KP = np.array([120., 120., 100., 90., 60., 40., 30.])
            KD = np.array([10., 10., 9., 8., 6., 5., 4.])
            mj.mj_forward(self.model, self.data)
            q, qd = self.arm_q(), self.data.qvel[self.vadr]
            tau = KP*(q_des-q) + KD*(qd_des-qd) + self.data.qfrc_bias[self.vadr]
            self.data.ctrl[self.arm_act] = np.clip(tau, [-87,-87,-87,-87,-12,-12,-12],
                                                  [87,87,87,87,12,12,12])
            self.data.ctrl[self.gripper_act] = self.gripper_command
            mj.mj_step(self.model, self.data)
            if self.attachment is not None:
                self.set_block(self.data, self.tool_pose() @ self.attachment)
                mj.mj_forward(self.model, self.data)
        else:
            raise ValueError('Unknown control mode')
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError('Non-finite simulation state; stopping')
