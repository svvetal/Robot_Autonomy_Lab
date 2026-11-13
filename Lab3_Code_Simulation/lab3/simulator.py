from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import (
    BLOCK_NAMES,
    BLOCK_HALF,
    TABLE_TOP,
    GRIP_CLOSE,
    GRIP_OPEN,
    HOME,
    Settings,
    initial_block_positions,
)
from .geometry import (
    make_transform,
    matrix_to_mjquat,
    quat_to_matrix,
    transform_inv,
)
from .scene import prepare_scene

@dataclass(frozen=True)
class ScriptedAttachment:

    block_name: str
    tool_to_block: np.ndarray

class Lab3Sim:

    def __init__(self, cfg: Settings, *, control: str = 'kinematic'):
        if control not in ('kinematic', 'torque'):
            raise ValueError("control must be 'kinematic' or 'torque'")
        try:
            import mujoco as mj
        except ImportError as exc:
            raise ImportError('Install MuJoCo: python -m pip install -r requirements.txt') from exc

        self.mj = mj
        self.cfg = cfg
        self.control = control
        self.model = mj.MjModel.from_xml_path(str(prepare_scene()))
        self.data = mj.MjData(self.model)
        self.scratch = mj.MjData(self.model)

        self.jids = np.array(
            [self._id(mj.mjtObj.mjOBJ_JOINT, f'joint{k}') for k in range(1, 8)],
            dtype=int,
        )
        self.qadr = np.asarray(self.model.jnt_qposadr[self.jids], dtype=int)
        self.vadr = np.asarray(self.model.jnt_dofadr[self.jids], dtype=int)
        self.limits = np.asarray(self.model.jnt_range[self.jids], float).copy()
        self.limits[:, 0] += cfg.safe_joint_margin
        self.limits[:, 1] -= cfg.safe_joint_margin
        if np.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError('Safety joint margin exceeds an arm joint range')

        self.tool_id = self._id(mj.mjtObj.mjOBJ_SITE, 'lab3_tool')
        self.arm_act = np.array(
            [self._id(mj.mjtObj.mjOBJ_ACTUATOR, f'actuator{k}') for k in range(1, 8)],
            dtype=int,
        )
        self.gripper_act = self._id(mj.mjtObj.mjOBJ_ACTUATOR, 'actuator8')
        finger_jids = [
            self._id(mj.mjtObj.mjOBJ_JOINT, f'finger_joint{k}')
            for k in (1, 2)
        ]
        self.finger_qadr = np.array(
            [int(self.model.jnt_qposadr[j]) for j in finger_jids], dtype=int
        )
        self.finger_vadr = np.array(
            [int(self.model.jnt_dofadr[j]) for j in finger_jids], dtype=int
        )

        self.block_body_ids: dict[str, int] = {}
        self.block_qadr: dict[str, int] = {}
        self.block_vadr: dict[str, int] = {}
        self.block_geoms: dict[str, set[int]] = {name: set() for name in BLOCK_NAMES}
        for name in BLOCK_NAMES:
            body_id = self._id(mj.mjtObj.mjOBJ_BODY, name)
            j = int(self.model.body_jntadr[body_id])
            if j < 0 or self.model.jnt_type[j] != mj.mjtJoint.mjJNT_FREE:
                raise RuntimeError(f'{name} must have one free joint')
            self.block_body_ids[name] = body_id
            self.block_qadr[name] = int(self.model.jnt_qposadr[j])
            self.block_vadr[name] = int(self.model.jnt_dofadr[j])

        self.robot_geoms: set[int] = set()
        self.grasp_geoms: set[int] = set()
        self.world_geoms: set[int] = set()
        self.geom_to_block: dict[int, str] = {}
        self.geom_to_body_name: dict[int, str] = {}
        for g in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[g])
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, body_id) or 'world'
            self.geom_to_body_name[g] = body_name
            if body_name in self.block_body_ids:
                self.block_geoms[body_name].add(g)
                self.geom_to_block[g] = body_name
            elif body_name.startswith(('link', 'hand', 'left_finger', 'right_finger')):

                if body_name != 'link0':
                    self.robot_geoms.add(g)
                if body_name in ('hand', 'left_finger', 'right_finger'):
                    self.grasp_geoms.add(g)
            else:
                self.world_geoms.add(g)

        if not self.robot_geoms or not self.world_geoms:
            raise RuntimeError('Robot or environment collision geometry missing')
        if any(not geoms for geoms in self.block_geoms.values()):
            raise RuntimeError('Missing one or more physical cube geoms')

        self.rng = np.random.default_rng(cfg.seed)
        self.attachment: Optional[ScriptedAttachment] = None
        self.gripper_command = GRIP_OPEN
        self.reset()

    def _id(self, obj_type, name: str) -> int:
        obj_id = self.mj.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise RuntimeError(f'Missing MuJoCo object: {name}')
        return int(obj_id)

    def arm_q(self, d=None) -> np.ndarray:
        d = self.data if d is None else d
        return np.asarray(d.qpos[self.qadr], float).copy()

    def set_arm(self, d, q, qd=None) -> None:
        q = np.asarray(q, float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('Arm position must contain seven finite values')
        d.qpos[self.qadr] = q
        if qd is None:
            d.qvel[self.vadr] = 0.0
        else:
            qd = np.asarray(qd, float)
            if qd.shape != (7,) or not np.isfinite(qd).all():
                raise ValueError('Arm velocity must contain seven finite values')
            d.qvel[self.vadr] = qd

    def _check_block_name(self, name: str) -> None:
        if name not in self.block_qadr:
            raise ValueError(f'Unknown block {name!r}; choose from {BLOCK_NAMES!r}')

    def set_block_pose(self, name: str, pose, d=None) -> None:

        self._check_block_name(name)
        d = self.data if d is None else d
        H = np.asarray(pose, dtype=float)
        if H.shape != (4, 4) or not np.isfinite(H).all():
            raise ValueError('Block pose must be a finite 4x4 transform')
        a = self.block_qadr[name]
        v = self.block_vadr[name]
        d.qpos[a:a + 3] = H[:3, 3]
        d.qpos[a + 3:a + 7] = matrix_to_mjquat(H[:3, :3])
        d.qvel[v:v + 6] = 0.0

    def block_pose(self, name: str, d=None) -> np.ndarray:

        self._check_block_name(name)
        d = self.data if d is None else d
        a = self.block_qadr[name]
        return make_transform(
            d.qpos[a:a + 3].copy(),
            quat_to_matrix(d.qpos[a + 3:a + 7].copy()),
        )

    def block_positions(self) -> dict[str, np.ndarray]:

        return {name: self.block_pose(name)[:3, 3] for name in BLOCK_NAMES}

    def tool_pose(self, d=None) -> np.ndarray:
        d = self.data if d is None else d
        return make_transform(
            d.site_xpos[self.tool_id].copy(),
            d.site_xmat[self.tool_id].reshape(3, 3).copy(),
        )

    def _sync_scratch(self) -> None:

        self.scratch.qpos[:] = self.data.qpos
        self.scratch.qvel[:] = self.data.qvel
        self.scratch.ctrl[:] = self.data.ctrl
        self.scratch.time = self.data.time

    def evaluate(self, q, carrying: bool = False) -> np.ndarray:

        self._sync_scratch()
        self.set_arm(self.scratch, q)
        self.scratch.qpos[self.finger_qadr] = self.gripper_command
        self.scratch.qvel[self.finger_vadr] = 0.0
        self.mj.mj_forward(self.model, self.scratch)
        if carrying:
            if self.attachment is None:
                raise RuntimeError('Cannot check carrying motion without an attachment')
            attached = self.attachment
            pose = self.tool_pose(self.scratch) @ attached.tool_to_block
            self.set_block_pose(attached.block_name, pose, self.scratch)
            self.mj.mj_forward(self.model, self.scratch)
        return self.tool_pose(self.scratch)

    def _geom_label(self, geom_id: int) -> str:
        name = self.mj.mj_id2name(
            self.model, self.mj.mjtObj.mjOBJ_GEOM, geom_id
        )
        return name or self.geom_to_body_name.get(geom_id, f'geom#{geom_id}')

    def collision_reasons(
        self,
        q,
        carrying: bool = False,
        max_reports: int = 8,
        allow_gripper_contact_with: Optional[str] = None,
        allow_table_support_with: Optional[str] = None,
    ) -> list[str]:

        if allow_gripper_contact_with is not None:
            self._check_block_name(allow_gripper_contact_with)
        if allow_table_support_with is not None:
            self._check_block_name(allow_table_support_with)
            if not carrying or self.attachment is None or self.attachment.block_name != allow_table_support_with:
                raise ValueError("Table support relaxation is only for the currently carried block")
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            return ['invalid arm configuration']
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            return ['arm joints outside safety limits']

        self.evaluate(q, carrying=carrying)
        held = self.attachment.block_name if carrying else None
        issues: list[str] = []

        for contact in self.scratch.contact[:self.scratch.ncon]:

            if float(contact.dist) >= -2e-5:
                continue
            a, b = int(contact.geom1), int(contact.geom2)
            a_robot, b_robot = a in self.robot_geoms, b in self.robot_geoms
            a_world, b_world = a in self.world_geoms, b in self.world_geoms
            a_block = self.geom_to_block.get(a)
            b_block = self.geom_to_block.get(b)
            label = None

            if (a_robot and b_world) or (b_robot and a_world):
                label = 'robot/environment'
            elif a_robot and b_robot:

                if not (a in self.grasp_geoms and b in self.grasp_geoms):
                    label = 'robot/self'
            elif a_block is not None and b_block is not None and a_block != b_block:

                label = 'block/block'
            elif (a_robot and b_block is not None) or (b_robot and a_block is not None):
                robot_geom = a if a_robot else b
                block_name = b_block if a_robot else a_block
                allow_intentional = (
                    robot_geom in self.grasp_geoms
                    and (block_name == held or block_name == allow_gripper_contact_with)
                )
                if not allow_intentional:
                    label = 'robot/block'
            elif held is not None and (
                (a_block == held and b_world) or (b_block == held and a_world)
            ):

                world_geom = b if a_block == held else a
                center_z = float(self.scratch.qpos[self.block_qadr[held] + 2])
                bottom = center_z - BLOCK_HALF
                shallow_table_support = (
                    allow_table_support_with == held
                    and self._geom_label(world_geom) == 'TablePlane'
                    and abs(bottom - TABLE_TOP) <= 0.004
                    and float(contact.dist) >= -0.001
                )
                if not shallow_table_support:
                    label = 'carried-block/environment'

            if label is not None:
                issues.append(
                    f'{self._geom_label(a)} <-> {self._geom_label(b)}: '
                    f'{label}, penetration={-min(0.0, float(contact.dist)):.5f} m'
                )
                if len(issues) >= max_reports:
                    break
        return issues

    def collision_free(
        self,
        q,
        carrying: bool = False,
        allow_gripper_contact_with: Optional[str] = None,
        allow_table_support_with: Optional[str] = None,
    ) -> bool:
        return not self.collision_reasons(
            q,
            carrying=carrying,
            max_reports=1,
            allow_gripper_contact_with=allow_gripper_contact_with,
            allow_table_support_with=allow_table_support_with,
        )

    def reset(self) -> None:

        self.mj.mj_resetData(self.model, self.data)
        self.mj.mj_resetData(self.model, self.scratch)
        self.attachment = None
        self.gripper_command = GRIP_OPEN
        self.set_arm(self.data, HOME)
        for name, xyz in initial_block_positions().items():
            self.set_block_pose(name, make_transform(xyz, np.eye(3)), self.data)
        self.data.qpos[self.finger_qadr] = GRIP_OPEN
        self.data.qvel[self.finger_vadr] = 0.0
        self.data.ctrl[self.gripper_act] = GRIP_OPEN
        self.mj.mj_forward(self.model, self.data)
        self._sync_scratch()
        self.mj.mj_forward(self.model, self.scratch)

    def attach(self, block_name: str) -> None:

        self._check_block_name(block_name)
        if self.attachment is not None:
            raise RuntimeError('Cannot attach while another cube is held')
        self.mj.mj_forward(self.model, self.data)
        tool = self.tool_pose()
        cube = self.block_pose(block_name)
        distance = float(np.linalg.norm(tool[:3, 3] - cube[:3, 3]))
        if distance > 0.045:
            raise RuntimeError(
                f'Tool not at {block_name}: offset {distance:.3f} m (>0.045 m)'
            )
        self.attachment = ScriptedAttachment(block_name, transform_inv(tool) @ cube)
        self.gripper_command = GRIP_CLOSE

    def release(self) -> str:

        if self.attachment is None:
            raise RuntimeError('Cannot release: no block is attached')
        self.mj.mj_forward(self.model, self.data)
        attached = self.attachment
        pose = self.tool_pose() @ attached.tool_to_block
        self.set_block_pose(attached.block_name, pose, self.data)
        self.attachment = None
        # Detaching is a mode transition, not an open-gripper command.  Keep
        # the collision-checked release aperture until the fingers have
        # retreated above neighbouring cubes.
        self.mj.mj_forward(self.model, self.data)
        return attached.block_name

    def advance(self, q_des, qd_des, dt: float) -> None:

        q_des = np.asarray(q_des, dtype=float)
        qd_des = np.asarray(qd_des, dtype=float)
        if q_des.shape != (7,) or qd_des.shape != (7,):
            raise ValueError('Expected seven desired joint positions/velocities')
        if not np.isfinite(q_des).all() or not np.isfinite(qd_des).all():
            raise ValueError('Non-finite desired joint state')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')

        mj = self.mj
        if self.control == 'kinematic':
            self.set_arm(self.data, q_des, qd_des)
            self.data.ctrl[self.gripper_act] = self.gripper_command
            self.data.qpos[self.finger_qadr] = self.gripper_command
            self.data.qvel[self.finger_vadr] = 0.0
            mj.mj_forward(self.model, self.data)
            if self.attachment is not None:
                attached = self.attachment
                self.set_block_pose(
                    attached.block_name,
                    self.tool_pose() @ attached.tool_to_block,
                    self.data,
                )
                mj.mj_forward(self.model, self.data)
            self.data.time += float(dt)
        else:
            kp = np.array([120., 120., 100., 90., 60., 40., 30.])
            kd = np.array([10., 10., 9., 8., 6., 5., 4.])
            mj.mj_forward(self.model, self.data)
            q = self.arm_q()
            qd = self.data.qvel[self.vadr]
            tau = kp * (q_des - q) + kd * (qd_des - qd) + self.data.qfrc_bias[self.vadr]
            lower = np.array([-87., -87., -87., -87., -12., -12., -12.])
            upper = -lower
            self.data.ctrl[self.arm_act] = np.clip(tau, lower, upper)
            self.data.ctrl[self.gripper_act] = self.gripper_command

            if not np.isclose(dt, self.model.opt.timestep, rtol=0, atol=1e-9):
                raise ValueError(
                    f'torque mode dt must equal MuJoCo timestep '
                    f'{self.model.opt.timestep:.6f}s'
                )
            mj.mj_step(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            if self.attachment is not None:
                attached = self.attachment
                self.set_block_pose(
                    attached.block_name,
                    self.tool_pose() @ attached.tool_to_block,
                    self.data,
                )
                mj.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError('Non-finite simulator state; stopped')