"""Manipulation sequence for pickup, shelf placement and regrasp."""
from dataclasses import dataclass, asdict
from .config import (HOME, TOP_R, SIDE_POS_Y_R, SIDE_NEG_Y_R,
                     Settings, targets_for_trial, trial_seed)
from .geometry import make_transform
from .ik import PoseIK
from .planning import JointPlanner
import numpy as np

@dataclass
class StageRecord:
    stage: str
    goal_xyz: list | None
    waypoints: int
    duration_s: float
    carrying: bool

class ManipulationTask:
    def __init__(self, sim, cfg: Settings, player):
        self.sim, self.cfg, self.player = sim, cfg, player
        self.ik = PoseIK(sim, cfg)
        self.planner = JointPlanner(sim, cfg)
        self.logs = []

    def _move(self, name, xyz, orient, carrying=False):
        sim = self.sim
        if carrying and sim.attachment is None:
            raise RuntimeError(f'{name}: requested attached-object motion without a grasp')
        target = make_transform(xyz, orient)
        if carrying:

            target[:3, 3] -= orient @ sim.attachment[:3, 3]
        initial = sim.arm_q()
        try:
            goal = self.ik.solve(target, initial, carrying=carrying)
            move = self.planner.build_move(name, initial, goal, carrying=carrying)
        except Exception as exc:
            raise RuntimeError(f'Stage {name}: {exc}') from exc
        self.player.move(move)
        if sim.control == 'torque' and np.linalg.norm(sim.arm_q()-goal) > 0.13:
            raise RuntimeError(f'{name}: torque tracking did not converge; stop and retune gains')
        self.logs.append(StageRecord(name, np.asarray(xyz).round(5).tolist(),
                                     len(move.path), sum(move.durations), carrying))

    def _grip(self, name, close, release_target=None):
        sim = self.sim
        if close:
            sim.attach()
        else:
            sim.release(release_target)
        self.player.hold(self.cfg.command_hold_seconds)
        self.logs.append(StageRecord(name, None, 0, self.cfg.command_hold_seconds,
                                     sim.attachment is not None))

    def run_one(self, trial_number, targets=None):
        """Execute one full manipulation trial."""
        cfg, sim = self.cfg, self.sim
        self.logs.clear()
        self.player.frame()
        left, right, right_shelf = (targets_for_trial(cfg, trial_number)
                                    if targets is None else targets)
        obj = sim.block_pose()[:3, 3].copy()

        self._move('01_table_pregrasp_top', obj+[0, 0, cfg.hover_z], TOP_R)
        self._move('02_table_grasp_top', obj, TOP_R)
        self._grip('03_close_gripper_attach', close=True)
        self._move('04_lift_cube_from_table', obj+[0, 0, cfg.lift_z], TOP_R, True)

        self._move('05_left_shelf_preplacement', left+[0, 0, cfg.hover_z], TOP_R, True)
        self._move('06_place_left_from_above', left, TOP_R, True)
        self._grip('07_release_on_left_shelf', False, release_target=left)
        self._move('08_retract_after_left_place', left+[0, 0, cfg.hover_z], TOP_R)

        self._move('09_side_grasp_approach', left+[0, -0.13, 0], SIDE_POS_Y_R)
        self._move('10_side_grasp', left, SIDE_POS_Y_R)
        self._grip('11_close_gripper_regrasp', True)

        self._move('12_side_grasp_lift_clear_shelf', left+[0, 0, 0.07], SIDE_POS_Y_R, True)
        self._move('13_side_grasp_retreat', left+[0, -0.13, 0.07], SIDE_POS_Y_R, True)

        self._move('14_right_shelf_side_approach', right+[0, +0.14, 0], SIDE_NEG_Y_R, True)
        self._move('15_place_on_right_shelf', right, SIDE_NEG_Y_R, True)
        self._grip('16_release_on_right_shelf', False, release_target=right)
        self._move('17_right_shelf_retract', right+[0, +0.14, 0], SIDE_NEG_Y_R)

        goal = self.planner.build_move('18_return_home', sim.arm_q(), HOME, False)
        self.player.move(goal)
        self.logs.append(StageRecord('18_return_home', None,
                                     len(goal.path), sum(goal.durations), False))
        self.player.hold(0.6)
        return {
            'trial': int(trial_number+1),
            'seed': int(trial_seed(cfg.seed, trial_number)),
            'sim_time_end_s': float(sim.data.time) if hasattr(sim, 'data') else None,
            'left_shelf_block_xyz': np.asarray(left).round(5).tolist(),
            'right_shelf_index_0based': int(right_shelf),
            'right_shelf_block_xyz': np.asarray(right).round(5).tolist(),
            'final_arm_error_rad': float(np.linalg.norm(sim.arm_q()-HOME)),
            'stages': [asdict(l) for l in self.logs],
            'status': 'executed_in_mujoco',
            'control': sim.control,
            'block_attachment': 'scripted_kinematic_follow_not_force_based_grasp',
        }
