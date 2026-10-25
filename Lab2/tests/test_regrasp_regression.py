"""Regression tests for Lab 2 shelf-side regrasp failure (no MuJoCo needed)."""
import numpy as np
from lab2.collisions import supported_pickup_contact
from lab2.config import Settings, HOME
from lab2.task import ManipulationTask, StageRecord

SHELF_CENTER = [.3475, .3465, .1425]
SHELF_HALF = [.2525, .0675, .005]
CUBE_HALF = [.02, .02, .02]


def near_support(**kwargs):
    return supported_pickup_contact(
        shelf_body='LShelfBottom', shelf_center=SHELF_CENTER,
        shelf_half_size=SHELF_HALF,
        block_center=[.50, .32, .1675], block_half_size=CUBE_HALF,
        contact_distance=0.0, **kwargs)


def test_initial_shelf_pickup_contact_is_legal():
    assert near_support()
    assert supported_pickup_contact(
        shelf_body='LShelfBottom', shelf_center=SHELF_CENTER,
        shelf_half_size=SHELF_HALF,
        block_center=[.50, .32, .1673], block_half_size=CUBE_HALF,
        contact_distance=-.0005)


def test_near_shelf_contact_only_not_deep_penetration():
    assert not supported_pickup_contact(
        shelf_body='LShelfBottom', shelf_center=SHELF_CENTER,
        shelf_half_size=SHELF_HALF,
        block_center=[.50, .32, .1595], block_half_size=CUBE_HALF,
        contact_distance=-.008)
    assert not supported_pickup_contact(
        shelf_body='LShelfBottom', shelf_center=SHELF_CENTER,
        shelf_half_size=SHELF_HALF,
        block_center=[.50, .32, .1675], block_half_size=CUBE_HALF,
        contact_distance=-.005)


def test_non_support_contacts_cannot_be_whitelisted():
    for body in ('TablePlane', 'LShelfMid', 'RShelfMiddleSupp3', 'LShelfBottomSuppB'):
        assert not supported_pickup_contact(
            shelf_body=body, shelf_center=SHELF_CENTER,
            shelf_half_size=SHELF_HALF,
            block_center=[.5, .32, .1675], block_half_size=CUBE_HALF,
            contact_distance=0.)
    assert not supported_pickup_contact(
        shelf_body='LShelfBottom', shelf_center=SHELF_CENTER,
        shelf_half_size=SHELF_HALF,
        block_center=[.8, 0., .1675], block_half_size=CUBE_HALF,
        contact_distance=0.)


def test_regrasp_must_lift_before_lateral_retreat():
    class Sim:
        rng = np.random.default_rng(13)
        attachment = None
        control = 'kinematic'
        def reset(self):
            self.attachment = None
        def block_pose(self):
            H = np.eye(4); H[:3, 3] = [.59, 0, .0221]; return H
        def arm_q(self):
            return HOME.copy()
    class Player:
        def frame(self): pass
        def move(self, m): pass
        def hold(self, s): pass
    sim = Sim()
    task = ManipulationTask(sim, Settings(trials=1), Player())
    records = []
    def mock_move(name, xyz, orient, carrying=False):
        records.append((name, np.array(xyz, dtype=float), carrying))
        task.logs.append(StageRecord(name, list(xyz), 2, 1., carrying))
    def mock_grip(name, close, release_target=None):
        sim.attachment = object() if close else None
        records.append((name, None, close))
        task.logs.append(StageRecord(name, None, 0, 0., close))
    task._move = mock_move
    task._grip = mock_grip
    class Planner:
        def build_move(self, *a):
            class Move:
                path = [HOME, HOME]
                durations = [.8]
            return Move()
    task.planner = Planner()
    result = task.run_one(0)
    names = [r[0] for r in records]
    assert names.index('11_close_gripper_regrasp') < names.index('12_side_grasp_lift_clear_shelf')
    assert names.index('12_side_grasp_lift_clear_shelf') < names.index('13_side_grasp_retreat')
    lift = records[names.index('12_side_grasp_lift_clear_shelf')][1]
    retreat = records[names.index('13_side_grasp_retreat')][1]
    assert retreat[2] == lift[2]
    assert retreat[1] < lift[1]
    assert len(result['stages']) == 18
