"""No ROS, no MuJoCo: tests plan schema, frame-guard and frankapy adapter."""
import numpy as np
import pytest
from lab2.config import Settings, HOME, targets_for_trial, trial_seed
from hardware.spec import STAGES, GRIP_ACTIONS, validate_plan
from hardware.execute import execute
from hardware.robot import FrankaRobot, HardwareError


class FakeArm:
    def __init__(self, H=None):
        self.q = HOME.copy()
        self.H = np.eye(4) if H is None else np.asarray(H).copy()
        self.width = .08
        self.commands = []

    def get_joints(self):
        return self.q.copy()

    def get_pose(self):
        class Pose:
            pass
        pose = Pose()
        pose.translation = self.H[:3, 3].copy()
        pose.rotation = self.H[:3, :3].copy()
        return pose

    def get_gripper_width(self):
        return self.width

    def open_gripper(self, block=True):
        self.width = .08
        self.commands.append(('open', block))

    def goto_gripper(self, **kwargs):
        self.width = kwargs['width']
        self.commands.append(('grasp', kwargs.copy()))

    def goto_joints(self, q, **kwargs):
        self.q = np.asarray(q, float)
        self.commands.append(('move', kwargs.copy()))

    def stop_skill(self):
        self.commands.append(('stop',))


def stub_plan(trials=3):
    cfg = Settings(seed=13, trials=trials)
    runs = []
    for i in range(trials):
        left, right, shelf = targets_for_trial(cfg, i)
        items = []
        held = False
        for name in STAGES:
            if name in GRIP_ACTIONS:
                op = GRIP_ACTIONS[name]
                items.append({'name': name, 'type': 'grip', 'action': op})
                held = op == 'close'
            else:
                items.append({'name': name, 'type': 'move',
                              'carrying': held, 'path': [HOME.tolist(), HOME.tolist()],
                              'durations': [2.0], 'goal_tool_pose': np.eye(4).tolist(),
                              'goal_xyz': None})
        runs.append({'trial': i+1, 'seed': trial_seed(cfg.seed, i),
                     'left_xyz': left.tolist(), 'right_xyz': right.tolist(),
                     'right_shelf': shelf+1, 'stages': items})
    return {'schema_version': 1, 'home': HOME.tolist(),
            'home_tool_pose': np.eye(4).tolist(),
            'seed': cfg.seed, 'trials': cfg.trials,
            'runs': runs}


def test_stage_count_structure_and_xyz():
    doc = stub_plan(3)
    assert validate_plan(doc)
    assert all(len(run['stages']) == 18 for run in doc['runs'])
    assert len({round(run['left_xyz'][0], 3) for run in doc['runs']}) == 3


def test_reject_corrupt_motion_order():
    doc = stub_plan(3)
    doc['runs'][0]['stages'][0]['name'] = 'unexpected_move'
    with pytest.raises(ValueError, match='18-stage'):
        validate_plan(doc)


def test_reject_wrong_object_state():
    doc = stub_plan(1)
    doc['runs'][0]['stages'][3]['carrying'] = False
    with pytest.raises(ValueError, match='carry state'):
        validate_plan(doc)


def test_mock_hardware_all_three_trials(tmp_path):
    doc = stub_plan(3)
    fake = FakeArm()
    execute(doc, FrankaRobot(arm=fake), selected_trials=[1, 2, 3],
            auto_stages=True, confirm=lambda _: None,
            log_path=tmp_path/'fake.jsonl')
    records = (tmp_path/'fake.jsonl').read_text().splitlines()
    assert sum('trial_completed' in r for r in records) == 3
    assert sum(command[0] == 'grasp' for command in fake.commands) == 6
    assert sum(command[0] == 'move' for command in fake.commands) == 42  # 14 moves/trial
    assert not any(command[0] == 'stop' for command in fake.commands)
    # Each move passes the safety options through the real API adapter.
    assert all(c[1]['ignore_virtual_walls'] is False
               for c in fake.commands if c[0] == 'move')


def test_bad_home_prevents_movement(tmp_path):
    fake = FakeArm()
    fake.q[0] += .3
    with pytest.raises(HardwareError, match='Initial HOME'):
        execute(stub_plan(1), FrankaRobot(arm=fake), confirm=lambda _: None,
                log_path=tmp_path/'fake.jsonl')
    assert not fake.commands


def test_tool_frame_mismatch_prevents_movement(tmp_path):
    pose = np.eye(4)
    pose[0, 3] = .07
    fake = FakeArm(H=pose)
    with pytest.raises(HardwareError, match='frame mismatch'):
        execute(stub_plan(1), FrankaRobot(arm=fake), confirm=lambda _: None,
                log_path=tmp_path/'fake.jsonl')
    assert not fake.commands


def test_execution_exceptions_stop_without_retreat(tmp_path):
    fake = FakeArm()
    def fail_goto(*args, **kwargs):
        raise RuntimeError('simulated hardware fault')
    fake.goto_joints = fail_goto
    with pytest.raises(RuntimeError, match='fault'):
        execute(stub_plan(1), FrankaRobot(arm=fake), confirm=lambda _: None,
                log_path=tmp_path/'fake.jsonl')
    assert fake.commands[-1][0] == 'stop'
    assert 'ABORTED' in (tmp_path/'fake.jsonl').read_text()
