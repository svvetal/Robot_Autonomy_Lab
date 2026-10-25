"""Real MuJoCo checks. Skipped gracefully on systems without the MuJoCo wheel."""
import os
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco', reason='install requirements.txt for MuJoCo integration checks')

from lab2.config import Settings, TABLE_BLOCK_POS, TOP_R, HOME
from lab2.geometry import make_transform
from lab2.simulator import LabSim
from lab2.ik import PoseIK
from lab2.rrt import JointRRT


def test_load_unmodified_scene_geometry_and_named_objects():
    sim = LabSim(Settings(), control='kinematic')
    assert sim.model.nq >= 16
    assert len(sim.jids) == 7
    assert len(sim.block_geoms) >= 1
    assert len(sim.world_geoms) >= 31
    assert np.allclose(sim.arm_q(), HOME)
    assert np.allclose(sim.block_pose()[:3, 3], TABLE_BLOCK_POS)


def test_home_valid_and_scratch_isolation():
    sim = LabSim(Settings(), control='kinematic')
    snapshot = sim.data.qpos.copy()
    assert sim.collision_free(sim.arm_q())
    assert np.allclose(sim.data.qpos, snapshot)


def test_top_table_grasp_ik_and_segment():
    sim = LabSim(Settings(), control='kinematic')
    pose = make_transform(TABLE_BLOCK_POS, TOP_R)
    q = PoseIK(sim, Settings()).solve(pose, sim.arm_q())
    assert sim.collision_free(q)
    assert np.linalg.norm(sim.evaluate(q)[:3, 3] - TABLE_BLOCK_POS) < .008
    # A path must pass collision sampling; it may require RRT-Connect, so avoid
    # assuming that direct interpolation is safe.
    planner = JointRRT(lambda x: sim.collision_free(x), sim.limits,
                       np.random.default_rng(13), max_iter=3000)
    path = planner.plan(HOME, q)
    assert all(planner.edge_free(a, b) for a, b in zip(path, path[1:]))


@pytest.mark.skipif(os.environ.get('LAB2_FULL_SMOKE') != '1',
                    reason='Set LAB2_FULL_SMOKE=1 to run the full 18-stage simulation smoke test')
def test_single_complete_trial_in_simulation():
    from lab2.playback import Player
    from lab2.task import ManipulationTask
    cfg = Settings(trials=1)
    sim = LabSim(cfg, control='kinematic')
    with Player(sim, cfg, visualize=False, record_path=None) as player:
        results = ManipulationTask(sim, cfg, player).run_one(0)
    assert results['status'] == 'executed_in_mujoco'
    assert len(results['stages']) == 18
    assert results['final_arm_error_rad'] < 1.e-5


def test_side_regrasp_start_accepted_after_closing_on_supported_block():
    """Reproduces user crash at stage 12, without having to render the full task.

    An IK pose must first be collision-free with an open gripper. After
    attaching, a shallow cube/shelf-top contact must not become a false
    collision. All link/shelf and deep contacts are still forbidden.
    """
    from lab2.config import SIDE_POS_Y_R, shelf_targets, BLOCK_HALF
    from lab2.geometry import make_transform
    cfg = Settings(seed=13, trials=1)
    sim = LabSim(cfg)
    left, _, _ = shelf_targets(sim.rng, cfg, 0)
    actual = np.asarray(left).copy()
    actual[2] -= cfg.placement_gap
    sim.set_block(sim.data, make_transform(actual, np.eye(3)))
    sim.mj.mj_forward(sim.model, sim.data)
    pose = make_transform(left, SIDE_POS_Y_R)
    q = PoseIK(sim, cfg).solve(pose, sim.arm_q(), carrying=False)
    sim.set_arm(sim.data, q)
    sim.mj.mj_forward(sim.model, sim.data)
    sim.attach()
    reasons = sim.collision_reasons(q, carrying=True)
    assert not reasons, f'New grasp must be a valid RRT start: {reasons}'
    from lab2.config import GRIP_CLOSE
    assert np.allclose(sim.scratch.qpos[sim.finger_qadr], GRIP_CLOSE)


def test_respawn_preserves_mujoco_instance_and_simulation_clock():
    from lab2.config import RIGHT_TOPS, GRIP_OPEN
    sim = LabSim(Settings(trials=3), control='kinematic')
    old_model, old_data = sim.model, sim.data
    sim.data.time = 17.7
    sim.set_block(sim.data, make_transform([.5, -.32, RIGHT_TOPS[0]+.02], np.eye(3)))
    sim.mj.mj_forward(sim.model, sim.data)
    sim.prepare_next_trial()
    assert sim.model is old_model and sim.data is old_data
    assert abs(sim.data.time-17.7) < 1e-10
    assert np.allclose(sim.arm_q(), HOME)
    assert np.allclose(sim.block_pose()[:3, 3], TABLE_BLOCK_POS)
    assert sim.attachment is None
    assert sim.gripper_command == GRIP_OPEN


@pytest.mark.skipif(os.environ.get('LAB2_CONTINUOUS_SMOKE') != '1',
                    reason='Set LAB2_CONTINUOUS_SMOKE=1 for full 3-trial MuJoCo session')
def test_three_trials_in_one_real_mujoco_session(tmp_path):
    from run_lab2 import build_placements, execute_session
    from lab2.playback import Player
    cfg = Settings(seed=13, trials=3)
    sim = LabSim(cfg, control='kinematic')
    old_data = sim.data
    with Player(sim, cfg, visualize=False) as player:
        results = execute_session(sim, cfg, player, build_placements(cfg), tmp_path)
    assert sim.data is old_data
    assert [x['status'] for x in results] == ['executed_in_mujoco']*3
    times = [x['sim_time_end_s'] for x in results]
    assert times == sorted(times) and len(set(times)) == 3
    assert all(len(x['stages']) == 18 for x in results)
