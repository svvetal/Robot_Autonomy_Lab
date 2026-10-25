"""Run with pytest -q (does not require MuJoCo)."""
from pathlib import Path
import numpy as np
import pytest
from lab2.config import Settings, shelf_targets, RIGHT_TOPS, BLOCK_HALF
from lab2.scene import prepare_scene, scene_shelf_geometry
from lab2.geometry import make_transform, transform_inv, quat_to_matrix, matrix_to_mjquat
from lab2.trajectory import quintic, duration_for_edge
from lab2.rrt import JointRRT, PlanFailure, interpolate_edge
from lab2.offline_fk import XmlKinematics


def test_scene_geometry_exactly_unchanged():
    before = scene_shelf_geometry()
    assert len(before) >= 30
    out = Path('franka_emika_panda/lab2_runtime_shelves.xml')
    prepare_scene(out)
    assert before == scene_shelf_geometry(out)
    assert len(before) == len(scene_shelf_geometry(out))


def test_reproducible_distinct_sampled_shelves():
    cfg = Settings()
    def get():
        r = np.random.default_rng(13)
        return [shelf_targets(r, cfg, k) for k in range(3)]
    cases = get()
    assert all(np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
               for a, b in zip(cases, get()))
    assert len(set(tuple(np.round(x[0], 4)) for x in cases)) == 3
    assert [x[2] for x in cases] == [0, 1, 2]
    for i, (left, right, level) in enumerate(cases):
        # Each trial occupies a different horizontal bin, not merely a jitter
        # around one fixed shelf position. The right side traverses reverse.
        left_bin = int((left[0]-cfg.left_x[0]) / ((cfg.left_x[1]-cfg.left_x[0])/3))
        right_bin = int((right[0]-cfg.right_x[0]) / ((cfg.right_x[1]-cfg.right_x[0])/3))
        assert left_bin == i
        assert right_bin == 2-i
        assert cfg.left_x[0] <= left[0] <= cfg.left_x[1]
        assert 0.31 < left[1] < 0.33
        assert right[1] < 0
        assert abs(right[2] - (RIGHT_TOPS[level] + BLOCK_HALF + cfg.placement_gap)) < 1e-9


def test_quintic_endpoint_rest_and_velocity_bound():
    a, b = np.zeros(7), np.array([0.5, -0.7, 0.1, 0, 0, 0, 0])
    T = duration_for_edge(a, b, 0.65)
    for t in np.linspace(0, T, 250):
        q, v, acc = quintic(a, b, t, T)
        assert np.max(np.abs(v)) <= 0.65+1.e-8
    q, v, acc = quintic(a, b, 0, T)
    assert np.allclose(q, a) and np.allclose(v, 0) and np.allclose(acc, 0)
    q, v, acc = quintic(a, b, T, T)
    assert np.allclose(q, b) and np.allclose(v, 0) and np.allclose(acc, 0, atol=1e-10)


def test_poses_and_quaternions_roundtrip():
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler('xyz', [.2, -.4, .7]).as_matrix()
    H = make_transform([1, 2, 3], R)
    assert np.allclose(H @ transform_inv(H), np.eye(4))
    assert np.allclose(quat_to_matrix(matrix_to_mjquat(R)), R)


def test_edge_checker_sampling_includes_endpoints():
    ps = list(interpolate_edge([0, 0], [1, 1], .05))
    assert np.allclose(ps[0], 0) and np.allclose(ps[-1], 1)
    assert np.max(np.abs(np.diff(ps, axis=0))) <= .05+1.e-9


def test_rrt_straight_path():
    rng = np.random.default_rng(42)
    planner = JointRRT(lambda q: True, [[-2, 2], [-2, 2]], rng)
    p = planner.plan([-1, 0], [1, 0])
    assert len(p) == 2 and planner.edge_free(*p)


def test_rrt_around_box():
    # 2D path from left to right around central collision box.
    rng = np.random.default_rng(7)
    def free(q):
        return not (-.3 < q[0] < .3 and -.6 < q[1] < .6)
    p = JointRRT(free, [[-2, 2], [-2, 2]], rng, step=.15,
                 resolution=.025, max_iter=2000).plan([-1, 0], [1, 0])
    assert all(free(q) for q in p)
    for a,b in zip(p, p[1:]):
        assert all(free(q) for q in interpolate_edge(a,b,.025))


def test_rrt_rejects_bad_goal():
    r = JointRRT(lambda q: q[0] < 0, [[-1, 1]], np.random.default_rng(0))
    with pytest.raises(PlanFailure):
        r.plan([-.9], [.4])


def test_robot_fk_finite_and_correct_joint_count():
    robot = XmlKinematics()
    H = robot.forward(np.zeros(7))
    assert np.all(np.isfinite(H))
    assert np.allclose(H[3], [0, 0, 0, 1])


def test_task_grasp_poses_kinematically_reachable():
    from lab2.config import TOP_R, SIDE_POS_Y_R, SIDE_NEG_Y_R
    robot = XmlKinematics()
    targets = [([.59,0,.0221],TOP_R), ([.51,.32,.1705],TOP_R),
               ([.51,.32,.1705],SIDE_POS_Y_R), ([.51,-.32,.5705],SIDE_NEG_Y_R)]
    for i, (xyz, R) in enumerate(targets):
        yes, err = robot.pose_reachable(xyz, R, seed=i, attempts=12)
        assert yes, (xyz, err)
