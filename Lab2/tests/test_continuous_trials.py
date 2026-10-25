"""Regression tests for uninterrupted viewer/session and randomized X placements.

These are pure-Python tests, with the separate MuJoCo tests optional.
"""
from types import SimpleNamespace
import numpy as np
import pytest

from lab2.config import Settings, HOME, targets_for_trial, trial_seed, BLOCK_HALF
from run_lab2 import build_placements, execute_session


def test_trial_targets_stratified_along_both_shelves_and_reproducible():
    cfg = Settings(seed=13, trials=3)
    placements = build_placements(cfg)
    assert placements == build_placements(Settings(seed=13, trials=3))
    assert len(placements) == 3
    assert [p['seed'] for p in placements] == [13, 7932, 15851]
    assert [p['right_shelf'] for p in placements] == [0, 1, 2]
    left_x = [p['left'][0] for p in placements]
    right_x = [p['right'][0] for p in placements]
    assert left_x[0] < left_x[1] < left_x[2]
    assert right_x[0] > right_x[1] > right_x[2]
    assert left_x[2] - left_x[0] > .06
    assert right_x[0] - right_x[2] > .06
    assert all(cfg.left_x[0] <= x <= cfg.left_x[1] for x in left_x)
    assert all(cfg.right_x[0] <= x <= cfg.right_x[1] for x in right_x)
    assert all(cfg.left_y[0] <= p['left'][1] <= cfg.left_y[1] for p in placements)
    assert all(cfg.right_y[0] <= p['right'][1] <= cfg.right_y[1] for p in placements)
    # A seed changes random placements; sampling is not hardcoded.
    other = build_placements(Settings(seed=14, trials=3))
    assert any(not np.allclose(a['left'], b['left']) for a, b in zip(placements, other))


def test_trial_targets_independent_of_previous_planner_draws():
    cfg = Settings(seed=13, trials=3)
    p2 = targets_for_trial(cfg, 2)
    noisy = np.random.default_rng(trial_seed(13, 0))
    noisy.random(100000)  # former trial doing more RRT steps does not alter target
    again = targets_for_trial(cfg, 2)
    assert all(np.allclose(a, b) if isinstance(a, np.ndarray) else a == b
               for a, b in zip(p2, again))


def test_session_uses_one_sim_player_monotonic_time_and_two_cube_only_transitions(monkeypatch, tmp_path):
    import lab2.task as task_module
    cfg = Settings(seed=13, trials=3)
    specs = build_placements(cfg)

    class Sim:
        def __init__(self):
            self.data = SimpleNamespace(time=5.0)
            self.rng = np.random.default_rng(0)
            self.attachment = None
            self.transitions = []
        def prepare_next_trial(self):
            self.transitions.append(self.data.time)
            assert self.attachment is None
            assert self.data.time > 5
    class Player:
        def __init__(self):
            self.sim = sim
            self.held = []
        def hold(self, duration):
            self.held.append(duration)
            sim.data.time += duration
        def frame(self):
            pass

    sim = Sim()
    player = Player()
    seen = []

    def fake_task_run(self, index, targets):
        assert self.sim is sim and self.player is player
        assert index == len(seen)
        assert len(targets) == 3
        seen.append((index, sim.data.time, sim.rng.random(), targets[0][0]))
        sim.data.time += 30
        return {'trial': index+1, 'status': 'executed_in_mujoco', 'stages': [None]*18}

    monkeypatch.setattr(task_module.ManipulationTask, 'run_one', fake_task_run)
    results = execute_session(sim, cfg, player, specs, tmp_path)
    assert len(results) == 3
    assert len(sim.transitions) == 2
    assert len({id(sim) for _ in seen}) == 1
    assert all(b[1] > a[1] for a, b in zip(seen, seen[1:]))
    assert sim.data.time > 95
    assert results[0]['transition_before'] == 'initial_scene'
    assert all(x['transition_before'] == 'scripted_cube_respawn_ONLY_no_scene_reset'
               for x in results[1:])
    assert (tmp_path/'results.json').exists()


def test_task_run_does_not_reset_scene_or_erase_sim_clock():
    import inspect
    from lab2.task import ManipulationTask
    code = inspect.getsource(ManipulationTask.run_one)
    assert 'sim.reset(' not in code
    assert 'mj_resetData' not in code
