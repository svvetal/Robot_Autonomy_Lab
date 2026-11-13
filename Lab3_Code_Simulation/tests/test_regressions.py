"""Regression checks that do not require MuJoCo or a graphical display.

Run from the project root: python -m unittest discover -s tests -v
The optional validate_lab3.py runner exercises actual MuJoCo transfers.
"""
from __future__ import annotations

from types import SimpleNamespace
import inspect
import unittest

import numpy as np

from lab3.config import BLOCK_NAMES, GRIP_CLOSE, GRIP_OPEN, initial_block_positions
from lab3.geometry import make_transform
from lab3.pick_place import PickPlaceError, PickPlaceSkill
from lab3.grasp_planner import GraspPlanner
from lab3.simulator import Lab3Sim, ScriptedAttachment
from lab3.task_executor import TaskExecutor
from lab3.task_planner import (
    INDEX, LOCATIONS, PATTERNS, PATTERN_ORDER, ColorState, SymbolicStateError,
    TaskMove, TaskPlanner, _center_trap_after, apply_symbolic_move,
    goal_reached, initial_state, observe_simulator, state_from_block_locations,
)


def home_locations():
    return {name: ("R" if name.startswith("red_") else "B") + name[-1]
            for name in BLOCK_NAMES}


class AssignmentPatterns(unittest.TestCase):
    def test_exact_figure_two_patterns(self):
        self.assertEqual(PATTERNS, {
            "pickaxe": ("BBB", ".R.", ".R."),
            "axe": ("BB.", "BR.", ".R."),
            "box": ("BRB", "R.R", "BRB"),
            "bow": ("BR.", "B.R", "BR."),
        })
        self.assertEqual(PATTERN_ORDER, ("pickaxe", "axe", "box", "bow"))

    def test_required_colors_fit_available_blocks(self):
        for pattern, rows in PATTERNS.items():
            with self.subTest(pattern=pattern):
                self.assertEqual(tuple(map(len, rows)), (3, 3, 3))
                self.assertLessEqual("".join(rows).count("R"), 4)
                self.assertLessEqual("".join(rows).count("B"), 4)

    def test_assignment_stack_defaults_to_top_grasps(self):
        for component in (GraspPlanner, PickPlaceSkill, TaskExecutor):
            with self.subTest(component=component.__name__):
                self.assertEqual(inspect.signature(component).parameters["grasp_modes"].default,
                                 ("top",))


class SymbolicPlanning(unittest.TestCase):
    def check_plan(self, state, pattern):
        plan = TaskPlanner().plan(state, pattern)
        self.assertEqual(plan.status, "solved", (pattern, plan))
        for move in plan.moves:
            before = state
            state = apply_symbolic_move(state, move)
            self.assertEqual(state.at(move.source), ".")
            self.assertEqual(state.at(move.destination), move.color)
            self.assertEqual(state.occupied.count("R"), 4)
            self.assertEqual(state.occupied.count("B"), 4)
            changed = {loc for loc in LOCATIONS if before.at(loc) != state.at(loc)}
            self.assertEqual(changed, {move.source, move.destination})
        self.assertTrue(goal_reached(state, pattern))
        self.assertEqual(state.grid_rows(), PATTERNS[pattern])
        return state

    def test_each_pattern_from_staging(self):
        for pattern in PATTERN_ORDER:
            with self.subTest(pattern=pattern):
                if pattern == "axe":
                    plan = TaskPlanner().plan(initial_state(), pattern)
                    self.assertEqual(plan.status, "infeasible")
                    self.assertFalse(plan.succeeded)
                else:
                    self.check_plan(initial_state(), pattern)

    def test_pattern_transitions(self):
        state = initial_state()
        for pattern in PATTERN_ORDER:
            with self.subTest(pattern=pattern):
                if pattern == "axe":
                    self.assertEqual(TaskPlanner().plan(state, pattern).status, "infeasible")
                else:
                    state = self.check_plan(state, pattern)

    def test_abstract_occupancy_mode_can_solve_all_four_patterns(self):
        state = initial_state()
        for pattern in PATTERN_ORDER:
            with self.subTest(pattern=pattern):
                plan = TaskPlanner(check_accessibility=False).plan(state, pattern)
                self.assertEqual(plan.status, "solved")
                for move in plan.moves:
                    state = apply_symbolic_move(state, move)
                self.assertTrue(goal_reached(state, pattern))

    def test_move_preconditions(self):
        state = initial_state()
        for move in (TaskMove("B", "R0", "G11"),
                     TaskMove("R", "R0", "R1"),
                     TaskMove("R", "R0", "R0"),
                     TaskMove("R", "R0", "G33")):
            with self.subTest(move=move):
                with self.assertRaises(SymbolicStateError):
                    apply_symbolic_move(state, move)

    def test_search_limit_is_not_no_solution(self):
        plan = TaskPlanner().plan(initial_state(), "box", max_expansions=1)
        self.assertEqual(plan.status, "search_limit")
        self.assertFalse(plan.succeeded)

    def test_exhausted_action_space_is_no_solution(self):
        plan = TaskPlanner(move_filter=lambda _state, _move: False).plan(
            initial_state(), "pickaxe")
        self.assertEqual(plan.status, "no_solution")
        self.assertFalse(plan.succeeded)

    def test_finishing_ring_cannot_trap_existing_center(self):
        locations = home_locations()
        locations.update(red_0="G11", red_1="G01", blue_0="G10", blue_1="G12")
        state = state_from_block_locations(locations)
        self.assertTrue(_center_trap_after(state, TaskMove("R", "R2", "G21")))

    def test_placing_center_inside_ring_is_rejected(self):
        locations = home_locations()
        locations.update(red_1="G01", red_2="G21", blue_0="G10", blue_1="G12")
        state = state_from_block_locations(locations)
        self.assertTrue(_center_trap_after(state, TaskMove("R", "R0", "G11")))

    def test_empty_center_of_ring_is_allowed(self):
        locations = home_locations()
        locations.update(red_1="G01", blue_0="G10", blue_1="G12")
        state = state_from_block_locations(locations)
        self.assertFalse(_center_trap_after(state, TaskMove("R", "R2", "G21")))


class ObservationRegression(unittest.TestCase):
    def test_observed_staging_matches_symbolic_state(self):
        sim = SimpleNamespace(block_positions=initial_block_positions)
        state, locations = observe_simulator(sim)
        self.assertEqual(state, initial_state())
        self.assertEqual(locations, home_locations())

    def test_duplicate_observed_cell_is_rejected(self):
        positions = initial_block_positions()
        positions["red_1"] = positions["red_0"].copy()
        with self.assertRaises(SymbolicStateError):
            observe_simulator(SimpleNamespace(block_positions=lambda: positions))

    def test_off_grid_block_is_rejected(self):
        positions = initial_block_positions()
        positions["red_0"] = positions["red_0"] + np.array([0.03, 0.03, 0.])
        with self.assertRaises(SymbolicStateError):
            observe_simulator(SimpleNamespace(block_positions=lambda: positions))


class GripperModeRegression(unittest.TestCase):
    def test_release_preserves_selected_aperture(self):
        writes = []
        local = make_transform([0.0, 0.0, 0.015], np.eye(3))
        sim = SimpleNamespace(
            attachment=ScriptedAttachment("red_0", local),
            mj=SimpleNamespace(mj_forward=lambda _m, _d: None),
            model=object(), data=object(), gripper_command=0.0215,
            tool_pose=lambda: np.eye(4),
            set_block_pose=lambda *args: writes.append(args),
        )
        self.assertEqual(Lab3Sim.release(sim), "red_0")
        self.assertIsNone(sim.attachment)
        self.assertEqual(sim.gripper_command, 0.0215)
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][0], "red_0")
        np.testing.assert_allclose(writes[0][1], local)

    def test_release_without_attachment_fails(self):
        with self.assertRaises(RuntimeError):
            Lab3Sim.release(SimpleNamespace(attachment=None))

    def test_closed_gripper_query_restores_aperture(self):
        probes = []
        sim = SimpleNamespace(gripper_command=GRIP_OPEN)

        def collides(_q, **_kw):
            probes.append(sim.gripper_command)
            return False

        sim.collision_free = collides
        skill = PickPlaceSkill.__new__(PickPlaceSkill)
        skill.sim = sim
        self.assertFalse(skill._closed_gripper_is_safe(np.zeros(7), "red_0"))
        self.assertEqual(sim.gripper_command, GRIP_OPEN)
        self.assertEqual(probes, [GRIP_CLOSE])

    def test_failed_transfer_preflight_restores_attachment_and_aperture(self):
        original = make_transform([0.3, 0.18, 0.0206], np.eye(3))
        sim = SimpleNamespace(
            attachment=None, gripper_command=GRIP_OPEN,
            block_pose=lambda _name: original.copy(),
            evaluate=lambda _q: np.eye(4),
        )
        skill = PickPlaceSkill.__new__(PickPlaceSkill)
        skill.sim = sim

        def reject(*_args, **_kw):
            self.assertIsNotNone(sim.attachment)
            self.assertEqual(sim.gripper_command, GRIP_CLOSE)
            raise PickPlaceError("synthetic lift obstruction")

        skill._vertical_path_is_free = reject
        grasp = SimpleNamespace(mode="top", grasp_q=np.zeros(7),
                                approach_qs=(np.zeros(7), np.ones(7) * .01))
        with self.assertRaises(PickPlaceError):
            skill._preplan_transfer("red_0", grasp, np.array([.46, 0., .0206]), np.eye(3))
        self.assertIsNone(sim.attachment)
        self.assertEqual(sim.gripper_command, GRIP_OPEN)
        np.testing.assert_array_equal(sim.block_pose("red_0"), original)

    def test_gripper_sweep_catches_mid_sweep_collision_and_restores_opening(self):
        probes = []
        sim = SimpleNamespace(gripper_command=GRIP_CLOSE)

        def collision_free(_q, **_kwargs):
            probes.append(sim.gripper_command)
            return not (0.025 < sim.gripper_command < 0.027)

        sim.collision_free = collision_free
        skill = PickPlaceSkill.__new__(PickPlaceSkill)
        skill.sim = sim
        # Both endpoints are clear; the interior passes through an obstacle.
        self.assertFalse(skill._gripper_sweep_is_free(
            np.zeros(7), 0.0215, 0.030, target_block="red_0"))
        self.assertGreater(len(probes), 2)
        self.assertTrue(any(.025 < p < .027 for p in probes))
        self.assertEqual(sim.gripper_command, GRIP_CLOSE)

    def test_released_retreat_checks_destination_and_restores_all_state(self):
        for reject_retreat in (False, True):
            with self.subTest(reject_retreat=reject_retreat):
                source = make_transform([.31, .18, .0206], np.eye(3))
                destination = make_transform([.46, 0., .0206], np.eye(3))
                attachment = ScriptedAttachment("red_0", np.eye(4))
                sim = SimpleNamespace(
                    attachment=attachment, gripper_command=GRIP_CLOSE,
                    scratch=object(), current_pose=source.copy(),
                )
                sim.block_pose = lambda _name, data=None: (
                    destination.copy() if data is sim.scratch else sim.current_pose.copy())
                sim.set_block_pose = lambda _name, pose: setattr(sim, "current_pose", pose.copy())
                sim.evaluate = lambda _q, carrying=False: destination.copy()
                skill = PickPlaceSkill.__new__(PickPlaceSkill)
                skill.sim = sim
                checks = []

                def check_retreat(_path, **kwargs):
                    checks.append("retreat")
                    self.assertIsNone(sim.attachment)
                    self.assertEqual(sim.gripper_command, .0215)
                    self.assertFalse(kwargs["carrying"])
                    self.assertEqual(kwargs["target_contact"], "red_0")
                    np.testing.assert_allclose(sim.current_pose, destination)
                    if reject_retreat:
                        raise PickPlaceError("synthetic released-finger collision")

                def check_opening(_q, start, end, **_kwargs):
                    checks.append("opening")
                    self.assertIsNone(sim.attachment)
                    np.testing.assert_allclose(sim.current_pose, destination)
                    self.assertEqual((start, end), (.0215, GRIP_OPEN))
                    return True

                skill._vertical_path_is_free = check_retreat
                skill._gripper_sweep_is_free = check_opening
                result = skill._release_retreat_is_free(
                    (np.zeros(7), np.ones(7) * .01), "red_0", .0215)
                self.assertEqual(result, not reject_retreat)
                self.assertIs(sim.attachment, attachment)
                self.assertEqual(sim.gripper_command, GRIP_CLOSE)
                np.testing.assert_array_equal(sim.current_pose, source)
                self.assertEqual(checks, ["retreat"] if reject_retreat else ["retreat", "opening"])


    def test_execution_reuses_aperture_with_preflighted_retreat(self):
        """Endpoint-only reselection must not undo a complete release check."""
        source = np.array([.31, .18, .0206])
        destination = np.array([.46, 0., .0206])
        position = [source.copy()]
        q = np.zeros(7)
        path = (q.copy(), q.copy())
        sim = SimpleNamespace(
            attachment=None, data=SimpleNamespace(time=0.),
            arm_q=lambda: q.copy(),
            block_pose=lambda _name: make_transform(position[0], np.eye(3)),
            collision_free=lambda *_args, **_kwargs: True,
        )
        sim.attach = lambda _name: setattr(sim, "attachment", object())
        sim.release = lambda: setattr(sim, "attachment", None)
        grasp = SimpleNamespace(
            mode="top", approach_qs=path, yaw_deg=90.,
            quality_min_singular=1., quality_volume=1.,
            force_closure_approx=True, contact_height_offset_m=0.,
            approach_opening_m=.024,
        )
        skill = PickPlaceSkill.__new__(PickPlaceSkill)
        skill.sim, skill.phases = sim, []
        skill.xy_tol, skill.z_tol = .006, .008
        skill._destination_is_available = lambda *_args: None
        selected_aperture = .024  # The smaller endpoint-safe gap failed retreat.
        move = SimpleNamespace(path=path)
        skill._choose_complete_transfer = lambda *_args: (
            grasp, move, path, move, path, 0., "vertical", selected_aperture,
        )
        skill._collision_checked_motion = lambda *_args, **_kwargs: None
        skill._closed_gripper_is_safe = lambda *_args: True
        skill._select_release_opening = lambda *_args, **_kwargs: self.fail(
            "Execution reselected an aperture instead of using the full preflight"
        )
        commands = []
        skill._animate_gripper = lambda target, stage, **_kwargs: commands.append((stage, target))

        def vertical(_path, stage, **_kwargs):
            if stage == "vertical_place_approach":
                position[0] = destination.copy()

        skill._vertical_motion = vertical
        result = skill.move("red_0", destination)
        self.assertEqual(result.position_error_m, 0.)
        self.assertIn(("open_gripper", selected_aperture), commands)
        self.assertIsNone(sim.attachment)


class ReplanningRegression(unittest.TestCase):
    def test_rejected_source_does_not_blacklist_other_sources(self):
        state = initial_state()
        failed = TaskMove("R", "R0", "G11")
        alternative = TaskMove("R", "R1", "G11")
        executor = TaskExecutor.__new__(TaskExecutor)
        executor.rejected = {(state.occupied, failed)}
        self.assertFalse(executor._allow_move(state, failed))
        self.assertTrue(executor._allow_move(state, alternative))


if __name__ == "__main__":
    unittest.main()
