from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import BLOCK_NAMES, Settings
from .pick_place import PickPlaceError, PickPlaceSkill
from .rrt import PlanFailure
from .simulator import Lab3Sim
from .task_planner import (
    GRID_CELLS,
    PATTERNS,
    location_xy,
    PATTERN_ORDER,
    LOCATIONS,
    ColorState,
    SymbolicStateError,
    TaskMove,
    TaskPlanner,
    apply_symbolic_move,
    goal_reached,
    observe_simulator,
    resolve_block,
)

class TaskExecutionError(RuntimeError):
    pass

def _grid_string(state: ColorState) -> str:
    return "\n".join("    " + " ".join(row) for row in state.grid_rows())

class TaskExecutor:

    def __init__(
        self,
        sim: Lab3Sim,
        cfg: Settings,
        *,
        step_callback: Callable[[str, object], None] | None = None,
        max_expansions: int = 50_000,
        max_moves_per_goal: int = 40,
        max_preflight_rejections: int = 12,
        max_state_mismatches: int = 3,
        grasp_modes: Sequence[str] = ("top",),
    ) -> None:
        if sim.control != "kinematic":
            raise ValueError("TaskExecutor currently requires kinematic Lab3Sim")
        if min(max_expansions, max_moves_per_goal,
               max_preflight_rejections, max_state_mismatches) < 1:
            raise ValueError("All planner/execution budgets must be positive")
        self.sim = sim
        self.cfg = cfg
        self.skill = PickPlaceSkill(
            sim, cfg, step_callback=step_callback, grasp_modes=grasp_modes,
        )
        self.available_grasp_modes = tuple(grasp_modes)
        self.max_expansions = int(max_expansions)
        self.max_moves_per_goal = int(max_moves_per_goal)
        self.max_preflight_rejections = int(max_preflight_rejections)
        self.max_state_mismatches = int(max_state_mismatches)
        self.rejected: set[tuple[tuple[str, ...], TaskMove]] = set()
        self.rejected_places: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
        self.blocked_goals: dict[str, set[tuple[str, str]]] = {}
        self.active_pattern: str | None = None
        self.visited_states: dict[tuple[str, ...], int] = {}

    @staticmethod
    def _placement_neighbors(state: ColorState, move: TaskMove) -> tuple[tuple[str, str], ...]:
        x, y = location_xy(move.destination)
        around = []
        for cell in GRID_CELLS:
            if cell == move.destination:
                continue
            cx, cy = location_xy(cell)
            if (cx - x)**2 + (cy - y)**2 <= 0.11**2:
                color = "." if cell == move.source else state.at(cell)
                around.append((cell, color))
        return tuple(around)

    def _allow_move(self, state: ColorState, move: TaskMove) -> bool:
        # A failed complete transfer is evidence about this action in this
        # state. It is not evidence that every pickup/IK branch aimed at the
        # same destination will fail.
        if (state.occupied, move) in self.rejected:
            return False
        return True

    @staticmethod
    def _next_occupancy(state: ColorState, move: TaskMove) -> tuple[str, ...]:
        return apply_symbolic_move(state, move).occupied

    def _record_blockers(
        self, destination: str, message: str,
        block_locations: dict[str, str], moving_block: str,
    ) -> None:
        blocks = set(re.findall(r"\b(?:red|blue)_[0-3](?=_geom\b)", message))
        blocking = {
            (block_locations[name], "R" if name.startswith("red_") else "B")
            for name in blocks if name in block_locations
            and name != moving_block and block_locations[name] in GRID_CELLS
        }
        if blocking and destination in GRID_CELLS:
            self.blocked_goals[destination] = blocking
            print(f"  Blocking cells at {destination}: "
                  f"{sorted(cell for cell, _ in blocking)}", flush=True)

    def _move_with_modes(
        self, block_name: str, destination: str, modes: Sequence[str],
    ):
        previous = self.skill.grasp_modes
        try:
            self.skill.grasp_modes = tuple(modes)
            return self.skill.move(block_name, destination)
        finally:
            self.skill.grasp_modes = previous

    def _observe(self) -> tuple[ColorState, dict[str, str]]:
        if self.sim.attachment is not None:
            raise TaskExecutionError(
                "Cannot interpret stable table occupancy while a block is attached"
            )
        return observe_simulator(self.sim)

    def _unchanged_since(
        self,
        q_before: np.ndarray,
        positions_before: dict[str, np.ndarray],
    ) -> bool:

        if self.sim.attachment is not None:
            return False
        if not np.allclose(self.sim.arm_q(), q_before, atol=1e-8, rtol=0):
            return False
        current = self.sim.block_positions()
        return (set(current) == set(positions_before) and all(
            np.allclose(current[name], xyz, atol=1e-8, rtol=0)
            for name, xyz in positions_before.items()
        ))

    def run_goal(self, pattern: str) -> dict[str, Any]:
        if pattern not in PATTERN_ORDER:
            raise ValueError(f"Unknown pattern {pattern!r}; choose {PATTERN_ORDER}")
        self.rejected.clear()
        self.rejected_places.clear()
        self.blocked_goals.clear()
        self.visited_states.clear()
        self.active_pattern = pattern
        moves_done: list[dict[str, Any]] = []
        errors: list[str] = []
        rejected_attempts = 0
        phase_rejections = 0
        mismatches = 0
        searches = 0
        last_expansions = 0
        top_failures: set[tuple[tuple[str, ...], TaskMove]] = set()
        alternatives = tuple(mode for mode in self.available_grasp_modes if mode != "top")
        stage = "top" if "top" in self.available_grasp_modes else "fallback"
        fallback_count = 0
        previous_observed: tuple[str, ...] | None = None
        infeasibility_certificate: dict | None = None

        print(f"\n{'=' * 55}\nGOAL: {pattern.upper()}\n{'=' * 55}", flush=True)

        def report(status: str, reason: str = "") -> dict[str, Any]:
            try:
                state, _ = self._observe()
                final_grid: list[str] | None = list(state.grid_rows())
            except (SymbolicStateError, TaskExecutionError) as exc:
                final_grid = None
                if not reason:
                    reason = str(exc)
            print(f"[{pattern}] {status}: {reason or 'finished'}", flush=True)
            if final_grid is not None:
                print("Final observed grid:\n" + "\n".join(
                    "    " + " ".join(row) for row in final_grid
                ), flush=True)
            return {
                "pattern": pattern,
                "status": status,
                "reason": reason,
                "moves_completed": len(moves_done),
                "movements": moves_done,
                "preflight_rejections": rejected_attempts,
                "grasp_strategy_fallbacks": fallback_count,
                "state_mismatches": mismatches,
                "planner_calls": searches,
                "last_search_expansions": last_expansions,
                "final_grid": final_grid,
                "simulation_time_s": float(self.sim.data.time),
                "errors": errors,
                "infeasibility_certificate": infeasibility_certificate,
            }

        while True:
            try:
                state, block_locations = self._observe()
            except (SymbolicStateError, TaskExecutionError) as exc:
                return report("stopped_observation", str(exc))

            if state.occupied != previous_observed:
                self.visited_states[state.occupied] = self.visited_states.get(state.occupied, 0) + 1
                previous_observed = state.occupied
            if goal_reached(state, pattern):
                return report("achieved")
            if len(moves_done) >= self.max_moves_per_goal:
                return report(
                    "stopped_move_limit",
                    f"Reached {self.max_moves_per_goal} completed moves before goal",
                )
            if phase_rejections >= self.max_preflight_rejections:
                if stage == "top" and alternatives:
                    stage = "fallback"
                    fallback_count += 1
                    phase_rejections = 0
                    self.rejected.clear()
                    self.rejected_places.clear()
                    self.blocked_goals.clear()
                    print(
                        f"[{pattern}] Top-only action alternatives exhausted "
                        f"({self.max_preflight_rejections} preflight rejections); "
                        "enabling tilted/side grasps", flush=True,
                    )
                    continue
                return report(
                    "stopped_rejection_limit",
                    f"Reached {self.max_preflight_rejections} rejected preflights "
                    f"in the {stage} search phase",
                )
            if searches >= 3 * (self.max_moves_per_goal + 2 * self.max_preflight_rejections):
                return report("stopped_search_limit", "Too many repeated task searches")

            planner = TaskPlanner(
                move_filter=self._allow_move,
                check_accessibility=self.available_grasp_modes == ("top",),
            )
            symbolic_plan = planner.plan(
                state, pattern, max_expansions=self.max_expansions
            )
            searches += 1
            last_expansions = symbolic_plan.expansions
            print(
                f"[{pattern}] A*: {symbolic_plan.status}, "
                f"planned moves={len(symbolic_plan.moves)}, "
                f"expanded={symbolic_plan.expansions}",
                flush=True,
            )
            if symbolic_plan.status == "infeasible":
                infeasibility_certificate = symbolic_plan.infeasibility_certificate
                return report("skipped_top_grasp_infeasible", symbolic_plan.reason)
            if symbolic_plan.status == "search_limit":
                return report(
                    "stopped_search_limit",
                    "Search expansion budget reached; physical infeasibility NOT established",
                )
            if symbolic_plan.status == "no_solution":
                if stage == "top" and alternatives:
                    stage = "fallback"
                    fallback_count += 1
                    phase_rejections = 0
                    self.rejected.clear()
                    self.rejected_places.clear()
                    self.blocked_goals.clear()
                    print(
                        f"[{pattern}] No remaining top-feasible symbolic plan; "
                        "trying tilted/side grasps after exploring other block moves",
                        flush=True,
                    )
                    continue
                return report(
                    "skipped_no_plan",
                    "No plan under current symbolic constraints and rejected "
                    "preflight actions; not proof of geometric infeasibility",
                )
            if not symbolic_plan.moves:
                return report(
                    "stopped_inconsistent_plan",
                    "Planner returned no moves although the goal is not observed",
                )

            action = symbolic_plan.moves[0]
            try:
                block_name = resolve_block(action, block_locations)
                predicted = apply_symbolic_move(state, action)
            except SymbolicStateError as exc:
                return report("stopped_inconsistent_plan", str(exc))

            print(
                f"[{pattern}] Next: {action.label()} "
                f"[physical cube: {block_name}]", flush=True
            )
            before_q = self.sim.arm_q()
            before_xyz = self.sim.block_positions()

            if stage == "top":
                modes = ("top",)
            elif (state.occupied, action) in top_failures and alternatives:
                modes = alternatives
            else:
                modes = self.available_grasp_modes
            print(f"[{pattern}] Grasp policy: {stage}; trying {', '.join(modes)}", flush=True)
            self.skill.phases.clear()
            try:
                result = self._move_with_modes(block_name, action.destination, modes)
            except (PickPlaceError, PlanFailure, RuntimeError) as exc:
                reason = f"{action.label()}: {type(exc).__name__}: {exc}"
                errors.append(reason)
                print(f"[{pattern}] Transfer rejected/failed: {reason}", flush=True)

                if (not self.skill.phases and
                        self._unchanged_since(before_q, before_xyz)):
                    self.rejected.add((state.occupied, action))
                    if stage == "top":
                        top_failures.add((state.occupied, action))
                    message = str(exc)
                    self._record_blockers(
                        action.destination, message, block_locations, block_name,
                    )
                    rejected_attempts += 1
                    phase_rejections += 1
                    print(
                        f"[{pattern}] {stage} preflight rejection "
                        f"({phase_rejections}/{self.max_preflight_rejections}); "
                        "leave block untouched and replan a different action",
                        flush=True,
                    )
                    continue
                return report(
                    "stopped_execution",
                    "Failure occurred after motion began or the scene changed; "
                    "inspect/reset simulator before retrying. " + reason,
                )

            moves_done.append({
                "index": len(moves_done) + 1,
                "action": action.label(),
                "block_name": block_name,
                "source": action.source,
                "destination": action.destination,
                "grasp_yaw_deg": float(result.selected_grasp_yaw_deg),
                "grasp_mode": result.selected_grasp_mode,
                "grasp_quality_sigma_min": float(result.grasp_quality_min_singular),
                "grasp_quality_volume": float(result.grasp_quality_volume),
                "grasp_force_closure_proxy": bool(result.force_closure_approx),
                "placement_rotation_deg": float(result.placement_rotation_deg),
                "placement_approach": result.placement_approach,
                "grasp_height_offset_m": float(result.grasp_height_offset_m),
                "final_position_error_m": float(result.position_error_m),
                "simulation_time_s": float(result.simulation_time_s),
                "attachment_method": result.attachment_method,
            })
            # Arm posture and physical block identity can change after a move.
            # Only failures at the unchanged observed state are reusable.
            self.rejected.clear()
            print(
                f"[{pattern}] Completed move {len(moves_done)}: "
                f"{block_name} -> {action.destination}, "
                f"pickup={result.selected_grasp_mode} "
                f"{result.selected_grasp_yaw_deg:.0f}°, "
                f"G_sigma_min={result.grasp_quality_min_singular:.3f}, "
                f"place_turn={result.placement_rotation_deg:.0f}°, "
                f"approach={result.placement_approach}, "
                f"error={result.position_error_m * 1000:.1f} mm",
                flush=True,
            )

            try:
                actual, actual_locations = self._observe()
            except (SymbolicStateError, TaskExecutionError) as exc:
                return report(
                    "stopped_observation",
                    "Transfer returned, but observation is not a valid grid state: "
                    + str(exc),
                )

            if (actual != predicted or
                    actual_locations.get(block_name) != action.destination):
                mismatches += 1
                errors.append(
                    f"Observed {actual.grid_rows()} vs expected "
                    f"{predicted.grid_rows()} after {action.label()}"
                )
                print(
                    f"[{pattern}] State mismatch; replanning from observation "
                    f"({mismatches}/{self.max_state_mismatches})",
                    flush=True,
                )
                if mismatches >= self.max_state_mismatches:
                    return report(
                        "stopped_mismatch_limit",
                        "Too many observed state disagreements; inspect scene",
                    )
            print("Observed grid:\n" + _grid_string(actual), flush=True)
            if stage == "fallback" and "top" in self.available_grasp_modes:
                stage = "top"
                phase_rejections = 0
                self.rejected.clear()
                self.rejected_places.clear()
                self.blocked_goals.clear()
                print(f"[{pattern}] Scene changed; resume top-first action planning", flush=True)


    @staticmethod
    def _home_slot(block_name: str) -> str:
        color, number = block_name.split("_")
        return ("R" if color == "red" else "B") + number

    @staticmethod
    def _reset_location_key(block_locations: dict[str, str]) -> tuple[str, ...]:
        return tuple(block_locations[name] for name in BLOCK_NAMES)

    @staticmethod
    def _reset_crowding(location: str, occupied: set[str]) -> int:
        x, y = location_xy(location)
        return sum(
            1 for other in occupied if other != location
            and sum((a - b) ** 2 for a, b in zip((x, y), location_xy(other))) < 0.080 ** 2
        )

    def _reset_candidates(
        self,
        block_locations: dict[str, str],
        attempted: set[tuple[tuple[str, ...], str, str]],
        visited: dict[tuple[str, ...], int],
        blocking_names: set[str],
    ) -> list[tuple[str, str, str]]:
        taken = {location: name for name, location in block_locations.items()}
        current_key = self._reset_location_key(block_locations)
        occupied = set(taken)
        candidates: list[tuple[tuple, str, str, str]] = []

        def add(name: str, destination: str, priority: int) -> None:
            source = block_locations[name]
            if source == destination or destination in taken:
                return
            if (current_key, name, destination) in attempted:
                return
            next_locations = dict(block_locations)
            next_locations[name] = destination
            if visited.get(self._reset_location_key(next_locations), 0) >= 2:
                return
            source_neighbors = self._reset_crowding(source, occupied)
            destination_neighbors = self._reset_crowding(destination, occupied)
            sx, sy = location_xy(source)
            dx, dy = location_xy(destination)
            score = (priority, source_neighbors, destination_neighbors,
                     (sx - dx) ** 2 + (sy - dy) ** 2, name, destination)
            candidates.append((score, name, source, destination))

        for name in BLOCK_NAMES:
            if block_locations[name] != self._home_slot(name):
                add(name, self._home_slot(name), 0)

        misplaced = [name for name in BLOCK_NAMES
                     if block_locations[name] != self._home_slot(name)]
        home_blockers = {
            taken[self._home_slot(name)] for name in misplaced
            if self._home_slot(name) in taken
        }
        movable = set(home_blockers) | (blocking_names & set(BLOCK_NAMES))
        if not candidates or all(
            (current_key, name, dest) in attempted
            for _, name, _, dest in candidates
        ):
            movable.update(misplaced)

        empty_grid = [cell for cell in GRID_CELLS if cell not in taken]
        for name in sorted(movable):
            if name in home_blockers:
                priority = 1
            elif name in blocking_names:
                priority = 2
            elif block_locations[name] != self._home_slot(name):
                priority = 3
            else:
                priority = 4
            for cell in empty_grid:
                add(name, cell, priority)

        candidates.sort(key=lambda record: record[0])
        return [(name, source, dest) for _, name, source, dest in candidates]

    def reset_to_origins(
        self, *, max_moves: int = 36, max_rejections: int = 40,
    ) -> dict[str, Any]:
        print(f"\n{'=' * 55}\nRESET: RETURN ALL BLOCKS TO STAGING\n{'=' * 55}",
              flush=True)
        moves: list[dict[str, Any]] = []
        errors: list[str] = []
        attempted_top: set[tuple[tuple[str, ...], str, str]] = set()
        attempted_all: set[tuple[tuple[str, ...], str, str]] = set()
        visited: dict[tuple[str, ...], int] = {}
        blocking_names: set[str] = set()
        rejected = 0

        def report(status: str, reason: str = "") -> dict[str, Any]:
            try:
                _, current = self._observe()
            except (SymbolicStateError, TaskExecutionError):
                current = {}
            remaining = [name for name in BLOCK_NAMES
                         if current.get(name) != self._home_slot(name)]
            print(f"[reset] {status}: {reason or 'finished'}; "
                  f"remaining={remaining}", flush=True)
            return {
                "status": status,
                "reason": reason,
                "moves_completed": len(moves),
                "movements": moves,
                "preflight_rejections": rejected,
                "remaining_blocks": remaining,
                "final_locations": current,
                "simulation_time_s": float(self.sim.data.time),
                "errors": errors,
            }

        if self.sim.attachment is not None:
            return report("stopped_unsafe", "A block is still attached")
        for _ in range(max_moves):
            try:
                _, locations = self._observe()
            except (SymbolicStateError, TaskExecutionError) as exc:
                return report("stopped_observation", str(exc))
            remaining = [name for name in BLOCK_NAMES
                         if locations[name] != self._home_slot(name)]
            if not remaining:
                return report("achieved")
            if rejected >= max_rejections:
                return report("stopped_rejection_limit", "Too many failed reset preflights")
            key = self._reset_location_key(locations)
            visited[key] = visited.get(key, 0) + 1
            moved = False
            alternatives = tuple(mode for mode in self.available_grasp_modes if mode != "top")
            phases = ("top", "fallback") if "top" in self.available_grasp_modes and alternatives else ("all",)
            for phase in phases:
                previously_attempted = attempted_top if phase == "top" else attempted_all
                top_limit = min(12, max(1, max_rejections // 2))
                failures_this_phase = 0
                if phase == "fallback":
                    print("[reset] Other top-grasp block moves did not clear the scene; "
                          "trying tilted/side transfers", flush=True)
                while rejected < max_rejections and not moved:
                    if phase == "top" and failures_this_phase >= top_limit:
                        break
                    candidates = self._reset_candidates(
                        locations, previously_attempted, visited, blocking_names,
                    )
                    if not candidates:
                        break
                    name, source, destination = candidates[0]
                    action_key = (key, name, destination)
                    if phase == "top":
                        modes = ("top",)
                    elif action_key in attempted_top and alternatives:
                        modes = alternatives
                    else:
                        modes = self.available_grasp_modes
                    print(f"[reset] Move {name}: {source} -> {destination} "
                          f"[modes={','.join(modes)}]", flush=True)
                    before_q = self.sim.arm_q()
                    before_xyz = self.sim.block_positions()
                    self.skill.phases.clear()
                    try:
                        result = self._move_with_modes(name, destination, modes)
                    except (PickPlaceError, PlanFailure, RuntimeError) as exc:
                        reason = f"{name}: {source} -> {destination}: {exc}"
                        if self.skill.phases or not self._unchanged_since(before_q, before_xyz):
                            errors.append(reason)
                            return report("stopped_execution", reason)
                        rejected += 1
                        failures_this_phase += 1
                        previously_attempted.add(action_key)
                        blocking_names.update(re.findall(
                            r"\b(?:red|blue)_[0-3](?=_geom\b)", str(exc)
                        ))
                        print(f"[reset] Safe {phase} preflight rejection "
                              f"({rejected}/{max_rejections}); choosing another block/action",
                              flush=True)
                        continue
                    moves.append({
                        "block_name": name,
                        "source": source,
                        "destination": destination,
                        "grasp_mode": result.selected_grasp_mode,
                        "final_position_error_m": float(result.position_error_m),
                        "simulation_time_s": float(result.simulation_time_s),
                    })
                    try:
                        _, checked = self._observe()
                    except (SymbolicStateError, TaskExecutionError) as exc:
                        return report("stopped_observation", str(exc))
                    if checked.get(name) != destination:
                        return report("stopped_mismatch", f"{name} did not reach {destination}")
                    moved = True
                    break
                if moved or rejected >= max_rejections:
                    break
            if not moved:
                return report("stopped_no_reset_plan", "All alternative reset actions failed preflight")
        return report("stopped_move_limit", f"Reset exceeded {max_moves} moves")

    def run(
        self, goals: Sequence[str] = PATTERN_ORDER, *,
        reset_after_goal: bool = True,
    ) -> list[dict[str, Any]]:
        reports: list[dict[str, Any]] = []
        for pattern in goals:
            summary = self.run_goal(pattern)
            reports.append(summary)
            safe_to_reset = (
                self.sim.attachment is None
                and summary["status"] in {
                    "achieved", "skipped_no_plan", "skipped_top_grasp_infeasible", "stopped_rejection_limit",
                    "stopped_search_limit", "stopped_move_limit",
                }
            )
            if not safe_to_reset:
                print("Stopping: robot state is not safe for an autonomous reset.",
                      flush=True)
                break
            try:
                self._observe()
            except (SymbolicStateError, TaskExecutionError) as exc:
                print(f"Stopping: cannot observe a valid reset state: {exc}",
                      flush=True)
                break
            if reset_after_goal:
                reset_result = self.reset_to_origins()
                summary["reset"] = reset_result
                if reset_result["status"] != "achieved":
                    print("Stopping remaining goals: autonomous reset failed.",
                          flush=True)
                    break
            else:
                summary["reset"] = None
        return reports

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lab 3 closed-loop task planner + kinematic MuJoCo executor"
    )
    goals = parser.add_mutually_exclusive_group()
    goals.add_argument(
        "--goal", choices=PATTERN_ORDER, default="pickaxe",
        help="Execute a single goal, starting from the initial staging layout",
    )
    goals.add_argument(
        "--all", action="store_true",
        help="Run four patterns and autonomously return all blocks to staging after each",
    )
    parser.add_argument("--view", action="store_true", help="Open live MuJoCo viewer")
    parser.add_argument("--no-reset", action="store_true",
                        help="Keep the previous pattern instead of resetting to staging")
    mode_options = parser.add_mutually_exclusive_group()
    mode_options.add_argument("--top-only", action="store_true",
                              help="Use only top grasps as specified in Lab 3")
    mode_options.add_argument("--side-only", action="store_true",
                              help="Run with side grasps only for diagnosis")
    mode_options.add_argument("--tilted-only", action="store_true",
                              help="Run with tilted grasps only for diagnosis")
    parser.add_argument("--max-expansions", type=int, default=50_000)
    parser.add_argument("--max-moves", type=int, default=40,
                        help="Maximum completed transfers per goal")
    parser.add_argument("--max-rejections", type=int, default=12,
                        help="Maximum preflight action rejections per goal")
    parser.add_argument("--log", type=Path,
                        help="Optional JSON result file, e.g. outputs/lab3/task_run.json")
    args = parser.parse_args()

    from contextlib import nullcontext
    cfg = Settings()
    sim = Lab3Sim(cfg, control="kinematic")

    if args.view:
        from mujoco import viewer
        context = viewer.launch_passive(sim.model, sim.data)
    else:
        context = nullcontext(None)

    try:
        with context as viewer_handle:
            previous_phase = [None]
            if viewer_handle is not None:
                viewer_handle.cam.distance = 2.3
                viewer_handle.cam.azimuth = 140
                viewer_handle.cam.elevation = -30

            def on_step(stage: str, _sim: object) -> None:
                if stage != previous_phase[0]:
                    print("    ->", stage, flush=True)
                    previous_phase[0] = stage
                if viewer_handle is not None:
                    if not viewer_handle.is_running():
                        raise PickPlaceError("MuJoCo viewer closed during execution")
                    viewer_handle.sync()

            executor = TaskExecutor(
                sim, cfg, step_callback=on_step,
                max_expansions=args.max_expansions,
                max_moves_per_goal=args.max_moves,
                max_preflight_rejections=args.max_rejections,
                grasp_modes=(("top",) if args.top_only else
                             ("side",) if args.side_only else
                             ("tilted",) if args.tilted_only else
                             ("top",)),
            )
            targets = PATTERN_ORDER if args.all else (args.goal,)
            summaries = executor.run(targets, reset_after_goal=not args.no_reset)
            print("\n=== LAB 3 EXECUTION SUMMARY ===")
            for item in summaries:
                print(
                    f"  {item['pattern']:8s} {item['status']:24s} "
                    f"moves={item['moves_completed']} "
                    f"rejected={item['preflight_rejections']} "
                    f"reset={(item.get('reset') or {}).get('status', 'off')}"
                )
            if args.log is not None:
                args.log.parent.mkdir(parents=True, exist_ok=True)
                args.log.write_text(
                    json.dumps({
                        "simulation_only": True,
                        "scripted_attachment": True,
                        "goals_attempted": list(targets),
                        "reset_after_goal": not args.no_reset,
                        "results": summaries,
                    }, indent=2) + "\n", encoding="utf-8"
                )
                print(f"Saved run log: {args.log}")
            success = (len(summaries) == len(targets) and all(
                (r["status"] == "achieved" or
                 (r["status"] == "skipped_top_grasp_infeasible" and
                  bool(r.get("infeasibility_certificate"))))
                and (args.no_reset or (r.get("reset") or {}).get("status") == "achieved")
                for r in summaries
            ))
            if viewer_handle is not None and success:
                print("Close the MuJoCo viewer to exit (or press Ctrl+C).")
                import time
                while viewer_handle.is_running():
                    viewer_handle.sync()
                    time.sleep(0.05)
            if not success:
                raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nInterrupted; requesting viewer shutdown.", flush=True)
        return

if __name__ == "__main__":
    main()
