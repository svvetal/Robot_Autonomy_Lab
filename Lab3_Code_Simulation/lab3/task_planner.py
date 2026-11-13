from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
from itertools import count
from math import hypot, sin, cos, radians
from typing import Callable, Mapping, Sequence

from .config import (
    BLOCK_CENTER_Z,
    BLOCK_NAMES,
    GRID_CELLS,
    STAGING_BLUE_Y,
    STAGING_RED_Y,
    STAGING_X,
    grid_position,
)
from .access import blocked_core, is_recoverable, peel_order, top_access_options

RED_SLOTS = tuple(f"R{i}" for i in range(4))
BLUE_SLOTS = tuple(f"B{i}" for i in range(4))
STAGING_SLOTS = RED_SLOTS + BLUE_SLOTS
LOCATIONS = tuple(GRID_CELLS) + STAGING_SLOTS
INDEX = {name: index for index, name in enumerate(LOCATIONS)}
GRID_INDEX = {name: INDEX[name] for name in GRID_CELLS}

PATTERNS: dict[str, tuple[str, str, str]] = {
    "pickaxe": ("BBB", ".R.", ".R."),
    "axe":     ("BB.", "BR.", ".R."),
    "box":     ("BRB", "R.R", "BRB"),
    "bow":     ("BR.", "B.R", "BR."),
}
PATTERN_ORDER = ("pickaxe", "axe", "box", "bow")

class SymbolicStateError(ValueError):
    pass

@dataclass(frozen=True)
class ColorState:

    occupied: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.occupied) != len(LOCATIONS):
            raise SymbolicStateError(
                f"Expected {len(LOCATIONS)} occupancy entries, got {len(self.occupied)}"
            )
        if any(c not in ("R", "B", ".") for c in self.occupied):
            raise SymbolicStateError("Occupancies must be 'R', 'B' or '.'")
        if self.occupied.count("R") != 4 or self.occupied.count("B") != 4:
            raise SymbolicStateError("The scene must contain exactly four cubes of each color")

    def at(self, location: str) -> str:
        return self.occupied[INDEX[location]]

    def grid_rows(self) -> tuple[str, str, str]:
        return tuple(
            "".join(self.occupied[GRID_INDEX[f"G{r}{c}"]] for c in range(3))
            for r in range(3)
        )

@dataclass(frozen=True)
class TaskMove:

    color: str
    source: str
    destination: str

    def label(self) -> str:
        return f"Move({self.color}, {self.source} -> {self.destination})"

@dataclass(frozen=True)
class TaskPlan:
    pattern: str
    status: str
    moves: tuple[TaskMove, ...]
    expansions: int
    explored_states: int
    reason: str = ""
    infeasibility_certificate: dict | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "solved"

SymbolicMoveFilter = Callable[[ColorState, TaskMove], bool]

def _color_of_block(block_name: str) -> str:
    if block_name not in BLOCK_NAMES:
        raise SymbolicStateError(f"Unknown block {block_name!r}")
    return "R" if block_name.startswith("red_") else "B"

def initial_state() -> ColorState:

    return state_from_block_locations({
        **{f"red_{i}": f"R{i}" for i in range(4)},
        **{f"blue_{i}": f"B{i}" for i in range(4)},
    })

def state_from_block_locations(block_locations: Mapping[str, str]) -> ColorState:

    if set(block_locations) != set(BLOCK_NAMES):
        missing = sorted(set(BLOCK_NAMES) - set(block_locations))
        unexpected = sorted(set(block_locations) - set(BLOCK_NAMES))
        raise SymbolicStateError(
            f"Expected all eight block names; missing={missing}; extra={unexpected}"
        )
    occupied = ["."] * len(LOCATIONS)
    for block_name, location in block_locations.items():
        if location not in INDEX:
            raise SymbolicStateError(f"Unknown location {location!r} for {block_name}")
        if occupied[INDEX[location]] != ".":
            raise SymbolicStateError(f"Two blocks assigned to location {location}")
        occupied[INDEX[location]] = _color_of_block(block_name)
    return ColorState(tuple(occupied))

def location_xy(location: str) -> tuple[float, float]:

    if location in GRID_CELLS:
        xyz = grid_position(int(location[1]), int(location[2]))
        return float(xyz[0]), float(xyz[1])
    if location in RED_SLOTS:
        return float(STAGING_X[int(location[1])]), float(STAGING_RED_Y)
    if location in BLUE_SLOTS:
        return float(STAGING_X[int(location[1])]), float(STAGING_BLUE_Y)
    raise SymbolicStateError(f"Unknown location {location!r}")

def observe_simulator(
    sim,
    *,
    xy_tolerance_m: float = 0.020,
    z_tolerance_m: float = 0.018,
) -> tuple[ColorState, dict[str, str]]:

    if xy_tolerance_m <= 0 or z_tolerance_m <= 0:
        raise ValueError("Observation tolerances must be positive")
    centers = sim.block_positions()
    if set(centers) != set(BLOCK_NAMES):
        raise SymbolicStateError("Simulator did not report exactly the eight Lab 3 cubes")
    site_xy = {loc: location_xy(loc) for loc in LOCATIONS}
    block_locations: dict[str, str] = {}
    assigned: set[str] = set()
    for name in BLOCK_NAMES:
        pos = centers[name]
        if len(pos) < 3:
            raise SymbolicStateError(f"Invalid observed cube pose for {name}")
        x, y, z = map(float, pos[:3])
        if not (abs(z - BLOCK_CENTER_Z) <= z_tolerance_m):
            raise SymbolicStateError(
                f"{name} is not at table block height (z={z:.4f}); "
                "cannot assign a safe symbolic location"
            )
        distances = sorted(
            (hypot(x - xy[0], y - xy[1]), loc) for loc, xy in site_xy.items()
        )
        error, closest = distances[0]
        if error > xy_tolerance_m:
            raise SymbolicStateError(
                f"{name} is {error:.3f} m from nearest known cell {closest}; "
                "recover the physical state before replanning"
            )
        if closest in assigned:
            raise SymbolicStateError(f"More than one cube detected at {closest}")
        assigned.add(closest)
        block_locations[name] = closest
    return state_from_block_locations(block_locations), block_locations

def resolve_block(move: TaskMove, block_locations: Mapping[str, str]) -> str:

    found = [
        name for name, loc in block_locations.items()
        if loc == move.source and _color_of_block(name) == move.color
    ]
    if len(found) != 1:
        raise SymbolicStateError(
            f"Cannot identify a unique {move.color} cube at {move.source}: {found}"
        )
    return found[0]

def goal_reached(state: ColorState, pattern: str) -> bool:

    if pattern not in PATTERNS:
        raise ValueError(f"Unknown Lab 3 pattern: {pattern}")
    return state.grid_rows() == PATTERNS[pattern]

def apply_symbolic_move(state: ColorState, move: TaskMove) -> ColorState:

    if move.source == move.destination:
        raise SymbolicStateError("Moving a block to its own location is not an action")
    if move.source not in INDEX or move.destination not in INDEX:
        raise SymbolicStateError("Action references an unknown location")
    if state.at(move.source) != move.color:
        raise SymbolicStateError(f"No {move.color} block at {move.source}")
    if state.at(move.destination) != ".":
        raise SymbolicStateError(f"Destination {move.destination} is occupied")
    nxt = list(state.occupied)
    nxt[INDEX[move.source]] = "."
    nxt[INDEX[move.destination]] = move.color
    return ColorState(tuple(nxt))

def _center_trap_after(state: ColorState, move: TaskMove) -> bool:
    # Enclosing an existing block is just as unsafe as putting a new block
    # into an enclosure. Check the result of every action, including moves
    # whose destination is a neighboring square.
    after = apply_symbolic_move(state, move)
    return after.at("G11") != "." and all(
        after.at(loc) != "." for loc in ("G01", "G10", "G12", "G21")
    )

def _heuristic(occupancy: tuple[str, ...], desired: tuple[str, ...]) -> int:

    missing_colored = 0
    misplaced_occupied = 0
    for loc in GRID_CELLS:
        a, target = occupancy[INDEX[loc]], desired[INDEX[loc]]
        if target != "." and a != target:
            missing_colored += 1
        if a != "." and a != target:
            misplaced_occupied += 1
    return max(missing_colored, misplaced_occupied)

def _desired_occupancy(pattern: str) -> tuple[str, ...]:
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown Lab 3 pattern {pattern!r}")
    desired = ["."] * len(LOCATIONS)
    for r, row in enumerate(PATTERNS[pattern]):
        for c, color in enumerate(row):
            desired[INDEX[f"G{r}{c}"]] = color
    return tuple(desired)

def _adjacency_degree(cell: str, desired: tuple[str, ...]) -> int:
    if cell not in GRID_CELLS:
        return 0
    r, c = int(cell[1]), int(cell[2])
    return sum(
        1 for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        if 0 <= r + dr < 3 and 0 <= c + dc < 3
        and desired[INDEX[f"G{r + dr}{c + dc}"]] != "."
    )

def _placement_risk(state: ColorState, destination: str) -> float:
    x, y = location_xy(destination)
    centers = [location_xy(loc) for loc in LOCATIONS
               if loc != destination and state.at(loc) != "."]
    if not centers:
        return 0.0
    risks = []
    for angle in (0., 45., 90., 135.):
        a = radians(angle)
        jaw = (cos(a), sin(a))
        risk = 0.0
        for sign in (-1., 1.):
            for opening in (0.021, 0.024, 0.026):
                fx, fy = x + sign * opening * jaw[0], y + sign * opening * jaw[1]
                for ox, oy in centers:
                    dx, dy = max(abs(fx - ox) - 0.020, 0.), max(abs(fy - oy) - 0.020, 0.)
                    distance = (dx * dx + dy * dy) ** .5
                    risk += max(0., 0.006 - distance) / 0.006
        risks.append(risk)
    return min(risks)

def _remaining_access_risk(state: ColorState, desired: tuple[str, ...]) -> float:
    return sum(
        _placement_risk(state, loc)
        for loc in GRID_CELLS
        if desired[INDEX[loc]] != "." and state.at(loc) != desired[INDEX[loc]]
        and state.at(loc) == "."
    )

def _ordered_moves(
    state: ColorState,
    desired: tuple[str, ...],
    accept_move: SymbolicMoveFilter | None,
    pattern: str | None = None,
    check_accessibility: bool = True,
):
    filled = [(loc, state.at(loc)) for loc in LOCATIONS if state.at(loc) != "."]
    empty = [loc for loc in LOCATIONS if state.at(loc) == "."]
    ranked = []
    old_h = _heuristic(state.occupied, desired)
    occupied_locations = tuple(loc for loc, _ in filled)
    for source, color in filled:
        if check_accessibility:
            pickup_heights = {
                option.height_offset_m for option in
                top_access_options(source, occupied_locations, phase="pick")
            }
            if not pickup_heights:
                continue
        for destination in empty:
            if destination in RED_SLOTS and color != "R":
                continue
            if destination in BLUE_SLOTS and color != "B":
                continue
            if source in STAGING_SLOTS and destination in STAGING_SLOTS:
                continue
            move = TaskMove(color, source, destination)
            if _center_trap_after(state, move):
                continue
            if check_accessibility:
                others = tuple(loc for loc in occupied_locations if loc != source)
                place_heights = {
                    option.height_offset_m for option in
                    top_access_options(destination, others, phase="place")
                }
                # A carried cube keeps its tool-to-block transform, so the
                # same contact height must work at pickup and placement.
                if not pickup_heights.intersection(place_heights):
                    continue
                if not is_recoverable(others + (destination,)):
                    continue
            if accept_move is not None and not accept_move(state, move):
                continue
            nxt = apply_symbolic_move(state, move)
            h_next = _heuristic(nxt.occupied, desired)
            proper_goal = destination in GRID_CELLS and desired[INDEX[destination]] == color
            replacing_good = source in GRID_CELLS and desired[INDEX[source]] == color
            immediate_risk = _placement_risk(state, destination)
            future_risk = _remaining_access_risk(nxt, desired)
            degree = _adjacency_degree(destination, desired) if proper_goal else 0
            sx, sy = location_xy(source)
            dx, dy = location_xy(destination)
            travel = hypot(sx - dx, sy - dy)
            # Every pick-and-place is one action. Clearance and travel order
            # equal-cost alternatives; they must not change A*'s objective.
            cost = 1.0
            rank = (h_next - old_h, not proper_goal, immediate_risk,
                    future_risk, replacing_good, -degree, travel, source, destination)
            ranked.append((rank, move, nxt.occupied, cost))
    ranked.sort(key=lambda item: item[0])
    for _, move, nxt, cost in ranked:
        yield move, nxt, cost

class TaskPlanner:

    def __init__(self, *, move_filter: SymbolicMoveFilter | None = None,
                 check_accessibility: bool = True):
        self.move_filter = move_filter
        self.check_accessibility = bool(check_accessibility)

    def plan(
        self,
        start: ColorState,
        pattern: str,
        *,
        max_expansions: int = 50_000,
    ) -> TaskPlan:
        if max_expansions < 1:
            raise ValueError("max_expansions must be at least 1")
        goal = _desired_occupancy(pattern)
        initial = start.occupied
        required_cells = tuple(loc for loc in GRID_CELLS if goal[INDEX[loc]] != ".")
        if self.check_accessibility and peel_order(required_cells) is None:
            # Any reversible construction has a valid reverse removal order.
            # The local model rejects a mutually blocked cluster before search,
            # without confusing an exhausted A*/IK budget with this evidence.
            from .access import (
                TOP_YAWS_DEG, TOP_HEIGHT_OFFSETS_M, PICKUP_OPENING_M, RELEASE_OPENING_M,
            )
            certificate = {
                "model": "original Panda finger collision hull and five fingertip pads",
                "scope": "supported conservative vertical face-grasp model; not arbitrary manipulation",
                "yaw_degrees": list(TOP_YAWS_DEG),
                "height_offsets_m": list(TOP_HEIGHT_OFFSETS_M),
                "pickup_half_opening_m": PICKUP_OPENING_M,
                "release_half_opening_m": RELEASE_OPENING_M,
                "required_cells": list(required_cells),
                "blocked_cells": list(blocked_core(required_cells)),
                "removal_order_exists": False,
                "test": "greedy obstacle removal; accessibility is monotone under removal",
            }
            return TaskPlan(
                pattern, "infeasible", (), 0, 1,
                "A mutually blocked cluster has no reversible construction under "
                "the supported top-grasp geometry. The assignment permits skipping it.",
                certificate,
            )
        if goal_reached(start, pattern):
            return TaskPlan(pattern, "solved", (), 0, 1)

        frontier: list[tuple[int, int, int, tuple[str, ...]]] = []
        sequence = count()
        g_score: dict[tuple[str, ...], float] = {initial: 0.0}
        parent: dict[tuple[str, ...], tuple[tuple[str, ...], TaskMove]] = {}
        h0 = _heuristic(initial, goal)
        heapq.heappush(frontier, (h0, h0, next(sequence), initial))
        expansions = 0

        while frontier:
            f, h, _, current = heapq.heappop(frontier)
            if abs(f - g_score[current] - _heuristic(current, goal)) > 1e-9:
                continue
            current_state = ColorState(current)
            if goal_reached(current_state, pattern):
                steps = []
                node = current
                while node != initial:
                    prev, move = parent[node]
                    steps.append(move)
                    node = prev
                steps.reverse()
                return TaskPlan(pattern, "solved", tuple(steps),
                                expansions, len(g_score))
            if expansions >= max_expansions:
                return TaskPlan(pattern, "search_limit", (),
                                expansions, len(g_score))
            expansions += 1
            for move, neighbor, move_cost in _ordered_moves(
                current_state, goal, self.move_filter, pattern, self.check_accessibility,
            ):
                candidate_cost = g_score[current] + move_cost
                if candidate_cost >= g_score.get(neighbor, float("inf")):
                    continue
                g_score[neighbor] = candidate_cost
                parent[neighbor] = (current, move)
                neighbor_h = _heuristic(neighbor, goal)
                heapq.heappush(frontier, (
                    candidate_cost + neighbor_h,
                    neighbor_h,
                    next(sequence),
                    neighbor,
                ))
        return TaskPlan(pattern, "no_solution", (), expansions, len(g_score))

def _show_state(state: ColorState) -> str:
    return "\n".join("  " + " ".join(row) for row in state.grid_rows())

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print-only Lab 3 symbolic task planner (no MuJoCo motion)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo", action="store_true", help="Plan all four patterns in sequence")
    group.add_argument("--goal", choices=PATTERN_ORDER, help="Plan one goal from the initial state")
    parser.add_argument("--max-expansions", type=int, default=50_000)
    args = parser.parse_args()

    planner = TaskPlanner()
    state = initial_state()
    targets = PATTERN_ORDER if args.demo or args.goal is None else (args.goal,)
    for goal in targets:
        print(f"\n=== Goal: {goal} ===")
        print("Start grid:\n" + _show_state(state))
        result = planner.plan(state, goal, max_expansions=args.max_expansions)
        print(f"Status: {result.status}; expansions: {result.expansions}; "
              f"visited: {result.explored_states}")
        if not result.succeeded:
            print(result.reason or "Goal not achieved; a search limit is not infeasibility.")
            continue
        for i, move in enumerate(result.moves, 1):
            print(f" {i:02d}. {move.label()}")
            state = apply_symbolic_move(state, move)
        print("Result grid:\n" + _show_state(state))
        assert goal_reached(state, goal), "Internal symbolic plan does not satisfy goal"

    print("\nTask planning with local gripper clearance: whole-arm IK and motion execution were not run.")

if __name__ == "__main__":
    main()
