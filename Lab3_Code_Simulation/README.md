# Lab 3: task planning with close-spaced blocks

This is the corrected simulation stack for the supplied Lab 3 handout. It retains
the original Panda collision meshes, eight 40 mm cubes, 50 mm grid pitch,
all shelves, and the exact four color patterns from Figure 2. Assignment runs
use vertical top grasps with selectable wrist orientation by default.

## Run it

From the extracted `Lab3` directory, in your existing Python environment:

```bash
python3 -m pip install -r requirements.txt
python3 run_lab3.py --check-scene
python3 run_lab3.py --check-simulator
python3 -m lab3.task_executor --all --view --log outputs/lab3/task_run.json
```

The default returns the cubes to staging with robot transfers after each pattern.
To move directly between patterns:

```bash
python3 -m lab3.task_executor --all --no-reset --view --log outputs/lab3/transitions.json
```

For one goal use `--goal pickaxe`, `--goal axe`, `--goal box`, or `--goal bow`.
`--top-only` is accepted and is already the default. Close the viewer after the
completion summary; closing it while the robot is moving stops execution.

## What the assignment expects

The handout explicitly permits skipping a pattern if no feasible plan exists.
Its close spacing is intentional: widening the grid would remove the grasp
orientation problem that the assignment asks you to solve.

| Figure 2 | Pattern rows | Result in the supported top-grasp model |
|---|---|---|
| 1: pickaxe | `BBB / .R. / .R.` | Valid assembly/removal order; 5 transfers from staging |
| 2: axe | `BB. / BR. / .R.` | Skip the mutually blocked 2x2 cluster |
| 3: box | `BRB / R.R / BRB` | Valid order; put corners in before obstructing edges; 8 transfers from staging |
| 4: bow | `BR. / B.R / BR.` | Valid assembly/removal order; 6 transfers from staging |

The axe leaves `G00, G01, G10, G11` mutually blocked after its tail is removed.
Every cube in that cluster has neighbors on both possible jaw axes. The planner
tests the original finger convex hull and five fingertip pads, including vertical
and aperture sweeps. It emits `skipped_top_grasp_infeasible` with a scoped
certificate before moving a block for this goal.

This certificate applies to the configured conservative library of face-aligned
vertical top grasps (yaw 0/90/180/270 degrees, contact offsets 0/7/11 mm,
48 mm pickup gap and at least 43 mm release gap). It is not a theorem about
every possible continuous grasp, pushing maneuver, or different gripper.
Experimental `--side-only` and `--tilted-only` options remain diagnostic modes;
they are not used in the assignment's default run.

## Why the old version failed

1. Release silently changed the carefully selected partial opening to the full
   80 mm opening. That made fingers collide with a neighboring cube on retreat.
   Release now preserves the preplanned aperture until the hand is above the cubes.
2. Pickup approached adjacent blocks at full opening. Candidates now carry a
   consistent 48 mm approach aperture, and jaw sweeps are checked.
3. Symbolic occupancy was mistaken for manipulation feasibility. A cached
   geometry predicate now checks source pickup, destination placement, compatible
   contact height, and recoverability before A* accepts a move.
4. A box corner can be temporarily blocked while an edge remains removable.
   The planner therefore checks for a complete removal order instead of requiring
   every cube to be immediately graspable. This gives useful block sequencing.
5. A failed transfer from one source could blacklist a destination for other
   sources. Rejections now apply only to the failed state/action and expire after
   successful motion changes the arm state.
6. A collision-free RRT path could become colliding after smoothing. Joint
   trajectories now use rest-to-rest quintics on each checked straight edge.
   Pickup, lift, place, jaw opening, released-cube retreat, and opening above the
   blocks are preflighted before a complete transfer is selected.

The original `task_run.json` also recorded a viewer closure during the axe
attempt. That execution stop did not prove infeasibility. New reports separate
geometric model infeasibility, A* budget exhaustion, rejected motion preflights,
and execution failures. An unexplained skipped/no-plan result does not count as
a successful validation run.

## Lecture connection and optimization

Lecture 11's action preconditions/effects and semantic attachment idea
(`CanGrasp`, `Reachable`) are implemented as task-action guards plus IK/RRT checks.
Lecture 10's hybrid manipulation modes are reflected in approach, close,
attach, lift, carry, place, release, and retreat, with contact allowances specific
to the mode. Every completed action is checked against observed cube locations.

A* minimizes the number of transfers in its discrete model using unit action
cost and the admissible maximum of missing and misplaced goal cubes. Clearance
and travel distance order equal-cost alternatives. This does not claim globally
minimum execution time or optimal motion across the entire four-goal sequence.

## Validate

Run the full headless suite on the machine where MuJoCo is installed:

```bash
python3 validate_lab3.py --all --log outputs/lab3/validation.json
```

It tests all goals with autonomous reset, direct transitions, and placement plus
pickup next to a neighbor on all four sides. It checks exact final grids,
top-only execution, neighbor poses, and reset completion. Only achieved goals
or explicitly certified top-grasp infeasibility count as expected outcomes.
Search or motion budget exhaustion is a failure.

Checks without MuJoCo (NumPy/SciPy are still required):

```bash
python3 validate_lab3.py --offline --log outputs/lab3/offline_validation.json
python3 -m lab3.task_planner --demo
```

The offline suite covers the exact handout patterns, all 512 grid occupancy
arrangements, adjacent-direction grasp orientation, removal certificates,
observation errors, release aperture preservation, gripper sweeps, state
restoration after failed preflight, preplanned release reuse, and trajectory
containment/speed.

## Verification limits

Offline tests and geometry-aware task planning were run while preparing this
correction. Full MuJoCo execution was attempted but could not start because the
runtime has no MuJoCo package and package download was blocked. The bundled
`outputs/lab3/validation.json` records those checks as failed due to the missing
dependency; it does not claim completed robot trajectories. The separate
`offline_validation.json` and `task_geometry_validation.json` record the checks
that actually ran. Run the full validation command above to obtain simulation
evidence on your machine.

Execution remains kinematic MuJoCo with scripted attachment, as in the uploaded
project. Cubes move through the planned tool attachment; there is no goal-state
teleport or simulator reset used to produce a pattern. Force-controlled grasp
physics and real-robot execution are not implemented or validated by this code.
