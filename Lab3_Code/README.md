# Robot Autonomy — Lab 3 (Task Planning)

Rearrange 4 red + 4 blue cubes into the four 3x3 patterns
(pickaxe, axe, box, bow) with a HW3 A* symbolic planner on top of the Lab 2
IK + RRT-Connect motion stack, in MuJoCo first and then on the Franka.

## Setup and running

```bash
source ~/robot_autonomy/bin/activate        # needs MuJoCo >= 3.1 (system python3 has 2.2.1)
cd Lab3_Code

python3 -m tests.test_grasp_axis            # ~3 s   finger rule holds at the motion level
python3 -m tests.test_symbolic              # ~2 min symbolic plans for all patterns
python3 run_lab3.py --plan-only             # print the chained symbolic plans, no sim

python3 run_lab3.py                         # MuJoCo viewer, all four patterns (~4 min sim)
python3 run_lab3.py --patterns box bow      # subset (starts from the Figure 1 setup)
python3 run_lab3.py --control torque        # PD + gravity-comp tracking instead of kinematic
MUJOCO_GL=egl python3 run_lab3.py --headless --record   # -> outputs/lab3_all_patterns.mp4
```

Every run writes `outputs/results.json`, with one entry per pattern: status
(`done` / `infeasible` / `failed`), the symbolic actions, the executed actions
with the grasp axis actually used, the replan count, the final grid, and any
cube more than 1.2 cm from its cell. The symbolic planning step (~2 min,
mostly proving axe infeasible) runs before the viewer opens.

## Package layout

```
Lab3_Code/
├── run_lab3.py              entry: pre-plan all patterns -> execute -> log/record
├── shelf_heights.json       (Lab 2) measured shelf heights; shelves stay in the scene
├── franka_emika_panda/      (Lab 2) MJCF + meshes; lab3_runtime_scene.xml is generated
├── lab3/
│   ├── config.py            Settings, HOME, grip widths, top_grasp_R(yaw), GRASP_YAW
│   ├── layout.py            grid/slot coordinates, finger-neighbour table, PATTERNS
│   ├── scene.py             shelves + 9 grid tiles + 8 free cubes + tool site
│   ├── symbolic/
│   │   ├── SymPlanner.py    (HW3) A* STRIPS planner, unchanged
│   │   ├── domain.py        pick/place(cell, colour, axis) with finger clearance
│   │   └── problem.py       occupancy -> init facts, pattern -> goal, plan roll-out
│   ├── simulator.py         Lab 2 LabSim refactored for 8 named cubes + held cube
│   ├── world.py             WorldModel: cell -> cube, occupancy, drift check
│   ├── primitives.py        fully pre-planned pick / place (IK, RRT-Connect, straight moves)
│   ├── executor.py          symbolic actions -> primitives, axis fallback, replanning
│   ├── ik.py                (Lab 2) least-squares pose IK + collision check (uses sim.fk)
│   ├── rrt.py               (Lab 2) RRT-Connect + shortcutting
│   ├── planning.py          (Lab 2) JointPlanner + chain_move for straight-line moves
│   ├── trajectory.py        (Lab 2) quintic interpolation
│   ├── playback.py          (Lab 2) viewer / video, + smooth polyline playback, grid camera
│   └── geometry.py, shelf_heights.py   (Lab 2)
└── tests/
    ├── test_symbolic.py     pattern feasibility regression
    └── test_grasp_axis.py   crowded-row grasp collides on y, is free on x
```

## Key design decisions

**Geometry.** The grid is centred at (0.50, 0) with a 5.5 cm pitch. The Lab 3
pre-grasp opening is 3 cm per finger, so a pad clears its own cube by 1 cm
but enters a neighbouring cell (the neighbour's face is 3.5 cm from centre).
This is the collision the handout requires. The storage columns sit at
y = ±0.17 with 8 cm spacing, so cubes there can be grasped on either axis
(at 7 cm the finger mesh, which reaches ~5.3 cm out, hit the neighbouring cube). Grid row 0
is the row farthest from the robot, and red is on the robot's left (+Y).

**Symbolic domain.** The domain uses colour-level facts (`has(cell, colour)`)
instead of block identities. This removes the 4!·4! symmetry and keeps
A* within the HW3 50k-expansion budget. Each action carries the closing
axis `x`/`y`, which becomes the grasp yaw (`config.GRASP_YAW`). Both `pick` and
`place` require the two finger-neighbour cells on that axis to be `empty`. The
pseudo-cell `out` is always empty. This one rule enforces the finger-collision
constraint and also stops the planner from dropping a block where it can never
be picked again, such as the middle of a ring.

**Colour-matched storage.** `place` also requires `accepts(cell, colour)`. Grid
cells accept either colour, but the red column (`sr*`) accepts only red and the
blue column (`sb*`) only blue. This holds between patterns too, and for the
executor's recovery "park" step. `tests/test_symbolic.py` asserts it.

**Feasibility result** (chained from the Figure 1 start):

| pattern | result |
|---|---|
| pickaxe | 10 actions |
| axe | **infeasible → skipped** (in the target, only g21 can go in last, and the remaining four cannot be built in any order) |
| box | 10 actions |
| bow | 10 actions |

## Status

Verified on 2026-11-15 in both `kinematic` and `torque` control. Pickaxe, box
and bow are built with no motion failures and no replans. Axe is skipped as
infeasible. All cubes end within 1.2 cm of their cells.

How the pieces fit together at run time:
1. `run_lab3.py` chains the A* plans through the patterns.
2. `TaskExecutor` checks that the world model matches the start state each
   plan expected, and replans if it doesn't.
3. Each `pick`/`place` action is planned in full by `Primitives`: hover IK
   (yaw and yaw ± π, keeping the solution nearest the current joints), then
   RRT-Connect, then a straight descent, then the grip, then a straight lift
   or retreat. Only then does it run.
   - The hover pose is polished to under 0.2 mm before the descent starts.
   - Straight moves use continuous differential IK (damped least squares,
     2 mm steps). The 7th DoF holds the arm's starting posture through an
     exact null-space projector, so the elbow and wrist move monotonically.
     Independent per-waypoint IK solves used to land on different postures
     and made those joints jitter.
4. If the symbolic axis fails at the motion level, the other axis is tried
   when the finger rule allows it.
5. If a place still fails, the cube is parked in a free slot of its own
   colour's storage column and the pattern is replanned (up to 3 times).

## Hardware

The real-robot package is `../Lab3_Hardware`. Its README covers planning,
validation, calibration (`table_layout.json`) and execution. It records this
simulation's run (same code, same seed) and streams the identical trajectories
to the Franka with frankapy. `run_lab3.py` pins `PYTHONHASHSEED=0` so the
symbolic plans are the same every run.

**Performance.** The HW3 planner unifies preconditions against the whole
state, so each pattern takes about 5–60 s. If that gets annoying, cache plans per
(occupancy, pattern), or precompute the static `fn` lookup inside `domain.py`.
