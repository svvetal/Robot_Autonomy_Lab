# Robot Autonomy — Async Lab 2, one continuous MuJoCo session (v3)

This version uses the uploaded shelf environment and Franka model to perform **top grasp from table → left shelf placement → side regrasp → transfer onto one of three right shelves → release → return home**, three times **in one MuJoCo process / viewer / simulation timeline**. It builds on the previous v2 shelf-contact fix; the original shelf geometry is never moved or resized.

## Quick start

From the extracted `Lab2` directory, in your existing MuJoCo Python environment:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python preflight.py --trials 3 --seed 13
python run_lab2.py --check-scene --trials 3 --seed 13
python run_lab2.py --trials 3 --seed 13
```

To make **one continuous MP4** covering all three trials:

```bash
python run_lab2.py --trials 3 --seed 13 --record
```

Outputs:

- `outputs/all_trials_continuous.mp4` — one uninterrupted MP4 (with `--record`).
- `outputs/placements.json` — exact table of target positions and trial seeds.
- `outputs/results.json` — stage-by-stage log for successful trials and failures.
- `outputs/offline_preflight.json` — optional kinematics screening (NOT MuJoCo execution).

For Linux headless recordings, run with a suitable rendering backend, e.g. `MUJOCO_GL=egl python run_lab2.py --trials 3 --seed 13 --headless --record`. Headless **without** `--record` makes no visible output; the pipeline and JSON logs still run.

## What happens between trials?

**The MuJoCo scene, viewer, simulation data, robot, and elapsed clock are NOT restarted.** The arm finishes each trial at `HOME`, then the next trial begins. The starter scene has **only one cube**: after trial 1 and 2, we explicitly **reposition only that cube** from the destination shelf to the table, then start the next table pickup. You will see the cube reappear on the table at the transition. This repositioning is a **scripted trial setup**, not physically grasping it again and bringing it back. It is only done between trials, with the gripper open and arm at home; it cannot run with an attached cube or arm far from home. This is the only practical way to repeat identical table-start tasks with this single-cube scene without adding two more cube objects or an extra reverse-transfer routine.

This is **one uninterrupted simulation and one recorded video**, but it is **not** proof of continuous physical block handling across the boundaries. For a future hardware implementation, the next trial's block should be manually replenished by authorized operators, or the planner should perform a separately validated return-to-table action. No real-robot commands are in this package.

### Placement coordinates and horizontal sampling

In the supplied scene, shelf **length runs along world X**. Previous versions varied X only slightly (`0.480–0.545 m`), which looked almost fixed. This version chooses a **random X coordinate in a larger interval** `[0.430, 0.560] m` on **both** sides. For three trials, it divides the range into three separate horizontal bands and samples randomly inside one band per trial. That ensures visibly distinct positions without hardcoding a single location. On the left, bands go from low X to high X; on the right, high X to low X. World Y varies slightly at the aisle-facing edge so side regrasp remains possible, and right-shelf Z selects bottom, middle, then top (repeating if >3 trials).

Settings are in `lab2/config.py`:

```python
left_x  = (0.430, 0.560)   # meters, horizontal across single shelf
left_y  = (0.311, 0.329)   # front edge
right_x = (0.430, 0.560)   # meters, horizontal across destination shelves
right_y = (-0.329, -0.311)
```

The cube half-width is `0.020 m`. Sampling inside the configured X intervals only checks the numeric boundaries; full robot/world collision checks happen when finding IK and an RRT path. The static `preflight.py` uses an independent FK model to screen sampled poses but **cannot certify collision-free MuJoCo execution**. If MuJoCo reports a collision for a particular location, inspect the reported stage/contact and tighten the intervals; never disable collision checking to force a demo.

Example placements from `--trials 3 --seed 13`:

| Trial | Left X (m) | Destination X (m) | Destination shelf |
|---|---:|---:|---|
| 1 | 0.4675 | 0.5518 | Bottom |
| 2 | 0.4931 | 0.4914 | Middle |
| 3 | 0.5471 | 0.4350 | Top |

## Exactly what `--seed` means

`--seed` controls a **pseudorandom number generator**, not MuJoCo speed or a physical robot setting. It lets you reproduce the same sampled locations and random planning choices. For example:

- `--seed 13`: draws the same positions and stochastic planning starts for every identical run with the same code/configuration.
- `--seed 42`: draws a different (but reproducible) set of positions and search samples.
- Internally, trial `i` (0-based) uses `placement_seed = base_seed + 7919*i`. The planner/IK draw from a **different** stream starting at `placement_seed + 104729`, so additional RRT iterations in trial 1 do not silently change trial 2's shelf positions.
- **Stratification depends on the total trial count:** the first placement with `--trials 1` need not equal the first with `--trials 3`, even for the same base seed.
- A seed does not guarantee reaching the target or avoiding collisions; those must be checked independently. Physics/controller/platform differences can also affect actual playback.

You can preview sampled locations **without launching MuJoCo**:

```bash
python run_lab2.py --check-scene --trials 3 --seed 13
python run_lab2.py --check-scene --trials 3 --seed 42
```

## Design and evaluation

```text
run_lab2.py           <-- one LabSim and one Player created OUTSIDE the trial loop
  |-- build_placements: independent, seeded, horizontal stratification
  |-- trial loop:
       |-- trial 1: initial scene / trials 2+ prepare_next_trial() [cube ONLY]
       |-- lab2/task.py: 18 named manipulation stages (NO sim.reset())
       |    |-- lab2/ik.py: collision-checked, multistart pose IK
       |    |-- lab2/planning.py -> lab2/rrt.py: bi-RRT + shortcuts
       |    `-- lab2/playback.py: quintic playback, gripper holds, video frames
       `-- outputs/results.json: completed stage records and failure details
  `-- ONE viewer and optionally ONE movie for all trials
```

- `lab2/scene.py`: preserves original shelf geometries; fixes malformed starter names and adds tool site.
- `lab2/simulator.py`: live state vs scratch MuJoCo collision checking, gripper script, and **cube-only between-trial preparation**. Sim clock never reset in the loop.
- `lab2/config.py`: all physical/planning parameters, seed definitions, target samplers.
- `lab2/collisions.py`: tightly limited shelf-support-contact exception during side regrasp; rejects deep penetration and robot/shelf collisions.
- `tests/test_continuous_trials.py`: seeds, independent target sampling, monotonic simulated session, and no task-level resetting.
- `tests/test_mujoco_integration.py`: real MuJoCo tests, skipped without `mujoco`; optional full 3-trial smoke test (see below).

## Acceptance checks on your MuJoCo machine

1. `python -m pytest -q` should pass, including MuJoCo-specific non-smoke tests.
2. `LAB2_CONTINUOUS_SMOKE=1 python -m pytest -q tests/test_mujoco_integration.py` executes all 3 full tasks (no viewer) and verifies a monotonic clock.
3. `python run_lab2.py --trials 3 --seed 13 --record` should complete 3 × 18 stages in **one** window and create `outputs/all_trials_continuous.mp4`.
4. Watch the entire MP4: cube is picked top-down, set near edge, regrasped from side, lifted off shelf, transported to right shelf, arm returns home, cube reappears on table for next trial; repeat 3 times with distinct X targets.
5. Check each `outputs/results.json` trial has `status: executed_in_mujoco` and increasing `sim_time_end_s`. Any exception is a real failure, not a silently completed trial.

**Verification boundary:** This archive was built where the `mujoco` Python package is unavailable. The pure Python tests and independent offline IK checks ran; **a full MuJoCo simulation of all three trials could not be executed here**. In particular, robot/shelf collision feasibility at newly sampled extremes, physical grasps, viewer handling and rendered MP4 need local evaluation. The default `--control kinematic` uses a **scripted cube attachment** for reproducible visualization, not a friction-validated grasp. `--control torque` is experimental and unverified. This archive targets simulation only, not hardware.

**Lab safety:** The lab instructions require a partner and an emergency-stop operator when using physical robots. Do not directly transfer kinematic or scripted grasping behavior to the physical arm.
