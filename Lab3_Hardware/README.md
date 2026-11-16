# Lab 3 on the real Franka (frankapy)

This package runs **exactly** the motions of the MuJoCo simulation
(`../Lab3_Code`) on the real robot:

1. **plan**: runs the same `run_lab3.execute_patterns` with the same seed
   in MuJoCo and records every move (joint path + timing) and every grip into
   `outputs/hardware_plan.json`.
2. **validate**: independently replays that plan in MuJoCo. It collision-checks
   every joint state at 0.01 rad, re-executes the grips and checks where every
   cube ends up, then writes `outputs/hardware_plan_validation.json`.
3. **execute**: on the robot PC, streams each move to frankapy's dynamic joint
   skill with the same quintic time profile the simulator played. It checks
   joints and tool frame against the simulation after every move and stops on
   any deviation.

Tests verify that the simulation and the hardware plan are identical, not just
similar:

| test | what it proves |
|---|---|
| `tests/test_capture_matches_sim.py` | all 123 events of a `run_lab3.py` run are bit-identical to the saved plan |
| `tests/test_trajectory_matches_sim.py` | streamed q(t) equals the simulator playback (max error 0.0 rad over 89,776 samples) |
| `tests/test_dry_run.py` | the hardware executor, run against MuJoCo, rebuilds every pattern (stream and waypoint modes) |

Symbolic planning is deterministic. `run_lab3.py` and `plan` both pin
`PYTHONHASHSEED=0`, because the HW3 planner iterates over Python sets.

## Files

```
Lab3_Hardware/
├── hardware_main.py      CLI: plan | validate | inspect | layout | measure | home | check-hardware | execute
├── lab3_hw/
│   ├── spec.py           plan format + checks, gripper widths, joint limits   (robot PC)
│   ├── trajectory.py     q(t) identical to lab3/playback.py                   (robot PC)
│   ├── robot.py          frankapy: streaming, gripper, guide mode            (robot PC)
│   ├── execute.py        supervised execution with safety checks            (robot PC)
│   ├── capture.py        plan = recorded simulation run                     (MuJoCo PC)
│   ├── validate.py       independent collision replay                       (MuJoCo PC)
│   ├── simrobot.py       MuJoCo stand-in for the robot, for dry runs        (MuJoCo PC)
│   └── simlink.py        finds ../Lab3_Code (or $LAB3_SIM_ROOT)
├── tests/
└── outputs/              hardware_plan.json, hardware_plan_validation.json, run logs
```

The robot-PC files need only `numpy` plus the lab's frankapy/ROS install, and
are Python 3.8 compatible. MuJoCo is **not** needed on the robot PC.

## Gripper: never fully open

The Lab 1 `fa.open_gripper()` opens to 8 cm, which hits the neighbouring cubes
in the grid. This package only ever commands:
- **6.0 cm** (pre-grasp; = 2 × the 3 cm per-finger sim opening), and
- a **4 cm grasp**, accepted only if the measured width is 3.2–4.8 cm.

---

# How to run on the hardware

## A. On your own PC (MuJoCo), before going to the lab

```bash
source ~/robot_autonomy/bin/activate
cd ~/CMU_MRSD_Sem1/Robot_Autonomy/Homework/LAB3/Lab3_Hardware

python3 hardware_main.py plan          # ~2 min
python3 hardware_main.py validate      # ~10 s, must print PASS (incl. reset->HOME move)
python3 hardware_main.py inspect       # symbolic plans + expected tables
python3 hardware_main.py layout        # coordinates of all 17 cells

# Rehearse the exact hardware code path against MuJoCo (watch it in the viewer):
python3 hardware_main.py execute --ack-safety --dry-run sim --viewer --auto-stages --time-scale 1.0

python3 -m tests.test_trajectory_matches_sim
python3 -m tests.test_dry_run
python3 -m tests.test_capture_matches_sim   # ~2 min
```

## B. In the lab: start the robot (Lab 1 procedure)

1. FCI switch ON, then power on the Control PC and User PC. Log in as `student`.
2. `ssh -X student@iam-<name>`, then `google-chrome`, open `172.16.0.2`, and
   **unlock the joints**. The lights go **blue**.
3. In a new terminal: `roscore`.
4. In another terminal:
   `cd /home/student/Documents/frankapy && bash ./bash_scripts/start_control_pc.sh -u student -i iam-<name>`
   (three terminals open).
5. Copy this package into your team folder (only `Lab3_Hardware/` is needed on
   the robot PC, **including `outputs/`**):
   `~/16662_RobotAutonomy/src/devel_packages/<team>/Lab3_Hardware`
6. `cd ~/16662_RobotAutonomy && bash run_docker.sh`, then in a new terminal
   `bash terminal_docker.sh`, and `cd` to the package inside the container.
7. Reset the arm: `python3 reset_joints.py` (frankapy reset pose
   `[0, -π/4, 0, -3π/4, 0, π/2, π/4]`). The plan starts from a different HOME,
   `[0, -0.5, 0, -2.0, 0, 1.5, 0.8]` (the Lab 2 HOME, also used in simulation).
   Move there with the validated move:
   ```bash
   python3 hardware_main.py home --ack-safety
   ```
   `validate` collision-checks this move too. It refuses to run if the arm is
   not at the reset pose.

**Safety, every time:** one person keeps a hand on the E-STOP, and nobody is
inside the table edges while the robot runs. After *any* E-STOP, close the three
control-PC terminals, rerun `start_control_pc.sh`, and re-check HOME.

## C. Calibrate the table once per session

The plan assumes the grid centre (cell `g11`) is at **x = 0.500, y = 0.000 m**
in the robot base frame, the storage columns are at **y = ±0.17 m**, and the
table top is at z ≈ 0.
1. Run `python3 hardware_main.py layout` and mark the 17 cube centres on the
   table with tape (5.5 cm grid pitch, storage cubes 8 cm apart).
2. Put a cube on `g11`, then run (robot PC, inside docker):
   ```bash
   python3 hardware_main.py measure --guide 30 --cell g11
   ```
   For 30 s the arm is hand-guidable. Bring the 6 cm-open fingers down around
   the cube at grasp depth, centred, with the fingers parallel to the grid
   lines. The command prints the offset between the real cube and the plan.
3. If any offset is above **3 mm**, copy the printed JSON into
   `Lab3_Code/table_layout.json` on your MuJoCo PC. Then re-run **plan** and
   **validate**, and copy `outputs/` to the robot PC again. Alternatively, move
   the tape so the offset is zero. Plans are stamped with the layout, so a stale
   plan is rejected.

## D. Read-only check (no motion)

```bash
python3 hardware_main.py check-hardware
```
This checks that the validation record matches the plan, that the robot is at
HOME, and that the frankapy `franka_tool` frame matches the simulated tool
frame (within 1 cm / 0.1 rad). If the tool frame check fails, stop: the
frankapy EE offset differs from the sim (`0.1034 m` from the flange) and must
be fixed first.

## E. Execute

First time: one pattern, slow, with a prompt before every move.
```bash
python3 hardware_main.py execute --ack-safety --only pickaxe --time-scale 2.0
```
The program shows the required cube arrangement and waits for `YES`. It then
asks for `YES` before every move, after every grasp (check the cube is held),
and at the end of the pattern (check the table).

Full run for the video (the simulation sequence, axe is skipped automatically):
```bash
python3 hardware_main.py execute --ack-safety --auto-stages --time-scale 1.5
```
- `--time-scale 1.0` is simulation speed; larger is slower. It cannot go below 1.0.
- `--start box` resumes from a pattern. Arrange the cubes as printed (that
  pattern's start table).
- `--motion waypoints` falls back to Lab 2 style `goto_joints` per waypoint
  (stops at each waypoint). Use it only if the dynamic joint skill misbehaves.
- Every event is logged to `outputs/hardware_run_log.jsonl`.

**On any abort** (joint error > 0.03 rad, tool error > 1 cm, missed grasp,
operator says no): the program stops the skill and leaves the arm where it is.
It does **not** retreat automatically. Recover under supervision with guide
mode, then run `reset_joints.py` and `hardware_main.py home --ack-safety`,
re-arrange the cubes to a pattern's start table (printed by `inspect`), and
resume with `--start <pattern>`.

## F. Shut down (Lab 1)

`python3 reset_joints.py`, then close the control-PC terminals. In the web UI,
choose **Shutdown**, wait about a minute, then switch the FCI off.

## What can still differ from simulation

- **Grasping is real.** In simulation the cube is attached to the gripper; on
  the robot it is held by friction. The 4 cm grasp width check catches misses,
  not slips.
- **The dynamic joint skill has small tracking lag.** After each move the
  program settles the arm with a 1 s `goto_joints` if the error is
  0.008–0.03 rad, and aborts above 0.03 rad.
- **Release drops the cube 3 mm** (the planned placement gap). In simulation
  the cube is placed exactly on the table.
- Real-world alignment of the table and cubes relies on calibration step C.
