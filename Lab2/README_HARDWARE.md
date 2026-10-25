# Lab 2 — Franka Panda real-hardware execution from the supplied MuJoCo flow

## Read this first

**This is a hardware interface and an off-robot trajectory exporter, NOT a report that the physical robot has already performed the experiment.** `frankapy` is not connected here and MuJoCo could not be installed in the package-building environment. Test the complete simulation, inspect the trajectories, check the physical setup with a TA, and perform attended, stage-gated trials. An exported path is **not** a guarantee of real-world collision-free operation.

The student assignment explicitly requires working with a partner; the partner must hold the accessible **physical emergency stop** whenever robot programs move the arm. Both people must stand outside the robot workspace. Stop immediately on suspicious motion/contact. **Do not disable the robot's virtual walls or change the fixed shelf setup.** After an e-stop, restart the control-PC Franka server per Lab 1, and recover the arm safely with the TA before continuing.

The real video must show **three actual pick/place/regrasp/transfer cycles with visibly different shelf positions**, not simulated block teleportation. The MuJoCo code respawns its only cube between simulation trials; on hardware the program prompts the operator to **manually replace the real cube on the table while the arm is stationary at HOME**. This is explicitly *between* trials and is not an extra robot motion.

## What was reused from your existing solution

- Directly calls `lab2.task.ManipulationTask.run_one()` for every trial. **Same 18 stages, same sequence, orientations, offsets, sampling, and `HOME`.** No new Cartesian pick sequence is substituted.
- Uses the original `lab2.ik.PoseIK`, `lab2.planning.JointPlanner`, `lab2.rrt.JointRRT`, `lab2.scene`, and `lab2.simulator.LabSim`. Collision-checked joint waypoints are exported from those same functions.
- Uses original seed logic (`seed + 7919 * trial_index`) and its separate planning seed (`+104729`). Left shelf X is sampled across 0.430–0.560 m, right shelf X across the same interval in reverse trial order. The right destination level is bottom, middle, top.
- **Only** `hardware/robot.py` sends actual commands to `FrankaArm`. This avoids installing `frankapy` in the local simulation environment, and avoids installing MuJoCo on the lab robot computer for playback.

### File roles and code execution flow

```text
Off-robot computer / MuJoCo Python >= 3.10
    hardware_main.py plan --seed 13 --trials 3
      -> hardware/plan.py CapturePlayer
      -> lab2/task.py ORIGINAL 18-step task
          -> lab2/ik.py          collision-checked pose IK
          -> lab2/planning.py    plans between joint vectors
          -> lab2/rrt.py         collision-checked RRT edges
          -> lab2/simulator.py   identical shelf/block/carry model
      -> outputs/hardware_plan_seed13.json

    hardware_main.py validate --plan outputs/hardware_plan_seed13.json
      -> hardware/validate.py    independent re-scan of saved edge samples

Real Franka lab robot PC / existing Lab 1 frankapy Docker (Python >= 3.8)
    hardware_main.py inspect   -> prints every planned joint motion and sampled xyz
    hardware_main.py execute   -> hardware/spec.py stage/sample/checks
      -> hardware/execute.py   18-stage sequencer + safety gates + log
        -> hardware/robot.py  FrankaArm() API -> physical arm and gripper
```

**What each module does:** `hardware/spec.py` locks the expected stage order and performs structural + joint continuity + sampled coordinate verification. `hardware/plan.py` captures each named motion from the original task, and simulates its attachment/release and next-trial reset *off-robot*. `hardware/validate.py` reruns arm/shelf collision sampling on every saved segment, detecting scene/model edits. `hardware/robot.py` wraps Lab 1 `FrankaArm` readouts, `goto_joints`, `open_gripper`, `goto_gripper(grasp=True)` and `stop_skill`. `hardware/execute.py` checks starting/ending joint errors and model-vs-robot tool transforms, scales down motion speed, pauses for permission, and appends a hardware event log. `hardware_main.py` is the one command-line entry point. `tests/test_hardware_mock.py` replaces the real robot with an instrumented fake to verify commands and stop conditions.

### Coordinates and reproducible samples

All coordinates are in **meters in the original MuJoCo robot-base/world frame**. The table cube center is `[0.5900, 0.0000, 0.0221]`; cube dimensions are 40×40×40 mm. Shelf XYZ means **block-center target**, not wrist/tool pose.

| Trial | Left shelf target center `(X,Y,Z)` m | Right target center `(X,Y,Z)` m | Right shelf |
|:--|:--|:--|:--|
| 1 | `(0.467475, 0.326395, 0.170500)` | `(0.551811, -0.324294, 0.170500)` | Bottom |
| 2 | `(0.493052, 0.326716, 0.170500)` | `(0.491378, -0.316866, 0.370500)` | Middle |
| 3 | `(0.547136, 0.323584, 0.170500)` | `(0.434984, -0.327175, 0.570500)` | Top |

These examples use `seed=13` and `trials=3`, not the old seed-0 JSON sample bundled in the original simulation archive. Use **`hardware_main.py inspect` as the ground truth for the actual exported plan**. Changing trial count will change X bin widths and therefore the sampled coordinates, even if seed is unchanged. Table location stays constant in every trial; shelf target positions differ.

**The important coordinate caveat:** Even if table and shelves have the identical nominal dimensions and positions, your real Franka base/tool frames, gripper finger geometry and block pose can differ slightly from the model. Hardware mode compares the observed `fa.get_pose()` against the simulated tool FK and aborts above 12 mm translation or 0.12 rad orientation error. **Those guard tolerances are stop conditions, not sufficient calibration accuracy for a 40 mm cube**; visually verify fixture alignment much more tightly *before* contact motions. Do not defeat this check to make the arm move. Ask the TA to diagnose frame/tool offsets if it rejects.

### Original 18 stages, in order

| Phase | Stages and operation |
|:--|:--|
| Top grasp from table | `01` table pregrasp → `02` descend to top grasp → `03` grasp → `04` lift |
| Place on single shelf | `05` left shelf preplacement → `06` place from above → `07` release → `08` retract |
| Side regrasp | `09` side approach from aisle → `10` side grasp → `11` grasp → `12` lift 70 mm → `13` retreat 130 mm |
| Move to right shelf | `14` approach from aisle (+Y side) → `15` place → `16` release → `17` retract |
| Finish | `18` collision-checked joint trajectory back to exactly `HOME = [0, -0.5, 0, -2, 0, 1.5, 0.8]` rad |

**Important:** Simulation's `attach()`/`release()` instantly repositions the cube. The real robot uses a *physical 40 mm grasp* (`fa.goto_gripper(width=0.040, grasp=True, speed=0.02, force=10.0)`), physical release (`fa.open_gripper()`), and checks gripper width/requests human confirmation before lifting. Physical grasp friction and contact are **not** proven by a width reading; the lab instructions permit gripper-malfunction footage but do not justify ignoring unsafe motions.

## End-to-end instructions

### A — Off-robot: verify your existing working MuJoCo simulation

Extract the supplied ZIP. Work from its `Lab2` folder (this ZIP already includes your full original simulation modules and unchanged Franka/shelf MJCF assets).

```bash
cd Lab2
python3 --version                  # use Python 3.10+ for original simulation modules
python3 -m pip install -r requirements.txt
python3 -m pytest -q
python3 run_lab2.py --check-scene --trials 3 --seed 13
LAB2_CONTINUOUS_SMOKE=1 python3 -m pytest -q tests/test_mujoco_integration.py
python3 run_lab2.py --trials 3 --seed 13 --record
```

Inspect `outputs/all_trials_continuous.mp4` and `outputs/results.json`. Require 3 successful, 18-stage trials before ever using physical hardware. MuJoCo's `--record` run is optional for **hardware execution** once an equivalent full simulation has already passed, but it is the best reference video.

### B — Off-robot: export, inspect and validate the exact joint trajectories

Still in `Lab2` on the MuJoCo-enabled computer:

```bash
python3 hardware_main.py sample --trials 3 --seed 13
python3 hardware_main.py plan --trials 3 --seed 13 --out outputs/hardware_plan_seed13.json
python3 hardware_main.py inspect --plan outputs/hardware_plan_seed13.json
python3 hardware_main.py validate --plan outputs/hardware_plan_seed13.json
python3 -m pytest -q tests/test_hardware_mock.py
```

**Do not proceed if `plan` or `validate` fails.** Any collision/IK failure means you must examine and fix the original simulation's target range/configuration; do not remove collision checks or force execution. Keep the entire `Lab2` code directory and `outputs/hardware_plan_seed13.json` together for reference. Only `hardware_main.py`, the `hardware` folder, and `outputs/hardware_plan_seed13.json` are required by the physical *execution* process; those files are also included in the complete package for convenience.

### C — The real robot: Lab 1 startup and server setup (in sequence)

1. **Get your partner and the TA's approval**, physically inspect the shelf fixtures, cube, tabletop and robot. Clear the workspace. Keep an e-stop operator next to the emergency stop *throughout motion*. Confirm physical axis orientation and cube coordinates against the simulation. Do not relocate shelves.
2. Turn on the Franka Control Interface and both PCs following **Lab 1, page 2**. Log in to the lab user PC with its lab-provided credentials.
3. In terminal 1, `ssh -X student@iam-<control-pc-name>` (replace with the name physically printed on the control PC); start `google-chrome` and open `http://172.16.0.2` (Franka Desk). Unlock the joints. **Blue** means program mode; see Lab 1 pages 2–3. Follow TA guidance if the UI or robot differs.
4. In a new user-PC terminal, start the ROS master:

   ```bash
   roscore
   ```

5. In another new user-PC terminal, start the **installed** `frankapy` Control PC bridge. The installed package uses `bash_scripts/start_control_pc.sh` (underscores are important):

   ```bash
   cd /home/student/Documents/frankapy
   bash ./bash_scripts/start_control_pc.sh -u student -i iam-<control-pc-name>
   ```

   Keep the bridge's spawned terminals alive. If your local Lab 1 setup has slightly different folder naming or flags, use the working command from that machine/TA instead; this package does not replace the already installed bridge.

6. Enter your lab's existing Franka-enabled Docker environment per **Lab 1, page 4**. On the User PC, first launch the Docker container, then attach a terminal:

   ```bash
   cd /home/student/16662_RobotAutonomy
   bash run_docker.sh
   # In a DIFFERENT User PC terminal:
   cd /home/student/16662_RobotAutonomy
   bash terminal_docker.sh
   ```

   Your actual Lab 1 repo spelling/location may differ; use the on-machine Lab 1 scripts. Verify that `python3 -c "from frankapy import FrankaArm"` works *inside the execution terminal*. You do **not** need to `pip install frankapy` into a new non-ROS environment.

7. Copy this package's `hardware_main.py`, `hardware/` and exported `outputs/hardware_plan_seed13.json` into the team source folder inside the existing Docker-mounted `src/devel_packages` folder. For example, if the mounted directory is `/home/student/16662_RobotAutonomy/src/devel_packages/TEAM_LAB2`, execute there. If Docker mount mapping differs, find it first and put the files in the shared directory; do not guess ROS container paths.
8. Make the arm start at exactly the **MuJoCo HOME** configuration `[0,-0.5,0,-2,0,1.5,0.8]`, not an arbitrary pose or necessarily `fa.reset_joints()`'s different home. Use the lab's approved, supervised guide-mode preparation. **This code intentionally does not move an arm from an unverified starting configuration to HOME**; it rejects the start instead. Have the TA confirm any necessary manual arm adjustment and gripper clearance.

### D — Physical dry run, one stage at a time

Inside the **Franka-enabled Docker terminal**, from the folder holding `hardware_main.py` and `hardware/`:

```bash
python3 hardware_main.py inspect --plan outputs/hardware_plan_seed13.json
python3 hardware_main.py execute --plan outputs/hardware_plan_seed13.json --trial 1 --ack-safety
```

- The program **first reads** current 7 joints and the real measured `franka_tool` pose. If any don't match the exported simulation HOME (0.060 rad max per joint, 12 mm translation, 0.12 rad orientation), it **sends no motion command**. Diagnose the mismatch; do not relax tolerances as a workaround.
- For trial 1, carefully place a 40 mm cube at the table start, verify the left `(0.467475,0.326395,0.1705)` and right `(0.551811,-0.324294,0.1705)` shelf targets, verify that the physical setup matches model and the e-stop is held, then type `YES` when prompted.
- The program first opens the gripper, then explicitly requests `YES` **for each of the 18 named stages**. Review clearance before moving into low grasps or shelves; do not allow another person to enter the work area. It executes RRT **joint** waypoints with Franka's `goto_joints(..., use_impedance=False, ignore_errors=False, ignore_virtual_walls=False)`, with durations stretched by ×1.8 and a 2-second minimum.
- At each grasp it uses a gentle 10 N limit and prompts for human confirmation of the actual block hold before any upward motion. At each segment end it checks the measured joint angle; at each stage end it checks measured tool pose. Error => stop/abort, **no automatic unsafe retreat**.
- At stage 18 it returns to the exact sim HOME configuration, not a fresh blind `reset_joints()` motion. Robot hold/release and collision behavior must be visually checked. It appends JSON lines to `outputs/hardware_run_log.jsonl`.
- If the cube is dropped, a grip jams, the arm is too near a fixture or you hear unusual contact, **press the emergency stop immediately**. Per Lab 1, stop the Franka server terminals, safely recover the robot and restart the server. Only rerun the failed trial once the task and pose have been corrected.

### E — Physical acceptance and final assignment video

Once trial 1 is physically satisfactory, run the other two one at a time (same fixed table pick location; each trial has a new right shelf height and different sampled shelf X):

```bash
python3 hardware_main.py execute --plan outputs/hardware_plan_seed13.json --trial 2 --ack-safety
python3 hardware_main.py execute --plan outputs/hardware_plan_seed13.json --trial 3 --ack-safety
```

For a **single continuous physical filming session** after all paths, grasps and fixture clearances have been demonstrated safely, start your phone/camera recording and run:

```bash
python3 hardware_main.py execute --plan outputs/hardware_plan_seed13.json --all --ack-safety --auto-stages
```

`--auto-stages` only removes **per-stage** `YES` prompts. It **does NOT remove** the pre-trial cube setup approval or the two grip confirmations per trial. The operator places the cube back on the table by hand **only after the prior trial has reached HOME and all arm motion has ceased**; then grants approval for the next trial. Do not enable `--auto-stages` before completing the stage-gated testing or without your TA's permission.

The camera should show (1) grasp from table, (2) placement on single shelf, (3) side regrasp, (4) move to right shelf, (5) release and return HOME; repeat with **bottom**, **middle**, **top** right shelves and distinct left/right shelf X. Keep your camera recording and make sure both whole shelves, the cube, and the robot are in view; your operator should stay outside the workspace. For the assignment submit the **real robot's** three-cycle video plus the code. You can record three separate approved runs and combine clips if a single filming session is impractical; do not portray manual cube replenishment as autonomous return-to-table motion.

### F — Shutdown

After safely completing all trials, do **Lab 1, page 7**: return robot to safe home under TA supervision, close the Control PC server terminals, use Franka Desk's shutdown function, wait for shutdown to finish, then switch off the physical FCI. Do not force the arm after e-stop.

## Limits and how to evaluate this package

- The same MuJoCo RRT/IK sequence is preserved, but physics-free `scripted_kinematic_follow_not_force_based_grasp` cannot predict real contact, slip, clamp force, or grasp success. Required operator confirmations are intentional.
- `LabSim.collision_reasons()` checks robot/shelf and carried-cube/shelf contacts, and does **not** comprehensively model every robot/unheld-cube finger interaction. MuJoCo's joint paths are not automatically certified by any physical sensor. Fixture calibration and progressive commissioning are required.
- Physical Franka `goto_joints` smooths each exported joint segment using its own min-jerk interpolation, not the same time stepping as MuJoCo. The joint *waypoint sequence*, stage sequencing and endpoints remain identical, and the commanded durations are slower.
- If your existing Lab 1 package uses different gripper force limits, finger pads, tool offsets or a different base/fixture transform, **review with the TA and recalibrate/regenerate** rather than ignoring checks. Strict frame mismatch and safe HOME checks exist to stop an unsafe test.
- In this environment **only the non-MuJoCo unit/mock tests were run**. A successful simulation run from the supplied archive is a user report, not a hardware acceptance test, and the attached original archive itself noted an inability to verify full MuJoCo integration in its build environment. Run the `plan` and `validate` commands where MuJoCo works, then perform real-world testing with a partner.

## Common failure messages

| Message | Likely reason and next action |
|:--|:--|
| `No module named mujoco` on `plan` or `validate` | Use the MuJoCo-enabled off-robot Python environment; playback-only `execute` does not need MuJoCo |
| `from frankapy import FrankaArm` fails | Use the existing Lab 1 ROS/Docker terminal after starting the Control PC bridge |
| `Initial HOME` joint error | Real arm not in simulator's `[0,-0.5,0,-2,0,1.5,0.8]`; supervised positioning required; not safe to jump automatically |
| `frame mismatch` | Real robot-base/tool calibration is not the model's FK; check arm, gripper tool offset, frame alignment and robot identity before motion |
| `Collision during capture`, `No collision-free IK`, `RRT start is in collision` | The seed's exact sampled target is not verified in MuJoCo; inspect physics scene and regenerate plan. Do not disable collision checks |
| `Unexpected grasp width` | Block was not in fingers, gripper jammed, or commanded width inappropriate; stop and inspect with partner |
| `max joint error` or `ABORTED` | Motor tracking, controller fault, interruption or e-stop; do not force recovery inside this script |

**Sources:** Supplied Lab 2 PDF (task/safety/video), supplied Lab 1 PDF (Franka startup, ROS/Docker, state, Cartesian/joint/gripper API), supplied `lab2_sim_sol.zip` (identical simulation sequencer, random shelf sampler, model, geometry). Franka method signatures were cross-checked with the [frankapy API documentation](https://iamlab-cmu.github.io/frankapy/frankaarm.html).
