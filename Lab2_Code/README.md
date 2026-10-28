# Robot Autonomy — Lab 2

This project performs cube pickup, placement on the left shelf, side regrasp,
transfer to a right shelf, and return to home. It includes a MuJoCo planner and
a FrankaPy execution interface. The three right-shelf levels are used in order.

## Configuration

Edit `shelf_heights.json` to set the measured **top surface height** of each
shelf, in **millimetres** relative to the Franka base-frame origin:

```json
{
  "left_bottom_mm": 139.0,
  "right_bottom_mm": 146.0,
  "right_middle_mm": 336.0,
  "right_top_mm": 542.0
}
```

The values above are the supplied configuration. Confirm them against the
physical setup before using the robot. `lab2/scene.py` shifts the boards and
their local supports in Z, while `lab2/config.py` uses the same measurements
for cube-centre placement targets. X/Y positions and the remaining rack
geometry are unchanged.

## Simulation

From this `Lab2` directory (Python 3.10+ recommended):

```bash
python3 -m pip install -r requirements.txt
python3 run_lab2.py --check-scene --trials 3
python3 run_lab2.py --trials 3 --seed 13
```

To record a continuous video, add `--record`. To run without a viewer, use
`--headless`; headless video capture may require `MUJOCO_GL=egl` on Linux.
Simulation outputs, including the generated runtime XML, are created on demand.

## Hardware planning

Generate the trajectories on the MuJoCo computer and validate them against
the calibrated shelf scene:

```bash
python3 hardware_main.py plan --trials 3 --seed 13
python3 hardware_main.py inspect
python3 hardware_main.py validate
```

This generates `outputs/hardware_plan_measured.json` and
`outputs/hardware_plan_validation.json`. They are not included in the source
submission because they must be regenerated for the active measurements.
Whenever `shelf_heights.json` or the model changes, regenerate and revalidate
the plan. Hardware execution rejects plans whose measurement or validation
record does not match.

## Franka execution

The execution computer must have a working, lab-approved FrankaPy/ROS
installation. Copy `hardware_main.py`, `hardware/`, `shelf_heights.json`, and
the two validated plan JSON files into the same relative layout. Do not run
`plan` or `validate` in an environment without MuJoCo.

First, perform the read-only joint and tool-frame check:

```bash
python3 hardware_main.py check-hardware
```

After verifying the physical rack, the robot-base and tool frames, the starting
cube pose, clearance, and the emergency-stop procedure with the lab supervisor,
execute one trial with a designated emergency-stop attendant:

```bash
python3 hardware_main.py execute --trial 1 --ack-safety
```

Stage-by-stage approval is the default. The script checks joint continuity,
tool-pose agreement and gripper readings, and stops on detected errors without
an automatic retreat. Passing offline collision checks does not by itself
establish physical collision clearance or grasp success. Only repeat or extend
execution after supervised commissioning.

## Source layout

- `run_lab2.py`: simulation entry point and trial loop
- `hardware_main.py`: trajectory planning, validation, inspection and execution
- `shelf_heights.json`: physical shelf-top measurements
- `lab2/`: scene generation, kinematics, path planning and simulation task
- `hardware/`: FrankaPy adapter, trajectory capture, validation and execution
- `franka_emika_panda/`: source MuJoCo shelf model and required mesh assets

The robot model files are distributed with their original Apache 2.0 license
in `franka_emika_panda/LICENSE`.
