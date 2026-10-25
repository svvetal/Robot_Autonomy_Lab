# Validation record supplied with the package

**Scope: Async Lab 2 (not Homework 2).**

## Verified locally in package-build environment

- Unmodified source shelf scene contains **31** shelf/table parts. The generated runtime scene retains every pose and half-extent, verified by an automated test.
- `python -m pytest -q tests/test_core.py`: **10 passed**.
- `python preflight.py --trials 3`: completed; reported all selected pose targets kinematically reachable using an independent XML forward-kinematics model and multistart inverse search.
- `python -m compileall -q lab2 preflight.py run_lab2.py`: passed.
- Sampled left- and right-shelf positions use fixed independent per-trial seeds; right shelf chosen bottom, middle, top.

## Not verified in build environment

- The third-party `mujoco` Python module is not present and package installation failed due to blocked network. The actual simulator could **not** be started.
- MuJoCo model compilation, runtime contact rejection, actual 18-stage sequence, smooth viewer playback, three MP4 videos, and torque-based tracking are **unverified**.
- Scripted carry in kinematic mode is a visual simulation convenience, **not a physically verified grip**.

## Exact next acceptance step

```bash
python -m pip install -r requirements.txt
python -m pytest -q
LAB2_FULL_SMOKE=1 python -m pytest -q tests/test_mujoco_integration.py
python run_lab2.py --trials 3 --seed 13 --record
```

Review every video. A zero-error exit and an MP4 file alone are **not** sufficient proof that contact-rich manipulation is physically correct.


## Follow-up: fixed side-grasp pickup transition (2026-10-22)

A user runtime trace showed an unanticipated failure on the regrasp-retreat
planning start. Version 2 adds close-gripper-aware collision FK, a tightly
constrained support-contact exemption and lift-before-retreat geometry.
Regression tests for these conditions have been executed offline, but this
build environment still has no MuJoCo and therefore cannot verify full physics
or integrated viewer behavior. Test and review the 18-stage end-to-end run
locally before submission.

## v3: single persistent MuJoCo session + horizontal randomized shelf placements

- Replaced per-trial `LabSim`, `Player`, and `sim.reset()` with ONE persistent MuJoCo model/data, ONE Player/viewer, and ONE continuous video `outputs/all_trials_continuous.mp4`.
- Between trial boundaries, explicitly respawn ONLY the single block on the table, preserving arm state and the simulation clock. This is a simulation setup step, not a physically executed robot reverse transfer.
- Both the single left shelf and three destination shelves use a wider **horizontal world-X range** `[0.430,0.560] m`, stratified by trial. World Y remains near the front edge for side regrasp. The right level cycles bottom/middle/top.
- Reproducible independent placement streams: `seed + 7919*trial`; distinct planner/IK RNG stream: `placement_seed + 104729`.
- `run_lab2.py --check-scene` previews **exact** CLI positions without MuJoCo; `preflight.py` now uses the same sampler.
- Pure-Python regression tests passed in build environment. Independent FK preflight for all three seed-13 samples passed. MuJoCo execution remains **unverified in this build environment**; inspect the actual simulator and outputs on the user's machine.
