"""Run offline regressions and optional real MuJoCo task regressions.

Examples:
    python validate_lab3.py --offline
    python validate_lab3.py --all --log outputs/lab3/validation.json
    python validate_lab3.py --adjacent

The simulator runs are kinematic with scripted attachment, as in the lab stack.
An unavailable simulator is a validation failure, never a passing/skipped run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np


def offline_checks():
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return {
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "failures": [{"test": str(t), "traceback": tb} for t, tb in result.failures],
        "errors": [{"test": str(t), "traceback": tb} for t, tb in result.errors],
        "skipped": [{"test": str(t), "reason": reason} for t, reason in result.skipped],
    }


def task_checks(*, transitions=False):
    from lab3.config import Settings
    from lab3.simulator import Lab3Sim
    from lab3.task_executor import TaskExecutor
    from lab3.task_planner import PATTERNS, PATTERN_ORDER

    cfg = Settings()
    sim = Lab3Sim(cfg, control="kinematic")
    executor = TaskExecutor(sim, cfg, grasp_modes=("top",))
    results = executor.run(PATTERN_ORDER, reset_after_goal=not transitions)
    ok = len(results) == len(PATTERN_ORDER)
    for result in results:
        achieved = (result["status"] == "achieved" and
                    tuple(result["final_grid"] or ()) == PATTERNS[result["pattern"]])
        certified_skip = (result["status"] == "skipped_top_grasp_infeasible" and
                          bool(result.get("infeasibility_certificate")))
        ok &= achieved or certified_skip
        ok &= all(m["grasp_mode"] == "top" for m in result["movements"])
        if not transitions:
            ok &= (result.get("reset") or {}).get("status") == "achieved"
    return {"status": "passed" if ok else "failed", "results": results}


def adjacent_checks():
    from lab3.config import Settings, grid_position
    from lab3.geometry import make_transform
    from lab3.pick_place import PickPlaceSkill
    from lab3.simulator import Lab3Sim
    from lab3.task_planner import observe_simulator

    results = []
    for neighbor_cell in ("G01", "G21", "G10", "G12"):
        cfg = Settings()
        sim = Lab3Sim(cfg, control="kinematic")
        # Fixture setup only: execution below uses the actual pick/place skill.
        row, col = int(neighbor_cell[1]), int(neighbor_cell[2])
        sim.set_block_pose("blue_0", make_transform(grid_position(row, col), np.eye(3)))
        sim.mj.mj_forward(sim.model, sim.data)
        skill = PickPlaceSkill(sim, cfg, grasp_modes=("top",))
        for direction, destination in (("place_adjacent", "G11"),
                                       ("pick_adjacent", "R0")):
            before = {name: sim.block_pose(name).copy() for name in sim.block_positions()
                      if name != "red_0"}
            item = {"neighbor_cell": neighbor_cell, "direction": direction,
                    "moving_block": "red_0", "destination": destination}
            try:
                move = skill.move("red_0", destination)
                _, locations = observe_simulator(sim)
                neighbors_unchanged = all(np.allclose(sim.block_pose(name), pose,
                                                     rtol=0., atol=1e-8)
                                          for name, pose in before.items())
                success = (locations["red_0"] == destination and neighbors_unchanged
                           and move.selected_grasp_mode == "top" and sim.attachment is None)
                item.update(status="passed" if success else "failed",
                            neighbors_unchanged=neighbors_unchanged, result=asdict(move))
            except Exception as exc:
                item.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            results.append(item)
            if item["status"] != "passed":
                break  # No dependent pickup after a failed placement.
    ok = len(results) == 8 and all(r["status"] == "passed" for r in results)
    return {"status": "passed" if ok else "failed", "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Run regression tests without MuJoCo")
    parser.add_argument("--all", action="store_true", help="Run offline, patterns/reset, transitions, adjacent pairs")
    parser.add_argument("--patterns", action="store_true", help="Run four patterns with autonomous reset")
    parser.add_argument("--transitions", action="store_true", help="Run direct transitions between four patterns")
    parser.add_argument("--adjacent", action="store_true", help="Test adjacent placement and pickup on four sides")
    parser.add_argument("--log", type=Path, default=Path("outputs/lab3/validation.json"))
    args = parser.parse_args()
    if not any((args.offline, args.all, args.patterns, args.transitions, args.adjacent)):
        args.offline = True

    started = time.monotonic()
    integration_requested = any((args.all, args.patterns, args.transitions, args.adjacent))
    report = {"simulation_only": True, "scripted_attachment": True,
              "validation_scope": "mujoco_requested" if integration_requested else "offline_only",
              "python_version": sys.version.split()[0], "checks": {}}
    checks = []
    if args.offline or args.all:
        checks.append(("offline", offline_checks))
    if args.patterns or args.all:
        checks.append(("patterns_with_reset", lambda: task_checks(transitions=False)))
    if args.transitions or args.all:
        checks.append(("pattern_transitions", lambda: task_checks(transitions=True)))
    if args.adjacent or args.all:
        checks.append(("adjacent_pairs", adjacent_checks))
    for name, check in checks:
        print(f"\nVALIDATION: {name}", flush=True)
        try:
            report["checks"][name] = check()
        except Exception as exc:
            report["checks"][name] = {"status": "failed",
                                      "reason": f"{type(exc).__name__}: {exc}"}
        print(f"{name}: {report['checks'][name]['status']}", flush=True)
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = ("passed" if all(c["status"] == "passed"
                                         for c in report["checks"].values()) else "failed")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nValidation {report['status']}; report: {args.log}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
