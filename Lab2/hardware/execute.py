"""Physical 18-stage task sequencer with fail-closed state checks.

No automatic retreat on exception. Never defeats Franka virtual walls.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from .spec import HOME
from .robot import FrankaRobot, HardwareError, pose_mismatch
from .spec import validate_plan


class SafetyLimits:
    start_joint_rad = .060
    goal_joint_rad = .055
    pose_translation_m = .012
    pose_angle_rad = .12
    duration_scale = 1.8
    min_duration_s = 2.0


def require_close(actual, expected, tol, label):
    err = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    if err > tol:
        raise HardwareError(f'{label}: max joint error {err:.4f} rad exceeds {tol:.4f}; STOP')
    return err


def check_tool(robot, expected_H, limits, label):
    pos, rad = pose_mismatch(np.asarray(expected_H, float), robot.tool_pose())
    if pos > limits.pose_translation_m or rad > limits.pose_angle_rad:
        raise HardwareError(f'{label}: robot/sim frame mismatch: '
                            f'{pos:.3f} m / {rad:.3f} rad. STOP; never override this')
    return pos, rad


def prompt_operator(msg):
    response = input(msg + ' Type YES to continue (anything else aborts): ').strip()
    if response != 'YES':
        raise HardwareError('Operator cancelled; robot remains at its current location')


def log_event(path, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'utc': datetime.now(timezone.utc).isoformat(), **fields}
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(payload) + '\n')
        f.flush()


def execute(doc, robot, *, selected_trials=None, auto_stages=False,
            log_path='outputs/hardware_run_log.jsonl',
            confirm=prompt_operator, limits=None):
    """Execute whole plan or selected sequential trials on a real robot.

    All runs start at HOME and with a block manually placed on the table.
    The robot must remain at HOME at trial boundaries; no scripted respawn.
    """
    validate_plan(doc)
    limits = limits or SafetyLimits()
    if selected_trials is None:
        selected_trials = list(range(1, doc['trials']+1))
    if not selected_trials or len(set(selected_trials)) != len(selected_trials):
        raise ValueError('Select at least one distinct trial')
    if any(not isinstance(i, int) or i < 1 or i > doc['trials'] for i in selected_trials):
        raise ValueError('Unknown trial number')

    log_path = Path(log_path)
    print('Franka Lab2 execution: explicit e-stop attendant REQUIRED.')
    print('Verify cube at table xyz [0.590, 0.000, 0.0221] m and shelves unchanged.')
    require_close(robot.joints(), HOME, limits.start_joint_rad, 'Initial HOME')
    check_tool(robot, doc['home_tool_pose'], limits, 'HOME tool-frame calibration')
    print('Initial robot joint pose and robot/simulation tool frame: PASS')

    for n in selected_trials:
        run = doc['runs'][n-1]
        print(f'\nTRIAL {n}: left {np.round(run["left_xyz"],4)} m | '
              f'right {np.round(run["right_xyz"],4)} m '
              f'(right shelf level {run["right_shelf"]})')
        confirm('Operator: place cube on TABLE at indicated position, '
                'check chosen shelf targets, empty workspace, e-stop held.')
        require_close(robot.joints(), HOME, limits.start_joint_rad, f'Trial {n} HOME')
        check_tool(robot, doc['home_tool_pose'], limits, f'Trial {n} home tool pose')
        # The offline simulation begins with open fingers (0.04 m PER finger).
        # Match this on hardware (0.08 m TOTAL opening) before stage 1.
        initial_width = robot.open_gripper()
        print(f'  initial open gripper width: {initial_width:.4f} m')
        log_event(log_path, trial=n, status='trial_started',
                  left=run['left_xyz'], right=run['right_xyz'])
        try:
            for st in run['stages']:
                name = st['name']
                print(f'  {name}')
                if not auto_stages:
                    confirm(f'Review and approve stage: {name}.')
                if st['type'] == 'grip':
                    if st['action'] == 'close':
                        width = robot.grasp_block()
                        print(f'    grasp width: {width:.4f} m')
                        # Even with auto-stages, friction and seating are NOT proven by width.
                        confirm('Visually confirm gripper is properly engaged on the cube '
                                'BEFORE the next lift.')
                    else:
                        width = robot.open_gripper()
                        print(f'    open width: {width:.4f} m')
                    log_event(log_path, trial=n, stage=name, status='completed',
                              gripper_width_m=width)
                    continue
                path = [np.asarray(q, float) for q in st['path']]
                require_close(robot.joints(), path[0], limits.start_joint_rad,
                              f'Stage start {name}')
                for k, (goal, duration) in enumerate(zip(path[1:], st['durations'])):
                    actual_duration = max(limits.min_duration_s,
                                          limits.duration_scale*float(duration))
                    print(f'    waypoint {k+1}/{len(path)-1}: '
                          f'duration={actual_duration:.2f}s')
                    robot.move_joints(goal, actual_duration)
                    require_close(robot.joints(), goal, limits.goal_joint_rad,
                                  f'{name} waypoint {k+1}')
                pos_err, rot_err = check_tool(robot, st['goal_tool_pose'],
                                              limits, name)
                log_event(log_path, trial=n, stage=name, status='completed',
                          tool_position_error_m=pos_err,
                          tool_orientation_error_rad=rot_err)
            require_close(robot.joints(), HOME, limits.start_joint_rad,
                          f'Trial {n} final HOME')
            log_event(log_path, trial=n, status='trial_completed')
            print(f'  Trial {n} finished at simulation HOME, gripper released.')
        except (BaseException) as exc:
            robot.stop()
            log_event(log_path, trial=n, status='ABORTED', error=str(exc))
            print('ABORT: No automatic retreat, arm reset, or gripper release attempted.')
            print('Use the physical e-stop if needed; recover using Lab1 procedures and TA.')
            raise
    print('\nAll selected trials completed. Physical placements still require visual inspection.')
