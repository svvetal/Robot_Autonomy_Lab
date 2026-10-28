"""Supervised execution of validated Franka trajectories."""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .spec import HOME,validate_plan
from .robot import HardwareError,pose_mismatch

class SafetyLimits:
    start_joint_rad = .060
    goal_joint_rad = .055
    pose_translation_m = .012
    pose_angle_rad = .12
    duration_scale = 1.8
    min_duration_s = 2.0

def require_close(actual,expected,tol,label):
    err=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    if not np.isfinite(err) or err > tol:
        raise HardwareError('{}: joint error {:.4f} rad exceeds {:.4f}; STOP'.format(label,err,tol))
    return err

def check_tool(robot,expected_H,limits,label):
    pos,rad=pose_mismatch(np.asarray(expected_H,float),robot.tool_pose())
    if not np.isfinite(pos+rad) or pos > limits.pose_translation_m or rad > limits.pose_angle_rad:
        raise HardwareError('{}: sim/robot tool frame mismatch {:.3f} m / {:.3f} rad; STOP'.format(label,pos,rad))
    return pos,rad

def prompt_operator(msg):
    response=input(msg+' Type YES to continue; anything else aborts: ').strip()
    if response != 'YES':
        raise HardwareError('Operator cancelled; arm remains at its current pose')

def log_event(path,**fields):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    record={'utc':datetime.now(timezone.utc).isoformat(),**fields}
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()

def execute(doc,robot,*,selected_trials=None,auto_stages=False,
            log_path='outputs/hardware_run_log.jsonl',confirm=prompt_operator,limits=None):
    validate_plan(doc)
    limits=limits or SafetyLimits()
    if selected_trials is None: selected_trials=list(range(1,doc['trials']+1))
    if (not selected_trials or len(set(selected_trials))!=len(selected_trials) or
        any(type(i) is not int or i<1 or i>doc['trials'] for i in selected_trials)):
        raise ValueError('Invalid trial selection')
    print('Supervised Franka task; physical E-STOP attendant REQUIRED.')
    print('Measured shelf TOPs (mm):',doc['shelf_heights_mm'])
    print('Confirm real shelves and tool frame coincide with model; measured Z alone is insufficient.')
    require_close(robot.joints(),HOME,limits.start_joint_rad,'Initial HOME')
    check_tool(robot,doc['home_tool_pose'],limits,'Initial HOME tool frame')
    for n in selected_trials:
        run=doc['runs'][n-1]
        print('TRIAL {}: left {} m, right {} m at level {}'.format(
            n,run['left_xyz'],run['right_xyz'],run['right_shelf']))
        confirm('Verify real cube at x=0.590,y=0,z=0.0221 m, shelf measured, empty workspace, E-STOP operator ready.')
        require_close(robot.joints(),HOME,limits.start_joint_rad,'Trial HOME')
        check_tool(robot,doc['home_tool_pose'],limits,'Trial HOME tool frame')
        try:
            width=robot.open_gripper()
            log_event(log_path,trial=n,status='trial_started',heights_mm=doc['shelf_heights_mm'],
                      height_fingerprint=doc['shelf_heights_sha256'],
                      left=run['left_xyz'],right=run['right_xyz'],opening_m=width)
            for stage in run['stages']:
                name=stage['name']
                if not auto_stages:
                    confirm('Review stage {}: clearance, fixtures and human safety.'.format(name))
                if stage['type']=='grip':
                    if stage['action']=='close':
                        width=robot.grasp_block()
                        confirm('Visually verify actual cube grip before any subsequent lift.')
                    else:
                        width=robot.open_gripper()
                    log_event(log_path,trial=n,stage=name,status='completed',width_m=width)
                    continue
                path=[np.asarray(q,float) for q in stage['path']]
                require_close(robot.joints(),path[0],limits.start_joint_rad,name+' start')
                for k,(goal,duration) in enumerate(zip(path[1:],stage['durations'])):
                    actual_dur=max(limits.min_duration_s,limits.duration_scale*float(duration))
                    print('  {}: waypoint {}/{} in {:.2f}s'.format(name,k+1,len(path)-1,actual_dur))
                    robot.move_joints(goal,actual_dur)
                    require_close(robot.joints(),goal,limits.goal_joint_rad,name+' waypoint')
                pos_err,rot_err=check_tool(robot,stage['goal_tool_pose'],limits,name)
                log_event(log_path,trial=n,stage=name,status='completed',
                          tool_position_error_m=pos_err,tool_angle_error_rad=rot_err)
            require_close(robot.joints(),HOME,limits.start_joint_rad,'Trial final HOME')
            log_event(log_path,trial=n,status='trial_completed')
        except BaseException as exc:
            robot.stop()
            log_event(log_path,trial=n,status='ABORTED',error=str(exc))
            print('ABORT. No automatic retreat, arm reset, or gripper release. Supervised recovery needed.')
            raise
    print('All selected trials completed; visually inspect physical placements.')
