"""Validate hardware trajectories against the calibrated scene."""
import numpy as np
from lab2.config import Settings, HOME
from lab2.scene import ORIGINAL_MJCF, GENERATED_MJCF, prepare_scene, scene_shelf_geometry
from lab2.simulator import LabSim
from lab2.rrt import interpolate_edge
from .heights import assert_matching_heights
from .spec import sha256_file, validate_plan, COLLISION_RESOLUTION_RAD

def validate_collisions(doc):
    validate_plan(doc)
    assert_matching_heights(doc)
    if doc['source_scene_sha256'] != sha256_file(ORIGINAL_MJCF):
        raise ValueError('Stock source scene changed since planning; rebuild plan')
    prepare_scene()
    if doc['runtime_scene_sha256'] != sha256_file(GENERATED_MJCF):
        raise ValueError('Calibrated runtime scene changed since planning; rebuild plan')
    geometry = scene_shelf_geometry(GENERATED_MJCF)
    mapping = {'left_bottom_mm':'LShelfBottom','right_bottom_mm':'RShelfBottom',
               'right_middle_mm':'RShelfMiddle','right_top_mm':'RShelfTop'}
    for key, name in mapping.items():
        pos, size = geometry[name]
        if abs(pos[2]+size[2]-doc['shelf_heights_mm'][key]/1000.) > 1e-8:
            raise RuntimeError('Simulated shelf geometry does not match '+key)
    cfg = Settings(seed=doc['seed'], trials=doc['trials'])
    sim = LabSim(cfg, control='kinematic')
    checked = 0
    for idx, run in enumerate(doc['runs']):
        if idx:
            sim.prepare_next_trial()
        for stage in run['stages']:
            name = stage['name']
            if stage['type'] == 'grip':
                if stage['action'] == 'close':
                    sim.attach()
                else:
                    target = run['left_xyz'] if name.startswith('07_') else run['right_xyz']
                    sim.release(np.asarray(target,float))
                continue
            if np.max(np.abs(sim.arm_q()-stage['path'][0])) > .001:
                raise RuntimeError('Saved stage path discontinuity at '+name)
            for a,b in zip(stage['path'],stage['path'][1:]):
                for q in interpolate_edge(a,b,COLLISION_RESOLUTION_RAD):
                    reasons = sim.collision_reasons(q,carrying=stage['carrying'])
                    if reasons:
                        raise RuntimeError('Collision in trial {} / {}: {}'.format(idx+1,name,reasons))
                    sim.advance(q,np.zeros(7),.001)
                    checked += 1
        if np.max(np.abs(sim.arm_q()-HOME)) > .001:
            raise RuntimeError('Saved trial does not finish at HOME')
    return checked
