"""Revalidate saved stage paths against the original MuJoCo collision scene."""
import numpy as np
from lab2.config import Settings, HOME
from lab2.geometry import make_transform
from lab2.scene import ORIGINAL_MJCF
from lab2.simulator import LabSim
from lab2.rrt import interpolate_edge
from .spec import sha256_file, validate_plan


def validate_collisions(doc):
    validate_plan(doc)
    if doc.get('scene_sha256') != sha256_file(ORIGINAL_MJCF):
        raise ValueError('Original scene XML changed since plan generation')
    cfg = Settings(seed=doc['seed'], trials=doc['trials'])
    sim = LabSim(cfg, control='kinematic')
    checked = 0
    for trial_index, run in enumerate(doc['runs']):
        if trial_index:
            sim.prepare_next_trial()
        for stage in run['stages']:
            name = stage['name']
            if stage['type'] == 'grip':
                if stage['action'] == 'close':
                    sim.attach()
                else:
                    target = run['left_xyz'] if name.startswith('07_') else run['right_xyz']
                    sim.release(np.asarray(target, dtype=float))
                continue
            if np.max(np.abs(sim.arm_q()-stage['path'][0])) > .001:
                raise RuntimeError(f'Saved path does not begin at current model state: {name}')
            for a, b in zip(stage['path'], stage['path'][1:]):
                for q in interpolate_edge(a, b, .02):
                    reason = sim.collision_reasons(q, carrying=stage['carrying'])
                    if reason:
                        raise RuntimeError(f'Trial {trial_index+1}, {name}: {reason}')
                    sim.advance(q, np.zeros(7), .001)
                    checked += 1
        if np.max(np.abs(sim.arm_q()-HOME)) > .001:
            raise RuntimeError(f'Trial {trial_index+1}: simulated end is not HOME')
    return checked
