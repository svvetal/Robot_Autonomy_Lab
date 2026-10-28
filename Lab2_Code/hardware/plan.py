"""Generate hardware trajectories from the MuJoCo task."""
from datetime import datetime, timezone
import numpy as np
from lab2.config import Settings, HOME, PLANNER_SEED_OFFSET, targets_for_trial, trial_seed
from lab2.scene import ORIGINAL_MJCF, GENERATED_MJCF, prepare_scene
from lab2.shelf_heights import load_heights, heights_as_mm
from lab2.simulator import LabSim
from lab2.task import ManipulationTask
from lab2.rrt import interpolate_edge
from .heights import load_measured_heights, heights_fingerprint
from .spec import (STAGES, GRIP_ACTIONS, SCHEMA_VERSION, COLLISION_RESOLUTION_RAD,
                   sha256_file, validate_plan)

class CapturePlayer:
    """Record planned joint paths and tool poses."""
    def __init__(self, sim, cfg):
        self.sim, self.cfg, self.moves = sim, cfg, {}

    def frame(self):
        pass

    def hold(self, seconds):
        if seconds < 0:
            raise ValueError('Negative hold')
        self.sim.advance(self.sim.arm_q(), np.zeros(7), max(.0001, seconds))

    def move(self, motion):
        if motion.name in self.moves:
            raise RuntimeError('Duplicate stage '+motion.name)
        if len(motion.path) < 2 or len(motion.durations) != len(motion.path)-1:
            raise RuntimeError('Invalid trajectory at '+motion.name)
        carrying = self.sim.attachment is not None
        if np.max(np.abs(self.sim.arm_q()-motion.path[0])) > .001:
            raise RuntimeError('Trajectory discontinuity at '+motion.name)

        for edge_idx, (a, b) in enumerate(zip(motion.path, motion.path[1:]), 1):
            for q in interpolate_edge(a, b, COLLISION_RESOLUTION_RAD):
                issue = self.sim.collision_reasons(q, carrying=carrying)
                if issue:
                    raise RuntimeError(
                        'Collision in stage {} edge {}/{} (carrying={}): {}. '
                        'The requested measured shelf geometry/target may not have a '
                        'collision-free solution; do NOT relax the collision test.'
                        .format(motion.name, edge_idx, len(motion.path)-1, carrying, issue))
        for a, b, dt in zip(motion.path, motion.path[1:], motion.durations):
            for q in interpolate_edge(a, b, COLLISION_RESOLUTION_RAD):
                self.sim.advance(q, np.zeros(7), max(.0001, float(dt)/20))
        pose = self.sim.evaluate(np.asarray(motion.path[-1], float), carrying=carrying)
        self.moves[motion.name] = dict(
            type='move', name=motion.name, carrying=bool(carrying),
            path=[np.asarray(q,float).tolist() for q in motion.path],
            durations=[float(t) for t in motion.durations], goal_tool_pose=pose.tolist())

def build_plan(seed=13, trials=3):
    cfg = Settings(seed=int(seed), trials=int(trials))
    if cfg.trials < 1:
        raise ValueError('Require at least one trial')
    if cfg.rrt_edge_resolution > COLLISION_RESOLUTION_RAD or cfg.rrt_edge_resolution <= 0:
        raise RuntimeError('RRT must collision-check edges at least as finely as the hardware plan')
    hw_heights = load_measured_heights()
    sim_heights = heights_as_mm(load_heights())
    if any(abs(hw_heights[k]-sim_heights[k]) > 1e-7 for k in hw_heights):
        raise RuntimeError('Sim and hardware measurement readers disagree')

    prepare_scene()
    source_scene_sha = sha256_file(ORIGINAL_MJCF)
    runtime_scene_sha = sha256_file(GENERATED_MJCF)
    sim = LabSim(cfg, control='kinematic')
    home_tool_pose = sim.evaluate(HOME).tolist()
    captures=[]
    for i in range(cfg.trials):
        if i:
            sim.prepare_next_trial()
        sim.rng = np.random.default_rng(trial_seed(cfg.seed, i) + PLANNER_SEED_OFFSET)
        player = CapturePlayer(sim, cfg)
        left, right, shelf = targets_for_trial(cfg, i)
        result = ManipulationTask(sim, cfg, player).run_one(i, targets=(left,right,shelf))
        if len(result['stages']) != 18:
            raise RuntimeError('Expected exactly 18 stages')
        stages=[]
        for rec in result['stages']:
            name = rec['stage']
            if name in GRIP_ACTIONS:
                stages.append(dict(type='grip', name=name, action=GRIP_ACTIONS[name]))
            else:
                move = dict(player.moves[name]); move['goal_xyz'] = rec['goal_xyz']; stages.append(move)
        if tuple(s['name'] for s in stages) != STAGES:
            raise RuntimeError('Simulation stage order changed')
        if np.max(np.abs(sim.arm_q()-HOME)) > .001:
            raise RuntimeError('Trial finished away from HOME')
        captures.append(dict(trial=i+1, seed=trial_seed(cfg.seed,i),
                             left_xyz=left.tolist(), right_xyz=right.tolist(),
                             right_shelf=shelf+1, stages=stages))

    if heights_fingerprint(hw_heights) != heights_fingerprint(load_measured_heights()):
        raise RuntimeError('Measurement file changed during planning. Re-run.')
    doc = {
        'schema_version': SCHEMA_VERSION,
        'collision_resolution_rad': COLLISION_RESOLUTION_RAD,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'MuJoCo calibrated shelf geometry + lab2.task 18-stage RRT',
        'shelf_heights_mm': hw_heights, 'shelf_heights_sha256': heights_fingerprint(hw_heights),
        'source_scene_sha256': source_scene_sha, 'runtime_scene_sha256': runtime_scene_sha,
        'seed':cfg.seed, 'trials':cfg.trials,
        'home':HOME.tolist(), 'home_tool_pose':home_tool_pose,
        'table_block_xyz':[.59,0.,.0221], 'runs':captures,
    }
    validate_plan(doc)
    return doc
