"""Capture motions by RUNNING THE ORIGINAL sim task, not recreating its sequence.

No Franka commands here; it needs MuJoCo on the off-robot planning computer.
"""
from datetime import datetime, timezone
import numpy as np

from lab2.config import Settings, HOME, PLANNER_SEED_OFFSET, targets_for_trial, trial_seed
from lab2.scene import ORIGINAL_MJCF
from lab2.simulator import LabSim
from lab2.task import ManipulationTask
from lab2.rrt import interpolate_edge
from .spec import STAGES, GRIP_ACTIONS, SCHEMA_VERSION, sha256_file, validate_plan


class CapturePlayer:
    """Implements original task's Player interface; stores every planned RRT move.

    Kinematically moves through checked joints in sim to preserve its block
    attachment state. No GUI and no 1 kHz wall-clock wait required.
    """
    def __init__(self, sim, cfg):
        self.sim, self.cfg = sim, cfg
        self.moves = {}

    def frame(self):
        pass

    def hold(self, seconds):
        if seconds < 0:
            raise ValueError('negative hold')
        self.sim.advance(self.sim.arm_q(), np.zeros(7), max(0.0001, seconds))

    def move(self, motion):
        if motion.name in self.moves:
            raise RuntimeError(f'Duplicate stage name {motion.name}')
        if len(motion.path) < 2 or len(motion.durations) != len(motion.path) - 1:
            raise RuntimeError(f'Invalid RRT path {motion.name}')
        carrying = self.sim.attachment is not None
        if np.max(np.abs(self.sim.arm_q() - motion.path[0])) > 0.001:
            raise RuntimeError(f'Playback start does not match live sim at {motion.name}')
        for q0, q1, dt in zip(motion.path, motion.path[1:], motion.durations):
            # Higher density than the original planner's joint-space edge check.
            for q in interpolate_edge(q0, q1, min(.025, self.cfg.rrt_edge_resolution)):
                reasons = self.sim.collision_reasons(q, carrying=carrying)
                if reasons:
                    raise RuntimeError(f'Collision during capture {motion.name}: {reasons}')
                self.sim.advance(q, np.zeros(7), max(0.0001, float(dt)/20))
        goal_tool_pose = self.sim.evaluate(np.asarray(motion.path[-1], float), carrying=carrying)
        self.moves[motion.name] = {
            'type': 'move', 'name': motion.name,
            'carrying': bool(carrying),
            'path': [np.asarray(q, float).tolist() for q in motion.path],
            'durations': [float(t) for t in motion.durations],
            'goal_tool_pose': goal_tool_pose.tolist(),
        }


def build_plan(seed=13, trials=3):
    cfg = Settings(seed=int(seed), trials=int(trials))
    if cfg.trials < 1:
        raise ValueError('Need at least one trial')
    sim = LabSim(cfg, control='kinematic')
    home_tool_pose = sim.evaluate(HOME).tolist()
    captures = []
    for i in range(cfg.trials):
        if i:
            sim.prepare_next_trial()  # sim-only respawn, same existing sim session
        sim.rng = np.random.default_rng(trial_seed(cfg.seed, i) + PLANNER_SEED_OFFSET)
        player = CapturePlayer(sim, cfg)
        left, right, shelf = targets_for_trial(cfg, i)
        result = ManipulationTask(sim, cfg, player).run_one(i,
                         targets=(left, right, shelf))
        if len(result['stages']) != 18:
            raise RuntimeError('Simulation did not complete 18 stages')
        stages = []
        for rec in result['stages']:
            name = rec['stage']
            if name in GRIP_ACTIONS:
                stages.append({'type': 'grip', 'name': name,
                               'action': GRIP_ACTIONS[name]})
            else:
                move = dict(player.moves[name])
                move['goal_xyz'] = rec['goal_xyz']
                stages.append(move)
        if tuple(s['name'] for s in stages) != STAGES:
            raise RuntimeError('Original task stage list unexpectedly changed')
        if np.max(np.abs(sim.arm_q() - HOME)) > .001:
            raise RuntimeError('Sim ended away from the intended HOME')
        captures.append({
            'trial': i + 1,
            'seed': trial_seed(cfg.seed, i),
            'left_xyz': left.tolist(),
            'right_xyz': right.tolist(),
            'right_shelf': shelf + 1,
            'stages': stages,
        })
    doc = {
        'schema_version': SCHEMA_VERSION,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': 'lab2.task.ManipulationTask.run_one / MuJoCo collision-checked RRT',
        'scene_sha256': sha256_file(ORIGINAL_MJCF),
        'seed': cfg.seed, 'trials': cfg.trials,
        'home': HOME.tolist(),
        'home_tool_pose': home_tool_pose,
        'table_block_xyz': [0.59, 0.0, 0.0221],
        'runs': captures,
    }
    validate_plan(doc)
    return doc
