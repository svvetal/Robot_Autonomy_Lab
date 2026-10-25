"""Interpolated playback, optional passive MuJoCo viewer and MP4 capture."""
from pathlib import Path
import numpy as np
from .trajectory import quintic


class Player:
    def __init__(self, sim, cfg, *, visualize=False, record_path=None):
        self.sim, self.cfg = sim, cfg
        self.viewer = None
        self.renderer = None
        self.writer = None
        self.record_path = record_path
        self.visualize = visualize
        self.video_frames = 0
        self.next_frame_time = 0.0
        self.next_view_time = 0.0

    def __enter__(self):
        mj = self.sim.mj
        if self.visualize:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.sim.model, self.sim.data)
            self.viewer.cam.distance = 2.6
            self.viewer.cam.azimuth = 140
            self.viewer.cam.elevation = -23
        if self.record_path:
            from imageio.v2 import get_writer
            self.renderer = mj.Renderer(self.sim.model,
                                        height=self.cfg.render_height,
                                        width=self.cfg.render_width)
            out = Path(self.record_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            self.writer = get_writer(str(out), fps=self.cfg.video_fps, codec='libx264',
                                     quality=7, macro_block_size=16)
        return self

    def __exit__(self, et, ev, tb):
        try:
            if self.writer is not None:
                self.writer.close()
            if self.renderer is not None:
                self.renderer.close()
        finally:
            if self.viewer is not None:
                self.viewer.close()

    def frame(self):
        sim = self.sim
        if self.viewer is not None and sim.data.time >= self.next_view_time:
            if not self.viewer.is_running():
                raise RuntimeError('MuJoCo viewer closed during task')
            self.viewer.sync()
            self.next_view_time = sim.data.time + 1/30
        if self.renderer is not None:
            if sim.data.time + 1e-9 >= self.next_frame_time:
                self.renderer.update_scene(sim.data)
                self.writer.append_data(self.renderer.render())
                self.next_frame_time = sim.data.time + 1 / self.cfg.video_fps
                self.video_frames += 1

    def hold(self, seconds):
        sim, cfg = self.sim, self.cfg
        dt = sim.model.opt.timestep
        q = sim.arm_q()
        for _ in range(max(1, round(seconds/dt))):
            sim.advance(q, np.zeros(7), dt)
            self.frame()

    def move(self, motion):
        sim = self.sim
        dt = float(sim.model.opt.timestep)
        for q_start, q_goal, duration in zip(motion.path, motion.path[1:], motion.durations):
            n = max(2, int(np.ceil(duration / dt)))
            step_dt = duration/n
            for k in range(1, n+1):
                des, vel, _ = quintic(q_start, q_goal, k/n*duration, duration)
                # For torque mode actual integrator timestep is fixed model.opt.timestep.
                sim.advance(des, vel, dt)
                self.frame()
        self.hold(0.17)  # settle at each state transition
