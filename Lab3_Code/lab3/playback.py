"""MuJoCo playback, viewer and video recording."""
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

    @staticmethod
    def _aim(cam):
        """Look at the 3x3 grid from in front of the table (robot in the background)."""
        cam.lookat[:] = (0.45, 0.0, 0.08)
        cam.distance = 1.35
        cam.azimuth = 180
        cam.elevation = -48

    def __enter__(self):
        mj = self.sim.mj
        if self.visualize:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.sim.model, self.sim.data)
            self._aim(self.viewer.cam)
        if self.record_path:
            from imageio.v2 import get_writer
            self.cam = mj.MjvCamera()
            self._aim(self.cam)
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
                self.renderer.update_scene(sim.data, camera=self.cam)
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
        if getattr(motion, 'smooth', False):
            return self._move_smooth(motion)
        sim = self.sim
        dt = float(sim.model.opt.timestep)
        for q_start, q_goal, duration in zip(motion.path, motion.path[1:], motion.durations):
            n = max(2, int(np.ceil(duration / dt)))
            step_dt = duration/n
            for k in range(1, n+1):
                des, vel, _ = quintic(q_start, q_goal, k/n*duration, duration)
                if k == n:
                    des = np.asarray(q_goal, float).copy()   # exact segment end

                sim.advance(des, vel, dt)
                self.frame()
        self.hold(0.17)

    def _move_smooth(self, motion):
        """Follow a joint polyline with a single quintic time scaling."""
        sim = self.sim
        dt = float(sim.model.opt.timestep)
        pts = np.asarray(motion.path, float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.r_[0.0, np.cumsum(seg)]
        total, T = cum[-1], float(motion.durations[0])

        def at(arc):
            k = min(int(np.searchsorted(cum, arc, side='right')) - 1, len(seg) - 1)
            u = 0.0 if seg[k] < 1e-12 else (arc - cum[k]) / seg[k]
            return pts[k] + np.clip(u, 0, 1) * (pts[k + 1] - pts[k]), k
        n = max(2, int(np.ceil(T / dt)))
        for i in range(1, n + 1):
            t = i / n * T
            u = t / T
            s = 10*u**3 - 15*u**4 + 6*u**5
            sd = (30*u**2 - 60*u**3 + 30*u**4) / T
            q, k = at(s * total)
            if i == n:
                q = pts[-1].copy()        # land exactly on the goal (no arc-length round-off)
            direction = (pts[k + 1] - pts[k]) / max(seg[k], 1e-12)
            sim.advance(q, direction * sd * total, dt)
            self.frame()
        self.hold(0.17)
