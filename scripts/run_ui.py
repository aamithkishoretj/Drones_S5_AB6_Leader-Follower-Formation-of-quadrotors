#!/usr/bin/env python3
"""Desktop live-control UI for the leader-follower drone simulation."""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from collections import deque

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from dq_control import get_gains, LeaderTrajectory, LemniscateParams, PotatoChipTrajectory, PotatoChipParams
from simulators import SimConfig, LeaderFollowerSimulation
from simulators.base import SimulationStopped


class SimulationUI:
    def __init__(self, root):
        self.root = root
        root.title("Leader–Follower Drone Simulation")
        root.geometry("440x790")
        root.minsize(420, 740)
        self.events = queue.Queue(maxsize=1)
        self.plot_window = None
        self.plot_points = [deque(maxlen=600), deque(maxlen=600)]
        self.last_plot = 0.0
        self.choices = []
        self.closing = False
        self.live = {"lock": threading.Lock(), "values": {}}
        self.stop_event = None
        self.worker = None
        self.running = False
        self.worker_done = False
        self._build()
        root.protocol("WM_DELETE_WINDOW", self._close)
        root.after(40, self._poll)

    def _build(self):
        self.root.configure(bg="#101820")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#101820")
        style.configure("TLabel", background="#101820", foreground="#e6edf3", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 17, "bold"), foreground="#ffffff")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=7)
        style.configure("TCombobox", padding=5)
        outer = ttk.Frame(self.root, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Leader–Follower Flight Lab", style="Header.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Controls stay here; simulators open in their own windows.").pack(anchor="w", pady=(3, 12))
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        panel = ttk.Frame(body)
        panel.pack(fill="both", expand=True)
        self.simulator = tk.StringVar(value="pybullet")
        self.trajectory = tk.StringVar(value="lemniscate")
        self.controller = tk.StringVar(value="learned")
        self._choice(panel, "Simulator", self.simulator, ["pybullet", "gazebo", "mujoco", "kinematic", "ardupilot"])
        self._choice(panel, "Leader route", self.trajectory, ["lemniscate", "potato_chip"])
        self._choice(panel, "Controller", self.controller, ["analytic", "learned"])
        ttk.Label(panel, text="Live parameters", style="Header.TLabel").pack(anchor="w", pady=(16, 3))
        self.speed = tk.DoubleVar(value=float(np.pi / 15))
        self.distance = tk.DoubleVar(value=0.8)
        self.radius = tk.DoubleVar(value=0.85)
        self.altitude = tk.DoubleVar(value=1.0)
        self._scale(panel, "Leader speed (rad/s)", self.speed, 0.05, 0.3, "leader_speed")
        self._scale(panel, "Follower gap (m)", self.distance, 0.65, 2.5, "follow_distance")
        self._scale(panel, "Route radius (m)", self.radius, 0.6, 3.0, "route_radius")
        self._scale(panel, "Flight altitude (m)", self.altitude, 0.8, 3.0, "altitude")
        self.status = tk.StringVar(value="Ready")
        ttk.Label(panel, textvariable=self.status, wraplength=380).pack(anchor="w", pady=(10, 12))
        buttons = ttk.Frame(panel)
        buttons.pack(fill="x", side="bottom")
        self.start_button = ttk.Button(buttons, text="▶  Start simulation", command=self.start)
        self.start_button.pack(fill="x", pady=3)
        self.stop_button = ttk.Button(buttons, text="■  Stop simulation", command=self.stop, state="disabled")
        self.stop_button.pack(fill="x", pady=3)

        self.help_text = tk.StringVar()
        ttk.Label(panel, textvariable=self.help_text, wraplength=380).pack(anchor="w", pady=8)
        self.applied_text = tk.StringVar(value="Move sliders to set targets. Applied values appear during flight.")
        ttk.Label(panel, textvariable=self.applied_text, wraplength=380).pack(anchor="w", pady=8)
        self.simulator.trace_add("write", lambda *_: self._selection_changed())
        self._selection_changed()

    def _selection_changed(self):
        name = self.simulator.get()
        if name == "kinematic":
            self.controller.set("analytic")
            self.controller_box.configure(state="disabled")
            self.help_text.set("Kinematic reference: ideal motion without motors or physics. Opens a separate 3D path plot; no trained model is needed.")
        elif name == "ardupilot":
            self.controller.set("learned")
            self.controller_box.configure(state="readonly")
            self.speed.set(.105); self.radius.set(2.55)
            self.distance.set(1.5); self.altitude.set(2.)
            self.help_text.set("ArduPilot SITL + Gazebo (through WSL on Windows). Requires built ArduCopter, ardupilot_gazebo and pymavlink; set ARDUPILOT_PATH if ArduCopter is not in the sibling ardupilot folder. Learned uses the SITL-identified data-driven model.")
        else:
            self.speed.set(float(np.pi / 15)); self.radius.set(.85)
            self.distance.set(.8); self.altitude.set(1.)
            self.controller.set("learned")
            self.controller_box.configure(state="readonly")
            self.help_text.set("Sliders change the running flight, with gradual transitions. Red = leader; blue = follower. MuJoCo uses simplified force control with yaw tracking." if name == "mujoco" else "Sliders change the running flight, with gradual transitions. Red = leader; blue = follower.")

    def _choice(self, parent, label, variable, options):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(7, 2))
        box = ttk.Combobox(parent, textvariable=variable, values=options, state="readonly")
        box.pack(fill="x")
        self.choices.append(box)
        if label == "Controller": self.controller_box = box

    def _scale(self, parent, label, variable, low, high, key):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(9, 0))
        ttk.Label(row, text=label).pack(anchor="w")
        value = ttk.Label(row, width=8, anchor="e")
        value.pack(side="right")
        def changed(raw):
            v = float(raw)
            value.configure(text=f"{v:.2f}")
            with self.live["lock"]:
                self.live["values"][key] = v
        scale = tk.Scale(row, variable=variable, from_=low, to=high, resolution=(0.01 if high > 1 else 0.005),
                         orient="horizontal", showvalue=False, length=230,
                         bg="#101820", fg="#e6edf3", troughcolor="#344454", highlightthickness=0)
        scale.pack(fill="x")
        variable.trace_add("write", lambda *_: changed(variable.get()))
        changed(variable.get())

    def start(self):
        if self.running:
            return
        name = self.simulator.get()
        trajectory_name = self.trajectory.get()
        try:
            if trajectory_name == "lemniscate":
                traj = LeaderTrajectory(LemniscateParams(w_d=self.speed.get(), r_x=self.radius.get(),
                    r_y=self.radius.get()*.65/.85, z0=self.altitude.get()))
            else:
                traj = PotatoChipTrajectory(PotatoChipParams(w=self.speed.get(), r=self.radius.get(), z0=self.altitude.get()))
            cfg = SimConfig(simulator=name, controller=self.controller.get(),
                            learned_model=os.path.join(_ROOT, "models", f"{name}_response.npz"),
                            gui=True, duration_sec=1e9, trails=True, follow_distance=self.distance.get(),
                            gazebo_render_engine="ogre",
                            follower_offset_mode="path",
                            # ArduPilot demos and its learned model were validated at 24 Hz.
                            **({"ctrl_freq": 24} if name == "ardupilot" else {}))
            sim = LeaderFollowerSimulation(get_gains("real_eig"), cfg, traj)
        except Exception as exc:
            messagebox.showerror("Could not start simulation", str(exc))
            return
        self.stop_event = threading.Event()
        self.running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.run_name = name
        for choice in self.choices: choice.configure(state="disabled")
        self.status.set(f"Starting {name}…")
        self.plot_points = [deque(maxlen=600), deque(maxlen=600)]
        if name == "kinematic": self._open_plot()
        with self.live["lock"]:
            self.live["values"] = dict(leader_speed=self.speed.get(), follow_distance=self.distance.get(),
                                      route_radius=self.radius.get(), altitude=self.altitude.get())
        next_wall = time.monotonic()


        def on_step(t, states):
            nonlocal next_wall
            if name == "kinematic":
                next_wall += cfg.control_timestep()
                self.stop_event.wait(max(0., next_wall-time.monotonic()))
            self._post_event((t, [s.position.copy() for s in states], dict(sim.live_applied)))
        def run():
            try:
                sim.run(stop_event=self.stop_event, step_callback=on_step, live_parameters=self.live)
                self._post_event((None, None, "Simulation stopped."))
            except (SimulationStopped, KeyboardInterrupt) as exc:
                self._post_event((None, None, str(exc) or "Simulation stopped."))
            except Exception as exc:
                self._post_event((None, None, f"Simulation error: {exc}"))
            finally:
                self.worker_done = True
        self.worker = threading.Thread(target=run, name="drone-simulation", daemon=True)
        self.worker_done = False
        self.worker.start()

    def stop(self):
        if self.stop_event:
            self.stop_event.set()
            self.status.set("Stopping simulation…")
            self.stop_button.configure(state="disabled")

    def _finished(self):
        self.running = False
        for choice in self.choices: choice.configure(state="readonly")
        if self.simulator.get() == "kinematic":
            self.controller_box.configure(state="disabled")
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if self.status.get() == "Stopping simulation…":
            self.status.set("Stopped. Ready to launch another run.")

    def _poll(self):
        latest = None
        while True:
            try: latest = self.events.get_nowait()
            except queue.Empty: break
        if latest is not None:
            t, positions, info = latest
            if isinstance(info, str):
                self.status.set(info)
            elif t is not None:
                self.status.set(f"Running | t = {t:.1f} s | {self.run_name}")
                self.applied_text.set("Applied: speed {leader_speed:.2f} rad/s | gap {follow_distance:.2f} m\nRadius {route_radius:.2f} m | altitude {altitude:.2f} m".format(**info))
                if self.run_name == "kinematic": self._update_plot(positions)
        if self.running and self.worker_done:
            self._finished()
        if self.closing and not self.running:
            self.root.destroy()
            return
        if self.root.winfo_exists(): self.root.after(40, self._poll)

    def _post_event(self, event):
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try: self.events.get_nowait()
            except queue.Empty: pass
            try: self.events.put_nowait(event)
            except queue.Full: pass

    def _open_plot(self):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        if self.plot_window is not None and self.plot_window.winfo_exists():
            self.plot_window.destroy()
        self.plot_window = tk.Toplevel(self.root)
        self.plot_window.title("Kinematic reference - ideal motion (no physics)")
        fig = Figure(figsize=(7, 5))
        self.axes = fig.add_subplot(111, projection="3d")
        self.axes.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
        self.plot_lines = [self.axes.plot([], [], [], color=c, label=n)[0]
                           for c, n in [("red", "Leader"), ("blue", "Follower")]]
        self.axes.legend()
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_window)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.plot_window.protocol("WM_DELETE_WINDOW", self._close_plot)

    def _close_plot(self):
        self.stop()
        self.plot_window.destroy()
        self.plot_window = None

    def _update_plot(self, positions):
        if self.plot_window is None or not self.plot_window.winfo_exists(): return
        for points, position in zip(self.plot_points, positions): points.append(position)
        if time.monotonic()-self.last_plot < .1: return
        self.last_plot = time.monotonic()
        for line, points in zip(self.plot_lines, self.plot_points):
            array = np.asarray(points)
            line.set_data_3d(array[:, 0], array[:, 1], array[:, 2])
        all_points = np.concatenate([np.asarray(p) for p in self.plot_points])
        low, high = all_points.min(axis=0)-.4, all_points.max(axis=0)+.4
        self.axes.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
        self.canvas.draw_idle()

    def _close(self):
        self.closing = True
        if self.running:
            self.stop()
            self.status.set("Closing after simulator cleanup. ArduPilot may need time to land.")
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    SimulationUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
