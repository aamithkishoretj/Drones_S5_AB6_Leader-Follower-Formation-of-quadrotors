#!/usr/bin/env python3
"""Desktop live-control UI for the leader-follower drone simulation."""
from __future__ import annotations

import os
import queue
import sys
import threading
import ctypes
import ctypes.wintypes

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
        root.geometry("1120x760")
        root.minsize(900, 620)
        self.events = queue.Queue(maxsize=1)
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
        ttk.Label(outer, text="Choose a simulator, launch the flight, and adjust its controls while it runs.").pack(anchor="w", pady=(3, 12))
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        panel = ttk.Frame(body, padding=(0, 0, 16, 0), width=280)
        panel.pack(side="left", fill="y")
        panel.pack_propagate(False)
        self.simulator = tk.StringVar(value="pybullet")
        self.trajectory = tk.StringVar(value="lemniscate")
        self.controller = tk.StringVar(value="analytic")
        self._choice(panel, "Simulator", self.simulator, ["pybullet", "gazebo", "mujoco", "kinematic"])
        self._choice(panel, "Leader route", self.trajectory, ["lemniscate", "potato_chip"])
        self._choice(panel, "Controller", self.controller, ["analytic", "learned"])
        ttk.Label(panel, text="Live parameters", style="Header.TLabel").pack(anchor="w", pady=(16, 3))
        self.speed = tk.DoubleVar(value=float(np.pi / 15))
        self.distance = tk.DoubleVar(value=0.8)
        self.radius = tk.DoubleVar(value=0.85)
        self.altitude = tk.DoubleVar(value=1.0)
        self._scale(panel, "Leader speed (rad/s)", self.speed, 0.03, 0.8, "leader_speed")
        self._scale(panel, "Follower gap (m)", self.distance, 0.25, 2.5, "follow_distance")
        self._scale(panel, "Route radius (m)", self.radius, 0.2, 2.0, "route_radius")
        self._scale(panel, "Flight altitude (m)", self.altitude, 0.3, 3.0, "altitude")
        self.status = tk.StringVar(value="Ready")
        ttk.Label(panel, textvariable=self.status, wraplength=255).pack(anchor="w", pady=(10, 12))
        buttons = ttk.Frame(panel)
        buttons.pack(fill="x", side="bottom")
        self.start_button = ttk.Button(buttons, text="▶  Start simulation", command=self.start)
        self.start_button.pack(fill="x", pady=3)
        self.stop_button = ttk.Button(buttons, text="■  Stop simulation", command=self.stop, state="disabled")
        self.stop_button.pack(fill="x", pady=3)

        self.native_host = ttk.Frame(body, relief="sunken", borderwidth=2)
        self.native_host.pack(side="right", fill="both", expand=True)
        self.native_message = tk.StringVar(
            value="Start a run to display the selected simulator's native application here."
        )
        self.native_label = ttk.Label(
            self.native_host, textvariable=self.native_message, anchor="center",
            justify="center", wraplength=560,
        )
        self.native_label.place(relx=.5, rely=.5, anchor="center")
        self.native_hwnd = None
        self.native_host.bind("<Configure>", self._resize_native_window)

    def _choice(self, parent, label, variable, options):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(7, 2))
        box = ttk.Combobox(parent, textvariable=variable, values=options, state="readonly")
        box.pack(fill="x")
        setattr(self, label.lower().replace(" ", "_"), box)

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
                         orient="horizontal", showvalue=False, length=230, command=changed,
                         bg="#101820", fg="#e6edf3", troughcolor="#344454", highlightthickness=0)
        scale.pack(fill="x")
        changed(variable.get())

    def start(self):
        if self.running:
            return
        name = self.simulator.get()
        trajectory_name = self.trajectory.get()
        try:
            if trajectory_name == "lemniscate":
                traj = LeaderTrajectory(LemniscateParams(w_d=self.speed.get()))
            else:
                traj = PotatoChipTrajectory(PotatoChipParams(w=self.speed.get()))
            cfg = SimConfig(simulator=name, controller=self.controller.get(),
                            learned_model=os.path.join(_ROOT, "models", f"{name}_response.npz"),
                            gui=True, duration_sec=1e9, trails=True, follow_distance=self.distance.get(),
                            follower_offset_mode="path")
            sim = LeaderFollowerSimulation(get_gains("real_eig"), cfg, traj)
        except Exception as exc:
            messagebox.showerror("Could not start simulation", str(exc))
            return
        self.native_hwnd = None
        self.stop_event = threading.Event()
        self.running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.native_message.set(self._viewer_start_message(name))
        self.status.set(f"Starting {name} native viewer…")

        def on_step(t, states):
            try:
                self.events.put_nowait((t, None, None))
            except queue.Full:
                try: self.events.get_nowait()
                except queue.Empty: pass
                try: self.events.put_nowait((t, None, None))
                except queue.Full: pass
        def run():
            try:
                sim.run(stop_event=self.stop_event, step_callback=on_step, live_parameters=self.live)
                self._post_event((None, None, "Simulation stopped."))
            except SimulationStopped as exc:
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
            if info:
                self.status.set(info)
            elif t is not None:
                self.status.set(f"Running • t = {t:.1f} s • {self.simulator.get()}")
        if self.running and self.native_hwnd is None and self.simulator.get() in ("pybullet", "mujoco"):
            self._try_attach_native_window()
        if self.running and self.worker_done:
            self._finished()
        if self.root.winfo_exists(): self.root.after(40, self._poll)

    def _post_event(self, event):
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try: self.events.get_nowait()
            except queue.Empty: pass
            try: self.events.put_nowait(event)
            except queue.Full: pass

    def _viewer_start_message(self, backend):
        if backend == "kinematic":
            return "Kinematic mode has no native simulator application. Choose PyBullet or MuJoCo for a physics viewer."
        if backend == "gazebo":
            return "Starting Gazebo's native application. Gazebo runs in WSL and opens in its own WSLg window."
        return f"Starting the {backend} native viewer…"

    def _try_attach_native_window(self):
        if sys.platform != "win32":
            return
        user32 = ctypes.windll.user32
        title_words = {"pybullet": ("pybullet", "bullet"), "mujoco": ("mujoco",)}.get(self.simulator.get(), ())
        windows = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

        def collect(hwnd, _):
            if not user32.IsWindowVisible(hwnd) or int(hwnd) == int(self.root.winfo_id()):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            title = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, title, length + 1)
            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if title.value:
                windows.append((int(hwnd), title.value, pid.value))
            return True

        callback = callback_type(collect)
        user32.EnumWindows(callback, 0)
        candidates = [item for item in windows if any(word in item[1].lower() for word in title_words)]
        if not candidates:
            candidates = [item for item in windows if item[2] == os.getpid()]
        if not candidates:
            return
        hwnd, title, _pid = candidates[0]
        try:
            host = int(self.native_host.winfo_id())
            style = user32.GetWindowLongW(hwnd, -16)  # GWL_STYLE
            style = (style & ~0x80000000) | 0x40000000  # WS_POPUP -> WS_CHILD
            user32.SetWindowLongW(hwnd, -16, style)
            user32.SetParent(hwnd, host)
            self.native_hwnd = hwnd
            self.native_label.place_forget()
            self.native_message.set(f"Native {self.simulator.get()} viewer: {title}")
            self._resize_native_window()
        except Exception as exc:
            self.native_message.set(f"Native viewer is running in its own window ({exc}).")

    def _resize_native_window(self, _event=None):
        if self.native_hwnd is None or sys.platform != "win32":
            return
        user32 = ctypes.windll.user32
        user32.MoveWindow(self.native_hwnd, 0, 0, max(1, self.native_host.winfo_width()),
                          max(1, self.native_host.winfo_height()), True)

    def _close(self):
        if self.stop_event: self.stop_event.set()
        self.root.destroy()


def main():
    root = tk.Tk()
    SimulationUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
