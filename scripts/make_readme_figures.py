#!/usr/bin/env python3
"""Generate the schematic README figures in assets/.

    python scripts/make_readme_figures.py

Produces:
    assets/fig7_formation_geometry.png    (Section 10, path-following formation)
    assets/fig8_dual_quaternion_pose.png  (Section 3.5, pose as one dual quaternion)
    assets/fig9_thrust_alignment.png      (Section 9, thrust-vector alignment)

3D scenes are drawn with a fixed oblique projection onto the page, so every label
position is controlled explicitly.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Polygon

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
from dq_control import Quaternion  # noqa: E402

ASSETS = os.path.join(_ROOT, "assets")
LEADER = "#d62728"     # red, same as the simulator trails
FOLLOWER = "#1f5fbf"   # blue, same as the simulator trails
WORLD = "#555555"
ACCENT = "#e07b00"
GREEN = "#2a9d3f"
PURPLE = "#7b3fbf"
GHOST = "#b0b0b0"
plt.rcParams.update({"font.size": 13, "mathtext.fontset": "cm"})

# Oblique (cabinet-style) projection: x to the right, z up, y receding up-right.
_DEPTH, _ANGLE = 0.5, np.deg2rad(32)


def set_view(depth, angle_deg):
    global _DEPTH, _ANGLE
    _DEPTH, _ANGLE = depth, np.deg2rad(angle_deg)


def proj(p):
    p = np.asarray(p, dtype=float)
    return np.array([p[0] + _DEPTH * p[1] * np.cos(_ANGLE), p[2] + _DEPTH * p[1] * np.sin(_ANGLE)])


def arrow(ax, a, b, color, lw=2.2, ms=16, ls="-", z=5, style="-|>"):
    ax.add_patch(FancyArrowPatch(proj(a), proj(b), arrowstyle=style, mutation_scale=ms, color=color,
                                 lw=lw, linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))


def line(ax, pts, color, lw=1.5, ls="-", z=3, alpha=1.0):
    q = np.array([proj(p) for p in pts])
    ax.plot(q[:, 0], q[:, 1], color=color, lw=lw, ls=ls, zorder=z, alpha=alpha, solid_capstyle="round")


def label(ax, p, text, color="black", dx=0.0, dy=0.0, fs=14, ha="center", va="center", z=8):
    q = proj(p)
    ax.text(q[0] + dx, q[1] + dy, text, color=color, fontsize=fs, ha=ha, va=va, zorder=z)


def frame(ax, o, R, length, color, names, fs=14, lw=2.2, offsets=None):
    offsets = offsets or [(0, 0)] * 3
    for k in range(3):
        tip = o + R[:, k] * length
        arrow(ax, o, tip, color, lw=lw)
        if names[k]:
            label(ax, o + R[:, k] * length * 1.2, names[k], color, dx=offsets[k][0], dy=offsets[k][1], fs=fs)


def drone(ax, p, R, size, color):
    """X-frame quadrotor: two arms and four rotor discs in the body x-y plane."""
    for a in (np.array([1.0, 1.0, 0.0]), np.array([1.0, -1.0, 0.0])):
        a = a / np.linalg.norm(a) * size
        line(ax, [p + R @ a, p - R @ a], "#222222", lw=3.0, z=6)
    th = np.linspace(0, 2 * np.pi, 48)
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        c = np.array([sx, sy, 0.0]) / np.sqrt(2) * size
        ring = [p + R @ (c + 0.42 * size * np.array([np.cos(t), np.sin(t), 0.0])) for t in th]
        line(ax, ring, color, lw=1.8, z=6)
    q = proj(p)
    ax.scatter([q[0]], [q[1]], color=color, s=36, zorder=7)


def yaw_matrix(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ground(ax, xs, ys):
    for x in xs:
        line(ax, [(x, ys[0], 0), (x, ys[-1], 0)], "#e4e4e4", lw=0.8, z=1)
    for y in ys:
        line(ax, [(xs[0], y, 0), (xs[-1], y, 0)], "#e4e4e4", lw=0.8, z=1)


def equation_box(ax, x, y, title, lines, fs=13.5):
    text = "\n".join([title] + lines)
    ax.text(x, y, text, fontsize=fs, va="top", ha="left", linespacing=1.95, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.7", facecolor="#f7f7f7", edgecolor="#bbbbbb"))


def new_figure(width, height):
    fig = plt.figure(figsize=(width, height), dpi=150)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(ASSETS, name), bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------ figure 7
def fig_formation():
    set_view(0.9, 38)
    t = np.linspace(0.0, 1.0, 800)
    path = np.stack([7.5 * t, 1.3 * np.sin(2.0 * np.pi * t), 1.6 + 0.15 * np.sin(np.pi * t)], axis=1)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    d_gap = None
    p_L = path[-1]
    d_gap = s[-1] - s[len(path) // 2]          # follower sits half-way along this illustration
    idx = int(np.searchsorted(s, s[-1] - d_gap))
    p_F = path[idx]

    def heading(i):
        d = path[min(i + 1, len(path) - 1)] - path[max(i - 1, 0)]
        return yaw_matrix(np.arctan2(d[1], d[0]))

    fig, ax = new_figure(17, 7.4)
    ground(ax, np.linspace(-1.0, 7.5, 18), np.linspace(-2.0, 2.0, 9))

    # world frame
    o = np.array([-1.0, -2.0, 0.0])
    frame(ax, o, np.eye(3), 0.9, WORLD, [r"$x_w$", r"$y_w$", r"$z_w$"], lw=2.4, offsets=[(0.15, -0.05), (0.1, 0.05), (-0.2, 0.0)])
    label(ax, o, r"world $\{W\}$", WORLD, dx=0.05, dy=-0.35, fs=15)

    # position vectors from the world origin
    line(ax, [o, p_F], FOLLOWER, lw=1.3, ls="--", z=2, alpha=0.75)
    line(ax, [o, p_L], LEADER, lw=1.3, ls="--", z=2, alpha=0.75)
    label(ax, 0.45 * o + 0.55 * p_F, r"$p_{Fd}$", FOLLOWER, dx=-0.15, dy=0.28, fs=16)
    label(ax, 0.5 * o + 0.5 * p_L, r"$p_L$", LEADER, dx=0.1, dy=-0.3, fs=16)

    # recorded trail with sampled history, and the last d_gap of trail
    line(ax, path[: idx + 1], LEADER, lw=2.0, alpha=0.5)
    samples = np.array([proj(p) for p in path[:idx:40]])
    ax.scatter(samples[:, 0], samples[:, 1], color=LEADER, s=16, zorder=4, alpha=0.8)
    line(ax, path[idx:], ACCENT, lw=7.0, alpha=0.85, z=3)
    label(ax, path[len(path) // 10], "recorded leader trail\n" + r"$p_{L,0},\ p_{L,1},\ \ldots,\ p_{L,k}$", LEADER,
          dx=-1.0, dy=0.85, fs=14)
    mid = path[(idx + 3 * len(path)) // 4]
    label(ax, mid, r"$d_{\mathrm{gap}}$ (arc length along the trail)", ACCENT, dx=0.6, dy=-0.62, fs=16)
    line(ax, [p_F, p_L], "#555555", lw=2.0, ls=":", z=4)
    label(ax, 0.5 * (p_F + p_L), r"straight-line gap $< d_{\mathrm{gap}}$", "#555555", dx=0.4, dy=-0.38, fs=13)

    # bracketing samples a, b around the follower target
    for k, name, dx, dy in [(idx - 120, r"$p_{L,a}$", -0.5, 0.05), (idx + 120, r"$p_{L,b}$", 0.55, -0.1)]:
        q = proj(path[k])
        ax.scatter([q[0]], [q[1]], color="black", s=30, zorder=9)
        ax.text(q[0] + dx, q[1] + dy, name, fontsize=14, ha="center", va="center", zorder=9)

    # vehicles and body frames
    R_L, R_F = heading(len(path) - 1), heading(idx)
    drone(ax, p_L, R_L, 0.38, LEADER)
    frame(ax, p_L, R_L, 0.85, LEADER, [r"$x_L$", r"$y_L$", r"$z_L$"], offsets=[(0.1, -0.15), (0.15, 0.05), (0.0, 0.08)])
    label(ax, p_L, r"leader $\{L\}$", LEADER, dx=0.2, dy=1.45, fs=16)
    drone(ax, p_F, R_F, 0.38, FOLLOWER)
    frame(ax, p_F, R_F, 0.85, FOLLOWER, [r"$x_F$", r"$y_F$", r"$z_F$"], offsets=[(-0.3, -0.05), (0.15, 0.05), (0.0, 0.08)])
    label(ax, p_F, r"follower target $\{F\}$", FOLLOWER, dx=-0.2, dy=1.45, fs=16)

    ax.set_xlim(-2.4, 17.6)
    ax.set_ylim(-2.2, 5.0)
    ax.text(-2.3, 4.9, "Leader–follower formation geometry", fontsize=19, va="top")
    equation_box(ax, 0.62, 0.86, r"$\mathbf{Path\ following\ (Section\ 10.1)}$", [
        r"Leader pose: $Q_L = \bar{q}_L + \varepsilon\,\frac{1}{2}\,\bar{p}_L \circ \bar{q}_L$",
        r"Trail length: $s_k = s_{k-1} + \Vert p_{L,k} - p_{L,k-1} \Vert$",
        r"Target: $s_F^{\mathrm{target}} = \max(0,\ s_k - d_{\mathrm{gap}})$",
        r"Position: $p_{Fd} = (1-\tau)\,p_{L,a} + \tau\,p_{L,b}$",
        r"Attitude: $\bar{q}_{Fd} = \mathrm{SLERP}(\bar{q}_{L,a},\ \bar{q}_{L,b},\ \tau)$",
        r"Follower pose: $Q_{Fd} = \bar{q}_{Fd} + \varepsilon\,\frac{1}{2}\,\bar{p}_{Fd} \circ \bar{q}_{Fd}$",
    ])
    ax.text(0.62, 0.17, "Red: leader and its recorded trail.   Blue: follower reference.\n"
            "Orange: the last $d_{\\mathrm{gap}}$ of trail.  On curves the straight-line gap is shorter.",
            transform=ax.transAxes, fontsize=12, color="#444444", va="top", linespacing=1.6)
    save(fig, "fig7_formation_geometry.png")


# ------------------------------------------------------------ figure 8
def fig_pose():
    set_view(0.55, 30)
    n = np.array([0.35, 0.15, 1.0])
    n /= np.linalg.norm(n)
    theta = np.deg2rad(60)
    R = Quaternion.from_rotvec(n, theta).rotation_matrix()
    p = np.array([4.0, 0.6, 2.5])
    x_b = np.array([1.5, -0.9, -0.9])
    x_w = R @ x_b + p
    o = np.zeros(3)
    L = 1.35

    fig, ax = new_figure(16, 7.4)
    ground(ax, np.linspace(-0.4, 7.0, 12), np.linspace(-0.6, 2.4, 5))

    frame(ax, o, np.eye(3), 1.1, WORLD, [r"$x_w$", r"$y_w$", r"$z_w$"], lw=2.4,
          offsets=[(0.12, -0.12), (0.35, 0.05), (-0.22, 0.0)])
    label(ax, o, r"world $\{W\}$", WORLD, dx=-0.1, dy=-0.42, fs=15)

    # translation p
    arrow(ax, o, p, "black", lw=1.6, ls="--", ms=14, z=4)
    label(ax, 0.45 * p, r"translation $p$", "black", dx=-0.55, dy=0.35, fs=15)

    # frame after translation only (gray), rotation axis n and angle theta
    frame(ax, p, np.eye(3), L, GHOST, ["", "", ""], lw=1.8)
    line(ax, [p - 0.9 * n, p + 2.3 * n], PURPLE, lw=1.8, ls="-.", z=4)
    label(ax, p + 2.35 * n, r"rotation axis $n$", PURPLE, dy=0.28, fs=15)
    u = np.cross(n, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    centre = p + 1.75 * n
    angles = np.linspace(-0.6, theta - 0.6, 40)
    arc = [centre + 0.9 * (np.cos(a) * u + np.sin(a) * w) for a in angles]
    line(ax, arc[:-1], PURPLE, lw=2.4, z=5)
    arrow(ax, arc[-4], arc[-1], PURPLE, lw=2.4, ms=18)
    a_mid = angles[len(angles) // 2]
    label(ax, centre + 1.25 * (np.cos(a_mid) * u + np.sin(a_mid) * w), r"$\theta$", PURPLE, fs=19)

    # resulting body frame and vehicle
    drone(ax, p, R, 0.45, FOLLOWER)
    frame(ax, p, R, L, FOLLOWER, [r"$x_b$", r"$y_b$", r"$z_b$"], lw=2.4,
          offsets=[(0.15, -0.1), (-0.05, 0.22), (-0.05, 0.3)])
    label(ax, p, r"body $\{B\}$", FOLLOWER, dx=-1.3, dy=-0.55, fs=15)

    # a point fixed on the body: body vector (green) and world position (orange)
    arrow(ax, p, x_w, GREEN, lw=2.4, ms=18, z=6)
    label(ax, 0.5 * (p + x_w), r"$R(\bar{q})\,x_b$", GREEN, dx=0.65, dy=0.1, fs=16)
    line(ax, [o, x_w], ACCENT, lw=1.8, ls=":", z=3)
    q = proj(x_w)
    ax.scatter([q[0]], [q[1]], color=ACCENT, s=60, zorder=9)
    label(ax, x_w, r"point $x$", ACCENT, dx=0.7, dy=-0.15, fs=16)
    label(ax, 0.62 * x_w, r"$x_w$", ACCENT, dx=0.1, dy=-0.32, fs=16)

    ax.set_xlim(-1.4, 15.6)
    ax.set_ylim(-1.0, 5.8)
    ax.text(-1.3, 5.7, "A unit dual quaternion as a rigid-body pose", fontsize=19, va="top")
    equation_box(ax, 0.59, 0.84, r"$\mathbf{One\ object\ for\ the\ whole\ pose\ (Section\ 3.5)}$", [
        r"Rotation: $\bar{q} = (n \sin\frac{\theta}{2},\ \cos\frac{\theta}{2})$",
        r"Translation: $\bar{p} = (p,\ 0)$",
        r"Pose: $Q = \bar{q} + \varepsilon\,\frac{1}{2}\,\bar{p} \circ \bar{q} = (1 + \varepsilon\,\frac{1}{2}\,\bar{p}) \circ \bar{q}$",
        r"Point map: $x_w = R(\bar{q})\,x_b + p$",
        r"Recovery: $\bar{q} = \mathcal{P}(Q),\quad \bar{p} = 2\,\mathcal{D}(Q) \circ \mathcal{P}(Q)^{*}$",
    ])
    ax.text(0.59, 0.2, "Gray: frame after translating by $p$.   Purple: rotation by $\\theta$ about $n$.\n"
            "Blue: the body frame described by $Q$.   Orange: a body point in world coordinates.",
            transform=ax.transAxes, fontsize=12, color="#444444", va="top", linespacing=1.6)
    save(fig, "fig8_dual_quaternion_pose.png")


# ------------------------------------------------------------ figure 9
def fig_thrust():
    g = 9.81
    a_h = np.array([3.6, 0.0])            # horizontal part of p_dd + K_a (p_dot - v)
    U = a_h + np.array([0.0, g])
    tilt = np.arctan2(U[0], U[1])

    fig, ax = plt.subplots(figsize=(15, 7.0), dpi=150)
    ax.set_aspect("equal")
    ax.axis("off")
    c = np.array([0.0, 0.0])
    scale = 0.32

    ax.plot([-6.5, 5.0], [-3.3, -3.3], color="#999999", linewidth=1.2)
    for k in range(-13, 11):
        ax.plot([0.5 * k, 0.5 * k - 0.25], [-3.3, -3.55], color="#bbbbbb", linewidth=0.8)
    wo = np.array([-6.0, -2.8])
    for v, lab in [((1.2, 0.0), r"$x_w$"), ((0.0, 1.2), r"$z_w$")]:
        ax.add_patch(FancyArrowPatch(wo, wo + v, arrowstyle="-|>", mutation_scale=15, color=WORLD, lw=1.6))
        ax.text(*(wo + 1.17 * np.array(v) + np.array([0.12, 0.08])), lab, color=WORLD, fontsize=15)

    # tilted quadrotor, side view: body z axis (thrust axis) along U_p
    Rb = np.array([[np.cos(tilt), np.sin(tilt)], [-np.sin(tilt), np.cos(tilt)]])
    body = np.array([[-0.45, -0.12], [0.45, -0.12], [0.45, 0.12], [-0.45, 0.12]])
    ax.add_patch(Polygon((Rb @ body.T).T + c, closed=True, facecolor="#333333", edgecolor="#111111", zorder=4))
    for sx in (-1, 1):
        ax.plot(*zip(c, Rb @ np.array([sx * 1.4, 0.05]) + c), color="#222222", linewidth=3, zorder=3)
        rotor = (Rb @ (np.array([[-0.5, 0.0], [0.5, 0.0]]) + np.array([sx * 1.4, 0.22])).T).T + c
        ax.plot(rotor[:, 0], rotor[:, 1], color=FOLLOWER, linewidth=4, solid_capstyle="round", zorder=5)
        post = (Rb @ np.array([[sx * 1.4, 0.05], [sx * 1.4, 0.22]]).T).T + c
        ax.plot(post[:, 0], post[:, 1], color="#222222", linewidth=2, zorder=4)

    def vec(start, v, color, lw=2.6, ms=20, ls="-", z=6):
        ax.add_patch(FancyArrowPatch(start, start + v, arrowstyle="-|>", mutation_scale=ms, color=color,
                                     lw=lw, linestyle=ls, zorder=z, shrinkA=0, shrinkB=0))

    gv, av, Uv = np.array([0.0, g]) * scale, a_h * scale, U * scale
    vec(c, gv, "#7f7f7f")
    vec(c, av, ACCENT)
    vec(c + gv, av, ACCENT, lw=1.4, ls="--", ms=14)
    vec(c + av, gv, "#7f7f7f", lw=1.4, ls="--", ms=14)
    vec(c, Uv, FOLLOWER, lw=3.4, ms=24, z=7)
    vec(c, np.array([0.0, -g]) * 0.22, "#111111", lw=2.0)
    ax.plot([0, 0], [3.2, 3.9], color="#cccccc", linestyle=":", linewidth=1.2)

    ax.text(*(c + gv + np.array([-1.15, -0.15])), r"$g\,e_z$", color="#5f5f5f", fontsize=18)
    ax.text(2.5, -1.55, r"$\ddot{p}_{jd} + K_a(\dot{p}_j - v)$", color=ACCENT, fontsize=16)
    ax.add_patch(FancyArrowPatch((2.45, -1.35), (0.9, -0.05), arrowstyle="-", color=ACCENT, lw=1.0, zorder=5))
    ax.text(*(c + Uv + np.array([0.18, 0.05])), r"$U_p\ \parallel\ n_j$ (thrust axis)", color=FOLLOWER, fontsize=17)
    ax.text(0.2, -2.45, r"$-g\,e_z$ (gravity)", color="#111111", fontsize=15)

    ax.add_patch(Arc(c, 4.4, 4.4, angle=0, theta1=90 - np.degrees(tilt), theta2=90, color=PURPLE, lw=2.2))
    ax.text(1.45, 1.55, r"tilt $= \arctan\,\frac{\Vert a_h \Vert}{g}$", color=PURPLE, fontsize=16)
    ax.add_patch(FancyArrowPatch((1.42, 1.62), (0.55, 2.12), arrowstyle="-", color=PURPLE, lw=1.0, zorder=5))
    ax.text(1.45, 0.95, r"$a_h$: horizontal part of $U_p$", color="#666666", fontsize=13)

    ax.set_xlim(-6.7, 14.0)
    ax.set_ylim(-3.8, 4.4)
    box = "\n".join([
        r"$\mathbf{Thrust\ vector\ alignment\ (Section\ 9)}$",
        r"Desired thrust acceleration:",
        r"$\quad U_p = \ddot{p}_{jd} + K_a(\dot{p}_j - v) + g\,e_z$",
        r"Shortest rotation taking $n_j$ onto $U_p$:",
        r"$\quad \bar{q}'_{j\phi\theta} = (n_j \times U_p,\ \ n_j^{\mathsf{T}} U_p + \Vert U_p \Vert)$",
        r"Add yaw about the thrust axis:",
        r"$\quad \bar{q}_{jd} = \bar{q}_{j\phi\theta} \circ \bar{q}_{j\psi}$",
        r"Hover: $U_p = g\,e_z\ \Rightarrow$ no tilt",
    ])
    ax.text(6.3, 3.1, box, fontsize=14, va="top", ha="left", linespacing=1.95,
            bbox=dict(boxstyle="round,pad=0.7", facecolor="#f7f7f7", edgecolor="#bbbbbb"))
    ax.text(-6.6, 4.35, "How a quadrotor turns a velocity command into a tilt", fontsize=19, va="top")
    fig.savefig(os.path.join(ASSETS, "fig9_thrust_alignment.png"), bbox_inches="tight", pad_inches=0.25,
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    fig_formation()
    fig_pose()
    fig_thrust()
    print("Figures written to", ASSETS)
