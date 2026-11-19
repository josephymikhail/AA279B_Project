#!/usr/bin/env python3
"""
Natural Motion Circumnavigation (NMC) visualizer
================================================

Shows the relative motion of a chaser around a target in LEO, from the
target's point of view (the Hill / LVLH frame), using relative orbital
elements (ROEs) in the Keplerian, small-separation approximation:

    x (radial)       = -a*de * cos(u) + a*da
    y (along-track)  =  2*a*de * sin(u) - 1.5*a*da*u
    z (cross-track)  =  a*di * sin(u - Delta)

    u      target's mean argument of latitude (advances 2*pi per orbit)
    a*de   length of the relative eccentricity vector  (km)
    a*di   length of the relative inclination vector   (km)
    Delta  angle between the de and di vectors (deg)
           0 = parallel, 180 = antiparallel, 90 = perpendicular
    a*da   semi-major axis mismatch (km). 0 means the no-drift condition
           ydot0 = -2*n*x0 holds; anything else makes the chaser drift.

Usage
-----
    python nmc_visualizer.py                    # interactive animation with sliders
    python nmc_visualizer.py --static           # compare parallel / antiparallel / perpendicular
    python nmc_visualizer.py --save nmc.gif     # write the animation to a file (no window)
    python nmc_visualizer.py --de 3 --di 1.5 --phase 180 --da 0.1

Requires: numpy, matplotlib   (pip install numpy matplotlib)
For --save with .gif, Pillow is also needed (pip install pillow).
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# ----------------------------------------------------------------------------
# Relative motion model
# ----------------------------------------------------------------------------

def hill_state(u, ade, adi, delta_deg, ada=0.0):
    """Hill-frame position (x radial, y along-track, z cross-track) in km."""
    d = np.deg2rad(delta_deg)
    x = -ade * np.cos(u) + ada
    y = 2.0 * ade * np.sin(u) - 1.5 * ada * u
    z = adi * np.sin(u - d)
    return x, y, z


def min_xz_distance(ade, adi, delta_deg, ada=0.0, n=2000):
    """Closest approach to the target in the radial/cross-track (x-z) plane."""
    u = np.linspace(0, 2 * np.pi, n)
    x, _, z = hill_state(u, ade, adi, delta_deg, ada)
    return float(np.min(np.hypot(x - ada, z)))


def min_3d_distance(ade, adi, delta_deg, ada=0.0, n_orbits=1, n=4000):
    """Closest 3D approach to the target over n_orbits."""
    u = np.linspace(0, 2 * np.pi * n_orbits, n)
    x, y, z = hill_state(u, ade, adi, delta_deg, ada)
    return float(np.min(np.sqrt(x**2 + y**2 + z**2)))


def case_label(delta_deg):
    d = delta_deg % 360
    if d < 5 or d > 355:
        return "parallel"
    if abs(d - 180) < 5:
        return "antiparallel"
    if abs(d - 90) < 5 or abs(d - 270) < 5:
        return "perpendicular"
    return "offset"


# ----------------------------------------------------------------------------
# Colors
# ----------------------------------------------------------------------------
C_TARGET = "#c77700"
C_CHASER = "#0a7d8c"
C_WARN = "#c2255c"
C_EARTH = "#b9c7d3"
C_GRID = "#e3e8ed"

ECC_EXAGGERATION = 0.16  # eccentricity used only in the Earth-view panel (not to scale)


# ----------------------------------------------------------------------------
# Interactive / saved animation
# ----------------------------------------------------------------------------

def build_animation(args):
    P = dict(de=args.de, di=args.di, ph=args.phase, da=args.da)

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle("Natural motion circumnavigation, seen four ways", fontsize=14, y=0.98)
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.97, top=0.92, bottom=0.27,
                          wspace=0.18, hspace=0.28)
    ax_earth = fig.add_subplot(gs[0, 0])
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xz = fig.add_subplot(gs[1, 0])
    ax_3d = fig.add_subplot(gs[1, 1], projection="3d")

    # ---- Panel 1: around Earth (inertial, exaggerated) ----
    R, E = 1.0, ECC_EXAGGERATION
    th = np.linspace(0, 2 * np.pi, 400)
    ax_earth.set_aspect("equal")
    ax_earth.add_patch(plt.Circle((0, 0), 0.34, color=C_EARTH))
    ax_earth.text(0, 0, "Earth", ha="center", va="center", fontsize=9)
    ax_earth.plot(R * np.cos(th), R * np.sin(th), "--", color=C_TARGET, alpha=0.6, lw=1.4,
                  label="Target orbit (circle)")
    rch = R * (1 - E * np.cos(th))
    ax_earth.plot(rch * np.cos(th), rch * np.sin(th), color=C_CHASER, lw=1.6,
                  label="Chaser orbit (same a, slightly elliptical)")
    t_dot_e, = ax_earth.plot([], [], "o", color=C_TARGET, ms=9)
    c_dot_e, = ax_earth.plot([], [], "o", color=C_CHASER, ms=7)
    link_e, = ax_earth.plot([], [], color="gray", alpha=0.5, lw=1)
    ax_earth.set_xlim(-1.35, 1.35)
    ax_earth.set_ylim(-1.35, 1.35)
    ax_earth.set_xticks([])
    ax_earth.set_yticks([])
    ax_earth.set_title("1. Around Earth (eccentricity exaggerated)", fontsize=10)
    ax_earth.legend(loc="lower center", fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02))

    # ---- Panel 2: x-y ----
    ax_xy.set_title("2. Orbit plane (x-y): the 2:1 ellipse", fontsize=10)
    ax_xy.set_xlabel("y along-track (km)")
    ax_xy.set_ylabel("x radial (km)")
    ax_xy.set_xlim(-10, 10)
    ax_xy.set_ylim(-6.5, 6.5)
    ax_xy.grid(color=C_GRID)
    ax_xy.axhline(0, color="gray", lw=0.8)
    ax_xy.axvline(0, color="gray", lw=0.8)
    path_xy, = ax_xy.plot([], [], color=C_CHASER, alpha=0.5, lw=1.5)
    c_dot_xy, = ax_xy.plot([], [], "o", color=C_CHASER, ms=8)
    ax_xy.plot(0, 0, "o", color=C_TARGET, ms=10)
    ax_xy.annotate("target", (0, 0), xytext=(8, -16), textcoords="offset points",
                   color=C_TARGET, fontsize=9)

    # ---- Panel 3: x-z ----
    ax_xz.set_title("3. Looking along the orbit (x-z)", fontsize=10)
    ax_xz.set_xlabel("z cross-track (km)")
    ax_xz.set_ylabel("x radial (km)")
    ax_xz.set_xlim(-6, 6)
    ax_xz.set_ylim(-4.5, 4.5)
    ax_xz.set_aspect("equal")
    ax_xz.grid(color=C_GRID)
    ax_xz.axhline(0, color="gray", lw=0.8)
    ax_xz.axvline(0, color="gray", lw=0.8)
    path_xz, = ax_xz.plot([], [], color=C_CHASER, alpha=0.5, lw=1.5)
    c_dot_xz, = ax_xz.plot([], [], "o", color=C_CHASER, ms=8)
    ax_xz.plot(0, 0, "o", color=C_TARGET, ms=10)
    safe_circle = plt.Circle((0, 0), 0.0, fill=False, ls="--", color=C_WARN, lw=1.5)
    ax_xz.add_patch(safe_circle)
    xz_text = ax_xz.text(0.02, 0.03, "", transform=ax_xz.transAxes, fontsize=9, color=C_WARN)

    # ---- Panel 4: 3D ----
    ax_3d.set_title("4. All three together (drag to rotate)", fontsize=10)
    ax_3d.set_xlabel("y")
    ax_3d.set_ylabel("z")
    ax_3d.set_zlabel("x")
    ax_3d.set_xlim(-8, 8)
    ax_3d.set_ylim(-5, 5)
    ax_3d.set_zlim(-5, 5)
    ax_3d.view_init(elev=22, azim=-55)
    path_3d, = ax_3d.plot([], [], [], color=C_CHASER, alpha=0.6, lw=1.6)
    c_dot_3d, = ax_3d.plot([], [], [], "o", color=C_CHASER, ms=7)
    ax_3d.plot([0], [0], [0], "o", color=C_TARGET, ms=9)

    # ---- Sliders and buttons ----
    sl_de = Slider(fig.add_axes([0.12, 0.17, 0.30, 0.025]), "a·δe (km)", 0.5, 3.0, valinit=P["de"])
    sl_di = Slider(fig.add_axes([0.12, 0.12, 0.30, 0.025]), "a·δi (km)", 0.0, 3.0, valinit=P["di"])
    sl_ph = Slider(fig.add_axes([0.60, 0.17, 0.30, 0.025]), "Angle δe↔δi (°)", 0, 180, valinit=P["ph"])
    sl_da = Slider(fig.add_axes([0.60, 0.12, 0.30, 0.025]), "a·δa drift (km)", 0.0, 1.0, valinit=P["da"])
    b_par = Button(fig.add_axes([0.12, 0.04, 0.12, 0.04]), "Parallel")
    b_anti = Button(fig.add_axes([0.26, 0.04, 0.14, 0.04]), "Antiparallel")
    b_perp = Button(fig.add_axes([0.42, 0.04, 0.14, 0.04]), "Perpendicular")
    b_play = Button(fig.add_axes([0.58, 0.04, 0.10, 0.04]), "Pause")
    readout = fig.text(0.70, 0.045, "", fontsize=9, va="center")

    state = {"u": 0.0, "playing": True}

    def refresh_paths():
        n_orb = 1 if P["da"] == 0 else 3
        uu = np.linspace(0, 2 * np.pi * n_orb, 600 * n_orb)
        x, y, z = hill_state(uu, P["de"], P["di"], P["ph"], P["da"])
        path_xy.set_data(y, x)
        path_xz.set_data(z, x)
        path_3d.set_data(y, z)
        path_3d.set_3d_properties(x)

        m = min_xz_distance(P["de"], P["di"], P["ph"], P["da"])
        safe_circle.set_radius(m)
        xz_text.set_text(f"{case_label(P['ph'])}: closest x-z approach = {m:.2f} km")
        d3 = min_3d_distance(P["de"], P["di"], P["ph"], P["da"], n_orbits=n_orb)
        readout.set_text(f"min 3D distance (over {n_orb} orbit{'s' if n_orb > 1 else ''}): {d3:.2f} km")

    def on_slider(_):
        P["de"], P["di"], P["ph"], P["da"] = sl_de.val, sl_di.val, sl_ph.val, sl_da.val
        refresh_paths()
        fig.canvas.draw_idle()

    for s in (sl_de, sl_di, sl_ph, sl_da):
        s.on_changed(on_slider)

    b_par.on_clicked(lambda e: sl_ph.set_val(0))
    b_anti.on_clicked(lambda e: sl_ph.set_val(180))
    b_perp.on_clicked(lambda e: sl_ph.set_val(90))

    def toggle(e):
        state["playing"] = not state["playing"]
        b_play.label.set_text("Pause" if state["playing"] else "Play")
        fig.canvas.draw_idle()

    b_play.on_clicked(toggle)

    def update(frame):
        if state["playing"]:
            state["u"] = frame * args.step
        u = state["u"]
        x, y, z = hill_state(u, P["de"], P["di"], P["ph"], P["da"])
        # Earth view
        t_dot_e.set_data([R * np.cos(u)], [R * np.sin(u)])
        rc = R * (1 - E * np.cos(u))
        c_dot_e.set_data([rc * np.cos(u)], [rc * np.sin(u)])
        link_e.set_data([R * np.cos(u), rc * np.cos(u)], [R * np.sin(u), rc * np.sin(u)])
        # Hill frame views
        c_dot_xy.set_data([y], [x])
        c_dot_xz.set_data([z], [x])
        c_dot_3d.set_data([y], [z])
        c_dot_3d.set_3d_properties([x])
        return (t_dot_e, c_dot_e, link_e, c_dot_xy, c_dot_xz, c_dot_3d)

    refresh_paths()
    # keep references alive
    fig._nmc_widgets = (sl_de, sl_di, sl_ph, sl_da, b_par, b_anti, b_perp, b_play)
    anim = FuncAnimation(fig, update, frames=args.frames, interval=30,
                         blit=False, repeat=True)
    return fig, anim


# ----------------------------------------------------------------------------
# Static comparison figure
# ----------------------------------------------------------------------------

def static_figure(args, out_path):
    cases = [("Parallel (0°)", 0.0), ("Antiparallel (180°)", 180.0), ("Perpendicular (90°)", 90.0)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    fig.suptitle(
        f"Effect of δe/δi alignment   (a·δe = {args.de:.1f} km, a·δi = {args.di:.1f} km, δa = 0)",
        fontsize=13)
    u = np.linspace(0, 2 * np.pi, 600)
    for col, (name, d) in enumerate(cases):
        x, y, z = hill_state(u, args.de, args.di, d, 0.0)

        ax = axes[0, col]
        ax.plot(z, x, color=C_CHASER, lw=2)
        ax.plot(0, 0, "o", color=C_TARGET, ms=10)
        m = min_xz_distance(args.de, args.di, d)
        if m > 0.02:
            ax.add_patch(plt.Circle((0, 0), m, fill=False, ls="--", color=C_WARN))
        ax.annotate("", xy=(z[40], x[40]), xytext=(z[30], x[30]),
                    arrowprops=dict(arrowstyle="->", color=C_CHASER, lw=2))
        ax.set_title(f"{name}\nx-z view, closest approach {m:.2f} km", fontsize=10)
        ax.set_xlabel("z cross-track (km)")
        ax.set_ylabel("x radial (km)")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.grid(color=C_GRID)

        ax = axes[1, col]
        ax.plot(y, x, color=C_CHASER, lw=2)
        ax.plot(0, 0, "o", color=C_TARGET, ms=10)
        ax.set_title("x-y view (orbit plane)", fontsize=10)
        ax.set_xlabel("y along-track (km)")
        ax.set_ylabel("x radial (km)")
        ax.set_xlim(-9, 9)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.grid(color=C_GRID)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    return fig


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Natural motion circumnavigation visualizer")
    ap.add_argument("--de", type=float, default=2.0, help="a*delta_e magnitude in km (default 2.0)")
    ap.add_argument("--di", type=float, default=2.0, help="a*delta_i magnitude in km (default 2.0)")
    ap.add_argument("--phase", type=float, default=0.0,
                    help="angle between delta_e and delta_i in degrees (default 0)")
    ap.add_argument("--da", type=float, default=0.0, help="a*delta_a mismatch in km (default 0)")
    ap.add_argument("--static", action="store_true",
                    help="make a static parallel/antiparallel/perpendicular comparison figure")
    ap.add_argument("--save", metavar="FILE",
                    help="save the animation to FILE (.gif or .mp4) instead of opening a window")
    ap.add_argument("--frames", type=int, default=240, help="number of animation frames")
    ap.add_argument("--step", type=float, default=2 * np.pi / 120,
                    help="advance in u (radians) per frame; default is 120 frames per orbit")
    args = ap.parse_args()

    if args.static:
        static_figure(args, "nmc_comparison.png")
        if not args.save:
            plt.show()
        return

    fig, anim = build_animation(args)
    if args.save:
        writer = "pillow" if args.save.lower().endswith(".gif") else None
        anim.save(args.save, writer=writer, fps=30, dpi=90)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
