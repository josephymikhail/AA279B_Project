#!/usr/bin/env python3
"""
Interactive relative-orbital-element (ROE) visualizer.

Shows the chaser's path in the target's Hill frame (x radial, y along-track,
z cross-track) built from the ROEs, in the Keplerian, small-separation model:

    x = a*da - a|de| cos(u - phi)
    y = a*dl - 1.5*a*da*u + 2 a|de| sin(u - phi)
    z = a|di| sin(u - theta)

Sliders are in meters (the factor "a" is folded in). This is one common sign
convention; check D'Amico's papers before reusing it in guidance code.
J2 and drag are not modeled.

Usage:
    python roe_visualizer.py                       # interactive window
    python roe_visualizer.py --preset "Unsafe"     # start on a preset (substring match)
    python roe_visualizer.py --preset 6 --save out.png   # save a static figure and exit

Requires: numpy, matplotlib
"""
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

TAU = 2 * np.pi

# name, params, explanation
PRESETS = [
    ("Baseline: de only",
     dict(da=0, dl=0, de=500, phi=0, di=0, th=0, orbits=3),
     "Only de is set: a flat 2:1 ellipse in the orbit plane (500 m radial, 1000 m along-track). "
     "No cross-track motion, so not passively safe on its own."),
    ("da != 0: drift",
     dict(da=60, dl=0, de=500, phi=0, di=0, th=0, orbits=5),
     "A small da breaks the closed ellipse: the center shifts radially and the loop walks "
     "backward along-track every orbit (same as violating ydot0 = -2 n x0)."),
    ("dl: offset",
     dict(da=0, dl=900, de=500, phi=0, di=0, th=0, orbits=3),
     "dl slides the whole ellipse along-track without changing its shape: a standoff loop "
     "centered 900 m ahead of the target."),
    ("Bigger |de|",
     dict(da=0, dl=0, de=900, phi=0, di=0, th=0, orbits=3),
     "|de| scales the ellipse. It is the range knob: bigger loop, farther away, lower resolution."),
    ("Change phi",
     dict(da=0, dl=0, de=500, phi=90, di=0, th=0, orbits=3),
     "phi changes timing, not shape: the same ellipse is reached at a different point in the "
     "orbit. Use it to phase the loop against sunlight."),
    ("Add di",
     dict(da=0, dl=0, de=500, phi=0, di=500, th=0, orbits=3),
     "Adding di lifts the ellipse into a 3D loop. With theta = phi the radial/cross-track "
     "motion circles the target."),
    ("Safe: de || di",
     dict(da=0, dl=0, de=500, phi=0, di=500, th=0, orbits=3),
     "In the radial vs cross-track plot the path circles the origin; closest approach there "
     "is about min(a|de|, a|di|), which bounds the 3D distance."),
    ("Unsafe: 90 deg apart",
     dict(da=0, dl=0, de=500, phi=0, di=500, th=90, orbits=3),
     "theta = phi + 90 deg: radial and cross-track motion are in phase and collapse onto a "
     "line through the target in the x-z plot. Only along-track phasing keeps you apart."),
]

# key, label, min, max, step
SLIDERS = [
    ("da", "a*da  [m]", -150, 150, 5),
    ("dl", "a*dl  [m]", -1500, 1500, 25),
    ("de", "a|de|  [m]", 0, 1000, 10),
    ("phi", "phi  [deg]", 0, 360, 5),
    ("di", "a|di|  [m]", 0, 1000, 10),
    ("th", "theta  [deg]", 0, 360, 5),
    ("orbits", "orbits shown", 1, 8, 1),
]


def hill_path(u, p):
    """Hill-frame position (x radial, y along-track, z cross-track) in meters."""
    phi, th = np.radians(p["phi"]), np.radians(p["th"])
    x = p["da"] - p["de"] * np.cos(u - phi)
    y = p["dl"] - 1.5 * p["da"] * u + 2 * p["de"] * np.sin(u - phi)
    z = p["di"] * np.sin(u - th)
    return x, y, z


def path_stats(p):
    u = np.linspace(0, TAU * p["orbits"], 400 * int(p["orbits"]) + 1)
    x, y, z = hill_path(u, p)
    d_xz = np.hypot(x, z)
    d3 = np.sqrt(x**2 + y**2 + z**2)
    smaller = min(p["de"], p["di"])
    return dict(u=u, x=x, y=y, z=z, min3d=d3.min(), min_xz=d_xz.min(),
                safe=(smaller > 0 and d_xz.min() > 0.5 * smaller))


class Visualizer:
    def __init__(self, start_preset=0):
        self.p = dict(PRESETS[start_preset][1])
        self.blurb_text = PRESETS[start_preset][2]
        self.updating = False
        self.frame = 0

        self.fig = plt.figure(figsize=(13, 9))
        self.fig.suptitle("How each relative orbital element shapes the loop", fontsize=14)
        gs = self.fig.add_gridspec(2, 2, left=0.06, right=0.97, top=0.86,
                                   bottom=0.43, hspace=0.55, wspace=0.15)
        self.ax3 = self.fig.add_subplot(gs[0, 0], projection="3d")
        self.axxy = self.fig.add_subplot(gs[0, 1])
        self.axxz = self.fig.add_subplot(gs[1, 0])
        self.axyz = self.fig.add_subplot(gs[1, 1])

        self._setup_axes()
        self._setup_widgets()
        self.refresh()

    # ----------------------------------------------------------------- axes
    def _setup_axes(self):
        trail = "tab:blue"
        self.l3, = self.ax3.plot([], [], [], color=trail, lw=1.8)
        self.d3, = self.ax3.plot([], [], [], "o", color="tab:orange", ms=7)
        self.ax3.plot([0], [0], [0], "o", color="black", ms=6)
        self.ax3.set_xlabel("along-track")
        self.ax3.set_ylabel("cross-track")
        self.ax3.set_zlabel("radial")
        self.ax3.set_title("3D (drag to rotate)", fontsize=10)
        try:
            self.ax3.set_box_aspect((1, 1, 1), zoom=1.15)
        except TypeError:
            self.ax3.set_box_aspect((1, 1, 1))

        specs = [
            (self.axxy, "along-track [m]", "radial [m]", "Radial vs along-track"),
            (self.axxz, "cross-track [m]", "radial [m]", "Radial vs cross-track (safety plane)"),
            (self.axyz, "along-track [m]", "cross-track [m]", "Cross-track vs along-track"),
        ]
        self.lines2d, self.dots2d = [], []
        for ax, xl, yl, title in specs:
            line, = ax.plot([], [], color=trail, lw=1.8)
            dot, = ax.plot([], [], "o", color="tab:orange", ms=7)
            ax.plot([0], [0], "o", color="black", ms=6)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.axvline(0, color="0.6", lw=0.8)
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.set_title(title, fontsize=10)
            ax.grid(alpha=0.3)
            self.lines2d.append(line)
            self.dots2d.append(dot)

        # circle of radius min(de, di) in the x-z plane = guaranteed standoff
        self.safe_circle, = self.axxz.plot([], [], "--", color="tab:green", lw=1.2)

    # -------------------------------------------------------------- widgets
    def _setup_widgets(self):
        # preset buttons
        n = len(PRESETS)
        w = 0.94 / n
        self.buttons = []
        for i, (name, _, _) in enumerate(PRESETS):
            bax = self.fig.add_axes([0.03 + i * w, 0.89, w - 0.006, 0.04])
            b = Button(bax, name.replace(": ", ":\n", 1), hovercolor="0.85")
            b.label.set_fontsize(8)
            b.on_clicked(lambda _e, i=i: self.apply_preset(i))
            self.buttons.append(b)

        self.readout = self.fig.text(0.05, 0.36, "", fontsize=10)
        self.blurb = self.fig.text(0.05, 0.31, "", fontsize=9.5, wrap=True, color="0.25")

        self.sliders = {}
        ys = [0.21, 0.15, 0.09, 0.03]
        for idx, (key, label, lo, hi, step) in enumerate(SLIDERS):
            col, row = divmod(idx, 4)
            x0 = 0.12 if col == 0 else 0.60
            sax = self.fig.add_axes([x0, ys[row], 0.30, 0.03])
            s = Slider(sax, label, lo, hi, valinit=self.p[key], valstep=step)
            s.on_changed(lambda v, k=key: self.on_slider(k, v))
            self.sliders[key] = s

    # ------------------------------------------------------------ callbacks
    def on_slider(self, key, val):
        if self.updating:
            return
        self.p[key] = float(val)
        self.blurb_text = ""
        self.refresh()

    def apply_preset(self, i):
        self.updating = True
        self.p = dict(PRESETS[i][1])
        for k, s in self.sliders.items():
            s.set_val(self.p[k])
        self.updating = False
        self.blurb_text = PRESETS[i][2]
        self.frame = 0
        self.refresh()

    # -------------------------------------------------------------- drawing
    def refresh(self):
        self.st = path_stats(self.p)
        st = self.st
        x, y, z = st["x"], st["y"], st["z"]

        self.l3.set_data(y, z)
        self.l3.set_3d_properties(x)
        self.lines2d[0].set_data(y, x)
        self.lines2d[1].set_data(z, x)
        self.lines2d[2].set_data(y, z)

        ext = lambda a: max(np.abs(a).max(), 200.0) * 1.15
        ex, ey, ez = ext(x), ext(y), ext(z)
        r = max(ex, ey, ez)
        self.ax3.set_xlim(-r, r)
        self.ax3.set_ylim(-r, r)
        self.ax3.set_zlim(-r, r)
        for ax, h, v in [(self.axxy, ey, ex), (self.axxz, ez, ex), (self.axyz, ey, ez)]:
            ax.set_xlim(-h, h)
            ax.set_ylim(-v, v)
            ax.set_aspect("equal", adjustable="box")

        smaller = min(self.p["de"], self.p["di"])
        c = np.linspace(0, TAU, 100)
        if smaller > 0:
            self.safe_circle.set_data(smaller * np.sin(c), smaller * np.cos(c))
        else:
            self.safe_circle.set_data([], [])

        flag = "holds (de and di sized and aligned)" if st["safe"] else "not guaranteed"
        self.readout.set_text(
            f"Closest approach, 3D: {st['min3d']:.0f} m     "
            f"radial/cross-track only: {st['min_xz']:.0f} m     "
            f"radial/cross-track separation: {flag}"
        )
        self.readout.set_color("tab:green" if st["safe"] else "tab:red")
        self.blurb.set_text(self._wrap(self.blurb_text, 150))
        self.fig.canvas.draw_idle()

    @staticmethod
    def _wrap(text, width):
        words, lines, cur = text.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > width:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
        return "\n".join(lines)

    def animate(self, _i):
        n = len(self.st["u"])
        self.frame = (self.frame + 2) % n
        x, y, z = self.st["x"][self.frame], self.st["y"][self.frame], self.st["z"][self.frame]
        self.d3.set_data([y], [z])
        self.d3.set_3d_properties([x])
        self.dots2d[0].set_data([y], [x])
        self.dots2d[1].set_data([z], [x])
        self.dots2d[2].set_data([y], [z])
        return []

    def set_dot(self, frac=0.15):
        self.frame = int(frac * (len(self.st["u"]) - 1))
        self.animate(0)
        self.frame -= 2


def pick_preset(arg):
    if arg is None:
        return 0
    if arg.isdigit() and 0 <= int(arg) < len(PRESETS):
        return int(arg)
    for i, (name, _, _) in enumerate(PRESETS):
        if arg.lower() in name.lower():
            return i
    names = "\n".join(f"  {i}: {n}" for i, (n, _, _) in enumerate(PRESETS))
    sys.exit(f"Unknown preset '{arg}'. Available:\n{names}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", help="preset index or name substring")
    ap.add_argument("--save", metavar="FILE", help="save a static figure and exit (no window)")
    args = ap.parse_args()

    viz = Visualizer(pick_preset(args.preset))
    if args.save:
        viz.set_dot()
        viz.fig.savefig(args.save, dpi=150)
        print(f"Saved {args.save}")
        return
    viz.anim = FuncAnimation(viz.fig, viz.animate, interval=30, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
