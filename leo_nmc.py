import numpy as np
import matplotlib.pyplot as plt
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody, orbitalMotion
from Basilisk.simulation import spacecraft

from Basilisk.utilities.supportDataTools.dataFetcher import get_path, DataFile

# ---------------- target orbit (LEO, circular) ----------------
R_E   = 6378.137e3
ALT   = 500e3
A_T   = R_E + ALT
I_T   = np.deg2rad(97.4)
OMEGA_T, U_T = 0.0, 0.0        # RAAN and mean arg. of latitude at t=0 (e=0, so omega_t=0, f_t=u_t)
DT    = 10.0                   # s


# ---------------- ROE -> chaser classical elements ----------------
def roe_to_chaser_elements(roe):
    """roe = [da, dlam, dex, dey, dix, diy], all dimensionless (rad). First-order, circular target."""
    da, dlam, dex, dey, dix, diy = roe
    oe = orbitalMotion.ClassicElements()
    oe.a = A_T * (1.0 + da)
    oe.e = float(np.hypot(dex, dey))                       # target e = 0
    oe.omega = float(np.arctan2(dey, dex)) if oe.e > 0 else 0.0
    oe.i = I_T + dix
    dOmega = diy / np.sin(I_T)
    oe.Omega = OMEGA_T + dOmega
    u = U_T + dlam - dOmega * np.cos(I_T)                  # from dlam = du + dOmega*cos(i)
    M = np.mod(u - oe.omega, 2 * np.pi)
    E = orbitalMotion.M2E(M, oe.e)
    oe.f = orbitalMotion.E2f(E, oe.e)
    return oe


def make_roe(A=10e3, B=0.0, phi_deg=0.0, theta_deg=0.0, dlam_m=0.0, drift_per_orbit=0.0):
    """
    A: radial half-amplitude a*|de| (m).   Ellipse is 2A long in-track.
    B: cross-track amplitude a*|di| (m).
    phi_deg / theta_deg: phase angles of the de and di vectors.
    dlam_m: along-track offset of ellipse center (m).
    drift_per_orbit: along-track drift of the center per orbit (m), + is forward. (Tong Eq. 4)
    """
    phi, th = np.deg2rad(phi_deg), np.deg2rad(theta_deg)
    da   = -drift_per_orbit / (3 * np.pi * A_T)            # relative da
    dlam = dlam_m / A_T
    dex, dey = (A / A_T) * np.cos(phi), (A / A_T) * np.sin(phi)
    dix, diy = (B / A_T) * np.cos(th),  (B / A_T) * np.sin(th)
    return np.array([da, dlam, dex, dey, dix, diy])


# ---------------- Basilisk propagation ----------------
def propagate(roe, hours):
    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("proc")
    proc.addTask(sim.CreateNewTask("task", macros.sec2nano(DT)))

    grav = simIncludeGravBody.gravBodyFactory()
    earth = grav.createEarth()
    earth.isCentralBody = True


    USE_J2 = False   # or make this a parameter of propagate()
    if USE_J2:
        earth.useSphericalHarmParams = True
        simIncludeGravBody.loadGravFromFile(
            get_path(DataFile.LocalGravData.GGM03S_J2_only),
            earth.spherHarm,
            2,
        )

    mu = earth.mu

    def make_sc(name, oe):
        sc = spacecraft.Spacecraft()
        sc.ModelTag = name
        sc.gravField.gravBodies = spacecraft.GravBodyVector(list(grav.gravBodies.values()))
        rN, vN = orbitalMotion.elem2rv(mu, oe)
        sc.hub.r_CN_NInit = rN
        sc.hub.v_CN_NInit = vN
        sim.AddModelToTask("task", sc)
        rec = sc.scStateOutMsg.recorder()
        sim.AddModelToTask("task", rec)
        return rec

    oe_t = orbitalMotion.ClassicElements()
    oe_t.a, oe_t.e, oe_t.i = A_T, 0.0, I_T
    oe_t.Omega, oe_t.omega, oe_t.f = OMEGA_T, 0.0, U_T

    rec_t = make_sc("target", oe_t)
    rec_c = make_sc("chaser", roe_to_chaser_elements(roe))

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.hour2nano(hours))
    sim.ExecuteSimulation()

    rt, vt = np.array(rec_t.r_BN_N), np.array(rec_t.v_BN_N)
    rc = np.array(rec_c.r_BN_N)

    # inertial -> target RIC (x = radial, y = in-track, z = cross-track)
    r_hat = rt / np.linalg.norm(rt, axis=1)[:, None]
    h = np.cross(rt, vt)
    c_hat = h / np.linalg.norm(h, axis=1)[:, None]
    i_hat = np.cross(c_hat, r_hat)
    dr = rc - rt
    rel = np.column_stack([np.sum(dr * r_hat, 1), np.sum(dr * i_hat, 1), np.sum(dr * c_hat, 1)])

    t_hr = np.array(rec_t.times()) * macros.NANO2SEC / 3600.0
    return t_hr, rel


def period_hr():
    mu = 3.986004415e14
    return 2 * np.pi * np.sqrt(A_T**3 / mu) / 3600.0


# ---------------- Case 1: baseline in-plane NMC ----------------
def case_baseline():
    t, rel = propagate(make_roe(A=10e3), hours=6.0)
    km = rel / 1e3
    print("\n[baseline] period = %.2f min" % (period_hr() * 60))
    print("  radial   min/max (km):", km[:, 0].min(), km[:, 0].max(), " (expect +-10)")
    print("  in-track min/max (km):", km[:, 1].min(), km[:, 1].max(), " (expect +-20)")
    print("  cross    min/max (km):", km[:, 2].min(), km[:, 2].max(), " (expect ~0)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(km[:, 1], km[:, 0]); ax[0].plot(0, 0, "r*", ms=12)
    ax[0].set_xlabel("In-track (km)"); ax[0].set_ylabel("Radial (km)")
    ax[0].set_title("Baseline NMC (RIC)"); ax[0].axis("equal"); ax[0].grid(True)
    ax[1].plot(t, km[:, 0], label="radial"); ax[1].plot(t, km[:, 1], label="in-track")
    ax[1].set_xlabel("Time (hr)"); ax[1].set_ylabel("km"); ax[1].legend(); ax[1].grid(True)
    plt.tight_layout(); plt.savefig("case1_baseline.png", dpi=150)


# ---------------- Case 2: add cross-track ----------------
def case_crosstrack():
    # de and di parallel (phi = theta = 0) -> passively safe geometry
    t, rel = propagate(make_roe(A=10e3, B=5e3, phi_deg=0, theta_deg=0), hours=6.0)
    km = rel / 1e3
    print("\n[cross-track]")
    print("  radial   min/max (km):", km[:, 0].min(), km[:, 0].max(), " (expect +-10)")
    print("  in-track min/max (km):", km[:, 1].min(), km[:, 1].max(), " (expect +-20)")
    print("  cross    min/max (km):", km[:, 2].min(), km[:, 2].max(), " (expect +-5)")
    print("  min 3D distance (km):", np.linalg.norm(km, axis=1).min())

    fig = plt.figure(figsize=(14, 4.5))
    ax = fig.add_subplot(1, 4, 1); ax.plot(km[:, 1], km[:, 0]); ax.plot(0, 0, "r*")
    ax.set_xlabel("In-track"); ax.set_ylabel("Radial"); ax.set_title("R-I"); ax.axis("equal"); ax.grid(True)
    ax = fig.add_subplot(1, 4, 2); ax.plot(km[:, 2], km[:, 0]); ax.plot(0, 0, "r*")
    ax.set_xlabel("Cross-track"); ax.set_ylabel("Radial"); ax.set_title("R-C"); ax.axis("equal"); ax.grid(True)
    ax = fig.add_subplot(1, 4, 3); ax.plot(km[:, 1], km[:, 2]); ax.plot(0, 0, "r*")
    ax.set_xlabel("In-track"); ax.set_ylabel("Cross-track"); ax.set_title("I-C"); ax.axis("equal"); ax.grid(True)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    ax.plot(km[:, 1], km[:, 2], km[:, 0]); ax.scatter(0, 0, 0, c="r")
    ax.set_xlabel("In-track"); ax.set_ylabel("Cross"); ax.set_zlabel("Radial"); ax.set_title("3D")
    plt.tight_layout(); plt.savefig("case2_crosstrack.png", dpi=150)


# ---------------- Case 3: passive safety under along-track drift ----------------
def case_safety():
    A, B = 10e3, 5e3
    bound_km = min(A, B) / 1e3
    # Failure scenario: ellipse center starts 24 km behind the target and drifts forward
    # 300 m/orbit, sweeping across the target. Run ~2.5 days.
    common = dict(A=A, B=B, phi_deg=0, dlam_m=-24e3, drift_per_orbit=300.0)
    cases = {
        "parallel (theta = phi)":        dict(theta_deg=0),
        "antiparallel (theta = phi+180)": dict(theta_deg=180),
        "perpendicular (theta = phi+90)": dict(theta_deg=90),
    }
    results = {}
    print("\n[safety] bound min(A,B) = %.1f km" % bound_km)
    for name, kw in cases.items():
        t, rel = propagate(make_roe(**common, **kw), hours=60.0)
        km = rel / 1e3
        d3 = np.linalg.norm(km, axis=1)
        dxz = np.hypot(km[:, 0], km[:, 2])
        results[name] = (t, km, d3)
        print("  %-32s min 3D = %6.2f km | min x-z = %6.2f km" % (name, d3.min(), dxz.min()))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, (t, km, d3) in results.items():
        ax[0].plot(t, d3, label=name)
        ax[1].plot(km[:, 2], km[:, 0], lw=0.6, label=name)   # x-z projection
    ax[0].axhline(bound_km, color="k", ls="--", label="min(A,B) bound")
    ax[0].set_xlabel("Time (hr)"); ax[0].set_ylabel("3D distance to target (km)")
    ax[0].set_title("Separation under along-track drift"); ax[0].legend(fontsize=8); ax[0].grid(True)
    ax[1].plot(0, 0, "r*", ms=12)
    ax[1].set_xlabel("Cross-track z (km)"); ax[1].set_ylabel("Radial x (km)")
    ax[1].set_title("x-z projection"); ax[1].axis("equal"); ax[1].legend(fontsize=8); ax[1].grid(True)
    plt.tight_layout(); plt.savefig("case3_safety.png", dpi=150)


if __name__ == "__main__":
    case_baseline()
    case_crosstrack()
    case_safety()
    plt.show()