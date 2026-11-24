from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import macros, simIncludeGravBody
import numpy as np
from itertools import combinations


class RelOrbitMaintenance(sysModel.SysModel):
    def __init__(self, chief_oe, m, dt, window_length=2, guidance=False, roe_des=None):
        super().__init__()
        if guidance:
            self.desROEInMsg = messaging.THRArrayCmdForceMsgReader() 
        else:
            self.desROEInMsg = None
            self.roe_des = roe_des if roe_des is not None else ValueError("No Desired ROEs provided!")
        #^^a hacky way of getting desired ROEs as a list; the thrForce item in the message is a list of the des ROEs
        self.chaserTransInMsg = messaging.SCStatesMsgReader()
        self.chiefTransInMsg = messaging.SCStatesMsgReader()
        self.cmdForceOutMsg = messaging.CmdForceInertialMsg()
        self.chief_oe = np.array(chief_oe[:-1]) #a, e, i
        self.u0 = np.array(chief_oe)[-1] #argument of latitude


        #find chief orbital period assuming earth central body
        gravity = simIncludeGravBody.gravBodyFactory()
        earth = gravity.createEarth()
        self.MU = earth.mu
        self.R_E = earth.radEquator
        self.J2 = 1.082626683e-3
        self.period = 2 * np.pi * np.sqrt(self.chief_oe[0]**3 / self.MU) #s

        #initialize some variables used for windowing
        self.window_start = 0
        self.window_length = window_length # units: chief orbital periods
        self.burn_times = np.array([])
        self.burns = np.array([])
        self.next_plan = 0
        self.m = m
        self.dt = dt

    def UpdateState(self, CurrentSimNanos):
        if CurrentSimNanos >= self.next_plan:
            opt = self.optimal_control()
            self.window_start = CurrentSimNanos
            self.next_plan = CurrentSimNanos + 2 * np.pi * self.window_length / opt["params"]["u_dot"] * 1e9
        forceOutMsgBuffer = messaging.CmdForceInertialMsgPayload()
        if len(self.burn_times) and CurrentSimNanos / 1e9 >= self.burn_times[0] + self.window_start / 1e9:
            dv_rtn = self.burns[0, :]
            chiefTrans = self.chiefTransInMsg()
            r = np.array(chiefTrans.r_BN_N)
            v = np.array(chiefTrans.v_BN_N)
            R = r / np.linalg.norm(r); N = np.cross(r, v); N /= np.linalg.norm(N); T = np.cross(N, R)
            burn = self.m * (dv_rtn[0]*R + dv_rtn[1]*T + dv_rtn[2]*N) / self.dt
            forceOutMsgBuffer.forceRequestInertial = np.array(burn)
            self.burns = self.burns[1:, :]
            self.burn_times = self.burn_times[1:]
        else:
            forceOutMsgBuffer.forceRequestInertial = np.zeros((3))
        self.cmdForceOutMsg.write(forceOutMsgBuffer, CurrentSimNanos)

    def j2_params(self, a, e, i):
        """Secular J2 coefficients used by the near-circular STM (Eqs. 6, 9, 28)."""
        n = np.sqrt(self.MU / a**3)
        eta = np.sqrt(1.0 - e**2)
        kappa = 0.75 * self.J2 * self.R_E**2 * np.sqrt(self.MU) / (a**3.5 * eta**4)
        P = 3 * np.cos(i)**2 - 1
        Q = 5 * np.cos(i)**2 - 1
        S = np.sin(2 * i)
        T = np.sin(i)**2
        u_dot = n + kappa * (eta * P + Q)                   # mean arg. of latitude rate
        return dict(
            a=a, n=n, eta=eta, kappa=kappa, P=P, Q=Q, S=S, T=T, u_dot=u_dot,
            c=kappa * Q / u_dot,                             # omega_dot / u_dot
            dlam_da=-(1.5 * n + 3.5 * kappa * (1 + eta) * P),  # d(dlambda)/dt per da
            dlam_dix=-kappa * (4 + 3 * eta) * S,               # d(dlambda)/dt per dix
            diy_da=3.5 * kappa * S,                            # d(diy)/dt per da
            diy_dix=2 * kappa * T,                             # d(diy)/dt per dix
        )


    def stm(tau, p):
        """J2 STM for mean quasi-nonsingular ROE, near-circular chief (Eq. 8)."""
        w = p["kappa"] * p["Q"] * tau
        Phi = np.eye(6)
        Phi[1, 0] = p["dlam_da"] * tau
        Phi[1, 4] = p["dlam_dix"] * tau
        Phi[2:4, 2:4] = [[np.cos(w), -np.sin(w)], [np.sin(w), np.cos(w)]]
        Phi[5, 0] = p["diy_da"] * tau
        Phi[5, 4] = p["diy_dix"] * tau
        return Phi


    def control_matrix(u, p):
        """Impulsive control input matrix for near-circular orbits (Eq. 10)."""
        s, c = np.sin(u), np.cos(u)
        return np.array([[0, 2, 0],
                        [-2, 0, 0],
                        [s, 2 * c, 0],
                        [-c, 2 * s, 0],
                        [0, 0, c],
                        [0, 0, s]]) / (p["n"] * p["a"])


    # ---------------------------------------------------------------------------
    # Out-of-plane: single cross-track burn (Eqs. 23-26)
    # ---------------------------------------------------------------------------
    def _oop_maneuver(d_ix, d_iy, u0, uf, p, tol=1e-13, max_iter=20):
        na = p["n"] * p["a"]
        if np.hypot(d_ix, d_iy) == 0.0:
            return None
        K = p["diy_dix"] / p["u_dot"]                       # 2 kappa T / u_dot

        def g(u):                                           # d_iy per unit d_ix-burn
            return np.sin(u) + K * (uf - u) * np.cos(u)

        def f(u):                                           # Eq. 26 * cos(u) * d_ix
            return d_ix * g(u) - d_iy * np.cos(u)

        def fp(u):
            return (d_ix * (np.cos(u) - K * np.cos(u) - K * (uf - u) * np.sin(u))
                    + d_iy * np.sin(u))

        u_ideal = np.arctan2(d_iy, d_ix)                    # Eq. 23
        m0 = int(np.ceil((u0 - u_ideal) / np.pi)) - 1
        best = None
        for m in range(m0, m0 + int((uf - u0) / np.pi) + 3):
            u = u_ideal + m * np.pi
            for _ in range(max_iter):                       # Newton on Eq. 26
                du = f(u) / fp(u)
                u -= du
                if abs(du) < tol:
                    break
            if not (u0 <= u <= uf):
                continue
            cu, gu = np.cos(u), g(u)
            dvN = na * (d_ix * cu + d_iy * gu) / (cu**2 + gu**2)   # Eq. 25, robust form
            if best is None or abs(dvN) < abs(best[1]) - 1e-12:
                best = (u, dvN)
        if best is None:
            raise ValueError("No admissible cross-track burn location in the window; "
                            "the control window must span at least half an orbit.")
        return best


    # ---------------------------------------------------------------------------
    # In-plane: triplet of along-track burns (Eqs. 27-32, Tables 3-4)
    # ---------------------------------------------------------------------------
    def _ip_locations(d_ex, d_ey, u0, uf, p):
        c = p["c"]
        U_bar = np.arctan2(d_ey, d_ex) if np.hypot(d_ex, d_ey) > 0 else 0.0   # Eq. 32
        k_min = int(np.ceil(((1 - c) * u0 + c * uf - U_bar) / np.pi))
        k_max = int(np.floor(((1 - c) * uf + c * uf - U_bar) / np.pi))
        ks = np.arange(k_min, k_max + 1)
        us = (U_bar + ks * np.pi - c * uf) / (1 - c)                          # Eq. 31
        return ks, us


    def _ip_solve(ks, us, d_a, d_lam, d_e_norm, uf, p):
        """Solve Eq. 27 for given burn locations (collinear de increments)."""
        na = p["n"] * p["a"]
        D = p["dlam_da"] / p["u_dot"]                # d(dlambda)/du per unit da
        sgn = np.where(np.asarray(ks) % 2 == 0, 1.0, -1.0)
        A = np.vstack([2 * np.ones_like(us),
                    2 * D * (uf - np.asarray(us)),
                    2 * sgn])
        return np.linalg.solve(A, na * np.array([d_a, d_lam, d_e_norm]))


    def _ip_maneuvers(d_a, d_lam, d_ex, d_ey, u0, uf, p, k=None):
        d_e = np.hypot(d_ex, d_ey)
        ks, us = RelOrbitMaintenance._ip_locations(d_ex, d_ey, u0, uf, p)
        if k is not None:
            k = np.asarray(k)
            sel = [np.where(ks == kk)[0] for kk in k]
            if any(len(s) == 0 for s in sel):
                raise ValueError(f"Requested k={list(k)} not all inside window; "
                                f"admissible k: {list(ks)}")
            idx = [s[0] for s in sel]
            return us[idx], RelOrbitMaintenance._ip_solve(ks[idx], us[idx], d_a, d_lam, d_e, uf, p), ks[idx]

        best = None
        for idx in combinations(range(len(ks)), 3):
            kk = ks[list(idx)]
            if np.all(kk % 2 == kk[0] % 2):          # same sign -> singular system
                continue
            uu = us[list(idx)]
            dv = RelOrbitMaintenance._ip_solve(kk, uu, d_a, d_lam, d_e, uf, p)
            cost = np.sum(np.abs(dv))
            if best is None or cost < best[0] * (1 - 1e-9):
                best = (cost, uu, dv, kk)
        if best is None:
            raise ValueError("Window too short for an in-plane triplet; it must span "
                            "at least ~1 orbit (two opposite-sign burn slots + one more).")
        return best[1], best[2], best[3]

    def optimal_control(self, coupling_iter=5):
        """
        Closed-form impulsive reconfiguration in J2-perturbed near-circular orbits.

        Parameters
        ----------
        roe_true : (6,) current mean ROE [da, dlambda, dex, dey, dix, diy] (dimensionless)
        roe_des  : (6,) desired mean ROE at the end of the control window
        a, i     : chief mean semi-major axis [m] and inclination [rad]
        u0       : chief mean argument of latitude at window start [rad]
        du       : control window length in mean argument of latitude [rad]
        e        : chief eccentricity (near zero; only enters kappa and rates)
        coupling_iter : passes to fold in-plane/out-of-plane J2 couplings

        Returns
        -------
        dict with
        'maneuvers' : list of dicts sorted in time, each with
                        u [rad, unwrapped], dt [s from t0], dv_rtn [m/s] (3,)
        'dv_total'  : sum of burn magnitudes [m/s]
        'dv_lb_inplane', 'dv_oop_ideal' : delta-v lower bounds [m/s]
        'ecc_dominant' : True if the paper's dominance condition holds
                        (||D_de|| >= |D_da| and ||D_de|| >= |D_dlambda|/(1.5*du))
        'ip_cost_over_lb' : in-plane dv / in-plane lower bound (1.0 = optimal)
        'droe_target'  : pre-compensated desired ROE change
        """
        roe_true, self.chief_oe, self.u0 = self.rv_to_roe()
        a, e, i = self.chief_oe
        du = 2 * np.pi * self.window_length
        
        if self.desROEInMsg is not None:
            roe_des = np.array(self.desROEInMsg().thrForce)[:6]
        else:
            roe_des = self.roe_des
        p = self.j2_params(a, e, i)
        na, uf = p["n"] * a, self.u0 + du
        tau = lambda u_from: (uf - u_from) / p["u_dot"]

        # 1) Pre-compensation of free motion over the full window (Eq. 15)
        droe = roe_des - RelOrbitMaintenance.stm(du / p["u_dot"], p) @ roe_true
        d_a, d_lam, d_ex, d_ey, d_ix, d_iy = droe

        # 2)+3) with fixed-point folding of the small J2 cross-couplings
        lam_corr, iy_corr = 0.0, 0.0
        for _ in range(max(1, coupling_iter)):
            oop = RelOrbitMaintenance._oop_maneuver(d_ix, d_iy - iy_corr, self.u0, uf, p)
            if oop is not None:
                uN, dvN = oop
                lam_corr = p["dlam_dix"] * tau(uN) * np.cos(uN) * dvN / na
            u_ip, dv_ip, k_used = RelOrbitMaintenance._ip_maneuvers(d_a, d_lam - lam_corr, d_ex, d_ey,
                                                self.u0, uf, p)
            iy_corr = np.sum(p["diy_da"] * tau(u_ip) * 2 * dv_ip / na)

        mans = [dict(u=u, dt=(u - self.u0) / p["u_dot"], dv_rtn=np.array([0.0, dv, 0.0]),
                    k=int(k)) for u, dv, k in zip(u_ip, dv_ip, k_used)]
        if oop is not None:
            mans.append(dict(u=uN, dt=(uN - self.u0) / p["u_dot"],
                            dv_rtn=np.array([0.0, 0.0, dvN])))
        mans.sort(key=lambda m: m["u"])

        # Delta-v lower bound for in-plane control (Eq. 22, J2-consistent drift)
        D = abs(p["dlam_da"] / p["u_dot"])
        lb_ip = na * max(abs(d_a) / 2, abs(d_lam) / max(2.0, 2 * D * du),
                        np.hypot(d_ex, d_ey) / 2)
        ecc_dom = np.hypot(d_ex, d_ey) / 2 >= lb_ip / na * (1 - 1e-12)
        dv_ip_tot = float(np.sum(np.abs(dv_ip)))

        '''
        chiefTrans = self.chiefTransInMsg()
        r = np.array(chiefTrans.r_BN_N)
        v = np.array(chiefTrans.v_BN_N)
        R = r / np.linalg.norm(r); N = np.cross(r, v); N /= np.linalg.norm(N); T = np.cross(N, R)

        
        burns = []
        for i in mans:
            dv_rtn = i['dv_rtn']
            burns.append(self.m * (dv_rtn[0]*R + dv_rtn[1]*T + dv_rtn[2]*N) / self.dt)
        '''
        self.burn_times = np.array([i['dt'] for i in mans])
        self.burns = np.array([i['dv_rtn'] for i in mans])
        return dict(maneuvers=mans,
                    dv_total=float(sum(np.linalg.norm(m["dv_rtn"]) for m in mans)),
                    dv_lb_inplane=float(lb_ip),
                    dv_oop_ideal=float(na * np.hypot(d_ix, d_iy)),
                    ecc_dominant=bool(ecc_dom),
                    ip_cost_over_lb=dv_ip_tot / lb_ip if lb_ip > 0 else 1.0,
                    droe_target=droe, params=p)


    def propagate_linear(roe0, maneuvers, a, i, u0, du, e=0.0):
        """Propagate ROE through the same linear model to verify a maneuver plan."""
        p = RelOrbitMaintenance.j2_params(a, e, i)
        roe, u_prev = np.asarray(roe0, float).copy(), u0
        for m in sorted(maneuvers, key=lambda m: m["u"]):
            roe = RelOrbitMaintenance.stm((m["u"] - u_prev) / p["u_dot"], p) @ roe
            roe = roe + RelOrbitMaintenance.control_matrix(m["u"], p) @ m["dv_rtn"]
            u_prev = m["u"]
        return RelOrbitMaintenance.stm((u0 + du - u_prev) / p["u_dot"], p) @ roe


    def _wrap(x):
        """Wrap angle(s) to (-pi, pi]."""
        return np.pi - np.mod(np.pi - np.asarray(x), 2 * np.pi)
    

    def rv_to_oe(r, v, mu):
        """
        Inertial position/velocity [m, m/s] -> orbital elements.

        Uses the eccentricity-vector components (ex, ey) and the argument of
        latitude so that the quantities the ROE need (a, ex, ey, i, RAAN, u)
        stay well defined for circular orbits. Singular only for i = 0.

        Returns dict: a, e, i, raan, argp, nu (true anom.), M (mean anom.),
        ex = e cos(argp), ey = e sin(argp), theta (true arg. of latitude),
        u (mean arg. of latitude). Angles in rad, wrapped to [0, 2pi).
        """
        r, v = np.asarray(r, float), np.asarray(v, float)
        rn, vn = np.linalg.norm(r), np.linalg.norm(v)
        h = np.cross(r, v)
        hn = np.linalg.norm(h)
        w_hat = h / hn                                      # orbit normal

        a = 1.0 / (2.0 / rn - vn**2 / mu)
        if a <= 0:
            raise ValueError("Orbit is not elliptic (a <= 0).")
        i = np.arccos(np.clip(w_hat[2], -1.0, 1.0))

        node = np.cross([0.0, 0.0, 1.0], h)
        nn = np.linalg.norm(node)
        if nn < 1e-12 * hn:
            raise ValueError("Equatorial orbit: RAAN and these ROE are singular.")
        n_hat = node / nn
        raan = np.arctan2(n_hat[1], n_hat[0])

        # In-plane axes: p along ascending node, q 90 deg ahead in orbit plane
        q_hat = np.cross(w_hat, n_hat)
        e_vec = np.cross(v, h) / mu - r / rn
        ex, ey = e_vec @ n_hat, e_vec @ q_hat               # e cos w, e sin w
        e = np.hypot(ex, ey)
        argp = np.arctan2(ey, ex) if e > 0 else 0.0

        theta = np.arctan2(r @ q_hat, r @ n_hat)            # true arg. of latitude
        nu = theta - argp
        E = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2),
                            np.sqrt(1 + e) * np.cos(nu / 2))
        M = E - e * np.sin(E)
        u = argp + M                                        # mean arg. of latitude

        tw = lambda x: np.mod(x, 2 * np.pi)
        return dict(a=a, e=e, i=i, raan=tw(raan), argp=tw(argp), nu=tw(nu),
                    M=tw(M), ex=ex, ey=ey, theta=tw(theta), u=tw(u))

    def elements_to_roe(oe_c, oe_d):
        """Chief/deputy element dicts (from rv_to_oe) -> ROE per Eq. 4."""
        d_raan = RelOrbitMaintenance._wrap(oe_d["raan"] - oe_c["raan"])
        return np.array([
            (oe_d["a"] - oe_c["a"]) / oe_c["a"],
            RelOrbitMaintenance._wrap(oe_d["u"] - oe_c["u"]) + d_raan * np.cos(oe_c["i"]),
            oe_d["ex"] - oe_c["ex"],
            oe_d["ey"] - oe_c["ey"],
            oe_d["i"] - oe_c["i"],
            d_raan * np.sin(oe_c["i"]),
        ])

    def rv_to_roe(self):
        """
        Chief and deputy (chaser) inertial states -> quasi-nonsingular ROE.

        Returns (roe, oe_chief): roe is the dimensionless 6-vector
        [da, dlambda, dex, dey, dix, diy]; multiply by oe_chief['a'] for metres.
        """
        chaserTrans = self.chaserTransInMsg()
        chiefTrans = self.chiefTransInMsg()
        chaserPos = np.array(chaserTrans.r_BN_N)
        chaserVelo = np.array(chaserTrans.v_BN_N)
        chiefPos = np.array(chiefTrans.r_BN_N)
        chiefVelo = np.array(chiefTrans.v_BN_N)
        oe_c = RelOrbitMaintenance.rv_to_oe(chiefPos, chiefVelo, self.MU)
        oe_d = RelOrbitMaintenance.rv_to_oe(chaserPos, chaserVelo, self.MU) 
        return RelOrbitMaintenance.elements_to_roe(oe_c, oe_d), np.array([oe_c['a'], oe_c['e'], oe_c['i']]), oe_c["u"]
