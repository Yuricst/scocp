"""Fixed-time 3-DoF Mars rocket landing benchmark."""

from dataclasses import dataclass

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


@dataclass
class Mars3DoFRocketLandingResult:
    """Solved fixed-time Mars rocket landing trajectory."""

    times: np.ndarray
    r: np.ndarray
    v: np.ndarray
    z: np.ndarray
    u: np.ndarray
    xi: np.ndarray
    cost: float
    mass: np.ndarray
    thrust: np.ndarray
    thrust_norm: np.ndarray
    pointing_angle: np.ndarray
    status: str


class FixedTimeMars3DoFRocketLanding:
    """Fixed-time 3-DoF Mars landing problem matching the SCPToolbox benchmark."""

    def __init__(
        self,
        times: np.ndarray | None = None,
        *,
        latitude_deg: float = 30.0,
        gravity_mars: float = 3.7114,
        sidereal_day_s: float = 24.6229 * 3600.0,
        m_dry: float = 1505.0,
        m_wet: float = 1905.0,
        isp_s: float = 225.0,
        n_engines: int = 6,
        cant_angle_deg: float = 27.0,
        thrust_max_single_n: float = 3.1e3,
        thrust_min_ratio: float = 0.3,
        thrust_max_ratio: float = 0.8,
        glide_slope_deg: float = 86.0,
        pointing_angle_deg: float = 40.0,
        vmax_mps: float = 500.0 * 1e3 / 3600.0,
        r0_m: np.ndarray | None = None,
        v0_mps: np.ndarray | None = None,
        solver=cp.CLARABEL,
        verbose_solver: bool = False,
    ):
        self.times = (
            np.linspace(0.0, 75.0, 76) if times is None else np.asarray(times, dtype=float)
        )
        if self.times.ndim != 1 or len(self.times) < 2:
            raise ValueError("times must be a 1D array with at least two nodes")
        if np.any(np.diff(self.times) <= 0.0):
            raise ValueError("times must be strictly increasing")

        self.solver = solver
        self.verbose_solver = verbose_solver

        e_x = np.array([1.0, 0.0, 0.0])
        e_y = np.array([0.0, 1.0, 0.0])
        e_z = np.array([0.0, 0.0, 1.0])

        self.g = -gravity_mars * e_z
        theta = np.deg2rad(latitude_deg)
        self.omega = (2.0 * np.pi / sidereal_day_s) * (
            e_x * np.cos(theta) + e_y * 0.0 + e_z * np.sin(theta)
        )

        self.m_dry = m_dry
        self.m_wet = m_wet
        self.isp = isp_s
        self.n_engines = n_engines
        self.phi = np.deg2rad(cant_angle_deg)
        self.gamma_gs = np.deg2rad(glide_slope_deg)
        self.gamma_p = np.deg2rad(pointing_angle_deg)
        self.v_max = vmax_mps

        thrust_min_single = thrust_min_ratio * thrust_max_single_n
        thrust_max_single = thrust_max_ratio * thrust_max_single_n
        self.rho_min = n_engines * thrust_min_single * np.cos(self.phi)
        self.rho_max = n_engines * thrust_max_single * np.cos(self.phi)

        g0 = 9.807
        self.alpha = 1.0 / (self.isp * g0 * np.cos(self.phi))

        self.r0 = (
            np.array([2.0e3, 0.0, 1.5e3], dtype=float)
            if r0_m is None
            else np.asarray(r0_m, dtype=float)
        )
        self.v0 = (
            np.array([80.0, 30.0, -75.0], dtype=float)
            if v0_mps is None
            else np.asarray(v0_mps, dtype=float)
        )
        self.rf = np.zeros(3)
        self.vf = np.zeros(3)

        omega_x = self._skew(self.omega)
        self.A_c = np.block(
            [
                [np.zeros((3, 3)), np.eye(3), np.zeros((3, 1))],
                [-(omega_x @ omega_x), -2.0 * omega_x, np.zeros((3, 1))],
                [np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 1))],
            ]
        )
        self.B_c = np.block(
            [
                [np.zeros((3, 4))],
                [np.eye(3), np.zeros((3, 1))],
                [np.zeros((1, 3)), np.array([[-self.alpha]])],
            ]
        )
        self.p_c = np.concatenate((np.zeros(3), self.g, np.zeros(1)))
        self.nx = 7
        self.nu = 4
        self.status = "not_solved"

        self._H_gs = np.array(
            [
                [np.cos(self.gamma_gs), 0.0, -np.sin(self.gamma_gs)],
                [-np.cos(self.gamma_gs), 0.0, -np.sin(self.gamma_gs)],
                [0.0, np.cos(self.gamma_gs), -np.sin(self.gamma_gs)],
                [0.0, -np.cos(self.gamma_gs), -np.sin(self.gamma_gs)],
            ]
        )
        self._scale_data = self._build_scaling()

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ]
        )

    def _build_scaling(self) -> dict[str, np.ndarray | float]:
        s_r = np.zeros(3)
        S_r = np.diag(np.maximum(1.0, np.abs(self.r0)))
        s_v = np.zeros(3)
        S_v = np.diag(np.maximum(1.0, np.abs(self.v0)))
        s_z = 0.5 * (np.log(self.m_dry) + np.log(self.m_wet))
        S_z = np.log(self.m_wet) - s_z
        s_u = np.array(
            [
                0.0,
                0.0,
                0.5
                * (
                    self.rho_min / self.m_wet * np.cos(self.gamma_p)
                    + self.rho_max / self.m_dry
                ),
            ]
        )
        S_u = np.diag(
            [
                self.rho_max / self.m_dry * np.sin(self.gamma_p),
                self.rho_max / self.m_dry * np.sin(self.gamma_p),
                self.rho_max / self.m_dry - s_u[2],
            ]
        )
        return {
            "s_r": s_r,
            "S_r": S_r,
            "s_v": s_v,
            "S_v": S_v,
            "s_z": s_z,
            "S_z": S_z,
            "s_u": s_u,
            "S_u": S_u,
            "s_xi": s_u[2],
            "S_xi": S_u[2, 2],
        }

    def _discretize(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        aug = np.zeros((self.nx + self.nu + 1, self.nx + self.nu + 1))
        aug[: self.nx, : self.nx] = self.A_c
        aug[: self.nx, self.nx : self.nx + self.nu] = self.B_c
        aug[: self.nx, -1] = self.p_c
        exp_aug = expm(aug * dt)
        A_d = exp_aug[: self.nx, : self.nx]
        B_d = exp_aug[: self.nx, self.nx : self.nx + self.nu]
        p_d = exp_aug[: self.nx, -1]
        return A_d, B_d, p_d

    def _build_result(
        self,
        times: np.ndarray,
        r: np.ndarray,
        v: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        xi: np.ndarray,
        cost: float,
        status: str,
    ) -> Mars3DoFRocketLandingResult:
        mass = np.exp(z)
        if u.shape[0] == mass.shape[0]:
            thrust = u * mass[:, None]
        elif u.shape[0] == mass.shape[0] - 1:
            thrust = u * mass[:-1, None]
        else:
            raise ValueError("control history length is incompatible with mass history length")
        thrust_norm = np.linalg.norm(thrust, axis=1)
        if thrust.shape[0] == 0:
            pointing_angle = np.zeros((0,))
        else:
            cos_gamma = np.divide(
                thrust[:, 2],
                np.maximum(thrust_norm, 1e-12),
            )
            pointing_angle = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
        return Mars3DoFRocketLandingResult(
            times=np.asarray(times, dtype=float).copy(),
            r=r,
            v=v,
            z=z,
            u=u,
            xi=xi,
            cost=float(cost),
            mass=mass,
            thrust=thrust,
            thrust_norm=thrust_norm,
            pointing_angle=pointing_angle,
            status=status,
        )

    def solve(self) -> Mars3DoFRocketLandingResult:
        """Solve the fixed-time convex Mars landing problem."""

        N = len(self.times)
        dts = np.diff(self.times)
        scales = self._scale_data

        r_s = cp.Variable((3, N), name="r_s")
        v_s = cp.Variable((3, N), name="v_s")
        z_s = cp.Variable(N, name="z_s")
        u_s = cp.Variable((3, N - 1), name="u_s")
        xi_s = cp.Variable(N - 1, name="xi_s")

        r = scales["S_r"] @ r_s + scales["s_r"][:, None]
        v = scales["S_v"] @ v_s + scales["s_v"][:, None]
        z = scales["S_z"] * z_s + scales["s_z"]
        u = scales["S_u"] @ u_s + scales["s_u"][:, None]
        xi = scales["S_xi"] * xi_s + scales["s_xi"]

        constraints = [r[:, 0] == self.r0, v[:, 0] == self.v0, z[0] == np.log(self.m_wet)]
        constraints += [r[:, -1] == self.rf, v[:, -1] == self.vf, z[-1] >= np.log(self.m_dry)]

        for k, dt in enumerate(dts):
            A_d, B_d, p_d = self._discretize(float(dt))
            x_k = cp.hstack([r[:, k], v[:, k], z[k]])
            x_kp1 = cp.hstack([r[:, k + 1], v[:, k + 1], z[k + 1]])
            u_k = cp.hstack([u[:, k], xi[k]])
            constraints.append(x_kp1 == A_d @ x_k + B_d @ u_k + p_d)

            z0_k = np.log(self.m_wet - self.alpha * self.rho_max * self.times[k])
            delta_z = z[k] - z0_k
            mu_min = self.rho_min * np.exp(-z0_k)
            mu_max = self.rho_max * np.exp(-z0_k)
            constraints.append(mu_min * (1.0 - delta_z + 0.5 * cp.square(delta_z)) <= xi[k])
            constraints.append(xi[k] <= mu_max * (1.0 - delta_z))

            constraints.append(cp.SOC(xi[k], u[:, k]))
            constraints.append(u[2, k] >= xi[k] * np.cos(self.gamma_p))

        for k, t_k in enumerate(self.times):
            z0_k = np.log(self.m_wet - self.alpha * self.rho_max * t_k)
            z1_k = np.log(self.m_wet - self.alpha * self.rho_min * t_k)
            constraints.append(z0_k <= z[k])
            constraints.append(z[k] <= z1_k)
            constraints.append(self._H_gs @ r[:, k] <= np.zeros(4))
            constraints.append(cp.SOC(self.v_max, v[:, k]))

        objective = cp.Minimize(cp.sum(cp.multiply(dts, xi)))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=self.solver, verbose=self.verbose_solver)

        self.status = problem.status
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"rocket landing solve failed with status {problem.status}")

        return self._build_result(
            times=self.times,
            r=np.asarray(r.value.T),
            v=np.asarray(v.value.T),
            z=np.asarray(z.value),
            u=np.asarray(u.value.T),
            xi=np.asarray(xi.value),
            cost=problem.value,
            status=problem.status,
        )

    def simulate(
        self,
        solution: Mars3DoFRocketLandingResult,
        dt: float = 1e-2,
    ) -> Mars3DoFRocketLandingResult:
        """Simulate the continuous dynamics with ZOH controls."""

        tf = float(solution.times[-1])
        t_eval = np.linspace(0.0, tf, int(round(tf / dt)) + 1)
        x0 = np.concatenate((self.r0, self.v0, [np.log(self.m_wet)]))

        def control_at(t: float) -> np.ndarray:
            idx = np.searchsorted(solution.times[1:], t, side="right")
            idx = min(idx, len(solution.xi) - 1)
            return np.concatenate((solution.u[idx], [solution.xi[idx]]))

        def rhs(t: float, x: np.ndarray) -> np.ndarray:
            return self.A_c @ x + self.B_c @ control_at(t) + self.p_c

        sol = solve_ivp(rhs, (0.0, tf), x0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
        x_hist = sol.y.T
        z = x_hist[:, 6]
        control_hist = np.vstack([control_at(t) for t in sol.t])
        return self._build_result(
            times=sol.t,
            r=x_hist[:, 0:3],
            v=x_hist[:, 3:6],
            z=z,
            u=control_hist[:, 0:3],
            xi=control_hist[:, 3],
            cost=0.0,
            status="simulated",
        )

    def _control_plot_series(
        self,
        solution: Mars3DoFRocketLandingResult,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return control-panel series in consistent acceleration units."""

        times = solution.times[:-1]
        control = solution.xi
        z0 = np.log(self.m_wet - self.alpha * self.rho_max * times)
        delta_z = solution.z[:-1] - z0
        min_accel = self.rho_min * np.exp(-z0) * (1.0 - delta_z + 0.5 * delta_z**2)
        max_accel = self.rho_max * np.exp(-z0) * (1.0 - delta_z)
        return times, control, min_accel, max_accel

    def plot_summary(
        self,
        solution: Mars3DoFRocketLandingResult,
        sim: Mars3DoFRocketLandingResult | None = None,
        path: str | None = None,
    ):
        """Create a compact 2x3 summary figure aligned with the SCPToolbox example."""

        traj = sim if sim is not None else solution
        fig = plt.figure(figsize=(12, 7))

        thrust_scale = 1e-3
        mass_scale = 1e-3
        speed_scale = 3.6
        range_scale = 1e-3

        thrust_dt = solution.thrust_norm * thrust_scale
        thrust_ct = traj.thrust_norm * thrust_scale
        min_thrust = self.rho_min * thrust_scale
        max_thrust = self.rho_max * thrust_scale
        mass_dt = solution.mass * mass_scale
        mass_ct = traj.mass * mass_scale
        velocity_dt = np.linalg.norm(solution.v, axis=1) * speed_scale
        velocity_ct = np.linalg.norm(traj.v, axis=1) * speed_scale
        angle_dt = np.rad2deg(solution.pointing_angle)
        angle_ct = np.rad2deg(traj.pointing_angle)
        downrange_dt = solution.r[:, 0] * range_scale
        downrange_ct = traj.r[:, 0] * range_scale
        crossrange_dt = solution.r[:, 1] * range_scale
        crossrange_ct = traj.r[:, 1] * range_scale
        altitude_dt = solution.r[:, 2] * range_scale
        altitude_ct = traj.r[:, 2] * range_scale

        ax_thrust = fig.add_subplot(2, 3, 1)
        ax_thrust.grid(True, alpha=0.5)
        ax_thrust.axhline(min_thrust, color="grey", linestyle="--", label="Min thrust")
        ax_thrust.axhline(max_thrust, color="r", linestyle=":", label="Max thrust")
        ax_thrust.plot(traj.times[: len(thrust_ct)], thrust_ct, color="b", lw=1.5, label="T(t)")
        ax_thrust.plot(
            solution.times[: len(thrust_dt)],
            thrust_dt,
            linestyle="none",
            marker="o",
            markersize=2.5,
            markerfacecolor="k",
            markeredgecolor="white",
            markeredgewidth=0.2,
            label="T_k",
        )
        ax_thrust.set(xlabel="Time [s]", ylabel="Thrust [kN]")
        ax_thrust.legend()

        ax_mass = fig.add_subplot(2, 3, 2)
        ax_mass.grid(True, alpha=0.5)
        ax_mass.axhline(self.m_dry * mass_scale, color="r", linestyle="--")
        ax_mass.axhline(self.m_wet * mass_scale, color="grey", linestyle=":")
        ax_mass.plot(traj.times, mass_ct, color="b", lw=1.5, label="m(t)")
        ax_mass.plot(
            solution.times,
            mass_dt,
            linestyle="none",
            marker="o",
            markersize=2.5,
            markerfacecolor="k",
            markeredgecolor="white",
            markeredgewidth=0.2,
            label="m_k",
        )
        ax_mass.text(
            traj.times[0],
            solution.mass[-1] * mass_scale + 0.005,
            f"m_f = {solution.mass[-1] * mass_scale:.3f} t",
            color="r",
        )
        ax_mass.set(xlabel="Time [s]", ylabel="Mass [t]")
        ax_mass.legend()

        ax_vel = fig.add_subplot(2, 3, 3)
        ax_vel.grid(True, alpha=0.5)
        ax_vel.axhline(self.v_max * speed_scale, color="r", linestyle="--", label="v max")
        ax_vel.plot(traj.times, velocity_ct, color="b", lw=1.5, label="|v(t)|")
        ax_vel.plot(
            solution.times,
            velocity_dt,
            linestyle="none",
            marker="o",
            markersize=2.5,
            markerfacecolor="k",
            markeredgecolor="white",
            markeredgewidth=0.2,
            label="|v_k|",
        )
        ax_vel.set(xlabel="Time [s]", ylabel="Speed [km/h]")
        ax_vel.legend()

        ax_angle = fig.add_subplot(2, 3, 4)
        ax_angle.grid(True, alpha=0.5)
        ax_angle.axhline(
            np.rad2deg(self.gamma_p),
            color="r",
            linestyle="--",
            label="angle max",
        )
        ax_angle.plot(traj.times[: len(angle_ct)], angle_ct, color="b", lw=1.5, label="theta(t)")
        ax_angle.plot(
            solution.times[: len(angle_dt)],
            angle_dt,
            linestyle="none",
            marker="o",
            markersize=2.5,
            markerfacecolor="k",
            markeredgecolor="white",
            markeredgewidth=0.2,
            label="theta_k",
        )
        ax_angle.set(xlabel="Time [s]", ylabel="Angle [deg]")
        ax_angle.legend()

        ax_down = fig.add_subplot(2, 3, 5)
        ax_down.grid(True, alpha=0.5)
        ax_down.plot(downrange_ct, altitude_ct, color="b", lw=1.5, label="r(t)")
        ax_down.plot(
            downrange_dt,
            altitude_dt,
            linestyle="none",
            marker="o",
            markersize=3,
            markerfacecolor="k",
            markeredgecolor="white",
            markeredgewidth=0.2,
            label="r_k",
        )
        ax_down.set(xlabel="Downrange [km]", ylabel="Altitude [km]")
        ax_down.legend()

        ax_cross = fig.add_subplot(2, 3, 6)
        ax_cross.grid(True, alpha=0.5)
        ax_cross.plot(crossrange_ct, altitude_ct, color="b", lw=1.5, label="r(t)")
        ax_cross.plot(
            crossrange_dt,
            altitude_dt,
            linestyle="none",
            marker="o",
            markersize=3,
            markerfacecolor="k",
            markeredgecolor="white",
            markeredgewidth=0.2,
            label="r_k",
        )
        ax_cross.set(xlabel="Crossrange [km]", ylabel="Altitude [km]")
        ax_cross.legend()

        plt.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=300)
        return fig
