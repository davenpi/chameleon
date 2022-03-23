import copy
import numpy as np


class Chameleon:
    """
    Chameleon environment
    """

    def __init__(
        self,
        L: float = 1,
        rho: float = 1,
        E: float = 1,
        alpha: float = 1,
        n_elems: int = 100,
        dt: float = 0.01,
    ) -> None:
        self.rho = rho
        self.E = E
        self.a = alpha
        self.L = L
        self.n_elems = n_elems
        self.dt = dt
        self.pos_init = np.linspace(0, self.L, self.n_elems)
        self.dx = self.pos_init[1] - self.pos_init[0]
        self.r = np.ones(self.n_elems)
        self.pos = copy.deepcopy(self.pos_init)
        self.u = self.pos - self.pos_init
        self.displacement_hist = []
        self.displacement_hist.append(self.u)
        self.displacement_hist.append(
            self.u
        )  # trick so we can compute time derivative from start
        self.active_stress_hist = []
        self.active_stress_hist.append(np.zeros(self.n_elems))
        self.time = 0

    def get_velocity(self) -> np.ndarray:
        """Compute v = du_dt."""
        v = (self.displacement_hist[-1] - self.displacement_hist[-2]) / self.dt
        return v

    def get_volume(self) -> float:
        """Compute volume of chameleon. Should be conserved."""
        v = np.trapz(self.r, dx=self.dx)
        return v

    def get_tip_strain(self) -> float:
        """
        Compute strain by solving the boundary condition for the stress.

        We can solve the stress boundary condition analytically and this is the result.

        Parameters
        ----------

        Returns
        -------
        tip_strain : float
            du_dx evaluated at the tip. AKA the strain at the tip.
        """
        gamma = self.E / self.a
        time_grid = np.arange(
            0, self.time + self.dt / 10, self.dt
        )  # want final time included in interval
        tip_active_stress = np.array([stress[-1] for stress in self.active_stress_hist])
        integral = np.trapz(np.exp(gamma * time_grid) * tip_active_stress, dx=self.dt)
        tip_strain = -integral * (np.exp(gamma * self.time)) / (self.a)
        return tip_strain

    def compute_stress(self, active_stress: np.ndarray) -> np.ndarray:
        """Compute the stress from each component and return sum."""
        elastic_stress = self.E * np.gradient(self.u, self.dx)
        du_dt = self.get_velocity()
        viscous_stress = self.a * np.gradient(du_dt, self.dx)
        return elastic_stress + viscous_stress + active_stress

    def update_u(self, active_stress: np.ndarray) -> None:
        """
        Use the dynamical equation to update the displacement u.

        Need to make sure we satisfy the boundary condition that u(0, t) = 0
        and sigma(L, t) = 0. The sigma(L, t) = 0 BC gives me an ODE for the
        strain at the boundary. Solve the ODE and plug in answer when updating
        u.

        Parameters
        ----------
        active_stress : np.ndarray
            active stress

        Returns
        -------
        None
        """
        self.time += self.dt
        self.active_stress_hist.append(active_stress)
        stress = self.compute_stress(active_stress=active_stress)
        v_old = self.get_velocity()
        v_new = v_old + self.dt * (np.gradient(stress * self.r**2, self.dx)) / (
            self.rho * self.r**2
        )
        self.u = self.u + self.dt * v_new
        self.u[0] = 0
        tip_strain = self.get_tip_strain()
        self.u[-1] = self.u[-2] + self.dx * tip_strain
        self.displacement_hist.append(self.u)

    def update_r(self) -> None:
        """
        Use the volume conservation equation to update r
        """
        du_dt = self.get_velocity()
        d2u_dxdt = np.gradient(du_dt, self.dx)
        du_dx = np.gradient(self.u, self.dx)
        update = -(self.r**2) * (d2u_dxdt) / (2 * (1 + du_dx))
        self.r = self.r + self.dt * update
        pass

    def one_step(self, active_stress: np.ndarray) -> None:
        """
        Take one step to update the displacement and radius.
        """
        self.update_u(active_stress)
        self.pos = self.pos_init + self.u
        self.update_r()
