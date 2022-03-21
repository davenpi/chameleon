import copy
from turtle import pos
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
        n_elems: int = 50,
        dt: float = 0.001,
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

    def update_r(self):
        """
        Use the volume conservation equation to update r
        """
        v = self.get_velocity()
        grad_v = np.gradient(v, self.dx, edge_order=2)
        grad_u = np.gradient(self.u, edge_order=2)
        self.r = self.r - self.dt * self.r * grad_v / (2 * (1 + grad_u))

    def get_velocity(self):
        """
        Compute v = du_dt
        """
        v = (self.displacement_hist[-1] - self.displacement_hist[-2]) / self.dt
        return v

    def compute_stress(self, active_stress: np.ndarray) -> np.ndarray:
        v = self.get_velocity()
        elastic_stress = self.E * np.gradient(self.u, self.dx)
        viscous_stress = self.a * np.gradient(v, self.dx)
        tot_stress = active_stress + elastic_stress + viscous_stress
        return tot_stress

    def update_u(self, active_stress: np.ndarray) -> None:
        """
        Use the dynamical equation to update the displacement u.

        Need to make sure we satisfy the boundary condition that u(0, t) = 0
        and sigma(L, t) = 0. The sigma(L, t) = 0 BC gives me an ODE for the
        strain at the boundary. Solve the ODE and plug in answer when updating
        u. It's just strange because it seems to involve an integral in time
        of the active stress.

        Parameters
        ----------
        active_stress : np.ndarray
            active stress

        Returns
        -------
        None
        """
        pass

    def one_step(self, active_stress: np.ndarray) -> None:
        """
        Take one step to update the displacement and radius.
        """
        self.update_u(active_stress)
        self.update_r()
        self.pos = self.u + self.pos_init
