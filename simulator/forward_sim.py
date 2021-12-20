"""
This script implements the forward simulation of the dynamical equation
of motion for the chameleon system.
"""
import numpy as np
import derivatives as d
from collections import deque


class Chameleon:
    def __init__(
        self,
        target_pos: float = 2,
        n_elems: int = 10,
        length: float = 1,
        E: float = 1,
        alpha: float = 1,
        dt: float = 1e-3,
        final_time: float = 1,  # if i have a Chameleon does it really make
        # sense to give it a final time attribute? not really.
    ):
        self.target_pos = target_pos
        self.n_elems = n_elems
        self.length = length
        self.E = E
        self.alpha = alpha
        self.pos_0 = np.linspace(0, length, n_elems)
        self.pos_f = np.linspace(0, length, n_elems)
        self.disp_current = self.pos_f - self.pos_0
        self.disp_previous = 0  # not really used anywhere right now.
        self.dt = dt
        self.final_time = final_time
        self.n_steps = int(self.final_time / self.dt)
        self.position_history = deque([], maxlen=self.n_steps)

    def update_disp(self, active_stress: np.ndarray):
        """
        Update the displacement using the dynamical equation of motion.
        
        Our equation of motion is E*d^2u/dx^2 + d(sigma_a)/dx = alpha * du/dt
        In general we know that u(t) = u(t-1) + delta_t * du/dt so we use the
        equation of motion laid out above to give an update rule for u.
        First we compute the internal force which is the second derivative
        of displacement with respect to position. Then we compute the external
        force which is the derivative of the externally applied stress with
        respect to position. Since the sum of those two (times some constants)
        is equal to du/dt, we update u using the computed spatial derivatives
        times the appropriate constants. In the end we need to make sure the
        point on the left end of the rod is stationary so we manually set it's
        displacement to zero (that may not be correct and is something I will
        ask about).
        
        Parameters
        ----------
        active_stress : np.ndarray
            Externally applied stress.
        """
        internal_stress = (self.E / self.alpha) * d.second_space_derivative(
            self.disp_current, self.pos_f
        )
        external_stress = (1 / self.alpha) * d.first_space_deriv(
            active_stress, self.pos_f
        )
        new_disp = self.disp_current + self.dt * (internal_stress + external_stress)
        # should set initial displacement to 0 right? just to satisfy left BC
        # and be sure that I have the correct forces computed.
        new_disp[0] = 0
        self.disp_previous = self.disp_current
        self.disp_current = new_disp

    def update_pos(self):
        """
        Update the final position of an element
        
        Since displacement refers to global displacement we need to use the
        initial position of the rod plus it's overall displacement to compute
        it's final position i.e. u = x_f - x_0 => x_f = u + x_0. Since the left
        end of the rod is fixed to the boundary, we make sure to enforce the
        boundary condition in the computation of displacements as that is where
        forces will be computed. I am not sure if there is a better way to do
        this BC enforcement but it seems it should be alright for now.
        """
        self.pos_f = self.disp_current + self.pos_0

    def step(self, active_stress: np.ndarray):
        """
        Move the simulation forward one step in time by upddating the position
        of each element.

        Parameters
        ----------
        active_torque : np.ndarray
            Active torque specified at this moment in time.
        """
        self.update_disp(active_stress)
        self.update_pos()
        self.position_history.append(self.pos_f)

    def forward_simulate(self, active_stress: np.ndarray, **sim_steps):
        """
        Forward simulate the tongue for a fixed amount of time with a given
        active stress.

        If no number of forward steps is specified, then we will just simulate
        the rod forward in time for the number of steps given in self.n_steps.

        Parameters
        ----------
        active_stress : np.ndarray
            Active stress to apply for duration of forward simulation.

        kwargs
        ------
        sim_steps : int
            optional keyword argument that can specify how many steps to
            simulate forward for.

        """
        sim_steps = sim_steps.get("sim_steps")
        if sim_steps:
            T = sim_steps
        else:
            T = self.n_steps
        for i in range(T):
            self.step(active_stress)

    def check_target_dist(self):
        """
        Check distance from tip of rod to target.
        """
        dist = self.pos_f[-1] - self.target_pos
        if dist < 0:
            print("haven't reached")
        if dist > 0:
            print("overshot")
        return dist
