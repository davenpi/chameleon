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
        target_pos=2,
        n_elems: int = 10,
        length: float = 1,
        E: float = 1,
        alpha: float = 1,
        dt: float = 1e-3,
        T: float = 10,
    ):
        self.target_pos = target_pos
        self.n_elems = n_elems
        self.length = length
        self.E = E
        self.alpha = alpha
        self.pos_0 = np.linspace(0, length, n_elems)
        self.pos_f = np.linspace(0, length, n_elems)
        self.disp_current = self.pos_f - self.pos_0
        self.disp_previous = 0
        self.dt = dt
        self.T = T
        self.n_steps = int(self.T / self.dt)
        self.position_history = deque([], maxlen=self.n_steps)

    def update_disp(self, active_torque: np.ndarray):
        internal_stress = (self.E / self.alpha) * d.u_second_space_derivative(
            self.disp_current, self.pos_f
        )
        external_stress = (1 / self.alpha) * d.active_torque_deriv(
            active_torque, self.pos_f
        )
        new_disp = self.disp_current + self.dt * (internal_stress + external_stress)
        self.disp_previous = self.disp_current
        self.disp_current = new_disp

    def update_pos(self):
        # think I should use previous position to get new position
        self.pos_f = self.disp_current + self.pos_f
        self.pos_f[0] = 0  # boundary condition

    def step(self, torque: np.ndarray):
        """
        Move the simulation forward one step in time. Update the position of
        each element.
        """
        self.update_disp(torque)
        self.update_pos()
        self.position_history.append(self.pos_f)

    def forward_simulate(self, torque):
        """
        Forward simulate the tongue for a fixed amount of time
        """
        for i in range(self.n_steps):  # running full forward simulation now.
            self.step(torque)

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
