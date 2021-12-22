"""
This script implements the Chameleon system.
"""
import numpy as np
import derivatives as d
from collections import deque
import forward_helpers as fh


class Chameleon:
    def __init__(
        self,
        target_pos: float = 2,
        n_elems: int = 10,
        length: float = 1,
        E: float = 1,
        alpha: float = 1,
        dt: float = 1e-3,  # this as an attribute for the chameleon is questionable
        final_time: float = 1,  # if i have a Chameleon does it really make
        # sense to give it a final time attribute? not really. using it now
        # to compute self.n_steps which DOES get used in forward helpers.
        # will refactor.
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
        self.position_history.append(self.pos_f)

    def step(self, active_stress: np.ndarray, **sim_steps):
        """
        Move the simulation forward one step in time by upddating the position
        of each element.

        Parameters
        ----------
        active_torque : np.ndarray
            Active torque specified at this moment in time.

        kwargs
        ------
        sim_steps : int
            Optional keyword argument to specify how many steps to simulate
            for.
        """
        steps = sim_steps.get("sim_steps")
        fh.forward_simulate(self, active_stress, sim_steps=steps)

    def watch_history(self):
        pass

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
