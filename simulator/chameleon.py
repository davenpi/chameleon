"""
This script implements the Chameleon system.
"""
from typing import Tuple
import gym
import numpy as np
import derivatives as d
from collections import deque
import forward_helpers as fh


class Chameleon(gym.Env):
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
        super(Chameleon, self).__init__()
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
        # coefficients in a + bx + cx^2
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(3,))
        # observation space is just going to consist of tip of tongue
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,))

    def step(self, action: np.ndarray, **sim_steps) -> Tuple[float, float, bool, dict]:
        """
        Move the simulation forward one step in time by upddating the position
        of each element.

        Parameters
        ----------
        action : np.ndarray
            action used to specify active torque. Right now they are the
            coefficients of the constant, linear, and quadratic terms in
            the stress.

        kwargs
        ------
        sim_steps : int
            Optional keyword argument to specify how many steps to simulate
            for.

        Returns
        -------
        state : float
            State of system after step. Just the tip position.
        reward : float
            Reward computed after the step is over.
        done : bool
            True if episode is done, false otehrwise. For now I will only
            consider an episode done when the simulation breaks.
        info : dict
            Logging info.
        """
        steps = sim_steps.get("sim_steps")
        constant = action[0] * np.ones(self.n_elems)
        linear = action[1] * self.pos_f
        quadratic = action[2] * self.pos_f ** 2
        active_stress = constant + linear + quadratic
        done = False
        try:
            fh.forward_simulate(self, active_stress, sim_steps=steps)
            state = self.pos_f[-1]
            # get negative reward for distance and one negative reward for time
            reward = -(np.abs(state - self.target_pos)) - 1
            if state == self.target_pos:
                done = True
        except:
            print("Simulation messed up somewhere!")
            reward = -1000
            done = True
            state = np.NaN
        return state, reward, done, {}

    def reset(self):
        self.position_history.clear()
        self.pos_f = self.pos_0

    def render(self):
        pass

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
