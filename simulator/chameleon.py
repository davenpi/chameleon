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
        init_pos: np.ndarray = np.linspace(0, 1, 50),  # hardcoding for now
        current_disp: np.ndarray = np.zeros(50),
        target_pos: float = 2,
        E: float = 1,
        alpha: float = 1,
        dt: float = 1e-3,  # this as an attribute for the chameleon is questionable
        with_grad: bool = True,
    ):
        super(Chameleon, self).__init__()
        self.target_pos = target_pos
        self.n_elems = np.size(init_pos)
        self.E = E
        self.alpha = alpha
        self.pos_0 = init_pos
        self.pos_f = self.pos_0
        self.disp_current = current_disp  # can be different than pos_f - pos_0
        self.dt = dt
        self.with_grad = with_grad
        self.n_steps = 1000
        self.position_history = deque([], maxlen=self.n_steps)
        self.position_history.append(self.pos_f)
        self.displacement_history = deque([], maxlen=self.n_steps)
        self.displacement_history.append(self.disp_current)
        self.active_stress_history = deque([], maxlen=self.n_steps)
        self.active_stress_history.append(np.zeros(self.n_elems))
        # coefficients in a + bx + cx^2
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(3,))
        # observation space is just going to consist of tip of tongue
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(1,))

    def step(
        self, action: np.ndarray, **sim_steps
    ) -> Tuple[np.ndarray, float, bool, dict]:
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
        print(action)
        print(active_stress)
        done = False
        # try:
        fh.forward_simulate(self, active_stress, sim_steps=steps)
        state = np.array([self.pos_f[-1]], dtype=np.float32)
        # get negative reward for distance and one negative reward for time
        reward = -np.abs(state - self.target_pos).item() - 1
        if state.item() == self.target_pos:
            done = True
            # negative reward for large velocity at the end. want to reach
            # target and be slowing down or nearly stopped
            reward -= (
                self.position_history[-1][-1] - self.position_history[-2][-1]
            ) / self.dt
            self.reset()
        # except:
        #     print("Simulation messed up somewhere!")
        #     reward = -1000.0
        #     done = True
        #     state = np.array([self.pos_f[-1]])
        #     self.reset()
        return state, reward, done, {}

    def reset(self):
        self.position_history.clear()
        self.pos_f = self.pos_0
        state = np.array([self.pos_f[-1]], dtype=np.float32)
        return state

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
