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
        length: float = 0.1,
        n_elems: int = 50,
        target_pos: float = 0.13,
        E: float = 1,
        alpha: float = 1,
        dt: float = 1e-5,  # this as an attribute for the chameleon is questionable
        with_grad: bool = True,
    ):
        super(Chameleon, self).__init__()
        self.length = length
        self.target_pos = target_pos
        self.n_elems = n_elems
        self.E = E
        self.alpha = alpha
        self.pos_0 = np.linspace(0, length, n_elems)
        self.pos_f = self.pos_0
        self.disp_current = np.zeros(self.n_elems)
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
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        # observation space is just going to consist of tip of tongue
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.step_counter = 0
        self.ep_length = 15

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Move the simulation forward one step in time by upddating the position
        of each element.

        Parameters
        ----------
        action : np.ndarray
            Action used to specify active torque. Right now this array should
            be length three and contain the coefficients of the constant,
            linear, and quadratic terms in the stress. Each coefficient must
            lie in the range (0, 1).

        Returns
        -------
        state : np.ndarray
            State of system after step. Just the tip position.
        reward : float
            Reward computed after the step is over.
        done : bool
            True if episode is done, false otehrwise. For now I will only
            consider an episode done when the simulation breaks.
        info : dict
            Logging info.
        """
        constant = action[0] * np.ones(self.n_elems)
        linear = action[1] * self.pos_f
        quadratic = action[2] * self.pos_f ** 2
        active_stress = 10 * (constant + linear + quadratic)
        done = False
        try:
            fh.forward_simulate(self, active_stress, sim_steps=1_000)
            self.position_history.append(self.pos_f)
            self.step_counter += 1
            state = np.array([self.pos_f[-1]], dtype=np.float32)
            close = np.isclose(state.item(), self.target_pos, rtol=0.01)
            # overshot = state.item() > self.target_pos
            overtime = self.step_counter > self.ep_length
            if close:
                done = True
                # negative reward for large velocity at the end. want to reach
                # target and be slowing down or nearly stopped

                reward = 0
                #     - 10 * (self.pos_f[-1] - self.position_history[-2][-1]) / self.dt
                # )
                print(f"State is {state}")
                print(f"Reached in {self.step_counter} steps! =) with reward: {reward}")
                state = self.reset()
            elif overtime:  # fail for taking too long
                # if overshot:
                #     print("overshot target =(")
                print(f"took too long ( : > ( )")
                reward = -10
                done = True
                state = self.reset()
            else:  # get negative reward for distance and one negative reward for time
                # reward = -np.abs(state - self.target_pos).item() - 1
                reward = -1
        except:
            # print(
            #     "elements most likely out of order"
            # )  # terrible feedback. be more specific and differentiate
            done = True
            reward = -1000
            state = self.reset()

        return state, reward, done, {}

    def reset(self):
        self.step_counter = 0
        # self.position_history.clear()
        self.pos_0 = np.linspace(0, self.length, self.n_elems)
        self.pos_f = self.pos_0
        # self.position_history.append(self.pos_0)
        self.displacement_history.clear()
        self.disp_current = self.pos_f - self.pos_0
        self.displacement_history.append(self.disp_current)
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
