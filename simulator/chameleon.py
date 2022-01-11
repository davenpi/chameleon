import copy
from typing import Tuple
import gym
import numpy as np


class Chameleon(gym.Env):
    def __init__(
        self,
        E: float,
        alpha: float,
        n_elems: int = 50,
        dt: float = 1e-5,
        target_pos: float = 0.18,
    ) -> None:
        super(Chameleon, self).__init__()
        self.E = E
        self.alpha = alpha
        self.n_elems = n_elems
        self.target_pos = target_pos
        self.dt = dt
        self.pos_init = np.linspace(0, 0.1, self.n_elems)
        self.pos = copy.deepcopy(self.pos_init)
        self.u_current = self.pos - self.pos_init
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        self.learning_counter = 0
        self.episode_length = 24

    def one_step(self, active_stress: np.ndarray) -> None:
        u_gradient = np.gradient(
            self.u_current, edge_order=2
        )  # NEED TO TAKE DERIV WRT X_INIT otherwise unit spacing is used
        update = self.dt * (
            (self.E / self.alpha) * u_gradient + (1 / self.alpha) * active_stress
        )
        u_final = self.u_current + update
        u_final[0] = self.u_current[0]
        self.u_current = copy.deepcopy(u_final)
        self.pos = self.pos_init + self.u_current

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        const = action[0] * np.ones(self.n_elems)
        linear = action[1] * self.pos
        quad = action[2] * self.pos ** 2
        active_stress = const + linear + quad

        for i in range(1000):  # take 1000 steps per learning step
            self.one_step(active_stress)

        self.learning_counter += 1
        diffs = np.diff(self.pos)
        if any(diffs < 0):  # end sim if elements out of order
            state = self.reset()
            reward = -1000
            done = True
        else:
            state = np.array([self.pos[-1]], dtype=np.float32)
            overtime = self.learning_counter > self.episode_length
            close = np.isclose(state.item(), self.target_pos, rtol=0.01)
            if overtime:
                state = self.reset()
                done = True
                reward = -1
            elif close:
                print(f"Solved! in {self.learning_counter} steps")
                self.winning_pos = self.pos  # kludge to get last position
                state = self.reset()
                done = True
                reward = 0
            else:
                reward = -1
                done = False

        info = {}
        return state, reward, done, info

    def reset(self) -> np.ndarray:
        self.pos = copy.deepcopy(self.pos_init)
        self.u_current = self.pos - self.pos_init
        state = np.array([self.pos[-1]], dtype=np.float32)
        self.learning_counter = 0
        return state

    def render():
        pass
