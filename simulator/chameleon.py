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
        self.pos_init = np.linspace(0, 0.10, self.n_elems)
        self.pos = copy.deepcopy(self.pos_init)
        self.u_current = self.pos - self.pos_init
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
        self.learning_counter = 0
        self.episode_length = 50

    def one_step(self, active_stress: np.ndarray) -> None:
        diff = np.diff(self.pos_init)[0]
        u_gradient = np.gradient(self.u_current, diff, edge_order=2)
        u_gradient[-1] = active_stress[-1] / self.E  # Boundary condition
        C_t = -self.E * u_gradient[0] - active_stress[0]
        update = self.dt * (
            C_t / self.alpha
            + (self.E / self.alpha) * u_gradient
            + (1 / self.alpha) * active_stress
        )
        u_final = self.u_current + update
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
            close = np.isclose(state.item(), self.target_pos, rtol=0.05)
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
