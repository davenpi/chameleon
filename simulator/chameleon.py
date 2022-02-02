import copy
import gym
from typing import Tuple
from collections import deque
import numpy as np
from scipy.integrate import cumtrapz


class Chameleon(gym.Env):
    def __init__(
        self,
        E: float = 1.0,
        alpha: float = 1,
        C: float = 1,
        n_elems: int = 50,
        dt: float = 1e-3,
        init_length: float = 0.1,
        target_pos: float = 0.18,
        atol: float = 0.05,
        train: bool = True,
    ) -> None:
        super(Chameleon, self).__init__()
        self.E = E
        self.alpha = alpha
        self.C = C
        self.g = self.E / self.alpha
        self.length = init_length
        self.drag = False
        self.n_elems = n_elems
        self.target_pos = target_pos
        self.original_target_pos = target_pos
        self.dt = dt
        self.steps_per_sec = int(1 / self.dt)
        self.atol = atol
        self.pos_init = np.linspace(0, init_length, self.n_elems)
        self.dx = self.pos_init[1] - self.pos_init[0]
        self.pos = copy.deepcopy(self.pos_init)
        self.u_current = self.pos - self.pos_init
        self.u_hist = deque([], maxlen=10_000)
        self.u_hist.append(self.u_current)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.learning_counter = 0
        self.episode_length = 59  # length 60 eps but start counting at zero
        self.active_stress_hist = deque([], maxlen=self.episode_length + 1)
        self.active_stress_hist.append(np.zeros(self.n_elems))
        self.winning_stress_hist = deque([], maxlen=self.episode_length + 1)
        self.episode_counter = 0
        self.ep_rew = 0
        self.reward_history = []
        assert (
            self.target_pos < self.observation_space.high.item()
        ), "target outside observation space"
        self.train = train

    def one_step(self, active_stress: np.ndarray) -> None:
        """
        Update the displacement using the equation of motion.

        Satisfy the stress free equation at the boundary.

        Parameters
        ----------
        active_stress : np.ndarry
            Active steess to be applied at this time.
        """
        sig_int = cumtrapz(active_stress, dx=self.dx, initial=0)
        self.u_current = self.u_current + self.dt * (
            -self.g * self.u_current - (1 / self.alpha) * sig_int
        )
        self.pos = self.pos_init + self.u_current
        self.u_hist.append(self.u_current)

    def one_step_drag(self, active_stress: np.ndarray) -> None:
        """
        Step forward in time with drag at boundary.
        """
        sig_int = cumtrapz(active_stress, dx=self.dx, initial=0)
        du_dtL = -(1 / (1 + self.C * self.length / self.alpha)) * (
            self.g * self.u_current[-1] + (1 / self.alpha) * sig_int[-1]
        )
        self.u_current = self.u_current + self.dt * (
            -self.g * self.u_current
            - (1 / self.alpha) * sig_int
            - (self.C * self.pos_init / self.alpha) * du_dtL
        )
        self.pos = self.pos_init + self.u_current
        self.u_hist.append(self.u_current)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        const = action[0] * np.ones(self.n_elems)
        active_stress = const
        self.active_stress_hist.append(active_stress)

        if self.drag:
            for _ in range(self.steps_per_sec):
                self.one_step_drag(active_stress=active_stress)
        else:
            for _ in range(self.steps_per_sec):
                self.one_step(active_stress=active_stress)

        self.learning_counter += 1
        diffs = np.diff(self.pos)
        if any(diffs < 0):
            reward = -60
            self.ep_rew += reward
            done = True
            state = self.reset()
        else:
            difference = self.target_pos - self.pos[-1]
            state = np.array([difference], dtype=np.float32)
            out_of_bounds = (
                state.item() > self.observation_space.high.item()
                or state.item() < self.observation_space.low.item()
            )
            overtime = self.learning_counter > self.episode_length
            close = np.isclose(0, state.item(), atol=self.atol)
            if overtime:
                reward = -1
                self.ep_rew += reward
                state = self.reset()
                done = True
            elif out_of_bounds:
                reward = -1
                self.ep_rew += reward
                state = self.reset()
                done = True
            elif close:
                if self.target_pos == self.pos_init[-1]:
                    self.winning_pos = self.pos  # kludge to get last position
                    self.winning_stress_hist = copy.copy(self.active_stress_hist)
                    reward = 1
                    self.ep_rew += reward
                    state = self.reset()
                    done = True
                else:  # reward for reaching first target and update target position
                    self.target_pos = self.pos_init[-1]
                    reward = -1
                    self.ep_rew += reward
                    done = False
                    self.drag = True
            else:
                reward = -1
                self.ep_rew += reward
                done = False

        info = {}
        return state, reward, done, info

    def reset(self) -> np.ndarray:
        self.pos = copy.deepcopy(self.pos_init)
        self.u_current = self.pos - self.pos_init
        self.active_stress_hist.clear()
        self.active_stress_hist.append(np.zeros(self.n_elems))
        if self.train:
            self.target_pos = (
                (self.original_target_pos - self.pos[-1]) * np.random.sample(1)
                + self.pos[-1]
                + self.atol
            )
        else:
            self.target_pos = self.original_target_pos
        difference = self.target_pos - self.pos[-1]
        state = np.array([difference], dtype=np.float32)
        self.ep_rew = 0
        self.episode_counter += 1
        self.learning_counter = 0
        self.drag = False
        return state

    def render():
        pass
