import copy
from typing import Tuple
from collections import deque
import gym
import numpy as np
from scipy.integrate import cumtrapz, solve_ivp


class Chameleon(gym.Env):
    def __init__(
        self,
        E: float = 1,
        alpha: float = 1,
        c: float = 1,
        m: float = 1,
        n_elems: int = 50,
        dt: float = 1e-2,
        init_length: float = 1,
        target_pos: float = 1.8,
        atol: float = 0.05,
        train: bool = True,
        t_max: float = 10,
    ) -> None:
        super(Chameleon, self).__init__()
        self.E = E
        self.alpha = alpha
        self.c = c
        self.m = m
        self.tau = self.alpha / self.E
        self.g = self.E / self.alpha
        self.length = init_length
        self.drag = False
        self.n_elems = n_elems
        self.target_pos = target_pos
        self.original_target_pos = target_pos
        self.dt = dt
        self.atol = atol
        self.pos_init = np.linspace(0, init_length, self.n_elems)
        self.dx = self.pos_init[1] - self.pos_init[0]
        self.pos = copy.deepcopy(self.pos_init)
        self.u_current = self.pos - self.pos_init
        self.u_hist = deque([], maxlen=10_000)
        self.u_hist.append(self.u_current)
        self.u_velocity_history = []
        self.observation_space = gym.spaces.Box(
            low=-10 * self.length, high=10 * self.length, shape=(1,)
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.learning_counter = 0
        self.episode_length = 9  # episode starts counting at zero
        self.active_stress_hist = deque([], maxlen=self.episode_length + 1)
        self.active_stress_hist.append(np.zeros(self.n_elems))
        self.winning_stress_hist = deque([], maxlen=self.episode_length + 1)
        self.episode_counter = 0
        self.ep_rew = 0
        self.train = train
        self.t_max = t_max
        self.t_arr = np.arange(0, self.t_max + self.dt, self.dt)
        self.time = 0
        self.one_steps = 0
        self.U0 = [0, 0]
        self.F = []
        self.returning = False

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
        du_dt = (self.u_current - self.u_hist[-1])[-1] / self.dt
        self.u_hist.append(self.u_current)
        self.u_velocity_history.append(du_dt)

    def compute_boundary_stress(self, active_stress: np.ndarray) -> float:
        """Compute boundary stress F for a given active stress."""
        sig_int = cumtrapz(active_stress, dx=self.dx)

        def system(t, y, m, L, c, a, E, sig_int) -> np.ndarray:
            """
            System of equations coming from u_L ODE.

            This function returns dy/dt (vector), the vector that comes from
            turning the second order equation for u_L into a system of first
            order equations.

            Parameters
            ----------
            t : float
                Current time in simulation.
            y : np.ndarray
                Current values of variables being solved for in ODE.
            m : float
                Mass terms.
            L : float
                Length of rod.
            c : float
                Drag coefficient.
            a : float
                Strain rate drag coefficient.
            E : float
                Young's Modulus.
            sig_int : np.ndarray
                Cumulative integral of sigma along the rod.

            Returns
            -------
            y_out : np.ndarray
                Current value of dy/dt for the ODE system.
            """
            mat = np.array([[0, 1], [-E / (L * m), -(L * c + a) / (L * m)]])
            y_out = np.matmul(mat, y) + np.array([0, -sig_int[-1] / (L * m)])
            return y_out

        if self.time == 0:
            duL_dt = self.U0[1]
            d2uL_dt = (1 / (self.length * self.m)) * (
                -self.E * self.U0[0]
                - (self.length * self.c + self.alpha) * duL_dt
                - sig_int[-1]
            )
            u = [self.U0[0]]
            self.u_tiny = u
            # self.time_crunch = [0]
            # self.u_tiny_deriv = [0]
        else:
            sol = solve_ivp(
                system,
                # t_span=[self.time - self.dt, self.time],
                t_span=[self.time, self.time + self.dt],
                method="BDF",
                y0=[self.U0[0], self.U0[1]],
                args=[self.m, self.length, self.c, self.alpha, self.E, sig_int],
            )
            t = sol["t"]
            # self.time_crunch = t
            u = sol["y"][0]
            self.u_tiny = u
            duL_dt = sol["y"][1]
            # self.u_tiny_deriv = duL_dt
            try:
                d2uL_dt = np.gradient(duL_dt, t, edge_order=2)
            except:
                d2uL_dt = np.gradient(duL_dt, t, edge_order=1)
            duL_dt = duL_dt[-1]
            d2uL_dt = d2uL_dt[-1]
            self.U0[0] = u[-1]

        self.U0[1] = duL_dt
        f = -self.c * duL_dt - self.m * d2uL_dt
        return f

    def one_step_return(self, active_stress: np.ndarray) -> None:
        """
        Method which implements one step of the dynamics during return steps.

        The equation of motion is the same in this case but instead of using
        a stress free boundary condition we add a drag and an inertial term at
        the boundary. These represent effects from the food on the tongue of
        the chameleon.
        """
        sig_int = cumtrapz(active_stress, dx=self.dx, initial=0)
        f = self.compute_boundary_stress(active_stress)

        self.u_current = self.u_current + self.dt * (
            -self.g * self.u_current
            - (1 / self.alpha) * sig_int
            + f * self.pos_init / self.alpha
        )
        # self.u_current[-1] = self.u_tiny[-1]
        # self.time += self.dt  # UNCOMMENT FOR CALIBRATION
        du_dt = (self.u_current - self.u_hist[-1])[-1] / self.dt
        self.u_hist.append(self.u_current)
        self.u_velocity_history.append(du_dt)
        self.U0[0] = self.u_current[-1]
        self.U0[1] = du_dt
        self.pos = self.pos_init + self.u_current

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        const = action[0] * np.ones(self.n_elems)
        if self.returning:
            active_stress = 5_000 * const
            self.one_step_return(active_stress=active_stress)
        else:
            active_stress = 50 * const
            self.one_step(active_stress=active_stress)

        self.learning_counter += 1
        diffs = np.diff(self.pos)
        if any(diffs < 0):
            reward = -self.episode_length - 1
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
                reward = -self.episode_length - 1
                self.ep_rew += reward
                done = True
                state = self.reset()

            elif close:
                if self.target_pos == self.pos_init[-1]:
                    self.winning_pos = self.pos  # kludge to get last position
                    self.winning_stress_hist = copy.copy(self.active_stress_hist)
                    reward = 10
                    self.ep_rew += reward
                    state = self.reset()
                    done = True
                else:  # reward for reaching first target and update target position
                    self.target_pos = self.pos_init[-1]
                    reward = -1
                    self.ep_rew += reward
                    done = False
                    self.returning = True
                    self.U0[0] = self.u_current[-1]
                    self.U0[1] = self.u_velocity_history[-1]

            else:
                reward = -1
                self.ep_rew += reward
                done = False

        if done:
            info = {"target_pos": round(self.init_tp, 3)}
        else:
            info = {}
        diff = np.abs(self.target_pos - self.pos[-1])
        reward -= diff
        self.time += self.dt
        return state, reward, done, info

    def reset(self) -> np.ndarray:
        self.clear_history()
        self.pos = copy.deepcopy(self.pos_init)
        self.u_current = self.pos - self.pos_init
        self.u_hist.append(self.u_current)
        self.active_stress_hist.append(np.zeros(self.n_elems))
        if self.train:
            self.target_pos = (
                (self.original_target_pos - self.pos[-1]) * np.random.sample(1)
                + self.pos[-1]
                + self.atol
            )
        else:
            self.target_pos = np.array([self.original_target_pos])
        self.target_pos = self.target_pos.item()
        self.init_tp = copy.deepcopy(self.target_pos)
        difference = self.target_pos - self.pos[-1]
        state = np.array([difference], dtype=np.float32)
        self.ep_rew = 0
        self.episode_counter += 1
        self.learning_counter = 0
        self.returning = False
        self.one_steps = 0
        self.time = 0
        return state

    def clear_history(self):
        """
        Clear all of the history data for a episode.
        """
        self.u_hist.clear()
        self.u_velocity_history.clear()
        self.active_stress_hist.clear()
        self.winning_stress_hist.clear()

    def render():
        pass

    def one_step_drag(self, active_stress: np.ndarray) -> None:
        """
        Step forward in time with drag at boundary.
        """
        sig_int = cumtrapz(active_stress, dx=self.dx, initial=0)
        du_dtL = -(1 / (1 + self.c * self.length / self.alpha)) * (
            self.g * self.u_current[-1] + (1 / self.alpha) * sig_int[-1]
        )
        self.u_current = self.u_current + self.dt * (
            -self.g * self.u_current
            - (1 / self.alpha) * sig_int
            - (self.C * self.pos_init / self.alpha) * du_dtL
        )
        self.pos = self.pos_init + self.u_current
        self.u_hist.append(self.u_current)
