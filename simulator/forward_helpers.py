"""Some helper functions for running the simulation of the chameleon"""

import numpy as np
import derivatives as d


def update_pos(chameleon):
    """
    Update the final position of an element

    Since displacement refers to global displacement we need to use the
    initial position of the rod plus it's overall displacement to compute
    it's final position i.e. u = x_f - x_0 => x_f = u + x_0. Since the left
    end of the rod is fixed to the boundary, we make sure to enforce the
    boundary condition in the computation of displacements as that is where
    forces will be computed. I am not sure if there is a better way to do
    this BC enforcement but it seems it should be alright for now.

    Parameters
    ----------
    chameleon : Chameleon
        Chameleon object whose tongue we are simulating.
    """
    chameleon.pos_f = chameleon.disp_current + chameleon.pos_0


def update_disp(chameleon, active_stress: np.ndarray):
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
    displacement to zero and we need to make sure that Edu/dx=-singma_a at
    the right end. I am using a trick described
    http://hplgit.github.io/prog4comp/doc/pub/p4c-sphinx-Python/._pylight006.html
    to ensure that the derivative boundary condition is satisfied.

    Parameters
    ----------
    chameleon :
        Chameleon object whose tongue we are simulating.
    active_stress : np.ndarray
        Externally applied stress.
    """
    if chameleon.with_grad:  # use eqn where drag has gradient on it.
        internal_term = (chameleon.E / chameleon.alpha) * d.first_space_deriv(
            chameleon.disp_current, chameleon.pos_0
        )
        external_term = (1 / chameleon.alpha) * active_stress
        new_disp = chameleon.disp_current + chameleon.dt * (
            internal_term + external_term
        )
    else:
        internal_term = (chameleon.E / chameleon.alpha) * d.second_space_derivative(
            chameleon.disp_current, chameleon.pos_0
        )
        external_term = (1 / chameleon.alpha) * d.first_space_deriv(
            active_stress, chameleon.pos_0
        )

        new_disp = chameleon.disp_current + chameleon.dt * (
            internal_term + external_term
        )
        # satisfying derivative bc.
        dx = chameleon.pos_0[-1] - chameleon.pos_0[-2]
        ## THIS LOOKS WRONG. SHOULD PROBABLY USE PREVIOUS DISPLACEMENT
        ## USE FIRST ORDER METHOD FOR NOW BUT COME BACK IF IT WORKS.
        # update = (chameleon.dt / chameleon.alpha) * (
        #     active_stress[-1]
        #     + chameleon.E
        #     * (
        #         2 * chameleon.disp_current[-2]
        #         - 2 * chameleon.disp_current[-1]
        #         - (2 * dx * active_stress[-1]) / (chameleon.E)
        #     )
        #     / (dx ** 2)
        # )
        # last_element_disp = chameleon.disp_current[-1] + update
        last_element_disp = (-active_stress[-1] * dx) / (chameleon.E) + new_disp[-2]
        new_disp[-1] = last_element_disp
    new_disp[0] = 0
    chameleon.disp_previous = chameleon.disp_current
    chameleon.disp_current = new_disp
    chameleon.active_stress_history.append(active_stress)


def one_step(chameleon, active_stress: np.ndarray):
    """
    Move the simulation forward one step in time by upddating the position
    of each element.

    Parameters
    ----------
    chameleon : Chameleon
        Chameleon object whose tongue we are simulating.
    active_torque : np.ndarray
        Active torque specified at this moment in time.
    """
    update_disp(chameleon, active_stress)
    update_pos(chameleon)
    chameleon.position_history.append(chameleon.pos_f)
    chameleon.displacement_history.append(chameleon.disp_current)


def forward_simulate(chameleon, active_stress: np.ndarray, **sim_steps):
    """
    Forward simulate the chameleon tongue for a fixed amount of time with a given
    active stress.

    If no number of forward steps is specified, then we will just simulate
    the rod forward in time for the number of steps given in chameleon.n_steps.

    Parameters
    ----------
    chameleon : Chameleon
        Chameleon object whose tongue we are simulating.
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
        T = chameleon.n_steps
    for i in range(T):
        diffs = np.diff(chameleon.pos_f)
        elements_increasing = np.all(diffs > 0)
        # last_element_last = diffs[-1] > 0
        if elements_increasing:
            one_step(chameleon, active_stress)
        else:
            raise ValueError("The rod elements have become out of order")
