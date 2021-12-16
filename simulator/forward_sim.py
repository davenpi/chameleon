"""
This script implements the forward simulation of the dynamical equation
of motion for the chameleon system
"""
import numpy as np

class Chameleon:
    def __init__(
        self, n_elems: int = 10, length: float = 1, E: float = 1, alpha: float = 1
    ):
        self.n_elems = n_elems
        self.length = length
        self.E = E
        self.alpha = alpha
        self.pos_0 = np.linspace(1, length, n_elems)
        self.pos_f = np.linspace(1, length, n_elems)
        self.disp = self.pos_f - self.pos_0

def disp_time_derivative(disp_new: np.ndarray, disp_old: np.ndarray, delta_t: float) -> np.ndarray:
    """
    Compute the time derivative of u. (using Backwards Euler? so method here
    may be too simple)
    """
    du_dt = (disp_new - disp_old)/delta_t
    return du_dt

def u_space_derivative(disp: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Compute the first spatial derivative of displacement with repect
    to position. I think numpy handles the averaging of left and right and
    also handles the boundaries properly. Might need to do some kind of
    interpolation if things are too bad.
    """
    # (u_i - u_(i-1)) / (x_i - x_(i-1))
    deriv = np.gradient(disp, varargs=pos)
    return deriv

def u_second_space_derivative(disp: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Compute second spatial derivative of dipslacement with respect to position.
    Don't see any easy built in functions built to handle this. Probably exists
    """
    pass

def update_pos(pos_0: np.ndarray, disp: np.ndarray) -> np.ndarray:
    """
    Compute current position. May want to use position from last frame instead
    of original position but the basic idea is here.
    """
    pos_f = disp + pos_0
    return pos_f

def check_dist_target(pos_tongue: np.ndarray, pos_target: np.ndarray) -> float:
    """
    Compute distance between tip of the tongue and the target
    """
    distance = pos_tongue[-1] - pos_target[0]
    if distance < 0:
        print("haven't reached")
    if distance > 0:
        print("overshot")
    return distance

def active_torque_deriv(torque: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    compute derivative of the active torque. 
    """
    deriv = np.gradient(torque, x)
    return deriv