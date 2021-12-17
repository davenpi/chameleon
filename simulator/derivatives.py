"""
This script holds all of the derivative computations that must be done for the
simulator.
"""
import numpy as np
from scipy.interpolate import UnivariateSpline


def disp_time_derivative(
    disp_curr: np.ndarray, disp_prev: np.ndarray, delta_t: float
) -> np.ndarray:
    """
    Compute the time derivative of u. 
    
    Maha mentioned using Backwards Euler but I am not really sure how to do
    that and my current approach may be too simple.

    Parameters
    ----------
    disp_new : np.ndarray
        Current displacement of each element.
    disp_prev : np.ndarray
        Displacement of each element at previous time step.
    delta_t : np.ndarray
        TIme step size.
    
    Returns
    -------
    du_dt : np.ndarray
        Derivative of displacement with respect to time. 
    
    """
    du_dt = (disp_curr - disp_prev) / delta_t
    return du_dt


def u_second_space_derivative(disp: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Compute second spatial derivative of dipslacement with respect to position.
    
    We first create a spline of the displacement and then use the built in 
    scipy method to evaluate it's second derivative. We then select the
    derivative at points we have specified. Don't see any easy built in 
    functions built to handle this but the probably exist. I know I can use
    finite difference methods to compute derivatives but handling the boundary
    seems a little annoying.

    Parameters
    ----------
    disp : np.ndarray
        Current displacement of elements.
    pos : np.ndarray
        Current position of elements.
    
    Returns
    -------
    deriv : np.ndarray
        Value of second derivative of displacement at input positions.
    """
    spline = UnivariateSpline(pos, disp)
    deriv = spline.derivative(n=2)(pos)
    return deriv


def active_torque_deriv(torque: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """
    Compute derivative of the active torque. 

    We first create a spline of the torque over the data and then use a built
    in scipy method to compute the derivative. We also have the option of
    working directly with the data and computing derivatives using numpy but
    we want to be consistent with how we compute second derivatives.

    Parameters
    ----------
    torque : np.ndarray
        Specificed active torque at each element of the rod.
    x : np.ndarray
        Current x position of elements along the rod.
    
    Returns
    -------
    deriv : np.ndarray
        Derivative of torque with respect to position.
    """
    spline = UnivariateSpline(pos, torque)
    deriv = spline.derivative(n=1)(pos)
    # deriv = np.gradient(torque, varargs=pos)
    return deriv
