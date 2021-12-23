"""This script holds all of the derivative functions for the simulator."""

import numpy as np
from scipy.interpolate import UnivariateSpline


def time_derivative(
    y_curr: np.ndarray, y_prev: np.ndarray, delta_t: float
) -> np.ndarray:
    """
    Compute the time derivative of u.

    Maha mentioned using Backwards Euler but I am not really sure how to do
    that and my current approach may be too simple.

    Parameters
    ----------
    y_curr : np.ndarray
        Current y value at each element.
    y_prev : np.ndarray
        Y value of each element at previous time step.
    delta_t : np.ndarray
        Time step size.

    Returns
    -------
    dy_dt : np.ndarray
        Derivative of y respect to time.

    """
    dy_dt = (y_curr - y_prev) / delta_t
    return dy_dt


def first_space_deriv(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute derivative of y with respect to x.

    We first create a spline of y over x and then use a built in scipy method
    to compute the derivative. We also have the option of working directly with
    the data and computing derivatives using numpy but we want to be consistent
    with how we compute second derivatives.

    Parameters
    ----------
    y : np.ndarray
        Specificed active torque at each element of the rod.
    x : np.ndarray
        Current x position of elements along the rod.

    Returns
    -------
    deriv : np.ndarray
        Derivative of torque with respect to position.
    """
    spline = UnivariateSpline(x, y)
    deriv = spline.derivative(n=1)(x)
    # deriv = np.gradient(y, varargs=x)
    return deriv


def second_space_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute second spatial derivative of y with respect to x.

    We first create a spline of y and then use the built in scipy method to
    evaluate it's second derivative. We then select the derivative at points we
    have specified. I don't see any easy built in numpy methods to do this but
    they probably exist. I know I could use a finite difference method to
    compute derivatives but I don't know how to handle the boundary and the
    scipy method is likely faster.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable which depends on x.
    x : np.ndarray
        Independent variable.

    Returns
    -------
    deriv : np.ndarray
        Value of second derivative at input positions.
    """
    spline = UnivariateSpline(x, y)
    deriv = spline.derivative(n=2)(x)
    return deriv
