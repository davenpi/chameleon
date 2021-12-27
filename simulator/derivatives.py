"""This script holds all of the derivative functions for the simulator."""

import numpy as np
from scipy.interpolate import UnivariateSpline


def time_derivative(
    y_curr: np.ndarray, y_prev: np.ndarray, delta_t: float
) -> np.ndarray:
    """
    Compute the time derivative of u.

    Compute time derivative by looking at difference between current and
    previous values of variable and dividing by the time step.

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

    Using built in numpy method to compute gradient. Using a second order
    method to compute derivatives at the boundary.

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
    # spline = UnivariateSpline(x, y)
    # deriv = spline.derivative(n=1)(x)
    deriv = np.gradient(y, x, edge_order=2)
    return deriv


def second_space_derivative(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute second spatial derivative of y with respect to x.

    Using numpy gradient twice to compute second derivative. Using a second
    order method to compute derivative at boundary.

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
    # spline = UnivariateSpline(x, y)
    # deriv = spline.derivative(n=2)(x)
    first_deriv = first_space_deriv(y, x)
    second_deriv = np.gradient(first_deriv, x, edge_order=2)
    return second_deriv
