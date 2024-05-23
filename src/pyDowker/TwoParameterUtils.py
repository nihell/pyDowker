import numpy as np
from pyrivet import rivet

def graded_rank_at_value(betti, x,y):
    """
    Evaluates the Hilbert function of a bipersistence module at point (x,y)

    Parameters
    ----------
    betti : rivet.Betti
        bipersistence module
    x : float
        first parameter
    y : float
        second parameter

    Returns
    -------
    int
        dimension of the bipersistence module at the given point.
    """
    for i in range(betti.graded_rank.shape[1]):
        this_x = betti.dimensions.x_grades[i]
        if this_x > x:
            i -= 1
            break

    for j in range(betti.graded_rank.shape[0]):
        this_y = betti.dimensions.y_grades[j]
        if this_y > y:
            j -= 1
            break

    if j < 0 or i < 0 or i == betti.graded_rank.shape[1] or j == betti.graded_rank.shape[0]:
        return 0
    return betti.graded_rank[j,i]

def discretize_graded_rank(betti, x_grid, y_grid):
    """
    Evaluate the Hilbert function on a user-defined grid.

    Parameters
    ----------
    betti : rivet.Betti
        bipersistence module
    x_grid : numpy.array
        grid of values for first parameter
    y_grid : numpy.array
        grid of values for second parameter

    Returns
    -------
    numpy.array
        values of the Hilbert function evaluated at the grid points
    """
    betti_grid = np.zeros((len(x_grid), len(y_grid)))

    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            betti_grid[i,j] = graded_rank_at_value(betti, x, y)

    # just to be consistent with pyRivet
    return betti_grid.T


def ecp_value(contributions, x, y):
    """
    Evaluates an Euler characteristic profile, given as a list of contributions, at the point (x,y) in two parameter space.
    Assumes that the first parameter is ordered w.r.t. <= and the second parameter w.r.t. >=

    Parameters
    ----------
    contributions : list
        list of pairs (bidegree, contribution)
    x : float
        first parameter
    y : float
        second parameter

    Returns
    -------
    int
        Vale of the ECP at (x,y)
    """
    return sum([c[1] for c in contributions if (c[0][0] <= x) and
                                               (c[0][1] >= y)])# y-reverse !!!


def grid_ECP(contributions, x_grid, y_grid):
    """
    Evaluate the Euler characteristic profile on a user-defined grid.
    

    Parameters
    ----------
    contributions : list
        list of pairs (bidegree, contribution)
    x_grid : numpy.array
        grid of values for first parameter
    y_grid : numpy.array
        grid of values for second parameter

    Returns
    -------
    numpy.array
        values of the Hilbert function evaluated at the grid points
    """
    ecp_grid = np.zeros((len(x_grid), len(y_grid)))

    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            ecp_grid[i,j] = ecp_value(contributions, x, y)

    # just to be consistent with pyRivet
    return ecp_grid.T