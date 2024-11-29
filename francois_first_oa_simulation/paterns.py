# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 09:02:14 2024

@author: fleroux
"""

import numpy as np

def GetCartesianCoordinates(nrows, **kargs):
    """
    GetCartesianCoordinates : generate two arrays representing the cartesian coordinates

    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.05, Brest
    Comments : For even support size, coordinates are defined like [-2,-1,0,1] (N = 4)
               (i.e the origin is the top right pixel of the four central ones)

    Inputs : MANDATORY : nrows {int} : the number of rows

              OPTIONAL : ncols {int} : the number of columns

    Outputs : [X,Y], two arrays representing the cartesian coordinates
    """

    # read optinal parameters values
    ncols = kargs.get("ncols", nrows)

    [X, Y] = np.meshgrid(
        np.arange(-ncols // 2 + ncols % 2, ncols // 2 + ncols % 2),
        np.arange(nrows // 2 - 1 + nrows % 2, (-nrows) // 2 - 1 + nrows % 2, step=-1),
    )

    return np.array([X, Y])

def Disk(n_points):
    """
    disk : generate a disk

    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.04.05, Brest
    Comments :

    Inputs : MANDATORY : nPointsX {int} : number of sampling points in the X direction

    Outputs : a disk
    """

    disk = np.zeros((n_points, n_points))

    [X, Y] = GetCartesianCoordinates(n_points)

    radial_coordinate = (X**2 + Y**2) ** 0.5
    disk[radial_coordinate < n_points / 2] = 1

    return disk