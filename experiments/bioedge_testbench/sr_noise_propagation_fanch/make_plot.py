# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:29:55 2025

@author: fleroux
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:46:18 2025

@author: fleroux
"""

# pylint: disable=undefined-variable

import platform
import pathlib

from copy import deepcopy

import dill

from parameter_file import get_parameters

import numpy as np
import matplotlib.pyplot as plt

#%% Import parameter file

param = get_parameters()

#%% Load OOPAO calibration objects computed in analysis_file.py

if platform.system() == 'Windows':
    temp = deepcopy(pathlib.PosixPath)
    pathlib.PosixPath = pathlib.WindowsPath

dill.load_session(param['path_analysis'] / pathlib.Path('analysis'+str(param['filename'])+'.pkl'))

if platform.system() == 'Windows':
    pathlib.PosixPath = temp

param = get_parameters()

#%% Plots

# SVD - Normalized Eigenvalues  

    # --------------- pyramid --------------------

plt.figure()
plt.semilogy(singular_values_pyramid/singular_values_pyramid.max(), 'b', label='no SR')
plt.semilogy(singular_values_pyramid_sr/singular_values_pyramid_sr.max(), 'r', label='SR')
plt.semilogy(singular_values_pyramid_oversampled/singular_values_pyramid_oversampled.max(), 'c', label='oversampled')
plt.title('singular_values_pyramid, '+str(param['n_subaperture'])+' subapertures, ' + param['modal_basis'] + ' modes used, 0.25 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')


    # --------------- gbioedge --------------------

plt.figure()
plt.semilogy(singular_values_gbioedge/singular_values_gbioedge.max(), 'b', label='no SR')
plt.semilogy(singular_values_gbioedge_sr/singular_values_gbioedge_sr.max(), 'r', label='SR')
plt.semilogy(singular_values_gbioedge_oversampled/singular_values_gbioedge_oversampled.max(), 'c', label='oversampled')
plt.title('singular_values_gbioedge, '+str(param['n_subaperture'])+' subapertures, ' + param['modal_basis'] + ' modes used, 0.25 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')

    # --------------- sbioedge --------------------

plt.figure()
plt.semilogy(singular_values_sbioedge/singular_values_sbioedge.max(), 'b', label='no SR')
plt.semilogy(singular_values_sbioedge_sr/singular_values_sbioedge_sr.max(), '--r', label='SR')
plt.semilogy(singular_values_sbioedge_oversampled/singular_values_sbioedge_oversampled.max(), 'c', label='oversampled')
plt.title('singular_values_sbioedge, '+str(param['n_subaperture'])+' subapertures, ' + param['modal_basis'] + ' modes used, 0.25 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')