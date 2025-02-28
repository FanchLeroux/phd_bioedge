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

#%% Import parameter file

param = get_parameters()

#%% Load OOPAO calibration objects computed in make_calibration_file.py

if platform.system() == 'Windows':
    temp = deepcopy(pathlib.PosixPath)
    pathlib.PosixPath = pathlib.WindowsPath

dill.load_session(param['path_calibration'] / pathlib.Path('calibration'+str(param['filename'])+'.pkl'))

if platform.system() == 'Windows':
    pathlib.PosixPath = temp

param = get_parameters()

#%% Analysis

    # ------------------ pyramid ---------------------- #

# extract singular values

singular_values_pyramid = calib_pyramid.eigenValues
singular_values_pyramid_sr = calib_pyramid_sr.eigenValues
singular_values_pyramid_oversampled = calib_pyramid_oversampled.eigenValues

# Calibration matrix truncation, inversion and noise propagation computation

R_gbiedge_oversampled = np.linalg.pinv(calib_pyramid_oversampled.D)
noise_propagation_pyramid_oversampled = np.diag(R_gbiedge_oversampled @ R_gbiedge_oversampled.T)/calib_pyramid_oversampled.D.shape[0]

noise_propagation_pyramid = []
noise_propagation_pyramid_sr = []

for n_modes in param['list_modes_to_keep']:

    R = np.linalg.pinv(calib_pyramid.D[:,:n_modes])
    R_sr = np.linalg.pinv(calib_pyramid_sr.D[:,:n_modes])

    noise_propagation_pyramid.append(np.diag(R @ R.T)/calib_pyramid.D.shape[0])
    noise_propagation_pyramid_sr.append(np.diag(R_sr @ R_sr.T)/calib_pyramid_sr.D.shape[0])

    
    # ------------------ gbioedge ---------------------- #

# extract singular values

singular_values_gbioedge = calib_gbioedge.eigenValues
singular_values_gbioedge_sr = calib_gbioedge_sr.eigenValues
singular_values_gbioedge_oversampled = calib_gbioedge_oversampled.eigenValues

# Calibration matrix truncation, inversion and noise propagation computation

R_gbiedge_oversampled = np.linalg.pinv(calib_gbioedge_oversampled.D)
noise_propagation_gbioedge_oversampled = np.diag(R_gbiedge_oversampled @ R_gbiedge_oversampled.T)/calib_gbioedge_oversampled.D.shape[0]

noise_propagation_gbioedge = []
noise_propagation_gbioedge_sr = []

for n_modes in param['list_modes_to_keep']:

    R = np.linalg.pinv(calib_gbioedge.D[:,:n_modes])
    R_sr = np.linalg.pinv(calib_gbioedge_sr.D[:,:n_modes])

    noise_propagation_gbioedge.append(np.diag(R @ R.T)/calib_gbioedge.D.shape[0])
    noise_propagation_gbioedge_sr.append(np.diag(R_sr @ R_sr.T)/calib_gbioedge_sr.D.shape[0])
    

    # ------------------ sbioedge ---------------------- #

# extract singular values

singular_values_sbioedge = calib_sbioedge.eigenValues
singular_values_sbioedge_sr = calib_sbioedge_sr.eigenValues
singular_values_sbioedge_oversampled = calib_sbioedge_oversampled.eigenValues

# Calibration matrix truncation, inversion and noise propagation computation

R_gbiedge_oversampled = np.linalg.pinv(calib_sbioedge_oversampled.D)
noise_propagation_sbioedge_oversampled = np.diag(R_gbiedge_oversampled @ R_gbiedge_oversampled.T)/calib_sbioedge_oversampled.D.shape[0]

noise_propagation_sbioedge = []
noise_propagation_sbioedge_sr = []

for n_modes in param['list_modes_to_keep']:

    R = np.linalg.pinv(calib_sbioedge.D[:,:n_modes])
    R_sr = np.linalg.pinv(calib_sbioedge_sr.D[:,:n_modes])

    noise_propagation_sbioedge.append(np.diag(R @ R.T)/calib_sbioedge.D.shape[0])
    noise_propagation_sbioedge_sr.append(np.diag(R_sr @ R_sr.T)/calib_sbioedge_sr.D.shape[0])
    
#%% remove all the variables we do not want to save in the pickle file provided by dill.load_session

for obj in dir():
    #checking for built-in variables/functions
    if not (obj.startswith('noise_propagation_') or obj.startswith('singular_values_') or obj.startswith('__') or obj.startswith('param')\
            or obj.startswith('pathlib') or obj.startswith('dill')):
        #deleting the said obj, since a user-defined function
        del globals()[obj]
del obj

#%% save all variables

origin = str(pathlib.Path(__file__)) # keep a trace of where the saved objects come from

dill.dump_session(param['path_analysis'] / pathlib.Path('analysis'+str(param['filename'])+'.pkl'))
