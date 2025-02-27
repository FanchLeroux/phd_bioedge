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


#%% Import parameter file

param = get_parameters()

#%% Load OOPAO calibration objects computed in make_calibration_file.py

if platform.system() == 'Windows':
    temp = deepcopy(pathlib.PosixPath)
    pathlib.PosixPath = pathlib.WindowsPath

dill.load_session(param['path_calibration'] / pathlib.Path('calibration'+str(param['filename'])+'.pkl'))

if platform.system() == 'Windows':
    pathlib.PosixPath = temp

#%% Analysis

# extract singular values

singular_values_gbioedge = calib_gbioedge.eigenValues

# Calibration matrix truncation, inversion and noise propagation computation

R_gbiedge_oversampled = np.linalg.pinv(calib_bioedge_oversampled.D)
noise_propagation_gbioedge_oversampled = np.diag(R_gbiedge_oversampled @ R_gbiedge_oversampled.T)/wfs_oversampled.nSignal

for n_modes in range(param['list_modes_to_keep'].shape[0]):

    n_modes_no_sr = n_modes_list[n_modes]
    n_modes_sr = n_modes_list[n_modes]

    R = np.linalg.pinv(calib.D[:,:n_modes_no_sr])
    R_sr = np.linalg.pinv(calib_sr.D[:,:n_modes_sr])

    noise_propagation_no_sr.append(np.diag(R @ R.T)/wfs.nSignal)
    noise_propagation_sr.append(np.diag(R_sr @ R_sr.T)/wfs.nSignal)
