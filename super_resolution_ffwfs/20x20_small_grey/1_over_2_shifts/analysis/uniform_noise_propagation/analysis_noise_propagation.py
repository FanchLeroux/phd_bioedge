# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:46:18 2025

@author: fleroux
"""

# pylint: disable=undefined-variable

#%%

import pathlib
import sys
import platform
import dill

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from fanch.tools.save_load import save_vars, load_vars

from OOPAO.tools.displayTools import cl_plot

#%% Get parameter file

path_parameter_file = pathlib.Path(__file__).parent.parent.parent / "parameter_file.pkl"
load_vars(path_parameter_file, ['param'])

#%% path type compatibility issues

if platform.system() == 'Windows':
    temp = deepcopy(pathlib.PosixPath)
    pathlib.PosixPath = pathlib.WindowsPath
elif platform.system() == 'Linux':
    temp = deepcopy(pathlib.WindowsPath)
    pathlib.WindowsPath = pathlib.PosixPath

#%% Load objects computed in build_object_file.py 

# load_vars(param['path_object'] / pathlib.Path('object'+str(param['filename'])+'.pkl'), 
#           ['parameters_object', 'origin_object',\
#            'tel','atm', 'dm', 'ngs', 'M2C',\
#            'pyramid', 'pyramid_sr', 'pyramid_oversampled',\
#            'sbioedge', 'sbioedge_sr', 'sbioedge_oversampled',\
#            'gbioedge', 'gbioedge_sr', 'gbioedge_oversampled',\
#            'sgbioedge', 'sgbioedge_sr','sgbioedge_oversampled'])

#%% Load OOPAO calibration objects computed in make_calibration_file.py

path_calibration = param['path_calibration']

#%%

dill.load_session(path_calibration / pathlib.Path('calibration_pyramid'+param['filename']+'.pkl'))
dill.load_session(path_calibration / pathlib.Path('calibration_gbioedge'+param['filename']+'.pkl'))
dill.load_session(path_calibration / pathlib.Path('calibration_sbioedge'+param['filename']+'.pkl'))
dill.load_session(path_calibration / pathlib.Path('calibration_sgbioedge'+param['filename']+'.pkl'))

#%%

param = get_parameters() # to get current machine path (cluster or laptop issue ...)

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

# ------------------ sgbioedge ---------------------- #

# extract singular values

singular_values_sgbioedge = calib_sgbioedge.eigenValues
singular_values_sgbioedge_sr = calib_sgbioedge_sr.eigenValues
singular_values_sgbioedge_oversampled = calib_sgbioedge_oversampled.eigenValues

# Calibration matrix truncation, inversion and noise propagation computation

R_gbiedge_oversampled = np.linalg.pinv(calib_sgbioedge_oversampled.D)
noise_propagation_sgbioedge_oversampled = np.diag(R_gbiedge_oversampled @ R_gbiedge_oversampled.T)/calib_sgbioedge_oversampled.D.shape[0]

noise_propagation_sgbioedge = []
noise_propagation_sgbioedge_sr = []

for n_modes in param['list_modes_to_keep']:

    R = np.linalg.pinv(calib_sgbioedge.D[:,:n_modes])
    R_sr = np.linalg.pinv(calib_sgbioedge_sr.D[:,:n_modes])

    noise_propagation_sgbioedge.append(np.diag(R @ R.T)/calib_sgbioedge.D.shape[0])
    noise_propagation_sgbioedge_sr.append(np.diag(R_sr @ R_sr.T)/calib_sgbioedge_sr.D.shape[0])

    
#%% remove all the variables we do not want to save in the pickle file provided by dill.load_session

parameters = deepcopy(param)

for obj in dir():
    #checking for built-in variables/functions
    if not obj in ['parameters',\
                   'singular_values_pyramid','singular_values_pyramid_sr', 'singular_values_pyramid_oversampled',\
                   'singular_values_sbioedge','singular_values_sbioedge_sr', 'singular_values_sbioedge_oversampled',\
                   'singular_values_gbioedge','singular_values_gbioedge_sr', 'singular_values_gbioedge_oversampled',\
                   'singular_values_sgbioedge','singular_values_sgbioedge_sr', 'singular_values_sgbioedge_oversampled',\
                   'noise_propagation_pyramid','noise_propagation_pyramid_sr', 'noise_propagation_pyramid_oversampled',\
                   'noise_propagation_sbioedge','noise_propagation_sbioedge_sr', 'noise_propagation_sbioedge_oversampled',\
                   'noise_propagation_sgbioedge','noise_propagation_sgbioedge_sr', 'noise_propagation_sgbioedge_oversampled',\
                   'noise_propagation_gbioedge','noise_propagation_gbioedge_sr', 'noise_propagation_gbioedge_oversampled',\
                   'get_parameters', 'dill', 'pathlib']\
    and not obj.startswith('__'):
        #deleting the said obj, since a user-defined function
        del globals()[obj]
del obj

#%% save all variables

origin = str(pathlib.Path(__file__)) # keep a trace of where the saved objects come from

#%%

dill.dump_session(pathlib.Path(__file__).parent / "data_analysis" /pathlib.Path('analysis_noise_propagation'+
                                                                                str(parameters['filename'])+'.pkl'))
