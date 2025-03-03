# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:08:49 2025

@author: fleroux
"""

# pylint: disable=undefined-variable

import platform
import pathlib

import dill

from copy import deepcopy
from parameter_file import get_parameters

from OOPAO.calibration.InteractionMatrix import InteractionMatrix

#%% Import parameter file

param = get_parameters()

#%% Load everything

if platform.system() == 'Windows':
    temp = deepcopy(pathlib.PosixPath)
    pathlib.PosixPath = pathlib.WindowsPath

#%% load objects
dill.load_session(param['path_object'] / pathlib.Path('object'+str(param['filename'])+'.pkl'))

#%%
param = get_parameters()

#%% load calibrations

path_calibration = param['path_calibration']

dill.load_session(path_calibration / pathlib.Path('calibration_pyramid'+param['filename']+'.pkl'))
dill.load_session(path_calibration / pathlib.Path('calibration_gbioedge'+param['filename']+'.pkl'))
dill.load_session(path_calibration / pathlib.Path('calibration_sbioedge'+param['filename']+'.pkl'))

#%%
param = get_parameters()

#%% load analysis
dill.load_session(param['path_analysis'] / pathlib.Path('analysis'+str(param['filename'])+'.pkl'))

#%%
param = get_parameters()

#%%
if platform.system() == 'Windows':
    pathlib.PosixPath = temp
    
#%% save all variables

origin = str(pathlib.Path(__file__)) # keep a trace of where the saved objects come from

dill.dump_session(param['path_calibration'].parent / pathlib.Path('recap'+str(param['filename'])+'.pkl'))

#%% load recap

dill.load_session(param['path_calibration'].parent / pathlib.Path('recap'+str(param['filename'])+'.pkl'))
