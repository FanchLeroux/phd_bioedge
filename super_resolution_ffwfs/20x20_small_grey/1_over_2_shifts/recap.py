# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:08:49 2025

@author: fleroux
"""

# pylint: disable=undefined-variable

import platform
import pathlib

from copy import deepcopy

from fanch.tools.save_load import save_vars, load_vars

#%% path type compatibility issues

if platform.system() == 'Windows':
    temp = deepcopy(pathlib.PosixPath)
    pathlib.PosixPath = pathlib.WindowsPath
elif platform.system() == 'Linux':
    temp = deepcopy(pathlib.WindowsPath)
    pathlib.WindowsPath = pathlib.PosixPath

#%% Get parameter file

path_parameter_file = pathlib.Path(__file__).parent / "parameter_file.pkl"
load_vars(path_parameter_file, ['param'])

#%% load objects

load_vars(param['path_object'] / pathlib.Path('object'+str(param['filename'])+'.pkl'))

#%% load calibrations

path_calibration = param['path_calibration']

load_vars(path_calibration / pathlib.Path('calibration_pyramid'+param['filename']+'.pkl'))
load_vars(path_calibration / pathlib.Path('calibration_gbioedge'+param['filename']+'.pkl'))
load_vars(path_calibration / pathlib.Path('calibration_sbioedge'+param['filename']+'.pkl'))
load_vars(path_calibration / pathlib.Path('calibration_sgbioedge'+param['filename']+'.pkl'))

#%% save all variables

origin_recap = str(pathlib.Path(__file__)) # keep a trace of where the saved objects come from

#%%

save_vars(param['path_calibration'].parent / pathlib.Path('recap'+str(param['filename'])+'.pkl'))

#%% load recap

load_vars(param['path_calibration'].parent / pathlib.Path('recap'+str(param['filename'])+'.pkl'), load_session=True)

