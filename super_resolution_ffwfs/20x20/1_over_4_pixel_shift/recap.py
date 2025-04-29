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

#%% Define paths

path = pathlib.Path(__file__).parent
path_data = path.parent.parent.parent.parent / "phd_bioedge_data" / pathlib.Path(*path.parts[-3:]) # could be done better

#%% Get parameter file

path_parameter_file = path_data / "parameter_file.pkl"
load_vars(path_parameter_file, ['param'])

#%% path type compatibility issues

if platform.system() == 'Windows':
    temp = deepcopy(pathlib.PosixPath)
    pathlib.PosixPath = pathlib.WindowsPath
elif platform.system() == 'Linux':
    temp = deepcopy(pathlib.WindowsPath)
    pathlib.WindowsPath = pathlib.PosixPath
    
#%% load objects

load_vars(param['path_object'] / pathlib.Path('all_objects'+str(param['filename'])+'.pkl'))

#%% load calibrations

load_vars(param['path_calibration'] / pathlib.Path('calibration_pyramid'+param['filename']+'.pkl'))
load_vars(param['path_calibration'] / pathlib.Path('calibration_gbioedge'+param['filename']+'.pkl'))
load_vars(param['path_calibration'] / pathlib.Path('calibration_sbioedge'+param['filename']+'.pkl'))
load_vars(param['path_calibration'] / pathlib.Path('calibration_sgbioedge'+param['filename']+'.pkl'))

#%% load analysis results

path_results_closed_loop = pathlib.Path(__file__).parent\
    / 'analysis' / 'closed_loop' / 'data_analysis' / pathlib.Path('analysis_closed_loop' + param['filename'])

load_vars(path_results_closed_loop)

#%% save all variables

origin_recap = str(pathlib.Path(__file__)) # keep a trace of where the saved objects come from

#%%

save_vars(param['path_calibration'].parent / pathlib.Path('recap'+str(param['filename'])+'.pkl'))

#%% load recap

load_vars(param['path_calibration'].parent / pathlib.Path('recap'+str(param['filename'])+'.pkl'), load_session=True)

