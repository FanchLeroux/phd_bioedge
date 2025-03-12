# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:08:49 2025

@author: fleroux
"""

# pylint: disable=undefined-variable

import importlib.util
import sys

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

#%% import parameter file from any repository

# weird method from https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path

path_parameters = pathlib.Path(__file__).parent / "parameter_file.py"

spec = importlib.util.spec_from_file_location("get_parameters", path_parameters)
foo = importlib.util.module_from_spec(spec)
sys.modules["parameter_file"] = foo
spec.loader.exec_module(foo)

param = foo.get_parameters()

#%% load objects

#load_vars(param['path_object'] / pathlib.Path('object'+str(param['filename'])+'.pkl'))

#%% load calibrations

path_calibration = param['path_calibration']

#load_vars(path_calibration / pathlib.Path('calibration_pyramid'+param['filename']+'.pkl'))
#load_vars(path_calibration / pathlib.Path('calibration_gbioedge'+param['filename']+'.pkl'))
#load_vars(path_calibration / pathlib.Path('calibration_sbioedge'+param['filename']+'.pkl'))
load_vars(path_calibration / pathlib.Path('calibration_sgbioedge'+param['filename']+'.pkl'))

#%% save all variables

origin_recap = str(pathlib.Path(__file__)) # keep a trace of where the saved objects come from

#%%

save_vars(param['path_calibration'].parent / pathlib.Path('recap'+str(param['filename'])+'.pkl'), var_names=['calib_sgbioedge'])

#%% load recap

load_vars(param['path_calibration'].parent / pathlib.Path('recap'+str(param['filename'])+'.pkl'))
