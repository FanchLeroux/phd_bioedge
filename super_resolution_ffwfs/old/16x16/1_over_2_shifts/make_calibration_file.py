# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:45:55 2025

@author: fleroux
"""

# pylint: disable=undefined-variable
# pylint: disable=undefined-loop-variable

import platform
import pathlib

import dill

from copy import deepcopy
from parameter_file import get_parameters

from OOPAO.calibration.InteractionMatrix import InteractionMatrix

#%% Import parameter file

param = get_parameters()

#%% Load objects computed in build_object_file.py

if platform.system() == 'Windows':
    temp = deepcopy(pathlib.PosixPath)
    pathlib.PosixPath = pathlib.WindowsPath
elif platform.system() == 'Linux':
    temp = deepcopy(pathlib.WindowsPath)
    pathlib.WindowsPath = pathlib.PosixPath

dill.load_session(param['path_object'] / pathlib.Path('object'+str(param['filename'])+'.pkl'))

if platform.system() == 'Windows':
    pathlib.PosixPath = temp
elif platform.system() == 'Linux':
    pathlib.WindowsPath = temp

param = get_parameters()

#%% Make calibrations

calib_pyramid = InteractionMatrix(ngs, atm, tel, dm, pyramid, M2C = M2C, 
                                  stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_pyramid_sr = InteractionMatrix(ngs, atm, tel, dm, pyramid_sr, M2C = M2C, 
                                      stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_pyramid_oversampled = InteractionMatrix(ngs, atm, tel, dm, pyramid_oversampled, M2C = M2C, 
                                              stroke = param['stroke'], single_pass=param['single_pass'], display=True)

#%% remove all the variables we do not want to save in the pickle file provided by dill.load_session

for obj in dir():
    #checking for built-in variables/functions
    if not (obj.startswith('calib_pyramid') or obj.startswith('__') or obj.startswith('param')\
            or obj.startswith('pathlib') or obj.startswith('dill') or obj.startswith('InteractionMatrix')):
        #deleting the said obj, since a user-defined function
        del globals()[obj]
del obj

#%% save all variables

origin = str(pathlib.Path(__file__)) # keep a trace of where the saved objects come from

dill.dump_session(param['path_calibration'] / pathlib.Path('calibration_pyramid'+str(param['filename'])+'.pkl'))

#%% Make calibrations

dill.load_session(param['path_object'] / pathlib.Path('object'+str(param['filename'])+'.pkl'))

calib_gbioedge = InteractionMatrix(ngs, atm, tel, dm, gbioedge, M2C = M2C, 
                                  stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_gbioedge_sr = InteractionMatrix(ngs, atm, tel, dm, gbioedge_sr, M2C = M2C, 
                                      stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_gbioedge_oversampled = InteractionMatrix(ngs, atm, tel, dm, gbioedge_oversampled, M2C = M2C, 
                                              stroke = param['stroke'], single_pass=param['single_pass'], display=True)

#%% remove all the variables we do not want to save in the pickle file provided by dill.load_session

for obj in dir():
    #checking for built-in variables/functions
    if not (obj.startswith('calib_gbioedge') or obj.startswith('__') or obj.startswith('param')\
            or obj.startswith('pathlib') or obj.startswith('dill') or obj.startswith('InteractionMatrix')):
        #deleting the said obj, since a user-defined function
        del globals()[obj]
del obj

#%% save all variables

origin = str(pathlib.Path(__file__)) # keep a trace of where the saved objects come from

dill.dump_session(param['path_calibration'] / pathlib.Path('calibration_gbioedge'+str(param['filename'])+'.pkl'))

#%% Make calibrations

dill.load_session(param['path_object'] / pathlib.Path('object'+str(param['filename'])+'.pkl'))

calib_sbioedge = InteractionMatrix(ngs, atm, tel, dm, sbioedge, M2C = M2C, 
                                  stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_sbioedge_sr = InteractionMatrix(ngs, atm, tel, dm, sbioedge_sr, M2C = M2C, 
                                      stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_sbioedge_oversampled = InteractionMatrix(ngs, atm, tel, dm, sbioedge_oversampled, M2C = M2C, 
                                              stroke = param['stroke'], single_pass=param['single_pass'], display=True)

#%% remove all the variables we do not want to save in the pickle file provided by dill.load_session

for obj in dir():
    #checking for built-in variables/functions
    if not (obj.startswith('calib_sbioedge') or obj.startswith('__') or obj.startswith('param')\
            or obj.startswith('pathlib') or obj.startswith('dill') or obj.startswith('InteractionMatrix')):
        #deleting the said obj, since a user-defined function
        del globals()[obj]
del obj

#%% save all variables

origin = str(pathlib.Path(__file__)) # keep a trace of where the saved objects come from

dill.dump_session(param['path_calibration'] / pathlib.Path('calibration_sbioedge'+str(param['filename'])+'.pkl'))
