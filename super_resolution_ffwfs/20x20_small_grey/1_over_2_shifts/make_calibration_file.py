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

#%%

do_pyramid = True
do_sbioedge = True
do_gbioedge = True
do_sgbioedge = True

#%% path type compatibility issues

if platform.system() == 'Windows':
    temp = deepcopy(pathlib.PosixPath)
    pathlib.PosixPath = pathlib.WindowsPath
elif platform.system() == 'Linux':
    temp = deepcopy(pathlib.WindowsPath)
    pathlib.WindowsPath = pathlib.PosixPath


#%% ------------------------- Make pyramid calibrations ------------------------------------

if do_pyramid:

    # Import parameter file
    param = get_parameters()
    
    # Load objects computed in build_object_file.py
    dill.load_session(param['path_object'] / pathlib.Path('object'+str(param['filename'])+'.pkl'))
    
    #%%
    calib_pyramid = InteractionMatrix(ngs, atm, tel, dm, pyramid, M2C = M2C, 
                                      stroke = param['stroke'], single_pass=param['single_pass'], 
                                      display=True)
    calib_pyramid_sr = InteractionMatrix(ngs, atm, tel, dm, pyramid_sr, M2C = M2C, 
                                          stroke = param['stroke'], single_pass=param['single_pass'], 
                                          display=True)
    calib_pyramid_oversampled = InteractionMatrix(ngs, atm, tel, dm, pyramid_oversampled, M2C = M2C, 
                                                  stroke = param['stroke'], single_pass=param['single_pass'], 
                                                  display=True)
    
    #%% remove all the variables we do not want to save in the pickle file provided by dill.load_session
    
    parameters = deepcopy(param)
    
    for obj in dir():
        #checking for built-in variables/functions
        if not obj in ['parameters',\
                       'calib_pyramid', 'calib_pyramid_sr', 'calib_pyramid_oversampled',\
                       'pathlib', 'dill', 'InteractionMatrix', 'get_parameters']\
        and not obj.startswith('_'): 
            #deleting the said obj, since a user-defined function
            del globals()[obj]
    del obj
    
    #%% save all variables
    
    origin = str(pathlib.Path(__file__)) # keep a trace of where the saved objects come from
    
    dill.dump_session(parameters['path_calibration'] / pathlib.Path('calibration_pyramid'+str(parameters['filename'])+'.pkl'))


#%% -------------------- Make sbioedge calibrations --------------------------

if do_sbioedge:

    param = get_parameters()
    
    dill.load_session(param['path_object'] / pathlib.Path('object'+str(param['filename'])+'.pkl'))
    
    #%%
    calib_sbioedge = InteractionMatrix(ngs, atm, tel, dm, sbioedge, M2C = M2C, 
                                      stroke = param['stroke'], single_pass=param['single_pass'], display=True)
    calib_sbioedge_sr = InteractionMatrix(ngs, atm, tel, dm, sbioedge_sr, M2C = M2C, 
                                          stroke = param['stroke'], single_pass=param['single_pass'], display=True)
    calib_sbioedge_oversampled = InteractionMatrix(ngs, atm, tel, dm, sbioedge_oversampled, M2C = M2C, 
                                                  stroke = param['stroke'], single_pass=param['single_pass'], display=True)
    
    #%% remove all the variables we do not want to save in the pickle file provided by dill.load_session
    
    parameters = deepcopy(param)
    
    for obj in dir():
        #checking for built-in variables/functions
        if not obj in ['parameters',\
                       'calib_sbioedge', 'calib_sbioedge_sr', 'calib_sbioedge_oversampled',\
                       'pathlib', 'dill', 'InteractionMatrix', 'get_parameters']\
        and not obj.startswith('_'): 
            #deleting the said obj, since a user-defined function
            del globals()[obj]
    del obj
    
    #%% save all variables
    
    origin = str(pathlib.Path(__file__)) # keep a trace of where the saved objects come from
    
    dill.dump_session(parameters['path_calibration'] / pathlib.Path('calibration_sbioedge'+str(parameters['filename'])+'.pkl'))


#%% -------------------- Make gbioedge calibrations -----------------------

if do_gbioedge:

    param = get_parameters()
    
    dill.load_session(param['path_object'] / pathlib.Path('object'+str(param['filename'])+'.pkl'))
    
    #%%
    calib_gbioedge = InteractionMatrix(ngs, atm, tel, dm, gbioedge, M2C = M2C, 
                                      stroke = param['stroke'], single_pass=param['single_pass'], display=True)
    calib_gbioedge_sr = InteractionMatrix(ngs, atm, tel, dm, gbioedge_sr, M2C = M2C, 
                                          stroke = param['stroke'], single_pass=param['single_pass'], display=True)
    calib_gbioedge_oversampled = InteractionMatrix(ngs, atm, tel, dm, gbioedge_oversampled, M2C = M2C, 
                                                  stroke = param['stroke'], single_pass=param['single_pass'], display=True)
    
    #%% remove all the variables we do not want to save in the pickle file provided by dill.load_session
    
    parameters = deepcopy(param)
    
    for obj in dir():
        #checking for built-in variables/functions
        if not obj in ['parameters',\
                       'calib_gbioedge', 'calib_gbioedge_sr', 'calib_gbioedge_oversampled',\
                       'pathlib', 'dill', 'InteractionMatrix', 'get_parameters']\
        and not obj.startswith('_'): 
            #deleting the said obj, since a user-defined function
            del globals()[obj]
    del obj
    
    #%% save all variables
    
    origin = str(pathlib.Path(__file__)) # keep a trace of where the saved objects come from
    
    dill.dump_session(parameters['path_calibration'] / pathlib.Path('calibration_gbioedge'+str(parameters['filename'])+'.pkl'))


#%% -------------------- Make sgbioedge calibrations -----------------------

if do_sgbioedge:

    param = get_parameters()
    
    dill.load_session(param['path_object'] / pathlib.Path('object'+str(param['filename'])+'.pkl'))
    
    #%%
    calib_sgbioedge = InteractionMatrix(ngs, atm, tel, dm, sgbioedge, M2C = M2C, 
                                      stroke = param['stroke'], single_pass=param['single_pass'], display=True)
    calib_sgbioedge_sr = InteractionMatrix(ngs, atm, tel, dm, sgbioedge_sr, M2C = M2C, 
                                          stroke = param['stroke'], single_pass=param['single_pass'], display=True)
    calib_sgbioedge_oversampled = InteractionMatrix(ngs, atm, tel, dm, sgbioedge_oversampled, M2C = M2C, 
                                                  stroke = param['stroke'], single_pass=param['single_pass'], display=True)
    
    #%% remove all the variables we do not want to save in the pickle file provided by dill.load_session
    
    parameters = deepcopy(param)
    
    for obj in dir():
        #checking for built-in variables/functions
        if not obj in ['parameters',\
                       'calib_sgbioedge', 'calib_sgbioedge_sr', 'calib_sgbioedge_oversampled',\
                       'pathlib', 'dill', 'InteractionMatrix', 'get_parameters']\
        and not obj.startswith('_'): 
            #deleting the said obj, since a user-defined function
            del globals()[obj]
    del obj
    
    #%% save all variables
    
    origin = str(pathlib.Path(__file__)) # keep a trace of where the saved objects come from
    
    dill.dump_session(parameters['path_calibration'] / pathlib.Path('calibration_sgbioedge'+str(parameters['filename'])+'.pkl'))

