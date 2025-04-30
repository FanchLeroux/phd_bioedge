# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:45:55 2025

@author: fleroux²
"""

# pylint: disable=undefined-variable
# pylint: disable=undefined-loop-variable

import platform
import pathlib

from fanch.tools.save_load import save_vars, load_vars

from copy import deepcopy

from OOPAO.calibration.InteractionMatrix import InteractionMatrix

#%% Define paths

path = pathlib.Path(__file__).parent
path_data = path.parent.parent.parent.parent / "phd_bioedge_data" / pathlib.Path(*path.parts[-3:]) # could be done better

#%% Get parameter file

path_parameter_file = path_data / "parameter_file.pkl"
load_vars(path_parameter_file, ['param'])

#%% Load objects computed in build_object_file.py 

load_vars(pathlib.Path(param['path_object']) / pathlib.Path('all_objects'+str(param['filename'])+'.pkl'), 
          ['parameters_object', 'origin_object',\
           'tel','atm', 'dm', 'ngs', 'M2C',\
           'pyramid', 'pyramid_sr', 'pyramid_oversampled',\
           'sbioedge', 'sbioedge_sr', 'sbioedge_oversampled',\
           'gbioedge', 'gbioedge_sr', 'gbioedge_oversampled',\
           'sgbioedge', 'sgbioedge_sr','sgbioedge_oversampled'])

#%% ------------------------- Make calibrations ------------------------------------

parameters_calib = deepcopy(param)
origin_calib = str(pathlib.Path(__file__)) # keep a trace of where the saved objects come from

#%% -------------------- Make pyramid calibrations --------------------------

calib_pyramid = InteractionMatrix(ngs, atm, tel, dm, pyramid, M2C = M2C, 
                                  stroke = param['stroke'], single_pass=param['single_pass'], 
                                  display=True)
calib_pyramid_sr = InteractionMatrix(ngs, atm, tel, dm, pyramid_sr, M2C = M2C, 
                                      stroke = param['stroke'], single_pass=param['single_pass'], 
                                      display=True)
calib_pyramid_oversampled = InteractionMatrix(ngs, atm, tel, dm, pyramid_oversampled, M2C = M2C, 
                                              stroke = param['stroke'], single_pass=param['single_pass'], 
                                              display=True)

#%% save pyramid calibrations

save_vars(pathlib.Path(param['path_calibration']) / pathlib.Path('calibration_pyramid'+str(param['filename'])+'.pkl'),
          ['parameters_calib', 'origin_calib',\
           'calib_pyramid', 'calib_pyramid_sr', 'calib_pyramid_oversampled'])


#%% -------------------- Make sbioedge calibrations --------------------------

calib_sbioedge = InteractionMatrix(ngs, atm, tel, dm, sbioedge, M2C = M2C, 
                                  stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_sbioedge_sr = InteractionMatrix(ngs, atm, tel, dm, sbioedge_sr, M2C = M2C, 
                                      stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_sbioedge_oversampled = InteractionMatrix(ngs, atm, tel, dm, sbioedge_oversampled, M2C = M2C, 
                                              stroke = param['stroke'], single_pass=param['single_pass'], display=True)

#%% save sbioedge calibrations

save_vars(pathlib.Path(param['path_calibration']) / pathlib.Path('calibration_sbioedge'+str(param['filename'])+'.pkl'),
          ['parameters_calib', 'origin_calib',\
           'calib_sbioedge', 'calib_sbioedge_sr', 'calib_sbioedge_oversampled'])

#%% -------------------- Make gbioedge calibrations -----------------------


calib_gbioedge = InteractionMatrix(ngs, atm, tel, dm, gbioedge, M2C = M2C, 
                                  stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_gbioedge_sr = InteractionMatrix(ngs, atm, tel, dm, gbioedge_sr, M2C = M2C, 
                                      stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_gbioedge_oversampled = InteractionMatrix(ngs, atm, tel, dm, gbioedge_oversampled, M2C = M2C, 
                                              stroke = param['stroke'], single_pass=param['single_pass'], display=True)

#%% save gbioedge calibrations

save_vars(pathlib.Path(param['path_calibration']) / pathlib.Path('calibration_gbioedge'+str(param['filename'])+'.pkl'),
          ['parameters_calib', 'origin_calib',\
           'calib_gbioedge', 'calib_gbioedge_sr', 'calib_gbioedge_oversampled'])
    
#%% -------------------- Make sgbioedge calibrations -----------------------

calib_sgbioedge = InteractionMatrix(ngs, atm, tel, dm, sgbioedge, M2C = M2C, 
                                  stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_sgbioedge_sr = InteractionMatrix(ngs, atm, tel, dm, sgbioedge_sr, M2C = M2C, 
                                      stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_sgbioedge_oversampled = InteractionMatrix(ngs, atm, tel, dm, sgbioedge_oversampled, M2C = M2C, 
                                              stroke = param['stroke'], single_pass=param['single_pass'], display=True)
    
#%% save sgbioedge calibrations

save_vars(pathlib.Path(param['path_calibration']) / pathlib.Path('calibration_sgbioedge'+str(param['filename'])+'.pkl'),
          ['parameters_calib', 'origin_calib',\
           'calib_sgbioedge', 'calib_sgbioedge_sr', 'calib_sgbioedge_oversampled'])
    
#%% save all calibrations

save_vars(pathlib.Path(param['path_calibration']) / pathlib.Path('all_calibrations'+str(param['filename'])+'.pkl'),
          ['parameters_calib', 'origin_calib',\
           'calib_pyramid', 'calib_pyramid_sr', 'calib_pyramid_oversampled',\
           'calib_sbioedge', 'calib_sbioedge_sr', 'calib_sbioedge_oversampled',\
           'calib_gbioedge', 'calib_gbioedge_sr', 'calib_gbioedge_oversampled',\
           'calib_sgbioedge', 'calib_sgbioedge_sr', 'calib_sgbioedge_oversampled'])