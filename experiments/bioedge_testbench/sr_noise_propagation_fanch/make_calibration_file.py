# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:45:55 2025

@author: fleroux
"""

# pylint: disable=undefined-variable
# pylint: disable=undefined-loop-variable


from pathlib import Path

import dill

from parameter_file import get_parameters

from OOPAO.calibration.InteractionMatrix import InteractionMatrix

#%% Import parameter file

param = get_parameters()

#%% Load objects computed in build_object_file.py

dill.load_session(param['path_object'] / Path('object'+str(param['filename'])+'.pkl'))

#%% Make calibrations

calib_pyramid = InteractionMatrix(ngs, atm, tel, dm, pyramid, M2C = M2C, 
                                  stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_pyramid_sr = InteractionMatrix(ngs, atm, tel, dm, pyramid_sr, M2C = M2C, 
                                     stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_pyramid_oversampled = InteractionMatrix(ngs, atm, tel, dm, pyramid_oversampled, M2C = M2C, 
                                              stroke = param['stroke'], single_pass=param['single_pass'], display=True)

calib_gbioedge = InteractionMatrix(ngs, atm, tel, dm, gbioedge, M2C = M2C, 
                                   stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_gbioedge_sr = InteractionMatrix(ngs, atm, tel, dm, gbioedge_sr, M2C = M2C, 
                                   stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_gbioedge_oversampled = InteractionMatrix(ngs, atm, tel, dm, gbioedge_oversampled, M2C = M2C, 
                                   stroke = param['stroke'], single_pass=param['single_pass'], display=True)

calib_sbioedge = InteractionMatrix(ngs, atm, tel, dm, sbioedge, M2C = M2C, 
                                   stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_sbioedge_sr = InteractionMatrix(ngs, atm, tel, dm, sbioedge_sr, M2C = M2C, 
                                   stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_sbioedge_oversampled = InteractionMatrix(ngs, atm, tel, dm, sbioedge_oversampled, M2C = M2C, 
                                   stroke = param['stroke'], single_pass=param['single_pass'], display=True)

#%% remove all the variables we do not want to save in the pickle file provided by dill.load_session

for obj in dir():
    #checking for built-in variables/functions
    if not (obj.startswith('calib') or obj.startswith('__') or obj.startswith('param')\
            or obj.startswith('pathlib') or obj.startswith('dill')):
        #deleting the said obj, since a user-defined function
        del globals()[obj]
del obj

#%% save all variables

origin = str(Path(__file__)) # keep a trace of where the saved objects come from

dill.dump_session(param['path_calibration'] / Path('calibration'+str(param['filename'])+'.pkl'))
