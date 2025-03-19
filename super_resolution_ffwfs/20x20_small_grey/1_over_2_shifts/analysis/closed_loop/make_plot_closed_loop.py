# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:29:55 2025

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

from fanch.tools.save_load import load_vars

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

#%% Load objects computed inanalysis_closed_loop.py 

path_analysis_data = pathlib.Path(__file__).parent / 'data_analysis'

load_vars(path_analysis_data / pathlib.Path('analysis_closed_loop' + param['filename']), 
          ['parameters_analysis',\
           'total_gbioedge', 'residual_gbioedge', 'strehl_gbioedge',\
           'residual_gbioedge_sr', 'strehl_gbioedge_sr',\
           'residual_gbioedge_oversampled', 'strehl_gbioedge_oversampled',\
           'residual_sgbioedge', 'strehl_sgbioedge',\
           'residual_sgbioedge_sr', 'strehl_sgbioedge_sr',\
           'residual_sgbioedge_oversampled', 'strehl_sgbioedge_oversampled'])
    
    
#%% path plots

path_plots = pathlib.Path(__file__).parent / 'plots'
    
#%% plots

for k in range(len(param['seeds'])):

    seed = param['seeds'][k]
    
    plt.figure()
    plt.plot(1e9*total_gbioedge[k, :], 'g', label='Open loop',)
    plt.plot(residual_gbioedge[k, :], 'b', label=str(param['n_subaperture'])+'x'+
              str(param['n_subaperture'])+', no SR '+str(param['n_modes_to_show'] )+' modes')
    plt.plot(residual_gbioedge_sr[k, :], 'r', label=str(param['n_subaperture'])+'x'+
              str(param['n_subaperture'])+', SR ' +str(param['n_modes_to_show_sr'] )+' modes shown\n'
              +str(param['n_modes_to_control_sr'])+' modes controlled')
    plt.plot(residual_gbioedge_oversampled[k, :], 'k', label=str(2*param['n_subaperture'])+'x'+
              str(2*param['n_subaperture'])+', '+str(param['n_modes_to_show_oversampled'] )+' modes')
    plt.title('Closed Loop residuals gbioedge\n'
              'loop frequency : '+str(np.round(1/param['sampling_time']/1e3, 1))+'kHz\n'
              'Telescope diameter: '+str(param['diameter']) + ' m\n'
              'Half grey width : '+str(param['modulation'])+' lambda/D\n'
              'seed : '+str(seed))
    plt.xlabel('Iteration')
    plt.ylabel('residuals (nm)')
    plt.legend()
    plt.savefig(path_plots / "gbioedge_closed_loop_severals_seeds.png", bbox_inches = 'tight')
    
    plt.figure()
    plt.plot(1e9*total_gbioedge[k, :], 'g', label='Open loop')
    plt.plot(residual_sgbioedge[k, :], 'b', label=str(param['n_subaperture'])+'x'+
              str(param['n_subaperture'])+', no SR '+str(param['n_modes_to_show'] )+' modes')
    plt.plot(residual_sgbioedge_sr[k, :], 'r', label=str(param['n_subaperture'])+'x'+
              str(param['n_subaperture'])+', SR ' +str(param['n_modes_to_show_sr'] )+' modes shown\n'
              +str(param['n_modes_to_control_sr'])+' modes controlled')
    plt.plot(residual_sgbioedge_oversampled[k, :], 'k', label=str(2*param['n_subaperture'])+'x'+
              str(2*param['n_subaperture'])+', '+str(param['n_modes_to_show_oversampled'] )+' modes')
    plt.title('Closed Loop residuals sgbioedge\n'
              'loop frequency : '+str(np.round(1/param['sampling_time']/1e3, 1))+'kHz\n'
              'Telescope diameter: '+str(param['diameter']) + ' m\n'
              'Half grey width : '+str(param['modulation'])+' lambda/D\n'
              'seed : '+str(seed))
    plt.xlabel('Iteration')
    plt.ylabel('residuals (nm)')
    plt.legend()
    plt.savefig(path_plots / "sgbioedge_closed_loop_severals_seeds.png", bbox_inches = 'tight')
