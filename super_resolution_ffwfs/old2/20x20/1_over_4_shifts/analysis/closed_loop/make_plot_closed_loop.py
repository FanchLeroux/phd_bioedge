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

from PIL import Image

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

load_vars(path_analysis_data / pathlib.Path('analysis_closed_loop' + param['filename']+'.pkl'), 
          ['parameters_analysis',\
           'total_gbioedge', 'residual_gbioedge', 'strehl_gbioedge',\
           'residual_gbioedge_sr', 'strehl_gbioedge_sr',\
           'residual_gbioedge_oversampled', 'strehl_gbioedge_oversampled',\
           'residual_sgbioedge', 'strehl_sgbioedge',\
           'residual_sgbioedge_sr', 'strehl_sgbioedge_sr',\
           'residual_sgbioedge_oversampled', 'strehl_sgbioedge_oversampled'])
    
param = parameters_analysis
    
#%% path plots

path_plots = pathlib.Path(__file__).parent / 'plots'
    
#%% plots

images = []

ylim = total_gbioedge.max() + 10

for k in range(len(param['seeds'])):

    seed = param['seeds'][k]
    
    plt.figure()
    plt.plot(total_gbioedge[k, :], 'g', label='Open loop')
    plt.plot(residual_gbioedge[k, :], 'b', label='gbioedge '+str(param['n_subaperture'])+'x'+
              str(param['n_subaperture'])+', no SR '+str(param['n_modes_to_show'] )+' modes')
    plt.plot(residual_gbioedge_sr[k, :], 'r', label='gbioedge_sr '+str(param['n_subaperture'])+'x'+
              str(param['n_subaperture'])+', ' +str(param['n_modes_to_show_sr'] )+' modes shown\n'
              +str(param['n_modes_to_control_sr'])+' modes controlled')
    plt.plot(residual_sgbioedge_sr[k, :], 'm', label='sgbioedge_sr '+str(param['n_subaperture'])+'x'+
              str(param['n_subaperture'])+', ' +str(param['n_modes_to_show_sr'] )+' modes shown\n'
              +str(param['n_modes_to_control_sr'])+' modes controlled')
    plt.plot(residual_gbioedge_oversampled[k, :], 'k', label='gbioedge '+str(2*param['n_subaperture'])+'x'+
              str(2*param['n_subaperture'])+', '+str(param['n_modes_to_show_oversampled'] )+' modes')
    plt.title('Closed Loop residuals\n'
              'loop frequency : '+str(np.round(1/param['sampling_time']/1e3, 1))+'kHz\n'
              'Telescope diameter: '+str(param['diameter']) + ' m\n'
              'Half grey width : '+str(param['modulation'])+' lambda/D\n'
              'seed : '+str(seed))
    plt.ylim((0,ylim))
    plt.xlabel('Iteration')
    plt.ylabel('phase std (nm)')
    plt.legend(loc='upper left')
    
    path_image = path_plots / pathlib.Path('gbioedge_sgbioedge_SR_closed_loop_'+str(k)+'_seed'+'.png')
    
    plt.savefig(path_image, bbox_inches = 'tight')
    
    images.append(Image.open(path_image))
    
#%% make gif
    
images[0].save(path_plots / 'animated_plot.gif', save_all=True, append_images=images, duration=1000, loop=0)
    
#%%

ylim = 1000

plt.figure()

plt.plot(residual_gbioedge[k,:], 'b', label='gbioedge '+str(param['n_subaperture'])+'x'+
          str(param['n_subaperture'])+', '+str(param['n_modes_to_show'] )+' modes')
plt.plot(residual_gbioedge_sr[k,:], '--r', label='gbioedge_sr '+str(param['n_subaperture'])+'x'+
          str(param['n_subaperture'])+', SR ' +str(param['n_modes_to_show_sr'] )+' modes shown\n'
          +str(param['n_modes_to_control_sr'])+' modes controlled')
plt.plot(residual_sgbioedge_sr[k,:], 'm', label='sgbioedge_sr '+str(param['n_subaperture'])+'x'+
          str(param['n_subaperture'])+', SR ' +str(param['n_modes_to_show_sr'] )+' modes shown\n'
          +str(param['n_modes_to_control_sr'])+' modes controlled')
plt.plot(residual_gbioedge_oversampled[k, :], 'k', label='gbioedge '+str(2*param['n_subaperture'])+'x'+
          str(2*param['n_subaperture'])+', '+str(param['n_modes_to_show_oversampled'] )+' modes')

for k in range(1,len(param['seeds'])):
    plt.plot(residual_gbioedge[k,:], 'b')
    plt.plot(residual_gbioedge_sr[k,:], '--r')
    plt.plot(residual_sgbioedge_sr[k,:], 'm')
    plt.plot(residual_gbioedge_oversampled[k, :], 'k')
    
plt.legend()

plt.title('Closed Loop residuals - all seeds\n'
          'loop frequency : '+str(np.round(1/param['sampling_time']/1e3, 1))+'kHz\n'
          'Telescope diameter: '+str(param['diameter']) + ' m\n'
          'Half grey width : '+str(param['modulation'])+' lambda/D\n'+ str(len(param['seeds']))+' seeds')
plt.ylim((0,ylim))
plt.xlabel('Iteration')
plt.ylabel('phase std (nm)')
plt.savefig(path_plots / "sr_broken_gbioedge_vs_sgbioedge.png", bbox_inches = 'tight')

#%% Colect only  broken loops

threshold = 96. # [nm] : std phase over which the loop is considered broken
n_iter_bootstarp = 100

broken_seeds_gbioedge = []
broken_seeds_sgbioedge = []

broken_loops_gbioedge = []
broken_loops_sgbioedge = []

for k in range(len(param['seeds'])):
    
    if residual_gbioedge_sr[k, n_iter_bootstarp:].max() >= threshold:
        broken_seeds_gbioedge.append(param['seeds'][k])
        broken_loops_gbioedge.append(residual_gbioedge_sr[k, :])
        
    if residual_sgbioedge_sr[k, n_iter_bootstarp:].max() >= threshold:
        broken_seeds_sgbioedge.append(param['seeds'][k])
        broken_loops_sgbioedge.append(residual_sgbioedge_sr[k, :])
        
broken_loops_gbioedge = np.asarray(broken_loops_gbioedge)
broken_loops_sgbioedge = np.asarray(broken_loops_sgbioedge)

#%% Plot only  broken loops

figure=plt.figure()


plt.title('Broken Closed Loop residuals\n'
          'loop frequency : '+str(np.round(1/param['sampling_time']/1e3, 1))+'kHz\n'
          'Telescope diameter: '+str(param['diameter']) + ' m\n'
          'Half grey width : '+str(param['modulation'])+' lambda/D\n'+ str(len(param['seeds']))+' seeds')

plt.plot(residual_gbioedge[k,:], 'b', label='gbioedge '+str(param['n_subaperture'])+'x'+
          str(param['n_subaperture'])+', '+str(param['n_modes_to_show'] )+' modes')
plt.plot(residual_gbioedge_oversampled[k, :], 'k', label='gbioedge '+str(2*param['n_subaperture'])+'x'+
          str(2*param['n_subaperture'])+', '+str(param['n_modes_to_show_oversampled'] )+' modes')

for n in range(broken_loops_sgbioedge.shape[0]):
    plt.plot(broken_loops_sgbioedge[n,:], color = 'm',  label=str(n+1) + ') sgbioedge, seed '+str(broken_seeds_sgbioedge[n]))
for m in range(broken_loops_gbioedge.shape[0]):
    plt.plot(broken_loops_gbioedge[m,:], color = 'r', linestyle=(0,(5,1)), label= str(m+1) + ') gbioedge, seed '+ str(broken_seeds_gbioedge[m]))
    
plt.legend(loc='upper left')
figure.set_size_inches(15, 8)
plt.xlabel('Iteration')
plt.ylabel('phase std (nm)')

plt.savefig(path_plots / "sr_only_broken_gbioedge_vs_sgbioedge.png", bbox_inches = 'tight')
