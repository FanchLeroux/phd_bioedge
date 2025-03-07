# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:29:55 2025

@author: fleroux
"""

# pylint: disable=undefined-variable

import platform
import pathlib

from copy import deepcopy

import dill

from parameter_file import get_parameters

import matplotlib.pyplot as plt

#%% Import parameter file

param = get_parameters()

#%% Load OOPAO calibration objects computed in analysis_file.py

if platform.system() == 'Windows':
    temp = deepcopy(pathlib.PosixPath)
    pathlib.PosixPath = pathlib.WindowsPath

dill.load_session(param['path_analysis'] / pathlib.Path('analysis'+str(param['filename'])+'.pkl'))

if platform.system() == 'Windows':
    pathlib.PosixPath = temp

param = get_parameters()

#%%###################### Plots #################

# SVD - Normalized Eigenvalues  

    # --------------- pyramid --------------------

plt.figure()
plt.semilogy(singular_values_pyramid/singular_values_pyramid.max(), 'b', label='no SR')
plt.semilogy(singular_values_pyramid_sr/singular_values_pyramid_sr.max(), 'r', label='SR')
plt.semilogy(singular_values_pyramid_oversampled/singular_values_pyramid_oversampled.max(), 'c', label='oversampled')
plt.title('singular_values_pyramid, '+str(param['n_subaperture'])+' subapertures, ' + param['modal_basis'] + ' modes used, 0.25 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')


    # --------------- gbioedge --------------------

plt.figure()
plt.semilogy(singular_values_gbioedge/singular_values_gbioedge.max(), 'b', label='no SR')
plt.semilogy(singular_values_gbioedge_sr/singular_values_gbioedge_sr.max(), 'r', label='SR')
plt.semilogy(singular_values_gbioedge_oversampled/singular_values_gbioedge_oversampled.max(), 'c', label='oversampled')
plt.title('singular_values_gbioedge, '+str(param['n_subaperture'])+' subapertures, ' + param['modal_basis'] + ' modes used, 0.25 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')


    # --------------- sbioedge --------------------

plt.figure()
plt.semilogy(singular_values_sbioedge/singular_values_sbioedge.max(), 'b', label='no SR')
plt.semilogy(singular_values_sbioedge_sr/singular_values_sbioedge_sr.max(), 'r', label='SR')
plt.semilogy(singular_values_sbioedge_oversampled/singular_values_sbioedge_oversampled.max(), 'c', label='oversampled')
plt.title('singular_values_sbioedge, '+str(param['n_subaperture'])+' subapertures, ' + param['modal_basis'] + ' modes used, 0.25 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')


#%% Noise Propagation - With SR - All WFS

i = 0
for n_modes in param['list_modes_to_keep']:

    plt.figure()
    plt.plot(noise_propagation_gbioedge_oversampled, 'k', label='gbioedge '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))
    plt.plot(noise_propagation_sbioedge_oversampled, 'g', label='sbioedge '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))
    plt.plot(noise_propagation_pyramid_oversampled, 'm', label='pyramid '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))

    #plt.plot(noise_propagation_gbioedge[i], label= str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' '+str(n_modes)+" modes")
    plt.plot(noise_propagation_gbioedge_sr[i], 'r' , label= 'gbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    plt.plot(noise_propagation_sbioedge_sr[i] , 'y', label= 'sbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    plt.plot(noise_propagation_pyramid_sr[i], 'b' , label= 'pyramid '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    i+=1
    plt.yscale('log')
    plt.title("Grey Bi-O-Edge VS MPYWFS Uniform noise propagation\n"+str(param['modulation'])+' lambda/D')
    plt.xlabel("mode ("+param['modal_basis']+") index")
    plt.ylabel("np.diag(R @ R.T)/wfs.nSignal")
    plt.legend()
    plt.savefig(param['path_plots'] / pathlib.Path(str(n_modes) + '_modes' + param['filename'] + '.png'), bbox_inches = 'tight')


#%% SR vs no SR behaviour with LO modes

index_n_modes_no_sr = 0
index_n_modes_sr = -4

plt.figure()
plt.plot(noise_propagation_gbioedge_oversampled, 'k', label='gbioedge '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))
plt.plot(noise_propagation_gbioedge_sr[index_n_modes_no_sr], 'b' , 
         label= 'gbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - no SR - '+str(param['list_modes_to_keep'][index_n_modes_no_sr])+" modes")
plt.plot(noise_propagation_gbioedge_sr[index_n_modes_sr], 'r--' , 
         label= 'gbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(param['list_modes_to_keep'][index_n_modes_sr])+" modes")
plt.yscale('log')
plt.title("Grey Bi-O-Edge SR and no SR, uniform noise propagation\n"+str(param['modulation'])+' lambda/D')
plt.xlabel("mode ("+param['modal_basis']+") index")
plt.ylabel("np.diag(R @ R.T)/wfs.nSignal")
plt.legend()
plt.savefig(param['path_plots'] / pathlib.Path(param['filename'] + '.png'), bbox_inches = 'tight')

# zoom on LO
plt.figure()
plt.plot(noise_propagation_gbioedge_oversampled, 'k', label='gbioedge '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))
plt.plot(noise_propagation_gbioedge_sr[index_n_modes_no_sr], 'b' , 
         label= 'gbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - no SR - '+str(param['list_modes_to_keep'][index_n_modes_no_sr])+" modes")
plt.plot(noise_propagation_gbioedge_sr[index_n_modes_sr], 'r--' , 
         label= 'gbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(param['list_modes_to_keep'][index_n_modes_sr])+" modes")
plt.yscale('log')
plt.title("ZOOM - Grey Bi-O-Edge SR and no SR, uniform noise propagation\n"+str(param['modulation'])+' lambda/D')
plt.xlabel("mode ("+param['modal_basis']+") index")
plt.ylabel("np.diag(R @ R.T)/wfs.nSignal")
plt.xlim(0,120)
plt.legend()
plt.savefig(param['path_plots'] / pathlib.Path('ZOOM'+ param['filename'] + '.png'), bbox_inches = 'tight')