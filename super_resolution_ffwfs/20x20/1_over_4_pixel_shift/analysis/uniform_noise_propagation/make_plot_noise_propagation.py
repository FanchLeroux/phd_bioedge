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

#%% Define paths

path = pathlib.Path(__file__).parent
path_data = path.parent.parent.parent.parent.parent.parent / "phd_bioedge_data" / pathlib.Path(*path.parts[-5:-2]) # could be done better

#%% Get parameter file

path_parameter_file = path_data / "parameter_file.pkl"
load_vars(path_parameter_file, ['param'])

#%% load analysis results

path_analysis_data = path_data / 'analysis' / 'uniform_noise_propagation'
load_vars(path_analysis_data / pathlib.Path('analysis_uniform_noise_propagation' + param['filename'] + '.pkl'))

#%% path plots

path_plots = path_data / 'plots' / 'uniform_noise_propagation'

#%%###################### Plots #################

# SVD - Normalized Eigenvalues  

    # --------------- pyramid --------------------

plt.figure()
plt.semilogy(singular_values_pyramid/singular_values_pyramid.max(), 'b', label='no SR')
plt.semilogy(singular_values_pyramid_sr/singular_values_pyramid_sr.max(), 'r', label='SR')
plt.semilogy(singular_values_pyramid_oversampled/singular_values_pyramid_oversampled.max(), 'c', label='oversampled')
plt.title('singular_values_pyramid, '+str(param['n_subaperture'])+' subapertures, ' + param['modal_basis'] 
          + ' modes used, 0.5 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')

    # --------------- sbioedge -------------------- #

plt.figure()
plt.semilogy(singular_values_sbioedge/singular_values_sbioedge.max(), 'b', label='no SR')
plt.semilogy(singular_values_sbioedge_sr/singular_values_sbioedge_sr.max(), 'r', label='SR')
plt.semilogy(singular_values_sbioedge_oversampled/singular_values_sbioedge_oversampled.max(), 'c', label='oversampled')
plt.title('singular_values_sbioedge, '+str(param['n_subaperture'])+' subapertures, ' + param['modal_basis'] + 
          ' modes used, 0.5 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')

    # --------------- sgbioedge -------------------- #

plt.figure()
plt.semilogy(singular_values_sgbioedge/singular_values_sgbioedge.max(), 'b', label='no SR')
plt.semilogy(singular_values_sgbioedge_sr/singular_values_sgbioedge_sr.max(), 'r', label='SR')
plt.semilogy(singular_values_sgbioedge_oversampled/singular_values_sgbioedge_oversampled.max(), 'c', label='oversampled')
plt.title('singular_values_sgbioedge, '+str(param['n_subaperture'])+' subapertures, ' + param['modal_basis'] 
          + ' modes used, 0.5 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')


# --------------- gbioedge -------------------- #

plt.figure()
plt.semilogy(singular_values_gbioedge/singular_values_gbioedge.max(), 'b', label='no SR')
plt.semilogy(singular_values_gbioedge_sr/singular_values_gbioedge_sr.max(), 'r', label='SR')
plt.semilogy(singular_values_gbioedge_oversampled/singular_values_gbioedge_oversampled.max(), 'c', label='oversampled')
plt.title('singular_values_gbioedge, '+str(param['n_subaperture'])+' subapertures, ' + param['modal_basis'] + 
          ' modes used, '+str(param['sr_amplitude'])+' pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')

#%% Normalized SVG - all

plt.figure()
plt.semilogy(singular_values_pyramid/singular_values_pyramid.max(), 'b', 
             label='pyramid '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' no SR')
plt.semilogy(singular_values_pyramid_sr/singular_values_pyramid_sr.max(), 'r', 
             label='pyramid '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' SR')
plt.semilogy(singular_values_sgbioedge/singular_values_sgbioedge.max(), 'c', 
             label='sgbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' no SR')
plt.semilogy(singular_values_sgbioedge_sr/singular_values_sgbioedge_sr.max(), 'm', 
             label='sgbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' SR')
plt.semilogy(singular_values_sgbioedge_oversampled/singular_values_sgbioedge_oversampled.max(), 'k', 
             label=str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')
plt.title('Singular Values of the Interraction Matrix Eigen modes')

#%%###################### Plots #################

# SVD - Raw Eigenvalues  

    # --------------- pyramid --------------------

plt.figure()
plt.semilogy(singular_values_pyramid, 'b', label='no SR')
plt.semilogy(singular_values_pyramid_sr, 'r', label='SR')
plt.semilogy(singular_values_pyramid_oversampled, 'c', label='oversampled')
plt.title('singular_values_pyramid, '+str(param['n_subaperture'])+' subapertures, ' + param['modal_basis'] 
          + ' modes used, 0.5 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('raw eigen value')

    # --------------- sbioedge -------------------- #

plt.figure()
plt.semilogy(singular_values_sbioedge, 'b', label='no SR')
plt.semilogy(singular_values_sbioedge_sr, 'r', label='SR')
plt.semilogy(singular_values_sbioedge_oversampled, 'c', label='oversampled')
plt.title('singular_values_sbioedge, '+str(param['n_subaperture'])+' subapertures, ' + param['modal_basis'] + 
          ' modes used, 0.5 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('raw eigen value')

    # --------------- sgbioedge -------------------- #

plt.figure()
plt.semilogy(singular_values_sgbioedge, 'b', label='no SR')
plt.semilogy(singular_values_sgbioedge_sr, 'r', label='SR')
plt.semilogy(singular_values_sgbioedge_oversampled, 'c', label='oversampled')
plt.title('singular_values_sgbioedge, '+str(param['n_subaperture'])+' subapertures, ' + param['modal_basis'] 
          + ' modes used, 0.5 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('raw eigen value')


# --------------- gbioedge -------------------- #

plt.figure()
plt.semilogy(singular_values_gbioedge, 'b', label='no SR')
plt.semilogy(singular_values_gbioedge_sr, 'r', label='SR')
plt.semilogy(singular_values_gbioedge_oversampled, 'c', label='oversampled')
plt.title('singular_values_gbioedge, '+str(param['n_subaperture'])+' subapertures, ' + param['modal_basis'] + 
          ' modes used, '+str(param['sr_amplitude'])+' pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('raw eigen value')

#%% Raw SVG - all

plt.figure()
plt.semilogy(singular_values_pyramid, 'b', 
             label='pyramid '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' no SR')
plt.semilogy(singular_values_pyramid_sr, 'r', 
             label='pyramid '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' SR')
plt.semilogy(singular_values_sgbioedge, 'c', 
             label='sgbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' no SR')
plt.semilogy(singular_values_sgbioedge_sr, 'm', 
             label='sgbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' SR')
plt.semilogy(singular_values_sgbioedge_oversampled, 'k', 
             label=str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('raw eigen value')
plt.title('Singular Values of the Interraction Matrix Eigen modes')

#%% Noise Propagation - With or Without SR - gbioedge

# --------------- sgbioedge -------------------- #

i = 0
for n_modes in param['list_modes_to_keep']:

    plt.figure()
    plt.plot(noise_propagation_sgbioedge_oversampled, 'k', 
             label='sgbioedge '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))
    #plt.plot(noise_propagation_pyramid_oversampled, 'm', label='pyramid '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))

    #plt.plot(noise_propagation_sgbioedge[i], label= str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' '+str(n_modes)+" modes")
    plt.plot(noise_propagation_sgbioedge_sr[i], 'r' , 
             label= 'sgbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    #plt.plot(noise_propagation_pyramid_sr[i], 'b' , label= 'pyramid '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    i+=1
    plt.yscale('log')
    plt.ylim(1e-15, 1e-9)
    plt.title("Grey Bi-O-Edge VS MPYWFS Uniform noise propagation\n"+str(param['modulation'])+' lambda/D')
    plt.xlabel("mode ("+param['modal_basis']+") index")
    plt.ylabel("np.diag(R @ R.T)/wfs.nSignal")
    plt.legend()
    plt.savefig(path_plots / pathlib.Path('figure_sgb'+str(i)+'.png'), bbox_inches = 'tight')

#%% --------------- gbioedge -------------------- #

i = 0
for n_modes in param['list_modes_to_keep']:

    plt.figure()
    plt.plot(noise_propagation_gbioedge_oversampled, 'k', 
             label='gbioedge '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))
    #plt.plot(noise_propagation_pyramid_oversampled, 'm', label='pyramid '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))

    #plt.plot(noise_propagation_gbioedge[i], label= str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' '+str(n_modes)+" modes")
    plt.plot(noise_propagation_gbioedge_sr[i], 'b' , 
             label= 'gbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    #plt.plot(noise_propagation_pyramid_sr[i], 'b' , label= 'pyramid '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    i+=1
    plt.yscale('log')
    plt.ylim(1e-15, 1e-9)
    plt.title("Grey Bi-O-Edge VS MPYWFS Uniform noise propagation\n"+str(param['modulation'])+' lambda/D')
    plt.xlabel("mode ("+param['modal_basis']+") index")
    plt.ylabel("np.diag(R @ R.T)/wfs.nSignal")
    plt.legend()
    plt.savefig(path_plots / pathlib.Path('figure_gb_'+str(i)+'.png'), bbox_inches = 'tight')
    
    
#%% --------------- gbioedge and sbioedge -------------------- #

i = 0
for n_modes in param['list_modes_to_keep']:

    plt.figure()
    plt.plot(noise_propagation_gbioedge_oversampled, 'k', 
             label='gbioedge '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))
    plt.plot(noise_propagation_gbioedge_sr[i], '--b' , 
             label= 'gbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    plt.plot(noise_propagation_sgbioedge_sr[i], 'r' , 
             label= 'sgbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    
    i+=1
    plt.yscale('log')
    plt.ylim(1e-15, 1e-9)
    plt.title("Grey Bi-O-Edge VS Small Grey Bi-O-Edge\nUniform noise propagation\n"
              +str(param['modulation'])+' lambda/D')
    plt.xlabel("mode ("+param['modal_basis']+") index")
    plt.ylabel("np.diag(R @ R.T)/wfs.nSignal")
    plt.legend()
    plt.savefig(path_plots / pathlib.Path('figure_sgb_gb_'+str(i)+'.png'), bbox_inches = 'tight')
    
#%% --------------- pyramid, gbioedge and sbioedge -------------------- #

i = 0
for n_modes in param['list_modes_to_keep']:

    plt.figure()
    
    plt.plot(noise_propagation_pyramid_oversampled, 'm', 
             label='pyramid '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))
    plt.plot(noise_propagation_gbioedge_oversampled, 'k', 
             label='gbioedge '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))
    
    plt.plot(noise_propagation_pyramid_sr[i], 'g' , 
             label= 'pyramid '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    plt.plot(noise_propagation_gbioedge_sr[i], 'b' , 
             label= 'gbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    plt.plot(noise_propagation_sgbioedge_sr[i], '--r' , 
             label= 'sgbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    
    
    i+=1
    plt.yscale('log')
    plt.ylim(1e-15, 1e-9)
    plt.title("Pyramid VS Grey Bi-O-Edge VS Small Grey Bi-O-Edge\nUniform noise propagation\n"
              +str(param['modulation'])+' lambda/D')
    plt.xlabel("mode ("+param['modal_basis']+") index")
    plt.ylabel("np.diag(R @ R.T)/wfs.nSignal")
    plt.legend()
    plt.savefig(path_plots / pathlib.Path('figure_sgb_gb_'+str(i)+'.png'), bbox_inches = 'tight')