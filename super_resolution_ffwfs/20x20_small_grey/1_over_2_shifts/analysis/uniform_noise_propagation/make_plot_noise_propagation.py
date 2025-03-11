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

#%% path type compatibility issues

if platform.system() == 'Windows':
    temp = deepcopy(pathlib.PosixPath)
    pathlib.PosixPath = pathlib.WindowsPath
elif platform.system() == 'Linux':
    temp = deepcopy(pathlib.WindowsPath)
    pathlib.WindowsPath = pathlib.PosixPath

#%%

path = pathlib.Path(__file__).parent.parent.parent # location of parameter_file.py

#%% import parameter file from a distinct repository than the one of this file (relou)

# weird method from https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path

import importlib.util
import sys
spec = importlib.util.spec_from_file_location("get_parameters", path / "parameter_file.py")
foo = importlib.util.module_from_spec(spec)
sys.modules["parameter_file"] = foo
spec.loader.exec_module(foo)

#%%

param = foo.get_parameters()

#%% Load analysis results

dill.load_session(pathlib.Path(__file__).parent / "data_analysis" /pathlib.Path('analysis_noise_propagation'+
                                                                                str(param['filename'])+'.pkl'))

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
          ' modes used, 0.5 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')


#%% Noise Propagation - With or Without SR - gbioedge

# --------------- sgbioedge -------------------- #

i = 0
for n_modes in param['list_modes_to_keep']:

    plt.figure()
    plt.plot(noise_propagation_sgbioedge_oversampled, 'k', label='sgbioedge '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))
    #plt.plot(noise_propagation_pyramid_oversampled, 'm', label='pyramid '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))

    #plt.plot(noise_propagation_sgbioedge[i], label= str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' '+str(n_modes)+" modes")
    plt.plot(noise_propagation_sgbioedge_sr[i], 'r' , label= 'sgbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    #plt.plot(noise_propagation_pyramid_sr[i], 'b' , label= 'pyramid '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    i+=1
    plt.yscale('log')
    plt.ylim(1e-15, 1e-9)
    plt.title("Grey Bi-O-Edge VS MPYWFS Uniform noise propagation\n"+str(param['modulation'])+' lambda/D')
    plt.xlabel("mode ("+param['modal_basis']+") index")
    plt.ylabel("np.diag(R @ R.T)/wfs.nSignal")
    plt.legend()
    plt.savefig(pathlib.Path(__file__).parent / "plots" / pathlib.Path('figure_sgb'+str(i)+'.png'), bbox_inches = 'tight')

#%% --------------- gbioedge -------------------- #

i = 0
for n_modes in param['list_modes_to_keep']:

    plt.figure()
    plt.plot(noise_propagation_gbioedge_oversampled, 'k', label='gbioedge '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))
    #plt.plot(noise_propagation_pyramid_oversampled, 'm', label='pyramid '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))

    #plt.plot(noise_propagation_gbioedge[i], label= str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' '+str(n_modes)+" modes")
    plt.plot(noise_propagation_gbioedge_sr[i], 'b' , label= 'gbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    #plt.plot(noise_propagation_pyramid_sr[i], 'b' , label= 'pyramid '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    i+=1
    plt.yscale('log')
    plt.ylim(1e-15, 1e-9)
    plt.title("Grey Bi-O-Edge VS MPYWFS Uniform noise propagation\n"+str(param['modulation'])+' lambda/D')
    plt.xlabel("mode ("+param['modal_basis']+") index")
    plt.ylabel("np.diag(R @ R.T)/wfs.nSignal")
    plt.legend()
    plt.savefig(pathlib.Path(__file__).parent / "plots" / pathlib.Path('figure_gb_'+str(i)+'.png'), bbox_inches = 'tight')
    
    
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
    plt.savefig(pathlib.Path(__file__).parent / "plots" / pathlib.Path('figure_sgb_gb_'+str(i)+'.png'), bbox_inches = 'tight')
    
#%% --------------- pyramid, sgb,  gbioedge and sbioedge -------------------- #

i = 0
for n_modes in param['list_modes_to_keep']:

    plt.figure()
    plt.plot(noise_propagation_gbioedge_oversampled, 'k', label='gbioedge '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))
    #plt.plot(noise_propagation_pyramid_oversampled, 'm', label='pyramid '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))

    #plt.plot(noise_propagation_gbioedge[i], label= str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' '+str(n_modes)+" modes")
    plt.plot(noise_propagation_gbioedge_sr[i], 'b' , label= 'gbioedge '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    #plt.plot(noise_propagation_pyramid_sr[i], 'b' , label= 'pyramid '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    i+=1
    plt.yscale('log')
    plt.ylim(1e-15, 1e-9)
    plt.title("Grey Bi-O-Edge VS MPYWFS Uniform noise propagation\n"+str(param['modulation'])+' lambda/D')
    plt.xlabel("mode ("+param['modal_basis']+") index")
    plt.ylabel("np.diag(R @ R.T)/wfs.nSignal")
    plt.legend()
    plt.savefig(pathlib.Path(__file__).parent / "plots" / pathlib.Path('figure_sgb_gb_'+str(i)+'.png'), bbox_inches = 'tight')