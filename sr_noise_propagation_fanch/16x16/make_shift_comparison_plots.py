# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 18:12:12 2025

@author: fleroux
"""

import platform
import pathlib

import matplotlib.pyplot as plt

from copy import deepcopy

import dill

#%%

path_plot = pathlib.Path(__file__).parent / 'shift_comparison_plots'

#%%

path_1_over_2 = pathlib.Path(__file__).parent / '1_over_2_shifts' / 'data_analysis'
filename = 'analysis_sr_noise_prop_I2_band_16x16_KL_basis.pkl'

dill.load_session(path_1_over_2 / filename)

singular_values_pyramid_1_over_2_shifts = singular_values_pyramid
singular_values_gbioedge_1_over_2_shifts = singular_values_gbioedge
singular_values_sbioedge_1_over_2_shifts = singular_values_sbioedge

singular_values_pyramid_sr_1_over_2_shifts = singular_values_pyramid_sr
singular_values_gbioedge_sr_1_over_2_shifts = singular_values_gbioedge_sr
singular_values_sbioedge_sr_1_over_2_shifts = singular_values_sbioedge_sr

singular_values_pyramid_oversampled_1_over_2_shifts = singular_values_pyramid_oversampled
singular_values_gbioedge_oversampled_1_over_2_shifts = singular_values_gbioedge_oversampled
singular_values_sbioedge_oversampled_1_over_2_shifts = singular_values_sbioedge_oversampled

noise_propagation_pyramid_1_over_2_shifts = noise_propagation_pyramid
noise_propagation_gbioedge_1_over_2_shifts = noise_propagation_gbioedge
noise_propagation_sbioedge_1_over_2_shifts = noise_propagation_sbioedge

noise_propagation_pyramid_sr_1_over_2_shifts = noise_propagation_pyramid_sr
noise_propagation_gbioedge_sr_1_over_2_shifts = noise_propagation_gbioedge_sr
noise_propagation_sbioedge_sr_1_over_2_shifts = noise_propagation_sbioedge_sr

noise_propagation_pyramid_oversampled_1_over_2_shifts = noise_propagation_pyramid_oversampled
noise_propagation_gbioedge_oversampled_1_over_2_shifts = noise_propagation_gbioedge_oversampled
noise_propagation_sbioedge_oversampled_1_over_2_shifts = noise_propagation_sbioedge_oversampled


#%%

path_1_over_4 = pathlib.Path(__file__).parent / '1_over_4_shifts' / 'data_analysis'
dill.load_session(path_1_over_4 / filename)


#%%

# pyramid
for k in range(len(param['list_modes_to_keep'])):
    plt.figure()
    plt.semilogy(noise_propagation_pyramid_sr_1_over_2_shifts[k], 'b', label='1 over 2 pixel shifts')
    plt.semilogy(noise_propagation_pyramid_sr[k], 'r', label='1 over 4 pixel Shifts')
    plt.title('Noise propagation for SR pyramid with ' + str(param['list_modes_to_keep'][k]) + 'modes')
    plt.legend()
    plt.xlabel('# KL mode')
    plt.ylabel('uniform noise propagation')
    plt.savefig(path_plot / pathlib.Path('pyramid_'+str(param['list_modes_to_keep'][k]) + '_modes' +'.png'), bbox_inches = 'tight')
   
# gbioedge
for k in range(len(param['list_modes_to_keep'])):
    plt.figure()
    plt.semilogy(noise_propagation_gbioedge_sr_1_over_2_shifts[k], 'b', label='1 over 2 pixel shifts')
    plt.semilogy(noise_propagation_gbioedge_sr[k], 'r', label='1 over 4 pixel Shifts')
    plt.title('Noise propagation for SR gbioedge with ' + str(param['list_modes_to_keep'][k]) + 'modes')
    plt.legend()
    plt.xlabel('# KL mode')
    plt.ylabel('uniform noise propagation')
    plt.savefig(path_plot / pathlib.Path('gbioedge_'+str(param['list_modes_to_keep'][k]) + '_modes' +'.png'), bbox_inches = 'tight')

# sbioedge
for k in range(len(param['list_modes_to_keep'])):
    plt.figure()
    plt.semilogy(noise_propagation_sbioedge_sr_1_over_2_shifts[k], 'b', label='1 over 2 pixel shifts')
    plt.semilogy(noise_propagation_sbioedge_sr[k], 'r', label='1 over 4 pixel Shifts')
    plt.title('Noise propagation for SR sbioedge with ' + str(param['list_modes_to_keep'][k]) + 'modes')
    plt.legend()
    plt.xlabel('# KL mode')
    plt.ylabel('uniform noise propagation')
    plt.savefig(path_plot / pathlib.Path('sbioedge_' + str(param['list_modes_to_keep'][k]) + '_modes' +'.png'), bbox_inches = 'tight')
    
