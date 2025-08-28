# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:53:03 2025

@author: fleroux
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

import copy

from OOPAO.calibration.CalibrationVault import CalibrationVault

from OOPAO.tools.displayTools import displayMap, display_wfs_signals

#%%

def set_binning(array, binning_factor, mode='sum'):
    
    if array.shape[0]%binning_factor == 0:
        if array.ndim == 2:
            new_shape = [int(np.round(array.shape[0]/binning_factor)),
                         int(np.round(array.shape[1]/binning_factor))]
            shape = (new_shape[0], array.shape[0] // new_shape[0], 
                     new_shape[1], array.shape[1] // new_shape[1])
            if mode == 'sum':
                return array.reshape(shape).sum(-1).sum(1)
            else:
                return array.reshape(shape).mean(-1).mean(1)
        else:
            new_shape = [int(np.round(array.shape[0]/binning_factor)),
                         int(np.round(array.shape[1]/binning_factor)),
                         array.shape[2]]
            shape = (new_shape[0], array.shape[0] // new_shape[0], 
                     new_shape[1], array.shape[1] // new_shape[1],
                     new_shape[2])
            if mode == 'sum':
                return array.reshape(shape).sum(-2).sum(1)
            else:
                return array.reshape(shape).mean(-2).mean(1)
    else:
        raise ValueError("""Binning factor {binning_factor} not compatible with 
                         the array size""")

#%%

dirc_data = pathlib.Path(__file__).parent.parent.parent.parent.parent / "data"\
    / "data_banc_proto_bioedge" / "matrices"
    
#%% Push-Pull interaction matrix : utc_2025-07-30_15-41-15

utc_measurement = "utc_2025-07-30_15-41-15"

reference_intensities_push_pull = np.load(dirc_data / utc_measurement /
                             (utc_measurement + 
                              "_reference_intensities_orca_inline.npy"))

raw_measurements_push_pull = np.load(dirc_data / utc_measurement /
                             (utc_measurement + "_interaction_matrix.npy"))

#%% Binning push_pull

raw_measurements_push_pull = set_binning(raw_measurements_push_pull, 16)
reference_intensities_push_pull = set_binning(reference_intensities_push_pull, 16)

#%% Select valid pixels - push_pull

raw_measurements_push_pull_std = np.std(raw_measurements_push_pull, axis=2)
raw_measurements_push_pull_std_normalized = raw_measurements_push_pull_std/\
    raw_measurements_push_pull_std.max()

#%%

threshold_push_pull = 0.2
valid_pixels_push_pull =\
    raw_measurements_push_pull_std_normalized > threshold_push_pull

#%% truncate some modes

raw_measurements_push_pull = raw_measurements_push_pull[:,:,:]

#%% Post-processing - push_pull

# keep only valid pixels
interaction_matrix_push_pull = np.reshape(\
    raw_measurements_push_pull[valid_pixels_push_pull==1],
    (int(valid_pixels_push_pull.sum()),raw_measurements_push_pull.shape[-1]))
    
# flux normalization
# interaction_matrix_push_pull = interaction_matrix_push_pull /\
#     np.sum(interaction_matrix_push_pull, axis=0)

# SVD computation
interaction_matrix_push_pull = CalibrationVault(
    interaction_matrix_push_pull)

#%% Eigen modes extraction - push_pull

n_mode = 0

eigen_mode_push_pull = np.zeros(valid_pixels_push_pull.shape, dtype=float)
eigen_mode_push_pull.fill(np.nan)

eigen_mode_push_pull[valid_pixels_push_pull==1] = \
    interaction_matrix_push_pull.U[n_mode,:]

# slicer_x = np.r_[np.s_[0:500], np.s_[800:1200], np.s_[1500:2048]]
# slicer_y = np.r_[np.s_[0:400], np.s_[700:1300], np.s_[1600:2048]]

slicer_x = np.r_[np.s_[0:30], np.s_[50:70], np.s_[92:120]]
slicer_y = np.r_[np.s_[0:20], np.s_[44:74], np.s_[100:120]]
   
eigen_mode_push_pull = np.delete(np.delete(eigen_mode_push_pull, slicer_x, 1), 
                                 slicer_y, 0)

plt.figure()
plt.imshow(eigen_mode_push_pull)

#%% Display

plt.figure()
plt.plot(interaction_matrix_push_pull.eigenValues, label="push_pull")
plt.yscale("log")
plt.title("Interaction matrices eigenvalues\nlog scale")
plt.xlabel("Eigen modes")
plt.ylabel("Eigen Values")
plt.legend()

plt.figure()
plt.imshow(eigen_mode_push_pull)
plt.title(f"Eigen mode {n_mode}, push_pull")

support = np.zeros(reference_intensities_push_pull.shape, dtype=float)
support.fill(np.nan)
support[valid_pixels_push_pull==1] =\
    reference_intensities_push_pull[valid_pixels_push_pull==1]

plt.figure()
plt.imshow(np.delete(np.delete(support, slicer_x, 1), slicer_y, 0))
plt.title("reference intensities, push_pull")
