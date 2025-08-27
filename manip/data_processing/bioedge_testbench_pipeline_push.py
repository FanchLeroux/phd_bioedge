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

#%% Push interaction matrix : utc_2025-07-30_12-51-21

utc_measurement = "utc_2025-07-30_12-51-21"

reference_intensities_push = np.load(dirc_data / utc_measurement /
                             (utc_measurement + 
                              "_reference_intensities_orca_inline.npy"))

raw_measurements_push = np.load(dirc_data / utc_measurement /
                             (utc_measurement + "_interaction_matrix.npy"))

#%% Binning push

raw_measurements_push = set_binning(raw_measurements_push, 16)
reference_intensities_push = set_binning(reference_intensities_push, 16)

#%% Select valid pixels - push

raw_measurements_push_std = np.std(raw_measurements_push, axis=2)
raw_measurements_push_std_normalized = raw_measurements_push_std/\
    raw_measurements_push_std.max()

threshold_push = 0.2
valid_pixels_push = raw_measurements_push_std_normalized > threshold_push

#%% Post-processing - push

# keep only valid pixels
interaction_matrix_push = np.reshape(\
    raw_measurements_push[valid_pixels_push==1],
    (int(valid_pixels_push.sum()),raw_measurements_push.shape[-1]))

reference_intensities_push_flat = np.reshape(\
    reference_intensities_push[valid_pixels_push==1],
    int(valid_pixels_push.sum()))
    
# flux normalization
interaction_matrix_push = interaction_matrix_push /\
    np.sum(interaction_matrix_push, axis=0)
    
reference_intensities_push_flat =\
    reference_intensities_push_flat/reference_intensities_push_flat.sum()
    
# substract reference intensities
interaction_matrix_push = interaction_matrix_push -\
    reference_intensities_push_flat[:,np.newaxis]

# truncate some modes
interaction_matrix_push = interaction_matrix_push[:,:30]

# SVD computation
interaction_matrix_push = CalibrationVault(
    interaction_matrix_push)

#%% Eigen modes extraction - push

n_mode = 0

eigen_mode_push = np.zeros(valid_pixels_push.shape, dtype=float)
eigen_mode_push.fill(np.nan)

eigen_mode_push[valid_pixels_push==1] = interaction_matrix_push.U[n_mode,:]

slicer_x = np.r_[np.s_[0:30], np.s_[50:70], np.s_[92:120]]
slicer_y = np.r_[np.s_[0:20], np.s_[44:74], np.s_[100:120]]
   
eigen_mode_push = np.delete(np.delete(eigen_mode_push, slicer_x, 1), 
                            slicer_y, 0)

#%% Interaction matrx visualization - push

n_mode_to_visualize = 3

mode_to_visualize = np.zeros(valid_pixels_push.shape, dtype=float)
mode_to_visualize.fill(np.nan)

mode_to_visualize[valid_pixels_push==1] =\
    interaction_matrix_push.D[:,n_mode_to_visualize]
   
mode_to_visualize = np.delete(np.delete(mode_to_visualize, slicer_x, 1), 
                            slicer_y, 0)

#%% Display

plt.figure()
plt.imshow(mode_to_visualize)
plt.title(f"calibration mode {mode_to_visualize}")

plt.figure()
plt.plot(interaction_matrix_push.eigenValues, 
         label="push", linestyle = 'dashed')
plt.yscale("log")
plt.title("Interaction matrices eigenvalues\nlog scale")
plt.xlabel("Eigen modes")
plt.ylabel("Eigen Values")
plt.legend()

plt.figure()
plt.imshow(eigen_mode_push)
plt.title(f"Eigen mode {n_mode}, push")

support = np.zeros(reference_intensities_push.shape, dtype=float)
support.fill(np.nan)
support[valid_pixels_push==1] =\
    reference_intensities_push[valid_pixels_push==1]

plt.figure()
plt.imshow(np.delete(np.delete(support, slicer_x, 1), slicer_y, 0))
plt.title(f"referecnce intensities, push")
