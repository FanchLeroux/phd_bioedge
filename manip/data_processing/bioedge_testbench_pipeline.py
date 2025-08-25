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

dirc_data = pathlib.Path(__file__).parent.parent.parent.parent.parent / "data"\
    / "data_banc_proto_bioedge" / "matrices"
    
#%% Push-Pull interaction matrix : utc_2025-07-30_15-41-15

utc_measurement = "utc_2025-07-30_15-41-15"

reference_intensities_push_pull = np.load(dirc_data / utc_measurement /
                             (utc_measurement + 
                              "_reference_intensities_orca_inline.npy"))

raw_measurements_push_pull = np.load(dirc_data / utc_measurement /
                             (utc_measurement + "_interaction_matrix.npy"))

#%% Push interaction matrix : utc_2025-07-30_12-51-21

utc_measurement = "utc_2025-07-30_12-51-21"

reference_intensities_push = np.load(dirc_data / utc_measurement /
                             (utc_measurement + 
                              "_reference_intensities_orca_inline.npy"))

raw_measurements_push = np.load(dirc_data / utc_measurement /
                             (utc_measurement + "_interaction_matrix.npy"))

#%% Select valid pixels - push_pull

raw_measurements_push_pull_std = np.std(raw_measurements_push_pull, axis=2)
raw_measurements_push_pull_std_normalized = raw_measurements_push_pull_std/\
    raw_measurements_push_pull_std.max()

#%%

threshold_push_pull = 0.2
valid_pixels_push_pull =\
    raw_measurements_push_pull_std_normalized > threshold_push_pull

#%% Select valid pixels - push

raw_measurements_push_std = np.std(raw_measurements_push, axis=2)
raw_measurements_push_std_normalized = raw_measurements_push_std/\
    raw_measurements_push_std.max()

#%%

threshold_push = 0.2
valid_pixels_push = raw_measurements_push_std_normalized > threshold_push

#%% Post-processing - push_pull

# keep only valid pixels
interaction_matrix_push_pull = np.reshape(\
    raw_measurements_push_pull[valid_pixels_push_pull==1],
    (int(valid_pixels_push_pull.sum()),raw_measurements_push_pull.shape[-1]))
    
# flux normalization
interaction_matrix_push_pull = interaction_matrix_push_pull /\
    np.sum(interaction_matrix_push_pull, axis=0)

# SVD computation
interaction_matrix_push_pull = CalibrationVault(
    interaction_matrix_push_pull)

#%% Post-processing - push

# keep only valid pixels
interaction_matrix_push = np.reshape(\
    raw_measurements_push[valid_pixels_push==1],
    (int(valid_pixels_push.sum()),raw_measurements_push.shape[-1]))

# substract reference intensities
interaction_matrix_push = interaction_matrix_push - np.reshape(\
    reference_intensities_push[valid_pixels_push==1],
    int(valid_pixels_push.sum()))[:,np.newaxis]
    
# flux normalization
interaction_matrix_push = interaction_matrix_push /\
    np.sum(interaction_matrix_push, axis=0)

# SVD computation
interaction_matrix_push = CalibrationVault(
    interaction_matrix_push)

#%% Eigen modes extraction - push_pull

n_mode = 0

eigen_mode_push_pull = np.zeros((2048,2048), dtype=float)
eigen_mode_push_pull.fill(np.nan)

eigen_mode_push_pull[valid_pixels_push_pull==1] = \
    interaction_matrix_push_pull.U[n_mode,:]

slicer_x = np.r_[np.s_[0:500], np.s_[800:1200], np.s_[1500:2048]]
slicer_y = np.r_[np.s_[0:400], np.s_[700:1300], np.s_[1600:2048]]
   
eigen_mode_push_pull = np.delete(np.delete(eigen_mode_push_pull, slicer_x, 1), 
                                 slicer_y, 0)

#%% Eigen modes extraction - push

n_mode = 0

eigen_mode_push = np.zeros((2048,2048), dtype=float)
eigen_mode_push.fill(np.nan)

eigen_mode_push[valid_pixels_push==1] = interaction_matrix_push.U[n_mode,:]

slicer_x = np.r_[np.s_[0:500], np.s_[800:1200], np.s_[1500:2048]]
slicer_y = np.r_[np.s_[0:400], np.s_[700:1300], np.s_[1600:2048]]
   
eigen_mode_push = np.delete(np.delete(eigen_mode_push, slicer_x, 1), 
                            slicer_y, 0)


#%% Display

plt.figure()
plt.plot(interaction_matrix_push_pull.eigenValues, label="push_pull")
plt.plot(interaction_matrix_push.eigenValues, 
         label="push", linestyle = 'dashed')
plt.yscale("log")
plt.title("Interaction matrices eigenvalues\nlog scale")
plt.xlabel("Eigen modes")
plt.ylabel("Eigen Values")
plt.legend()

plt.figure()
plt.imshow(eigen_mode_push_pull)
plt.title(f"Eigen mode {n_mode}, push_pull")

plt.figure()
plt.imshow(eigen_mode_push)
plt.title(f"Eigen mode {n_mode}, push")
