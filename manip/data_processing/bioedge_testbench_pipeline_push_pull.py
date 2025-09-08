# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 10:53:03 2025

@author: fleroux
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

# %%


def set_binning(array, binning_factor, mode='sum'):

    if array.shape[0] % binning_factor == 0:
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
        raise ValueError("Binning factor {binning_factor} not compatible with "
                         "the array size")

# %% parameters


# directory
dirc_data = pathlib.Path(
    __file__).parent.parent.parent.parent.parent / "data"

utc_measurement = "utc_2025-09-08_07-24-41"

dirc_interaction_matrix = dirc_data / "phd_bioedge" / \
    "manip" / "interaction_matrix" / utc_measurement

filename_calibration_modes = utc_measurement\
    + "_calibration_modes_slm_units.npy"

filename_reference_intensities_orca_inline = utc_measurement\
    + "_reference_intensities_orca_inline.npy"

filename_interaction_matrix_push_pull_orca_inline = utc_measurement\
    + "_interaction_matrix_push_pull_orca_inline.npy"

binning_factor = 16
threshold_push_pull = 0.2
n_mode = 0

# %% Load calibration modes

slm_phase_screens = np.load(dirc_interaction_matrix /
                            filename_calibration_modes,
                            mmap_mode='r')

# %% Push-Pull interaction matrix

reference_intensities_orca_inline =\
    np.load(dirc_interaction_matrix /
            filename_reference_intensities_orca_inline,
            mmap_mode='r')

raw_measurements_push_pull_orca_inline =\
    np.load(dirc_interaction_matrix /
            filename_interaction_matrix_push_pull_orca_inline,
            mmap_mode='r')

# %% Binning push_pull


raw_measurements_push_pull_orca_inline =\
    set_binning(raw_measurements_push_pull_orca_inline,
                binning_factor)
reference_intensities_orca_inline =\
    set_binning(reference_intensities_orca_inline,
                binning_factor)

# %% Select valid pixels - push_pull

raw_measurements_push_pull_orca_inline_std = np.std(
    raw_measurements_push_pull_orca_inline, axis=2)
raw_measurements_push_pull_orca_inline_std_normalized =\
    raw_measurements_push_pull_orca_inline_std /\
    raw_measurements_push_pull_orca_inline_std.max()

# %%


valid_pixels_push_pull =\
    raw_measurements_push_pull_orca_inline_std_normalized > threshold_push_pull

# %% Check valid pixels

plt.figure()
plt.imshow(valid_pixels_push_pull)
plt.title(f"Valid pixels, {threshold_push_pull} threshold")

# %% truncate some modes

raw_measurements_push_pull_orca_inline =\
    raw_measurements_push_pull_orca_inline[:, :, :]

# %% Post-processing - push_pull

# keep only valid pixels
interaction_matrix_push_pull = np.reshape(
    raw_measurements_push_pull_orca_inline[valid_pixels_push_pull == 1],
    (int(valid_pixels_push_pull.sum()),
     raw_measurements_push_pull_orca_inline.shape[-1]))

# flux normalization
# interaction_matrix_push_pull = interaction_matrix_push_pull /\
#     np.sum(interaction_matrix_push_pull, axis=0)

# SVD computation
U, s, VT = np.linalg.svd(interaction_matrix_push_pull, full_matrices=False)

# %% Eigen modes extraction - Control space - push_pull


eigen_modes_push_pull_control_space =\
    np.reshape(
        np.reshape(slm_phase_screens,
                   (slm_phase_screens.shape[1]*slm_phase_screens.shape[2],
                    slm_phase_screens.shape[0])) @ VT, slm_phase_screens.shape)

eigen_modes_push_pull_control_space =\
    np.reshape(
        slm_phase_screens.reshape(slm_phase_screens.shape[0], -1).T @ VT,
        slm_phase_screens.shape, order='A')

# %%

eigen_modes_push_pull_control_space = np.tensordot(
    VT, slm_phase_screens, axes=(1, 0))

# %% Eigen modes extraction - Measurements space - push_pull

eigen_modes_push_pull_measurements_space = np.zeros(
    (valid_pixels_push_pull.shape[0], valid_pixels_push_pull.shape[1],
     slm_phase_screens.shape[0]), dtype=float)
eigen_modes_push_pull_measurements_space.fill(np.nan)

eigen_modes_push_pull_measurements_space[valid_pixels_push_pull == 1] = U

# slicer_x = np.r_[np.s_[0:500], np.s_[800:1200], np.s_[1500:2048]]
# slicer_y = np.r_[np.s_[0:400], np.s_[700:1300], np.s_[1600:2048]]

slicer_x = np.r_[np.s_[0:30], np.s_[50:70], np.s_[92:120]]
slicer_y = np.r_[np.s_[0:20], np.s_[44:74], np.s_[100:120]]

eigen_modes_push_pull_measurements_space =\
    np.delete(np.delete(eigen_modes_push_pull_measurements_space, slicer_x, 1),
              slicer_y, 0)

# %% Display

plt.figure()
plt.plot(s, label="push_pull")
plt.yscale("log")
plt.title("Interaction matrices eigenvalues\nlog scale")
plt.xlabel("Eigen modes")
plt.ylabel("Eigen Values")
plt.legend()

support = np.zeros(reference_intensities_orca_inline.shape, dtype=float)
support.fill(np.nan)
support[valid_pixels_push_pull == 1] =\
    reference_intensities_orca_inline[valid_pixels_push_pull == 1]

plt.figure()
plt.imshow(np.delete(np.delete(support, slicer_x, 1), slicer_y, 0))
plt.title("reference intensities, push_pull")

plt.figure()
plt.imshow(eigen_modes_push_pull_control_space[n_mode, :, :])
plt.title(f"Eigen mode {n_mode}, control space, push_pull")

plt.figure()
plt.imshow(eigen_modes_push_pull_measurements_space[:, :, n_mode])
plt.title(f"Eigen mode {n_mode}, measurements space, push_pull")

plt.figure()
plt.plot(interaction_matrix_push_pull.sum(axis=0))
plt.title("""sum of the columns of interaction matrix\n
          (calibrations modes in measurements space)""")

plt.figure()
plt.plot(slm_phase_screens.sum(axis=(1, 2)))
plt.title("""sum of the calibration modes\n
          (calibrations modes in control space)""")

plt.figure()
plt.plot(U.sum(axis=0))
plt.title("sum of the colums of U\n(eigenmodes in measurements space)")

plt.figure()
plt.plot(eigen_modes_push_pull_control_space.sum(axis=(1, 2)))
plt.title("sum of the colums of VT\n(eigenmodes in control space)")
