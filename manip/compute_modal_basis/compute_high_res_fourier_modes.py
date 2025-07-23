# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:09:37 2025

@author: fleroux
"""

import pathlib

import numpy as np

from fanch.basis.fourier import compute_real_fourier_basis,\
                                sort_real_fourier_basis,\
                                extract_horizontal_frequencies,\
                                extract_vertical_frequencies,\
                                extract_diagonal_frequencies

#%%

dirc_data = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent / "data"

#%%

n_subaperture = 1152
n_pixels_in_slm_pupil = 20

#%%

fourier_modes_map = compute_real_fourier_basis(n_pixels_in_slm_pupil, return_map=True, npx_computation_limit=n_subaperture)

fourier_modes = sort_real_fourier_basis(fourier_modes_map)
horizontal_fourier_modes = extract_horizontal_frequencies(fourier_modes_map)
vertical_fourier_modes = extract_vertical_frequencies(fourier_modes_map)
diagonal_fourier_modes = extract_diagonal_frequencies(fourier_modes_map)

# remove piston
fourier_modes = fourier_modes[:,:,1:]
horizontal_fourier_modes = horizontal_fourier_modes[:,:,1:]
vertical_fourier_modes = vertical_fourier_modes[:,:,1:]
diagonal_fourier_modes = diagonal_fourier_modes[:,:,1:]

# std normalization
fourier_modes = fourier_modes / np.std(fourier_modes, axis=(0,1))
horizontal_fourier_modes = horizontal_fourier_modes / np.std(horizontal_fourier_modes, axis=(0,1))
vertical_fourier_modes = vertical_fourier_modes / np.std(vertical_fourier_modes, axis=(0,1))
diagonal_fourier_modes = diagonal_fourier_modes / np.std(diagonal_fourier_modes, axis=(0,1))

# set the mean value around pi
fourier_modes = fourier_modes + np.pi
horizontal_fourier_modes = horizontal_fourier_modes + np.pi
vertical_fourier_modes = vertical_fourier_modes + np.pi
diagonal_fourier_modes = diagonal_fourier_modes + np.pi

# scale from 0 to 255 for a 2pi phase shift
fourier_modes = fourier_modes * 255/(2*np.pi)
horizontal_fourier_modes = horizontal_fourier_modes * 255/(2*np.pi)
vertical_fourier_modes = vertical_fourier_modes * 255/(2*np.pi)
diagonal_fourier_modes = diagonal_fourier_modes * 255/(2*np.pi)

# convert to 8-bit integers
fourier_modes = fourier_modes.astype(np.uint8)
horizontal_fourier_modes = horizontal_fourier_modes.astype(np.uint8)
vertical_fourier_modes = vertical_fourier_modes.astype(np.uint8)
diagonal_fourier_modes = diagonal_fourier_modes.astype(np.uint8)

np.save(dirc_data / "slm" / "modal_basis" / "fourier_modes" / ("fourier_modes_" + 
                                                               str(n_pixels_in_slm_pupil) + 
                                                               "_pixels_in_slm_pupil_" +
                                                               str(n_subaperture) +
                                                               "_subapertures.npy"), fourier_modes)
np.save(dirc_data / "slm" / "modal_basis" / "fourier_modes" / ("horizontal_fourier_modes_" + 
                                                               str(n_pixels_in_slm_pupil) + 
                                                               "_pixels_in_slm_pupil_" +
                                                               str(n_subaperture) +
                                                               "_subapertures.npy"), horizontal_fourier_modes)
np.save(dirc_data / "slm" / "modal_basis" / "fourier_modes" / ("vertical_fourier_modes_" + 
                                                               str(n_pixels_in_slm_pupil) + 
                                                               "_pixels_in_slm_pupil_" +
                                                               str(n_subaperture) +
                                                               "_subapertures.npy"), vertical_fourier_modes)
np.save(dirc_data / "slm" / "modal_basis" / "fourier_modes" / ("diagonal_fourier_modes_" + 
                                                               str(n_pixels_in_slm_pupil) + 
                                                               "_pixels_in_slm_pupil_" +
                                                               str(n_subaperture) +
                                                               "_subapertures.npy"), diagonal_fourier_modes)
