# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:09:37 2025

@author: fleroux
"""

import pathlib

import numpy as np

from fanch.basis.fourier import compute_real_fourier_basis   

#%%

dirc_data = pathlib.Path(__file__).parent.parent.parent.parent / "data"

#%%

n_subaperture = 40
n_pixels_in_slm_pupil = 1152

#%%

fourier_modes = compute_real_fourier_basis(n_pixels_in_slm_pupil, return_map=False, npx_computation_limit=n_subaperture)

# remove piston
fourier_modes = fourier_modes[:,:,1:]

# std normalization
fourier_modes = fourier_modes / np.std(fourier_modes, axis=(0,1))

# set the mean value around pi
fourier_modes = fourier_modes + np.pi

# scale from 0 to 255 for a 2pi phase shift
fourier_modes = fourier_modes * 255/(2*np.pi)

# convert to 8-bit integers
fourier_modes = fourier_modes.astype(np.uint8)

np.save(dirc_data / "slm" / "modal_basis" / "fourier_modes" / ("fourier_modes_" + 
                                                               str(n_pixels_in_slm_pupil) + 
                                                               "_pixels_in_slm_pupil_" +
                                                               str(n_subaperture) +
                                                               "_subapertures.npy"), fourier_modes)
