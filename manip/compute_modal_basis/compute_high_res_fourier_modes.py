# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:09:37 2025

@author: fleroux
"""

import pathlib

import numpy as np

from fanch.basis.fourier import compute_real_fourier_basis   

#%%

dirc_data = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent / "data"

#%%

n_subaperture = 20
n_pixels_in_slm_pupil = 600

#%%

fourier_modes = compute_real_fourier_basis(n_pixels_in_slm_pupil, return_map=False, npx_computation_limit=n_subaperture)

np.save(dirc_data / "slm" / "modal_basis" / "fourier_modes" / ("fourier_modes_" + 
                                                               str(n_pixels_in_slm_pupil) + 
                                                               "_pixels_in_slm_pupil_" +
                                                               str(n_subaperture) +
                                                               "_subapertures.npy"), fourier_modes)
