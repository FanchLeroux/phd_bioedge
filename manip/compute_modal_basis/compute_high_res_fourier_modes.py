# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 16:09:37 2025

@author: fleroux
"""

import pathlib

from fanch.basis.fourier import compute_real_fourier_basis   

#%%

dirc_data = pathlib.Path(__file__).parent

#%%

n_subaperture = 20
n_pixels_in_slm_pupil = 500

#%%

fourier_modes = compute_real_fourier_basis(n_pixels_in_slm_pupil, return_map=False, npx_computation_limit=n_subaperture)
