# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 10:00:30 2025

@author: fleroux
"""

import matplotlib.pyplot as plt
import numpy as np

from fanch.basis.fourier import compute_real_fourier_basis, extract_subset, sort_real_fourier_basis
from fanch.tools.miscellaneous import get_tilt

#%%

npx_tel = 64

npx_wfs = 8

fourier_basis_tel = compute_real_fourier_basis(npx_tel, return_map=True)

fourier_basis_tel_sorted = sort_real_fourier_basis(fourier_basis_tel)

fourier_basis_wfs = extract_subset(fourier_basis_tel, npx_wfs)

fourier_basis_wfs_sorted = sort_real_fourier_basis(fourier_basis_wfs)

fourier_basis_wfs_sorted_piston_removed = fourier_basis_wfs_sorted[:,:,1:]

#%%

fourier_basis_limit = compute_real_fourier_basis(npx_tel, return_map=True, npx_computation_limit=npx_wfs)

fourier_basis_limit_wfs = extract_subset(fourier_basis_limit, npx_wfs)

fourier_basis_limit_wfs_sorted = sort_real_fourier_basis(fourier_basis_limit_wfs)

fourier_basis_limit_wfs_sorted_piston_removed = fourier_basis_limit_wfs_sorted[:,:,1:]

#%% check orthogonality

matrix = np.reshape(fourier_basis_wfs_sorted_piston_removed, (fourier_basis_wfs_sorted_piston_removed.shape[0]*\
                                                                    fourier_basis_wfs_sorted_piston_removed.shape[1],
                                                                    fourier_basis_wfs_sorted_piston_removed.shape[2]))
    
matrix_limit = np.reshape(fourier_basis_limit_wfs_sorted_piston_removed, (fourier_basis_limit_wfs_sorted_piston_removed.shape[0]*\
                                                                    fourier_basis_limit_wfs_sorted_piston_removed.shape[1],
                                                                    fourier_basis_limit_wfs_sorted_piston_removed.shape[2]))
    
#%%

plt.imshow(matrix.T @ matrix)

#%%

plt.imshow(matrix_limit.T @ matrix_limit)

#%% Tilt decomposition

tilt = get_tilt((npx_tel, npx_tel))

tilt_vector = np.reshape(tilt, tilt.shape[0]*tilt.shape[1])

#%%

tilt_decomposition_coefs = matrix.T @ tilt_vector

#tilt_decomposition_coefs = tilt_decomposition_coefs

reconstructed_tilt_vector = matrix @ tilt_decomposition_coefs

#%%

reconstructed_tilt = np.reshape(reconstructed_tilt_vector, (npx_tel, npx_tel))

#%%

plt.figure()
plt.imshow(tilt)

#%%

plt.figure(7)
plt.imshow(reconstructed_tilt)

#%% Use full basis

fourier_basis_tel_piston_remove = fourier_basis_tel_sorted[:,:,1:]

#%%

matrix_full = np.reshape(fourier_basis_tel_piston_remove, (fourier_basis_tel_piston_remove.shape[0]*\
                                                                    fourier_basis_tel_piston_remove.shape[1],
                                                                    fourier_basis_tel_piston_remove.shape[2]))

# check orthogonality
plt.figure()
plt.imshow(matrix_full.T @ matrix_full)
print(np.sum(matrix_full.T @ matrix_full))

tilt_full_decomposition_coefs = matrix_full.T @ tilt_vector

reconstructed_full_tilt_vector = matrix_full @ tilt_full_decomposition_coefs

reconstructed_full_tilt = np.reshape(reconstructed_full_tilt_vector, (npx_tel, npx_tel))

#%%
plt.figure()
plt.imshow(reconstructed_full_tilt)

#%%
plt.figure()
plt.imshow(np.abs(reconstructed_full_tilt-tilt))