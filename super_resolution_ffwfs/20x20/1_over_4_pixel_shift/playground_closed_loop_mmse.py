# -*- coding: utf-8 -*-
"""
Created on Mon May  5 16:18:37 2025

@author: fleroux
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 16:19:11 2025

@author: fleroux
"""

#%%

import pathlib
import sys
import platform

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

from fanch.tools.save_load import save_vars, load_vars

from fanch.tools.oopao import close_the_loop_delay, compute_mmse_reconstructor

from OOPAO.calibration.compute_KL_modal_basis import compute_M2C

from OOPAO.tools.displayTools import cl_plot

import pickle

from astropy.io import fits

#%% Define paths

path = pathlib.Path(__file__).parent
path_data = path.parent.parent.parent.parent / "phd_bioedge_data" / pathlib.Path(*path.parts[-3:]) # could be done better

#%% Get parameter file

path_parameter_file = path_data / "parameter_file.pkl"
load_vars(path_parameter_file, ['param'])

#%% Load objects computed in build_object_file.py 

load_vars(param['path_object'] / pathlib.Path('all_objects'+str(param['filename'])+'.pkl'), 
          ['parameters_object', 'origin_object',\
           'tel','atm', 'dm', 'ngs', 'M2C',\
           #'pyramid', 'pyramid_sr', 'pyramid_oversampled',\
           #'sbioedge', 'sbioedge_sr', 'sbioedge_oversampled',\
           'gbioedge', 'gbioedge_sr', 'gbioedge_oversampled',\
           'sgbioedge', 'sgbioedge_sr','sgbioedge_oversampled'])

#%% get KL modes covariance matrix

# output = compute_M2C(tel, atm, dm, HHtName = 'KL_covariance_matrix', nameFolder=str(pathlib.Path(__file__).parent / "output_compute_M2C"))

# with open(pathlib.Path(__file__).parent / "output_compute_M2CHHt_PSD_df_KL_covariance_matrix.pkl", 'rb') as f:
#     HHt, PSD_atm, df = pickle.load(f)
    
# cov_kl = M2C.T @ HHt @ M2C
    
#%% load calibrations

#load_vars(param['path_calibration'] / pathlib.Path('calibration_pyramid'+param['filename']+'.pkl'))
#load_vars(param['path_calibration'] / pathlib.Path('calibration_sbioedge'+param['filename']+'.pkl'))
load_vars(param['path_calibration'] / pathlib.Path('calibration_gbioedge'+param['filename']+'.pkl'))#, ['calib_gbioedge_sr'])
load_vars(param['path_calibration'] / pathlib.Path('calibration_sgbioedge'+param['filename']+'.pkl'))

#%%

param['n_modes_to_show'] = 250

param['n_modes_to_show_sr'] = 600#850
param['n_modes_to_control_sr'] = 600#700 # should be inferior to param['n_modes_to_show_sr']

param['n_modes_to_show_oversampled'] = 980

#%% Modal Basis

M2C_sr = deepcopy(M2C)
M2C_sr[:, param['n_modes_to_control_sr']:] = 0.

#%% reconstructors - pyramid

# R_pyramid = np.linalg.pinv(calib_pyramid.D[:,:param['n_modes_to_show']])
# R_pyramid_sr = np.linalg.pinv(calib_pyramid_sr.D[:,:param['n_modes_to_show_sr']])
# R_pyramid_oversampled = np.linalg.pinv(calib_pyramid_oversampled.D[:,:param['n_modes_to_show_oversampled']])

# reconstructor_pyramid = M2C[:, :param['n_modes_to_show']]@R_pyramid
# reconstructor_pyramid_sr = M2C_sr[:, :param['n_modes_to_show_sr']]@R_pyramid_sr
# reconstructor_pyramid_oversampled = M2C[:, :param['n_modes_to_show_oversampled']]@R_pyramid_oversampled

#%% reconstructors - gbioedge

R_gbioedge = np.linalg.pinv(calib_gbioedge.D[:,:param['n_modes_to_show']])
R_gbioedge_sr = np.linalg.pinv(calib_gbioedge_sr.D[:,:param['n_modes_to_show_sr']])
R_gbioedge_oversampled = np.linalg.pinv(calib_gbioedge_oversampled.D[:,:param['n_modes_to_show_oversampled']])

reconstructor_gbioedge = M2C[:, :param['n_modes_to_show']]@R_gbioedge
reconstructor_gbioedge_sr = M2C_sr[:, :param['n_modes_to_show_sr']]@R_gbioedge_sr
reconstructor_gbioedge_oversampled = M2C[:, :param['n_modes_to_show_oversampled']]@R_gbioedge_oversampled

#%% reconstructors - sgbioedge

R_sgbioedge = np.linalg.pinv(calib_sgbioedge.D[:,:param['n_modes_to_show']])
R_sgbioedge_sr = np.linalg.pinv(calib_sgbioedge_sr.D[:,:param['n_modes_to_show_sr']])
R_sgbioedge_oversampled = np.linalg.pinv(calib_sgbioedge_oversampled.D[:,:param['n_modes_to_show_oversampled']])

reconstructor_sgbioedge = M2C[:, :param['n_modes_to_show']]@R_sgbioedge
reconstructor_sgbioedge_sr = M2C_sr[:, :param['n_modes_to_show_sr']]@R_sgbioedge_sr
reconstructor_sgbioedge_oversampled = M2C[:, :param['n_modes_to_show_oversampled']]@R_sgbioedge_oversampled

#%% MMSE Reconstructors

# mode covariance

with fits.open(path_data / 'objects' / 'M2C_KL_basis_160_res.fits', 'readonly') as hdul:
    M2C_KL_full = hdul[1].data
       
with open(path_data / 'objects' / 'HHt_PSD_df_KL_covariance_matrix.pkl', 'rb') as f:
    HHt, PSD, df = pickle.load(f)
    
## COVARIANCE OF MODES IN ATMOSPHERE
tpup = tel.pupil.sum()
Cmo_B = (1./tpup**2.) * M2C_KL_full.T @ HHt @ M2C_KL_full *(0.5e-6/(2.*np.pi))**2

## R0 assumption for the MMSE REC
r0_guess=0.15
## Noise level (trust) assumption for the MMSE REC
noise_level_guess  = 100e-9
# n modes shown to the wfs
n_modes_shown_mmse = calib_gbioedge.D.shape[1] + 1 # Number of modes in the inverion

## COVARIANCE PISTON ECLUDEDOF CONTROLLED MODES (PISTON EXCLUDED)
C_phi = np.asmatrix(Cmo_B[1:n_modes_shown_mmse,1:n_modes_shown_mmse])*r0_guess**(-5./3.)

## Noise covariance matrix

C_noise = noise_level_guess**2 * np.asmatrix(np.identity(gbioedge.signal.shape[0]))
C_noise_oversampled = noise_level_guess**2 * np.identity(gbioedge_oversampled.signal.shape[0])

# intereaction matrix in meter
M_gbioedge = np.asmatrix(tel.src.wavelength/(2.*np.pi) * calib_gbioedge.D)

#%%
# compute reconstructor
alpha = 1.
reconstructor_gbioedge_mmse = (M_gbioedge.T @ C_noise.I @ M_gbioedge + alpha*C_phi.I).I @ M_gbioedge.T @ C_noise.I
reconstructor_gbioedge_mmse = np.asarray(M2C @ reconstructor_gbioedge_mmse)

#%% Allocate memory

# total_pyramid = np.zeros(param['n_iter'])
# residual_pyramid = np.zeros(param['n_iter'])
# strehl_pyramid = np.zeros(param['n_iter'])

# total_pyramid_sr = np.zeros(param['n_iter'])
# residual_pyramid_sr = np.zeros(param['n_iter'])
# strehl_pyramid_sr = np.zeros(param['n_iter'])         
                              
# total_pyramid_oversampled = np.zeros(param['n_iter'])
# residual_pyramid_oversampled = np.zeros(param['n_iter'])
# strehl_pyramid_oversampled = np.zeros(param['n_iter'])

total_gbioedge = np.zeros(param['n_iter'])
residual_gbioedge_1_frame_delay= np.zeros(param['n_iter'])
strehl_gbioedge = np.zeros(param['n_iter'])

total_gbioedge_sr = np.zeros(param['n_iter'])
residual_gbioedge_sr = np.zeros(param['n_iter'])
strehl_gbioedge_sr = np.zeros(param['n_iter'])

total_gbioedge_oversampled = np.zeros(param['n_iter'])
residual_gbioedge_oversampled = np.zeros(param['n_iter'])
strehl_gbioedge_oversampled = np.zeros(param['n_iter'])

total_sgbioedge = np.zeros(param['n_iter'])
residual_sgbioedge = np.zeros(param['n_iter'])
strehl_sgbioedge = np.zeros(param['n_iter'])

total_sgbioedge_sr = np.zeros(param['n_iter'])
residual_sgbioedge_sr = np.zeros(param['n_iter'])
strehl_sgbioedge_sr = np.zeros(param['n_iter'])
                               
total_sgbioedge_oversampled = np.zeros(param['n_iter'])
residual_sgbioedge_oversampled = np.zeros(param['n_iter'])
strehl_sgbioedge_oversampled = np.zeros(param['n_iter'])

total_gbioedge_mmse = np.zeros(param['n_iter'])
residual_gbioedge_mmse_1_frame_delay= np.zeros(param['n_iter'])
strehl_gbioedge_mmse = np.zeros(param['n_iter'])

total_gbioedge_mmse_sr = np.zeros(param['n_iter'])
residual_gbioedge_mmse_sr = np.zeros(param['n_iter'])
strehl_gbioedge_mmse_sr = np.zeros(param['n_iter'])         
                              
total_gbioedge_mmse_oversampled = np.zeros(param['n_iter'])
residual_gbioedge_mmse_oversampled = np.zeros(param['n_iter'])
strehl_gbioedge_mmse_oversampled = np.zeros(param['n_iter'])
                                       
total_sgbioedge_mmse = np.zeros(param['n_iter'])
residual_sgbioedge_mmse = np.zeros(param['n_iter'])
strehl_sgbioedge_mmse = np.zeros(param['n_iter'])

total_sgbioedge_mmse_sr = np.zeros(param['n_iter'])
residual_sgbioedge_mmse_sr = np.zeros(param['n_iter'])
strehl_sgbioedge_mmse_sr = np.zeros(param['n_iter'])
                               
total_sgbioedge_mmse_oversampled = np.zeros(param['n_iter'])
residual_sgbioedge_mmse_oversampled = np.zeros(param['n_iter'])
strehl_sgbioedge_mmse_oversampled = np.zeros(param['n_iter'])

#%% close the loop

seed = 12

total, residual, strehl, dm_coefs, turbulence_phase_screens,\
    residual_phase_screens, wfs_frames, short_exposure_psf, buffer_wfs_measure =\
    close_the_loop_delay(tel, ngs, atm, dm, gbioedge, reconstructor_gbioedge, 
                         loop_gain=param['loop_gain'], n_iter=param['n_iter'], 
                         delay=1, seed=seed, save_telemetry=True, save_psf=True)

#%% close the loop - MMSE

from fanch.tools.oopao import close_the_loop_delay

reconstructor_gbioedge_mmse = reconstructor_gbioedge_mmse * gbioedge.telescope.src.wavelength / (2*np.pi)

total, residual_mmse, strehl, dm_coefs, turbulence_phase_screens,\
    residual_phase_screens, wfs_frames, short_exposure_psf, buffer_wfs_measure =\
    close_the_loop_delay(tel, ngs, atm, dm, gbioedge, reconstructor_gbioedge_mmse, 
                         loop_gain=10, n_iter=100, 
                         delay=2, seed=seed, save_telemetry=True, save_psf=True)

plt.plot(residual_mmse)
