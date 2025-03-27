#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 14:18:06 2025

@author: fleroux
"""

#%%

import numpy as np

import pathlib

from parameter_file import get_parameters

from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from fanch.basis.fourier import compute_real_fourier_basis, extract_subset, extract_vertical_frequencies, extract_diagonal_frequencies
from OOPAO.Pyramid import Pyramid
from OOPAO.BioEdge import BioEdge

from OOPAO.calibration.InteractionMatrix import InteractionMatrix

#%%

path = pathlib.Path(__file__).parent

# %% Parameters

# initialize the dictionary
param = {}

# ------------------ ATMOSPHERE -----------------

param['r0'            ] = 0.1                                            # [m] value of r0 in the visibile
param['L0'            ] = 30                                             # [m] value of L0 in the visibile
param['fractionnal_r0'] = [0.45, 0.1, 0.1, 0.25, 0.1]                    # Cn2 profile (percentage)
param['wind_speed'    ] = [5,4,8,10,2]                                   # [m.s-1] wind speed of the different layers
param['wind_direction'] = [0,72,144,216,288]                             # [degrees] wind direction of the different layers
param['altitude'      ] = [0, 1000,5000,10000,12000 ]                    # [m] altitude of the different layers

# ------------------- TELESCOPE ------------------

param['diameter'               ] = 8                                         # [m] telescope diameter
param['n_subaperture'          ] = 16                                        # number of WFS subaperture along the telescope diameter
param['n_pixel_per_subaperture'] = 8                                         # [pixel] sampling of the WFS subapertures in 
                                                                             # telescope pupil space
param['resolution'             ] = param['n_subaperture']*\
                                   param['n_pixel_per_subaperture']          # resolution of the telescope driven by the WFS
param['size_subaperture'       ] = param['diameter']/param['n_subaperture']  # [m] size of a sub-aperture projected in the M1 space
param['sampling_time'          ] = 1/1000                                    # [s] loop sampling time
param['centralObstruction'     ] = 0                                         # central obstruction in percentage of the diameter

# ---------------------- NGS ----------------------

param['magnitude'            ] = 0                                          # magnitude of the guide star

# GHOST wavelength : 770 nm, full bandwidth = 20 nm
# 'I2' band : 750 nm, bandwidth? = 33 nm
param['optical_band'          ] = 'I2'                                      # optical band of the guide star

# ------------------------ DM ---------------------

param['n_actuator'] = 2*param['n_subaperture'] # number of actuators
param['is_dm_modal'] = False                   # set to True if the dm is a modal dm

# ----------------------- WFS ----------------------

param['modulation'            ] = 5.                  # [lambda/D] modulation radius or grey width
param['n_pix_separation'      ] = 10                   # [pixel] separation ratio between the PWFS pupils
param['psf_centering'          ] = False              # centering of the FFT and of the PWFS mask on the 4 central pixels
param['light_threshold'        ] = 0.3                # light threshold to select the valid pixels
param['post_processing'        ] = 'fullFrame'        # post-processing of the PWFS signals 'slopesMaps' ou 'fullFrame'

# super resolution
param['sr_amplitude']        = 0.25                   # [pixel] super resolution shifts amplitude
param['pupil_shift_bioedge'] = [[param['sr_amplitude'],\
                                 -param['sr_amplitude'],\
                                 param['sr_amplitude'],\
                                 -param['sr_amplitude']],\
                                [param['sr_amplitude'],\
                                -param['sr_amplitude'],\
                                -param['sr_amplitude'],\
                                param['sr_amplitude']]] # [pixel] [sx,sy] to be applied with wfs.apply_shift_wfs() method (for bioedge)
    
    
param['pupil_shift_pyramid'] = [[param['sr_amplitude'],\
                                 -param['sr_amplitude'],\
                                 -param['sr_amplitude'],\
                                 param['sr_amplitude']],\
                                [-param['sr_amplitude'],\
                                -param['sr_amplitude'],\
                                param['sr_amplitude'],\
                                param['sr_amplitude']]] # [pixel] [sx,sy] to be applied with wfs.apply_shift_wfs() method (for pyramid)

# -------------------- MODAL BASIS ----------------

param['modal_basis'] = 'KL'
param['list_modes_to_keep'] = np.linspace(int(0.5*(np.pi * (param['n_subaperture']/2)**2)), 
                                          int(np.pi * param['n_subaperture']**2), num=10, dtype=int)
param['stroke'] = 1e-9 # [m] actuator stoke for calibration matrices computation
param['single_pass'] = False    

# --------------------- FILENAME --------------------

# name of the system
param['filename'] = '_sr_noise_prop_' +  param['optical_band'] +'_band_'+ str(param['n_subaperture'])+'x'+ str(param['n_subaperture'])\
                    + '_' + param['modal_basis'] + '_basis'

# --------------------- FOLDERS ---------------------

# location of the modal basis data
param['path_object'] = path / 'data_object'

# location of the calibration data
param['path_calibration'] = path / 'data_calibration' 

# location of the analysis data
param['path_analysis'] = path / 'data_analysis'

# location of the plots
param['path_plots'] = path / 'plots'

#%% Objects

#%% -----------------------    TELESCOPE   -----------------------------

# create the Telescope object
tel = Telescope(resolution = param['resolution'], # [pixel] resolution of the telescope
                diameter   = param['diameter'])   # [m] telescope diameter

#%% -----------------------     NGS   ----------------------------------

# GHOST wavelength : 770 nm, full bandwidth = 20 nm
# 'I2' band : 750 nm, bandwidth? = 33 nm

# create the Natural Guide Star object
ngs = Source(optBand     = param['optical_band'],          # Source optical band (see photometry.py)
             magnitude   = param['magnitude'])            # Source Magnitude

ngs*tel

#%% -----------------------    ATMOSPHERE   ----------------------------

# create the Atmosphere object
atm = Atmosphere(telescope     = tel,                      # Telescope                              
                 r0            = param['r0'],              # Fried Parameter [m]
                 L0            = param['L0'],              # Outer Scale [m]
                 fractionalR0  = param['fractionnal_r0'],  # Cn2 Profile (percentage)
                 windSpeed     = param['wind_speed'],      # [m.s-1] wind speed of the different layers
                 windDirection = param['wind_direction'],  # [degrees] wind direction of the different layers
                 altitude      =  param['altitude'      ]) # [m] altitude of the different layers

#%% -------------------------     DM   ----------------------------------

if not(param['is_dm_modal']):
    dm = DeformableMirror(tel, nSubap=param['n_actuator'])
    
#%% ------------------------- MODAL BASIS -------------------------------

if param['modal_basis'] == 'KL':
    M2C = compute_KL_basis(tel, atm, dm) # matrix to apply modes on the DM

elif param['modal_basis'] == 'poke':
    M2C = np.identity(dm.nValidAct)

elif param['modal_basis'] == 'Fourier1D_diag':
    fourier_modes = compute_real_fourier_basis(tel.resolution, return_map=True)
    fourier_modes = extract_subset(fourier_modes, 2*param['n_subaperture'])
    fourier_modes = extract_diagonal_frequencies(fourier_modes, complete=False)
    modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))
    modes = modes[:,1:] # remove piston
    M2C = np.identity(modes.shape[1])
    
elif param['modal_basis'] == 'Fourier1D_vert':
    fourier_modes = compute_real_fourier_basis(tel.resolution, return_map=True)
    fourier_modes = extract_vertical_frequencies(fourier_modes)
    modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))
    modes = modes[:,1:] # remove piston
    M2C = np.identity(modes.shape[1])
    
elif param['modal_basis'] == 'Fourier2D':
    from fanch.basis.fourier import compute_real_fourier_basis
    fourier_modes = compute_real_fourier_basis(tel.resolution)
    modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))
    modes = modes[:,1:] # remove piston
    M2C = np.identity(modes.shape[1])
    
elif param['modal_basis'] == 'Fourier2Dsmall':
    from fanch.basis.fourier import compute_real_fourier_basis,\
        extract_subset, sort_real_fourier_basis
    fourier_modes = compute_real_fourier_basis(tel.resolution, return_map=True)
    fourier_modes = extract_subset(fourier_modes, 2*param['n_subaperture'])
    fourier_modes = sort_real_fourier_basis(fourier_modes)
    modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))
    modes = modes[:,1:] # remove piston
    M2C = np.identity(modes.shape[1])
    
elif param['modal_basis'] == 'Fourier2DsmallBis':
    from fanch.basis.fourier import compute_real_fourier_basis
    fourier_modes = compute_real_fourier_basis(tel.resolution)
    fourier_modes = fourier_modes[:,:,:int(np.pi * param['n_subaperture']**2)]
    modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))
    modes = modes[:,1:] # remove piston
    M2C = np.identity(modes.shape[1])
    
#%% -----------------------   MODAL DM   -------------------------------

if param['is_dm_modal']:
    dm = DeformableMirror(tel, nSubap=param['n_actuator'], modes = modes)

#%% --------------------------- WFSs -----------------------------------

# pramid
pyramid = Pyramid(nSubap = param['n_subaperture'], 
              telescope = tel, 
              modulation = param['modulation'], 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              n_pix_edge = param['n_pix_separation'],
              postProcessing = param['post_processing'],
              psfCentering = param['psf_centering'])

# pyramid SR
pyramid_sr = Pyramid(nSubap = param['n_subaperture'], 
              telescope = tel, 
              modulation = param['modulation'], 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              n_pix_edge = param['n_pix_separation'],
              postProcessing = param['post_processing'],
              psfCentering = param['psf_centering'])

pyramid_sr.apply_shift_wfs(param['pupil_shift_pyramid'][0], param['pupil_shift_pyramid'][1], units='pixels')
pyramid_sr.modulation = param['modulation'] # update reference intensities etc.

# pyramid oversampled
pyramid_oversampled = Pyramid(nSubap = 2*param['n_subaperture'], 
              telescope = tel, 
              modulation = param['modulation'], 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              n_pix_edge = param['n_pix_separation'],
              postProcessing = param['post_processing'],
              psfCentering = param['psf_centering'])

#%% Calibrations

calib_pyramid = InteractionMatrix(ngs, atm, tel, dm, pyramid, M2C = M2C, 
                                  stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_pyramid_sr = InteractionMatrix(ngs, atm, tel, dm, pyramid_sr, M2C = M2C, 
                                      stroke = param['stroke'], single_pass=param['single_pass'], display=True)
calib_pyramid_oversampled = InteractionMatrix(ngs, atm, tel, dm, pyramid_oversampled, M2C = M2C, 
                                              stroke = param['stroke'], single_pass=param['single_pass'], display=True)

#%% Analysis

    # ------------------ pyramid ---------------------- #

# extract singular values

singular_values_pyramid = calib_pyramid.eigenValues
singular_values_pyramid_sr = calib_pyramid_sr.eigenValues
singular_values_pyramid_oversampled = calib_pyramid_oversampled.eigenValues

# Calibration matrix truncation, inversion and noise propagation computation

R_gbiedge_oversampled = np.linalg.pinv(calib_pyramid_oversampled.D)
noise_propagation_pyramid_oversampled = np.diag(R_gbiedge_oversampled @ R_gbiedge_oversampled.T)/calib_pyramid_oversampled.D.shape[0]

noise_propagation_pyramid = []
noise_propagation_pyramid_sr = []

for n_modes in param['list_modes_to_keep']:

    R = np.linalg.pinv(calib_pyramid.D[:,:n_modes])
    R_sr = np.linalg.pinv(calib_pyramid_sr.D[:,:n_modes])

    noise_propagation_pyramid.append(np.diag(R @ R.T)/calib_pyramid.D.shape[0])
    noise_propagation_pyramid_sr.append(np.diag(R_sr @ R_sr.T)/calib_pyramid_sr.D.shape[0])
    
#%% Plots

#%% SVD - Normalized Eigenvalues  

    # --------------- pyramid --------------------

plt.figure()
plt.semilogy(singular_values_pyramid/singular_values_pyramid.max(), 'b', label='no SR')
plt.semilogy(singular_values_pyramid_sr/singular_values_pyramid_sr.max(), 'r', label='SR')
plt.semilogy(singular_values_pyramid_oversampled/singular_values_pyramid_oversampled.max(), 'c', label='oversampled')
plt.title('singular_values_pyramid, '+str(param['n_subaperture'])+' subapertures, ' + param['modal_basis'] + ' modes used, 0.25 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')

#%% Noise Propagation - With SR - All WFS

i = 0
for n_modes in param['list_modes_to_keep']:

    plt.figure()
    plt.plot(noise_propagation_pyramid_oversampled, 'm', label='pyramid '+str(2*param['n_subaperture'])+'x'+str(2*param['n_subaperture']))

    plt.plot(noise_propagation_pyramid_sr[i], 'b' , label= 'pyramid '+str(param['n_subaperture'])+'x'+str(param['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    i+=1
    plt.yscale('log')
    plt.title("MPYWFS Uniform noise propagation\n"+str(param['modulation'])+' lambda/D')
    plt.xlabel("mode ("+param['modal_basis']+") index")
    plt.ylabel("np.diag(R @ R.T)/wfs.nSignal")
    plt.legend()
    plt.savefig(param['path_plots'] / pathlib.Path(str(n_modes) + '_modes' + param['filename'] + '.png'), bbox_inches = 'tight')
