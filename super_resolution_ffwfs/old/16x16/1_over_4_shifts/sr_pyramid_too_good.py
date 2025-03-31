#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 14:18:06 2025

@author: fleroux
"""

#%%

import matplotlib.pyplot as plt
import numpy as np

import pathlib
import dill

from copy import deepcopy

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

# %% parameterseters

# initialize the dictionary
parameters = {}

# ------------------ ATMOSPHERE -----------------

parameters['r0'            ] = 0.1                                            # [m] value of r0 in the visibile
parameters['L0'            ] = 30                                             # [m] value of L0 in the visibile
parameters['fractionnal_r0'] = [0.45, 0.1, 0.1, 0.25, 0.1]                    # Cn2 profile (percentage)
parameters['wind_speed'    ] = [5,4,8,10,2]                                   # [m.s-1] wind speed of the different layers
parameters['wind_direction'] = [0,72,144,216,288]                             # [degrees] wind direction of the different layers
parameters['altitude'      ] = [0, 1000,5000,10000,12000 ]                    # [m] altitude of the different layers

# ------------------- TELESCOPE ------------------

parameters['diameter'               ] = 8                                         # [m] telescope diameter
parameters['n_subaperture'          ] = 16                                        # number of WFS subaperture along the telescope diameter
parameters['n_pixel_per_subaperture'] = 8                                         # [pixel] sampling of the WFS subapertures in 
                                                                             # telescope pupil space
parameters['resolution'             ] = parameters['n_subaperture']*\
                                   parameters['n_pixel_per_subaperture']          # resolution of the telescope driven by the WFS
parameters['size_subaperture'       ] = parameters['diameter']/parameters['n_subaperture']  # [m] size of a sub-aperture projected in the M1 space
parameters['sampling_time'          ] = 1/1000                                    # [s] loop sampling time
parameters['centralObstruction'     ] = 0                                         # central obstruction in percentage of the diameter

# ---------------------- NGS ----------------------

parameters['magnitude'            ] = 0                                          # magnitude of the guide star

# GHOST wavelength : 770 nm, full bandwidth = 20 nm
# 'I2' band : 750 nm, bandwidth? = 33 nm
parameters['optical_band'          ] = 'I2'                                      # optical band of the guide star

# ------------------------ DM ---------------------

parameters['n_actuator'] = 2*parameters['n_subaperture'] # number of actuators
parameters['is_dm_modal'] = False                   # set to True if the dm is a modal dm

# ----------------------- WFS ----------------------

parameters['modulation'            ] = 5.                  # [lambda/D] modulation radius or grey width
parameters['n_pix_separation'      ] = 10                   # [pixel] separation ratio between the PWFS pupils
parameters['psf_centering'          ] = False              # centering of the FFT and of the PWFS mask on the 4 central pixels
parameters['light_threshold'        ] = 0.3                # light threshold to select the valid pixels
parameters['post_processing'        ] = 'fullFrame'        # post-processing of the PWFS signals 'slopesMaps' ou 'fullFrame'

# super resolution
parameters['sr_amplitude']        = 0.25                   # [pixel] super resolution shifts amplitude
parameters['pupil_shift_bioedge'] = [[parameters['sr_amplitude'],\
                                 -parameters['sr_amplitude'],\
                                 parameters['sr_amplitude'],\
                                 -parameters['sr_amplitude']],\
                                [parameters['sr_amplitude'],\
                                -parameters['sr_amplitude'],\
                                -parameters['sr_amplitude'],\
                                parameters['sr_amplitude']]] # [pixel] [sx,sy] to be applied with wfs.apply_shift_wfs() method (for bioedge)
    
    
parameters['pupil_shift_pyramid'] = [[parameters['sr_amplitude'],\
                                 -parameters['sr_amplitude'],\
                                 -parameters['sr_amplitude'],\
                                 parameters['sr_amplitude']],\
                                [-parameters['sr_amplitude'],\
                                -parameters['sr_amplitude'],\
                                parameters['sr_amplitude'],\
                                parameters['sr_amplitude']]] # [pixel] [sx,sy] to be applied with wfs.apply_shift_wfs() method (for pyramid)

# -------------------- MODAL BASIS ----------------

parameters['modal_basis'] = 'KL'
parameters['list_modes_to_keep'] = np.linspace(int(0.5*(np.pi * (parameters['n_subaperture']/2)**2)), 
                                          int(np.pi * parameters['n_subaperture']**2), num=10, dtype=int)
parameters['stroke'] = 1e-9 # [m] actuator stoke for calibration matrices computation
parameters['single_pass'] = False    

# --------------------- FILENAME --------------------

# name of the system
parameters['filename'] = '_sr_noise_prop_' +  parameters['optical_band'] +'_band_'+ str(parameters['n_subaperture'])+'x'+ str(parameters['n_subaperture'])\
                    + '_' + parameters['modal_basis'] + '_basis'

# --------------------- FOLDERS ---------------------

# location of the modal basis data
parameters['path_object'] = path / 'data_object'

# location of the calibration data
parameters['path_calibration'] = path / 'data_calibration' 

# location of the analysis data
parameters['path_analysis'] = path / 'data_analysis'

# location of the plots
parameters['path_plots'] = path / 'plots'

#%% Objects

#%% Load objects

dill.load_session(parameters['path_object'] / pathlib.Path('object'+str(parameters['filename'])+'.pkl'))

#%%

pyramid_too_good = deepcopy(pyramid)
pyramid_too_good_sr = deepcopy(pyramid_sr)
pyramid_too_good_oversampled = deepcopy(pyramid_oversampled)

del pyramid, pyramid_sr, pyramid_oversampled, tel, atm, ngs, M2C, dm

#%% -----------------------    TELESCOPE   -----------------------------

# create the Telescope object
tel = Telescope(resolution = parameters['resolution'], # [pixel] resolution of the telescope
                diameter   = parameters['diameter'])   # [m] telescope diameter

#%% -----------------------     NGS   ----------------------------------

# GHOST wavelength : 770 nm, full bandwidth = 20 nm
# 'I2' band : 750 nm, bandwidth? = 33 nm

# create the Natural Guide Star object
ngs = Source(optBand     = parameters['optical_band'],          # Source optical band (see photometry.py)
             magnitude   = parameters['magnitude'])            # Source Magnitude

ngs*tel

#%% -----------------------    ATMOSPHERE   ----------------------------

# create the Atmosphere object
atm = Atmosphere(telescope     = tel,                      # Telescope                              
                 r0            = parameters['r0'],              # Fried parameterseter [m]
                 L0            = parameters['L0'],              # Outer Scale [m]
                 fractionalR0  = parameters['fractionnal_r0'],  # Cn2 Profile (percentage)
                 windSpeed     = parameters['wind_speed'],      # [m.s-1] wind speed of the different layers
                 windDirection = parameters['wind_direction'],  # [degrees] wind direction of the different layers
                 altitude      =  parameters['altitude'      ]) # [m] altitude of the different layers

#%% -------------------------     DM   ----------------------------------

if not(parameters['is_dm_modal']):
    dm = DeformableMirror(tel, nSubap=parameters['n_actuator'])
    
#%% ------------------------- MODAL BASIS -------------------------------

if parameters['modal_basis'] == 'KL':
    M2C = compute_KL_basis(tel, atm, dm) # matrix to apply modes on the DM

elif parameters['modal_basis'] == 'poke':
    M2C = np.identity(dm.nValidAct)

elif parameters['modal_basis'] == 'Fourier1D_diag':
    fourier_modes = compute_real_fourier_basis(tel.resolution, return_map=True)
    fourier_modes = extract_subset(fourier_modes, 2*parameters['n_subaperture'])
    fourier_modes = extract_diagonal_frequencies(fourier_modes, complete=False)
    modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))
    modes = modes[:,1:] # remove piston
    M2C = np.identity(modes.shape[1])
    
elif parameters['modal_basis'] == 'Fourier1D_vert':
    fourier_modes = compute_real_fourier_basis(tel.resolution, return_map=True)
    fourier_modes = extract_vertical_frequencies(fourier_modes)
    modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))
    modes = modes[:,1:] # remove piston
    M2C = np.identity(modes.shape[1])
    
elif parameters['modal_basis'] == 'Fourier2D':
    from fanch.basis.fourier import compute_real_fourier_basis
    fourier_modes = compute_real_fourier_basis(tel.resolution)
    modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))
    modes = modes[:,1:] # remove piston
    M2C = np.identity(modes.shape[1])
    
elif parameters['modal_basis'] == 'Fourier2Dsmall':
    from fanch.basis.fourier import compute_real_fourier_basis,\
        extract_subset, sort_real_fourier_basis
    fourier_modes = compute_real_fourier_basis(tel.resolution, return_map=True)
    fourier_modes = extract_subset(fourier_modes, 2*parameters['n_subaperture'])
    fourier_modes = sort_real_fourier_basis(fourier_modes)
    modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))
    modes = modes[:,1:] # remove piston
    M2C = np.identity(modes.shape[1])
    
elif parameters['modal_basis'] == 'Fourier2DsmallBis':
    from fanch.basis.fourier import compute_real_fourier_basis
    fourier_modes = compute_real_fourier_basis(tel.resolution)
    fourier_modes = fourier_modes[:,:,:int(np.pi * parameters['n_subaperture']**2)]
    modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))
    modes = modes[:,1:] # remove piston
    M2C = np.identity(modes.shape[1])
    
#%% -----------------------   MODAL DM   -------------------------------

if parameters['is_dm_modal']:
    dm = DeformableMirror(tel, nSubap=parameters['n_actuator'], modes = modes)

#%% --------------------------- WFSs -----------------------------------

# pramid
pyramid = Pyramid(nSubap = parameters['n_subaperture'], 
              telescope = tel, 
              modulation = parameters['modulation'], 
              lightRatio = parameters['light_threshold'],
              n_pix_separation = parameters['n_pix_separation'],
              n_pix_edge = parameters['n_pix_separation'],
              postProcessing = parameters['post_processing'],
              psfCentering = parameters['psf_centering'])

# pyramid SR
pyramid_sr = Pyramid(nSubap = parameters['n_subaperture'], 
              telescope = tel, 
              modulation = parameters['modulation'], 
              lightRatio = parameters['light_threshold'],
              n_pix_separation = parameters['n_pix_separation'],
              n_pix_edge = parameters['n_pix_separation'],
              postProcessing = parameters['post_processing'],
              psfCentering = parameters['psf_centering'])

pyramid_sr.apply_shift_wfs(parameters['pupil_shift_pyramid'][0], parameters['pupil_shift_pyramid'][1], units='pixels')
pyramid_sr.modulation = parameters['modulation'] # update reference intensities etc.

test = 0
if test:
    pyramid_sr.mask = pyramid_too_good_sr.mask
    pyramid_sr.modulation = parameters['modulation'] # update reference intensities etc.

# pyramid oversampled
pyramid_oversampled = Pyramid(nSubap = 2*parameters['n_subaperture'], 
              telescope = tel, 
              modulation = parameters['modulation'], 
              lightRatio = parameters['light_threshold'],
              n_pix_separation = parameters['n_pix_separation'],
              n_pix_edge = parameters['n_pix_separation'],
              postProcessing = parameters['post_processing'],
              psfCentering = parameters['psf_centering'])

#%% Calibrations - pyramid_too_good

calib_pyramid_too_good = InteractionMatrix(ngs, atm, tel, dm, pyramid_too_good, M2C = M2C, 
                                  stroke = parameters['stroke'], single_pass=parameters['single_pass'], display=True)
calib_pyramid_too_good_sr = InteractionMatrix(ngs, atm, tel, dm, pyramid_too_good_sr, M2C = M2C, 
                                      stroke = parameters['stroke'], single_pass=parameters['single_pass'], display=True)
calib_pyramid_too_good_oversampled = InteractionMatrix(ngs, atm, tel, dm, pyramid_too_good_oversampled, M2C = M2C, 
                                              stroke = parameters['stroke'], single_pass=parameters['single_pass'], display=True)

#%% Calibrations

calib_pyramid = InteractionMatrix(ngs, atm, tel, dm, pyramid, M2C = M2C, 
                                  stroke = parameters['stroke'], single_pass=parameters['single_pass'], display=True)
calib_pyramid_sr = InteractionMatrix(ngs, atm, tel, dm, pyramid_sr, M2C = M2C, 
                                      stroke = parameters['stroke'], single_pass=parameters['single_pass'], display=True)
calib_pyramid_oversampled = InteractionMatrix(ngs, atm, tel, dm, pyramid_oversampled, M2C = M2C, 
                                              stroke = parameters['stroke'], single_pass=parameters['single_pass'], display=True)

#%% Analysis pyramid_too_good

    # ------------------ pyramid_too_good ---------------------- #

# extract singular values

singular_values_pyramid_too_good = calib_pyramid_too_good.eigenValues
singular_values_pyramid_too_good_sr = calib_pyramid_too_good_sr.eigenValues
singular_values_pyramid_too_good_oversampled = calib_pyramid_too_good_oversampled.eigenValues

# Calibration matrix truncation, inversion and noise propagation computation

R_gbiedge_oversampled = np.linalg.pinv(calib_pyramid_too_good_oversampled.D)
noise_propagation_pyramid_too_good_oversampled = np.diag(R_gbiedge_oversampled @ R_gbiedge_oversampled.T)/calib_pyramid_too_good_oversampled.D.shape[0]

noise_propagation_pyramid_too_good = []
noise_propagation_pyramid_too_good_sr = []

for n_modes in parameters['list_modes_to_keep']:

    R = np.linalg.pinv(calib_pyramid_too_good.D[:,:n_modes])
    R_sr = np.linalg.pinv(calib_pyramid_too_good_sr.D[:,:n_modes])

    noise_propagation_pyramid_too_good.append(np.diag(R @ R.T)/calib_pyramid_too_good.D.shape[0])
    noise_propagation_pyramid_too_good_sr.append(np.diag(R_sr @ R_sr.T)/calib_pyramid_too_good_sr.D.shape[0])

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

for n_modes in parameters['list_modes_to_keep']:

    R = np.linalg.pinv(calib_pyramid.D[:,:n_modes])
    R_sr = np.linalg.pinv(calib_pyramid_sr.D[:,:n_modes])

    noise_propagation_pyramid.append(np.diag(R @ R.T)/calib_pyramid.D.shape[0])
    noise_propagation_pyramid_sr.append(np.diag(R_sr @ R_sr.T)/calib_pyramid_sr.D.shape[0])

#%% Plots - pyramid_too_good

#%% SVD - Normalized Eigenvalues  

    # --------------- pyramid --------------------

plt.figure()
plt.semilogy(singular_values_pyramid_too_good/singular_values_pyramid_too_good.max(), 'b', label='no SR')
plt.semilogy(singular_values_pyramid_too_good_sr/singular_values_pyramid_too_good_sr.max(), 'r', label='SR')
plt.semilogy(singular_values_pyramid_too_good_oversampled/singular_values_pyramid_too_good_oversampled.max(), 'c', label='oversampled')
plt.title('singular_values_pyramid_too_good, '+str(parameters['n_subaperture'])+' subapertures, ' + parameters['modal_basis'] + ' modes used, 0.25 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')

#%% Noise Propagation - With SR - All WFS

i = 0
for n_modes in parameters['list_modes_to_keep']:

    plt.figure()
    plt.plot(noise_propagation_pyramid_too_good_oversampled, 'm', label='pyramid_too_good '+str(2*parameters['n_subaperture'])+'x'+str(2*parameters['n_subaperture']))

    plt.plot(noise_propagation_pyramid_too_good_sr[i], 'b' , label= 'pyramid_too_good '+str(parameters['n_subaperture'])+'x'+str(parameters['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    i+=1
    plt.yscale('log')
    plt.title("MPYWFS Uniform noise propagation\n"+str(parameters['modulation'])+' lambda/D')
    plt.xlabel("mode ("+parameters['modal_basis']+") index")
    plt.ylabel("np.diag(R @ R.T)/wfs.nSignal")
    plt.legend()
    #plt.savefig(parameters['path_plots'] / pathlib.Path(str(n_modes) + '_modes' + parameters['filename'] + '.png'), bbox_inches = 'tight')


#%% Plots - pyramid

#%% SVD - Normalized Eigenvalues  

    # --------------- pyramid --------------------

plt.figure()
plt.semilogy(singular_values_pyramid/singular_values_pyramid.max(), 'b', label='no SR')
plt.semilogy(singular_values_pyramid_sr/singular_values_pyramid_sr.max(), 'r', label='SR')
plt.semilogy(singular_values_pyramid_oversampled/singular_values_pyramid_oversampled.max(), 'c', label='oversampled')
plt.title('singular_values_pyramid, '+str(parameters['n_subaperture'])+' subapertures, ' + parameters['modal_basis'] + ' modes used, 0.25 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')

#%% Noise Propagation - With SR - All WFS

i = 0
for n_modes in parameters['list_modes_to_keep']:

    plt.figure()
    plt.plot(noise_propagation_pyramid_oversampled, 'm', label='pyramid '+str(2*parameters['n_subaperture'])+'x'+str(2*parameters['n_subaperture']))

    plt.plot(noise_propagation_pyramid_sr[i], 'b' , label= 'pyramid '+str(parameters['n_subaperture'])+'x'+str(parameters['n_subaperture'])+' - SR - '+str(n_modes)+" modes")
    i+=1
    plt.yscale('log')
    plt.title("MPYWFS Uniform noise propagation\n"+str(parameters['modulation'])+' lambda/D')
    plt.xlabel("mode ("+parameters['modal_basis']+") index")
    plt.ylabel("np.diag(R @ R.T)/wfs.nSignal")
    plt.legend()
    #plt.savefig(parameters['path_plots'] / pathlib.Path(str(n_modes) + '_modes' + parameters['filename'] + '.png'), bbox_inches = 'tight')

#%% Mask analysis

plt.figure()
plt.imshow(np.abs(pyramid_too_good.referencePyramidFrame - pyramid_too_good_sr.referencePyramidFrame))

plt.figure()
plt.imshow(np.abs(pyramid_sr.referencePyramidFrame - pyramid.referencePyramidFrame))           

plt.figure()
plt.plot(np.unwrap(np.diag(np.angle(pyramid_sr.mask))), 'b', label='pyramid_sr')
plt.plot(np.unwrap(np.diag(np.angle(pyramid_too_good_sr.mask))), '--r', label='pyramid_too_good_sr')
plt.legend()
