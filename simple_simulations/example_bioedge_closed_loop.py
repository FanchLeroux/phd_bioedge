# -*- coding: utf-8 -*-
"""
Created on Mon May 12 14:37:17 2025

@author: fleroux
"""

import numpy as np
import matplotlib.pyplot as plt

from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
from OOPAO.BioEdge import BioEdge
from OOPAO.calibration.InteractionMatrix import InteractionMatrix

#%% define functions

def close_the_loop(tel, ngs, atm, dm, wfs, reconstructor, loop_gain, 
                   n_iter=100, delay=1, photon_noise = False, 
                   read_out_noise = 0., seed=0, 
                   save_telemetry=False, save_psf=False,
                   display=False):
    
    wfs.cam.photonNoise = photon_noise
    wfs.cam.readoutNoise = read_out_noise
    
    ngs*tel
    tel.computePSF() # just to get the shape
    
    # Memory allocation
    
    total = np.zeros(n_iter)     # turbulence phase std [nm]
    residual = np.zeros(n_iter)  # residual phase std [nm]
    strehl = np.zeros(n_iter)    # Strehl Ratio
    
    buffer_wfs_measure = np.zeros([wfs.signal.shape[0]]+[delay])
    
    if save_telemetry:
        dm_coefs = np.zeros([dm.nValidAct, n_iter])
        turbulence_phase_screens = np.zeros([tel.OPD.shape[0],tel.OPD.shape[1]]+[n_iter])
        residual_phase_screens = np.zeros([tel.OPD.shape[0],tel.OPD.shape[1]]+[n_iter])
        wfs_frames = np.zeros([wfs.cam.frame.shape[0],wfs.cam.frame.shape[1]]+[n_iter])
        
    if save_psf:
        short_exposure_psf = np.zeros([tel.PSF.shape[0],tel.PSF.shape[1]] + [n_iter])
    
    # initialization
    
    atm.initializeAtmosphere(tel)
    atm.generateNewPhaseScreen(seed = seed)
    tel+atm
    
    dm.coefs = 0
    
    ngs*tel*dm*wfs

    
    # close the loop
    
    for k in range(n_iter):
        
        atm.update()
        total[k] = np.std(tel.OPD[np.where(tel.pupil>0)])*1e9 # [nm]
        
        if  save_telemetry:
            turbulence_phase_screens[:,:,k] = tel.OPD
        
        ngs*tel*dm*wfs
        
        buffer_wfs_measure = np.roll(buffer_wfs_measure, -1, axis=1)
        buffer_wfs_measure[:,-1] = wfs.signal
        
        if save_telemetry:
            
            residual_phase_screens[:,:,k] = tel.OPD
            dm_coefs[:,k] = dm.coefs
            wfs_frames[:,:,k] = wfs.cam.frame
           
        residual[k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9 # [nm]
        strehl[k] = np.exp(-np.var(tel.src.phase[np.where(tel.pupil>0)]))
           
        dm.coefs = dm.coefs - loop_gain * np.matmul(reconstructor, buffer_wfs_measure[:,0])
           
        if save_psf:
            
            tel.computePSF()
            short_exposure_psf[:,:,k] = tel.PSF
       
    # return
    
    if save_telemetry and save_psf:
        return total, residual, strehl, dm_coefs, turbulence_phase_screens,\
                      residual_phase_screens, wfs_frames, short_exposure_psf
    elif save_telemetry:
        return total, residual, strehl, dm_coefs, turbulence_phase_screens,\
            residual_phase_screens, wfs_frames
    elif save_psf:
        total, residual, strehl, short_exposure_psf
    else:
        return total, residual, strehl

#%% Generate parameter file
    
# initialize the dictionary
param = {}

# fill the dictionary
# ------------------ ATMOSPHERE ----------------- #
   
param['r0'            ] = 0.15                                           # [m] value of r0 in the visibile
param['L0'            ] = 30                                             # [m] value of L0 in the visibile
param['fractionnal_r0'] = [0.45, 0.1, 0.1, 0.25, 0.1]                    # Cn2 profile (percentage)
param['wind_speed'    ] = [5,4,8,10,2]                                   # [m.s-1] wind speed of the different layers
param['wind_direction'] = [0,72,144,216,288]                             # [degrees] wind direction of the different layers
param['altitude'      ] = [0, 1000, 5000, 10000, 12000]                  # [m] altitude of the different layers
param['seeds']          = range(2)

# ------------------- TELESCOPE ------------------ #

param['diameter'               ] = 8                                         # [m] telescope diameter
param['n_subaperture'          ] = 20                                        # number of WFS subaperture along the telescope diameter
param['n_pixel_per_subaperture'] = 8                                         # [pixel] sampling of the WFS subapertures in 
                                                                             # telescope pupil space
param['resolution'             ] = param['n_subaperture']*\
                                   param['n_pixel_per_subaperture']          # resolution of the telescope driven by the WFS
param['size_subaperture'       ] = param['diameter']/param['n_subaperture']  # [m] size of a sub-aperture projected in the M1 space
param['sampling_time'          ] = 1/1000                                     # [s] loop sampling time
param['centralObstruction'     ] = 0                                         # central obstruction in percentage of the diameter

# ---------------------- NGS ---------------------- #

param['magnitude'            ] = 8                                          # magnitude of the guide star

# GHOST wavelength : 770 nm, full bandwidth = 20 nm
# 'I2' band : 750 nm, bandwidth? = 33 nm                                    # phot.R4 = [0.670e-6, 0.300e-6, 7.66e12]
param['optical_band'          ] = 'R4'                                      # optical band of the guide star

# ------------------------ DM --------------------- #

param['n_actuator'] = 2*param['n_subaperture'] # number of actuators

# ----------------------- WFS ---------------------- #

param['modulation'            ] = 2.                  # [lambda/D] modulation radius or grey width
param['grey_length']            = param['modulation'] # [lambda/D] grey length in case of small grey bioedge WFS
param['n_pix_separation'      ] = 10                  # [pixel] separation ratio between the PWFS pupils
param['psf_centering'          ] = False              # centering of the FFT and of the PWFS mask on the 4 central pixels
param['light_threshold'        ] = 0.3                # light threshold to select the valid pixels
param['post_processing'        ] = 'slopesMaps'        # post-processing of the PWFS signals 'slopesMaps' ou 'fullFrame'
param['detector_photon_noise']   = True
param['detector_read_out_noise']  = 0.                 # e- RMS

# super resolution
param['sr_amplitude']        = 0.25                   # [pixel] super resolution shifts amplitude

# [pixel] [sx,sy] to be applied with wfs.apply_shift_wfs() method (for bioedge)
param['pupil_shift_bioedge'] = [[param['sr_amplitude'],\
                                 -param['sr_amplitude'],\
                                 param['sr_amplitude'],\
                                 -param['sr_amplitude']],\
                                [param['sr_amplitude'],\
                                -param['sr_amplitude'],\
                                -param['sr_amplitude'],\
                                param['sr_amplitude']]]

# -------------------- CALIBRATION - MODAL BASIS ---------------- #

param['modal_basis'] = 'KL'
param['list_modes_to_keep'] = np.linspace(int(0.5*(np.pi * (param['n_subaperture']/2)**2)), 
                            int(np.pi * param['n_subaperture']**2), num=10, dtype=int)
param['stroke'] = 1e-9 # [m] actuator stroke for calibration matrices computation
param['single_pass'] = False # push-pull or push only for the calibration

# ----------------------- RECONSTRUCTION ------------------------ #

param['mmse_noise_level_guess'] = 100e-9 # noise level assumption for MMSE reconstruction
param['mmse_alpha'] = 1. # Weight for the turbulence statistics for MMSE reconstruction

# -------------------- LOOP ----------------------- #

param['n_modes_to_show'] = 300

param['loop_gain'] = 0.5

param['n_iter'] = 200

param['delay'] = 2

# --------------------- FILENAME -------------------- #

# name of the system
param['filename'] = '_' +  param['optical_band'] +'_band_'+\
    str(param['n_subaperture'])+'x'+ str(param['n_subaperture'])\
                    + '_' + param['modal_basis'] + '_basis'
                    
#%% Build objects

#% -----------------------    TELESCOPE   -----------------------------

# create the Telescope object
tel = Telescope(resolution = param['resolution'], # [pixel] resolution of the telescope
                diameter   = param['diameter'])   # [m] telescope diameter

#% -----------------------     NGS   ----------------------------------

# create the Natural Guide Star object
ngs = Source(optBand     = param['optical_band'],          # Source optical band (see photometry.py)
             magnitude   = param['magnitude'])            # Source Magnitude

#% -----------------------    ATMOSPHERE   ----------------------------

# coupling the telescope and the source is mandatory to generate Atmosphere object
ngs*tel

# create the Atmosphere object
atm = Atmosphere(telescope     = tel,                      # Telescope                              
                 r0            = param['r0'],              # Fried Parameter [m]
                 L0            = param['L0'],              # Outer Scale [m]
                 fractionalR0  = param['fractionnal_r0'],  # Cn2 Profile (percentage)
                 windSpeed     = param['wind_speed'],      # [m.s-1] wind speed of the different layers
                 windDirection = param['wind_direction'],  # [degrees] wind direction of the different layers
                 altitude      =  param['altitude'      ]) # [m] altitude of the different layers

#%% -------------------------     DM   ----------------------------------

dm = DeformableMirror(tel, nSubap=param['n_actuator'])
    
#%% ------------------------- MODAL BASIS -------------------------------

if param['modal_basis'] == 'KL':
    
    M2C_KL_full, HHt, PSD_atm, df = compute_M2C(telescope= tel,\
                              atmosphere         = atm,\
                              deformableMirror   = dm,\
                              param              = param,\
                              nameFolder         = '',\
                              remove_piston      = False,\
                              HHtName            = 'KL_covariance_matrix',\
                              baseName           = 'KL_basis' ,\
                              mem_available      = 6.1e9,\
                              minimF             = False,\
                              nmo                = 1350,\
                              ortho_spm          = True,\
                              SZ                 = np.int64(2*tel.OPD.shape[0]),\
                              nZer               = 3,\
                              NDIVL              = 1,\
                              lim_inversion=1e-5,
                              returnHHt_PSD_df=True)
        
    M2C = M2C_KL_full[:,1:param['n_modes_to_show']] # remove piston
    
    dm.coefs = np.zeros(dm.nValidAct) # reset dm.OPD
    
elif param['modal_basis'] == 'poke':
    M2C = np.identity(dm.nValidAct)
    
#%% ----------------------- Grey Bi-O-Edge ---------------------------- #

if param['post_processing'] == 'slopesMaps':
    # grey bioedge
    gbioedge = BioEdge(nSubap = param['n_subaperture'], 
                  telescope = tel, 
                  modulation = 0.,
                  grey_width = param['modulation'], 
                  lightRatio = param['light_threshold'],
                  n_pix_separation = param['n_pix_separation'],
                  postProcessing = param['post_processing'], 
                  psfCentering = param['psf_centering'])

elif param['post_processing'] == 'fullFrame':
    # super resolved grey bioedge
    gbioedge = BioEdge(nSubap = param['n_subaperture'], 
                  telescope = tel, 
                  modulation = 0.,
                  grey_width = param['modulation'], 
                  lightRatio = param['light_threshold'],
                  n_pix_separation = param['n_pix_separation'],
                  postProcessing = param['post_processing'], 
                  psfCentering = param['psf_centering'])
    
    gbioedge.apply_shift_wfs(param['pupil_shift_bioedge'][0], 
                             param['pupil_shift_bioedge'][1], units='pixels')
    gbioedge.modulation = 0. # update reference intensities etc.
    
#%% Calibration

calib = InteractionMatrix(ngs, tel, dm, gbioedge, M2C=M2C,
                stroke=param['stroke'], single_pass=param['single_pass'],
                noise = 'off', display=True)

#%% LSE Reconstructor computation

reconstructor_lse = M2C @ np.linalg.pinv(calib.D)

#%% MMSE Reconstructor computation

# COVARIANCE OF MODES IN ATMOSPHERE
C_phi_full_KL_basis = (1./tel.pupil.sum()**2.) * M2C_KL_full.T @ HHt @ M2C_KL_full *(tel.src.wavelength/(2.*np.pi))**2

# COVARIANCE OF CONTROLLED MODES (PISTON EXCLUDED)
C_phi = np.asmatrix(C_phi_full_KL_basis[1:param['n_modes_to_show'],1:param['n_modes_to_show']])*param['r0']**(-5./3.)

# COVARIANCE OF NOISE (assumed to be uncorrelated: Diagonal matrix)
C_n = np.asmatrix(param['mmse_noise_level_guess']**2 * np.identity(gbioedge.nSignal))

### INTERACTION MATRIX "IN METERS"
calib_D_meter = calib.D * ngs.wavelength/(2. * np.pi)

reconstructor_mmse = ngs.wavelength/(2. * np.pi) * np.asarray(M2C @ (calib_D_meter.T @ C_n.I @ calib_D_meter + param['mmse_alpha']*C_phi.I).I @ calib_D_meter.T @ C_n.I)

#%% Close the loop - LSE

seed = 0 # seed for atmosphere computation

total, residual, strehl, dm_coefs, turbulence_phase_screens,\
    residual_phase_screens, wfs_frames, short_exposure_psf =\
    close_the_loop(tel, ngs, atm, dm, gbioedge, reconstructor_lse,
                       param['loop_gain'], param['n_iter'], 
                       delay=param['delay'], photon_noise=param['detector_photon_noise'], 
                       read_out_noise=param['detector_read_out_noise'],  seed=seed, 
                       save_telemetry=True, save_psf=True,
                       display = False)
    
#%% Close the loop - MMSE

seed = 0 # seed for atmosphere computation

total, residual, strehl, dm_coefs, turbulence_phase_screens,\
    residual_phase_screens, wfs_frames, short_exposure_psf =\
    close_the_loop(tel, ngs, atm, dm, gbioedge, reconstructor_mmse,
                       param['loop_gain'], param['n_iter'], 
                       delay=param['delay'], photon_noise=param['detector_photon_noise'], 
                       read_out_noise=param['detector_read_out_noise'],  seed=seed, 
                       save_telemetry=True, save_psf=True,
                       display = False)
    
#%% post processing

long_exposure_psf = np.sum(short_exposure_psf[:,:,100:], axis=2)

#%% plots

plt.figure()
plt.plot(residual)
plt.xlabel('loop iteration')
plt.ylabel('residual phase RMS [nm]')
plt.title('Closed Loop residuals')
plt.ylim(0,200)

plt.figure()
plt.plot(strehl)
plt.xlabel('loop iteration')
plt.ylabel('Strehl Ratio')
plt.title('Closed Loop Strehl Ratio')

plt.figure()
plt.imshow(np.log(long_exposure_psf))
plt.title('Long Exposure PSF (log scale)')

# Bi-O-Edge complex amplitude mask
fig, axs = plt.subplots(ncols=2, nrows=2)
axs[0,0].imshow(np.abs(gbioedge.mask[0]))
axs[0,1].imshow(np.abs(gbioedge.mask[2]))
axs[1,0].imshow(np.angle(gbioedge.mask[0])) # The phase on the "dark" (amplitude = 0) side
axs[1,1].imshow(np.angle(gbioedge.mask[2])) # of the Foucault Knife Edge does not matter

