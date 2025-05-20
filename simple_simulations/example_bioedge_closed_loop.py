# -*- coding: utf-8 -*-
"""
Created on Mon May 12 14:37:17 2025

@author: fleroux
"""

import pathlib

import numpy as np
import matplotlib.pyplot as plt

from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
from OOPAO.BioEdge import BioEdge
from OOPAO.calibration.InteractionMatrix import InteractionMatrix

#%%

dirc = pathlib.Path(__file__).parent

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
        wfs_signals = np.zeros([wfs.signal.shape[0]]+[n_iter])
        
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
            wfs_signals[:,k] = buffer_wfs_measure[:,0]
           
        residual[k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9 # [nm]
        strehl[k] = np.exp(-np.var(tel.src.phase[np.where(tel.pupil>0)]))
           
        dm.coefs = dm.coefs - loop_gain * reconstructor @ buffer_wfs_measure[:,0]
           
        if save_psf:
            
            tel.computePSF()
            short_exposure_psf[:,:,k] = tel.PSF
       
    # return
    
    if save_telemetry and save_psf:
        return total, residual, strehl, dm_coefs, turbulence_phase_screens,\
                      residual_phase_screens, wfs_frames, wfs_signals, short_exposure_psf
    elif save_telemetry:
        return total, residual, strehl, dm_coefs, turbulence_phase_screens,\
            residual_phase_screens, wfs_frames
    elif save_psf:
        total, residual, strehl, short_exposure_psf
    else:
        return total, residual, strehl
    
# pseudo-open loop mmse reconstruction
def close_the_loop_pol(tel, ngs, atm, dm, wfs, M2C, 
                   interaction_matrix, reconstructor, loop_gain, 
                   n_iter=100, delay=1, photon_noise = False, 
                   read_out_noise = 0., seed=0, 
                   save_telemetry=False, save_psf=False,
                   display=False):
    
    C2M = np.linalg.pinv(M2C)
    
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
        wfs_signals = np.zeros([wfs.signal.shape[0]]+[n_iter])
        
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
            wfs_signals[:,k] = buffer_wfs_measure[:,0]
           
        residual[k]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9 # [nm]
        strehl[k] = np.exp(-np.var(tel.src.phase[np.where(tel.pupil>0)]))
        
        pseudo_open_loop_measures = buffer_wfs_measure[:,0] - interaction_matrix @ C2M @ dm.coefs
        
        dm.coefs = (1-loop_gain) * dm.coefs - loop_gain * M2C @ reconstructor @ pseudo_open_loop_measures 
        
        if save_psf:
            
            tel.computePSF()
            short_exposure_psf[:,:,k] = tel.PSF
       
    # return
    
    if save_telemetry and save_psf:
        return total, residual, strehl, dm_coefs, turbulence_phase_screens,\
                      residual_phase_screens, wfs_frames, wfs_signals, short_exposure_psf
    elif save_telemetry:
        return total, residual, strehl, dm_coefs, turbulence_phase_screens,\
            residual_phase_screens, wfs_frames
    elif save_psf:
        total, residual, strehl, short_exposure_psf
    else:
        return total, residual, strehl

#%% Define parameters
    
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
param['seeds']          = range(1)

# ------------------- TELESCOPE ------------------ #

param['diameter'               ] = 8                                         # [m] telescope diameter
param['n_subaperture'          ] = 20                                        # number of WFS subaperture along the telescope diameter
param['n_pixel_per_subaperture'] = 8                                         # [pixel] sampling of the WFS subapertures in 
                                                                             # telescope pupil space
param['resolution'             ] = param['n_subaperture']*\
                                   param['n_pixel_per_subaperture']          # resolution of the telescope driven by the WFS
param['size_subaperture'       ] = param['diameter']/param['n_subaperture']  # [m] size of a sub-aperture projected in the M1 space
param['sampling_time'          ] = 1/1000                                    # [s] loop sampling time
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
param['post_processing'        ] = 'fullFrame'       # post-processing of the WFS signals ('slopesMaps' or 'fullFrame')
param['detector_photon_noise']   = True
param['detector_read_out_noise']  = 0.                # e- RMS

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
param['stroke'] = 1e-9 # [m] actuator stroke for calibration matrices computation
param['single_pass'] = False # push-pull or push only for the calibration
param['compute_M2C_Folder'] = str(pathlib.Path(__file__).parent)

# ----------------------- RECONSTRUCTION ------------------------ #

param['mmse_noise_level_guess_slopes_maps'] = 10e-9 # noise level assumption for MMSE reconstruction
param['mmse_noise_level_guess_full_frame'] = 5e-11 # noise level assumption for MMSE full frame reconstruction
param['mmse_alpha'] = 1. # Weight for the turbulence statistics for MMSE reconstruction

# -------------------- LOOP ----------------------- #

param['n_modes_to_show_lse'] = 310
param['n_modes_to_show_lse_sr'] = 750
param['n_modes_to_show_mmse'] = 1300
param['n_modes_to_show_mmse_sr'] = 1300

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
                            nameFolder         = param['compute_M2C_Folder'],\
                            remove_piston      = False,\
                            HHtName            = 'KL_covariance_matrix',\
                            baseName           = 'KL_basis' ,\
                            mem_available      = 6.1e9,\
                            minimF             = False,\
                            nmo                = None,\
                            ortho_spm          = True,\
                            SZ                 = np.int64(2*tel.OPD.shape[0]),\
                            nZer               = 3,\
                            NDIVL              = 1,\
                            lim_inversion=1e-5,
                            returnHHt_PSD_df=True)
        
    M2C = M2C_KL_full[:,1:] # remove piston
    
    dm.coefs = np.zeros(dm.nValidAct) # reset dm.OPD
    
elif param['modal_basis'] == 'poke':
    M2C = np.identity(dm.nValidAct)
    
#%% ----------------------- Grey Bi-O-Edge ---------------------------- #

# grey bioedge
gbioedge_slopes_maps = BioEdge(nSubap = param['n_subaperture'], 
              telescope = tel,
              modulation = 0.,
              grey_width = param['modulation'], 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = 'slopesMaps', 
              psfCentering = param['psf_centering'])

# grey bioedge - fullFrame
gbioedge_full_frame = BioEdge(nSubap = param['n_subaperture'], 
              telescope = tel,
              modulation = 0.,
              grey_width = param['modulation'], 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = 'fullFrame', 
              psfCentering = param['psf_centering'])


# super resolved grey bioedge
gbioedge_sr = BioEdge(nSubap = param['n_subaperture'], 
              telescope = tel, 
              modulation = 0.,
              grey_width = param['modulation'], 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = 'fullFrame', 
              psfCentering = param['psf_centering'])

gbioedge_sr.apply_shift_wfs(param['pupil_shift_bioedge'][0], 
                         param['pupil_shift_bioedge'][1], units='pixels')
gbioedge_sr.modulation = 0. # update reference intensities etc.
    
#%% Calibration

calib_slopes_maps = InteractionMatrix(ngs, tel, dm, gbioedge_slopes_maps, M2C=M2C,
                stroke=param['stroke'], single_pass=param['single_pass'],
                noise = 'off', display=True)

calib_full_frame = InteractionMatrix(ngs, tel, dm, gbioedge_full_frame, M2C=M2C,
                stroke=param['stroke'], single_pass=param['single_pass'],
                noise = 'off', display=True)

calib_sr = InteractionMatrix(ngs, tel, dm, gbioedge_sr, M2C=M2C,
                stroke=param['stroke'], single_pass=param['single_pass'],
                noise = 'off', display=True)

#%% LSE Reconstructor computation

reconstructor_lse_slopes_maps = M2C[:,:param['n_modes_to_show_lse']] @ np.linalg.pinv(calib_slopes_maps.D[:,:param['n_modes_to_show_lse']])
reconstructor_lse_full_frame = M2C[:,:param['n_modes_to_show_lse']] @ np.linalg.pinv(calib_full_frame.D[:,:param['n_modes_to_show_lse']])
reconstructor_lse_sr = M2C[:,:param['n_modes_to_show_lse_sr']] @ np.linalg.pinv(calib_sr.D[:,:param['n_modes_to_show_lse_sr']])

#%% MMSE

# Covariance of KL modes in atmosphere - piston included
C_phi_full_KL_basis = (1./tel.pupil.sum()**2.) * M2C_KL_full.T @ HHt @ M2C_KL_full *(tel.src.wavelength/(2.*np.pi))**2

# COVARIANCE OF CONTROLLED MODES (PISTON EXCLUDED) - no SR
C_phi = np.asmatrix(C_phi_full_KL_basis[1:param['n_modes_to_show_mmse']+1,1:param['n_modes_to_show_mmse']+1])*param['r0']**(-5./3.)

# COVARIANCE OF CONTROLLED MODES (PISTON EXCLUDED) - SR
C_phi_sr = np.asmatrix(C_phi_full_KL_basis[1:param['n_modes_to_show_mmse_sr']+1,1:param['n_modes_to_show_mmse_sr']+1])*param['r0']**(-5./3.)

#%% MMSE Reconstructor computation - slopesMaps

# COVARIANCE OF NOISE (assumed to be uncorrelated: Diagonal matrix)
C_n_slopes_maps = np.asmatrix(param['mmse_noise_level_guess_slopes_maps']**2 * np.identity(gbioedge_slopes_maps.nSignal))

### INTERACTION MATRIX "IN METERS"
calib_D_meter_slopes_maps = calib_slopes_maps.D * ngs.wavelength/(2. * np.pi)

reconstructor_mmse_slopes_maps = np.zeros(calib_D_meter_slopes_maps.T.shape)

reconstructor_mmse_slopes_maps[:param['n_modes_to_show_mmse'], :] = ngs.wavelength/(2. * np.pi) *\
    np.asarray((calib_D_meter_slopes_maps[:,:param['n_modes_to_show_mmse']].T\
    @ C_n_slopes_maps.I @ calib_D_meter_slopes_maps[:,:param['n_modes_to_show_mmse']]\
    + param['mmse_alpha']*C_phi.I).I\
    @ calib_D_meter_slopes_maps[:,:param['n_modes_to_show_mmse']].T @ C_n_slopes_maps.I)
        
#%% MMSE Reconstructor computation - fullFrame

# COVARIANCE OF NOISE (assumed to be uncorrelated: Diagonal matrix)
C_n_full_frame = np.asmatrix(param['mmse_noise_level_guess_full_frame']**2 * np.identity(gbioedge_full_frame.nSignal))

### INTERACTION MATRIX "IN METERS"
calib_D_full_frame_meter = calib_full_frame.D * ngs.wavelength/(2. * np.pi)

reconstructor_mmse_full_frame = np.zeros(calib_full_frame.D.T.shape)

reconstructor_mmse_full_frame[:param['n_modes_to_show_mmse'], :] = ngs.wavelength/(2. * np.pi) *\
    np.asarray((calib_D_full_frame_meter[:,:param['n_modes_to_show_mmse']].T\
    @ C_n_full_frame.I @ calib_D_full_frame_meter[:,:param['n_modes_to_show_mmse']]\
    + param['mmse_alpha']*C_phi.I).I\
    @ calib_D_full_frame_meter[:,:param['n_modes_to_show_mmse']].T @ C_n_full_frame.I)
        
#%% MMSE - Reconstructor computation - SR

# COVARIANCE OF NOISE (assumed to be uncorrelated: Diagonal matrix)
C_n_sr = np.asmatrix(param['mmse_noise_level_guess_full_frame']**2 * np.identity(gbioedge_sr.nSignal))

### INTERACTION MATRIX "IN METERS"
calib_sr_D_meter = calib_sr.D * ngs.wavelength/(2. * np.pi)

reconstructor_mmse_sr = np.zeros(calib_sr.D.T.shape)

reconstructor_mmse_sr[:param['n_modes_to_show_mmse_sr'], :] = ngs.wavelength/(2. * np.pi) *\
    np.asarray((calib_sr_D_meter[:,:param['n_modes_to_show_mmse_sr']].T\
    @ C_n_sr.I @ calib_sr_D_meter[:,:param['n_modes_to_show_mmse_sr']]\
    + param['mmse_alpha']*C_phi.I).I\
    @ calib_sr_D_meter[:,:param['n_modes_to_show_mmse_sr']].T @ C_n_sr.I)

#%% SEED

seed = 12 # seed for atmosphere computation

#%% Close the loop - LSE - slopesMaps

total_lse_slopes_maps, residual_lse_slopes_maps, strehl_lse_slopes_maps, dm_coefs_lse_slopes_maps, turbulence_phase_screens_lse_slopes_maps,\
    residual_phase_screens_lse_slopes_maps, wfs_frames_lse_slopes_maps, wfs_signals_lse_slopes_maps, short_exposure_psf_lse_slopes_maps =\
    close_the_loop(tel, ngs, atm, dm, gbioedge_slopes_maps, reconstructor_lse_slopes_maps,
                       param['loop_gain'], param['n_iter'], 
                       delay=param['delay'], photon_noise=param['detector_photon_noise'], 
                       read_out_noise=param['detector_read_out_noise'],  seed=seed, 
                       save_telemetry=True, save_psf=True,
                       display = False)
    
#%% Close the loop - LSE - fullFrame

total_lse_full_frame, residual_lse_full_frame, strehl_lse_full_frame, dm_coefs_lse_full_frame, turbulence_phase_screens_lse_full_frame,\
    residual_phase_screens_lse_full_frame, wfs_frames_lse_full_frame, wfs_signals_lse_full_frame, short_exposure_psf_lse_full_frame =\
    close_the_loop(tel, ngs, atm, dm, gbioedge_full_frame, reconstructor_lse_full_frame,
                       param['loop_gain'], param['n_iter'], 
                       delay=param['delay'], photon_noise=param['detector_photon_noise'], 
                       read_out_noise=param['detector_read_out_noise'],  seed=seed, 
                       save_telemetry=True, save_psf=True,
                       display = False)
    
#%% Close the loop - LSE - SR

total_lse_sr, residual_lse_sr, strehl_lse_sr, dm_coefs_lse_sr, turbulence_phase_screens_lse_sr,\
    residual_phase_screens_lse_sr, wfs_frames_lse_sr, wfs_signals_lse_sr, short_exposure_psf_lse_sr =\
    close_the_loop(tel, ngs, atm, dm, gbioedge_sr, reconstructor_lse_sr,
                       param['loop_gain'], param['n_iter'],
                       delay=param['delay'], photon_noise=param['detector_photon_noise'], 
                       read_out_noise=param['detector_read_out_noise'],  seed=seed, 
                       save_telemetry=True, save_psf=True,
                       display = False)
    
#%% Close the loop - MMSE - SlopesMaps - pseudo open loop

total_mmse_slopes_maps, residual_mmse_slopes_maps, strehl_mmse_slopes_maps, dm_coefs_mmse_slopes_maps, turbulence_phase_screens_mmse_slopes_maps,\
    residual_phase_screens_mmse_slopes_maps, wfs_frames_mmse_slopes_maps, wfs_signals_mmse_slopes_maps, short_exposure_psf_mmse_slopes_maps =\
    close_the_loop_pol(tel, ngs, atm, dm, gbioedge_slopes_maps, M2C,
                       calib_slopes_maps.D, reconstructor_mmse_slopes_maps,
                       param['loop_gain'], param['n_iter'], 
                       delay=param['delay'], photon_noise=param['detector_photon_noise'], 
                       read_out_noise=param['detector_read_out_noise'],  seed=seed, 
                       save_telemetry=True, save_psf=True,
                       display = False)
    
#%% Close the loop - MMSE - fullFrame - pseudo open loop

total_mmse_full_frame, residual_mmse_full_frame, strehl_mmse_full_frame, dm_coefs_mmse_full_frame, turbulence_phase_screens_mmse_full_frame,\
    residual_phase_screens_mmse_full_frame, wfs_frames_mmse_full_frame, wfs_signals_mmse_full_frame, short_exposure_psf_mmse_full_frame =\
    close_the_loop_pol(tel, ngs, atm, dm, gbioedge_full_frame, M2C,
                       calib_full_frame.D, reconstructor_mmse_full_frame,
                       param['loop_gain'], param['n_iter'], 
                       delay=param['delay'], photon_noise=param['detector_photon_noise'], 
                       read_out_noise=param['detector_read_out_noise'],  seed=seed, 
                       save_telemetry=True, save_psf=True,
                       display = False)

#%% Close the loop - MMSE - SR - pseudo open loop 

total_mmse_sr, residual_mmse_sr, strehl_mmse_sr, dm_coefs_mmse_sr, turbulence_phase_screens_mmse_sr,\
    residual_phase_screens_mmse_sr, wfs_frames_mmse_sr, wfs_signals_mmse_sr, short_exposure_psf_mmse_sr =\
    close_the_loop_pol(tel, ngs, atm, dm, gbioedge_sr, M2C,
                       calib_sr.D, reconstructor_mmse_sr,
                       param['loop_gain'], param['n_iter'], 
                       delay=param['delay'], photon_noise=param['detector_photon_noise'], 
                       read_out_noise=param['detector_read_out_noise'],  seed=seed, 
                       save_telemetry=True, save_psf=True,
                       display = False)

#%% post processing

long_exposure_psf_lse_slopes_maps = np.sum(short_exposure_psf_lse_slopes_maps[:,:,100:], axis=2)
long_exposure_psf_lse_full_frame = np.sum(short_exposure_psf_lse_full_frame[:,:,100:], axis=2)
long_exposure_psf_lse_sr = np.sum(short_exposure_psf_lse_sr[:,:,100:], axis=2)
long_exposure_psf_mmse_slopes_maps = np.sum(short_exposure_psf_mmse_slopes_maps[:,:,100:], axis=2)
long_exposure_psf_mmse_full_frame = np.sum(short_exposure_psf_mmse_full_frame[:,:,100:], axis=2)
long_exposure_psf_mmse_sr = np.sum(short_exposure_psf_mmse_sr[:,:,100:], axis=2)

#%% plots

# residuals
plt.figure()
plt.plot(residual_lse_slopes_maps, label='residual_lse_slopes_maps')
plt.plot(residual_lse_full_frame, label='residual_lse_full_frame')
plt.plot(residual_lse_sr, label='residual_lse_sr')
plt.plot(residual_mmse_slopes_maps, label='residual_mmse_slopes_maps')
plt.plot(residual_mmse_full_frame, label='residual_mmse_full_frame')
plt.plot(residual_mmse_sr, label='residual_mmse_sr')
plt.xlabel('loop iteration')
plt.ylabel('residual phase RMS [nm]')
plt.title('Closed Loop residuals')
plt.legend()
plt.ylim(0,200)
plt.savefig(dirc / pathlib.Path('residuals'+'.png'), bbox_inches = 'tight')

#%%

# strehls
plt.figure()
plt.plot(strehl_lse_slopes_maps, label='strehl_lse_slopes_maps')
plt.plot(strehl_lse_full_frame, label='strehl_lse_full_frame')
plt.plot(strehl_lse_sr, label='strehl_lse_sr')
plt.plot(strehl_mmse_slopes_maps, label='strehl_mmse_slopes_maps')
plt.plot(strehl_mmse_full_frame, label='strehl_mmse_full_frame')
plt.plot(strehl_mmse_sr, label='strehl_mmse_sr')
plt.xlabel('loop iteration')
plt.ylabel('strehl phase RMS [nm]')
plt.title('Closed Loop strehls')
plt.legend()
plt.savefig(dirc / pathlib.Path('strehls'+'.png'), bbox_inches = 'tight')

#%%

fig, axs = plt.subplots(nrows=2, ncols=3)
axs[0,0].imshow(np.log(long_exposure_psf_lse_slopes_maps))
axs[0,0].set_title('long_exposure_psf_lse_slopes_maps')
axs[0,1].imshow(np.log(long_exposure_psf_lse_full_frame))
axs[0,1].set_title('long_exposure_psf_lse_full_frame')
axs[0,2].imshow(np.log(long_exposure_psf_lse_sr))
axs[0,2].set_title('long_exposure_psf_lse_sr')
axs[1,0].imshow(np.log(long_exposure_psf_mmse_slopes_maps))
axs[1,0].set_title('long_exposure_psf_mmse_slopes_maps')
axs[1,1].imshow(np.log(long_exposure_psf_mmse_full_frame))
axs[1,1].set_title('long_exposure_psf_mmse_full_frame')
axs[1,2].imshow(np.log(long_exposure_psf_mmse_sr))
axs[1,2].set_title('long_exposure_psf_mmse_sr')

#%% debug

#%% MMSE reconstruction - Slopesmaps

tel.resetOPD()
dm.coefs = 0
gbioedge_slopes_maps.signal = np.zeros(gbioedge_slopes_maps.signal.shape)
gbioedge_slopes_maps.cam.photonNoise = False

stroke = 1e-9

dm.coefs = stroke*M2C
ngs*tel*dm*gbioedge_slopes_maps

#%%

reconstructed_modes_mmse_slopes_maps = reconstructor_mmse_slopes_maps @ gbioedge_slopes_maps.signal

#%%

plt.figure()
plt.imshow(reconstructed_modes_mmse_slopes_maps)

#%%

plt.figure()
plt.plot(reconstructed_modes_mmse_slopes_maps[:,10]/stroke)

#%%

plt.figure()
plt.plot(np.diag(reconstructed_modes_mmse_slopes_maps)/stroke)

#%% MMSE reconstruction - fullFrame

tel.resetOPD()
dm.coefs = 0
gbioedge_full_frame.signal = np.zeros(gbioedge_full_frame.signal.shape)
gbioedge_full_frame.cam.photonNoise = False

stroke = 1e-9

dm.coefs = stroke*M2C
ngs*tel*dm*gbioedge_full_frame

#%%

reconstructed_modes_mmse_full_frame = reconstructor_mmse_full_frame @ gbioedge_full_frame.signal

#%%

plt.figure()
plt.imshow(reconstructed_modes_mmse_full_frame)

#%%

plt.figure()
plt.plot(reconstructed_modes_mmse_full_frame[:,10]/stroke)

#%%

plt.figure()
plt.plot(np.diag(reconstructed_modes_mmse_full_frame)/stroke)

#%% MMSE reconstruction - SR

tel.resetOPD()
dm.coefs = 0
gbioedge_sr.signal = np.zeros(gbioedge_sr.signal.shape)
gbioedge_sr.cam.photonNoise = False

stroke = 1e-9

dm.coefs = stroke*M2C
ngs*tel*dm*gbioedge_sr

#%%

reconstructed_modes_mmse_sr = reconstructor_mmse_sr @ gbioedge_sr.signal

#%%

plt.figure()
plt.imshow(reconstructed_modes_mmse_sr)

#%%

plt.figure()
plt.plot(reconstructed_modes_mmse_sr[:,10]/stroke)

#%%

plt.figure()
plt.plot(np.diag(reconstructed_modes_mmse_sr)/stroke)

#%% all

fig, axs = plt.subplots(nrows=2, ncols=3)
axs[0,0].imshow(reconstructed_modes_mmse_slopes_maps)
axs[0,0].set_title("reconstructed_modes_mmse_slopes_maps")
axs[0,1].imshow(reconstructed_modes_mmse_full_frame)
axs[0,1].set_title("reconstructed_modes_mmse_full_frame")
axs[0,2].imshow(reconstructed_modes_mmse_sr)
axs[0,2].set_title("reconstructed_modes_mmse_sr")
axs[1,0].plot(np.diag(reconstructed_modes_mmse_slopes_maps)/stroke)
axs[1,0].set_title('np.diag(reconstructed_modes_mmse_slopes_maps)/stroke')
axs[1,1].plot(np.diag(reconstructed_modes_mmse_full_frame)/stroke)
axs[1,1].set_title('np.diag(reconstructed_modes_mmse_full_frame)/stroke')
axs[1,2].plot(np.diag(reconstructed_modes_mmse_sr)/stroke)
axs[1,2].set_title('np.diag(reconstructed_modes_mmse_sr)/stroke')
