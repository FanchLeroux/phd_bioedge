# -*- coding: utf-8 -*-
"""
Created on Mon Aug 25 17:04:13 2025

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

#%% Define parameters
    
# initialize the dictionary
param = {}

# fill the dictionary
# ------------------ ATMOSPHERE ----------------- #
   
param['r0'] = 0.15 # [m] value of r0 in the visibile
param['L0'] = 30 # [m] value of L0 in the visibile
param['fractionnal_r0'] = [0.45, 0.1, 0.1, 
                           0.25, 0.1] # Cn2 profile (percentage)
param['wind_speed'] = [5, 4, 8, 10, 2] # [m.s-1] wind speed of  layers
param['wind_direction'] = [0, 72, 144, 216, 
                           288] # [degrees] wind direction of layers
param['altitude'] = [0, 1000, 5000, 10000, 12000] # [m] altitude of layers
param['seeds'] = range(1)

# ------------------- TELESCOPE ------------------ #

param['diameter'] = 8  # [m] telescope diameter
param['n_subaperture'] = 20  # number of WFS subaperture along the 
                             # telescope diameter
param['n_pixel_per_subaperture'] = 8 # [pixel] sampling of the WFS subapertures
                                     # in telescope pupil space
param['resolution'] = param['n_subaperture']*\
    param['n_pixel_per_subaperture'] # resolution of the telescope driven by 
                                     # the WFS
param['size_subaperture'] = param['diameter']/\
    param['n_subaperture'] # [m] size of a subaperture projected in M1 space
param['sampling_time'] = 1/1000 # [s] loop sampling time
param['centralObstruction'] = 0 # central obstruction in percentage 
                                # of the diameter

# ---------------------- NGS ---------------------- #

param['magnitude'] = 8 # magnitude of the guide star

# phot.R4 = [0.670e-6, 0.300e-6, 7.66e12]
param['optical_band'] = 'R4' # optical band of the guide star

# ------------------------ DM --------------------- #

param['n_actuator'] = param['n_subaperture'] # number of actuators

# ----------------------- WFS ---------------------- #

param['modulation'] = 2. # [lambda/D] modulation radius or grey width
param['grey_length'] = param['modulation'] # [lambda/D] grey length in case of 
                                           # small grey bioedge WFS
param['n_pix_separation'] = 10 # [pixel] separation ratio between the pupils
param['psf_centering'] = False # centering of the FFT and of the mask on 
                               # the 4 central pixels
param['light_threshold'] = 0.3 # light threshold to select the valid pixels
param['post_processing'] = 'fullFrame' # post-processing of the WFS signals 
                                       # ('slopesMaps' or 'fullFrame')
param['detector_photon_noise']   = True
param['detector_read_out_noise']  = 0. # e- RMS

# -------------------- CALIBRATION - MODAL BASIS ---------------- #

param['modal_basis'] = 'KL'
param['stroke'] = 1e-9 # [m] actuator stroke for 
                       # calibration matrices computation
param['single_pass'] = False # push-pull or push only for the calibration
param['compute_M2C_Folder'] = str(pathlib.Path(__file__).parent)

# ----------------------- RECONSTRUCTION ------------------------ #

param['n_modes_to_show_lse'] = 350

# -------------------- LOOP ----------------------- #

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
tel = Telescope(resolution = param['resolution'], # [pixel] resolution of 
                                                  # the telescope
                diameter   = param['diameter'])   # [m] telescope diameter

#% -----------------------     NGS   ----------------------------------

# create the Natural Guide Star object
ngs = Source(optBand     = param['optical_band'], # Source optical band 
                                                  # (see photometry.py)
             magnitude   = param['magnitude'])    # Source Magnitude

#% -----------------------    ATMOSPHERE   ----------------------------

# coupling telescope and source is mandatory to generate Atmosphere object
ngs*tel

# create the Atmosphere object
atm = Atmosphere(telescope     = tel, # Telescope                              
        r0 = param['r0'], # Fried Parameter [m]
        L0 = param['L0'], # Outer Scale [m]
        fractionalR0 = param['fractionnal_r0'], # Cn2 Profile (percentage)
        windSpeed = param['wind_speed'], # [m.s-1] wind speed of layers
        windDirection = param['wind_direction'], # [degrees] wind direction 
                                                 # of layers
        altitude =  param['altitude'      ]) # [m] altitude of layers

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

# grey bioedge - fullFrame
gbioedge_full_frame = BioEdge(nSubap = param['n_subaperture'], 
              telescope = tel,
              modulation = 0.,
              grey_width = param['modulation'], 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = 'fullFrame', 
              psfCentering = param['psf_centering'])

#%% Calibration

dm.coefs = 100. * param['stroke'] * M2C[:,2]
tel.resetOPD()
tel*dm
phase_offset = tel.OPD #* 2*np.pi / tel.src.wavelength

tel.resetOPD()

#%%

calib_full_frame = InteractionMatrix(ngs, tel, dm, gbioedge_full_frame, 
            M2C = M2C, stroke = param['stroke'], 
            phaseOffset = 0,
            single_pass = param['single_pass'],
            noise = 'off', display=True)

calib_full_frame_phase_offset = InteractionMatrix(ngs, tel, dm, 
            gbioedge_full_frame, M2C = M2C, stroke = param['stroke'], 
            phaseOffset = phase_offset, single_pass = param['single_pass'],
            noise = 'off', display=True)

#%% Reconstructor computation

reconstructor_lse_full_frame = M2C[:,:param['n_modes_to_show_lse']] @\
    np.linalg.pinv(calib_full_frame.D[:,:param['n_modes_to_show_lse']])

#%% Extract eigen mode

n_mode = -1

eigen_mode_push_pull = np.zeros(gbioedge_full_frame.bioSignal_2D.shape[:-1])
eigen_mode_push_pull.fill(np.nan)
eigen_mode_push_pull[gbioedge_full_frame.validSignal == True] =\
    calib_full_frame.U[n_mode,:]

slicer_x = np.r_[np.s_[0:18], np.s_[42:78], np.s_[102:120]]
slicer_y = np.r_[np.s_[0:18], np.s_[42:78], np.s_[102:120]]
   
eigen_mode_push_pull = np.delete(np.delete(eigen_mode_push_pull, slicer_x, 1), 
                            slicer_y, 0)

#%% Display

plt.figure()
plt.plot(calib_full_frame.eigenValues, label="push_pull")
# plt.plot(interaction_matrix_push.eigenValues, 
#          label="push", linestyle = 'dashed')
plt.yscale("log")
plt.title("Interaction matrice eigenvalues\nlog scale")
plt.xlabel("Eigen modes")
plt.ylabel("Eigen Values")
plt.legend()

plt.figure()
plt.plot(calib_full_frame_phase_offset.eigenValues, label="push_pull")
# plt.plot(interaction_matrix_push.eigenValues,
#          label="push", linestyle = 'dashed')
plt.yscale("log")
plt.title("Interaction matrice eigenvalues\nlog scale")
plt.xlabel("Eigen modes")
plt.ylabel("Eigen Values")
plt.legend()

#%%

plt.figure()
plt.imshow(eigen_mode_push_pull)
plt.title(f"Eigen mode {n_mode}, push_pull")

# plt.figure()
# plt.imshow(eigen_mode_push)
# plt.title(f"Eigen mode {n_mode}, push")

#%%

