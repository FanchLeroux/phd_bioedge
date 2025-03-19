# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:45:20 2025

@author: fleroux
"""

import numpy as np

import pathlib

from fanch.tools.save_load import save_vars

#%% Define path

path = pathlib.Path(__file__).parent

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
param['altitude'      ] = [0, 1000,5000,10000,12000 ]                    # [m] altitude of the different layers

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

param['magnitude'            ] = 0                                          # magnitude of the guide star

# GHOST wavelength : 770 nm, full bandwidth = 20 nm
# 'I2' band : 750 nm, bandwidth? = 33 nm
param['optical_band'          ] = 'I2'                                      # optical band of the guide star

# ------------------------ DM --------------------- #

param['n_actuator'] = 2*param['n_subaperture'] # number of actuators
param['is_dm_modal'] = False                   # set to True if the dm is a modal dm

# ----------------------- WFS ---------------------- #

param['modulation'            ] = 2.                  # [lambda/D] modulation radius or grey width
param['grey_length']            = param['modulation'] # [lambda/D] grey length in case of small grey bioedge WFS
param['n_pix_separation'      ] = 10                  # [pixel] separation ratio between the PWFS pupils
param['psf_centering'          ] = False              # centering of the FFT and of the PWFS mask on the 4 central pixels
param['light_threshold'        ] = 0.3                # light threshold to select the valid pixels
param['post_processing'        ] = 'fullFrame'        # post-processing of the PWFS signals 'slopesMaps' ou 'fullFrame'

# super resolution
param['sr_amplitude']        = 0.5                   # [pixel] super resolution shifts amplitude
param['pupil_shift_bioedge'] = [[param['sr_amplitude'],\
                                 0.,\
                                 param['sr_amplitude'],\
                                 0.],\
                                [param['sr_amplitude'],\
                                0.,\
                                0.,\
                                param['sr_amplitude']]] # [pixel] [sx,sy] to be applied with wfs.apply_shift_wfs() method 
                                                             # (for bioedge)
    
    
param['pupil_shift_pyramid'] = [[param['sr_amplitude'],\
                                 0.,\
                                 0.,\
                                 param['sr_amplitude']],\
                                [param['sr_amplitude'],\
                                param['sr_amplitude'],\
                                0.,\
                                0.]] # [pixel] [sx,sy] to be applied with wfs.apply_shift_wfs() method (for pyramid)

# -------------------- MODAL BASIS ---------------- #

param['modal_basis'] = 'KL'
param['list_modes_to_keep'] = np.linspace(int(0.5*(np.pi * (param['n_subaperture']/2)**2)), 
                                          int(np.pi * param['n_subaperture']**2), num=10, dtype=int)
param['stroke'] = 1e-9 # [m] actuator stoke for calibration matrices computation
param['single_pass'] = False    

# -------------------- LOOP ----------------------- #

param['n_modes_to_show'] = 250

param['n_modes_to_show_sr'] = 850
param['n_modes_to_control_sr'] = 700 # should be inferior to param['n_modes_to_show_sr']

param['n_modes_to_show_oversampled'] = 980

param['loop_gain'] = 0.5

param['n_iter'] = 100

# --------------------- FILENAME -------------------- #

# name of the system
param['filename'] = '_' +  param['optical_band'] +'_band_'+ str(param['n_subaperture'])+'x'+ str(param['n_subaperture'])\
                    + '_' + param['modal_basis'] + '_basis'

# --------------------- FOLDERS --------------------- #

# location of the objects data
param['path_object'] = path / 'data_object'

# location of the calibration data
param['path_calibration'] = path / 'data_calibration'

#%% Save parameter file

save_vars(path / pathlib.Path("parameter_file" + ".pkl"), ['param'])
