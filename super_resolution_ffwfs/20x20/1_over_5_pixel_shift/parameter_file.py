# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:45:20 2025

@author: fleroux
"""

import numpy as np

import pathlib

from fanch.tools.save_load import save_vars

#%% Define paths

path = pathlib.Path(__file__).parent
path_data = path.parent.parent.parent.parent / "phd_bioedge_data" / pathlib.Path(*path.parts[-3:]) # could be done better

#%% Create data folders

path_object = path_data / "objects"
path_object.mkdir(parents=True, exist_ok=True)

path_calibration = path_data / "calibrations"
path_calibration.mkdir(parents=True, exist_ok=True)

path_closed_loop = path_data / "closed_loop"
path_closed_loop.mkdir(parents=True, exist_ok=True)

path_analysis_closed_loop = path_data / "analysis" / "closed_loop"
path_analysis_closed_loop.mkdir(parents=True, exist_ok=True)

path_analysis_uniform_noise_propagation = path_data / "analysis" / "uniform_noise_propagation"
path_analysis_uniform_noise_propagation.mkdir(parents=True, exist_ok=True)

path_plots_closed_loop = path_data / "plots" / "closed_loop"
path_plots_closed_loop.mkdir(parents=True, exist_ok=True)

path_plots_uniform_noise_propagation = path_data / "plots" / "uniform_noise_propagation"
path_plots_uniform_noise_propagation.mkdir(parents=True, exist_ok=True)

#%% Generate parameter file
    
# initialize the dictionary
param = {}

# fill the dictionary

# ----------------- DATA FOLDERS ---------------- # # str type used for windows / linux compatibility

param['path_object'] = str(path_object)
param['path_calibration'] = str(path_calibration)
param['path_closed_loop'] = str(path_closed_loop)
param['path_analysis_closed_loop'] = str(path_analysis_closed_loop)
param['path_analysis_uniform_noise_propagation'] = str(path_analysis_uniform_noise_propagation)
param['path_plots_closed_loop'] = str(path_plots_closed_loop)
param['path_plots_uniform_noise_propagation'] = str(path_plots_uniform_noise_propagation)


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
param['sampling_time'          ] = 1/500                                     # [s] loop sampling time
param['centralObstruction'     ] = 0                                         # central obstruction in percentage of the diameter

# ---------------------- NGS ---------------------- #

param['magnitude'            ] = 0                                          # magnitude of the guide star

# GHOST wavelength : 770 nm, full bandwidth = 20 nm
# 'I2' band : 750 nm, bandwidth? = 33 nm                                    # phot.R4 = [0.670e-6, 0.300e-6, 7.66e12]
param['optical_band'          ] = 'R4'                                      # optical band of the guide star

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
param['sr_amplitude']        = 0.20                   # [pixel] super resolution shifts amplitude
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

# -------------------- MODAL BASIS ---------------- #

param['modal_basis'] = 'KL'
param['list_modes_to_keep'] = np.linspace(int(0.5*(np.pi * (param['n_subaperture']/2)**2)), 
                                          int(np.pi * param['n_subaperture']**2), num=10, dtype=int)
param['stroke'] = 1e-9 # [m] actuator stroke for calibration matrices computation
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

#%% Save parameter file

save_vars(path_data / pathlib.Path("parameter_file" + ".pkl"), ['param'])
