# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:45:38 2025

@author: fleroux
"""

# pylint: disable=undefined-variable
# pylint: disable=undefined-loop-variable

#%%

import pathlib
import sys

from fanch.tools.save_load import save_vars, load_vars
from fanch.tools.oopao import clean_wfs

import numpy as np

from copy import deepcopy

from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis, compute_M2C
from fanch.basis.fourier import compute_real_fourier_basis, extract_subset, extract_vertical_frequencies,\
    extract_diagonal_frequencies
from OOPAO.Pyramid import Pyramid
from OOPAO.BioEdge import BioEdge

#%% Define paths

path = pathlib.Path(__file__).parent
path_data = path.parent.parent.parent.parent / "phd_bioedge_data" / pathlib.Path(*path.parts[-3:]) # could be done better

#%% Get parameter file

path_parameter_file = path_data / "parameter_file.pkl"
load_vars(path_parameter_file, ['param'])

#%% -----------------------    TELESCOPE   -----------------------------

# create the Telescope object
tel = Telescope(resolution = param['resolution'], # [pixel] resolution of the telescope
                diameter   = param['diameter'])   # [m] telescope diameter

save_vars(pathlib.Path(param['path_object']) / 'tel.pkl', ['tel'])

#%% -----------------------     NGS   ----------------------------------

# create the Natural Guide Star object
ngs = Source(optBand     = param['optical_band'],          # Source optical band (see photometry.py)
             magnitude   = param['magnitude'])            # Source Magnitude

save_vars(pathlib.Path(param['path_object']) / 'ngs.pkl', ['ngs'])

#%% -----------------------    ATMOSPHERE   ----------------------------

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

save_vars(pathlib.Path(param['path_object']) / 'atm.pkl', ['atm'])

#%% -------------------------     DM   ----------------------------------

if not(param['is_dm_modal']):
    dm = DeformableMirror(tel, nSubap=param['n_actuator'])
    save_vars(pathlib.Path(param['path_object']) / 'dm.pkl', ['dm'])
    
#%% ------------------------- MODAL BASIS -------------------------------

if param['modal_basis'] == 'KL':
    # M2C = compute_KL_basis(tel, atm, dm) # matrix to apply modes on the DM
    
    if sys.platform.startswith('win'):
        nameFolder = str(param['path_object']) +"\\"
    else:
        nameFolder = str(param['path_object']) +"/"
    
    M2C_KL_full = compute_M2C(telescope          = tel,\
                              atmosphere         = atm,\
                              deformableMirror   = dm,\
                              param              = param,\
                              nameFolder         = nameFolder,\
                              remove_piston      = False,\
                              HHtName            = 'KL_covariance_matrix',\ # name of the saved cov matrix
                              baseName           = 'KL_basis' ,\
                              mem_available      = 6.1e9,\
                              minimF             = False,\
                              nmo                = 1350,\
                              ortho_spm          = True,\
                              SZ                 = np.int64(2*tel.OPD.shape[0]),\
                              nZer               = 3,\
                              NDIVL              = 1,\
                              lim_inversion=1e-5)
        
    M2C = M2C_KL_full[:,1:] # remove piston
    
    dm.coefs = np.zeros(dm.nValidAct) # reset dm.OPD
    
    save_vars(pathlib.Path(param['path_object']) / 'M2C.pkl', ['M2C'])

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
    save_vars(pathlib.Path(param['path_object']) / 'dm.pkl', ['dm'])
    save_vars(pathlib.Path(param['path_object']) / 'M2C.pkl', ['M2C'])

#%% --------------------------- WFSs -----------------------------------

# ---------------------- Pyramid ----------------------------- #

# pramid
pyramid = Pyramid(nSubap = param['n_subaperture'], 
              telescope = tel, 
              modulation = param['modulation'], 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              n_pix_edge = param['n_pix_separation'],
              postProcessing = param['post_processing'],
              psfCentering = param['psf_centering'])

pyramid = clean_wfs(pyramid)
save_vars(pathlib.Path(param['path_object']) / 'pyramid.pkl', ['pyramid'])

#%%

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

pyramid_sr = clean_wfs(pyramid_sr)
save_vars(pathlib.Path(param['path_object']) / 'pyramid_sr.pkl', ['pyramid_sr'])

# pyramid oversampled
pyramid_oversampled = Pyramid(nSubap = 2*param['n_subaperture'], 
              telescope = tel, 
              modulation = param['modulation'], 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              n_pix_edge = param['n_pix_separation'],
              postProcessing = param['post_processing'],
              psfCentering = param['psf_centering'])

pyramid_oversampled = clean_wfs(pyramid_oversampled)
save_vars(pathlib.Path(param['path_object']) / 'pyramid_oversampled.pkl', ['pyramid_oversampled'])

# ----------------------- Sharp Bi-O-Edge ---------------------------- #

# sharp bioedge
sbioedge = BioEdge(nSubap = param['n_subaperture'], 
              telescope = tel, 
              modulation = param['modulation'],
              grey_width = 0., 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = param['post_processing'], 
              psfCentering = param['psf_centering'])

sbioedge = clean_wfs(sbioedge)
save_vars(pathlib.Path(param['path_object']) / 'sbioedge.pkl', ['sbioedge'])

# sharp bioedge SR
sbioedge_sr = BioEdge(nSubap = param['n_subaperture'], 
              telescope = tel, 
              modulation = param['modulation'],
              grey_width = 0., 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = param['post_processing'], 
              psfCentering = param['psf_centering'])

sbioedge_sr.apply_shift_wfs(param['pupil_shift_bioedge'][0], param['pupil_shift_bioedge'][1], units='pixels')
sbioedge_sr.modulation = param['modulation'] # update reference intensities etc.

sbioedge_sr = clean_wfs(sbioedge_sr)
save_vars(pathlib.Path(param['path_object']) / 'sbioedge_sr.pkl', ['sbioedge_sr'])

# sharp bioedge oversampled
sbioedge_oversampled = BioEdge(nSubap = 2*param['n_subaperture'], 
              telescope = tel, 
              modulation = param['modulation'],
              grey_width = 0., 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = param['post_processing'], 
              psfCentering = param['psf_centering'])

sbioedge_oversampled = clean_wfs(sbioedge_oversampled)
save_vars(pathlib.Path(param['path_object']) / 'sbioedge_oversampled.pkl', ['sbioedge_oversampled'])

# ----------------------- Grey Bi-O-Edge ---------------------------- #

# grey bioedge
gbioedge = BioEdge(nSubap = param['n_subaperture'], 
              telescope = tel, 
              modulation = 0.,
              grey_width = param['modulation'], 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = param['post_processing'], 
              psfCentering = param['psf_centering'])

gbioedge = clean_wfs(gbioedge)
save_vars(pathlib.Path(param['path_object']) / 'gbioedge.pkl', ['gbioedge'])

# grey bioedge SR
gbioedge_sr = BioEdge(nSubap = param['n_subaperture'], 
              telescope = tel, 
              modulation = 0.,
              grey_width = param['modulation'], 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = param['post_processing'], 
              psfCentering = param['psf_centering'])

gbioedge_sr.apply_shift_wfs(param['pupil_shift_bioedge'][0], param['pupil_shift_bioedge'][1], units='pixels')
gbioedge_sr.modulation = 0. # update reference intensities etc.

gbioedge_sr = clean_wfs(gbioedge_sr)
save_vars(pathlib.Path(param['path_object']) / 'gbioedge_sr.pkl', ['gbioedge_sr'])

# grey bioedge oversampled
gbioedge_oversampled = BioEdge(nSubap = 2*param['n_subaperture'],
              telescope = tel,
              modulation = 0.,
              grey_width = param['modulation'], 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = param['post_processing'], 
              psfCentering = param['psf_centering'])

gbioedge_oversampled = clean_wfs(gbioedge_oversampled)
save_vars(pathlib.Path(param['path_object']) / 'gbioedge_oversampled.pkl', ['gbioedge_oversampled'])

# ---------------------- Small Grey Bi-O-Edge ------------------------------- #

# small grey bioedge
sgbioedge = BioEdge(nSubap = param['n_subaperture'], 
              telescope = tel, 
              modulation = 0.,
              grey_width = param['modulation'],
              grey_length = param['grey_length'],
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = param['post_processing'], 
              psfCentering = param['psf_centering'])

sgbioedge = clean_wfs(sgbioedge)
save_vars(pathlib.Path(param['path_object']) / 'sgbioedge.pkl', ['sgbioedge'])

# small grey bioedge SR
sgbioedge_sr = BioEdge(nSubap = param['n_subaperture'], 
              telescope = tel, 
              modulation = 0.,
              grey_width = param['modulation'],
              grey_length = param['grey_length'],
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = param['post_processing'], 
              psfCentering = param['psf_centering'])

sgbioedge_sr.apply_shift_wfs(param['pupil_shift_bioedge'][0], param['pupil_shift_bioedge'][1], units='pixels')
sgbioedge_sr.modulation = 0. # update reference intensities etc.

sgbioedge_sr = clean_wfs(sgbioedge_sr)
save_vars(pathlib.Path(param['path_object']) / 'sgbioedge_sr.pkl', ['sgbioedge_sr'])

# small grey bioedge oversampled
sgbioedge_oversampled = BioEdge(nSubap = 2*param['n_subaperture'], 
              telescope = tel, 
              modulation = 0.,
              grey_width = param['modulation'],
              grey_length = param['grey_length'],
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = param['post_processing'], 
              psfCentering = param['psf_centering'])

sgbioedge_oversampled = clean_wfs(sgbioedge_oversampled)
save_vars(pathlib.Path(param['path_object']) / 'sgbioedge_oversampled.pkl', ['sgbioedge_oversampled'])

#%% save all variables

parameters_object = deepcopy(param)

origin_object = str(pathlib.Path(__file__)) # keep a trace of where the saved objects come from

#%%

save_vars(pathlib.Path(parameters_object['path_object']) / pathlib.Path('all_objects'+str(parameters_object['filename'])+'.pkl'), 
          ['parameters_object', 'origin_object',\
           'tel','atm', 'dm', 'ngs', 'M2C',\
           'pyramid', 'pyramid_sr', 'pyramid_oversampled',\
           'sbioedge', 'sbioedge_sr', 'sbioedge_oversampled',\
           'gbioedge', 'gbioedge_sr', 'gbioedge_oversampled',\
           'sgbioedge', 'sgbioedge_sr','sgbioedge_oversampled'])