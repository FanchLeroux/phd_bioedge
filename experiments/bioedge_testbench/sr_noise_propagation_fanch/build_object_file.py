# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:45:38 2025

@author: fleroux
"""

import numpy as np

from pathlib import Path

from parameter_file import get_parameters

from OOPAO.Telescope import Telescope
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Source import Source
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
from fanch.basis.fourier import compute_real_fourier_basis, extract_subset, extract_vertical_frequencies, extract_diagonal_frequencies
from OOPAO.Pyramid import Pyramid
from OOPAO.BioEdge import BioEdge

param = get_parameters()

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


# grey bioedge
gbioedge = BioEdge(nSubap = param['n_subaperture'], 
              telescope = tel, 
              modulation = 0.,
              grey_width = param['modulation'], 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = param['post_processing'], 
              psfCentering = param['psf_centering'])

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

# grey bioedge oversampled
gbioedge_oversampled = BioEdge(nSubap = 2*param['n_subaperture'], 
              telescope = tel, 
              modulation = 0.,
              grey_width = param['modulation'], 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = param['post_processing'], 
              psfCentering = param['psf_centering'])

# sharp bioedge
sbioedge = BioEdge(nSubap = param['n_subaperture'], 
              telescope = tel, 
              modulation = param['modulation'],
              grey_width = 0., 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = param['post_processing'], 
              psfCentering = param['psf_centering'])

# sharp bioedge SR
sbioedge_sr = BioEdge(nSubap = param['n_subaperture'], 
              telescope = tel, 
              modulation = param['modulation'],
              grey_width = 0., 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = param['post_processing'], 
              psfCentering = param['psf_centering'])

gbioedge_sr.apply_shift_wfs(param['pupil_shift_bioedge'][0], param['pupil_shift_bioedge'][1], units='pixels')
gbioedge_sr.modulation = 0. # update reference intensities etc.

# sharp bioedge oversampled
sbioedge_oversampled = BioEdge(nSubap = 2*param['n_subaperture'], 
              telescope = tel, 
              modulation = param['modulation'],
              grey_width = 0., 
              lightRatio = param['light_threshold'],
              n_pix_separation = param['n_pix_separation'],
              postProcessing = param['post_processing'], 
              psfCentering = param['psf_centering'])
    
#%% save all variables

import dill

origin = Path(__file__) # keep a trace of where the saved objects come from

dill.dump_session(param['path_object'] / Path('object'+str(param['filename'])+'.pkl'))