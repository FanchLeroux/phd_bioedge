# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 11:40:19 2024

@author: fleroux
"""

import numpy as np
import matplotlib.pyplot as plt

# from OOPAO.Source import Source
# from OOPAO.Telescope import Telescope
# from OOPAO.DeformableMirror import DeformableMirror
# from OOPAO.Pyramid import Pyramid

from bi_dimensional_real_fourier_basis import compute_real_fourier_basis,\
    extract_subset, sort_real_fourier_basis

def zeros_padding(array, zeros_padding_factor):
    array = np.pad(array, (((zeros_padding_factor-1)*array.shape[0]//2, 
                                     (zeros_padding_factor-1)*array.shape[0]//2),
                                    ((zeros_padding_factor-1)*array.shape[1]//2, 
                                     (zeros_padding_factor-1)*array.shape[1]//2)))
    return array

#%%

n_subapertures = 32
    
#%% -----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope

# create the Telescope object
tel = Telescope(resolution           = 4*n_subapertures,   # resolution of the telescope in [pix]
                diameter             = 8,                 # diameter in [m]        
                samplingTime         = 1/1000,            # Sampling time in [s] of the AO loop
                centralObstruction   = 0.,                # Central obstruction in [%] of a diameter 
                display_optical_path = False,             # Flag to display optical path
                fov                  = 10)               # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of 
                                                          # the phase screens but is uncompatible with off-axis targets

#%% -----------------------     NGS   ----------------------------------

# GHOST wavelength : 770 nm, full bandwidth = 20 nm
# 'I2' band : 750 nm, bandwidth? = 33 nm

from OOPAO.Source import Source

# create the Natural Guide Star object
ngs = Source(optBand     = 'I2',          # Optical band (see photometry.py)
             magnitude   = 0,             # Source Magnitude
             coordinates = [0,0])         # Source coordinated [arcsec,deg]

# combine the NGS to the telescope using '*'
ngs*tel

#%% ----------------------- Atmosphere ----------------------------------

from OOPAO.Atmosphere import Atmosphere
           
# create the Atmosphere object
atm = Atmosphere(telescope     = tel,                               # Telescope                              
                 r0            = 0.15,                              # Fried Parameter [m]
                 L0            = 25,                                # Outer Scale [m]
                 fractionalR0  = [0.45 ,0.1  ,0.1  ,0.25  ,0.1   ], # Cn2 Profile
                 windSpeed     = [10   ,12   ,11   ,15    ,20    ], # Wind Speed in [m]
                 windDirection = [0    ,72   ,144  ,216   ,288   ], # Wind Direction in [degrees]
                 altitude      = [0    ,1000 ,5000 ,10000 ,12000 ]) # Altitude Layers in [m]

#%% --------------------------- Modulation ------------------------------------

modulation = 5.

#%% --------------------------- PYWFS ------------------------------------

from OOPAO.Pyramid import Pyramid

pywfs = Pyramid(nSubap = n_subapertures, 
              telescope = tel, 
              modulation = modulation, 
              lightRatio = 0.5,
              n_pix_separation = 2,
              postProcessing = 'fullFrame', 
              psfCentering=False)

#%% --------------------------- Bio Edge ------------------------------------

from OOPAO.BioEdge import BioEdge

bioedge = BioEdge(nSubap = n_subapertures, 
                   telescope = tel, 
                   modulation = modulation, 
                   grey_width = 0., 
                   lightRatio = 0.5,
                   postProcessing = 'fullFrame')

#%% ------------------------- Modal basis --------------------------------

fourier_modes = compute_real_fourier_basis(tel.resolution, return_map=True)
fourier_modes = extract_subset(fourier_modes, n_subapertures)
fourier_modes = sort_real_fourier_basis(fourier_modes)
dm_modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))

#%% --------------------------- Modal DM ---------------------------------

from OOPAO.DeformableMirror import DeformableMirror

dm = DeformableMirror(tel, nSubap=2*n_subapertures, modes=dm_modes) # modal dm
M2C = np.identity(dm.nValidAct)

#%% --------------------------- Calibration ------------------------------

from OOPAO.calibration.InteractionMatrix import InteractionMatrix

stroke = 1e-9 # [m]

#tel.resetOPD()
ngs*tel*dm
calib_pywfs = InteractionMatrix(ngs, atm, tel, dm, pywfs, M2C = M2C, stroke = stroke)

tel.resetOPD()
ngs*tel*dm
calib_bioedge = InteractionMatrix(ngs, atm, tel, dm, bioedge, M2C = M2C, stroke = stroke)

#%% ---------------------------- Sensitivity -----------------------------

sensitivity_matrix_pywfs= calib_pywfs.D.T @ calib_pywfs.D
sensitivity_matrix_bioedge = calib_bioedge.D.T @ calib_bioedge.D

modal_sensitivity_pywfs = np.diag(sensitivity_matrix_pywfs)
modal_sensitivity_bioedge = np.diag(sensitivity_matrix_bioedge)

#%%

from OOPAO.tools.displayTools import display_wfs_signals

#n_mode = 5

n_mode = 1

#%%
display_wfs_signals(pywfs, calib_pywfs.D[:,n_mode])

#%%
display_wfs_signals(bioedge, calib_bioedge.D[:,n_mode])

#%%

plt.figure(2)
plt.imshow(fourier_modes[:,:,n_mode])

# zeros_padding_factor = 2
# complex_amplitude = zeros_padding(tel.pupil*np.exp(1j*20*np.pi*fourier_modes[:,:,n_mode]), zeros_padding_factor)
# plt.figure(4)
# plt.imshow(np.abs(np.fft.fftshift((np.fft.fft2(complex_amplitude))))**2)

#%%

plt.figure(3)
plt.plot(modal_sensitivity_pywfs, '-b', label='pywfs')
plt.plot(modal_sensitivity_bioedge, '-r', label='bioedge')
plt.legend()

plt.figure(4)
plt.semilogy(modal_sensitivity_pywfs/modal_sensitivity_bioedge, 'b')
plt.plot(np.ones(modal_sensitivity_bioedge.size),'r')