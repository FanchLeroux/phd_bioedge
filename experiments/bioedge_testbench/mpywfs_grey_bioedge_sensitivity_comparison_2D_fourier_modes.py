# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 11:40:19 2024

@author: fleroux
"""

import numpy as np
import matplotlib.pyplot as plt

from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere
from OOPAO.Telescope import Telescope
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.Pyramid import Pyramid
from OOPAO.BioEdge import BioEdge
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import display_wfs_signals

from fanch.basis.fourier import compute_real_fourier_basis,\
    extract_subset, sort_real_fourier_basis, extract_vertical_frequencies,\
    extract_diagonal_frequencies
from fanch.tools import zeros_padding

#%%

n_subapertures = 32

modulation = 0
stroke = 1e-9 # [m]

n_calib = 1
n_mode = -2

is_grey = 0
    
#%% -----------------------     TELESCOPE   ----------------------------------

# create the Telescope object
tel = Telescope(resolution           = 4*n_subapertures,  # resolution of the telescope in [pix]
                diameter             = 8,                 # diameter in [m]        
                samplingTime         = 1/1000,            # Sampling time in [s] of the AO loop
                centralObstruction   = 0.,                # Central obstruction in [%] of a diameter 
                display_optical_path = False,             # Flag to display optical path
                fov                  = 10)                # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of 
                                                          # the phase screens but is uncompatible with off-axis targets

#%% -----------------------     NGS   ----------------------------------

# GHOST wavelength : 770 nm, full bandwidth = 20 nm
# 'I2' band : 750 nm, bandwidth? = 33 nm

# create the Natural Guide Star object
ngs = Source(optBand     = 'I2',          # Optical band (see photometry.py)
             magnitude   = 0,             # Source Magnitude
             coordinates = [0,0])         # Source coordinated [arcsec,deg]

# combine the NGS to the telescope using '*'
ngs*tel

#%% ----------------------- Atmosphere ----------------------------------
           
# create the Atmosphere object
atm = Atmosphere(telescope     = tel,                               # Telescope                              
                 r0            = 0.15,                              # Fried Parameter [m]
                 L0            = 25,                                # Outer Scale [m]
                 fractionalR0  = [0.45 ,0.1  ,0.1  ,0.25  ,0.1   ], # Cn2 Profile
                 windSpeed     = [10   ,12   ,11   ,15    ,20    ], # Wind Speed in [m]
                 windDirection = [0    ,72   ,144  ,216   ,288   ], # Wind Direction in [degrees]
                 altitude      = [0    ,1000 ,5000 ,10000 ,12000 ]) # Altitude Layers

#%% --------------------------- PYWFS ------------------------------------

pywfs = Pyramid(nSubap = n_subapertures, 
              telescope = tel, 
              modulation = modulation, 
              lightRatio = 0.5,
              n_pix_separation = 2,
              postProcessing = 'fullFrame', 
              psfCentering=False)

#%% --------------------------- Bio Edge ------------------------------------

bioedge = BioEdge(nSubap = n_subapertures, 
                   telescope = tel, 
                   modulation = float(not(is_grey))*modulation, 
                   grey_width = float(is_grey)*modulation, 
                   lightRatio = 0.5,
                   postProcessing = 'fullFrame')

#%% ------------------------- Modal basis --------------------------------

fourier_modes = compute_real_fourier_basis(tel.resolution, return_map=True)
fourier_modes = extract_subset(fourier_modes, n_subapertures)

#%%

vertical_fourier_modes = extract_vertical_frequencies(fourier_modes)

diagonal_fourier_modes = extract_diagonal_frequencies(fourier_modes, complete=False)

complete_fourier_modes = sort_real_fourier_basis(fourier_modes)

#%%

modes_calibrations = [complete_fourier_modes, vertical_fourier_modes,\
                      diagonal_fourier_modes]

pywfs_calibrations = []
bioedge_calibrations = []
    
pywfs_sensitivity_matrices = []
pywfs_modal_sensitivities = []
bioedge_sensitivity_matrices = []
bioedge_modal_sensitivities = []
    
#%%

for fourier_modes in modes_calibrations:

#%%

    dm_modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],
                                      fourier_modes.shape[2]))

#%% --------------------------- Modal DM ---------------------------------

    dm = DeformableMirror(tel, nSubap=2*n_subapertures, modes=dm_modes) # modal dm
    M2C = np.identity(dm.nValidAct)

#%% --------------------------- Calibration ------------------------------

    tel.resetOPD()
    ngs*tel*dm
    calib_pywfs = InteractionMatrix(ngs, atm, tel, dm, pywfs, M2C = M2C, 
                                    stroke = stroke)
    
    tel.resetOPD()
    ngs*tel*dm
    calib_bioedge = InteractionMatrix(ngs, atm, tel, dm, bioedge, M2C = M2C, 
                                      stroke = stroke)
    
    pywfs_calibrations.append(calib_pywfs)
    bioedge_calibrations.append(calib_bioedge)
    
#%% ---------------------------- Sensitivity -----------------------------
    
    sensitivity_matrix_pywfs= calib_pywfs.D.T @ calib_pywfs.D
    sensitivity_matrix_bioedge = calib_bioedge.D.T @ calib_bioedge.D
    
    modal_sensitivity_pywfs = np.diag(sensitivity_matrix_pywfs)**0.5
    modal_sensitivity_bioedge = np.diag(sensitivity_matrix_bioedge)**0.5

    pywfs_sensitivity_matrices.append(sensitivity_matrix_pywfs)
    bioedge_sensitivity_matrices.append(sensitivity_matrix_bioedge)
    
    pywfs_modal_sensitivities.append(modal_sensitivity_pywfs)
    bioedge_modal_sensitivities.append(modal_sensitivity_bioedge)

#%%

zeros_padding_factor = 2
complex_amplitude = zeros_padding(tel.pupil*\
                                  np.exp(1j*20*np.pi*fourier_modes[:,:,n_mode]),\
                                  zeros_padding_factor)
plt.figure(4)
plt.imshow(np.abs(np.fft.fftshift((np.fft.fft2(complex_amplitude))))**2)

#%%

plt.figure(3)
plt.plot(range(pywfs_modal_sensitivities[1][:-1].shape[0]),
         pywfs_modal_sensitivities[1][:-1], '-+c', label='vertical modes, pywfs')
plt.plot(range(pywfs_modal_sensitivities[1][:-1].shape[0]),
         bioedge_modal_sensitivities[1][:-1], '-+m', label='vertical modes, bioedge')
plt.plot(2**0.5*np.arange(pywfs_modal_sensitivities[1][:-1].shape[0]),
         pywfs_modal_sensitivities[2][:-1], '-+b', label='diagonal modes, pywfs')
plt.plot(2**0.5*np.arange(pywfs_modal_sensitivities[1][:-1].shape[0]),
         bioedge_modal_sensitivities[2][:-1], '-+r', label='diagonal modes, bioedge')
plt.legend()
plt.title("Diagonal and vertical Fourier modes sensitivity\n"+
          "Bi-O-Edge VS PYWFS\n"
          "modulation = "+str(modulation)+" lambda/D")
plt.xlabel("# cycle/pupil (phase amplitude = "+str(1e9*stroke)+" nm)")
plt.ylabel("Sensitivity")

# plt.figure(4)
# plt.semilogy(modal_sensitivity_pywfs/modal_sensitivity_bioedge, 'b')
# plt.plot(np.ones(modal_sensitivity_bioedge.size),'r')

#%%

plt.figure(2)
plt.imshow(tel.pupil*modes_calibrations[n_calib][:,:,n_mode])

#%%

display_wfs_signals(pywfs, pywfs_calibrations[n_calib].D[:,n_mode])

#%%

display_wfs_signals(bioedge, bioedge_calibrations[n_calib].D[:,n_mode])

#%%

print("sum abs bio edge\n"+
      str(np.abs(bioedge_calibrations[n_calib].D[:,n_mode]).sum()))

print("sum abs pywfs\n"+
      str(np.abs(pywfs_calibrations[n_calib].D[:,n_mode]).sum()))

print("sum square bio edge\n"+
      str((np.abs(bioedge_calibrations[n_calib].D[:,n_mode]**2).sum())))
print("sum square pywfs\n"+
      str((pywfs_calibrations[n_calib].D[:,n_mode]**2).sum()))