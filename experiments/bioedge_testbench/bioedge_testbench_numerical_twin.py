# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:08:36 2024

@author: fleroux
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from fanch.tools import get_tilt

#%% Parameters

n_subaperture = 16 # 124
modal_basis_name = 'KL' # 'Poke' ; 'Fourier1D' ; 'Fourier2D', 'Fourier2Dsmall'
#modal_basis_name = 'poke'
#modal_basis_name = 'Fourier1D_diag'
#modal_basis_name = 'Fourier1D_vert'
#modal_basis_name = 'Fourier2D'
#modal_basis_name = 'Fourier2Dsmall'
#modal_basis_name = 'Fourier2DsmallBis'

#%% Functions declarations

# def get_tilt(shape, theta=0., amplitude=1.):
#     [X,Y] = np.meshgrid(np.arange(0, shape[0]), np.arange(0, shape[1]))
#     tilt_theta = np.cos(theta) * X + np.sin(theta) * Y
#     Y = np.flip(Y, axis=0) # change orientation
#     tilt_theta = (tilt_theta - tilt_theta.min())/\
#     (tilt_theta.max()- tilt_theta.min())
    
#     return amplitude*tilt_theta

#%% -----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope

# create the Telescope object
tel = Telescope(resolution           = 8*n_subaperture,   # resolution of the telescope in [pix]
                diameter             = 8,                 # diameter in [m]        
                samplingTime         = 1/1000,            # Sampling time in [s] of the AO loop
                centralObstruction   = 0.,                # Central obstruction in [%] of a diameter 
                display_optical_path = False,             # Flag to display optical path
                fov                  = 10)                # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of 
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

#%% Compute PSF

tel.computePSF(zeroPaddingFactor=2)

#%% -----------------------     Bi-O-Edge WFS   ------------------------

from OOPAO.BioEdge import BioEdge

wfs = BioEdge(nSubap = n_subaperture, 
              telescope = tel, 
              modulation = 0.,
              grey_width = 2.5, 
              lightRatio = 0.5,
              n_pix_separation = 0,
              postProcessing = 'fullFrame', 
              psfCentering=False)

# propagate the light to the Wave-Front Sensor
tel*wfs
wfs.wfs_measure(tel.pupil)
flat_frame = wfs.cam.frame

#%% Atmosphere

from OOPAO.Atmosphere import Atmosphere
           
# create the Atmosphere object
atm = Atmosphere(telescope     = tel,                               # Telescope                              
                 r0            = 0.15,                              # Fried Parameter [m]
                 L0            = 25,                                # Outer Scale [m]
                 fractionalR0  = [0.45 ,0.1  ,0.1  ,0.25  ,0.1   ], # Cn2 Profile
                 windSpeed     = [10   ,12   ,11   ,15    ,20    ], # Wind Speed in [m]
                 windDirection = [0    ,72   ,144  ,216   ,288   ], # Wind Direction in [degrees]
                 altitude      = [0    ,1000 ,5000 ,10000 ,12000 ]) # Altitude Layers in [m]


#%% Modal basis

from OOPAO.DeformableMirror import DeformableMirror

if modal_basis_name == 'KL':
    from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
    dm = DeformableMirror(tel, nSubap=2*n_subaperture)
    M2C = compute_KL_basis(tel, atm, dm, lim = 1e-3) # matrix to apply modes on the DM
    #M2C = M2C[:,:200]

elif modal_basis_name == 'poke':
    dm = DeformableMirror(tel, nSubap=2*n_subaperture)
    M2C = np.identity(dm.nValidAct)

elif modal_basis_name == 'Fourier1D_diag':
    from fanch.basis.fourier import compute_real_fourier_basis, extract_subset,\
        extract_diagonal_frequencies
    fourier_modes = compute_real_fourier_basis(tel.resolution, return_map=True)
    fourier_modes = extract_subset(fourier_modes, 4*n_subaperture)
    fourier_modes = extract_diagonal_frequencies(fourier_modes, complete=False)
    fourier_modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))
    fourier_modes = fourier_modes[:,1:] # remove piston
    dm = DeformableMirror(tel, nSubap=2*n_subaperture, modes=fourier_modes) # modal dm
    M2C = np.identity(dm.nValidAct)
    
elif modal_basis_name == 'Fourier1D_vert':
    from fanch.basis.fourier import compute_real_fourier_basis, extract_subset,\
        extract_vertical_frequencies
    fourier_modes = compute_real_fourier_basis(tel.resolution, return_map=True)
    #fourier_modes = extract_subset(fourier_modes, 4*n_subaperture)
    fourier_modes = extract_vertical_frequencies(fourier_modes)
    fourier_modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))
    fourier_modes = fourier_modes[:,1:] # remove piston
    dm = DeformableMirror(tel, nSubap=2*n_subaperture, modes=fourier_modes) # modal dm
    M2C = np.identity(dm.nValidAct)
    
elif modal_basis_name == 'Fourier2D':
    from fanch.basis.fourier import compute_real_fourier_basis
    fourier_modes = compute_real_fourier_basis(tel.resolution)
    fourier_modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))
    fourier_modes = fourier_modes[:,1:] # remove piston
    dm = DeformableMirror(tel, nSubap=2*n_subaperture, modes=fourier_modes) # modal dm
    M2C = np.identity(dm.nValidAct)
    
elif modal_basis_name == 'Fourier2Dsmall':
    from fanch.basis.fourier import compute_real_fourier_basis,\
        extract_subset, sort_real_fourier_basis
    fourier_modes = compute_real_fourier_basis(tel.resolution, return_map=True)
    fourier_modes = extract_subset(fourier_modes, 2*n_subaperture)
    fourier_modes = sort_real_fourier_basis(fourier_modes)
    fourier_modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))
    fourier_modes = fourier_modes[:,1:] # remove piston
    dm = DeformableMirror(tel, nSubap=2*n_subaperture, modes=fourier_modes) # modal dm
    M2C = np.identity(dm.nValidAct)
    
elif modal_basis_name == 'Fourier2DsmallBis':
    from fanch.basis.fourier import compute_real_fourier_basis
    fourier_modes = compute_real_fourier_basis(tel.resolution)
    fourier_modes = fourier_modes[:,:,:int(np.pi * n_subaperture**2)]
    fourier_modes = fourier_modes.reshape((fourier_modes.shape[0]*fourier_modes.shape[1],fourier_modes.shape[2]))
    fourier_modes = fourier_modes[:,1:] # remove piston
    dm = DeformableMirror(tel, nSubap=2*n_subaperture, modes=fourier_modes) # modal dm
    M2C = np.identity(dm.nValidAct)

#%% Callibration - No SR

from OOPAO.calibration.InteractionMatrix import InteractionMatrix

stroke = 1e-9 # [m]

tel.resetOPD()
ngs*tel*dm
calib = InteractionMatrix(ngs, atm, tel, dm, wfs, M2C = M2C, stroke = stroke)

#%% Super Resolution

sr_amplitude = 0.25

sx = [-sr_amplitude, sr_amplitude, -sr_amplitude, sr_amplitude] # pixels 
sy = [sr_amplitude, sr_amplitude, -sr_amplitude, -sr_amplitude] # pixels

sx = np.array(sx)
sy = np.array(sy)

mask = wfs.mask
mask_sr_cam_frame = deepcopy(mask)

binning_factor = wfs.bioFrame.shape[0]/wfs.cam.frame.shape[0]
sx_cam_frame = [binning_factor * k for k in sx]
sy_cam_frame = [binning_factor * k for k in sy]

for k in range(len(mask)):
    tilt_x = get_tilt(mask[k].shape, theta = 0., amplitude = 2.*np.pi*sx_cam_frame[k])
    tilt_y = get_tilt(mask[k].shape, theta = np.pi/2, amplitude = 2.*np.pi*sy_cam_frame[k])
    mask_sr_cam_frame[k] = mask[k]*np.exp(1j*(tilt_x+tilt_y))

# replace usual-masks by SR-masks
wfs.mask = mask_sr_cam_frame
wfs.modulation = 0. # update reference intensities etc.

tel*wfs
wfs.wfs_measure(tel.pupil)
flat_frame_sr = wfs.cam.frame

#%% Callibration - SR

tel.resetOPD()
calib_sr = InteractionMatrix(ngs, atm, tel, dm, wfs, M2C = M2C, stroke = stroke)


#%% -----------------------     Bi-O-Edge WFS - 2x more samples   ------------------------

from OOPAO.BioEdge import BioEdge

wfs_oversampled = BioEdge(nSubap = 2*n_subaperture, 
              telescope = tel, 
              modulation = 0.,
              grey_width = 2.5, 
              lightRatio = 0.5,
              n_pix_separation = 0,
              postProcessing = 'fullFrame', 
              psfCentering=False)

# propagate the light to the Wave-Front Sensor
tel*wfs_oversampled
wfs_oversampled.wfs_measure(tel.pupil)
flat_frame_oversampled = wfs_oversampled.cam.frame

#%% Callibration - No SR - 2x oversampled

tel.resetOPD()
ngs*tel*dm
calib_oversampled = InteractionMatrix(ngs, atm, tel, dm, wfs_oversampled, M2C = M2C, stroke = stroke)

#%% Uniform noise sensitivity computation

sensitivity_matrix = np.abs(calib.D.T @ calib.D)
sensitivity_matrix_sr = np.abs(calib_sr.D.T @ calib_sr.D)
sensitivity_matrix_oversampled = np.abs(calib_oversampled.D.T @ calib_oversampled.D)

#%% Matrice de Covariance de l'erreur de phase

# no SR
phase_error_cov_matrix = calib.M @ calib.M.T #np.linalg.inv(calib.D.T @ calib.D)

# SR
phase_error_cov_matrix_sr = calib_sr.M @ calib_sr.M.T # np.linalg.inv(calib_sr.D.T @ calib_sr.D)

# oversampled
phase_error_cov_matrix_oversampled = calib_oversampled.M @ calib_oversampled.M.T # np.linalg.inv(calib_sr.D.T @ calib_sr.D)

# %% ------------------ PLOTS --------------------------------------------

from OOPAO.tools.displayTools import display_wfs_signals

plt.figure(1)
plt.imshow(np.abs(flat_frame_sr-flat_frame))
plt.title('SR pupils - No SR pupils (reference signal for a flat wavefront)\n0.25 pixel shifts')

#%% SVD - Eigenvalues

plt.figure(2)
plt.semilogy(calib.eigenValues/calib.eigenValues.max(), 'b', label='no SR')
plt.semilogy(calib_sr.eigenValues/calib_sr.eigenValues.max(), 'r', label='SR')
plt.semilogy(calib_oversampled.eigenValues/calib_oversampled.eigenValues.max(), 'c', label='oversampled')
plt.title('calib.eigenValues, '+str(n_subaperture)+' wfs subapertures, ' + modal_basis_name + ' modes used, 0.25 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')

#%%

# n_mode = -1

# display_wfs_signals(wfs, signals=calib.D[:,n_mode])
# plt.title('Bi-O-Edge signal, '+str(n_mode)+'th ' + modal_basis_name+' modes\n No SR')

# display_wfs_signals(wfs, signals=calib_sr.D[:,n_mode])
# plt.title('Bi-O-Edge signal, '+str(n_mode)+'th ' + modal_basis_name+' modes\n SR')

#%% Sensitivity Matrices

fig6, axs6 = plt.subplots(nrows=1, ncols=2)
img1 = axs6[0].imshow(np.abs(sensitivity_matrix))
axs6[0].set_title('Sensitivity matrix - ' + modal_basis_name + ' - No SR')
img2 = axs6[1].imshow(np.abs(sensitivity_matrix_sr))
axs6[1].set_title('Sensitivity matrix - ' + modal_basis_name + ' - SR')
plt.colorbar(img1, ax=axs6[0], fraction=0.046, pad=0.04)
plt.colorbar(img2, ax=axs6[1], fraction=0.046, pad=0.04)

#%% Noise propagation

plt.figure()
plt.plot(np.abs(np.diag(phase_error_cov_matrix)), 'b', label="no SR")
plt.plot(np.abs(np.diag(phase_error_cov_matrix_sr)),'r', label="SR")
plt.plot(np.diag(phase_error_cov_matrix_oversampled), 'c', label="oversampled")
plt.title("Uniform noise propagation\n"
          "Impact of Super Resolution")
plt.xlabel("mode ("+modal_basis_name+") index i")
plt.ylabel("np.diag(calib.M @ calib.M.T)")
plt.legend()

#%% Noise propagation - Log Scale

plt.figure()
plt.plot(np.diag(phase_error_cov_matrix), 'b', label="no SR")
plt.plot(np.diag(phase_error_cov_matrix_sr),'r', label="SR")
plt.plot(np.diag(phase_error_cov_matrix_oversampled), 'c', label="oversampled")
plt.yscale('log')
plt.title("Uniform noise propagation\n"
          "Impact of Super Resolution")
plt.xlabel("mode ("+modal_basis_name+") index i")
plt.ylabel("np.diag(calib.M @ calib.M.T)")
plt.legend()

#%% Noise propagation - Log-Log Scale

plt.figure()
plt.plot(np.diag(phase_error_cov_matrix), 'b', label="no SR")
plt.plot(np.diag(phase_error_cov_matrix_sr),'r', label="SR")
plt.plot(np.diag(phase_error_cov_matrix_oversampled), 'c', label="oversampled")
plt.xscale('log')
plt.yscale('log')
plt.title("Uniform noise propagation\n"
          "Impact of Super Resolution")
plt.xlabel("mode ("+modal_basis_name+") index i")
plt.ylabel("np.diag(calib.M @ calib.M.T)")
plt.legend()

#%% Uniform Noise sensitivity

plt.figure()
plt.plot(np.diag(sensitivity_matrix)**0.5,'b', label='no SR')
plt.plot(np.diag(sensitivity_matrix_sr)**0.5,'r', label='SR')
plt.plot(np.diag(sensitivity_matrix_oversampled)**0.5,'c', label='oversampled')
plt.legend()
plt.title("Uniform Noise Sensitivity")
plt.xlabel("mode ("+modal_basis_name+") index")
plt.ylabel("np.diag(calib.D.T @ calib.D)**0.5")