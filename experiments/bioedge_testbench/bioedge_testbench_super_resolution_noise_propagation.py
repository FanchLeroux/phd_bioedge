# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:48:11 2025

@author: fleroux
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from fanch.tools import get_tilt

#%%

from pathlib import Path

path = Path(__file__).parent

#%% Parameters

KL_computed = True

n_subaperture = 20

modal_basis_name = 'KL' # 'Poke' ; 'Fourier1D' ; 'Fourier2D', 'Fourier2Dsmall'
#modal_basis_name = 'poke'
#modal_basis_name = 'Fourier1D_diag'
#modal_basis_name = 'Fourier1D_vert'
#modal_basis_name = 'Fourier2D'
#modal_basis_name = 'Fourier2Dsmall'
#modal_basis_name = 'Fourier2DsmallBis'

grey_width = 5.





n_modes_list = np.arange(100, 1300, 100)

noise_propagation_no_sr = []
noise_propagation_sr = []

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
              grey_width = grey_width, 
              lightRatio = 0.5,
              n_pix_separation = 10,
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


#%% Load KL modes
if KL_computed:
    M2C = np.load(path / "M2C_KL.npy")

#%% Modal basis

from OOPAO.DeformableMirror import DeformableMirror

if modal_basis_name == 'KL':
    dm = DeformableMirror(tel, nSubap=2*n_subaperture)
    if not(KL_computed):
        from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
        M2C = compute_KL_basis(tel, atm, dm) # matrix to apply modes on the DM
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

#%% save KL modes

if modal_basis_name == 'KL' and not(KL_computed):
    np.save(path / "M2C_KL.npy", M2C)

#%% Callibration - No SR

from OOPAO.calibration.InteractionMatrix import InteractionMatrix

stroke = 1e-9 # [m]

tel.resetOPD()
ngs*tel*dm
calib = InteractionMatrix(ngs, atm, tel, dm, wfs, M2C = M2C, stroke = stroke, single_pass=False)

#%% Super Resolution

sr_amplitude = 0.25

sx = [sr_amplitude, -sr_amplitude, sr_amplitude, -sr_amplitude] # pixels 
sy = [sr_amplitude, -sr_amplitude, -sr_amplitude, sr_amplitude] # pixels

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
calib_sr = InteractionMatrix(ngs, atm, tel, dm, wfs, M2C = M2C, stroke = stroke, single_pass=False)

#%% -----------------------     Bi-O-Edge WFS - 2x more samples   ------------------------

from OOPAO.BioEdge import BioEdge

wfs_oversampled = BioEdge(nSubap = 2*n_subaperture, 
              telescope = tel, 
              modulation = 0.,
              grey_width = grey_width, 
              lightRatio = 0.5,
              n_pix_separation = 10,
              postProcessing = 'fullFrame', 
              psfCentering=False)

# propagate the light to the Wave-Front Sensor
tel*wfs_oversampled
wfs_oversampled.wfs_measure(tel.pupil)
flat_frame_oversampled = wfs_oversampled.cam.frame

#%% Callibration - No SR - 2x oversampled

tel.resetOPD()
ngs*tel*dm
calib_oversampled = InteractionMatrix(ngs, atm, tel, dm, wfs_oversampled, M2C = M2C, stroke = stroke, single_pass=False)

#%% Uniform noise sensitivity computation

sensitivity_matrix = np.abs(calib.D.T @ calib.D)
sensitivity_matrix_sr = np.abs(calib_sr.D.T @ calib_sr.D)
sensitivity_matrix_oversampled = np.abs(calib_oversampled.D.T @ calib_oversampled.D)

#%% Reconstructor computation 1 - Truncate calibration basis

R_oversampled = np.linalg.pinv(calib_oversampled.D)
noise_propagation_oversampled = np.diag(R_oversampled @ R_oversampled.T)/wfs_oversampled.nSignal

for n_modes in range(n_modes_list.shape[0]):

    n_modes_no_sr = n_modes_list[n_modes]
    n_modes_sr = n_modes_list[n_modes]

    R = np.linalg.pinv(calib.D[:,:n_modes_no_sr])
    R_sr = np.linalg.pinv(calib_sr.D[:,:n_modes_sr])

    noise_propagation_no_sr.append(np.diag(R @ R.T)/wfs.nSignal)
    noise_propagation_sr.append(np.diag(R_sr @ R_sr.T)/wfs.nSignal)

#%% Reconstructor computation 2 - Trucate eigen basis

# calib.nTrunc = calib.D.shape[1] - n_modes_no_sr
# calib_sr.nTrunc = calib_sr.D.shape[1] - n_modes_sr

# R = calib.Mtrunc
# R_sr = calib_sr.Mtrunc
# R_oversampled = calib_oversampled.M

#%% Matrice de Covariance de l'erreur de phase

# # no SR
# #phase_error_cov_matrix = calib.M @ calib.M.T #np.linalg.inv(calib.D.T @ calib.D)
# phase_error_cov_matrix = R @ R.T

# # SR
# #phase_error_cov_matrix_sr = calib_sr.M @ calib_sr.M.T # np.linalg.inv(calib_sr.D.T @ calib_sr.D)
# phase_error_cov_matrix_sr = R_sr @ R_sr.T

# # oversampled
# #phase_error_cov_matrix_oversampled = calib_oversampled.M @ calib_oversampled.M.T # np.linalg.inv(calib_sr.D.T @ calib_sr.D)
# phase_error_cov_matrix_oversampled = R_oversampled @ R_oversampled.T

# %% ------------------ PLOTS --------------------------------------------

plt.figure()
plt.imshow(np.abs(flat_frame_sr-flat_frame))
plt.title('SR pupils - No SR pupils (reference signal for a flat wavefront)\n0.25 pixel shifts')

#%% SVD - Eigenvalues  

plt.figure()
plt.semilogy(calib.eigenValues, 'b', label='no SR')
plt.semilogy(calib_sr.eigenValues, 'r', label='SR')
plt.semilogy(calib_oversampled.eigenValues, 'c', label='oversampled')
plt.title('calib.eigenValues, '+str(n_subaperture)+' wfs subapertures, ' + modal_basis_name + ' modes used, 0.25 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('Eigen value')

#%% SVD - Normalized Eigenvalues  

plt.figure()
plt.semilogy(calib.eigenValues/calib.eigenValues.max(), 'b', label='no SR')
plt.semilogy(calib_sr.eigenValues/calib_sr.eigenValues.max(), 'r', label='SR')
plt.semilogy(calib_oversampled.eigenValues/calib_oversampled.eigenValues.max(), 'c', label='oversampled')
plt.title('calib.eigenValues, '+str(n_subaperture)+' wfs subapertures, ' + modal_basis_name + ' modes used, 0.25 pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')

#%% Noise propagation - Log Scale - no SR

plt.figure()
plt.plot(noise_propagation_oversampled, 'k', label="40x40")

for n_modes in range(n_modes_list.shape[0]):
    plt.plot(noise_propagation_no_sr[n_modes], label= str(n_modes_list[n_modes])+" modes")

plt.yscale('log')
plt.title("20x20 Grey Bi-O-Edge Uniform noise propagation\n"
          "Without Super Resolution")
plt.xlabel("mode ("+modal_basis_name+") index")
plt.ylabel("np.diag(R @ R.T)")
plt.legend()

#%% Noise propagation - Log Scale - SR

plt.figure()
plt.plot(noise_propagation_oversampled, 'k', label="40x40")

for n_modes in range(n_modes_list.shape[0]-1):
    plt.plot(noise_propagation_sr[n_modes], label= str(n_modes_list[n_modes])+" modes")

plt.yscale('log')
plt.title("20x20 Grey Bi-O-Edge Uniform noise propagation\n"
          "Grey width = " + str(int(grey_width)) + " lambda/D\n"
          "With Super Resolution")
plt.xlabel("mode ("+modal_basis_name+") index")
plt.ylabel("np.diag(R @ R.T)/wfs.nSignal")
plt.legend()

# #%% Noise propagation - Log-Log Scale - no SR

# plt.figure()
# plt.plot(np.diag(phase_error_cov_matrix), 'b', label="no SR")
# plt.plot(np.diag(phase_error_cov_matrix_oversampled), 'c', label="oversampled")
# plt.xscale('log')
# plt.yscale('log')
# plt.title("Uniform noise propagation\n"
#           "Impact of Super Resolution")
# plt.xlabel("mode ("+modal_basis_name+") index i")
# plt.ylabel("np.diag(calib.M @ calib.M.T)")
# plt.legend()

# #%% Noise propagation - Log Scale - SR

# plt.figure()
# plt.plot(np.diag(phase_error_cov_matrix_sr)/wfs.nSignal,'r', label="SR")
# plt.plot(np.diag(phase_error_cov_matrix_oversampled)/wfs_oversampled.nSignal, 'c', label="oversampled")
# plt.yscale('log')
# plt.title("Uniform noise propagation\n"
#           "Impact of Super Resolution")
# plt.xlabel("mode ("+modal_basis_name+") index i")
# plt.ylabel("np.diag(calib.M @ calib.M.T)")
# plt.legend()

# #%% Noise propagation - Log-Log Scale - SR

# plt.figure()
# plt.plot(np.diag(phase_error_cov_matrix_sr),'r', label="SR")
# plt.plot(np.diag(phase_error_cov_matrix_oversampled), 'c', label="oversampled")
# plt.xscale('log')
# plt.yscale('log')
# plt.title("Uniform noise propagation\n"
#           "Impact of Super Resolution")
# plt.xlabel("mode ("+modal_basis_name+") index i")
# plt.ylabel("np.diag(calib.M @ calib.M.T)")
# plt.legend()
