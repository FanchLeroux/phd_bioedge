# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:08:36 2024

@author: fleroux
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

n_subaperture = 16 # 124

#%% Functions declarations

def get_tilt(shape, theta=0., amplitude=1.):
    [X,Y] = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    tilt_theta = np.cos(theta) * X + np.sin(theta) * Y
    Y = np.flip(Y, axis=0) # change orientation
    tilt_theta = (tilt_theta - tilt_theta.min())/(tilt_theta.max()- tilt_theta.min())
    
    return amplitude*tilt_theta

def compute_fourier_mode_x(shape, n_cycle):
    [X,_] = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    X = np.sin(2*np.pi * n_cycle/shape[0] * X)
    X = X/X.std()
    return X

#%% -----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope

# create the Telescope object

# You need to be VERY carefull with the telescope resolution. Setting it to only 4*n_subaperture leads to a too strong aliasing in the 
# input phases to be able to caracterize well enough the super resolution benefits

tel = Telescope(resolution           = 8*n_subaperture,   # resolution of the telescope in [pix]
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

#%% Modal basis

modal_basis_name = 'Fourier modes'

from OOPAO.tools.tools import compute_fourier_mode

n_modes = 2*n_subaperture
fourier_modes = np.empty((tel.pupil.shape[0]**2, n_modes-1))

for k in range(n_modes-1):
    fourier_mode = compute_fourier_mode(tel.pupil, spatial_frequency = k+1, angle_deg = 90.)
    fourier_mode = tel.pupil*fourier_mode
    fourier_modes[:,k] = np.reshape(fourier_mode, fourier_modes.shape[0])

#%% Deformable mirror

from OOPAO.DeformableMirror import DeformableMirror

dm = DeformableMirror(tel, nSubap=2*n_subaperture, modes=fourier_modes)

#%%

# dm.coefs[15] = 1.
# ngs*tel*dm

# plt.imshow(dm.OPD)

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

#%% Callibration - No SR

from OOPAO.calibration.InteractionMatrix import InteractionMatrix

M2C_modal_dm = np.identity(dm.nValidAct)

stroke = 10e-9 # [m]

tel.resetOPD()
ngs*tel*dm
calib = InteractionMatrix(ngs, atm, tel, dm, wfs, M2C = M2C_modal_dm, stroke = stroke)

#%% Super Resolution

sr_amplitude = 0.25 # pixel

sx = [-sr_amplitude, sr_amplitude, 0., 0.] # shift along meaningfull direction, no shift for blind pupils => we see a difference
sy = [0., 0., -0., -0.]


# sx = [0., 0., -0., -0.]
# sy = [-sr_amplitude, sr_amplitude, 0., 0.] # shift along meaningless direction, no shift for blind pupils => we see no difference



# sx = [-sr_amplitude, sr_amplitude, -sr_amplitude, sr_amplitude] # standard shifts
# sy = [sr_amplitude, sr_amplitude, -sr_amplitude, -sr_amplitude]

# sx = [-sr_amplitude, sr_amplitude, 0., 0.]
# sy = [0. , 0. , -sr_amplitude, sr_amplitude]

# sx = [0. , 0. , -sr_amplitude, sr_amplitude]
# sy = [-sr_amplitude, sr_amplitude, 0., 0.]

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
calib_sr = InteractionMatrix(ngs, atm, tel, dm, wfs, M2C = M2C_modal_dm, stroke = stroke)

# %% ------------------ PLOTS --------------------------------------------

from OOPAO.tools.displayTools import display_wfs_signals

plt.figure(1)
plt.imshow(np.abs(flat_frame_sr-flat_frame))
plt.title('SR pupils - No SR pupils (reference signal for a flat wavefront)\n0.25 pixel shifts')

plt.figure(2)
plt.semilogy(calib.eigenValues/calib.eigenValues.max(), 'b', label='no SR')
plt.semilogy(calib_sr.eigenValues/calib_sr.eigenValues.max(), 'r', label='SR')
plt.title('calib.eigenValues, '+str(n_subaperture)+' wfs subapertures, fourier basis used (x-axis only), '+str(sr_amplitude) +' pixels shift')
plt.legend()
plt.xlabel('# eigen mode')
plt.ylabel('normalized eigen value')

#%%

n_mode_display = 8

plt.figure(3)
plt.imshow(np.reshape(fourier_modes[:,n_mode_display], tel.pupil.shape))
plt.title("mode "+str(n_mode_display))

display_wfs_signals(wfs, calib.D[:,n_mode_display])
plt.title("No SR signal : mode "+str(n_mode_display))

display_wfs_signals(wfs, calib_sr.D[:,n_mode_display])
plt.title("SR signal : mode "+str(n_mode_display))

fig6, axs6 = plt.subplots(nrows=1, ncols=2)
img1 = axs6[0].imshow(np.abs(np.matmul(np.transpose(calib.D), calib.D)))
axs6[0].set_title('Sensitivity matrix - ' + modal_basis_name + ' - No SR\nAbsolute values')
img2 = axs6[1].imshow(np.abs(np.matmul(np.transpose(calib_sr.D), calib_sr.D)))
axs6[1].set_title('Sensitivity matrix - ' + modal_basis_name + ' - SR\nAbsolute values')
plt.colorbar(img1, ax=axs6[0], fraction=0.046, pad=0.04)
plt.colorbar(img2, ax=axs6[1], fraction=0.046, pad=0.04)