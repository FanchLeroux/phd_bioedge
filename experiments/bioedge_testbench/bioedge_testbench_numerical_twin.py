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
    [X,Y] = np.meshgrid(np.arange(0, shape[0]), np.arange(0, shape[1]))
    tilt_theta = np.cos(theta) * X + np.sin(theta) * Y
    Y = np.flip(Y, axis=0) # change orientation
    tilt_theta = (tilt_theta - tilt_theta.min())/(tilt_theta.max()- tilt_theta.min())
    
    return amplitude*tilt_theta

#%% -----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope

# create the Telescope object
tel = Telescope(resolution           = 4*n_subaperture,   # resolution of the telescope in [pix]
                diameter             = 8,                 # diameter in [m]        
                samplingTime         = 1/1000,            # Sampling time in [s] of the AO loop
                centralObstruction   = 0.,                # Central obstruction in [%] of a diameter 
                display_optical_path = False,             # Flag to display optical path
                fov                  = 10 )               # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of 
                                                          # the phase screens but is uncompatible with off-axis targets

#%% -----------------------     NGS   ----------------------------------

# GHOST wavelength : 770 nm, full bandwidth = 20 nm
# 'I2' band : 750 nm, bandwidth? = 33 nm

from OOPAO.Source import Source

# create the Natural Guide Star object
ngs = Source(optBand     = 'I2',          # Optical band (see photometry.py)
             magnitude   = 8,             # Source Magnitude
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
flat_raw_data = wfs.bioFrame



#%% super resolution

sx = [-0.25, 0.25, -0.25, 0.25] # pixels on wfs.bioFrame
sy = [0.25, 0.25, -0.25, -0.25]         # pixels on wfs.bioFrame
#wfs.apply_shift_wfs(sx = sx, sy = sy, units='pixels')

# extract masks         (convention : left|right ; top/bottom)
# mask[0] ---> 1|0
# mask[1] ---> 0|1
# mask[2] ---> 1/0   
# mask[3] ---> 0/1  
mask = wfs.mask
mask_sr_raw_data = deepcopy(mask)

# apply shifts manually - for the raw data

for k in range(len(mask)):
    tilt_x = get_tilt(mask[k].shape, theta = 0., amplitude = 2.*np.pi*sx[k])
    tilt_y = get_tilt(mask[k].shape, theta = np.pi/2, amplitude = 2.*np.pi*sy[k])
    mask_sr_raw_data[k] = mask[k]*np.exp(1j*(tilt_x+tilt_y))


# replace usual-masks by SR-masks
wfs.mask = mask_sr_raw_data

tel*wfs
wfs.wfs_measure(tel.pupil)
flat_raw_data_sr = wfs.bioFrame

# apply shifts manually - for the binned data (wfs.cam.frame)

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

tel*wfs
wfs.wfs_measure(tel.pupil)
flat_frame_sr = wfs.cam.frame

#%% Deformable mirror

from OOPAO.DeformableMirror import DeformableMirror

dm = DeformableMirror(tel, nSubap=n_subaperture)

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


#%% -----------------------     Modal Basis - Zernike  ----------------------------------

from OOPAO.Zernike import Zernike

#n_modes_callib

n_modes = int(np.pi * wfs.nSubap **2)

# create Zernike Object
Z = Zernike(tel,n_modes)
# compute polynomials for given telescope
Z.computeZernike(tel)

#%% Interaction Matrix

wfs.mask = mask # mask_sr_cam_frame

amplitude = 10e-9 # [m]

interraction_matrix_frames = np.empty([wfs.signal_2D.shape[0], wfs.signal_2D.shape[1], n_modes])
interraction_matrix = np.empty([wfs.signal.shape[0], n_modes])

for m in range(Z.modesFullRes.shape[2]):
    wfs.wfs_measure(amplitude * 2*np.pi/wfs.telescope.src.wavelength * Z.modesFullRes[:,:,m]/np.std(Z.modesFullRes[:,:,m]))
    interraction_matrix_frames[:,:,m] = wfs.signal_2D
    interraction_matrix[:,m] = wfs.signal
    

#%%
    
sensitivity_matrix = np.matmul(np.transpose(interraction_matrix), interraction_matrix)
plt.figure(1)
plt.imshow(sensitivity_matrix)

#%% Interaction Matrix - sr

wfs.mask = mask_sr_cam_frame # mask_sr_cam_frame

amplitude = 10e-9 # [m]

interraction_matrix_frames_sr = np.empty([wfs.signal_2D.shape[0], wfs.signal_2D.shape[1], n_modes])
interraction_matrix_sr = np.empty([wfs.signal.shape[0], n_modes])

for m in range(Z.modesFullRes.shape[2]):
    wfs.wfs_measure(amplitude * 2*np.pi/wfs.telescope.src.wavelength * Z.modesFullRes[:,:,m]/np.std(Z.modesFullRes[:,:,m]))
    interraction_matrix_frames_sr[:,:,m] = wfs.signal_2D
    interraction_matrix_sr[:,m] = wfs.signal
    

#%%
    
sensitivity_matrix_sr = np.matmul(np.transpose(interraction_matrix_sr), interraction_matrix_sr)
plt.figure(2)
plt.imshow(sensitivity_matrix_sr)

# %% ------------------ PLOTS --------------------------------------------

plt.figure(3)
plt.imshow(np.log(np.abs(sensitivity_matrix_sr-sensitivity_matrix)))

plt.figure(4)
plt.imshow(np.abs(flat_frame_sr-flat_frame))

print(np.trace(sensitivity_matrix_sr)-np.trace(sensitivity_matrix))

# plt.figure(2)
# plt.imshow(np.abs(flat_raw_data_sr-flat_raw_data))

# plt.figure(3)
# plt.imshow(wfs.cam.frame)

# plt.figure(4)
# plt.imshow(wfs.bioFrame)

#%%
 
# from OOPAO.tools.tools import compute_fourier_mode


# S = compute_fourier_mode(pupil=tel.pupil, spatial_frequency = 5, angle_deg=45) 
 
 
# wfs.wfs_measure(S*1e-9*tel.pupil)

# signal_sr = wfs.signal_2D

# plt.figure(3)
# plt.imshow(wfs.signal_2D)
# plt.title('WFS Camera Frame')