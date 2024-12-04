# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:08:36 2024

@author: fleroux
"""

import matplotlib.pyplot as plt
import numpy as np

n_subaperture = 32 # 124

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



#%% super resolution

sx = [0.025,0,0,0]
sy = [0.025,0,0,0]
wfs.apply_shift_wfs(sx = sx, sy = sy)

tel*wfs
wfs.wfs_measure(tel.pupil)
flat_frame_sr = wfs.cam.frame


# %% ------------------ PLOTS --------------------------------------------

# plt.figure(1)
# plt.imshow(np.abs(flat_frame_sr-flat_frame))

plt.figure(2)
plt.imshow(np.abs(np.array(wfs.mask)[0,:,:]))

plt.figure(3)
plt.imshow(np.angle(np.array(wfs.mask)[1,:,:]))

#%%
 
# from OOPAO.tools.tools import compute_fourier_mode


# S = compute_fourier_mode(pupil=tel.pupil, spatial_frequency = 5, angle_deg=45) 
 
 
# wfs.wfs_measure(S*1e-9*tel.pupil)

# signal_sr = wfs.signal_2D

# plt.figure(3)
# plt.imshow(wfs.signal_2D)
# plt.title('WFS Camera Frame')