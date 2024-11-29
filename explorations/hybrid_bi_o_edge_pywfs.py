# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:30:40 2024

@author: fleroux
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:35:32 2024

@author: fleroux
"""

import numpy as np
import matplotlib.pyplot as plt

from aoerror.ffwfs import *
from OOPAO.Telescope import Telescope
from OOPAO.Source import Source

#%%

def hybrid_pywfs(resolution, n_faces_1, n_faces_2, radius):
    
    mask = pyramid_mask(resolution, n_faces_1)
    
    coordinates = np.arange(-mask.shape[1]//2, mask.shape[1]//2) # N = 4 : [-2,-1,0,1]
    [X,Y] = np.meshgrid(coordinates,coordinates)
    radial_coordinates = (X**2+Y**2)**0.5
    
    mask1 =  pyramid_mask(resolution, n_faces_1)
    mask2 = pyramid_mask(resolution, n_faces_2)
    
    mask[radial_coordinates<radius] = mask2[radial_coordinates<radius]
    
    return mask, mask1

#%%

optical_band = 'V'
magnitude = 0

resolution = 256 # number of pixel accross pupil diameter
diameter = 10. # [m] telescope diameter

rmod = 5 # [lambda/D] modulation radius
samp = 2
n_faces = 4

zeros_padding_factor = 2

src = Source(optical_band, magnitude)

tel = Telescope(resolution, diameter)

src*tel

tel.computePSF(zeroPaddingFactor=zeros_padding_factor) # default zp = 2
psf_calib = tel.PSF
psf_calib = psf_calib/psf_calib.sum() # ! normalization

modu = modulation(resolution*zeros_padding_factor, rmod, samp)
pywfs_mask = pyramid_mask(resolution*zeros_padding_factor, n_faces)

bioedge_amplitude_mask_x = np.zeros((resolution*zeros_padding_factor, resolution*zeros_padding_factor), dtype='complex128')
bioedge_amplitude_mask_x[:bioedge_amplitude_mask_x.shape[0]//2, :] = 1.

n_faces_1 = 2
n_faces_2 = 4
radius = rmod*samp-1

hybrid_mask, mask1 = hybrid_pywfs(zeros_padding_factor*resolution, n_faces_1, n_faces_2, radius)

#%%

sensitivity_map_pywfs = ffwfs_sensitivity(pywfs_mask, modu, psf_calib, psf_calib)
sensitivity_map_bioedge = ffwfs_sensitivity(bioedge_amplitude_mask_x, modu, psf_calib, psf_calib)
sensitivity_hybrid_mask = ffwfs_sensitivity(hybrid_mask, modu, psf_calib, psf_calib)


#%%

fig1, axs1 = plt.subplots(ncols=2, nrows=2)

axs1[0,0].imshow(np.angle(pywfs_mask))
axs1[0,1].imshow(np.angle(hybrid_mask))

im1=axs1[1,0].imshow(sensitivity_map_pywfs)
plt.colorbar(im1, fraction=0.046, pad=0.04, ax=axs1[1,0])
im2=axs1[1,1].imshow(sensitivity_hybrid_mask)
plt.colorbar(im2, fraction=0.046, pad=0.04, ax=axs1[1,1])
