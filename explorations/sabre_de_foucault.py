# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:31:28 2024

@author: fleroux
"""

import numpy as np
import matplotlib.pyplot as plt

from aoerror.ffwfs import *
from OOPAO.Telescope import Telescope
from OOPAO.Source import Source

#%%

def foucault_saber(resolution, h, l):
    
    x = np.arange(0, resolution//2)
    y = np.arange(-resolution//2, resolution//2) # N = 4 : [-2,-1,0,1]
    [X,Y] = np.meshgrid(x,y)

    mask = np.zeros((resolution, resolution//2), dtype='complex128')

    # 1    
    mask1 = np.zeros((resolution, l))
    X1 = X[:,:l]
    Y1 = Y[:,:l]
    
    mask1[Y1<X1*h/float(l)] = 1.
    
    mask[:,:l] = mask1
    
    # 2    
    mask2 = np.zeros((resolution, resolution//2 - l))
    X2 = X[:,l:]-l
    Y2 = Y[:,l:]
    mask2[Y2<h-X2*h/(resolution//2-float(l))] = 1.
    
    mask[:,l:] = mask2    
    final_mask = np.zeros((resolution, resolution), dtype='complex128')
    final_mask[:, :resolution//2] = mask
    
    mask = np.abs(mask-1)
    
    final_mask[:, resolution//2:] = np.flip(mask,axis=0)
    
    
    return final_mask





#%%

optical_band = 'V'
magnitude = 0

resolution = 2048 # number of pixel accross pupil diameter
diameter = 10. # [m] telescope diameter

rmod = 80 # [lambda/D] modulation radius
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

foucault_saber_mask = foucault_saber(resolution*zeros_padding_factor, (resolution*zeros_padding_factor)//10, 
                                     (resolution*zeros_padding_factor)//8)

#%%

sensitivity_map_pywfs = ffwfs_sensitivity(pywfs_mask, modu, psf_calib, psf_calib)
sensitivity_map_bioedge = ffwfs_sensitivity(bioedge_amplitude_mask_x, modu, psf_calib, psf_calib)
sensitivity_map_foucault_saber = ffwfs_sensitivity(foucault_saber_mask, modu, psf_calib, psf_calib)


#%%

fig1, axs1 = plt.subplots(ncols=3, nrows=2)

axs1[0,0].imshow(np.angle(pywfs_mask))
axs1[0,1].imshow(np.abs(bioedge_amplitude_mask_x))
axs1[0,2].imshow(np.abs(foucault_saber_mask))

axs1[1,0].imshow(sensitivity_map_pywfs)
axs1[1,1].imshow(sensitivity_map_bioedge)
axs1[1,2].imshow(sensitivity_map_foucault_saber)
