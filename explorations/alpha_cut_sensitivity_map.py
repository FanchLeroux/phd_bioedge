# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from aoerror.ffwfs import *
from OOPAO.Telescope import Telescope
from OOPAO.Source import Source
from OOPAO.Atmosphere import Atmosphere

#%%

optical_band = 'V'
magnitude = 0

r0 = 0.1 # [m] fried parameter
L0 = 30 # [m] VonKarman external scale
wind_speed = 8 # [m/s]

resolution = 2048 # number of pixel accross pupil diameter
diameter = 10. # [m] telescope diameter

rmod = 0 # [lambda/D] modulation radius
samp = 2
n_faces = 4

zeros_padding_factor = 2

src = Source(optical_band, magnitude)

tel = Telescope(resolution, diameter)

src*tel

tel.computePSF(zeroPaddingFactor=zeros_padding_factor) # default zp = 2
psf_calib = tel.PSF
psf_calib = psf_calib/psf_calib.sum() # ! normalization

# atm = Atmosphere(telescope = tel, r0 = r0, L0 = L0, windSpeed = [wind_speed], fractionalR0 = [1.0], windDirection = [30], altitude = [0])
# atm.initializeAtmosphere(tel)
    
# tel+atm

tel.computePSF(zeroPaddingFactor=zeros_padding_factor) # default zp = 2
psf = tel.PSF
psf = psf/psf.sum() # ! normalization

modu = modulation(resolution*zeros_padding_factor, rmod, samp)
pywfs_mask = pyramid_mask(resolution*zeros_padding_factor, n_faces)

bioedge_amplitude_mask_x = np.zeros((resolution*zeros_padding_factor, resolution*zeros_padding_factor), dtype='complex128')
bioedge_amplitude_mask_x[:bioedge_amplitude_mask_x.shape[0]//2, :] = 1.

#%%

sensitivity_map_pywfs = ffwfs_sensitivity(pywfs_mask, modu, psf_calib, psf_calib)
sensitivity_map_bioedge = ffwfs_sensitivity(bioedge_amplitude_mask_x, modu, psf_calib, psf_calib)

sensitivity_map_pywfs_turb = ffwfs_sensitivity(pywfs_mask, modu, psf_calib, psf)
sensitivity_map_bioedge_turb = ffwfs_sensitivity(bioedge_amplitude_mask_x, modu, psf_calib, psf)

#%%

alpha_cut_list = [5,15,25,35,40,45]
alpha_cut_list = np.deg2rad(alpha_cut_list)
fig4, axs4 = plt.subplots(ncols=2, nrows=1)
axs4[0].imshow(sensitivity_map_pywfs)
axs4[1].set_xlabel('spatial frequency modulus')
axs4[1].set_ylabel('sensitivity')
axs4[1].plot(sensitivity_map_pywfs[sensitivity_map_pywfs.shape[0]//2, sensitivity_map_pywfs.shape[1]//2:], 'r',label='0° cut')

coordinates = np.arange(-sensitivity_map_pywfs.shape[1]//2, sensitivity_map_pywfs.shape[1]//2) # N = 4 : [-2,-1,0,1]
[X,Y] = np.meshgrid(coordinates,coordinates)
radial_coordinates = (X**2+Y**2)**0.5
angles = np.arctan2(Y,X)
angles_sup = np.arctan2(Y+0.5,X)
angles_inf = np.arctan2(Y-0.5,X)

for alpha_cut in alpha_cut_list:

    test = np.ones(angles.shape)
    test[angles_inf>=alpha_cut] = 0
    test[angles_sup<=alpha_cut] = 0
    test[test.shape[0]//2,:] = 0
    test[test.shape[0]//2-1,:] = 0
    coordinates_alpha_cut = radial_coordinates[test==1]
    
    
    # #%%
    
    # fig1, axs1 = plt.subplots(ncols=2, nrows=2)
    # axs1[0,0].imshow(sensitivity_map_pywfs)
    # axs1[0,1].imshow(sensitivity_map_bioedge)
    # axs1[1,0].imshow(sensitivity_map_pywfs_turb)
    # axs1[1,1].imshow(sensitivity_map_bioedge_turb)
    
    # fig2, axs2 = plt.subplots(ncols=2, nrows=2)
    # axs2[0,0].plot(np.diag(sensitivity_map_pywfs)[np.diag(sensitivity_map_pywfs).shape[0]//2:], 'b', label='45° cut')
    # axs2[0,0].plot(sensitivity_map_pywfs[sensitivity_map_pywfs.shape[0]//2, sensitivity_map_pywfs.shape[1]//2:], 'r',label='0° cut')
    # axs2[0,0].legend()
    
    # #%%
    
    # fig3, axs3 = plt.subplots(ncols=2, nrows=1)
    # axs3[0].imshow(sensitivity_map_pywfs)
    # axs3[1].set_xlabel('pixel')
    # axs3[1].set_ylabel('sensitivity')
    # axs3[1].plot(np.diag(sensitivity_map_pywfs)[np.diag(sensitivity_map_pywfs).shape[0]//2:], 'b', label='45° cut')
    # axs3[1].plot(sensitivity_map_pywfs[sensitivity_map_pywfs.shape[0]//2, sensitivity_map_pywfs.shape[1]//2:], 'r',label='0° cut')
    # axs3[1].legend()
    
   
    
    
    
    axs4[1].plot(coordinates_alpha_cut, sensitivity_map_pywfs[test==1], 
                 label= str(np.round(np.rad2deg(alpha_cut)))+'° cut')
    

axs4[1].legend()

