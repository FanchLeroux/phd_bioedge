# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 11:54:22 2025

@author: lgs
"""

import pathlib

import numpy as np

from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike

#%%

dirc_data = pathlib.Path(__file__).parent.parent.parent.parent.parent.parent / "data"

#%%

#%%

n_subaperture = 20
n_pixels_in_slm_pupil = 1100

#%%

tel = Telescope(n_pixels_in_slm_pupil, diameter=8)
Z = Zernike(tel,int(np.pi * (n_subaperture/2)**2)) # create a Zernike object considering 300 polynomials
Z.computeZernike(tel) # compute the Zernike
zernike_modes = Z.modesFullRes

# std normalization
zernike_modes = zernike_modes / np.std(zernike_modes, axis=(0,1))

# set the mean value around pi
zernike_modes = (zernike_modes + np.pi)*tel.pupil[:,:,np.newaxis]

# scale from 0 to 255 for a 2pi phase shift
zernike_modes = zernike_modes * 255/(2*np.pi)

# convert to 8-bit integers
zernike_modes = zernike_modes.astype(np.uint8)

np.save(dirc_data / "slm" / "modal_basis" / "zernike_modes" / ("zernike_modes_" + 
                                                               str(n_pixels_in_slm_pupil) +
                                                               "_pixels_in_slm_pupil_" +
                                                               str(n_subaperture) +
                                                               "_subapertures.npy"), zernike_modes)
