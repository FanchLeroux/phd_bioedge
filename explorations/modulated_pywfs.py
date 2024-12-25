# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:27:38 2024

@author: fleroux
"""

import matplotlib.pyplot as plt
import numpy as np

from fanch.tools import get_circular_pupil, get_tilt, zeros_padding

from fanch.pupil_masks import get_modulation_phase_screens

from fanch.focus_masks import get_4pywfs_phase_mask

from fanch.propagation import get_focal_plane_image, get_ffwfs_frame

#%% -------------- PARAMETERS ---------------------

# telescope
npx = 32

# modulation
modulation_radius = 5. # pixels/zeros_padding_factor [1/D m^-1]
n_modulation_points = 8

# focal plane sampling
zeros_padding_factor = 3

#%% ---------------- TELESCOPE --------------------

pupil = get_circular_pupil(npx)

#%% ---------------- MODULATOR --------------------

modulation_phase_screens = get_modulation_phase_screens(pupil, n_modulation_points, modulation_radius)
    
# %% ------------------ FOCAL PLANE MASK -----------

pywfs_tilt_amplitude = npx * zeros_padding_factor * 0.5 * np.pi 
mask = get_4pywfs_phase_mask(2*[npx*zeros_padding_factor], pywfs_tilt_amplitude)

# %% --------------------- PHASOR ------------------

pupil_pad = zeros_padding(pupil, zeros_padding_factor)

phasor = get_tilt(2*[npx], theta=1.25*np.pi) * pupil
phasor[pupil!=0] = (phasor[pupil!=0] - phasor[pupil!=0].min())/(phasor[pupil!=0].max()-phasor[pupil!=0].min()) # normalization
phasor = 1/zeros_padding_factor * 2**0.5 * np.pi * phasor

# %% ------------------ PROPAGATION ------------------

#%% Non-modulated pywfs
pupil_pad = zeros_padding(pupil*np.exp(1j*phasor), zeros_padding_factor)

focal_plane_detector = get_focal_plane_image(pupil_pad)

mask_complex_amplitude = np.exp(1j*mask)

pywfs_detector = get_ffwfs_frame(pupil_pad, mask_complex_amplitude)

#%% Modulated pywfs

mpywfs_detector = np.zeros([zeros_padding_factor*pupil.shape[0], zeros_padding_factor*pupil.shape[1]])
gsc_detector = np.zeros([zeros_padding_factor*pupil.shape[0], zeros_padding_factor*pupil.shape[1]])

for k in range(modulation_phase_screens.shape[2]):
    
    pupil_pad = zeros_padding(pupil*np.exp(1j*phasor)*np.exp(1j*modulation_phase_screens[:,:,k]), zeros_padding_factor)
    
    mpywfs_detector += get_ffwfs_frame(pupil_pad, mask_complex_amplitude)
    
    gsc_detector += get_focal_plane_image(pupil_pad)

#%% PLOTS

plt.figure(1)
plt.imshow(mpywfs_detector)

plt.figure(2)
plt.imshow(gsc_detector)