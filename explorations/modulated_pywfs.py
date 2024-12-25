# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:27:38 2024

@author: fleroux
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

from fanch.tools import get_circular_pupil, get_tilt, zeros_padding
from fanch.pupil_masks import get_modulation_phase_screens
from fanch.focus_masks import get_4pywfs_phase_mask
from fanch.propagation import get_focal_plane_image, get_ffwfs_frame
from fanch.basis.fourier import compute_real_fourier_basis, extract_diagonal_frequencies

#%%

dirc = Path(__file__).parent

filename = "lo_pywfs_resolved.mp4"

#%% -------------- PARAMETERS ---------------------

# telescope
n_px = 128

# modulation
modulation_radius = 10. # pixels/zeros_padding_factor [1/D m^-1]
n_modulation_points = 32

# focal plane sampling
zeros_padding_factor = 3

# input phase
input_phase_amplitude = 10.
n_mode = 3

#%% ---------------- TELESCOPE --------------------

pupil = get_circular_pupil(n_px)

#%% ---------------- MODULATOR --------------------

modulation_phase_screens = get_modulation_phase_screens(pupil, n_modulation_points, modulation_radius)
    
# %% ------------------ FOCAL PLANE MASK -----------

pywfs_tilt_amplitude = n_px * zeros_padding_factor * 0.5 * np.pi 
mask = get_4pywfs_phase_mask(2*[n_px*zeros_padding_factor], pywfs_tilt_amplitude)

# %% --------------------- PHASOR ------------------

pupil_pad = zeros_padding(pupil, zeros_padding_factor)

phasor = get_tilt(2*[n_px], theta=1.25*np.pi) * pupil
phasor[pupil!=0] = (phasor[pupil!=0] - phasor[pupil!=0].min())/(phasor[pupil!=0].max()-phasor[pupil!=0].min()) # normalization
phasor = 1/zeros_padding_factor * 2**0.5 * np.pi * phasor

# phasor = 0.*phasor

# %% ------------------ PROPAGATION ------------------

#%% Input phase

basis = compute_real_fourier_basis(n_px=n_px, return_map=True)

diagonal_basis = extract_diagonal_frequencies(basis)

phase_in = input_phase_amplitude*diagonal_basis[:,:,n_mode]

pupil = pupil*np.exp(1j*2*np.pi*phase_in)

#%% Non-modulated pywfs

pupil_pad = zeros_padding(pupil*np.exp(1j*phasor), zeros_padding_factor)

focal_plane_detector = get_focal_plane_image(pupil_pad)

mask_complex_amplitude = np.exp(1j*mask)

pywfs_detector = get_ffwfs_frame(pupil_pad, mask_complex_amplitude)

#%% Modulated pywfs

mpywfs_resolved_detector = np.zeros([zeros_padding_factor*pupil.shape[0], 
                            zeros_padding_factor*pupil.shape[1], 
                            modulation_phase_screens.shape[2]])
gsc_resolved_detector = np.zeros([zeros_padding_factor*pupil.shape[0], 
                         zeros_padding_factor*pupil.shape[1], 
                         modulation_phase_screens.shape[2]])

for k in range(modulation_phase_screens.shape[2]):
    
    pupil_pad = zeros_padding(pupil*np.exp(1j*phasor)*
                np.exp(1j*modulation_phase_screens[:,:,k]), zeros_padding_factor)
    
    mpywfs_resolved_detector[:,:,k] = get_ffwfs_frame(pupil_pad, mask_complex_amplitude)
    
    gsc_resolved_detector[:,:,k] = get_focal_plane_image(pupil_pad)

mpywfs_detector = mpywfs_resolved_detector.sum(axis=2)
gsc_detector = gsc_resolved_detector.sum(axis=2)

#%% PLOTS

# plt.figure(1)
# plt.imshow(phase_in)

# plt.figure(2)
# plt.imshow(mpywfs_detector)

# plt.figure(3)
# plt.imshow(gsc_detector)

# plt.figure(4)
# plt.imshow(focal_plane_detector)

# #%%

# modulation_point = 7

# fig, axs = plt.subplots(nrows=1, ncols=2)
# axs[0].imshow(gsc_resolved_detector[:,:,modulation_point])
# axs[1].imshow(mpywfs_resolved_detector[:,:,modulation_point])

#%%

fig, ax = plt.subplots()

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(mpywfs_resolved_detector.shape[2]):
    im = ax.imshow(mpywfs_resolved_detector[:,:,i])
    if i == 0:
        #ax.imshow(basis[:,:,i])  # show an initial one first
        ax.imshow(mpywfs_detector)
    ims.append([im])

ax.set_title('MPYWFS detector accross modulation points')

ani = animation.ArtistAnimation(fig, ims, interval=400, blit=True,
                                repeat_delay=2000)

ani.save(dirc / filename)
