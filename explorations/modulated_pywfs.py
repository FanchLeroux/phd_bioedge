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
from fanch.basis.fourier import compute_real_fourier_basis,\
                                extract_diagonal_frequencies

#%%

dirc = Path(__file__).parent.parent\
    .parent / "outputs" / "phd_bioedge" / "exploration"

filename = "lo_romanesco_resolved.mp4"

#%% -------------- PARAMETERS ---------------------

# telescope
n_px = 32

# modulation
modulation_radius = 3. # pixels/zeros_padding_factor [1/D m^-1]
n_modulation_points = 64

# focal plane sampling
zeros_padding_factor = 3

# input phase
input_phase_amplitude = 1. # 10 : see modal cross coupling
n_mode = 7

#%% ---------------- TELESCOPE --------------------

pupil = get_circular_pupil(n_px)

#%% ---------------- MODULATOR --------------------

modulation_phase_screens = get_modulation_phase_screens(pupil, 
                           n_modulation_points, modulation_radius)
    
# %% ------------------ FOCAL PLANE MASK -----------

pywfs_tilt_amplitude = n_px * zeros_padding_factor * 0.5 * np.pi 
mask = get_4pywfs_phase_mask(2*[n_px*zeros_padding_factor], pywfs_tilt_amplitude)

mask_complex_amplitude = np.exp(1j*mask)

# %% --------------------- PHASOR ------------------

phasor = get_tilt(2*[n_px], theta=1.25*np.pi) * pupil
phasor[pupil!=0] = (phasor[pupil!=0] - phasor[pupil!=0].min())\
                   /(phasor[pupil!=0].max()-phasor[pupil!=0].min()) # normalization
phasor = 1/zeros_padding_factor * 2**0.5 * np.pi * phasor

pupil = pupil * np.exp(1j*phasor)

# %% ------------------ PROPAGATION ------------------

#%% Reference frame without modulation

pupil_pad = zeros_padding(pupil, zeros_padding_factor)
frame_ref_no_modul = get_ffwfs_frame(pupil_pad, mask_complex_amplitude)

#%% Reference frame with modulation

mpywfs_resolved_detector_no_modul = np.zeros([zeros_padding_factor*pupil.shape[0], 
                            zeros_padding_factor*pupil.shape[1], 
                            modulation_phase_screens.shape[2]])
gsc_resolved_detector_no_modul = np.zeros([zeros_padding_factor*pupil.shape[0], 
                         zeros_padding_factor*pupil.shape[1], 
                         modulation_phase_screens.shape[2]])

for k in range(modulation_phase_screens.shape[2]):
    
    pupil_pad = zeros_padding(pupil*np.exp(1j*modulation_phase_screens[:,:,k]), 
                              zeros_padding_factor)
    
    mpywfs_resolved_detector_no_modul[:,:,k] = get_ffwfs_frame(pupil_pad, 
                                      mask_complex_amplitude)
    
    gsc_resolved_detector_no_modul[:,:,k] = get_focal_plane_image(pupil_pad)

mpywfs_detector_no_modul = mpywfs_resolved_detector_no_modul.sum(axis=2)
gsc_detector_no_modul = gsc_resolved_detector_no_modul.sum(axis=2)

#%% Input phase

basis = compute_real_fourier_basis(n_px=n_px, return_map=True)

diagonal_basis = extract_diagonal_frequencies(basis)

phase_in = input_phase_amplitude*diagonal_basis[:,:,n_mode]

pupil = pupil*np.exp(1j*2*np.pi*phase_in)

#%% Non-modulated pywfs

focal_plane_detector = get_focal_plane_image(zeros_padding(
                       pupil, zeros_padding_factor))



pywfs_detector = get_ffwfs_frame(pupil_pad, mask_complex_amplitude)

#%% Modulated pywfs

mpywfs_resolved_detector = np.zeros([zeros_padding_factor*pupil.shape[0], 
                            zeros_padding_factor*pupil.shape[1], 
                            modulation_phase_screens.shape[2]])
gsc_resolved_detector = np.zeros([zeros_padding_factor*pupil.shape[0], 
                         zeros_padding_factor*pupil.shape[1], 
                         modulation_phase_screens.shape[2]])

for k in range(modulation_phase_screens.shape[2]):
    
    pupil_pad = zeros_padding(pupil*np.exp(1j*modulation_phase_screens[:,:,k]), 
                              zeros_padding_factor)
    
    mpywfs_resolved_detector[:,:,k] = get_ffwfs_frame(pupil_pad, 
                                      mask_complex_amplitude)\
                                    - mpywfs_resolved_detector_no_modul[:,:,k]
    
    gsc_resolved_detector[:,:,k] = get_focal_plane_image(pupil_pad)

mpywfs_detector = mpywfs_resolved_detector.sum(axis=2)
gsc_detector = gsc_resolved_detector.sum(axis=2)

#%% Subtract reference signal

#mpywfs_signal = mpywfs_detector - mpywfs_detector_no_modul

#plt.imshow(mpywfs_detector)

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

# ani.save(dirc / filename)
