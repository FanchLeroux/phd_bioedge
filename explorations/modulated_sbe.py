# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 20:49:14 2024

@author: fleroux
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

from fanch.tools import get_circular_pupil, get_tilt, zeros_padding
from fanch.pupil_masks import get_modulation_phase_screens
from fanch.focus_masks import get_amplitude_bioedge_masks
from fanch.propagation import get_focal_plane_image, get_ffwfs_frame
from fanch.basis.fourier import compute_real_fourier_basis,\
                                extract_diagonal_frequencies
                                
#%%

dirc = Path(__file__).parent

filename = "lo_msbe_resolved.mp4"

#%% -------------- PARAMETERS ---------------------

# telescope
n_px = 32

# modulation
modulation_radius = 10. # pixels/zeros_padding_factor [1/D m^-1]
n_modulation_points = 32

# focal plane sampling
zeros_padding_factor = 2

# input phase
input_phase_amplitude = 1.
n_mode = 5

#%% ---------------- TELESCOPE --------------------

pupil = get_circular_pupil(n_px)

#%% ---------------- MODULATOR --------------------

modulation_phase_screens = get_modulation_phase_screens(pupil, 
                           n_modulation_points, modulation_radius)
    
# %% ------------------ FOCAL PLANE MASKS -----------

sbe_amplitude_masks = get_amplitude_bioedge_masks(zeros_padding_factor*n_px)

# %% --------------------- PHASOR ------------------

phasor = get_tilt(2*[n_px], theta=1.25*np.pi) * pupil
phasor[pupil!=0] = (phasor[pupil!=0] - phasor[pupil!=0].min())\
                   /(phasor[pupil!=0].max()-phasor[pupil!=0].min()) # normalization
phasor = 1/zeros_padding_factor * 2**0.5 * np.pi * phasor

pupil = pupil * np.exp(1j*phasor)

# %% ------------------ PROPAGATION ------------------

#%% Input phase

basis = compute_real_fourier_basis(n_px=n_px, return_map=True)

diagonal_basis = extract_diagonal_frequencies(basis)

phase_in = input_phase_amplitude*diagonal_basis[:,:,n_mode]

pupil = pupil*np.exp(1j*2*np.pi*phase_in)

#%% Modulated bio-edge

msbe_resolved_detector = np.zeros([zeros_padding_factor*pupil.shape[0], 
                            zeros_padding_factor*pupil.shape[1], 
                            modulation_phase_screens.shape[2]])

gsc_resolved_detector = np.zeros([zeros_padding_factor*pupil.shape[0], 
                         zeros_padding_factor*pupil.shape[1], 
                         modulation_phase_screens.shape[2]])

for k in range(modulation_phase_screens.shape[2]):
    
    pupil_pad = zeros_padding(pupil*np.exp(1j*modulation_phase_screens[:,:,k]), 
                              zeros_padding_factor)
    
    msbe_resolved_detector[:,:,k] = get_ffwfs_frame(pupil_pad, sbe_amplitude_masks[2])
    
    gsc_resolved_detector[:,:,k] = get_focal_plane_image(pupil_pad)

msbe_detector = msbe_resolved_detector.sum(axis=2)
gsc_detector = gsc_resolved_detector.sum(axis=2)

#%%

fig, ax = plt.subplots()

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(msbe_resolved_detector.shape[2]):
    im = ax.imshow(msbe_resolved_detector[:,:,i])
    if i == 0:
        #ax.imshow(basis[:,:,i])  # show an initial one first
        ax.imshow(msbe_detector)
    ims.append([im])

ax.set_title('MSBE detector accross modulation points')

ani = animation.ArtistAnimation(fig, ims, interval=400, blit=True,
                                repeat_delay=2000)

# ani.save(dirc / filename)
