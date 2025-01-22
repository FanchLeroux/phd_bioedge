# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 09:19:27 2024

@author: fleroux
"""

import matplotlib.pyplot as plt
import numpy as np

from OOPAO.tools.tools import compute_fourier_mode, zero_pad_array

from fanch.tools import get_circular_pupil

n = 256

angle_deg = 45.0

spatial_frequency = 5.0

zero_padding_factor = 4

fourier_mode_amplitude = .3

#pupil = np.ones(n)
pupil = get_circular_pupil(n)

# amplitude fourier mode

fourier_mode = pupil*fourier_mode_amplitude*compute_fourier_mode(pupil, spatial_frequency, angle_deg)

fourier_mode_zeros_padded = zero_pad_array(fourier_mode, padding=fourier_mode.shape[0]//2 * (zero_padding_factor-1))

fourier_mode_tilde = np.fft.fftshift(np.fft.fft2(fourier_mode))

fourier_mode_zeros_padded_tilde = np.fft.fftshift(np.fft.fft2(fourier_mode_zeros_padded))

#plt.imshow(np.abs(fourier_mode_zeros_padded_tilde))

# phase fourier mode

fourier_mode_phase = pupil*np.exp(1j*fourier_mode)

fourier_mode_phase_zeros_padded = zero_pad_array(fourier_mode_phase, padding=fourier_mode.shape[0]//2 * (zero_padding_factor-1))

fourier_mode_phase_zeros_padded_tilde = np.fft.fftshift(np.fft.fft2(fourier_mode_phase_zeros_padded))

fig2, axs2 = plt.subplots(nrows=1, ncols=5)
axs2[0].imshow(np.abs(fourier_mode_phase_zeros_padded)[384:640,384:640])
im=axs2[1].imshow(fourier_mode_zeros_padded[384:640,384:640])
axs2[2].imshow(np.abs(fourier_mode_phase_zeros_padded_tilde)**2)

# [360:660,360:660]

# foucault knife

amplitude_mask = np.zeros([n*zero_padding_factor]*2)
amplitude_mask[:amplitude_mask.shape[0]//2+10,:] = 1

fourier_mode_phase_zeros_padded_tilde_masked = fourier_mode_phase_zeros_padded_tilde * amplitude_mask

pupil_plane_complex_amplitude = np.fft.fft2(np.fft.ifftshift(fourier_mode_phase_zeros_padded_tilde_masked))

size = 50
axs2[3].imshow(np.abs(fourier_mode_phase_zeros_padded_tilde**2)
               [fourier_mode_phase_zeros_padded_tilde.shape[0]//2 - size:fourier_mode_phase_zeros_padded_tilde.shape[0]//2 + size,
                fourier_mode_phase_zeros_padded_tilde.shape[0]//2 - size:fourier_mode_phase_zeros_padded_tilde.shape[0]//2 + size])

axs2[4].imshow(np.abs(pupil_plane_complex_amplitude[384:640,384:640])**2)

axs2[0].set_title("amplitude")
axs2[1].set_title("phase")
axs2[2].set_title("focal plane irradiance")
axs2[3].set_title("zoom on focal plane irradiance")
axs2[4].set_title("pupil plane irradiance after filtering")

plt.colorbar(im,ax=axs2[1],fraction=0.046, pad=0.04)

fig, axs = plt.subplots(nrows=1,ncols=4)
axs[0].imshow(np.abs(fourier_mode_phase_zeros_padded)[384:640,384:640])
axs[1].imshow(fourier_mode_zeros_padded[384:640,384:640])
axs[2].imshow(np.abs(fourier_mode_phase_zeros_padded_tilde**2)
               [fourier_mode_phase_zeros_padded_tilde.shape[0]//2 - size:fourier_mode_phase_zeros_padded_tilde.shape[0]//2 + size,
                fourier_mode_phase_zeros_padded_tilde.shape[0]//2 - size:fourier_mode_phase_zeros_padded_tilde.shape[0]//2 + size])
axs[3].imshow(np.abs(pupil_plane_complex_amplitude[384:640,384:640])**2)

axs[0].set_title("amplitude")
axs[1].set_title("phase")
axs[2].set_title("focal plane irradiance")
axs[3].set_title("detector plane irradiance")