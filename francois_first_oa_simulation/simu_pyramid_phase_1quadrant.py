# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:43:01 2024

@author: fleroux
"""

import matplotlib.pyplot as plt
import numpy as np

from ifmta.paterns import Disk
from ifmta.phase_screens import Tilt

from tools import Zeros_padding

from amplitude_screens import AmplitudeScreensPywfs

plt.close('all')

# --------------- PARAMETERS --------------

pupil_support_size = 128
pupil_diameter = 1 # [m]

zeros_padding_factor = 4

delta_phi_pupil_plane = np.pi/(zeros_padding_factor) # phase shift to apply in pupil plane for centering the PSF on top of the PYWFS

delta_phi_focal_plane = 200.0*np.pi # phase shift to be applied in focal plane to separate the quadrants

# --------------- MAIN ---------------

# pupil generation
pupil_complex_amplitude = Disk(pupil_support_size)

# energy normalisation
pupil_complex_amplitude = pupil_complex_amplitude/np.sum(np.abs(pupil_complex_amplitude))**0.5

# centering phase application
tilt_x = Tilt(delta_phi=-delta_phi_pupil_plane, size_support=pupil_complex_amplitude.shape, direction='x')
tilt_y = Tilt(delta_phi=delta_phi_pupil_plane, size_support=pupil_complex_amplitude.shape, direction='y')
pupil_phase = tilt_x + tilt_y
pupil_centered_complex_amplitude = pupil_complex_amplitude * np.exp(1j*pupil_phase)

# zero padding
pupil_complex_amplitude = Zeros_padding(pupil_complex_amplitude, zeros_padding_factor=zeros_padding_factor)
pupil_centered_complex_amplitude = Zeros_padding(pupil_centered_complex_amplitude, zeros_padding_factor=zeros_padding_factor)

# propagation towards focal plane
focal_plane_complex_amplitude = np.fft.fftshift(np.fft.fft2(pupil_complex_amplitude, norm='ortho'))
focal_plane_centered_complex_amplitude = np.fft.fftshift(np.fft.fft2(pupil_centered_complex_amplitude, norm='ortho'))

# pywfs amplitude mask
pyfs_amplitude_mask = AmplitudeScreensPywfs(focal_plane_complex_amplitude.shape[0])

# pywfs phase mask 1 quadrant

focal_plane_phase_shift_11 = Tilt(delta_phi=-delta_phi_focal_plane, size_support=focal_plane_complex_amplitude.shape, direction='x') +\
                             Tilt(delta_phi=delta_phi_focal_plane, size_support=focal_plane_complex_amplitude.shape, direction='y')

# apply mask 1 quadrant
focal_plane_centered_complex_amplitude = focal_plane_centered_complex_amplitude * np.exp(1j*focal_plane_phase_shift_11)
focal_plane_centered_complex_amplitude = focal_plane_centered_complex_amplitude * pyfs_amplitude_mask[0]

# propagation towards conjugated pupil plane
pupil_plane_centered_complex_amplitude = np.fft.fft2(np.fft.ifftshift(focal_plane_centered_complex_amplitude), norm='ortho')

# --------------- PLOTS ---------------

# ok
fig1, axs1 = plt.subplots(nrows=2, ncols=2)
axs1[0,0].imshow(np.abs(focal_plane_complex_amplitude)**2)
axs1[0,1].imshow(np.abs(focal_plane_centered_complex_amplitude)**2)
axs1[1,0].imshow(np.abs(pupil_plane_centered_complex_amplitude)**2)

#ok
fig2, axs2 = plt.subplots(nrows=1, ncols=4)
axs2[0].imshow(pyfs_amplitude_mask[0]*np.abs(focal_plane_centered_complex_amplitude)**2)
axs2[1].imshow(pyfs_amplitude_mask[1]*np.abs(focal_plane_centered_complex_amplitude)**2)
axs2[2].imshow(pyfs_amplitude_mask[2]*np.abs(focal_plane_centered_complex_amplitude)**2)
axs2[3].imshow(pyfs_amplitude_mask[3]*np.abs(focal_plane_centered_complex_amplitude)**2)





