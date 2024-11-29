# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 14:43:01 2024

@author: fleroux
"""

import matplotlib.pyplot as plt
import numpy as np

from paterns import Disk
from phase_screens import Tilt

from tools import Zeros_padding

from amplitude_screens import AmplitudeScreensPywfs

plt.close('all')

# --------------- PARAMETERS --------------

pupil_support_size = 128
pupil_diameter = 1 # [m]

zeros_padding_factor = 4

delta_phi_pupil_plane = np.pi/(zeros_padding_factor) # phase shift to apply in pupil plane for centering the PSF on top of the PYWFS

delta_phi_focal_plane = zeros_padding_factor/2 * pupil_support_size*np.pi # phase shift to be applied in focal plane to separate the quadrants

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
pywfs_amplitude_mask = AmplitudeScreensPywfs(focal_plane_complex_amplitude.shape[0])

# pywfs phase mask
focal_plane_phase_shift_11 = Tilt(delta_phi=delta_phi_focal_plane, size_support=focal_plane_complex_amplitude.shape, direction='x') +\
                             Tilt(delta_phi=-delta_phi_focal_plane, size_support=focal_plane_complex_amplitude.shape, direction='y')
                             
pywfs_complex_amplitude_mask = pywfs_amplitude_mask[0] * np.exp(1j*focal_plane_phase_shift_11) +\
                               pywfs_amplitude_mask[1] * np.flip(np.exp(1j*focal_plane_phase_shift_11), axis=1) +\
                               pywfs_amplitude_mask[2] * np.flip(np.exp(1j*focal_plane_phase_shift_11), axis=0) +\
                               pywfs_amplitude_mask[3] * np.flip(np.exp(1j*focal_plane_phase_shift_11), axis=(0,1))
                   

# apply mask 4 quadrant

# focal_plane_complex_amplitude = focal_plane_complex_amplitude * np.exp(1j*focal_plane_phase_shift_22) * pywfs_amplitude_mask[0] +\
#                                          focal_plane_complex_amplitude * np.exp(1j*focal_plane_phase_shift_11) * pywfs_amplitude_mask[3] +\
#                                          focal_plane_complex_amplitude * np.exp(1j*focal_plane_phase_shift_21) * pywfs_amplitude_mask[2] +\
#                                          focal_plane_complex_amplitude * np.exp(1j*focal_plane_phase_shift_12) * pywfs_amplitude_mask[1]

# focal_plane_centered_complex_amplitude = focal_plane_centered_complex_amplitude * np.exp(1j*focal_plane_phase_shift_22) * pywfs_amplitude_mask[0] +\
#                                          focal_plane_centered_complex_amplitude * np.exp(1j*focal_plane_phase_shift_11) * pywfs_amplitude_mask[3] +\
#                                          focal_plane_centered_complex_amplitude * np.exp(1j*focal_plane_phase_shift_21) * pywfs_amplitude_mask[2] +\
#                                          focal_plane_centered_complex_amplitude * np.exp(1j*focal_plane_phase_shift_12) * pywfs_amplitude_mask[1]

focal_plane_complex_amplitude = focal_plane_complex_amplitude * pywfs_complex_amplitude_mask
focal_plane_centered_complex_amplitude = focal_plane_centered_complex_amplitude * pywfs_complex_amplitude_mask

# propagation towards conjugated pupil plane
conjugated_pupil_plane_complex_amplitude = np.fft.fft2(np.fft.ifftshift(focal_plane_complex_amplitude), norm='ortho')
conjugated_pupil_plane_centered_complex_amplitude = np.fft.fft2(np.fft.ifftshift(focal_plane_centered_complex_amplitude), norm='ortho')

# --------------- PLOTS ---------------

# ok
fig1, axs1 = plt.subplots(nrows=2, ncols=2)
axs1[0,0].imshow((np.abs(focal_plane_complex_amplitude)**2)[245:265,245:265])
axs1[0,1].imshow((np.abs(focal_plane_centered_complex_amplitude)**2)[245:265,245:265])
axs1[1,0].imshow(np.abs(conjugated_pupil_plane_complex_amplitude)**2)
axs1[1,1].imshow(np.abs(conjugated_pupil_plane_centered_complex_amplitude)**2)

#ok
fig2, axs2 = plt.subplots(nrows=2, ncols=5)
axs2[0,0].imshow((pywfs_amplitude_mask[0]*np.abs(focal_plane_complex_amplitude)**2)[245:265,245:265])
axs2[0,1].imshow((pywfs_amplitude_mask[1]*np.abs(focal_plane_complex_amplitude)**2)[245:265,245:265])
axs2[0,2].imshow((pywfs_amplitude_mask[2]*np.abs(focal_plane_complex_amplitude)**2)[245:265,245:265])
axs2[0,3].imshow((pywfs_amplitude_mask[3]*np.abs(focal_plane_complex_amplitude)**2)[245:265,245:265])
axs2[0,4].imshow((np.abs(focal_plane_complex_amplitude)**2)[245:265,245:265])

axs2[1,0].imshow((pywfs_amplitude_mask[0]*np.abs(focal_plane_centered_complex_amplitude)**2)[245:265,245:265])
axs2[1,1].imshow((pywfs_amplitude_mask[1]*np.abs(focal_plane_centered_complex_amplitude)**2)[245:265,245:265])
axs2[1,2].imshow((pywfs_amplitude_mask[2]*np.abs(focal_plane_centered_complex_amplitude)**2)[245:265,245:265])
axs2[1,3].imshow((pywfs_amplitude_mask[3]*np.abs(focal_plane_centered_complex_amplitude)**2)[245:265,245:265])
axs2[1,4].imshow((np.abs(focal_plane_centered_complex_amplitude)**2)[245:265,245:265])

fig3, axs3 = plt.subplots(nrows=1, ncols=1)
axs3.imshow(np.angle(pywfs_complex_amplitude_mask))

print(np.sum(np.abs(conjugated_pupil_plane_centered_complex_amplitude * pywfs_amplitude_mask[0])**2))
print(np.sum(np.abs(conjugated_pupil_plane_centered_complex_amplitude * pywfs_amplitude_mask[1])**2))
print(np.sum(np.abs(conjugated_pupil_plane_centered_complex_amplitude * pywfs_amplitude_mask[2])**2))
print(np.sum(np.abs(conjugated_pupil_plane_centered_complex_amplitude * pywfs_amplitude_mask[3])**2))
print(np.sum(np.abs(conjugated_pupil_plane_centered_complex_amplitude)**2))


