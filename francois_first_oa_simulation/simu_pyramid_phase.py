# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:03:43 2024

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

pupil_support_size = 100
pupil_diameter = 1 # [m]
wavelength = 0.5e-6 # [m]
zeros_padding_factor = 4
delta_phi_x = 0*np.pi/(zeros_padding_factor)

delta_phi = 200.0*np.pi

# --------------- CONSEQUENCES ---------------

pupil_plane_pp = pupil_diameter / pupil_support_size # [m]
 
# --------------- MAIN ---------------

# pupil generation
pupil = Disk(pupil_support_size)

# entrance phase generation
tilt_x = Tilt(delta_phi=delta_phi_x, size_support=[pupil_support_size, pupil_support_size], direction='x')
tilt_y = Tilt(delta_phi=-delta_phi_x, size_support=[pupil_support_size, pupil_support_size], direction='y')
pupil_phase = tilt_x + tilt_y
pupil_complex_amplitude = pupil * np.exp(1j*pupil_phase)

# energy normalisation
pupil_complex_amplitude = pupil_complex_amplitude/np.sum(np.abs(pupil_complex_amplitude))**0.5

# zero padding
pupil_complex_amplitude = Zeros_padding(pupil_complex_amplitude, zeros_padding_factor=zeros_padding_factor)
pupil_irradiance = np.abs(pupil_complex_amplitude)**2

# check energy conservation
print(np.round(pupil_irradiance.sum(), 1))

# propagation towards focal plane
focal_plane_complex_amplitude = np.fft.fftshift(np.fft.fft2(pupil_complex_amplitude, norm='ortho'))
focal_plane_irradiance = np.abs(focal_plane_complex_amplitude)**2 

# check energy conservation
print(np.round(focal_plane_irradiance.sum(), 1))


# compute PYWFS phase mask
focal_plane_phase_shift_11 = Tilt(delta_phi=-delta_phi, size_support=focal_plane_complex_amplitude.shape, direction='x') +\
                             Tilt(delta_phi=delta_phi, size_support=focal_plane_complex_amplitude.shape, direction='y')
                          
focal_plane_phase_shift_12 = Tilt(delta_phi=delta_phi, size_support=focal_plane_complex_amplitude.shape, direction='x') +\
                             Tilt(delta_phi=delta_phi, size_support=focal_plane_complex_amplitude.shape, direction='y')
                          
focal_plane_phase_shift_21 = Tilt(delta_phi=-delta_phi, size_support=focal_plane_complex_amplitude.shape, direction='x') +\
                             Tilt(delta_phi=-delta_phi, size_support=focal_plane_complex_amplitude.shape, direction='y')

focal_plane_phase_shift_22 = Tilt(delta_phi=delta_phi, size_support=focal_plane_complex_amplitude.shape, direction='x') +\
                             Tilt(delta_phi=-delta_phi, size_support=focal_plane_complex_amplitude.shape, direction='y')

pywfs_amplitude_mask = AmplitudeScreensPywfs(pupil_support_size*(zeros_padding_factor))                             
pywfs_phase_mask = pywfs_amplitude_mask[0] * focal_plane_phase_shift_11 + pywfs_amplitude_mask[1] * focal_plane_phase_shift_12 + \
                   pywfs_amplitude_mask[2] * focal_plane_phase_shift_21 + pywfs_amplitude_mask[3] * focal_plane_phase_shift_22

# addind phase tilt to complex amplitude in focal plane                        
focal_plane_complex_amplitude = focal_plane_complex_amplitude * np.exp(1j*pywfs_phase_mask)

# propagation towards conjugated pupil plane
conjugated_pupil_plane_complex_amplitude = np.fft.fft2(np.fft.ifftshift(focal_plane_complex_amplitude), norm='ortho')
conjugated_pupil_plane_irradiance = np.abs(conjugated_pupil_plane_complex_amplitude)**2

# check energy conservation
print(np.round(conjugated_pupil_plane_irradiance.sum(), 1))

# plots

fig, axs = plt.subplots(ncols=3, nrows=1)
axs[0].imshow(focal_plane_irradiance)
axs[1].imshow(pywfs_phase_mask)
axs[2].imshow(conjugated_pupil_plane_irradiance)

fig2, axs2 = plt.subplots(ncols=2, nrows=2)
axs2[0,0].imshow(focal_plane_irradiance*pywfs_amplitude_mask[0])
axs2[0,1].imshow(focal_plane_irradiance*pywfs_amplitude_mask[1])
axs2[1,0].imshow(focal_plane_irradiance*pywfs_amplitude_mask[2])
axs2[1,1].imshow(focal_plane_irradiance*pywfs_amplitude_mask[3])








