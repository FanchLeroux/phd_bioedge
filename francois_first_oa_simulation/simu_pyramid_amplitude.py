# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:23:42 2024

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
delta_phi_x = np.pi/(zeros_padding_factor)

# --------------- CONSEQUENCES ---------------

pupil_plane_pp = pupil_diameter / pupil_support_size # [m]
 
# --------------- MAIN ---------------

# pupil generation
pupil = Disk(pupil_support_size)

# entrance phase generation
tilt_x = Tilt(delta_phi=-delta_phi_x, size_support=[pupil_support_size, pupil_support_size], direction='x')
tilt_y = Tilt(delta_phi=delta_phi_x, size_support=[pupil_support_size, pupil_support_size], direction='y')
pupil_phase = tilt_x + tilt_y
pupil_complex_amplitude = pupil * np.exp(1j*pupil_phase)

# energy normalisation
pupil_complex_amplitude = pupil_complex_amplitude/np.sum(np.abs(pupil_complex_amplitude))**0.5

# zero padding
pupil_complex_amplitude = Zeros_padding(pupil_complex_amplitude, zeros_padding_factor=zeros_padding_factor)
pupil_irradiance = np.abs(pupil_complex_amplitude)**2

# propagation towards focal plane
focal_plane_complex_amplitude = np.fft.fftshift(np.fft.fft2(pupil_complex_amplitude, norm='ortho'))
focal_plane_irradiance = np.abs(focal_plane_complex_amplitude)**2

# propagation towards conjugated pupil plane
conjugated_pupil_plane_complex_amplitude = np.fft.fft2(np.fft.ifftshift(focal_plane_complex_amplitude), norm='ortho') # fft2 or ifft2, that is the question
conjugated_pupil_plane_irradiance = np.abs(conjugated_pupil_plane_complex_amplitude)**2

# check energy conservation
print(np.round(pupil_irradiance.sum(), 1))
print(np.round(focal_plane_irradiance.sum(), 1))
print(np.round(conjugated_pupil_plane_irradiance.sum(), 1))

# pyramid simulation

pyramid_amplitude_screens = AmplitudeScreensPywfs(pupil_support_size*(zeros_padding_factor))
pyramid_quadrants_complex_amplitude = []
for k in range(len(pyramid_amplitude_screens)):
    quadrant = np.fft.fft2(np.fft.ifftshift(focal_plane_complex_amplitude*pyramid_amplitude_screens[k]), norm='ortho')
    pyramid_quadrants_complex_amplitude.append(quadrant)
    
pyramid_quadrants_irradiance = np.abs(np.array(pyramid_quadrants_complex_amplitude))**2

# check energy conservation
print(np.round(pyramid_quadrants_irradiance.sum(), 1))

# check energy repartition over the pyramid quadrants
print("\n"+str(np.round(pyramid_quadrants_irradiance.sum(axis=(1,2)), 2)))

# plots
fig, axs = plt.subplots(nrows=1, ncols=4)

axs[0].imshow(pupil_irradiance)
axs[1].imshow(pupil*pupil_phase)
axs[1].set_title("pupil phase")
axs[2].imshow(focal_plane_irradiance)
axs[3].imshow(conjugated_pupil_plane_irradiance)

fig2, axs2 = plt.subplots(nrows=1, ncols=4)

axs2[0].imshow(pyramid_amplitude_screens[0])
axs2[1].imshow(pyramid_amplitude_screens[1])
axs2[2].imshow(pyramid_amplitude_screens[2])
axs2[3].imshow(pyramid_amplitude_screens[3])

fig3, axs3 = plt.subplots(nrows=1, ncols=4)

axs3[0].imshow(pyramid_quadrants_irradiance[0])
axs3[1].imshow(pyramid_quadrants_irradiance[1])
axs3[2].imshow(pyramid_quadrants_irradiance[2])
axs3[3].imshow(pyramid_quadrants_irradiance[3])

fig2, axs2 = plt.subplots(ncols=2, nrows=2)
axs2[0,0].imshow(focal_plane_irradiance*pyramid_amplitude_screens[0])
axs2[0,1].imshow(focal_plane_irradiance*pyramid_amplitude_screens[1])
axs2[1,0].imshow(focal_plane_irradiance*pyramid_amplitude_screens[2])
axs2[1,1].imshow(focal_plane_irradiance*pyramid_amplitude_screens[3])