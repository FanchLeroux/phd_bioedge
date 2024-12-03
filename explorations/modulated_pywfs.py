# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:27:38 2024

@author: fleroux
"""

import matplotlib.pyplot as plt
import numpy as np

def get_circular_pupil(npx):
    D = npx + 1
    x = np.linspace(-npx/2,npx/2,npx)
    xx,yy = np.meshgrid(x,x)
    circle = xx**2+yy**2
    pupil  = circle<(D/2)**2
    
    return pupil

def zeros_padding(array, zeros_padding_factor):
    array = np.pad(array, (((zeros_padding_factor-1)*array.shape[0]//2, 
                                     (zeros_padding_factor-1)*array.shape[0]//2),
                                    ((zeros_padding_factor-1)*array.shape[1]//2, 
                                     (zeros_padding_factor-1)*array.shape[1]//2)))
    return array

def get_tilt(shape, theta=0., amplitude=1.):
    [X,Y] = np.meshgrid(np.arange(0, shape[0]), np.arange(0, shape[1]))
    tilt_theta = np.cos(theta) * X + np.sin(theta) * Y
    Y = np.flip(Y, axis=0) # change orientation
    tilt_theta = (tilt_theta - tilt_theta.min())/(tilt_theta.max()- tilt_theta.min())
    
    return amplitude*tilt_theta

def get_4pywfs_phase_mask(shape, amplitude):
    
    mask = np.empty(shape, dtype=float)
    [X,Y] = np.meshgrid(np.arange(0, shape[0]), np.arange(0, shape[1]))
    Y = np.flip(Y, axis=0) # change orientation
    
    theta = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
    mask[:mask.shape[0]//2, mask.shape[1]//2:] = (np.cos(theta[0]) * X + np.sin(theta[0]) * Y)[:mask.shape[0]//2, mask.shape[1]//2:]
    mask[:mask.shape[0]//2, :mask.shape[1]//2] = (np.cos(theta[1]) * X + np.sin(theta[1]) * Y)[:mask.shape[0]//2, :mask.shape[1]//2]
    mask[mask.shape[0]//2:, :mask.shape[1]//2] = (np.cos(theta[2]) * X + np.sin(theta[2]) * Y)[mask.shape[0]//2:, mask.shape[1]//2:]
    mask[mask.shape[0]//2:, mask.shape[1]//2:] = (np.cos(theta[3]) * X + np.sin(theta[3]) * Y)[:mask.shape[0]//2, mask.shape[1]//2:]
    
    mask[:mask.shape[0]//2, mask.shape[1]//2:] = (mask[:mask.shape[0]//2, mask.shape[1]//2:]-mask[:mask.shape[0]//2, mask.shape[1]//2:].min())/(mask[:mask.shape[0]//2, mask.shape[1]//2:].max()-mask[:mask.shape[0]//2, mask.shape[1]//2:].min())
    mask[:mask.shape[0]//2, :mask.shape[1]//2] = (mask[:mask.shape[0]//2, :mask.shape[1]//2]-mask[:mask.shape[0]//2, :mask.shape[1]//2].min())/(mask[:mask.shape[0]//2, :mask.shape[1]//2].max()-mask[:mask.shape[0]//2, :mask.shape[1]//2].min())
    mask[mask.shape[0]//2:, :mask.shape[1]//2] = (mask[mask.shape[0]//2:, :mask.shape[1]//2]-mask[mask.shape[0]//2:, :mask.shape[1]//2].min())/(mask[mask.shape[0]//2:, :mask.shape[1]//2].max()-mask[mask.shape[0]//2:, :mask.shape[1]//2].min())
    mask[mask.shape[0]//2:, mask.shape[1]//2:] = (mask[mask.shape[0]//2:, mask.shape[1]//2:]-mask[mask.shape[0]//2:, mask.shape[1]//2:].min())/(mask[mask.shape[0]//2:, mask.shape[1]//2:].max()-mask[mask.shape[0]//2:, mask.shape[1]//2:].min())

    return amplitude*mask

def get_focal_plane_image(complex_amplitude):
    return np.abs(np.fft.fftshift(np.fft.fft2(complex_amplitude)))**2

def propagate_fourier_filtering(complex_amplitude, mask_complex_amplitude):
    focal_plane_complex_amplitude = np.fft.fftshift(np.fft.fft2(complex_amplitude))*mask_complex_amplitude
    return np.abs(np.fft.fft2(np.fft.ifftshift(focal_plane_complex_amplitude)))**2
    

#%% -------------- PARAMETERS ---------------------

# telescope
npx = 256

# modulation
modulation_radius = 3. # pixels/zeros_padding_factor [1/D m^-1]
n_modulation_points = 8

# focal plane sampling
zeros_padding_factor = 3

#%% ---------------- TELESCOPE --------------------

pupil = get_circular_pupil(npx)

#%% ---------------- MODULATOR --------------------

theta_list = np.arange(0., 2.*np.pi, 2.*np.pi/n_modulation_points)
tilt_amplitude = 2.*np.pi * modulation_radius

# modulation phase screens generation - Units : phase
modulation_phase_screens = np.empty((pupil.shape[0],
                                     pupil.shape[1],
                                     n_modulation_points), dtype=float)
[X,Y] = np.meshgrid(np.arange(0, pupil.shape[0]), np.arange(0, pupil.shape[0]))
Y = np.flip(Y, axis=0) # change orientation

for k in range(n_modulation_points):

    theta = theta_list[k]    

    tilt_theta = np.cos(theta) * X + np.sin(theta) * Y

    # pupil addition and normalization
    
    tilt_theta[pupil>0] = (tilt_theta[pupil>0] - tilt_theta[pupil>0].min())\
        /(tilt_theta[pupil>0].max()- tilt_theta[pupil>0].min())
    tilt_theta[pupil==0] = 0.
    
    modulation_phase_screens[:,:,k] = tilt_amplitude*tilt_theta
    
# %% ------------------ FOCAL PLANE MASK -----------

pywfs_tilt_amplitude = npx * zeros_padding_factor * 0.5 * np.pi 
mask = get_4pywfs_phase_mask(2*[npx*zeros_padding_factor], pywfs_tilt_amplitude)

# %% --------------------- PHASOR ------------------

pupil_pad = zeros_padding(pupil, zeros_padding_factor)

phasor = get_tilt(2*[npx], theta=1.25*np.pi) * pupil
phasor[pupil!=0] = (phasor[pupil!=0] - phasor[pupil!=0].min())/(phasor[pupil!=0].max()-phasor[pupil!=0].min()) # normalization
phasor = 1/zeros_padding_factor * 2**0.5 * np.pi * phasor

# %% ------------------ PROPAGATION ------------------

pupil_pad = zeros_padding(pupil*np.exp(1j*phasor), zeros_padding_factor)

focal_plane_detector = get_focal_plane_image(pupil_pad)

mask_complex_amplitude = np.exp(1j*mask)

pywfs_detector = propagate_fourier_filtering(pupil_pad, mask_complex_amplitude)