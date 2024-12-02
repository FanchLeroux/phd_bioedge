# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:31:35 2024

@author: fleroux
"""

import matplotlib.pyplot as plt
import numpy as np

# number of subaperture for the WFS
n_subaperture = 60
modulation = 3.
optBand = 'V'

#%% -----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope

# create the Telescope object
tel = Telescope(resolution           = 6*n_subaperture,                          # resolution of the telescope in [pix]
                diameter             = 8,                                        # diameter in [m]        
                samplingTime         = 1/1000,                                   # Sampling time in [s] of the AO loop
                centralObstruction   = 0.,                                       # Central obstruction in [%] of a diameter 
                display_optical_path = False,                                    # Flag to display optical path
                fov                  = 0.)                                       # field of view in [arcsec]. If set to 0 (default) this speeds up the computation of the phase screens but is uncompatible with off-axis targets

#%% -----------------------     NGS   ----------------------------------
from OOPAO.Source import Source

# create the Natural Guide Star object
ngs = Source(optBand     = optBand,           # Optical band (see photometry.py)
             magnitude   = 0,             # Source Magnitude
             coordinates = [0,0])         # Source coordinated [arcsec,deg]

# combine the NGS to the telescope using '*'
ngs*tel

#%% -----------------------     PYRAMID WFS   ----------------------------------
from OOPAO.Pyramid import Pyramid

# make sure tel and atm are separated to initialize the PWFS
tel.isPaired = False
tel.resetOPD()

wfs = Pyramid(nSubap            = n_subaperture,                # number of subaperture = number of pixel accros the pupil diameter
              telescope         = tel,                          # telescope object
              lightRatio        = 0.5,                          # flux threshold to select valid sub-subaperture
              modulation        = modulation,                   # Tip tilt modulation radius
              binning           = 1,                            # binning factor (applied only on the )
              n_pix_separation  = 4,                            # number of pixel separating the different pupils
              n_pix_edge        = 2,                            # number of pixel on the edges of the pupils
              postProcessing    = 'slopesMaps_incidence_flux')  # slopesMaps,

# propagate the light to the Wave-Front Sensor
tel*wfs

oopao_frame = wfs.cam.frame

#%% -------------------------- Fourier modes generation ------------------------

from OOPAO.tools.tools import compute_fourier_mode, zero_pad_array

#wavelength = ngs.photometry(optBand)[0]

angle_deg = 45.0
spatial_frequency = 5. # cycle/pupil
fourier_mode_amplitude = 50e-9 # [m] standard deviation (or variance ?)
fourier_mode = fourier_mode_amplitude*compute_fourier_mode(tel.pupil, spatial_frequency, angle_deg)

#%% --------------------------- Induce phase perturbation ----------------------

tel.OPD = tel.pupil * fourier_mode

#%% --------------------------------- PSF computation --------------------------

tel.computePSF(zeroPaddingFactor=4)

#%% ------------------------------ WFS frame computation -----------------------

tel*wfs

#%% --------------------------- Custom modulation simulation ------------------------

zeros_padding_factor = 2

#modulation parameters
modulation_radius = 3.
n_modulation_points = 5

# therefore
theta_list = np.arange(0., 2.*np.pi, 2.*np.pi/n_modulation_points)
tilt_amplitude = 2.*np.pi * modulation_radius

# modulation phase screens generation - Units : phase
modulation_phase_screens = np.empty((tel.pupil.shape[0],
                                     tel.pupil.shape[1],
                                     n_modulation_points), dtype=complex)
[X,Y] = np.meshgrid(np.arange(0, tel.pupil.shape[0]), np.arange(0, tel.pupil.shape[0]))
Y = np.flip(Y, axis=0) # change orientation

for k in range(n_modulation_points):

    theta = theta_list[k]    

    tilt_theta = np.cos(theta) * X + np.sin(theta) * Y

    # pupil addition and normalization
    
    tilt_theta[tel.pupil>0] = (tilt_theta[tel.pupil>0] - tilt_theta[tel.pupil>0].min())\
        /(tilt_theta[tel.pupil>0].max()- tilt_theta[tel.pupil>0].min())
    tilt_theta[tel.pupil==0] = 0.
    
    modulation_phase_screens[:,:,k] = tilt_amplitude*tilt_theta

#%% Modulated focal plane

detector_focal_plane = np.zeros((zeros_padding_factor*tel.pupil.shape[0],zeros_padding_factor*tel.pupil.shape[1]), dtype=float)

for k in range(n_modulation_points):
    
    complex_amplitude = tel.pupil * np.exp(1j*modulation_phase_screens[:,:,k])

    complex_amplitude_pad = np.pad(complex_amplitude, (((zeros_padding_factor-1)*complex_amplitude.shape[0]//2, 
                                     (zeros_padding_factor-1)*complex_amplitude.shape[0]//2),
                                    ((zeros_padding_factor-1)*complex_amplitude.shape[1]//2, 
                                     (zeros_padding_factor-1)*complex_amplitude.shape[1]//2)))

    detector_focal_plane = detector_focal_plane + np.abs(np.fft.fftshift(np.fft.fft2(complex_amplitude_pad)))**2

#%% OOPAO PYWFS detector

detector_pywfs = np.zeros(wfs.cam.frame.shape, dtype=float)

for k in range(n_modulation_points):
    
    # modulation phase screen is defined in phase [rad], not in opd [m]
    tel.OPD = tel.pupil * modulation_phase_screens[:,:,k] * ngs.photometry(optBand)[0]/(2.*np.pi)
    tel*wfs
    detector_pywfs = detector_pywfs + wfs.cam.frame


#%% Custom PYWFS detector

def get_4pywfs_phase_mask(shape):
    
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

    return mask

#%% Plots

plt.figure(1)
plt.imshow(oopao_frame)
plt.title("WFS Camera Frame - Flat wavefront\nOOPAO modulation ="+str(int(modulation)) + " lambda/D")

plt.figure(2)
plt.imshow(detector_focal_plane)

plt.figure(3)
plt.imshow(detector_pywfs)
plt.title("WFS Camera Frame - Flat wavefront\nFanch modulation ="+str(int(modulation_radius)) + " lambda/D")