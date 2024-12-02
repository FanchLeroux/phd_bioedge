# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 14:31:35 2024

@author: fleroux
"""

import matplotlib.pyplot as plt
import numpy as np

# number of subaperture for the WFS
n_subaperture = 60
modulation = 10
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

plt.figure(1)
plt.close('all')
plt.figure()
plt.imshow(wfs.cam.frame)
plt.title("WFS Camera Frame - Flat wavefront\nmodulation ="+str(int(modulation)) + " lambda/D")

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

[X,Y] = np.meshgrid(np.arange(0, tel.pupil.shape[0]), np.arange(0, tel.pupil.shape[0]))
Y = np.flip(Y, axis=0) # chage orientation

theta = 0.*np.pi

opd_tilt_theta = np.cos(theta) * X + np.sin(theta) * Y

opd_tilt_theta_normalized = opd_tilt_theta/opd_tilt_theta.max()

#%%

tilt_amplitude = 2.*np.pi

plt.figure(1)
plt.imshow(opd_tilt_theta_normalized)

plt.figure(2)
plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j*tilt_amplitude*opd_tilt_theta_normalized)))))

plt.figure(3)
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j*tilt_amplitude*opd_tilt_theta_normalized))))+1.))

#%%

zeros_padding_factor = 3

complex_amplitude = tel.pupil * np.exp(1j*tilt_amplitude*opd_tilt_theta_normalized)

complex_amplitude_pad = np.pad(complex_amplitude, (((zeros_padding_factor-1)*complex_amplitude.shape[0]//2, 
                                 (zeros_padding_factor-1)*complex_amplitude.shape[0]//2),
                                ((zeros_padding_factor-1)*complex_amplitude.shape[1]//2, 
                                 (zeros_padding_factor-1)*complex_amplitude.shape[1]//2)))

plt.figure(1)
plt.imshow(np.abs(complex_amplitude_pad))

#%% ----------------------------- Show many Fourier Modes ----------------------

from OOPAO.calibration.InteractionMatrix import InteractionMatrix

