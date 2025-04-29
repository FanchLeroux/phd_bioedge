# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np
import matplotlib.pyplot as plt

n = 512 # [pixel] side length

# cartesian coordinates. Convention for even support : [-2,-1,0,1] (n=4)
[X,Y] = np.meshgrid(np.arange(-n//2, n//2), np.arange(n//2-1, -n//2-1, -1))

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(X)
axs[1].imshow(Y)

radial_coordinate = (X**2 + Y **2)**0.5

pupil_diameter = n

pupil = radial_coordinate <= n/2

plt.figure()
plt.imshow(pupil)

zeros_padding_factor = 4 # number of pixels in psf fwhm

pupil_zeros_padded = np.pad(pupil, ((n//2 * (zeros_padding_factor-1),n//2 * (zeros_padding_factor-1)),))

psf = np.abs(np.fft.fftshift(np.fft.fft2(pupil_zeros_padded))) ** 2

fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(pupil_zeros_padded)
axs[1].imshow(np.abs(psf))

fto = np.fft.fftshift(np.fft.fft2(psf))
ftm = np.abs(fto)
ftm = ftm/ftm.max()

plt.figure()
plt.imshow(ftm)

plt.figure()
plt.plot(np.arange(-zeros_padding_factor*n //2 ,zeros_padding_factor*n //2 ), ftm[ftm.shape[0]//2, :], label='D = '+str(n) + ' pixels')
plt.xlabel('units = pixels')
plt.title('Module of the optical transfert function')
plt.legend()