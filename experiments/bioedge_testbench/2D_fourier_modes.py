# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:28:11 2024

@author: fleroux
"""

import numpy as np
import matplotlib.pyplot as plt

def zeros_padding(array, zeros_padding_factor):
    array = np.pad(array, (((zeros_padding_factor-1)*array.shape[0]//2, 
                                     (zeros_padding_factor-1)*array.shape[0]//2),
                                    ((zeros_padding_factor-1)*array.shape[1]//2, 
                                     (zeros_padding_factor-1)*array.shape[1]//2)))
    return array

def compute_fourier_mode_x(shape, n_cycle):
    [X,_] = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    X = np.sin(2*np.pi * n_cycle/shape[0] * X)
    X = X/X.std()
    return X

#%%

shape = 32*np.ones(2, dtype=int)
zeros_padding_factor = 1
fx = 0
fy = 15

spectrum =  np.zeros(shape, dtype=float)

spectrum[fx + spectrum.shape[0]//2, fy + spectrum.shape[0]//2] = 1. # null frequency : spectrum[spectrum.shape[0]//2, spectrum.shape[0]//2]

plt.figure()
plt.imshow(spectrum)

spectrum = np.fft.ifftshift(spectrum)

frequency = np.real(np.fft.fft2(spectrum))

frequency_imag = np.imag(np.fft.fft2(spectrum))



plt.figure()
plt.imshow(frequency)

#%%

fx_2 = -1
fy_2 = 0

spectrum_2 =  np.zeros(shape, dtype=float)

spectrum_2[fx_2 + spectrum_2.shape[0]//2, fy_2 + spectrum_2.shape[0]//2] = 1. # null frequency_2 : spectrum_2[spectrum_2.shape[0]//2, spectrum_2.shape[0]//2]

plt.figure()
plt.imshow(spectrum_2)

spectrum_2 = np.fft.ifftshift(spectrum_2)

frequency_2 = np.real(np.fft.fft2(spectrum_2))

frequency_2_imag = np.imag(np.fft.fft2(spectrum_2))

plt.figure()
plt.imshow(frequency_2)

#%%

# spectrum = zeros_padding(spectrum, 2)

spectrum = np.pad(spectrum, ((0,shape[0]*(zeros_padding_factor//2)), (0,shape[1]*(zeros_padding_factor//2))))

frequency_real = np.real(np.fft.fft2(spectrum))
frequency_real = frequency_real[:shape[0], :shape[1]]
#frequency_real = frequency_real/frequency_real.std()
frequency_imag = np.imag(np.fft.fft2(spectrum))
frequency_imag = frequency_imag[:shape[0], :shape[1]]
#frequency_imag = frequency_imag/frequency_imag.std()



from OOPAO.tools.tools import compute_fourier_mode

#oopao_shit = compute_fourier_mode(np.zeros(shape), n_cyles-1, angle_deg=90)

plt.figure()
plt.imshow(frequency_real)

plt.figure()
plt.imshow(frequency_imag)

plt.figure()
plt.plot(frequency_imag[0,:])

# plt.figure()
# plt.imshow(oopao_shit)

# plt.figure()
# img1=plt.imshow(np.abs(oopao_shit-frequency_real))
# plt.colorbar(img1)




