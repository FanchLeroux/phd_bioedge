# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 20:57:53 2024

@author: fleroux
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.animation as animation

from bi_dimensional_real_fourier_basis import compute_real_fourier_basis, extract_subset, sort_real_fourier_basis

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

n_px = 16
resolution = 2*n_px

basis = compute_real_fourier_basis(resolution, return_map=True)

basis = extract_subset(basis, n_px)

basis = sort_real_fourier_basis(basis)

pupil = get_circular_pupil(basis.shape[0])

zeros_padding_factor = 4

complex_amplitude_screen = np.empty((zeros_padding_factor*basis.shape[0], zeros_padding_factor*basis.shape[1], basis.shape[2]), dtype=complex)

for k in range(complex_amplitude_screen.shape[2]):
    complex_amplitude_screen[:,:,k] = zeros_padding(pupil*np.exp(1j*2*np.pi*basis[:,:,k]), zeros_padding_factor)

#%%

fig, ax = plt.subplots()

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(basis.shape[2]):
    im = ax.imshow(np.abs(np.fft.fftshift(np.fft.fft2(complex_amplitude_screen[:,:,i]))))
    if i == 0:
        #ax.imshow(basis[:,:,i])  # show an initial one first
        ax.imshow(np.abs(np.fft.fftshift(np.fft.fft2(complex_amplitude_screen[:,:,i])))**2)
    ims.append([im])

ax.set_title('2pi rad RMS')

ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                repeat_delay=1000)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)



plt.show()

ani.save("2D_fourier_basis_focal_plane_movie.mp4")