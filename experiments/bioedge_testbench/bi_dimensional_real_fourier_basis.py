# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:08:46 2024

@author: fleroux
"""

import numpy as np
import matplotlib.pyplot as plt

def compute_real_fourier_basis(n_px:int, *, return_map = False):
    
    basis = np.empty((n_px, n_px, n_px**2), dtype=float)
    basis.fill(np.nan)
    basis_map = np.empty((n_px, n_px, n_px, n_px//2+1, 2), dtype=float)
    basis_map.fill(np.nan)
    
    index = 0
    
    [X,Y] = np.meshgrid(np.arange(-n_px//2 + 1, n_px//2 + 1), np.flip(np.arange(-n_px//2 + 1, n_px//2 + 1)))
    
    for nu_x in np.arange(-n_px//2 + 1, n_px//2 + 1):
        for nu_y in np.arange(0, n_px//2 + 1):
            
            if (nu_x == 0 and nu_y == 0) or (nu_x == 0 and nu_y == n_px//2)\
            or (nu_x == n_px//2 and nu_y == n_px//2) or (nu_x == n_px//2 and nu_y == 0):
                
                basis[:,:,index] = 1./n_px * np.cos(2.*np.pi/n_px * (nu_x * X + nu_y * Y))
                index += 1
                basis_map[:,:,nu_x, nu_y, 0] = 1./n_px * np.cos(2.*np.pi/n_px * (nu_x * X + nu_y * Y))
                
            elif (nu_y != 0 or nu_x >= 0) and  (nu_y != n_px//2 or nu_x >= 0):
                
                basis[:,:,index] = 2**0.5/n_px * np.cos(2.*np.pi/n_px * (nu_x * X + nu_y * Y))
                basis[:,:,index+1] = 2**0.5/n_px * np.sin(2.*np.pi/n_px * (nu_x * X + nu_y * Y))
                index += 2
                basis_map[:,:,nu_x, nu_y, 0] = 2**0.5/n_px * np.cos(2.*np.pi/n_px * (nu_x * X + nu_y * Y))
                basis_map[:,:,nu_x, nu_y, 1] = 2**0.5/n_px * np.sin(2.*np.pi/n_px * (nu_x * X + nu_y * Y))
    
                
    if return_map:
        return basis_map
    else:
        return basis

def extract_subset(complete_real_fourier_basis, new_n_px):
    return np.roll(complete_real_fourier_basis[:,:,np.arange(-new_n_px//2+1,new_n_px//2+1),0:new_n_px//2+1,:]
                   , -new_n_px//2 + 1, axis = 2)

def basis_map2basis(basis_map):
    
    n_px = basis_map.shape[0]
    basis = np.empty((n_px, n_px, n_px**2))
    
    index = 0
    for nu_x in np.arange(-n_px//2 + 1, n_px//2 + 1):
        for nu_y in np.arange(0, n_px//2 + 1):
            
            if (nu_x == 0 and nu_y == 0) or (nu_x == 0 and nu_y == n_px//2)\
            or (nu_x == n_px//2 and nu_y == n_px//2) or (nu_x == n_px//2 and nu_y == 0):
                
                basis[:,:,index] = basis_map[:,:,nu_x, nu_y, 0]
                index += 1
                
            elif (nu_y != 0 or nu_x >= 0) and  (nu_y != n_px//2 or nu_x >= 0):
                
                basis[:,:,index] = basis_map[:,:,nu_x, nu_y, 0]
                basis[:,:,index+1] = basis_map[:,:,nu_x, nu_y, 1]
                index += 2
    
    return basis




#%%

# n_px = 8

# Check that the computed basis is orthonormal

# basis = compute_real_fourier_basis(n_px)
# basis_columns = basis.reshape((basis.shape[0]*basis.shape[1], basis.shape[2]))

# plt.figure(1)
# plt.imshow(basis_columns[:,0].reshape((n_px,n_px))==basis[:,:,0]) # check reshape function behaviour
# plt.figure(2)
# plt.imshow(basis_columns.T @ basis_columns)


# Plot basis

# basis = compute_real_fourier_basis(n_px)

# figure, axs = plt.subplots(nrows=n_px, ncols=n_px)

# index = 0
# for m in range(n_px):
#     for n in range(n_px):
#         axs[m,n].imshow(basis[:,:,index])
#         index += 1
        
# Plot basis fft2

# basis = compute_real_fourier_basis(n_px)

# figure, axs = plt.subplots(nrows=n_px, ncols=n_px)

# index = 0
# for m in range(n_px):
#     for n in range(n_px):
#         axs[m,n].imshow(np.fft.fftshift(np.abs(np.fft.fft2(basis[:,:,index]))))
#         index += 1
        
# Check basis_map2basis

# basis_map = compute_real_fourier_basis(n_px, return_map=True)
# basis = compute_real_fourier_basis(n_px)
# basis_retrieved = basis_map2basis(basis_map)

# print((basis == basis_retrieved).sum()/(basis == basis_retrieved).size)