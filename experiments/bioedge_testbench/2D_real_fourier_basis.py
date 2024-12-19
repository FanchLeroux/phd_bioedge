# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:08:46 2024

@author: fleroux
"""

import numpy as np
import matplotlib.pyplot as plt

def compute_real_fourier_basis(n_px:int, *, ordering = False):
    
    basis = np.empty((n_px, n_px, n_px**2), dtype=float)
    basis.fill(np.nan)
    basis_2 = np.empty((n_px, n_px, n_px, n_px//2+1, 2), dtype=float)
    basis_2.fill(np.nan)
    
    index = 0
    
    [X,Y] = np.meshgrid(np.arange(-n_px//2 + 1, n_px//2 + 1), np.flip(np.arange(-n_px//2 + 1, n_px//2 + 1)))
    
    for nu_x in np.arange(-n_px//2 + 1, n_px//2 + 1):
        for nu_y in np.arange(0, n_px//2 + 1):
            
            if (nu_x == 0 and nu_y == 0) or (nu_x == 0 and nu_y == n_px//2)\
            or (nu_x == n_px//2 and nu_y == n_px//2) or (nu_x == n_px//2 and nu_y == 0):
                
                basis[:,:,index] = 1./n_px * np.cos(2.*np.pi/n_px * (nu_x * X + nu_y * Y))
                index += 1
                basis_2[:,:,nu_x, nu_y, 0] = 1./n_px * np.cos(2.*np.pi/n_px * (nu_x * X + nu_y * Y))
                
            elif (nu_y != 0 or nu_x >= 0) and  (nu_y != n_px//2 or nu_x >= 0):
                
                basis[:,:,index] = 2**0.5/n_px * np.cos(2.*np.pi/n_px * (nu_x * X + nu_y * Y))
                basis[:,:,index+1] = 2**0.5/n_px * np.sin(2.*np.pi/n_px * (nu_x * X + nu_y * Y))
                index += 2
                basis_2[:,:,nu_x, nu_y, 0] = 2**0.5/n_px * np.cos(2.*np.pi/n_px * (nu_x * X + nu_y * Y))
                basis_2[:,:,nu_x, nu_y, 1] = 2**0.5/n_px * np.sin(2.*np.pi/n_px * (nu_x * X + nu_y * Y))
    
                
    if ordering:
        return basis_2
    else:
        return basis