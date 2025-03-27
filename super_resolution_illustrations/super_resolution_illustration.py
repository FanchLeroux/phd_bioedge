# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:17:04 2025

@author: fleroux
"""

import matplotlib.pyplot as plt
import numpy as np

def sinc(x):
    
    if x == 0.:
        return 1.
    else:
        return np.sin(np.pi * x) / (np.pi * x)

def sinc_2D(array):
    
    # coordinate convention : [-2, -1, 0, 1]
    [X,Y] = np.meshgrid(np.arange(-array.shape[1]//2, array.shape[1]//2), 
                        np.arange(array.shape[0]//2-1, -array.shape[0]//2-1, -1))
    
    return sinc(X) * sinc(Y)
    