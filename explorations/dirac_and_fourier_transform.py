# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 16:44:11 2024

@author: fleroux
"""

import numpy as np
import matplotlib.pyplot as plt

size = 256
delta_coordinate = 10


arr = np.zeros((size,size))
arr[arr.shape[0]//2, arr.shape[1]//2 + delta_coordinate] = 1

arr_tilde = np.fft.fftshift(np.fft.fft2(arr))

fig, axs = plt.subplots(nrows=1,ncols=2)
fig 