# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:18:24 2024

@author: fleroux
"""

import numpy as np

def Zeros_padding(array, zeros_padding_factor=2):
    zeros_padded_array = np.pad(array, np.array(array.shape)//2 * (zeros_padding_factor-1))
    return zeros_padded_array