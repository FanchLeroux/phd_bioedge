# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:40:53 2024

@author: fleroux
"""

import numpy as np

def PropagateComplexAmplitude(complex_amplitude):
    return np.fft.fftshift(np.fft.fft2(complex_amplitude))