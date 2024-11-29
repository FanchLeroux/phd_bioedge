# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:31:34 2024

@author: fleroux
"""

import numpy as np

def AmplitudeScreensPywfs(support_size):
    """
    AmplitudeScreensPywfs : generate 4 amplitude screens for PYWFS simulation

    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.10.08, Marseille
    Comments : For even support size, coordinates are defined like [-2,-1,0,1] (N = 4)
               (i.e the origin is the top right pixel of the four central ones)

    Inputs : MANDATORY : support_size {int}[px] : support side length in pixels
                         delta_phi_center {float}[rad] : phase shift aplied at the roof center
    Outputs : phase_screen 
    """
    amplitude_screens = [np.zeros((support_size,support_size), dtype='float'),
                         np.zeros((support_size,support_size)),
                         np.zeros((support_size,support_size)),
                         np.zeros((support_size,support_size))]
    
    amplitude_screens[0][:support_size//2, :support_size//2] = 1
    amplitude_screens[1][:support_size//2, support_size//2:support_size] = 1
    amplitude_screens[2][support_size//2:support_size, :support_size//2] = 1
    amplitude_screens[3][support_size//2:support_size, support_size//2:support_size] = 1
    
    return amplitude_screens