# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:06:04 2024

@author: fleroux
"""

import numpy as np

def PhaseScreenRoof(support_size, delta_phi_center):
    """
    PhaseScreenRoof : generate a phase screen corresponding to a rooftop (sort of 1D PYWFS)

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
    phase_screen=np.empty((support_size,support_size))
    phase_screen[:,:support_size//2] = Tilt(delta_phi_center, size_support=[support_size//2, support_size])
    
    return phase_screen

def Tilt(delta_phi, size_support, *, direction='x'):
    
    """
    Tilt : generate a phase screen correspnding to a tilt in x
    
    Author : Francois Leroux
    Contact : francois.leroux.pro@gmail.com
    Status : in progress
    Last update : 2024.03.04, Brest
    Comments : For even support size, coordinates are defined like [-2,-1,0,1] (N = 4)
      
    Inputs : MANDATORY : delta_phi [rad] : absolute phase difference between the edges of the phase screen
    
              OPTIONAL :  wavelenght {float}[m] : wavelenght - default value: 0.5 µm
                          size_support {tupple (1x2)}[pixel] : resolution of the support - default value: [128, 128]
                          n_levels {int} : number of levels over which the phase needs to be quantified. 
                                           default value: 0, no Discretization
                        
    Outputs : phase, values between -pi and pi
    """

    [X, Y] = np.meshgrid(np.arange(0,size_support[0]), np.arange(0,size_support[1]))
    
    if direction == 'x':
        X = np.asarray(X, dtype=np.float32)    
        X /= np.float32(size_support[0])
        phase = X * delta_phi
    
    elif direction == 'y':
        Y = np.asarray(Y, dtype=np.float32)    
        Y /= np.float32(size_support[0])
        phase = - Y * delta_phi    
    
    return phase
















