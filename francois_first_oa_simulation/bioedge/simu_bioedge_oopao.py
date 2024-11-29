# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 14:12:09 2024

@author: fleroux
"""

import numpy as np
import matplotlib.pyplot as plt

#%% ----------- PARAMETERS --------------

# telescope
tel_resolution = 16 # [px]
tel_diameter = 1 # [m]

# source
src_magnitude = 0
src_band = 'R'

# optical propagation
zero_padding_factor = 1

#%% ----------- TELESCOPE ---------------

from OOPAO.Telescope import Telescope

tel = Telescope(tel_resolution, tel_diameter)

#%% ------------ SOURCE -----------------

from OOPAO.Source import Source

src = Source(src_band, src_magnitude)

#%% ------ Light propagation ------------

src * tel
tel.computePSF(zeroPaddingFactor=zero_padding_factor)

#%% ------------- Plots ---------------

plt.figure(1)
plt.imshow(tel.pupil)

plt.figure(2)
plt.imshow(tel.PSF)