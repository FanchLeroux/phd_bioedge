# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:20:15 2024

@author: fleroux
"""

import numpy as np
import matplotlib.pyplot as plt

r = 3.
d = np.linspace(0., r, 1000)
alpha = np.linspace(0., 45., 5) # [°]
alpha = alpha 


plt.figure(1)

for alpha in alpha:
    alpha = np.deg2rad(alpha)
    sensitivity = 4 * (np.arcsin(d*np.sin(alpha)/r) + np.arcsin(d*np.cos(alpha)/r))
    sensitivity = sensitivity / (2*np.pi)
    plt.plot(d, sensitivity, label='alpha = '+str(int(np.rad2deg(alpha)))+'°')
    
plt.plot(d, d/r, 'r', label='d/r')
plt.xlabel('\nd : distance core - speckle, i.e spatial frequency')
plt.ylabel('duty cycle over one modulation period\n')
plt.legend()
plt.title('modulation radius : r = '+str(int(r))+'\n')