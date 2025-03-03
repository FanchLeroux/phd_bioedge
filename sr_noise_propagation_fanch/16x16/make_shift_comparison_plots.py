# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 18:12:12 2025

@author: fleroux
"""

import platform
import pathlib

from copy import deepcopy

import dill

#%%

path_1_over_2 = pathlib.Path(__file__).parent / '1_over_2_shifts' / 'data_analysis'
path_1_over_4 = pathlib.Path(__file__).parent / '1_over_4_shifts' / 'data_analysis'

filename = 'analysis_sr_noise_prop_I2_band_16x16_KL_basis.pkl'

dill.load_session(path_1_over_2 / filename)

#%%
dill.load_session(path_1_over_4 / filename)
