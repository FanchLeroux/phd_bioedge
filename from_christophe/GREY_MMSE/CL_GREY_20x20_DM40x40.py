# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:51:32 2020

@author: cheritie
"""
import pdb

import pathlib

import time
import copy

from OOPAO.calibration.ao_cockpit_psim import plt_imshow
from OOPAO.calibration.ao_cockpit_psim import plt_plot
from OOPAO.calibration.ao_cockpit_psim import load

import matplotlib.pyplot as plt
import numpy as np

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.BioEdge import BioEdge
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
from OOPAO.Zernike import Zernike
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap

from parameterFile_VLT_I_Band_BIO import initializeParameterFile

from astropy.io import fits

#%%

path = pathlib.Path(__file__).parent

# %% -----------------------     read parameter file   ----------------------------------

param = initializeParameterFile()

# %%

MMSE = True

## R0 assumption for the MMSE REC

r0_guess=0.15

## Noise level (trust) assumption for the MMSE REC
noise_level_guess  = 100e-9

alpha=1.0 # Weight for the turbulence statistics

n_modes_shown_mmse = 1300 # Number of modes in the inverion
n_iter = 1200 ## Iterations

#%% -----------------------     TELESCOPE   ----------------------------------

# create the Telescope object
tel = Telescope(resolution          = param['resolution'],\
                diameter            = param['diameter'],\
                samplingTime        = param['samplingTime'],\
                centralObstruction  = param['centralObstruction'])

pupil = tel.pupil
n_px_pupil = np.sum(tel.pupil)

#%% -----------------------     NGS   ----------------------------------

# create the Source object
ngs=Source(optBand   = param['opticalBand'],\
           magnitude = param['magnitude'])

# combine the NGS to the telescope using '*' operator:
ngs*tel

# create the Atmosphere object
atm=Atmosphere(telescope     = tel,\
               r0            = param['r0'],\
               L0            = param['L0'],\
               windSpeed     = param['windSpeed'],\
               fractionalR0  = param['fractionnalR0'],\
               windDirection = param['windDirection'],\
               altitude      = param['altitude'])
# initialize atmosphere
atm.initializeAtmosphere(tel)
atm.update()

# if no coordonates specified, create a cartesian dm
dm=DeformableMirror(telescope    = tel,\
                    nSubap       = 2*param['nSubaperture'],\
                    mechCoupling = param['mechanicalCoupling'])




#param['postProcessing'] = 'fullFrame_incidence_flux'
#param['postProcessing'] = 'fullFrame' 
param['postProcessing'] = 'slopesMaps_incidence_flux'

bio_20 = BioEdge(nSubap             = param['nSubaperture'],\
              telescope             = tel,\
              modulation            = 0,\
              grey_width            = 2,\
              lightRatio            = param['lightThreshold'],\
              n_pix_separation      = 0,\
              psfCentering          = False,\
              postProcessing        = param['postProcessing'],
              calibModulation=50)

#%% Combute modal basis    

M2C_KL_full = compute_M2C(telescope                  = tel,\
                                  atmosphere         = atm,\
                                  deformableMirror   = dm,\
                                  param              = param,\
                                  nameFolder         = str(path),\
                                  nameFile           = 'BASIS_SUPERR_nowok',\
                                  remove_piston      = False,\
                                  HHtName            = 'HHt_SUPERR_nowok',\
                                  baseName           = 'VLT_nowok' ,\
                                  mem_available      = 6.1e9,\
                                  minimF             = False,\
                                  nmo                = 1350,\
                                  ortho_spm          = True,\
                                  SZ                 = np.int64(2*tel.OPD.shape[0]),\
                                  nZer               = 3,\
                                  NDIVL              = 1,\
                                  lim_inversion=1e-5)

#n_modes_shown_mmse = M2C_KL_full.shape[1]

M2C_KL = M2C_KL_full[:,1:n_modes_shown_mmse]

#%%

stroke = 1e-9

#%% Callibration

calib_bio_20 = InteractionMatrix(ngs            = ngs,\
                                 atm            = atm,\
                                 tel            = tel,\
                                 dm             = dm,\
                                 wfs            = bio_20,\
                                 M2C            = M2C_KL,\
                                 stroke         = stroke,\
                                 nMeasurements  = 1,\
                                 noise          = 'off', 
                                 display = True,
                                 single_pass=False)

fits.writeto(path / 'RECMO_bio_20_slopesMap.fits', calib_bio_20.D,overwrite=True)

uniform_noise_propagation = np.diag(calib_bio_20.M @ calib_bio_20.M.T)

# modal projector
dm.coefs = M2C_KL_full
tel.resetOPD()

ngs*tel*dm

KL_1D = np.reshape(tel.OPD, [tel.resolution**2, tel.OPD.shape[2]])
KL_basis = np.squeeze(KL_1D[tel.pupilLogical,:])
KL_2D = np.reshape(KL_1D,[tel.OPD.shape[0], tel.OPD.shape[0], tel.OPD.shape[2]])

projector = np.linalg.pinv(KL_basis)

nn = np.array([n_modes_shown_mmse-1])

#%%

## MMSE RECONSTRUCTOR

# This is the full basis with Piston
B = M2C_KL_full.copy()

HHt, PSD, df = load('/diskb/cverinau/oopao_data/data_calibration/SUPERR/HHt_PSD_df_HHt_SUPERR_nowok.pkl')

## COVARIANCE OF MODES IN ATMOSPHERE
Cmo_B = (1./n_px_pupil**2.) * B.T @ HHt @ B *(tel.src.wavelength/(2.*np.pi))**2

## VERIFICATION: RMS ERROR PISTON INCLUDED: Full minus DM component = fitting error
rmsPSD_wiP = np.sqrt(np.sum(PSD*df**2))*0.5e-6/(2.*np.pi) # with piston
rmsDM_wiP = np.sqrt(np.sum(np.diag(Cmo_B[0:,0:])))
## FITTING ERROR FOR ALL MODES CORRECTED
fitting_error=np.sqrt(rmsPSD_wiP**2-rmsDM_wiP**2)


## COVARIANCE PISTON ECLUDEDOF CONTROLLED MODES (PISTON EXCLUDED)
C_phi = np.asmatrix(Cmo_B[1:n_modes_shown_mmse,1:n_modes_shown_mmse])*r0_guess**(-5./3.)
#for k in range(0,5):
#    C_phi[k,k] = C_phi[k,k] + UC_TERM_AMP**2

## COVARIANCE OF NOISE (assumed to be uncorrelated: Diagonal matrix)
nmeas=bio_20.signal.shape[0]
C_n = np.asmatrix(np.zeros([nmeas,nmeas]))
for k in range(0,nmeas):
    C_n[k,k] = noise_level_guess**2.

### INTERACTION MATRIX "IN METERS"
REF_LAM = bio_20.telescope.src.wavelength
TO_M0 = REF_LAM/(2. * np.pi)
IM_meter = calib_bio_20.D * TO_M0

M = np.asmatrix(IM_meter.copy())

REC_MMSE = (M.T @ C_n.I @ M + alpha*C_phi.I).I @ M.T @ C_n.I

#pdb.set_trace()

if MMSE==True:
    REC_A = REC_MMSE

if MMSE==False:
    REC_A = np.asmatrix(IM_meter).I






out_perf = []
n = nn[0] # 980
index = np.arange(n)
#index[300:] = 300+np.argsort(uniform_noise_propagation[300:n])
bio = bio_20
atm.generateNewPhaseScreen(10)
tel.resetOPD()
# initialize DM commands
dm.coefs=0
ngs*tel*dm*bio
tel+atm
# dm.coefs[100] = -1
#tel.computePSF(4)
# These are the calibration data used to close the loop
# combine telescope with atmosphere
tel+atm
# initialize DM commands
dm.coefs=0
ngs*tel*dm*bio
#plt.show()
param['n_iter'] = n_iter
# allocate memory to save data
SR                      = np.zeros(param['n_iter'])
total                   = np.zeros(param['n_iter'])
residual                = np.zeros(param['n_iter'])
bioSignal               = np.arange(0,bio.nSignal)*0
# loop parameters
gainCL                  = 0.7 #0.4
bio.cam.photonNoise     = True
display                 = False
reconstructor = np.asarray(M2C_KL @ REC_A) #M2C_KL@calib_bio_20.M
modes_in = []
modes_out = []
for i in range(param['n_iter']):
    a=time.time()
    # update phase screens => overwrite tel.OPD and consequently tel.src.phase
    atm.update()
    # save phase variance
    total[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    # save turbulent phase
    turbPhase = tel.src.phase
    modes_in.append(projector@np.squeeze(tel.OPD[np.where(tel.pupil==1)]))
    # uniform_noise_propagationgate to the bio with the CL commands applied
    tel*dm*bio
    modes_out.append(projector@np.squeeze(tel.OPD[np.where(tel.pupil==1)]))
    #pdb.set_trace()
    dm.coefs=dm.coefs-gainCL*np.matmul(reconstructor,bioSignal)
    # store the slopes after computing the commands => 2 frames delay
    bioSignal=bio.signal*TO_M0
    b= time.time()
    print('Elapsed time: ' + str(b-a) +' s')
    # update displays if required
    if display==True:        
        tel.computePSF(4)
        if i>15:
            SE_PSF.append(np.log10(tel.PSF_norma_zoom))
            LE_PSF = np.mean(SE_PSF, axis=0)
        cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD,bio.cam.frame,[np.arange(i+1),residual[:i+1]],dm.coefs,np.log10(tel.PSF_norma_zoom), LE_PSF],
                               plt_obj = plot_obj)
        plt.pause(0.1)
        if plot_obj.keep_going is False:
            break
    SR[i]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
    residual[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
    OPD=tel.OPD[np.where(tel.pupil>0)]
    print('Loop'+str(i)+'/'+str(param['n_iter'])+' Turbulence: '+str(total[i])+' -- Residual:' +str(residual[i])+ '\n')
modes_in = np.asarray(modes_in)
modes_out = np.asarray(modes_out)
out_perf.append([modes_in,modes_out, SR])

#%%

plt.figure()
plt.plot(residual,label='20x20 ')
plt.plot(total, 'g')
#plt.ylim(0,300)
plt.legend()

# amp = 1.e-9

# tel.OPD = KL_2D[:,:,270]*amp# KL_2D[:,:,300]*amp

# tel*bio

# signal_meter = bio.signal* TO_M0

# r_meter0 = np.array(REC_A  @ signal_meter)[0,:]
# plt_plot(r_meter0)

#%% from zoom

atm.initializeAtmosphere(tel)

atm_opd_s = np.reshape(atm.OPD, [tel.OPD.shape[0]*tel.OPD.shape[0]])
proj = KL_1D.T @ atm_opd_s/n_px_pupil

cor = KL_1D @ proj

res = atm_opd_s - cor

res_2D = np.reshape(res, [tel.OPD.shape[0],tel.OPD.shape[0]])

np.std(res_2D[np.where(pupil==1)])

fitting_error_analytical = (0.3*tel.D/dm.nAct / atm.r0)**0.5 **5./3 * 0.5 / (2*np.pi)

#%%

tel.resetOPD()
dm.coefs = 100e-9*M2C_KL

ngs*tel*dm*bio_20

reconstructed_modes = np.asarray(REC_A @ bio_20.bioSignal)

#%%

n_mode = 250

fig, axs = plt.subplots(nrows=1, ncols=3)
axs[0].imshow(tel.OPD[:,:, n_mode])
axs[1].imshow(bio_20.bioSignal_2D[:,:, n_mode])
axs[2].plot(reconstructed_modes[n_mode,:n_mode+20])

plt.figure()
plt.plot(np.diag(reconstructed_modes)[50:250])

# %% Pseudo inverse

n_modes_to_keep = 200
# reconstructor_LSE = np.asmatrix(calib_bio_20.D[:,:n_modes_to_keep].T @ calib_bio_20.D[:,:n_modes_to_keep]).I\
#     @ calib_bio_20.D[:,:n_modes_to_keep].T

reconstructor_LSE = np.linalg.pinv(calib_bio_20.D[:,:n_modes_to_keep])
    
#%%

n_mode = 2

reconstructed_modes_LSE = np.asarray(reconstructor_LSE @ bio_20.bioSignal)

plt.figure()
plt.plot(reconstructed_modes[n_mode,:n_mode+20])
#plt.plot(np.diag(reconstructed_modes_LSE)[50:250])