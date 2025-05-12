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

#%%

path = pathlib.Path(__file__).parent

# %% -----------------------     read parameter file   ----------------------------------

from parameterFile_VLT_I_Band_BIO import initializeParameterFile

from astropy.io import fits

def ca():
    plt.close('all')

param = initializeParameterFile()

# %%

plt.ion()

check_e2e_noise = True

MMSE = True

## R0 assumption for the MMSE REC

r0c=0.15

## Noise level (trust) assumption for the MMSE REC
noise_levelc  = 100e-9

## Terms uncorrelated with atmosphere (not used here)
UC_TERM_AMP = 300e-9*0.

alpha=1.0 # Weight for the turbulence statistics

nmot = 1300 # Number of modes in the inverion
nL = 1200 ## Iterations

#%% -----------------------     TELESCOPE   ----------------------------------

# create the Telescope object
tel = Telescope(resolution          = param['resolution'],\
                diameter            = param['diameter'],\
                samplingTime        = param['samplingTime'],\
                centralObstruction  = param['centralObstruction'])

dim = tel.OPD.shape[0]
pupil = tel.pupil
tpup = np.sum(pupil)
idxpup = np.where(pupil==1)

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

# mis-registrations object
misReg = MisRegistration(param)
# if no coordonates specified, create a cartesian dm
dm=DeformableMirror(telescope    = tel,\
                    nSubap       = 2*param['nSubaperture'],\
                    mechCoupling = param['mechanicalCoupling'],\
                    misReg       = misReg)




#param['postProcessing'] = 'fullFrame_incidence_flux'

param['postProcessing'] = 'slopesMaps_incidence_flux'

bio_20 = BioEdge(nSubap             = param['nSubaperture'],\
              telescope             = tel,\
              modulation            = 0,\
              grey_width            = 2,\
              lightRatio            = param['lightThreshold'],\
              n_pix_separation      = 0,\
              psfCentering          = False,\
              postProcessing        = param['postProcessing'],calibModulation=50)

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

#nmot = M2C_KL_full.shape[1]

M2C_KL = M2C_KL_full[:,1:nmot]

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
                                 noise          = 'off', display = True)

fits.writeto(path / 'RECMO_bio_20_slopesMap.fits', calib_bio_20.D,overwrite=True)



test_wfs = [bio_20]


test_calib = [calib_bio_20]

plt.close('all')



propa = []
for i_wfs in range(1):
    bio = copy.deepcopy(test_wfs[i_wfs])
    calib_CL    = test_calib[i_wfs]
    calib_CL = CalibrationVault(calib_CL.D[:,:nmot-1])
    #plt.figure(10)
    propa.append(np.diag(calib_CL.M@calib_CL.M.T)) # .M = reconstructor ??
    # plt.plot(propa[i_wfs])

# modal projector

dm.coefs = M2C_KL_full
tel.resetOPD()

ngs*tel*dm

KL_s = np.reshape(tel.OPD,[tel.resolution**2, tel.OPD.shape[2]])

KL_basis = np.squeeze(KL_s[tel.pupilLogical,:])

KL_2D = np.reshape(KL_s,[dim,dim, tel.OPD.shape[2]])

projector = np.linalg.pinv(KL_basis)


nn = np.array([nmot-1]) 

plt.close('all')

#pdb.set_trace()

#%% from zoom

atm.initializeAtmosphere(tel)

atm_opd_s = np.reshape(atm.OPD, [dim*dim])
proj = KL_s.T @ atm_opd_s/tpup

cor = KL_s @ proj

res = atm_opd_s - cor

res_2D = np.reshape(res, [dim,dim])

np.std(res_2D[idxpup])

fitting_error_analytical = (0.3*tel.D/dm.nAct / atm.r0)**0.5 **5./3 * 0.5 / (2*np.pi)

#%%

## MMSE RECONSTRUCTOR

# This is the full basis with Piston
B = M2C_KL_full.copy()

HHt, PSD, df = load('/diskb/cverinau/oopao_data/data_calibration/SUPERR/HHt_PSD_df_HHt_SUPERR_nowok.pkl')

## COVARIANCE OF MODES IN ATMOSPHERE
Cmo_B = (1./tpup**2.) * B.T @ HHt @ B *(0.5e-6/(2.*np.pi))**2

## VERIFICATION: RMS ERROR PISTON INCLUDED: Full minus DM component = fitting error
rmsPSD_wiP = np.sqrt(np.sum(PSD*df**2))*0.5e-6/(2.*np.pi) # with piston
rmsDM_wiP = np.sqrt(np.sum(np.diag(Cmo_B[0:,0:])))
## FITTING ERROR FOR ALL MODES CORRECTED
fitting_error=np.sqrt(rmsPSD_wiP**2-rmsDM_wiP**2)


## COVARIANCE PISTON ECLUDEDOF CONTROLLED MODES (PISTON EXCLUDED)
C_phi = np.asmatrix(Cmo_B[1:nmot,1:nmot])*r0c**(-5./3.)
#for k in range(0,5):
#    C_phi[k,k] = C_phi[k,k] + UC_TERM_AMP**2

## COVARIANCE OF NOISE (assumed to be uncorrelated: Diagonal matrix)
nmeas=bio_20.signal.shape[0]
C_n = np.asmatrix(np.zeros([nmeas,nmeas]))
for k in range(0,nmeas):
    C_n[k,k] = noise_levelc**2.

### INTERACTION MATRIX "IN METERS"
REF_LAM = bio_20.telescope.src.wavelength
TO_M0 = REF_LAM/(2. * np.pi)
IM_meter = calib_bio_20.D * TO_M0

M = np.asmatrix(IM_meter.copy())

REC_MMSE = (M.T @ C_n.I @ M + alpha*C_phi.I).I @ M.T @ C_n.I

residuals = np.zeros([param['nLoop'],1])




#pdb.set_trace()

if MMSE==True:
    REC_A = REC_MMSE

if MMSE==False:
    REC_A = np.asmatrix(IM_meter).I






out_perf = []
for i_wfs in range(1):
    tmp = propa[i_wfs]
    n = nn[i_wfs] # 980
    index = np.arange(n)
    #index[300:] = 300+np.argsort(tmp[300:n])
    bio = copy.deepcopy(test_wfs[i_wfs])
    calib_CL    = test_calib[i_wfs]
    calib_CL = CalibrationVault(calib_CL.D[:,index])
    M2C_CL      = M2C_KL[:,index]
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
    param['nLoop'] = nL
    # allocate memory to save data
    SR                      = np.zeros(param['nLoop'])
    total                   = np.zeros(param['nLoop'])
    residual                = np.zeros(param['nLoop'])
    bioSignal               = np.arange(0,bio.nSignal)*0
    # loop parameters
    gainCL                  = 0.7 #0.4
    bio.cam.photonNoise     = True
    display                 = False
    reconstructor = np.asarray(M2C_CL @ REC_A) #M2C_CL@calib_CL.M
    modes_in = []
    modes_out = []
    for i in range(param['nLoop']):
        a=time.time()
        # update phase screens => overwrite tel.OPD and consequently tel.src.phase
        atm.update()
        # save phase variance
        total[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
        # save turbulent phase
        turbPhase = tel.src.phase
        modes_in.append(projector@np.squeeze(tel.OPD[np.where(tel.pupil==1)]))
        # propagate to the bio with the CL commands applied
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
        residuals[i,i_wfs] = residual[i]
        OPD=tel.OPD[np.where(tel.pupil>0)]
        print('Loop'+str(i)+'/'+str(param['nLoop'])+' Turbulence: '+str(total[i])+' -- Residual:' +str(residual[i])+ '\n')
    modes_in = np.asarray(modes_in)
    modes_out = np.asarray(modes_out)
    out_perf.append([modes_in,modes_out, SR])

#%%

plt.figure()
plt.plot(residuals[0:,0],label='20x20 ')
#plt.ylim(0,300)
plt.legend()
plt.show(block=False)

#%%

# fits.writeto(path / pathlib.Path('RES_GRE20x20_KL_MMSE'+str(nmot)+'.fits'), residuals,overwrite='True')

#%%

if MMSE==True:
    fits.writeto(path / pathlib.Path('RES_GRE20x20_MMSE_nmoKL'+str(nmot)+'_r0c_'
                                     +str(r0c)+'_alpha_'+str(alpha)+'_noisec_'
                                     +str(noise_levelc)+'_nLoop'+str(param["nLoop"])
                                     +'_RMS_'+str(np.int64(np.mean(residual[100:param["nLoop"]])))+'nm.fits'), residuals,overwrite='True')

if MMSE==False:
    fits.writeto(path / pathlib.Path('RES_GRE20x20_LS_nmoKL'+str(nmot)+'_nLoop'+str(param["nLoop"])
                                     +'_RMS_'+str(np.int64(np.mean(residual[100:param["nLoop"]])))
                                     +'nm.fits'), residuals,overwrite='True')

# amp = 1.e-9

# tel.OPD = KL_2D[:,:,270]*amp# KL_2D[:,:,300]*amp

# tel*bio

# signal_meter = bio.signal* TO_M0

# r_meter0 = np.array(REC_A  @ signal_meter)[0,:]
# plt_plot(r_meter0)
