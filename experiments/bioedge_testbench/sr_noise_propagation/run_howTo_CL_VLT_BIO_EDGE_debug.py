# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:58:26 2025

@author: cheritier
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from OOPAO.Atmosphere import Atmosphere
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
from OOPAO.BioEdge import BioEdge
from OOPAO.Source import Source
from OOPAO.Telescope import Telescope
from OOPAO.Zernike import Zernike
from OOPAO.calibration.CalibrationVault import CalibrationVault
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap

# %% -----------------------     read parameter file   ----------------------------------
from parameterFile_VLT_I_Band_BIO import initializeParameterFile

param = initializeParameterFile()

# %%
plt.ion()

#%% -----------------------     TELESCOPE   ----------------------------------

# create the Telescope object
tel = Telescope(resolution          = param['resolution'],\
                diameter            = param['diameter'],\
                samplingTime        = param['samplingTime'],\
                centralObstruction  = param['centralObstruction'])

#%% -----------------------     NGS   ----------------------------------
# create the Source object
ngs=Source(optBand   = param['opticalBand'],\
           magnitude = param['magnitude'])

# combine the NGS to the telescope using '*' operator:
ngs*tel

tel.computePSF(zeroPaddingFactor = 6)
plt.figure()
plt.imshow(np.log10(np.abs(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,3])
plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()

#%% -----------------------     ATMOSPHERE   ----------------------------------

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

plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()


tel+atm
tel.computePSF(8)
plt.figure()
plt.imshow((np.log10(tel.PSF)),extent = [tel.xPSF_arcsec[0],tel.xPSF_arcsec[1],tel.xPSF_arcsec[0],tel.xPSF_arcsec[1]])
plt.clim([-1,3])

plt.xlabel('[Arcsec]')
plt.ylabel('[Arcsec]')
plt.colorbar()

#%% -----------------------     DEFORMABLE MIRROR   ----------------------------------
# mis-registrations object
misReg = MisRegistration(param)
# if no coordonates specified, create a cartesian dm
dm=DeformableMirror(telescope    = tel,\
                    nSubap       = 2*param['nSubaperture'],\
                    mechCoupling = param['mechanicalCoupling'],\
                    misReg       = misReg)

plt.figure()
plt.plot(dm.coordinates[:,0],dm.coordinates[:,1],'x')
plt.xlabel('[m]')
plt.ylabel('[m]')
plt.title('DM Actuator Coordinates')

#%% -----------------------     PYRAMID WFS   ----------------------------------


from OOPAO.Pyramid import Pyramid
param['postProcessing'] = 'fullFrame_incidence_flux'



bio_20 = BioEdge(nSubap             = param['nSubaperture'],\
              telescope             = tel,\
              modulation            = 0,\
              grey_width            = 2,\
              lightRatio            = 0,\
              n_pix_separation      = 0,\
              psfCentering          = True,\
              postProcessing        = param['postProcessing'],calibModulation=50)
    
VALID = bio_20.validSignal.copy()
    
bio_20_GSR = BioEdge(nSubap                = param['nSubaperture'],\
              telescope             = tel,\
              modulation            = 0,\
              grey_width            = 2,\
              lightRatio            = 0,\
              n_pix_separation      = 0,\
              psfCentering          = True,\
              postProcessing        = param['postProcessing'],calibModulation=50,userValidSignal=VALID)
    
    
bio_40 = BioEdge(nSubap                = 2*param['nSubaperture'],\
              telescope             = tel,\
              modulation            = 0,\
              grey_width            = 2,\
              lightRatio            = 0,\
              n_pix_separation      = 0,\
              psfCentering          = True,\
              postProcessing        = param['postProcessing'],calibModulation=50)    
    
plt.figure()
plt.imshow(bio_20.cam.frame)
#%%
plt.close('all')

# for i in range(21):
#     x = i*2
bio_20_GSR.apply_shift_wfs(sx=[0,0,0,0],sy=[0,0,0,0])
x = 0.5
bio_20_GSR.apply_shift_wfs(sx=[x/4,-x/3,-x/2,x/1],sy = [0,x,0,-x] )

plt.figure()
plt.imshow(bio_20_GSR.cam.frame-0*bio_20.cam.frame)

#%%
from OOPAO.tools.displayTools import display_wfs_signals, makeSquareAxes



display_wfs_signals(bio_20,bio_20.referenceSignal)
plt.colorbar()
display_wfs_signals(bio_20,bio_20_GSR.referenceSignal)
plt.colorbar()



plt.figure(),plt.imshow(bio_20_GSR.mask_TT[0])

#%%
from OOPAO.calibration.compute_KL_modal_basis import compute_M2C
M2C_KL_full = compute_M2C(telescope            = tel,\
                                  atmosphere         = atm,\
                                  deformableMirror   = dm,\
                                  param              = param,\
                                  nameFolder         = None,\
                                  nameFile           = None,\
                                  remove_piston      = True,\
                                  HHtName            = None,\
                                  baseName           = None ,\
                                  mem_available      = 6.1e9,\
                                  minimF             = False,\
                                  nmo                = 1100,\
                                  ortho_spm          = True,\
                                  SZ                 = int(2*tel.OPD.shape[0]),\
                                  nZer               = 3,\
                                  NDIVL              = 1)


#%%


M2C_KL = M2C_KL_full[:,:]
stroke = 1e-9


calib_bio_20 = InteractionMatrix(  ngs            = ngs,\
                            atm            = atm,\
                            tel            = tel,\
                            dm             = dm,\
                            wfs            = bio_20,\
                            M2C            = M2C_KL,\
                            stroke         = stroke,\
                            nMeasurements  = 2,\
                            noise          = 'off',single_pass=True,display=True)
    
calib_bio_20_GSR = InteractionMatrix(  ngs            = ngs,\
                                atm            = atm,\
                                tel            = tel,\
                                dm             = dm,\
                                wfs            = bio_20_GSR,\
                                M2C            = M2C_KL,\
                                stroke         = stroke,\
                                nMeasurements  = 2,\
                                noise          = 'off',single_pass=True,display=True)


calib_bio_40 = InteractionMatrix(  ngs            = ngs,\
                                atm            = atm,\
                                tel            = tel,\
                                dm             = dm,\
                                wfs            = bio_40,\
                                M2C            = M2C_KL,\
                                stroke         = stroke,\
                                nMeasurements  = 2,\
                                noise          = 'off',
                                single_pass=True,
                                display=True)



#%%

import copy

test_wfs = [bio_40,bio_20,bio_20_GSR]


test_calib = [calib_bio_40,calib_bio_20,calib_bio_20_GSR]
plt.close('all')

title_fig = ['40x40','20x20','20x20 + GSR'] 

for i_mode in [200,300,400,500,600]:
    propa = []

    for i_wfs in range(3):

        bio = copy.deepcopy(test_wfs[i_wfs])
        calib_CL    = test_calib[i_wfs]
        calib_CL = CalibrationVault(calib_CL.D[:,:i_mode])
    
        plt.figure(10)
        
        # propa.append(np.diag(calib_CL.M@calib_CL.M.T))
        plt.subplot(1,3,i_wfs+1)
        tmp = np.diag(calib_CL.M@calib_CL.M.T)
        
        plt.semilogy(tmp,label=str(i_mode)+'Modes' )
        plt.xlabel('KL Mode index')
        plt.ylabel('Noise Propagation Coefficient')
        plt.legend()
        plt.title(title_fig[i_wfs])
        # plt.legend(['PYR 40x40','PYR 20x20','PYR 20x20 WITH GSR'])
        # plt.ylim([1e-16 ,1e-12])
        makeSquareAxes()
        
        # plt.figure(15)
        # plt.plot(np.diag(calib_CL.D.T@calib_CL.D))
        # plt.xlabel('KL Mode index')
        # plt.ylabel('STD IM signals')
        # plt.legend(['PYR 40x40','PYR 20x20','PYR 20x20 WITH GSR'])
    
        plt.figure(20)
        # plt.subplot(1,3,i_wfs+1)
        plt.semilogy(calib_CL.eigenValues)
        # plt.xlabel('Singular Mode index')
        # plt.ylabel('Singular Mode value [Normalized]',label=str(i_mode) )
        plt.legend()

        # plt.legend(['PYR 40x40','PYR 20x20','PYR 20x20 WITH GSR'])
        makeSquareAxes()
#%%
for i in range(3):
    plt.figure()
    tmp = propa[i][50:]
    plt.plot(tmp[np.argsort(tmp)])







#%%
# modal projector

dm.coefs = M2C_KL_full
tel.resetOPD()

ngs*tel*dm

KL_basis = np.reshape(tel.OPD,[tel.resolution**2, tel.OPD.shape[2]])

KL_basis = np.squeeze(KL_basis[tel.pupilLogical,:])

projector = np.linalg.pinv(KL_basis)




#%%

plt.close('all')
tel.resetOPD()
# initialize DM commands
dm.coefs=0
ngs*tel*dm*bio
tel-atm

out_perf = []
for i_wfs in range(3):
    tmp = propa[i_wfs]
    n = 150
    index = np.arange(n)
    # index[50:] = 50+np.argsort(tmp[50:n])


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
    
    tel.computePSF(4)
        
    # These are the calibration data used to close the loop
    
    
    # combine telescope with atmosphere
    tel+atm
    
    # initialize DM commands
    dm.coefs=0
    ngs*tel*dm*bio
    
    
    plt.show()
    
    param['nLoop'] = 2000
    # allocate memory to save data
    SR                      = np.zeros(param['nLoop'])
    total                   = np.zeros(param['nLoop'])
    residual                = np.zeros(param['nLoop'])
    bioSignal               = np.arange(0,bio.nSignal)*0
    SE_PSF = []
    LE_PSF = np.log10(tel.PSF)
    
    plot_obj = cl_plot(list_fig          = [atm.OPD,tel.mean_removed_OPD,bio.cam.frame,[[0,0],[0,0]],[dm.coordinates[:,0],np.flip(dm.coordinates[:,1]),dm.coefs],np.log10(tel.PSF),np.log10(tel.PSF)],\
                        type_fig          = ['imshow','imshow','imshow','plot','scatter','imshow','imshow'],\
                        list_title        = ['Turbulence OPD','Residual OPD','bio Detector',None,None,None,None],\
                        list_lim          = [None,None,None,None,None,[-4,0],[-4,0]],\
                        list_label        = [None,None,None,None,['Time','WFE [nm]'],['DM Commands',''],['Short Exposure PSF',''],['Long Exposure_PSF','']],\
                        n_subplot         = [4,2],\
                        list_display_axis = [None,None,None,True,None,None,None],\
                        list_ratio        = [[0.95,0.95,0.1],[1,1,1,1]], s=20)
    # loop parameters
    gainCL                  = 0.4
    bio.cam.photonNoise     = False
    display                 = True
    
    reconstructor = M2C_CL@calib_CL.M
    
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
    
        dm.coefs=dm.coefs-gainCL*np.matmul(reconstructor,bioSignal)
        # store the slopes after computing the commands => 2 frames delay
        bioSignal=bio.signal
        b= time.time()
        print('Elapsed time: ' + str(b-a) +' s')
        # update displays if required
        if display==True:        
            tel.computePSF(4)
            if i>15:
                SE_PSF.append(np.log10(tel.PSF))
                LE_PSF = np.mean(SE_PSF, axis=0)
            
            cl_plot(list_fig   = [atm.OPD,tel.mean_removed_OPD,bio.cam.frame,[np.arange(i+1),residual[:i+1]],dm.coefs,np.log10(tel.PSF), LE_PSF],
                                   plt_obj = plot_obj)
            plt.pause(0.1)
            if plot_obj.keep_going is False:
                break
        
        SR[i]=np.exp(-np.var(tel.src.phase[np.where(tel.pupil==1)]))
        residual[i]=np.std(tel.OPD[np.where(tel.pupil>0)])*1e9
        OPD=tel.OPD[np.where(tel.pupil>0)]
    
        print('Loop'+str(i)+'/'+str(param['nLoop'])+' Turbulence: '+str(total[i])+' -- Residual:' +str(residual[i])+ '\n')
    
    modes_in = np.asarray(modes_in)
    modes_out = np.asarray(modes_out)
    
    
    out_perf.append([modes_in,modes_out, SR])
    
    
    
    # plt.figure(100)
    # plt.loglog(np.std(modes_in[50:,:],axis=0))
    # plt.loglog(np.std(modes_out[50:,:],axis=0))
    
#%%
plt.figure()
plt.loglog(np.std(out_perf[0][0][50:,:],axis=0),'k')
    
for i in range(3):
    plt.loglog(np.std(out_perf[i][1][50:,:],axis=0))
    
plt.legend(['Turbulence','PYR 20x20','PYR 10x10','PYR 10x10 WITH GSR'])
plt.grid(which='both')
plt.xlabel('KL Mode index')
plt.ylabel('WFE [nm]')


    #%%
plt.figure()
plt.plot(total)
plt.plot(residual)
plt.xlabel('Time')
plt.ylabel('WFE [nm]')

