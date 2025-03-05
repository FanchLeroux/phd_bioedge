
import scipy.ndimage as sp
import numexpr as ne


def PYR_MOD(IMC,Omega,MASK,nsplit):
    nmo = IMC.shape[2]
    nma = MASK.shape[2]
    idxO = np.where(Omega!=0)
    nmodu = len(idxO[0])
    PYRCCD = np.zeros([SZ,SZ,nmo,nma],dtype=np.float64)
    Osx = idxO[0]-SZ//2
    Osy = idxO[1]-SZ//2
    for k in range(0,nmodu,2):
        if Osx[k]<0:
            Osx[k] = Osx[k]+1
        if Osy[k]<0:
            Osy[k] = Osy[k]+1
    IMC2F = np.zeros([SZ,SZ,nmo],dtype=np.complex128)
    for ma in range(0,nma):
        print('MASK # ',ma)
        for mo in range(0,nmodu):
            for k in range(0,nmo):
                IMC2F[:,:,k] = aou.myshift2D(IMC[:,:,k],Osx[mo],Osy[mo])*MASK[:,:,ma]
        FT_ = aou.DO_FT_MULTI_C(IMC2F,nsplit)
        PYRCCD[:,:,:,ma] =  PYRCCD[:,:,:,ma]+np.abs(FT_)**2
    return PYRCCD

def PYR_MOD_C1w(IMC,Omega,MASK,nsplit):
    nmo = IMC.shape[2]
    nma = MASK.shape[2]
    idxO = np.where(Omega!=0)
    nmodu = len(idxO[0])
    PYRCCD = np.zeros([SZ,SZ,nmo,nma],dtype=np.float64)
    PYRCCDs = np.zeros([SZ,SZ,nmo,nma,nmodu],dtype=np.float64)
    Osx = idxO[1]-SZ//2
    Osy = idxO[0]-SZ//2
    IMC2F = np.zeros([SZ,SZ,nmo],dtype=np.complex128)
    IMC2Fs = np.zeros([SZ,SZ,nmo,nmodu,nma],dtype=np.complex128)
    for ma in range(0,nma):
        print('MASK # ',ma)
        for mo in range(0,nmodu):
            for k in range(0,nmo):
                IMC2F[:,:,k] = aou.myshift2D(IMC[:,:,k],Osx[mo],Osy[mo])*MASK[:,:,ma]
                IMC2Fs[:,:,k,mo,ma] = IMC2F[:,:,k]  #aou.myshift2D(IMC[:,:,k],Osx[mo],Osy[mo])*MASK[:,:,ma]
        FT_ = aou.DO_FT_MULTI_C(IMC2F,nsplit)
        pdb.set_trace()
        PYRCCDs[:,:,:,ma,mo] = np.abs(FT_)**2
        PYRCCD[:,:,:,ma] =  PYRCCD[:,:,:,ma]+np.abs(FT_)**2
    return PYRCCD, PYRCCDs,IMC2Fs




def PYR_MOD_C1(IMC,Omega,MASK,nsplit):
    nmo = IMC.shape[2]
    nma = MASK.shape[2]
    idxO = np.where(Omega!=0)
    nmodu = len(idxO[0])
    PYRCCD = np.zeros([SZ,SZ,nmo,nma],dtype=np.float64)
    PYRCCDs = np.zeros([SZ,SZ,nmo,nma,nmodu],dtype=np.float64)
    Osx = idxO[1]-SZ//2
    Osy = idxO[0]-SZ//2
    IMC2F = np.zeros([SZ,SZ,nmo],dtype=np.complex128)
    for ma in range(0,nma):
        print('MASK # ',ma)
        for mo in range(0,nmodu):
            for k in range(0,nmo):
                IMC2F[:,:,k] = aou.myshift2D(IMC[:,:,k],Osx[mo],Osy[mo])*MASK[:,:,ma]
            FT_ = aou.DO_FT_MULTI_C(IMC2F,nsplit)
            PYRCCD[:,:,:,ma] =  PYRCCD[:,:,:,ma]+np.abs(FT_)**2
    return PYRCCD



def PYR_MOD_C1_para(IMC,Omega,MASK,nsplit):
    SZ=IMC.shape[0]
    nmo = IMC.shape[2]
    nma = MASK.shape[2]
    idxO = np.where(Omega!=0)
    nmodu = len(idxO[0])
    PYRCCD = np.zeros([SZ,SZ,nmo,nma],dtype=np.float64)
    Osx = idxO[1]-SZ//2
    Osy = idxO[0]-SZ//2
    ## LOOP
    for ma in range(0,nma):
        IMC2Fs = np.zeros([SZ,SZ,nmo,nmodu],dtype=np.complex128)
        print('MASK # ',ma)
        print('FILLING IMC2FS...')
        for mo in range(0,nmodu):
            for k in range(0,nmo):
                IMC2Fs[:,:,k,mo] = aou.myshift2D(IMC[:,:,k],Osx[mo],Osy[mo])*MASK[:,:,ma]
        print('FILLING IMC2FS: DONE')
        IMC2F = np.reshape(IMC2Fs,[SZ,SZ,nmo*nmodu])
        print('COMPUTING MULTI FT...')
        FT_ = aou.DO_FT_MULTI_C(IMC2F,nsplit)
        print('COMPUTING MULTI FT: DONE')
        FT_A = np.reshape(FT_,[SZ,SZ,nmo,nmodu])
        FT_Aabs2 = np.abs(FT_A)**2
        SUM_OVER_MOD = np.sum(FT_Aabs2,axis=3)
        PYRCCD[:,:,:,ma] =  SUM_OVER_MOD
    return PYRCCD 



def PHASE_PYR_MOD_C1(IMC,Omega,sz,PM,sep,szb=0,nsplit=1):
    if szb == 0:
        szb=sz
    decal = np.int64(sz*sep/2.)
    nmo = IMC.shape[2]
    nma=4
    idxO = np.where(Omega!=0)
    nmodu = len(idxO[0])
    PYRCCD = np.zeros([szb,szb,nmo,nma],dtype=np.float64)
    Osx = idxO[1]-SZ//2
    Osy = idxO[0]-SZ//2
    PM_c = ne.evaluate('exp(1j*PM)')
    ## LOOP
    IMC2Fs = np.zeros([SZ,SZ,nmo,nmodu],dtype=np.complex128)
    print('FILLING IMC2FS...')
    for mo in range(0,nmodu):
        for k in range(0,nmo):
            ISMO = aou.myshift2D(IMC[:,:,k],Osx[mo],Osy[mo])
            IMC2Fs[:,:,k,mo] = ISMO*PM_c
    print('FILLING IMC2FS: DONE')
    IMC2F = np.reshape(IMC2Fs,[SZ,SZ,nmo*nmodu])
    print('COMPUTING MULTI FT...')
    FT_ = aou.DO_FT_MULTI_C(IMC2F,nsplit)
    print('COMPUTING MULTI FT: DONE')
    FT_A = np.reshape(FT_,[SZ,SZ,nmo,nmodu])
    FT_Aabs2 = np.abs(FT_A)**2
    SUM_OVER_MOD = np.sum(FT_Aabs2,axis=3)
    #pdb.set_trace()
    PYRCCD[:,:,:,0] = SUM_OVER_MOD[SZ//2-decal-szb//2 : SZ//2-decal+szb//2 , SZ//2+decal-szb//2 : SZ//2+decal+szb//2, :  ]
    PYRCCD[:,:,:,1] = SUM_OVER_MOD[SZ//2+decal-szb//2 : SZ//2+decal+szb//2 , SZ//2+decal-szb//2 : SZ//2+decal+szb//2, :  ]
    PYRCCD[:,:,:,2] = SUM_OVER_MOD[SZ//2-decal-szb//2 : SZ//2-decal+szb//2 , SZ//2-decal-szb//2 : SZ//2-decal+szb//2, :  ]
    PYRCCD[:,:,:,3] = SUM_OVER_MOD[SZ//2+decal-szb//2 : SZ//2+decal+szb//2 , SZ//2-decal-szb//2 : SZ//2-decal+szb//2, :  ]
    return PYRCCD, SUM_OVER_MOD


def PHASE_PYR_MOD0_C1(IMC,sz,PM,sep,szb=0,nsplit=1):
    if szb == 0:
        szb=sz
    decal = np.int64(sz*sep/2.)
    nmo = IMC.shape[2]
    nma=4
    PYRCCD = np.zeros([szb,szb,nmo,nma],dtype=np.float64)
    PM_c = ne.evaluate('exp(1j*PM)')
    print('FILLING IMC2F...')
    IMC2F = np.zeros([SZ,SZ,nmo],dtype=np.complex128)
    for k in range(0,nmo):
        IMC2F[:,:,k] = IMC[:,:,k]*PM_c
    print('FILLING IMC2F: DONE')
    print('COMPUTING MULTI FT: DONE')
    print('COMPUTING MULTI FT...')
    FT_A = aou.DO_FT_MULTI_C(IMC2F,nsplit)
    print('COMPUTING MULTI FT: DONE')
    FT_Aabs2 = np.abs(FT_A)**2
    #pdb.set_trace()
    PYRCCD[:,:,:,0] = FT_Aabs2[SZ//2-decal-szb//2 : SZ//2-decal+szb//2 , SZ//2+decal-szb//2 : SZ//2+decal+szb//2, :  ]
    PYRCCD[:,:,:,1] = FT_Aabs2[SZ//2+decal-szb//2 : SZ//2+decal+szb//2 , SZ//2+decal-szb//2 : SZ//2+decal+szb//2, :  ]
    PYRCCD[:,:,:,2] = FT_Aabs2[SZ//2-decal-szb//2 : SZ//2-decal+szb//2 , SZ//2-decal-szb//2 : SZ//2-decal+szb//2, :  ]
    PYRCCD[:,:,:,3] = FT_Aabs2[SZ//2+decal-szb//2 : SZ//2+decal+szb//2 , SZ//2-decal-szb//2 : SZ//2-decal+szb//2, :  ]
    return PYRCCD





def PHASE_PYRCCD_2_SIGNAL(PYRCCD,SZo,pupil,dd,dxo,nmodu):
    tpup=np.sum(pupil)
    SZ=PYRCCD.shape[0]
    sz=pupil.shape[0]
    nmoi=PYRCCD.shape[2]
    nma=PYRCCD.shape[3]
    npix = np.int64(dd/dxo)
    rs = SZ//npix
    rss = rs//2
    PUPIL=np.zeros([SZ,SZ])
    PUPIL[SZ//2-sz//2:SZ//2+sz//2,SZ//2-sz//2:SZ//2+sz//2]=pupil
    IDXPUP = np.where(PUPIL == 1)
    PUPIL_R_s = bin_ndarray(PUPIL,[rs,rs],operation='sum')
    PYR_CCD_R_s = np.zeros([rs,rs,nmoi,nma])
    normout = SZo**4*nmodu*tpup
    P_R = PYRCCD / normout *tpup/npix**2
    for k in range(0,nmoi):
        print(k, ' ', end='\r', flush=True)
        for j in range(0,nma):
            PYR_CCD_R_s[:,:,k,j] = bin_ndarray(P_R[:,:,k,j],[rs,rs],operation='sum')
    PYR_SX_s = (PYR_CCD_R_s[:,:,:,1] + PYR_CCD_R_s[:,:,:,3]) - (PYR_CCD_R_s[:,:,:,0] + PYR_CCD_R_s[:,:,:,2])
    PYR_SY_s = (PYR_CCD_R_s[:,:,:,0] + PYR_CCD_R_s[:,:,:,1]) - (PYR_CCD_R_s[:,:,:,2] + PYR_CCD_R_s[:,:,:,3])
    IDXP_S = np.where(PUPIL_R_s != 0)
    nsub = len(IDXP_S[0])
    Signal = np.zeros([nsub*2,nmoi])
    for k in range(0,nmoi):
        print(k, ' ', end='\r', flush=True)
        Signal[0:nsub,k] = PYR_SX_s[:,:,k][IDXP_S]
        Signal[nsub:,k] = PYR_SY_s[:,:,k][IDXP_S]
    return Signal,PYR_SX_s,PYR_SY_s,PYR_CCD_R_s


    


def PYR_MODf(IMC,Omega,MASK,nsplit,marge):
    SZ = IMC.shape[0]
    SM=SZ-marge
    nmo = IMC.shape[2]
    nma = MASK.shape[2]
    idxO = np.where(Omega!=0)
    nmodu = len(idxO[0])
    PYRCCD = np.zeros([SZ,SZ,nmo,nma],dtype=np.float64)
    Osx = idxO[0]-SZ//2
    Osy = idxO[1]-SZ//2
    for k in range(0,nmodu):
        if Osx[k]<0:
            Osx[k] = Osx[k]+1
        if Osy[k]<0:
            Osy[k] = Osy[k]+1
    IMC2F = np.zeros([SZ,SZ,nmo],dtype=np.complex128)
    #Imodu= np.zeros([SZ,SZ,nmo],dtype=np.float64)
    for ma in range(0,nma):
        print('MASK # ',ma)
        for mo in range(0,nmodu):
            #Imodu= np.zeros([SZ,SZ,nmo],dtype=np.float64)
            for k in range(0,nmo):
                IMC2F[SZ//2-SM//2:SZ//2+SM//2 , SZ//2-SM//2:SZ//2+SM//2,k] = IMC[SZ//2+Osx[mo]-SM//2:SZ//2+Osx[mo]+SM//2 , SZ//2+Osy[mo]-SM//2:SZ//2+Osy[mo]+SM//2,k]*MASK[SZ//2-SM//2:SZ//2+SM//2 , SZ//2-SM//2:SZ//2+SM//2,ma]
            FT_ = aou.DO_FT_MULTI_C(IMC2F,nsplit)
            PYRCCD[:,:,:,ma] =  PYRCCD[:,:,:,ma]+np.abs(FT_)**2
    return PYRCCD






#def PYR_MODUL(IMC,Omega,MASK,nsplit,marge):
def PYR_MODUL(IMC,Omega,MASK,nsplit):
    SZ = IMC.shape[0]
    #SM=SZ-marge
    nmo = IMC.shape[2]
    nma = MASK.shape[2]
    idxO = np.where(Omega!=0)
    nmodu = len(idxO[0])
    PYRCCD = np.zeros([SZ,SZ,nmo,nma],dtype=np.float64)
    Osx = idxO[0]-SZ//2
    Osy = idxO[1]-SZ//2
    for k in range(0,nmodu):
        if Osx[k]<0:
            Osx[k] = Osx[k]+1
        if Osy[k]<0:
            Osy[k] = Osy[k]+1
    IMC2F = np.zeros([SZ,SZ,nmo],dtype=np.complex128)
    #Imodu= np.zeros([SZ,SZ,nmo],dtype=np.float64)
    for ma in range(0,nma):
        print('MASK # ',ma)
        for mo in range(0,nmodu):
            for k in range(0,nmo):
                IMC2F[:,:,k] =  aou.myshift2D(IMC[:,:,k],Osx[mo],Osy[mo])*MASK[:,:,ma]
            FT_ = aou.DO_FT_MULTI_C(IMC2F,nsplit)
            PYRCCD[:,:,:,ma] =  PYRCCD[:,:,:,ma]+np.abs(FT_)**2
    return PYRCCD



# IMC2F[SZ//2-SM//2:SZ//2+SM//2 , SZ//2-SM//2:SZ//2+SM//2,k] = IMC[SZ//2+Osx[mo]-SM//2:SZ//2+Osx[mo]+SM//2 , SZ//2+Osy[mo]-SM//2:SZ//2+Osy[mo]+SM//2,k]*MASK[SZ//2-SM//2:SZ//2+SM//2 , SZ//2-SM//2:SZ//2+SM//2,ma]




def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray


def PYRCCD_2_SIGNAL(PYRCCD,pupil,dd,dxo,nmodu):
    SZ=PYRCCD.shape[0]
    sz=pupil.shape[0]
    nmoi=PYRCCD.shape[2]
    nma=PYRCCD.shape[3]
    npix = np.int64(dd/dxo)
    rs = SZ//npix
    rss = rs//2
    PUPIL=np.zeros([SZ,SZ])
    PUPIL[SZ//2-sz//2:SZ//2+sz//2,SZ//2-sz//2:SZ//2+sz//2]=pupil
    IDXPUP = np.where(PUPIL == 1)
    PUPIL_R = bin_ndarray(PUPIL,[rs,rs],operation='sum')
    PUPIL_R_s = PUPIL_R[rss-rss//2:rss+rss//2,rss-rss//2:rss+rss//2]
    PYR_CCD_R = np.zeros([rs,rs,nmoi,nma])
    normout = SZ**4*nmodu*tpup
    P_R = PYRCCD / normout *tpup/npix**2
    for k in range(0,nmoi):
        print(k, ' ', end='\r', flush=True)
        for j in range(0,nma):
            PYR_CCD_R[:,:,k,j] = bin_ndarray(P_R[:,:,k,j],[rs,rs],operation='sum')
    PYR_CCD_R_s = np.zeros([rss,rss,nmoi,nma])    
    for k in range(0,nmoi):
        print(k, ' ', end='\r', flush=True)
        for j in range(0,nma):
            PYR_CCD_R_s[:,:,k,j] = PYR_CCD_R[:,:,k,j][rss-rss//2:rss+rss//2,rss-rss//2:rss+rss//2]
    PYR_SX_s = (PYR_CCD_R_s[:,:,:,1] + PYR_CCD_R_s[:,:,:,3]) - (PYR_CCD_R_s[:,:,:,0] + PYR_CCD_R_s[:,:,:,2])
    PYR_SY_s = (PYR_CCD_R_s[:,:,:,0] + PYR_CCD_R_s[:,:,:,1]) - (PYR_CCD_R_s[:,:,:,2] + PYR_CCD_R_s[:,:,:,3])
    IDXP_S = np.where(PUPIL_R_s != 0)
    nsub = len(IDXP_S[0])
    Signal = np.zeros([nsub*2,nmoi])
    for k in range(0,nmoi):
        print(k, ' ', end='\r', flush=True)
        Signal[0:nsub,k] = PYR_SX_s[:,:,k][IDXP_S]
        Signal[nsub:,k] = PYR_SY_s[:,:,k][IDXP_S]
    return Signal



def MASK_OMEGA(SZ,sz,modulation_angle):
    M1=np.zeros([SZ,SZ])
    M2=np.zeros([SZ,SZ])
    M3=np.zeros([SZ,SZ])
    M4=np.zeros([SZ,SZ])
    M1[SZ//2:,0:SZ//2] = 1
    M2[SZ//2:,SZ//2:] = 1
    M3[0:SZ//2,0:SZ//2]=1
    M4[0:SZ//2,SZ//2:] = 1
    MASK = np.moveaxis(np.asarray( [M1,M2,M3,M4] ),0, -1   )
    radius_pix = modulation_angle*SZ/sz
    MODU_PATH = np.zeros([SZ,SZ])
    ep=1
    dim_x=SZ
    dim_y=SZ
    xc    = (dim_x-1)/2.
    yc    = (dim_y-1)/2.
    xx = np.resize(np.array(range(dim_x)), (dim_x, dim_y)) - xc
    yy = np.transpose(xx)
    rr = np.sqrt(xx**2+yy**2)
    idxin = np.where(rr > radius_pix)
    idxout = np.where(rr > radius_pix+ep)
    MODU_PATH[idxin]=1.0
    MODU_PATH[idxout]=0.
    MODU_PATH = MODU_PATH / np.sum(MODU_PATH)
    OMEGA =  MODU_PATH.copy()
    return MASK,OMEGA



def MASK_OMEGA_C1(SZ,sz,modulation_angle):
    M1=np.zeros([SZ,SZ])
    M2=np.zeros([SZ,SZ])
    M3=np.zeros([SZ,SZ])
    M4=np.zeros([SZ,SZ])
    ## M1
    M1[SZ//2:,0:SZ//2] = 1
    M1[SZ//2,0:SZ//2] = np.sqrt(0.5)
    M1[SZ//2:,SZ//2] = np.sqrt(0.5)
    M1[SZ//2,SZ//2] = np.sqrt(0.25)
    ## M2
    M2[SZ//2:,SZ//2:] = 1
    M2[SZ//2,SZ//2:]= np.sqrt(0.5)
    M2[SZ//2:,SZ//2]= np.sqrt(0.5)
    M2[SZ//2,SZ//2] = np.sqrt(0.25)
    # M3
    M3[0:SZ//2,0:SZ//2] = 1
    M3[SZ//2,0:SZ//2] = np.sqrt(0.5)
    M3[0:SZ//2,SZ//2] = np.sqrt(0.5)
    M3[SZ//2,SZ//2] = np.sqrt(0.25)
    #M4
    M4[0:SZ//2,SZ//2:] = 1
    M4[SZ//2,SZ//2:] = np.sqrt(0.5)
    M4[0:SZ//2,SZ//2] = np.sqrt(0.5)
    M4[SZ//2,SZ//2]  = np.sqrt(0.25)
    MASK = np.moveaxis(np.asarray( [M1,M2,M3,M4] ),0, -1   )
    radius_pix = modulation_angle*SZ/sz
    MODU_PATH = np.zeros([SZ,SZ])
    ep=1
    dim_x=SZ
    dim_y=SZ
    xc    = dim_x/2. #(dim_x-1)/2.
    yc    = dim_y/2. #(dim_y-1)/2.
    xx = np.resize(np.array(range(dim_x)), (dim_x, dim_y)) - xc
    yy = np.transpose(xx)
    rr = np.sqrt(xx**2+yy**2)
    idxin = np.where(rr > radius_pix-ep/2.)
    idxout = np.where(rr > radius_pix+ep/2.)
    MODU_PATH[idxin]=1.0
    MODU_PATH[idxout]=0.
    MODU_PATH = MODU_PATH / np.sum(MODU_PATH)
    OMEGA =  MODU_PATH.copy()
    return MASK,OMEGA


def MASK_OMEGA_C1_MOD_LIGHT(SZ,sz,modulation_angle):
    M1=np.zeros([SZ,SZ])
    M2=np.zeros([SZ,SZ])
    M3=np.zeros([SZ,SZ])
    M4=np.zeros([SZ,SZ])
    ## M1
    M1[SZ//2:,0:SZ//2] = 1
    M1[SZ//2,0:SZ//2] = np.sqrt(0.5)
    M1[SZ//2:,SZ//2] = np.sqrt(0.5)
    M1[SZ//2,SZ//2] = np.sqrt(0.25)
    ## M2
    M2[SZ//2:,SZ//2:] = 1
    M2[SZ//2,SZ//2:]= np.sqrt(0.5)
    M2[SZ//2:,SZ//2]= np.sqrt(0.5)
    M2[SZ//2,SZ//2] = np.sqrt(0.25)
    # M3
    M3[0:SZ//2,0:SZ//2] = 1
    M3[SZ//2,0:SZ//2] = np.sqrt(0.5)
    M3[0:SZ//2,SZ//2] = np.sqrt(0.5)
    M3[SZ//2,SZ//2] = np.sqrt(0.25)
    #M4
    M4[0:SZ//2,SZ//2:] = 1
    M4[SZ//2,SZ//2:] = np.sqrt(0.5)
    M4[0:SZ//2,SZ//2] = np.sqrt(0.5)
    M4[SZ//2,SZ//2]  = np.sqrt(0.25)
    MASK = np.moveaxis(np.asarray( [M1,M2,M3,M4] ),0, -1   )
    radius_pix = modulation_angle*SZ/sz
    MODU_PATH = np.zeros([SZ,SZ])
    ep=1
    dim_x=SZ
    dim_y=SZ
    xc    = dim_x/2. #(dim_x-1)/2.
    yc    = dim_y/2. #(dim_y-1)/2.
    xx = np.resize(np.array(range(dim_x)), (dim_x, dim_y)) - xc
    yy = np.transpose(xx)
    rr = np.sqrt(xx**2+yy**2)
    #idxin = np.where(rr > radius_pix-ep/2.)
    #idxout = np.where(rr > radius_pix+ep/2.)
    #pdb.set_trace()
    #a1 = np.zeros(nstep//4)
    #a2 = np.zeros(nstep//4)
    nstep = np.int64(8*modulation_angle)
    for k in range(0,nstep//4+1):
        a1 = np.int64((np.cos(2.*np.pi/nstep*k)*modulation_angle*SZ//sz))
        a2 = np.int64((np.sin(2.*np.pi/nstep*k)*modulation_angle*SZ//sz))
        #a1 = np.int64(np.round(np.cos(2.*np.pi/nstep*k)*modulation_angle*SZ//sz))
        #a2 = np.int64(np.round(np.sin(2.*np.pi/nstep*k)*modulation_angle*SZ//sz))
        MODU_PATH[SZ//2+a1,SZ//2+a2] =1.0
    #MODU_PATH[idxin]=1.0
    #MODU_PATH[idxout]=0.
    #pdb.set_trace()
    MODU_PATH = MODU_PATH / np.sum(MODU_PATH)
    OMEGA =  MODU_PATH.copy()
    return MASK,OMEGA





def PHASE_MASK_C1(SZ,sz,SEP,centering=False):
    print('Pyramid Mask initialization...')
    M = np.ones([SZ,SZ])
    # mask centered on 4 pixels
    if centering == True:
        M[:,-1+SZ//2] = 0
        M[:,SZ//2] = 0
        m = sp.morphology.distance_transform_edt(M) 
        m += np.transpose(m)
        # normalization of the mask 
        m[-1+SZ//2:SZ//2,-1+SZ//2:SZ//2] = m[-1+SZ//2:SZ//2,-1+SZ//2:SZ//2]/2
        m /= np.sqrt(2) 
        m -= m.mean()
        m*=(1+1/self.nSubap)     
    #        # create an amplitude mask for the debugging
    #            self.debugMask = np.ones([SZ,SZ])
    #            self.debugMask[:,-1+SZ//2:]=0
    #            self.debugMask[-1+SZ//2:,:]=0
    if centering == False:
        M[:,SZ//2] = 0
        m = sp.morphology.distance_transform_edt(M) 
        m += np.transpose(m)
        # normalization of the mask 
        m[SZ//2,SZ//2] = m[SZ//2,SZ//2]/2   
        m /= np.sqrt(2) 
     # apply the right angle to the faces
    if np.isscalar(SEP):
        # case with a single value for each face (perfect pyramid)
        m*=((np.pi/(SZ/sz))*SEP)/np.sin(np.pi/4)
    return m



def BUILD_Uii(FT_KLs,FT_KLs2,PUPIL):
    FT_P = np.fft.fftshift( np.fft.fft2(np.fft.fftshift(PUPIL))  )
    nmo = FT_KLs.shape[2]
    SZ = FT_KLs.shape[0]
    Uii = np.zeros([SZ,SZ,nmo])
    for k in range(0,nmo):
        print(k, ' ', end='\r', flush=True)
        FT_Uiik = 2.*(FT_KLs2[:,:,k] * FT_P).real - 2.*(  FT_KLs[:,:,k] **2 ).real
        Uii[:,:,k] =  np.fft.fftshift( np.fft.fft2(np.fft.fftshift(FT_Uiik))  ).real
    return Uii


def BUILD_Uii_(FT_KLs,FT_KLs2,PUPIL):
    FT_P = np.fft.fftshift( np.fft.fft2(np.fft.fftshift(PUPIL))  )
    nmo = FT_KLs.shape[2]
    SZ = FT_KLs.shape[0]
    Uii = np.zeros([SZ,SZ,nmo])
    for k in range(0,nmo):
        print(k, ' ', end='\r', flush=True)
        FT_Uiik = 2.*(FT_KLs2[:,:,k] * FT_P) - 2.*(  FT_KLs[:,:,k] **2 ).real
        Uii[:,:,k] =  np.fft.fftshift( np.fft.fft2(np.fft.fftshift(FT_Uiik))  )
    return Uii




def BUILD_Uii2(FT_KLs,FT_KLs2,PUPIL):
    FT_P = np.fft.fftshift( np.fft.fft2(np.fft.fftshift(PUPIL))  )
    nmo = FT_KLs.shape[2]
    SZ = FT_KLs.shape[0]
    Uii = np.zeros([SZ,SZ,nmo])
    for k in range(0,nmo):
        print(k, ' ', end='\r', flush=True)
        FT_Uiik = 2.*(FT_KLs2[:,:,k] * FT_P).real + 2.*(  FT_KLs[:,:,k] **2 ).real
        Uii[:,:,k] =  np.fft.fftshift( np.fft.fft2(np.fft.fftshift(FT_Uiik))  ).real
    return Uii

from scipy.signal import fftconvolve

    

def COMPUTE_UiiPUP(MODEi,MODEj,PUPIL):
    SZ=MODEi.shape[0]
    OUT = np.zeros([SZ,SZ])
    for k in range(-SZ//2+1,SZ//2):
        print(k, ' ', end='\r', flush=True)
        for j in range(-SZ//2+1,SZ//2):
            OUT[SZ//2-1+k,SZ//2-1+j]  =  np.sum(  (PUPIL*aou.myshift2D(PUPIL,k,j))  )
    return OUT


def COMPUTE_Uii(MODEi,MODEj,PUPIL):
    SZ=MODEi.shape[0]
    OUT = np.zeros([SZ,SZ])
    for k in range(-SZ//2,SZ//2):
        print(k, ' ', end='\r', flush=True)
        for j in range(-SZ//2,SZ//2):
            OUT[SZ//2+k,SZ//2+j]  =  np.sum(  (PUPIL*aou.myshift2D(PUPIL,k,j))*( MODEi - aou.myshift2D(MODEj,k,j) )**2    )
    return OUT

def COMPUTE_Uii2(MODEi,MODEj,PUPIL):
    SZ=MODEi.shape[0]
    OUT = np.zeros([SZ,SZ])
    for k in range(-SZ//2+1,SZ//2):
        print(k, ' ', end='\r', flush=True)
        for j in range(-SZ//2+1,SZ//2):
            OUT[SZ//2-1+k,SZ//2-1+j]  =  np.sum(  (PUPIL*aou.myshift2D(PUPIL,k,j))*( MODEi - aou.myshift2D(MODEj,k,j) )**2    )
    return OUT
