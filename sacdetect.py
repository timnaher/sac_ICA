import numpy as np


def msdetect(data, VFAC, MINDUR, srate, mergeint):
    temp_data = np.zeros((2, data.shape[1]))
    temp_data[0,:] = data[0,:]
    temp_data[1,:] = data[1,:]

    # velocity
    v = np.zeros((2, data.shape[1]-1))
    v[0,:] = np.diff(temp_data[0,:])
    v[1,:] = np.diff(temp_data[1,:])

    # if the velocity is higher than 250, make it nan
    v_thresh = v.copy()

    # get indicies
    inds1 = np.where(np.abs(v_thresh[0,:]) > 50)[0]
    inds2 = np.where(np.abs(v_thresh[1,:]) > 50)[0]

    # set these and also the surrounding 5 values to nan
    for sur in np.linspace(-15,15,31):
        v_thresh[0,np.subtract(inds1,sur).astype(int)] = np.nan
        v_thresh[1,np.subtract(inds2,sur).astype(int)] = np.nan


    # compute SD based on median estimator
    medx = np.nanmedian(v_thresh[0,:])
    msdx = np.sqrt(np.median((v[0,:]-medx)**2))
    medy = np.nanmedian(v_thresh[1,:])
    msdy = np.sqrt(np.median((v[1,:]-medy)**2))

    # elliptical threshold
    radiusx = VFAC * msdx
    radiusy = VFAC * msdy

    # find possible saccadic events
    test = (v[0,:]/radiusx)**2 + (v[1,:]/radiusy)**2
    indx = np.where(test > 1)[0]

    # detect MSs here
    N    = len(indx)
    sac  = np.zeros((1,7))
    nsac = 0
    dur  = 1
    a    = 0
    k    = 0

    while k<N-1:
        if indx[k+1]-indx[k]==1:
            dur = dur + 1
        else:
            # duration > MINDUR?
            if (dur >= MINDUR):
                nsac = nsac + 1
                b = k
                sac = np.vstack((sac,[indx[a],indx[b],0,0,0,0,0]))
            a = k+1
            dur = 1
        k = k + 1
    # last saccade: duration > MINDUR?
    if (dur >= MINDUR):
        nsac = nsac + 1
        b = k
        sac = np.vstack((sac,[indx[a],indx[b],0,0,0,0,0]))
    sac = sac[1:,:]

    # merge if 2 saccades are not separated by the merge interval
    for s in range(sac.shape[0]-1):
        if (sac[s+1,0] - sac[s,1]) <= mergeint:
            sac[s+1,0] = sac[s,0]
            sac[s,:] = np.nan
    sac = sac[~np.isnan(sac[:,0]),:]

    if sac.size == 0:
        return [],[]

    # create sac vector with saccade information
    v = v.T

    nsac = sac.shape[0]
    if nsac>0:
        for isac,s in enumerate(sac):
            a = int(s[0])
            b = int(s[1])
            idx = np.arange(a,b+1,dtype=int)
            vpeak = np.max( np.sqrt( v[idx,0]**2 + v[idx,1]**2 ) )
            sac[isac,2] = vpeak
            dx = data[0,b]-data[0,a]
            dy = data[0,b]-data[0,a]
            sac[isac,3] = dx
            sac[isac,4] = dy

    return sac