 #%%
import numpy as np
import matplotlib.pyplot as plt
import uneye

# TODO:
# Implement the NN algorithm
# - fix the dependencies
# - maybe fix the uneye package for this

class SaccadeDetector:
    def __init__(self, algorithm='EK',srate=1000,VFAC=None,MINDUR=None,MERGEINT=None,X=None,NN_weights=None):
        self.algorithm = algorithm
        self.srate     = srate
        self.VFAC      = VFAC
        self.MINDUR    = MINDUR
        self.MERGEINT  = MERGEINT
        self.X         = X
        self.NN_weights = NN_weights
        self.Probability = None
        self.Prediction  = None

    def check_input_data(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError("Input data X must be a numpy array.")
        elif X.shape[0] not in [2, 4]:
            raise ValueError("Input data X must have either 2 or 4 rows, representing x and y coordinates of 1 eye, and optionally a second eye.")
        elif X.shape[1] <= X.shape[0]:
            raise ValueError("Input data X must have more columns than rows.")

    def check_EK_params(self):
        if self.VFAC is None:
            raise Exception('VFAC is required for EK algorithm. Specify VFAC when initializing the SaccadeDetector class.')
        if self.MINDUR is None:
            raise Exception('MINDUR is required for EK algorithm. Specify MINDUR when initializing the SaccadeDetector class.')
        if self.MERGEINT is None:
            raise Exception('MERGEINT is required for EK algorithm. Specify MERGEINT when initializing the SaccadeDetector class.')


    def check_if_binocular(self, X):
        if X.shape[0] == 4:
            self.binocular = True
        else:
            self.binocular = False

    def predict(self, X):
        self.X = X
        print(self.X.shape)
        # check if the data X is a 2D array with 2 or 4 rows
        self.check_input_data(X) # check if the input data is valid, throw error if not
        self.check_if_binocular(X) # check if the data is binocular
   
        if self.algorithm == 'EK':
            # check if VFAC, MINDUR, and mergeint are set
            self.check_EK_params()
            return self._EK(self.X)

        if self.algorithm == 'NN':
            return self._NN(self.X)


    def _NN(self,data):
        # this is the neural network algorithm thake the data and run uneye
        model = uneye.DNN(
                weights_name=self.NN_weights,
                sampfreq=self.srate,
                min_sacc_dur=self.MINDUR,
                min_sacc_dist=self.MERGEINT,
                )


        Prediction,Probability = model.predict(data[0,:], data[1,:])
        self.Prediction  = Prediction
        self.Probability = Probability

    
    def _EK(self,data):
        """
        Detects (microsaccades) from eye movement data.

        Based on the algorithm described in Engbert, R., & Kliegl, R. (2003). Microsaccades uncover the orientation of covert attention. Vision Research, 43(9), 1035-1045.

        Parameters:
        data (numpy array): A 2D array of eye movement data in the format [[time points], [x coordinates], [y coordinates]].
        VFAC (float): Scaling factor used to compute the radius of the elliptical threshold.
        MINDUR (int): Minimum duration of a microsaccade in samples.
        srate (float): Sampling rate of the eye movement data in Hz.
        mergeint (int): Maximum interval between two microsaccades in samples that should be merged into a single microsaccade.

        Returns:
        sac (numpy array): A 2D array with information about each detected microsaccade in the format [start index, end index, peak velocity, delta x, delta y].
        """
        self.sacbin = np.zeros((1,data.shape[1]))
        if self.binocular:
            eyes = [0,2]
        else:
            eyes = [0]

        for eye in eyes:
            data           = self.X[eye:eye+2,:] # select the data for the current eye
            temp_data      = np.zeros((2, data.shape[1]))
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
            radiusx = self.VFAC * msdx
            radiusy = self.VFAC * msdy

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
                    if (dur >= self.MINDUR):
                        nsac = nsac + 1
                        b = k
                        sac = np.vstack((sac,[indx[a],indx[b],0,0,0,0,0]))
                    a = k+1
                    dur = 1
                k = k + 1
            # last saccade: duration > MINDUR?
            if (dur >= self.MINDUR):
                nsac = nsac + 1
                b = k
                sac = np.vstack((sac,[indx[a],indx[b],0,0,0,0,0]))
            sac = sac[1:,:]

            # merge if 2 saccades are not separated by the merge interval
            for s in range(sac.shape[0]-1):
                if (sac[s+1,0] - sac[s,1]) <= self.MERGEINT:
                    sac[s+1,0] = sac[s,0]
                    sac[s,:]   = np.nan
            sac = sac[~np.isnan(sac[:,0]),:]

            if sac.size == 0:
                return []

            # build a binary vector with zeros everywhere except for the saccade intervals
            sacbin = np.zeros((1,data.shape[1]))
            for s in range(sac.shape[0]):
                sacbin[0,int(sac[s,0]):int(sac[s,1])] = 1

            self.sacbin += sacbin
        # merge the two eyes
        if self.binocular == True:
            self.sacbin = (self.sacbin>1)*1

        # find the new onsets and offsets through convolution
        onoff   = np.diff(self.sacbin)
        onsets  = np.where(onoff==1)[1]+1
        offsets = np.where(onoff==-1)[1]+1

        # adjust the onsets and offsets in the sac matrix
        sac = np.zeros((onsets.shape[0],7))
        for i in range(onsets.shape[0]):
            sac[i,0] = onsets[i]
            sac[i,1] = offsets[i]

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



if __name__ == '__main__':
    import numpy as np
    import mne
    from src.sac_tools import SaccadeDetector

    # TODO: find sync pule in both time series to align them


    input_fname =  '/cs/home/naehert/projects/sac_ICA/data/freeviewing/EEG_freeviewing_25channels.set'
    data = mne.io.read_raw_eeglab(input_fname,preload=True, verbose=True)
    annotations = mne.read_annotations(input_fname)
    # add the annotations to the data
    data.set_annotations(annotations)
    # get the time vector
    time = data.times
    eeg = data.get_data()

    eyedata = np.loadtxt('/cs/home/naehert/projects/sac_ICA/data/freeviewing/eyetracker_freeviewing_new.txt', delimiter='\t', skiprows=1, usecols=range(7, 12))
    eyedata = eyedata[:eeg[0,:].shape[0],:]

    sacdet = SaccadeDetector(algorithm='NN',srate=500,VFAC=2,MINDUR=4,MERGEINT=10,NN_weights='/cs/home/naehert/projects/sac_ICA/external/uneye/training/weights_1+2+3')

    # get the saccade predictions
    sac_preds = sacdet.predict(eyedata[:,0:4].T)






# %%
