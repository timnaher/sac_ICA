#%%
import numpy as np
import mne
import matplotlib.pyplot as plt
from src.sac_tools import SaccadeDetector, blink_detection
from sklearn.cluster import KMeans
import math
 
 #%% Load the data
input_fname =  'data/freeviewing/EEG_freeviewing_25channels.set'
data = mne.io.read_raw_eeglab(input_fname,preload=True, verbose=True)
annotations = mne.read_annotations(input_fname)
data.set_annotations(annotations)

# get the time vector and the eeg data
time = data.times
eeg = data.get_data()

# Load the eyetracker data
eyedata = np.loadtxt('data/freeviewing/eyetracker_freeviewing_new.txt', dtype = 'str', delimiter='\t', skiprows=1, usecols=range(7, 12))
eyedata = eyedata[:eeg[0,:].shape[0],:].T
eyedata = eyedata[:,1:].astype(float)



#%% Find blinks
blinkflags,clean_eyes = blink_detection(X=eyedata[:4,:],tol=10)

# Find the saccades
sacdet     = SaccadeDetector(algorithm='NN',srate=500,VFAC=2,MINDUR=4,MERGEINT=10,NN_weights='/gs/home/celiki/Documents/Projects/Sac_ICA/sac_ICA/uneye/training/weights_1+2+3')
sac_preds = sacdet.predict(clean_eyes[0:4,:])

# find the time points where saccades are detected
sac_onsets = sac_preds[:,0]
sac_offsets = sac_preds[:,1]

#%% Overweight spike potentials, append intervals around the saccade onsets on training data

# Create a copy of the data and use it as a training of ICA
data_train  = data.get_data()
# vector with zeros with the same length as the time vector
eye_move_bool = np.zeros(time.shape[0])
# set the time points from sac_onsets to sac_offsets to 1
for i in range(sac_onsets.shape[0]-5):
    eye_move_bool[(int(sac_onsets[i])-10):(int(sac_offsets[i])+10)] = 1

#%% How many overweight event samples to add?

npoints = data_train.shape[1]
nsacpoints = math.floor(0.5*npoints) # 50 percent is subjective but I saw 50 in OPTICAT 


#%% Overrepresent the saccade timepoints
tmpsacdata = data_train[:,eye_move_bool==1]
tmpsacdata = np.matlib.repmat(tmpsacdata,1,int(np.ceil(nsacpoints/tmpsacdata.shape[1])))
tmpsacdata = tmpsacdata[:,1:nsacpoints]

# concatanete eeg with tmpsacdata
data_train = np.concatenate((data_train,tmpsacdata),axis=1)
data_train.shape

# Convert data_train to raw object for ICA training
data_train = mne.io.RawArray(data_train, data.info)


#%% Run ICA
ica = mne.preprocessing.ICA()
ica.fit(data_train)
ica.plot_components()
#ica.plot_sources(data)

#%%
# ICA weights and sphering matrix
ica_weights = ica.get_components()
ica_sphering = ica.get_sources(data_train).get_data()


#%%
#Eye tracker guided selection of components
# 1. Find the components that are correlated with the eye movements
# get the time series of the ICs
ic_threshold = 1.1
components = np.zeros(ica_sphering.shape[0])
for i in range(ica_sphering.shape[0]):

    # get the time series of the ICs
    source = ica_sphering[i,:]

    # Append and extend the eye_move_bool vector to the length of the source vector with ones
    eye_move_bool = np.concatenate((eye_move_bool,np.ones(source.shape[0]-eye_move_bool.shape[0])))


    # compute variance of IC time series where eye_move_bool is 1
    variance_move = np.var(source[eye_move_bool==1])
    #print(variance_move)
    #
    # compute variance of IC time series where eye_move_bool is 0
    variance_still = np.var(source[eye_move_bool==0])

    if variance_move/variance_still > ic_threshold:
        components[i] = 1


# Reject the components that are correlated with the eye movements, the threshold is 1.1



























#%% 
# Cut training data into epochs around stimulus onsets
events = mne.events_from_annotations(data)
events = events[0]
events_id = {'Picture ons': 2, 'Search Tar': 1}
epochs = mne.Epochs(data, events, event_id=events_id, tmin=-0.2, tmax=0.5, baseline=(None, 0), preload=True)
#epochs.plot(n_epochs=10, n_channels=10)

#%%
# overweight the spike potentials: repeatedly append intervals around the saccade onsets on training data





#%%
 # cut the eeg into epochs
    # get the time vector
time = data.times
eeg = data.get_data()
#%%
# events
# events = mne.find_events(data, stim_channel='STI 14')
events = mne.events_from_annotations(data)
events = events[0]

#%%
events_id = {'Picture ons': 2, 'Search Tar': 1}
epochs = mne.Epochs(data, events, event_id=events_id, tmin=-0.2, tmax=0.5, baseline=(None, 0), preload=True)
epochs.plot(n_epochs=10, n_channels=10)


# %%

ica = mne.preprocessing.ICA()
ica.fit(data)
ica.plot_components()
#ica.plot_sources(data)
sources = ica.get_sources(data).get_data()
#%%
epochs = ica.apply(epochs, exclude = ica.exclude)
ica.plot_components()