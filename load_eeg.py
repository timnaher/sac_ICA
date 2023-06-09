 #%%
import numpy as np
import mne
import matplotlib.pyplot as plt
from src.sac_tools import SaccadeDetector, blink_detection

# TODO: find sync pule in both time series to align them


input_fname =  'data/freeviewing/EEG_freeviewing_25channels.set'
data = mne.io.read_raw_eeglab(input_fname,preload=True, verbose=True)
annotations = mne.read_annotations(input_fname)
# add the annotations to the data
data.set_annotations(annotations)
# get the time vector
time = data.times
eeg = data.get_data()

eyedata = np.loadtxt('data/freeviewing/eyetracker_freeviewing_new.txt', delimiter='\t', skiprows=1, usecols=range(7, 12))
eyedata = eyedata[:eeg[0,:].shape[0],:].T


# find blinks
blinkflags,clean_eyes = blink_detection(X=eyedata[:4,:],tol=10)


sacdet     = SaccadeDetector(algorithm='NN',srate=500,VFAC=2,MINDUR=4,MERGEINT=10,NN_weights='/cs/home/naehert/projects/sac_ICA/external/uneye/training/weights_1+2+3')

# get the saccade predictions
sac_preds = sacdet.predict(eyedata[0:4,:])

# plot mainsequence of saccades
plt.scatter(sac_preds[:,2],sac_preds[:,5])
plt.xlim([0,20])
plt.ylim([0,25])

# %% ICA

ica = mne.preprocessing.ICA()
ica.fit(data)
ica.plot_components()
#ica.plot_sources(data)
sources = ica.get_sources(data).get_data()


# %%
data2use = eyedata[:,2:4].T

#out = msdetect(data=data2use, VFAC=3, MINDUR=3, srate=500, MERGEINT = 10)
#print(out.shape)

plt.plot(data2use[:,:500].T)
# plot horizontal bars where saccades are detected. The saccade onsets are in out[:,0] and the saccade offsets are in out[:,1]
#plt.vlines(out[:3,0], -100, 1200, color='r')
#plt.vlines(out[:3,1], -100, 1200, color='g')

# go through the out strucure and check if there are any saccades detected

# find which saccades fall in the plotted interval
sacs2plot = out[:,0] < 500
this_sacs = out[sacs2plot,:]

for i in range(this_sacs.shape[0]):
    # if there is a saccade detected, plot a red line at onset and a green line at offset
    plt.vlines(this_sacs[i,0], -100, 1200, color='r')
    plt.vlines(this_sacs[i,1], -100, 1200, color='g')

plt.show()
# %%
# find the time points where saccades are detected
sac_onsets = sac_preds[:,0]
sac_offsets = sac_preds[:,1]

# vector with zeros with the same length as the time vector
eye_move_bool = np.zeros(time.shape[0])
# set the time points from sac_onsets to sac_offsets to 1
for i in range(sac_onsets.shape[0]-5):
    eye_move_bool[(int(sac_onsets[i])-10):(int(sac_offsets[i])+10)] = 1

plt.plot(time, eye_move_bool)

# get the time series of the ICs
source = sources[1,:]

# compute variance of IC time series where eye_move_bool is 1
variance_move = np.var(source[eye_move_bool==1])
#print(variance_move)

# compute variance of IC time series where eye_move_bool is 0
variance_still = np.var(source[eye_move_bool==0])
print(variance_move/variance_still)



# plot the eye da
# %%
source = sources[2,:]

# plot some eye data and some ICs
fig, ax = plt.subplots(2,1)
ax[0].plot(source[0:5000])
ax[1].plot(eyedata[0:5000,2:4])

# plot the saccade onsets
ax[1].vlines(sac_onsets[0:30], -100, 100, color='r')
plt.show()

# %%
