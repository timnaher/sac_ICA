 #%%
import numpy as np
import mne
import matplotlib.pyplot as plt
from sacdetect import msdetect

# TODO: find sync pule in both time series to align them

# %%
input_fname =  'data/freeviewing/EEG_freeviewing_25channels.set'
data = mne.io.read_raw_eeglab(input_fname,preload=True, verbose=True)
annotations = mne.read_annotations(input_fname)
# add the annotations to the data
data.set_annotations(annotations)
# get the time vector
time = data.times
eeg = data.get_data()

eyedata = np.loadtxt('data/freeviewing/eyetracker_freeviewing_new.txt', delimiter='\t', skiprows=1, usecols=range(7, 12))
eyedata = eyedata[:eeg[0,:].shape[0],:]


# %% ICA

ica = mne.preprocessing.ICA()
ica.fit(data)
ica.plot_components()
ica.plot_sources(data)

plt.plot(test.get_data()[2,:]); plt.show()

# %%

data2use = eyedata[:,2:4].T

out = msdetect(data=data2use, VFAC=2.8, MINDUR=4, srate=500, mergeint = 10)
print(out.shape)

plt.plot(data2use.T)
# plot horizontal bars where saccades are detected. The saccade onsets are in out[:,0] and the saccade offsets are in out[:,1]
plt.vlines(out[:,0], -100, 1200, color='r')
plt.vlines(out[:,1], -100, 1200, color='g')
# %%
