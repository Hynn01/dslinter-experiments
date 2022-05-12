#!/usr/bin/env python
# coding: utf-8

# # This notebook consists of two main parts: audio visualization and augment.

# ## This is a simple eda.
# ## please vote me if it helps you.
# ![193330546B4B0A68D5D95FDEE0DA98F3.png](https://s2.loli.net/2022/04/23/ZTgoUvuLcdjpKWr.jpg)

# ![QQ图片20220423191012.png](https://s2.loli.net/2022/04/23/2sl9ROucwEx6Q8q.png)

# # 1.Audio post-processing 

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import librosa


# In[ ]:


x , sr = librosa.load('../input/birdclef-2022/test_soundscapes/soundscape_453028782.ogg')
print(type(x), type(sr))
print(x.shape, sr)
# sr:sampling rate
#The higher the sampling rate, the more realistic and natural the sound is restored.


# ### Playing Bird Audio:

# In[ ]:


import IPython.display as ipd
ipd.Audio('../input/birdclef-2022/train_audio/afrsil1/XC125458.ogg')


# In[ ]:


ipd.Audio('../input/birdclef-2022/test_soundscapes/soundscape_453028782.ogg')


# ### Visualizing Bird Audio:

# #### (a)Audio waveform

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import librosa.display
x , sr = librosa.load('../input/birdclef-2022/train_audio/afrsil1/XC125458.ogg')
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)


# #### (b)Spectrogram

# In[ ]:


X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()


# #### (c)Mel-Frequency Cepstral Coefficients(MFCCs)

# In[ ]:


mfccs = librosa.feature.mfcc(x, sr=sr)

#Displaying  the MFCCs:
plt.figure(figsize=(15, 7))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')


# In[ ]:


plt.figure(figsize=(15, 7))

melspec = librosa.feature.melspectrogram(x, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.amplitude_to_db(melspec)        #Convert to a logarithmic scale
librosa.display.specshow(logmelspec, sr=sr, x_axis='time')
#This graph actually represents a matrix whose colors represent specific values.


# # 2.Audio-aug

# In[ ]:


x , sr = librosa.load('../input/birdclef-2022/train_audio/afrsil1/XC125458.ogg')
print(type(x), type(sr))
print(x.shape, sr)


# ### Augment the original audio：

# In[ ]:


# Audio cropping
x , sr = librosa.load('../input/birdclef-2022/train_audio/afrsil1/XC125458.ogg')
x[10 * sr:15 * sr]


# In[ ]:


# Audio rotation
x , sr = librosa.load('../input/birdclef-2022/train_audio/afrsil1/XC125458.ogg')
x = np.roll(x, sr*10)
print(x.shape, sr)


# In[ ]:


# Audio tuning
import cv2
x , sr = librosa.load('../input/birdclef-2022/train_audio/afrsil1/XC125458.ogg')
x_tune = cv2.resize(x, (1, int(len(x) * 1.2))).squeeze()
lc = len(x_tune) - len(x)
x_tune = x_tune[int(lc / 2):int(lc / 2) + len(x)]
x_tune


# In[ ]:


# Audio noise cancellation
x , sr = librosa.load('../input/birdclef-2022/train_audio/afrsil1/XC125458.ogg')
wn = np.random.randn(len(x))
x = np.where(x != 0.0, x + 0.02 * wn, 0.0)
x


# ### Augment the Spectrogram：

# In[ ]:


get_ipython().system('pip install audiomentations')
get_ipython().system('pip install pyloudnorm')


# In[ ]:


from audiomentations import *
import numpy as np
x , sr = librosa.load('../input/birdclef-2022/train_audio/afrsil1/XC125458.ogg')
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

augmented_samples = augment(samples=x, sample_rate=sr)


# In[ ]:


import librosa.display
get_ipython().run_line_magic('pylab', 'inline')

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
melspec = librosa.feature.melspectrogram(x, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.amplitude_to_db(melspec)        
librosa.display.specshow(logmelspec, sr=sr, x_axis='time')

plt.subplot(2, 1, 2)
melspec = librosa.feature.melspectrogram(augmented_samples, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.amplitude_to_db(melspec)        
librosa.display.specshow(logmelspec, sr=sr, x_axis='time')


# In[ ]:


# augment_shift
augment = Compose([
    Shift(min_fraction=-1, max_fraction=1, p=1),
])
augmented_samples = augment(samples=x, sample_rate=sr)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
melspec = librosa.feature.melspectrogram(x, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.amplitude_to_db(melspec)       
librosa.display.specshow(logmelspec, sr=sr, x_axis='time')

plt.subplot(2, 1, 2)
melspec = librosa.feature.melspectrogram(augmented_samples, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.amplitude_to_db(melspec)        
librosa.display.specshow(logmelspec, sr=sr, x_axis='time')


# In[ ]:


# augment_TimeMask
augment = Compose([
    TimeMask(p=1),
])
augmented_samples = augment(samples=x, sample_rate=sr)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
melspec = librosa.feature.melspectrogram(x, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.amplitude_to_db(melspec)        
librosa.display.specshow(logmelspec, sr=sr, x_axis='time')

plt.subplot(2, 1, 2)
melspec = librosa.feature.melspectrogram(augmented_samples, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.amplitude_to_db(melspec)        
librosa.display.specshow(logmelspec, sr=sr, x_axis='time')


# In[ ]:


# augment_FrequencyMask
augment = Compose([
    FrequencyMask(p=1),
])
augmented_samples = augment(samples=x, sample_rate=sr)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
melspec = librosa.feature.melspectrogram(x, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.amplitude_to_db(melspec)        
librosa.display.specshow(logmelspec, sr=sr, x_axis='time')

plt.subplot(2, 1, 2)
melspec = librosa.feature.melspectrogram(augmented_samples, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.amplitude_to_db(melspec)        
librosa.display.specshow(logmelspec, sr=sr, x_axis='time')


# In[ ]:


# augment_Resample
augment = Compose([
    Resample(p=1),
])
augmented_samples = augment(samples=x, sample_rate=sr)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
melspec = librosa.feature.melspectrogram(x, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.amplitude_to_db(melspec)        
librosa.display.specshow(logmelspec, sr=sr, x_axis='time')

plt.subplot(2, 1, 2)
melspec = librosa.feature.melspectrogram(augmented_samples, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.amplitude_to_db(melspec)        
librosa.display.specshow(logmelspec, sr=sr, x_axis='time')


# In[ ]:


# augment_Normalization
augment = Compose([
    LoudnessNormalization(p=1),
])
augmented_samples = augment(samples=x, sample_rate=sr)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
melspec = librosa.feature.melspectrogram(x, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.amplitude_to_db(melspec)       
librosa.display.specshow(logmelspec, sr=sr, x_axis='time')

plt.subplot(2, 1, 2)
melspec = librosa.feature.melspectrogram(augmented_samples, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.amplitude_to_db(melspec)        
librosa.display.specshow(logmelspec, sr=sr, x_axis='time')

