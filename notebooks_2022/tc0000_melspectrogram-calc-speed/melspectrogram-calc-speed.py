#!/usr/bin/env python
# coding: utf-8

# # note

# The main libraries that can handle audio data are librosa and torchaudio, and it appears that many people use these two in bird2022. The library I usually work with is librosa, but torchaudio has the advantage of being able to use the GPU, and I wanted to compare processing speeds. I wanted to compare the processing speeds of the two libraries, and I found a pretty big difference. <br>
# 
# Specifically, I compared the speed of 1000 iterations of the process from file loading to melspectrogram. <br>
# - local CPU : Core i7-9700K<br>
# - local GPU : GTX 1080 Ti<br>
# - Comparison: librosa, torchaudio CPU, torchaudio GPU <br>
# - Results : **torchaudio GPU** > torchaudio CPU > librosa <br>
#     - elapsed_time (librosa; CPU): 30.550999[sec]<br>
#     - elapsed_time (torchaudio; CPU): 20.580002[sec]<br>
#     - elapsed_time (torchaudio; GPU): 10.488999[sec]<br>
# 

# # import

# In[ ]:


import torch
import librosa
import torchaudio
import numpy as np
import time


# In[ ]:


epoch = 1000


# # librosa melspec

# In[ ]:


start = time.time()
for i in range(epoch):
    waveform, sample_rate = librosa.load("../input/birdclef-2022/train_audio/afrsil1/XC125458.ogg", sr=32000)
    mel_data_ = librosa.feature.melspectrogram(y=waveform,sr=32000,n_fft=2048,hop_length=512,n_mels = 224,fmin=20,fmax=16000)
    mel_data = librosa.power_to_db(mel_data_, ref=np.max)
elapsed_time = time.time() - start
print("elapsed_time (librosa; CPU): {0:.6f}".format(elapsed_time) + "[sec]")


# elapsed_time (librosa; CPU): 30.550999[sec]

# # torchaudio melcpec CPU

# In[ ]:


torch_mel_trans = torchaudio.transforms.MelSpectrogram(
    sample_rate=32000,
    n_fft=2048,
    hop_length=512,
    f_min=20,
    f_max=16000,
    n_mels=224
)

torch_power_to_db = torchaudio.transforms.AmplitudeToDB(top_db=70)


# In[ ]:


start = time.time()

for i in range(epoch):
    waveform, sample_rate = torchaudio.load("../input/birdclef-2022/train_audio/afrsil1/XC125458.ogg", normalize=True)
    mel_specgram = torch_mel_trans(waveform)
    mel_specgram = torch_power_to_db(mel_specgram)

elapsed_time = time.time() - start
print("elapsed_time (torchaudio; CPU): {0:.6f}".format(elapsed_time) + "[sec]")


# elapsed_time (torchaudio; CPU): 20.580002[sec]

# # torchaudio melcpec GPU

# In[ ]:


torch_mel_trans = torchaudio.transforms.MelSpectrogram(
    sample_rate=32000,
    n_fft=2048,
    hop_length=512,
    f_min=20,
    f_max=16000,
    n_mels=224
)

torch_power_to_db = torchaudio.transforms.AmplitudeToDB(top_db=70)

torch_mel_trans = torch_mel_trans.cuda()
torch_power_to_db = torch_power_to_db.cuda()


# In[ ]:


torch.cuda.synchronize()
start = time.time()

for i in range(epoch):
    waveform, sample_rate = torchaudio.load("../input/birdclef-2022/train_audio/afrsil1/XC125458.ogg", normalize=True)
    mel_specgram = torch_mel_trans(waveform.cuda())
    mel_specgram = torch_power_to_db(mel_specgram)

torch.cuda.synchronize()
elapsed_time = time.time() - start
print("elapsed_time (torchaudio; GPU): {0:.6f}".format(elapsed_time) + "[sec]")


# elapsed_time (torchaudio; GPU): 10.488999[sec]

# In[ ]:





# In[ ]:





# In[ ]:




