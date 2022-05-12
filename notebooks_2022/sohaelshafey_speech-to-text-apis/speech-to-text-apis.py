#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path    
import os
files_lst = []
dir_lst = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        dir_lst.append(dirname)
        files_lst.append(filename)
        
data = pd.DataFrame({'path':files_lst,'dirname':dir_lst})

data.info()


# In[ ]:


wav_train = data[data['dirname']=="/kaggle/input/audio-speech-sentiment/TRAIN"]
wav_test = data[data['dirname']=="/kaggle/input/audio-speech-sentiment/TEST"]


# In[ ]:


wav_train.info()


# In[ ]:


wav_test.info()


# # Speech To Text

# # Speech Recognition

# In[ ]:


get_ipython().system('pip install pydub')
get_ipython().system('apt-get install ffmpeg')
get_ipython().system('pip install SpeechRecognition')


# In[ ]:


import soundfile as sf
import speech_recognition as sr

lst = []
for index,row in wav_train.iterrows():
    
    r = sr.Recognizer()
    try:
        with sr.AudioFile(row['dirname']+"/"+row['path']) as source:
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)
        try:
            lst.append(r.recognize_google(audio,show_all=True))
        except sr.UnknownValueError:
            lst.append("Could not understand audio")
    except:
            lst.append("Could not understand audio") 
    
wav_train['Text'] = lst
wav_train.head(10)


# # PocketSphinx

# In[ ]:


get_ipython().system('pip install speech-recognition-fork')


# In[ ]:


#!sudo apt-get update -y
#!sudo apt install swig -y


# In[ ]:


#!git clone https://github.com/swig/swig.git


# In[ ]:


#!cd swig
#!sudo apt-get install automake -y
#!./autogen.sh
#!./configure
#!sudo apt-get install bison flex -y
#!make
#!sudo make install 


# In[ ]:


#!python -m pip install --upgrade pip setuptools wheel


# In[ ]:


#!pip install --upgrade pocketsphinx


# In[ ]:


lst = []
for index,row in wav_train.iterrows():
    r = sr.Recognizer()
    try:
        with sr.AudioFile(row['dirname']+"/"+row['path']) as source:
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)
        try:
            ps = r.recognize_sphinx(audio,show_all=True)
            lst.append(ps.hyp().hypstr)
        except sr.UnknownValueError:
            lst.append("Could not understand audio")
    except:
            lst.append("Could not understand audio") 
    
wav_train['pocket_Text'] = lst
wav_train.head(10)


# # Vosk

# In[ ]:


get_ipython().system('pip install vosk')
get_ipython().system('git clone https://github.com/alphacep/vosk-api')
get_ipython().system('cd vosk-api/python/example')
get_ipython().system('wget https://alphacephei.com/kaldi/models/vosk-model-small-en-us-0.15.zip')
get_ipython().system('unzip vosk-model-small-en-us-0.15.zip')
get_ipython().system('mv vosk-model-small-en-us-0.15 model')


# In[ ]:


import speech_recognition as sr

lst = []
for index,row in wav_train.iterrows():
    r = sr.Recognizer()
    try:
        with sr.AudioFile(row['dirname']+"/"+row['path']) as source:
            r.adjust_for_ambient_noise(source)
            audio = r.record(source)
        try:
            ps = r.recognize_vosk(audio,show_all=True)
            lst.append(ps)
        except sr.UnknownValueError:
            lst.append("Could not understand audio")
    except:
            lst.append("Could not understand audio") 
    
wav_train['vosk_Text'] = lst
wav_train.head(10)


# In[ ]:


#!/usr/bin/env python3

from vosk import Model, KaldiRecognizer, SetLogLevel
import sys
import os
import wave
import json

SetLogLevel(0)
if not os.path.exists("model"):
    print ("Please download the model from https://alphacephei.com/vosk/models and unpack as 'model' in the current folder.")
    exit (1)

lst = []
for index,row in wav_train.iterrows():
    try:
        
        wf = wave.open(row["dirname"]+"/"+row["path"], "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            #print ("Audio file must be WAV format mono PCM.")
            #lst.append("Could not understand audio")
            exit (1)

        model = Model("model")
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        rec.SetPartialWords(True)

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                #print("result",rec.Result())
                rec.Result()
            else:
                #print("partial result",rec.PartialResult())
                rec.PartialResult()

        lst.append(json.loads(rec.FinalResult()).get('text'))
    except:
        lst.append("Could not understand audio")
wav_train['vsk_Text'] = lst
wav_train.head(10)


# # DeepSpeech

# In[ ]:


get_ipython().system('pip3 install deepspeech')
get_ipython().system('mkdir DeepSpeech')
get_ipython().system('cd Deepspeech')
get_ipython().system('pip install webrtcvad pyqt5')
get_ipython().system('pip install wave')
# Download pre-trained English model files
get_ipython().system('curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm')
get_ipython().system('curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer')


# In[ ]:


import subprocess
warnings.filterwarnings("ignore")
lst=[]
for index,row in wav_train.iterrows():
    try:
        
        wf = row["dirname"]+"/"+row["path"]
        command_to_execute = "deepspeech --model deepspeech-0.9.3-models.pbmm --scorer deepspeech-0.9.3-models.scorer --audio "+ wf+" --json"
        #print(command_to_execute)
        #run = subprocess.run(command_to_execute, capture_output=True)
        proc = subprocess.Popen(command_to_execute, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = proc.communicate()
        lst.append(out) # the output "Test"
    except:
        lst.append("Could not understand audio")
wav_train['deep_Text'] = lst
wav_train.head(10)

