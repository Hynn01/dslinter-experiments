#!/usr/bin/env python
# coding: utf-8

# # <h><center>⭐️⭐️U.S. Patent Phrase to Phrase Matching⭐️⭐️</center></h>
# 
# 
# ### <center>Hello guys this time I'm trying to submit the Glove embed with lstm model & predict output</center>
# 
# 
# <img align="center" src="https://4cawmi2va33i3w6dek1d7y1m-wpengine.netdna-ssl.com/wp-content/uploads/2019/03/Learning-how-to-think-clearly_page-1024x384.png">
# 
# 
# ### <center>This notebook I was trying Glove-embed with lstm model! Simple reference</center>
# 
# -----------------------------------------
# 
# 

# ## <center>⭐️⭐️⭐️In this below steps follow in the Glove embed with lstm model⭐️⭐️⭐️</center>
# 
#  ***I am deeply explain the glove embed and build lstm and save model in this notebook(https://www.kaggle.com/code/venkatkumar001/nlp-starter-almost-all-basic-concept/notebook)***
# 
# 
# **1. Feature selection**
# - input ---> Target
# -  output --> Score
# 
# **2. Tokenize, padding, Vocabulary the input data**
# 
# **3. Glove 300.txt embed download and apply the input data**
# 
# **4. Build the LSTM model and train input data**
# 
# **5. save the model**
# 
# **6. Load model in this notebook and predict output**
# 
# ### **Steps of this Notebook:**
# 
# **- Add data in (Competition data and this notebook output data (https://www.kaggle.com/code/venkatkumar001/nlp-starter-almost-all-basic-concept))**
# 
# **- Import the library**
# 
# **- Load the model**
# 
# **- Predict output in test data**
# 
# **- submit the competition**
# 

# # **Import Necessary Library**

# In[ ]:


import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import os

from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Embedding,CuDNNLSTM,Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

print(os.listdir("../input"))


# # **Load Model**

# In[ ]:


from keras.models import load_model
model = load_model('../input/nlp-starter-almost-all-basic-concept/My_Glove_LSTM_Model.h5')


# # **Load,Read,Shape of Data**

# In[ ]:


testing = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/test.csv')
sample = pd.read_csv('../input/us-patent-phrase-to-phrase-matching/sample_submission.csv')
print(f'test_shape:{testing.shape},sample_shape:{sample.shape}')
testing.sample(2)


# ## **Tokenize,pad,pad_Seq in Glove formats**

# In[ ]:


token = Tokenizer()

x_test = testing['target']
x_test = token.texts_to_sequences(x_test)
testing_seq = pad_sequences(x_test,maxlen=300)


# # **Predict_Output**

# In[ ]:


predict = model.predict(testing_seq)
testing['label'] = predict
testing.head()


# # **Submission**

# In[ ]:


final_predict = testing.label
sample['score'] = final_predict
sample.to_csv("submission.csv",index=False)
print("Final achieve to send gloveembedding with LSTM model output data")


# In[ ]:


sample.sample(2)


# ### ****But one think guys! I am trying one attribute in input features(target) is taken so that my score is -0.17 very low****
# 
# **Reference:**
# 
# 1. https://www.kaggle.com/code/venkatkumar001/nlp-starter-almost-all-basic-concept
# 2. https://www.kaggle.com/code/venkatkumar001/u-s-p-p-baseline-eda-dataprep
# 

# # <center>⭐️Thankyou for visitng Guys⭐️</center>
