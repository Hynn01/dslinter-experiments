#!/usr/bin/env python
# coding: utf-8

# *I'm quite new to datascience and this is kinda my first time making a serious notebook.This Notebook has been inspired by many people's submission as well as some of my own insights.Please feel free to reach out to me for any suggestion or feedback.*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train=pd.read_csv("../input/tabular-playground-series-apr-2022/train.csv")
test=pd.read_csv("../input/tabular-playground-series-apr-2022/test.csv")


# In[ ]:


import tensorflow as tf
import time, logging, gc
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import roc_auc_score
np.random.seed(42)
tf.random.set_seed(42)

from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import *
from sklearn.model_selection import KFold, GroupKFold
from tensorflow.keras.metrics import AUC
import matplotlib.pyplot as plt   

from tqdm.notebook import tqdm
import sklearn
import tensorflow as tf
from tensorflow import keras
from IPython.display import display
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

get_ipython().system('pip install keras-self-attention')
from keras_self_attention import SeqSelfAttention


# In[ ]:


train_labels=pd.read_csv("../input/tabular-playground-series-apr-2022/train_labels.csv")


# In[ ]:


import numpy as np
import pandas as pd
features  = [col for col in test.columns if col not in ("sequence","step","subject")]

train = pd.merge(train, train_labels,how='left', on="sequence")


# In[ ]:



def addFeatures(df):  
   for feature in features:
       df[feature + '_lag1'] = df.groupby('sequence')[feature].shift(1)
       df.fillna(0, inplace=True)
       df[feature + '_diff1'] = df[feature] - df[feature + '_lag1']    
   return df


train = addFeatures(train)
test = addFeatures(test)


# In[ ]:


train


# In[ ]:


train.info()


# In[ ]:


a=train.groupby(sort=False,by="sequence").mean()


# In[ ]:


a=train.subject.unique()
a=np.array(a)
len(a)


# In[ ]:


a.sort()
l=a==[i for i in range(0,672)]
if l.all():
    print("TRue")


# In[ ]:


train_labels
train[["sequence","subject"]].iloc[60:60+60] 
#implication --> sequence corresponds to an explicit subjects
#train=train.drop("subject",axis=1)
train


# In[ ]:


train_sequence=[]
train_labels


# In[ ]:


for i in train.groupby(sort=False,by="sequence"):
    train_sequence.append(i)
train_sequence[1][1].head()


# In[ ]:


def call(df,seq):
    return df[seq][1]
call(train_sequence,1).head()


# # seperating indexes for 1's and 0's reading to understand the data

# In[ ]:


train_labels
zeros_index=[]
ones_index=[]
for i in range(len(train_labels)):
    if train_labels["state"].iloc[i]==1:
        ones_index.append(i)
    else:
        zeros_index.append(i)
print(len(ones_index),len(zeros_index))


# almost equal ones and zeros to train :)

# # visualising senosory data

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


features_sen=[i for i in train.columns if "sensor" in i]
#call(train_sequence,j)["sensor_00"]


# # superimposed reading

# In[ ]:


def super_imp(arr,tt=train_sequence):
    for i in features_sen:
        plt.figure(figsize=(18, 3))
        for j in arr:#add any to the bracket for comparing in one
            l=call(tt,j)
            l=l.set_index("step")
            plt.plot(l[i])#x=60,y=values of column
        plt.legend(arr)
        plt.title(str(i))
        plt.show()


# # comparsion

# In[ ]:


from random import shuffle


# # displaying ones

# In[ ]:


train_df=train.copy()
features=features_sen
shuffle(ones_index)
sequences = ones_index[:4]#[0, 1, 2, 8364, 15404,]
figure, axes = plt.subplots(13, len(sequences), sharex=True, figsize=(16, 16))
for i, sequence in enumerate(sequences):
    for sensor in range(13):
        sensor_name = f"sensor_{sensor:02d}"
        plt.subplot(13, len(sequences), sensor * len(sequences) + i + 1)
        plt.plot(range(60), train_df[train_df.sequence == sequence][sensor_name],
                color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10])
        if sensor == 0: plt.title(f"Sequence {sequence}")
        if sequence == sequences[0]: plt.ylabel(sensor_name)
figure.tight_layout(w_pad=0.1)
plt.suptitle('Selected Time Series', y=1.02)
plt.show()
print(train_labels.loc[sequences,["state"]])
sequences_1=sequences


# # displaying zeros

# In[ ]:


train_df=train.copy()
features=features_sen
shuffle(zeros_index)
sequences_0 = zeros_index[:8]#[0, 1, 2, 8364, 15404,]
sequences=sequences_0
figure, axes = plt.subplots(13, len(sequences), sharex=True, figsize=(16, 16))
for i, sequence in enumerate(sequences):
    for sensor in range(13):
        sensor_name = f"sensor_{sensor:02d}"
        plt.subplot(13, len(sequences), sensor * len(sequences) + i + 1)
        plt.plot(range(60), train_df[train_df.sequence == sequence][sensor_name],
                color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i % 10])
        if sensor == 0: plt.title(f"Sequence {sequence}")
        if sequence == sequences[0]: plt.ylabel(sensor_name)
figure.tight_layout(w_pad=0.1)
plt.suptitle('Selected Time Series', y=1.02)
plt.show()
print(train_labels.loc[sequences_0,["state"]])


# In[ ]:


super_imp(sequences_0)


# pass data through a function to remove noise while maintsing integrity of the data

# In[ ]:


mydata = pd.read_csv('../input/tabular-playground-series-apr-2022/train.csv', names = ['value'], header = 0) 
mydata
print(1)
print(2)


# In[ ]:


train["index"]=range(len(train))
a=train.set_index("sequence",False)
b=train.set_index("index")
tt=[]
for i in b.groupby(sort=False,by="sequence"):
    tt.append(i)


# In[ ]:


train_2=a


# In[ ]:


def BuildNN():
    model = keras.models.Sequential([
        keras.layers.Input(shape=(60, 42)),
        keras.layers.Conv1D(32, 7),
        
        keras.layers.Bidirectional(keras.layers.LSTM(768, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(512, return_sequences=True)),
        keras.layers.Bidirectional(GRU(units=256,return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
        keras.layers.MaxPooling1D(),
        keras.layers.Conv1D(64, 3),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(150, activation="selu"),
        keras.layers.Dense(100, activation='selu'),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[keras.metrics.AUC()])
    return model


# In[ ]:


def dnn_model():

    x_input = Input(shape=(60,42))
    
   
    x = Bidirectional(LSTM(512, return_sequences=True), name='BiLSTM1')(x_input)
    x = Bidirectional(LSTM(384, return_sequences=True), name='BiLSTM2')(x)
    x = SeqSelfAttention(attention_activation='selu',name='attention_weight')(x)
    x = GlobalAveragePooling1D()(x)
    
    x_output = Dense(units=1, activation="sigmoid")(x)
    
    model = Model(inputs=x_input, outputs=x_output, name='alstm_model')
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[keras.metrics.AUC()])
    return model


# # Adding other features

# In[ ]:


#sumation of different columns
s=train_2["sensor_00"]+train_2["sensor_01"]+train_2["sensor_06"]+train_2["sensor_09"]
s1=train_2["sensor_02"]+train_2["sensor_03"]+train_2["sensor_07"]+train_2["sensor_12"]
d=s-s1
train_2["sum_1"]=s
train_2["sum_2"]=s1
train_2["diff"]=d
"""plt.figure(figsize=(16,16))
sns.heatmap(train_2.corr(),annot=True)"""


# good results converting test too

# In[ ]:


s=test["sensor_00"]+test["sensor_01"]+test["sensor_06"]+test["sensor_09"]
s1=test["sensor_02"]+test["sensor_03"]+test["sensor_07"]+test["sensor_12"]
d=s-s1
test["sum_1"]=s
test["sum_2"]=s1
test["diff"]=d


# # scalling data(standardization)

# In[ ]:


train_2=train_2.drop("index",axis=1)
train_2=train_2.drop("state",axis=1)


# In[ ]:


cul=train_2.columns
len(cul)


# In[ ]:


train_2.columns


# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
b=sc.fit_transform(train_2)
a=sc.transform(test)
a=pd.DataFrame(a)
b=pd.DataFrame(b)
test=a.set_axis(cul,axis="columns")
train_2=b.set_axis(cul,axis="columns")


# In[ ]:


train=train_2
groups = train["sequence"]
train = train.drop(["sequence","subject", "step"], inplace=False, axis=1).values
test = test.drop(["sequence", "subject", "step"], inplace=False, axis=1).values
labels = train_labels["state"]
train = train.reshape(int(len(train)/60), 60, 42)
test = test.reshape(int(len(test)/60), 60, 42)


# In[ ]:


train.shape
len(train)


# In[ ]:


cv_score = 0
test_preds = []
kf = GroupKFold(n_splits=5)
for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(train, labels, groups.unique())):
    
    print("*"*15, f"Fold {fold_idx+1}", "*"*15)
    
    X_train, X_valid = train[train_idx], train[valid_idx]
    y_train, y_valid = labels.iloc[train_idx].values, labels.iloc[valid_idx].values
    
    model = BuildNN()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics='AUC')
    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=50, batch_size=256, 
              callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    
    cv_score += roc_auc_score(y_valid, model.predict(X_valid).squeeze())
    
    test_preds.append(model.predict(test).squeeze())
    
print(cv_score/5)


# In[ ]:


submission=pd.read_csv("../input/tabular-playground-series-apr-2022/sample_submission.csv")


# In[ ]:


submission["state"] = sum(test_preds)/5
ans=[]
for i in submission["state"]:
    ans.append(i)


# In[ ]:


submission.to_csv("submission.csv_36", index=False)
submission


# feature 14 ->area under 13 sided polygon()

# ###
