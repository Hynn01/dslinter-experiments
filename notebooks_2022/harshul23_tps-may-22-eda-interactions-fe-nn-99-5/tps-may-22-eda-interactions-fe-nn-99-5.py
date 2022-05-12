#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

train_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
train_df.head()


# In[ ]:


test_df = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')
test_df.head()


# In[ ]:


train_df.info()


# In[ ]:


fig = plt.figure(figsize = (18,10))
sns.heatmap(train_df.corr(),  center=0, annot=True, fmt='.2f', )
plt.plot()


# Insights:
# 1. No feature is correlating with target 
# 2. f_28 is weakly correlated with f_00 - f_06 

# In[ ]:


continuos_col_float = []
for i in train_df.columns:
    if train_df[i].dtype == "float64" and i != "id" and i != "target":
        continuos_col_float.append(i)
len(continuos_col_float)        


# In[ ]:


fig = plt.figure(figsize = (15, 15))
for i, feat in enumerate(continuos_col_float):
    plt.subplot(4,4, i+1)
    sns.kdeplot(train_df[feat], hue = train_df['target'])
plt.tight_layout()    


# In[ ]:


continuos_col_int = []
for i in train_df.columns:
    if train_df[i].dtype == "int64" and i != "id" and i != "target":
        continuos_col_int.append(i)
len(continuos_col_int)        


# In[ ]:


fig = plt.figure(figsize = (15, 15))
for i, feat in enumerate(continuos_col_int):
    plt.subplot(4,4, i+1)
    sns.countplot(train_df[feat], hue = train_df['target'])
plt.tight_layout()   


# ### Making interactions

# In[ ]:


py1 = PolynomialFeatures(degree = 2, include_bias = False, interaction_only = True)
train_py = py1.fit_transform(train_df[continuos_col_int])
test_py = py1.transform(test_df[continuos_col_int])


# In[ ]:


py2 = PolynomialFeatures(degree = 2, include_bias = False, interaction_only = True)
train_py2 = py2.fit_transform(train_df[continuos_col_float])
test_py2 = py2.transform(test_df[continuos_col_float])


# In[ ]:


train_df_py = pd.DataFrame(train_py)
test_df_py = pd.DataFrame(test_py)


# In[ ]:


train_df_py2 = pd.DataFrame(train_py2)
test_df_py2 = pd.DataFrame(test_py2)


# In[ ]:


train_df_py.shape


# In[ ]:


train_df_py2.shape


# In[ ]:


def py2_to_py(df_py, df_py2):
    for i in df_py2.columns:
        df_py[str(i) + "_fl"] = df_py2[i]
    del [[df_py2]]
    gc.collect()
    
py2_to_py(train_df_py, train_df_py2)
del [[train_df_py2]]
gc.collect()

py2_to_py(test_df_py, test_df_py2)
del [[test_df_py2]]
gc.collect()


# In[ ]:


y_train = train_df['target']


# In[ ]:


def df_to_pydf(df, df_py):
    for i in range(10):
        df_py['f_27_'+str(i)] = df['f_27'].apply(lambda x: ord(x[i]) - ord("A"))
    df_py['unique_f_27'] = df['f_27'].apply(lambda x: len(set(x)))    
    del [[df]]
    gc.collect()
        
df_to_pydf(train_df, train_df_py)
del [[train_df]]
gc.collect()
df_to_pydf(test_df, test_df_py)       
del [[test_df]]
gc.collect()


# In[ ]:


li = [str(i) for i in train_df_py.columns if 'f_27' in str(i)]


# In[ ]:


li


# In[ ]:


cols = train_df_py.columns


# In[ ]:


scl = MinMaxScaler()
train_df_sc = scl.fit_transform(train_df_py)


# In[ ]:


del[[train_df_py]]
gc.collect()


# In[ ]:


test_df_sc = scl.transform(test_df_py)


# In[ ]:


del[[test_df_py]]
gc.collect()


# In[ ]:


train_df_sc.shape


# In[ ]:


from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[ ]:


model = Sequential()
model.add(Dense(1024, input_shape=[train_df_sc.shape[1]], activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.15))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


# In[ ]:


model.summary()


# In[ ]:


model.compile(optimizer='adadelta',
    loss= BinaryCrossentropy(from_logits=True),
    metrics=["AUC"])


# In[ ]:


mcallbacks = [EarlyStopping(monitor="val_loss",     # Quantity to be monitored
                            patience=20,                # How many epochs to wait before stopping
                            restore_best_weights=True), 
            ReduceLROnPlateau(monitor='val_loss', 
                                factor=0.5,                # Factor by which the learning rate will be reduced
                                patience=5)
            ]


# In[ ]:


mfit = model.fit(train_df_sc, y_train, batch_size = 9000, epochs = 150, validation_split = 0.1, use_multiprocessing=True, workers = -1, callbacks = mcallbacks, shuffle = True, verbose = 2 )


# In[ ]:


del [[train_df_sc]]
gc.collect()


# In[ ]:


y_pred = model.predict(test_df_sc)


# In[ ]:


del [[test_df_sc]]
gc.collect()


# In[ ]:


sub = pd.read_csv('../input/tabular-playground-series-may-2022/sample_submission.csv')
sub.shape


# In[ ]:


sub.target = y_pred


# In[ ]:


sub.head(10)


# In[ ]:


sub.to_csv("submission.csv", index = False)


# In[ ]:




