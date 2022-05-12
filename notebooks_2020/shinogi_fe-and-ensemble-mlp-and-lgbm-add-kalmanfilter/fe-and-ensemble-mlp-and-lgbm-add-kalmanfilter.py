#!/usr/bin/env python
# coding: utf-8

# # reference
# https://www.kaggle.com/martxelo/fe-and-ensemble-mlp-and-lgbm  
# https://www.kaggle.com/teejmahal20/single-model-lgbm-kalman-filter

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# imports
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, plot_confusion_matrix
from keras.models import Model
import keras.layers as L
import lightgbm as lgb


# # Load data
# ## data-without-drift
# Thanks to https://www.kaggle.com/cdeotte/data-without-drift.  
# ## clean-kalman
# Thanks to https://www.kaggle.com/ragnar123/clean-kalman.
# ## data-without-drift-with-kalman-filter
# Thanks to https://www.kaggle.com/michaln/data-without-drift-with-kalman-filter.  
# Thanks to https://www.kaggle.com/michaln/kalman-filter-on-clean-data.
# ## clean-2apply-kalman
# https://www.kaggle.com/shinogi/kalman-filter-2-apply-on-clean-data

# In[ ]:


# read data
# data = pd.read_csv('../input/data-without-drift/train_clean.csv')
# data = pd.read_csv('../input/clean-kalman/train_clean_kalman.csv')
# data = pd.read_csv('../input/data-without-drift-with-kalman-filter/train.csv')
# data = pd.read_csv('../input/clean-2apply-kalman/train_clean_2apply_kalman.csv')
# data = pd.read_csv('../input/clean-3apply-kalman/train_clean_3apply_kalman.csv')
# data = pd.read_csv('../input/clean-4apply-kalman/train_clean_4apply_kalman.csv')
data = pd.read_csv('../input/clean-5apply-kalman/train_clean_5apply_kalman.csv')

data.head()


# # Feature engineering
# Add to signal several other signals: gradients, rolling mean, std, low/high pass filters...
# 
# FE is the same as this notebook https://www.kaggle.com/martxelo/fe-and-simple-mlp with corrections in filters.

# ## definite feature enginnering function

# ### gradients

# In[ ]:


def calc_gradients(s, n_grads=4):
    '''
    Calculate gradients for a pandas series. Returns the same number of samples
    '''
    grads = pd.DataFrame()
    
    g = s.values
    for i in range(n_grads):
        g = np.gradient(g)
        grads['grad_' + str(i+1)] = g
        
    return grads


# ### low_pass

# In[ ]:


def calc_low_pass(s, n_filts=10):
    '''
    Applies low pass filters to the signal. Left delayed and no delayed
    '''
    wns = np.logspace(-2, -0.3, n_filts)
    
    low_pass = pd.DataFrame()
    x = s.values
    for wn in wns:
        b, a = signal.butter(1, Wn=wn, btype='low')
        zi = signal.lfilter_zi(b, a)
        low_pass['lowpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]
        low_pass['lowpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)
        
    return low_pass


# ### high pass

# In[ ]:


def calc_high_pass(s, n_filts=10):
    '''
    Applies high pass filters to the signal. Left delayed and no delayed
    '''
    wns = np.logspace(-2, -0.1, n_filts)
    
    high_pass = pd.DataFrame()
    x = s.values
    for wn in wns:
        b, a = signal.butter(1, Wn=wn, btype='high')
        zi = signal.lfilter_zi(b, a)
        high_pass['highpass_lf_' + str('%.4f' %wn)] = signal.lfilter(b, a, x, zi=zi*x[0])[0]
        high_pass['highpass_ff_' + str('%.4f' %wn)] = signal.filtfilt(b, a, x)
        
    return high_pass


# ### Rolling

# In[ ]:


def calc_roll_stats(s, windows=[10, 50, 100, 500, 1000]):
    '''
    Calculates rolling stats like mean, std, min, max...
    '''
    roll_stats = pd.DataFrame()
    for w in windows:
        roll_stats['roll_mean_' + str(w)] = s.rolling(window=w, min_periods=1).mean()
        roll_stats['roll_std_' + str(w)] = s.rolling(window=w, min_periods=1).std()
        roll_stats['roll_min_' + str(w)] = s.rolling(window=w, min_periods=1).min()
        roll_stats['roll_max_' + str(w)] = s.rolling(window=w, min_periods=1).max()
        roll_stats['roll_range_' + str(w)] = roll_stats['roll_max_' + str(w)] - roll_stats['roll_min_' + str(w)]
        roll_stats['roll_q10_' + str(w)] = s.rolling(window=w, min_periods=1).quantile(0.10)
        roll_stats['roll_q25_' + str(w)] = s.rolling(window=w, min_periods=1).quantile(0.25)
        roll_stats['roll_q50_' + str(w)] = s.rolling(window=w, min_periods=1).quantile(0.50)
        roll_stats['roll_q75_' + str(w)] = s.rolling(window=w, min_periods=1).quantile(0.75)
        roll_stats['roll_q90_' + str(w)] = s.rolling(window=w, min_periods=1).quantile(0.90)
    
    # add zeros when na values (std)
    roll_stats = roll_stats.fillna(value=0)
             
    return roll_stats


# ### exponential weighted functions

# In[ ]:


def calc_ewm(s, windows=[10, 50, 100, 500, 1000]):
    '''
    Calculates exponential weighted functions
    '''
    ewm = pd.DataFrame()
    for w in windows:
        ewm['ewm_mean_' + str(w)] = s.ewm(span=w, min_periods=1).mean()
        ewm['ewm_std_' + str(w)] = s.ewm(span=w, min_periods=1).std()
        
    # add zeros when na values (std)
    ewm = ewm.fillna(value=0)
        
    return ewm


# ## exec feature enginnering

# In[ ]:


def add_features(s):
    '''
    All calculations together
    '''
    print(s)
    gradients = calc_gradients(s)
    low_pass = calc_low_pass(s)
    high_pass = calc_high_pass(s)
    roll_stats = calc_roll_stats(s)
#     kf_stats = calc_roll_stats_kf(s)
    ewm = calc_ewm(s)
    
    return pd.concat([s, gradients, low_pass, high_pass, roll_stats, ewm], axis=1)


def divide_and_add_features(s, signal_size=500000):
    '''
    Divide the signal in bags of "signal_size".
    Normalize the data dividing it by 15.0
    '''
    # normalize
    s = s/15.0
    
    ls = []
    for i in tqdm(range(int(s.shape[0]/signal_size))):
        sig = s[i*signal_size:(i+1)*signal_size].copy().reset_index(drop=True)
        sig_featured = add_features(sig)
        ls.append(sig_featured)
    
    return pd.concat(ls, axis=0)


# In[ ]:


# apply every feature to data
df = divide_and_add_features(data['signal'])
df.head()


# ## plot
# Let's plot the signals to see how they look like.

# In[ ]:


# The low pass lfilter captures the trend of the signal for different cutoff frequencies
df[['signal',
    'lowpass_lf_0.0100',
    'lowpass_lf_0.0154',
    'lowpass_lf_0.0239',
    'lowpass_lf_0.0369',
    'lowpass_lf_0.5012']].iloc[:200].plot()


# In[ ]:


# The low pass filtfilt captures the trend of the signal for different cutoff frequencies
# but without delay
df[['signal',
    'lowpass_ff_0.0100',
    'lowpass_ff_0.0154',
    'lowpass_ff_0.0239',
    'lowpass_ff_0.0369',
    'lowpass_ff_0.5012']].iloc[:200].plot()


# In[ ]:


# The high pass lfilter captures fast variation of the signal for different cutoff frequencies
df[['signal',
    'highpass_lf_0.0100',
    'highpass_lf_0.0163',
    'highpass_lf_0.0264',
    'highpass_lf_0.3005',
    'highpass_lf_0.7943']].iloc[:100].plot()


# In[ ]:


# The high pass lfilter captures fast variation of the signal for different cutoff frequencies
# but without delay
df[['signal',
    'highpass_ff_0.0100',
    'highpass_ff_0.0163',
    'highpass_ff_0.0264',
    'highpass_ff_0.3005',
    'highpass_ff_0.7943']].iloc[:200].plot()


# In[ ]:


# rolling mean, quantiles and ewm also capture the trend
df[['signal',
    'roll_mean_10',
    'roll_mean_50',
    'roll_mean_100',
    'roll_q50_100',
    'ewm_mean_10',
    'ewm_mean_50',
    'ewm_mean_100']].iloc[:100].plot()


# In[ ]:


# quantiles, min, max
df[['signal',
    'roll_min_100',
    'roll_q10_100',
    'roll_q25_100',
    'roll_q50_100',
    'roll_q75_100',
    'roll_q90_100',
    'roll_max_100']].iloc[:1000].plot()


# In[ ]:


# rolling std, and emw std
df[['signal',
    'roll_std_10',
    'roll_std_50',
    'ewm_std_10',
    'ewm_std_50']].iloc[:100].plot()


# # Divide in train and test

# In[ ]:


# Get train and test data
x_train, x_test, y_train, y_test = train_test_split(df.values, data['open_channels'].values, test_size=0.2)

del data, df
print('x_train.shape=', x_train.shape)
print('x_test.shape=', x_test.shape)
print('y_train.shape=', y_train.shape)
print('y_test.shape=', y_test.shape)


# # Classes weights

# In[ ]:


def get_class_weight(classes, exp=1):
    '''
    Weight of the class is inversely proportional to the population of the class.
    There is an exponent for adding more weight.
    '''
    hist, _ = np.histogram(classes, bins=np.arange(12)-0.5)
    class_weight = hist.sum()/np.power(hist, exp)
    
    return class_weight

class_weight = get_class_weight(y_train)
print('class_weight=', class_weight)
plt.figure()
plt.title('classes')
plt.hist(y_train, bins=np.arange(12)-0.5)
plt.figure()
plt.title('class_weight')
plt.bar(np.arange(11), class_weight)
plt.title('class_weight')


# # Build a MLP model

# In[ ]:


def create_mpl(shape):
    '''
    Returns a keras model
    '''
    
    X_input = L.Input(shape)
    
    X = L.Dense(150, activation='relu')(X_input)
    X = L.Dense(150, activation='relu')(X)
    X = L.Dense(125, activation='relu')(X)
    X = L.Dense(100, activation='relu')(X)
    X = L.Dense(75, activation='relu')(X)
    X = L.Dense(50, activation='relu')(X)
    X = L.Dense(25, activation='relu')(X)
    X = L.Dense(11, activation='softmax')(X)
    
    model = Model(inputs=X_input, outputs=X)
    
    return model


mlp = create_mpl(x_train[0].shape)
mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
print(mlp.summary())


# In[ ]:


# fit the model
mlp.fit(x=x_train, y=y_train, epochs=30, batch_size=1024, class_weight=class_weight)


# In[ ]:


# plot history
plt.figure(1)
plt.plot(mlp.history.history['loss'], 'b', label='loss')
plt.xlabel('epochs')
plt.legend()
plt.figure(2)
plt.plot(mlp.history.history['sparse_categorical_accuracy'], 'g', label='sparse_categorical_accuracy')
plt.xlabel('epochs')
plt.legend()


# In[ ]:


# predict on test
mlp_pred = mlp.predict(x_test)
print('mlp_pred.shape=', mlp_pred.shape)


# # Build LGBM model
# Parameters from https://www.kaggle.com/ragnar123/single-model-lgbm. Thanks!

# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


# build model
dataset = lgb.Dataset(x_train, label=y_train)
params = {'boosting_type': 'gbdt',
          'metric': 'rmse',
          'objective': 'regression',
          'n_jobs': -1,
#           'n_jobs': 4,
          'seed': 236,
          'num_leaves': 280,
          'learning_rate': 0.026623466966581126,
          'max_depth': 73,
          'lambda_l1': 2.959759088169741,
          'lambda_l2': 1.331172832164913,
          'bagging_fraction': 0.9655406551472153,
          'bagging_freq': 9,
          'colsample_bytree': 0.6867118652742716,
#           'device': 'gpu',
#           'gpu_platform_id': 0,
#           'gpu_device_id': 0
}


# In[ ]:


import gc
gc.collect()


# In[ ]:


# fit the model
print('Training LGBM...')
gbc = lgb.train(params, dataset,
                num_boost_round=10000,
                verbose_eval=100)
print('LGBM trained!')


# In[ ]:


# predict on test
gbc_pred = gbc.predict(x_test)
print('gbc_pred.shape=', gbc_pred.shape)


# In[ ]:


def convert_to_one_hot(pred):
    '''
    Convert the prediction into probabilities
    Example: 1.6 --> [0, 0.4, 0.6, 0, 0, ...]
    1.6 is closer to 2 than to 1
    All rows will sum 1
    '''

    # clip results lower or higher than limits
    pred = np.clip(pred, 0, 10)

    # convert to "one-hot"
    pred = 1 - np.abs(pred.reshape((-1,1)) - np.arange(11))

    # clip results lower than 0
    pred = np.clip(pred, 0, 1)
    
    return pred
    
gbc_pred = convert_to_one_hot(gbc_pred)

print('gbc_pred.shape=', gbc_pred.shape)


# # Ensemble
# The idea is to mix both results with a parameter alpha ($0\le\alpha\le1$):
# 
# $y_{pred}=\alpha·mlp_{pred} + (1-\alpha)·gbc_{pred}$

# In[ ]:


# lists for keep results
f1s = []
alphas = np.linspace(0,1,101)

# loop for every alpha
for alpha in tqdm(alphas):
    y_pred = alpha*mlp_pred + (1 - alpha)*gbc_pred
    f1 = f1_score(y_test, np.argmax(y_pred, axis=-1), average='macro')
    f1s.append(f1)

# convert to numpy array
f1s = np.array(f1s)

# get best_alpha
best_alpha = alphas[np.argmax(f1s)]

print('best_f1=', f1s.max())
print('best_alpha=', best_alpha)


# In[ ]:


plt.plot(alphas, f1s)
plt.title('f1_score for ensemble')
plt.xlabel('alpha')
plt.ylabel('f1_score')


# # Confusion matrix

# In[ ]:


# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud
def plot_cm(y_true, y_pred, title):
    figsize=(16,16)
    y_pred = y_pred.astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap='viridis', annot=annot, fmt='', ax=ax)


# In[ ]:


f1_mlp = f1_score(y_test, np.argmax(mlp_pred, axis=-1), average='macro')
plot_cm(y_test, np.argmax(mlp_pred, axis=-1), 'Only MLP \n f1=' + str('%.4f' %f1_mlp))


# In[ ]:


f1_gbc = f1_score(y_test, np.argmax(gbc_pred, axis=-1), average='macro')
plot_cm(y_test, np.argmax(gbc_pred, axis=-1), 'Only GBC \n f1=' + str('%.4f' %f1_gbc))


# In[ ]:


y_pred = best_alpha*mlp_pred + (1 - best_alpha)*gbc_pred
f1_ens = f1_score(y_test, np.argmax(y_pred, axis=-1), average='macro')
plot_cm(y_test, np.argmax(y_pred, axis=-1), 'Ensemble \n f1=' + str('%.4f' %f1_ens))


# # Submit result

# In[ ]:


def submit_result(mlp, gbc, alpha):
    
    print('Reading data...')
#     data = pd.read_csv('../input/data-without-drift/test_clean.csv')
#     data = pd.read_csv('../input/clean-kalman/test_clean_kalman.csv')
#     data = pd.read_csv('../input/data-without-drift-with-kalman-filter/test.csv')
#     data = pd.read_csv('../input/clean-2apply-kalman/test-clean-2apply-kalman.csv')
#     data = pd.read_csv('../input/clean-3apply-kalman/test_clean_3apply_kalman.csv')
#     data = pd.read_csv('../input/clean-4apply-kalman/test_clean_4apply_kalman.csv')
    data = pd.read_csv('../input/clean-5apply-kalman/test_clean_5apply_kalman.csv')
    
    print('Feature engineering...')
    df = divide_and_add_features(data['signal'])
    
    print('Predicting MLP...')
    mlp_pred = mlp.predict(df.values)
    
    print('Predicting GBC...')
    gbc_pred = gbc.predict(df.values)
    gbc_pred = convert_to_one_hot(gbc_pred)
    
    print('Ensembling...')
    y_pred = alpha*mlp_pred + (1 - alpha)*gbc_pred
    y_pred = np.argmax(y_pred, axis=-1)
    
    print('Writing submission...')
    submission = pd.DataFrame()
    submission['time'] = data['time']
    submission['open_channels'] = y_pred
    submission.to_csv('submission.csv', index=False, float_format='%.4f')
    
    print('Submission finished!')


# In[ ]:


submit_result(mlp, gbc, best_alpha)

