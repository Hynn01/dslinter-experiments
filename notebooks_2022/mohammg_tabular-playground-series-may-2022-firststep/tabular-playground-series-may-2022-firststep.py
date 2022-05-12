#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
import datetime
import scipy.stats
import math
import random
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation,Input, InputLayer, Dense, BatchNormalization, Dropout,LayerNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train=pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
test=pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


print(train['f_27'].value_counts())
print(train['f_27'].nunique())


# In[ ]:


def label_encode_columns(df, columns, encoders=None):
	if encoders is None:
		encoders = {}
	
		for col in columns:

			unique_values = list(df[col].unique())
			unique_values.append('Unseen')
			le = LabelEncoder().fit(unique_values)
			df[col] = le.transform(df[col])
			encoders[col] = le
	
	else:
		for col in columns:
			le = encoders.get(col)
			df[col] = [x if x in le.classes_ else 'Unseen' for x in df[col]]
			df[col] = le.transform(df[col])

	return df, encoders


# In[ ]:





# In[ ]:


def processCol_str(df,colName,n):
    Newtrain=df[colName].str.split('',n=0,expand=True)
    for ncol in np.arange(1,n+1):
        df[colName+'_'+ str(ncol)]=Newtrain[ncol]
        
    df=df.drop(colName,axis=1,inplace=True)


# In[ ]:


processCol_str(train,'f_27',10)


# In[ ]:


processCol_str(test,'f_27',10)
catcol=[ 'f_27_1', 'f_27_2', 'f_27_3',
       'f_27_4', 'f_27_5', 'f_27_6', 'f_27_7', 'f_27_8', 'f_27_9', 'f_27_10']


# In[ ]:



train,encoder = label_encode_columns(train,catcol)
test,encoder=label_encode_columns(test,catcol,encoder)


# In[ ]:


y=train['target']
X=train.drop(['id','target'],axis=1,inplace=True)


# In[ ]:


X=train


# In[ ]:


X.head()


# In[ ]:





# In[ ]:





# In[ ]:


features=[ 'f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07',
       'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16',
       'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25',
       'f_26', 'f_28', 'f_29', 'f_30', 'f_27_1', 'f_27_2', 'f_27_3', 'f_27_4',
       'f_27_5', 'f_27_6', 'f_27_7', 'f_27_8', 'f_27_9', 'f_27_10']


# In[ ]:


def plot_history(history, *, n_epochs=None, plot_lr=False, title=None, bottom=None, top=None):
    """Plot (the last n_epochs epochs of) the training history
    
    Plots loss and optionally val_loss and lr."""
    plt.figure(figsize=(15, 6))
    from_epoch = 0 if n_epochs is None else max(len(history['loss']) - n_epochs, 0)
    
    # Plot training and validation losses
    plt.plot(np.arange(from_epoch, len(history['loss'])), history['loss'][from_epoch:], label='Training loss')
    try:
        plt.plot(np.arange(from_epoch, len(history['loss'])), history['val_loss'][from_epoch:], label='Validation loss')
        best_epoch = np.argmin(np.array(history['val_loss']))
        best_val_loss = history['val_loss'][best_epoch]
        if best_epoch >= from_epoch:
            plt.scatter([best_epoch], [best_val_loss], c='r', label=f'Best val_loss = {best_val_loss:.5f}')
        if best_epoch > 0:
            almost_epoch = np.argmin(np.array(history['val_loss'])[:best_epoch])
            almost_val_loss = history['val_loss'][almost_epoch]
            if almost_epoch >= from_epoch:
                plt.scatter([almost_epoch], [almost_val_loss], c='orange', label='Second best val_loss')
    except KeyError:
        pass
    if bottom is not None: plt.ylim(bottom=bottom)
    if top is not None: plt.ylim(top=top)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    if title is not None: plt.title(title)
        
    # Plot learning rate
    if plot_lr and 'lr' in history:
        ax2 = plt.gca().twinx()
        ax2.plot(np.arange(from_epoch, len(history['lr'])), np.array(history['lr'][from_epoch:]), color='g', label='Learning rate')
        ax2.set_ylabel('Learning rate')
        ax2.legend(loc='upper right')
        
    plt.show()


# In[ ]:


def my_model():
    """Simple sequential neural network with three hidden layers.
    
    Returns a (not yet compiled) instance of tensorflow.keras.models.Model.
    """
    activation = 'swish'
    inputs = Input(shape=(len(features)))
    x = Dense(256,
              kernel_regularizer=tf.keras.regularizers.l2(30e-6),
               use_bias = True,
              activation=activation,
             )(inputs)
    x = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              use_bias = True,
              activation=activation,
             )(inputs)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              use_bias = True,
              activation=activation,
             )(x)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              use_bias = True,
              activation=activation,
             )(x)
    x = Dense(32, kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              use_bias = True,
              activation=activation,
             )(x)
    x = Dense(16, kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              activation=activation,
             )(x)
    x = Dense(1,
              use_bias = True,
              kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              activation='sigmoid',
             )(x)
    model = Model(inputs, x)
    return model

plot_model(my_model(), show_layer_names=True, show_shapes=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Cross-validation of the classifier\n\nEPOCHS = 200\nEPOCHS_COSINEDECAY = 100\nVERBOSE = 0 # set to 0 for less output, or to 2 for more output\nDIAGRAMS = True\nUSE_PLATEAU = False\nBATCH_SIZE = 4096\n\n# see https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development\nnp.random.seed(1)\nrandom.seed(1)\ntf.random.set_seed(1)\n\ndef fit_model(X_tr, y_tr, X_va=None, y_va=None, run=0):\n    """Scale the data, fit a model, plot the training history and optionally validate the model\n    \n    Returns a trained instance of tensorflow.keras.models.Model.\n    \n    As a side effect, updates y_va_pred, history_list and score_list.\n    """\n    global y_va_pred\n    start_time = datetime.datetime.now()\n    \n    scaler = StandardScaler()\n    X_tr = scaler.fit_transform(X_tr)\n    \n    if X_va is not None:\n        X_va = scaler.transform(X_va)\n        validation_data = (X_va, y_va)\n    else:\n        validation_data = None\n\n    # Define the learning rate schedule and EarlyStopping\n    lr_start=0.01\n    if USE_PLATEAU and X_va is not None: # use early stopping\n        epochs = EPOCHS\n        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, \n                               patience=4, verbose=VERBOSE)\n        es = EarlyStopping(monitor="val_loss",\n                           patience=12, \n                           verbose=1,\n                           mode="min", \n                           restore_best_weights=True)\n        callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]\n\n    else: # use cosine learning rate decay rather than early stopping\n        epochs = EPOCHS_COSINEDECAY\n        lr_end=0.0002\n        def cosine_decay(epoch):\n            if epochs > 1:\n                w = (1 + math.cos(epoch / (epochs-1) * math.pi)) / 2\n            else:\n                w = 1\n            return w * lr_start + (1 - w) * lr_end\n\n        lr = LearningRateScheduler(cosine_decay, verbose=0)\n        callbacks = [lr, tf.keras.callbacks.TerminateOnNaN()]\n        \n    # Construct and compile the model\n    model = my_model()\n    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_start),\n                  #metrics=\'acc\',\n                  loss=tf.keras.losses.BinaryCrossentropy())\n    #model.compile(optimizer=tf.keras.optimizers.SGD(), loss=\'mse\')\n\n    # Train the model\n    history = model.fit(X_tr, y_tr, \n                        validation_data=validation_data, \n                        epochs=epochs,\n                        verbose=VERBOSE,\n                        batch_size=BATCH_SIZE,\n                        shuffle=True,\n                        callbacks=callbacks)\n\n    history_list.append(history.history)\n    callbacks, es, lr, history = None, None, None, None\n    print(f"Training loss:   {history_list[-1][\'loss\'][-1]:.3f}")\n    \n    if X_va is not None:\n        # Inference for validation\n        y_va_pred = model.predict(X_va, batch_size=BATCH_SIZE, verbose=VERBOSE)\n        #oof_list[run][val_idx] = y_va_pred\n        \n        # Evaluation: Execution time and AUC\n        score = roc_auc_score(y_va, y_va_pred)\n        print(f"Fold {run}.{fold} | {str(datetime.datetime.now() - start_time)[-12:-7]}"\n              f" | AUC: {score:.5f}")\n        score_list.append(score)\n        \n        if DIAGRAMS and fold == 0 and run == 0:\n            # Plot training history\n            plot_history(history_list[-1], \n                         title=f"Learning curve (validation AUC = {score:.5f})",\n                         plot_lr=True, n_epochs=110)\n\n            # Plot y_true vs. y_pred\n            plt.figure(figsize=(10, 4))\n            plt.hist(y_va_pred[y_va == 0], bins=np.linspace(0, 1, 21),\n                     alpha=0.5, density=True)\n            plt.hist(y_va_pred[y_va == 1], bins=np.linspace(0, 1, 21),\n                     alpha=0.5, density=True)\n            plt.xlabel(\'y_pred\')\n            plt.ylabel(\'density\')\n            plt.title(\'OOF Predictions\')\n            plt.show()\n\n    return model, scaler')


# In[ ]:


history_list = []
score_list = []
kf = KFold(n_splits=5)
for fold, (idx_tr, idx_va) in enumerate(kf.split(X)):
    X_tr = X.iloc[idx_tr][features]
    X_va = X.iloc[idx_va][features]
    y_tr = y.iloc[idx_tr]
    y_va = y.iloc[idx_va]
  
    fit_model(X_tr, y_tr, X_va, y_va)
    break # we only need the first fold


# In[ ]:


# Create submission
# Create submission
print(f"{len(features)} features")

X_tr = X
y_tr = y

pred_list = []
for seed in range(10):
    # see https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    model, scaler = fit_model(X_tr, y_tr, run=seed)
    pred_list.append(scipy.stats.rankdata(model.predict(scaler.transform(test[features]),
                                                        batch_size=BATCH_SIZE, verbose=VERBOSE)))
    print(f"{seed:2}", pred_list[-1])
print()
submission = test[['id']].copy()
submission['target'] = np.array(pred_list).mean(axis=0)
submission.to_csv('submission.csv', index=False)
submission


# refrence:
#     https://www.kaggle.com/code/ambrosm/tpsmay22-keras-quickstart/notebook?scriptVersionId=94617937

# In[ ]:





# In[ ]:


ord('A')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




