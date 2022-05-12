#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from math import ceil

import logging

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import mean_squared_error, roc_auc_score

SEED = 42

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


from pathlib import Path
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG


def create_logger(exp_version):
    log_file = ("{}.log".format(exp_version))

    # logger
    logger_ = getLogger(exp_version)
    logger_.setLevel(DEBUG)

    # formatter
    fmr = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    # file handler
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    fh.setFormatter(fmr)

    # stream handler
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmr)

    logger_.addHandler(fh)
    logger_.addHandler(ch)


def get_logger(exp_version):
    return getLogger(exp_version)

VERSION = "007"
create_logger(VERSION)
get_logger(VERSION).info("Logger Started")


# In[ ]:


debug = False

dig_df = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
X_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[ ]:


dig_df.shape


# In[ ]:


y = dig_df["label"]


# In[ ]:


X = dig_df.drop(columns="label")
X.head()


# In[ ]:


X.shape


# In[ ]:


def show_multiple_img(images, targets):
    """

    :param images:
    :param targets:
    :return:
    """

    if len(images) < 6:
        my_cols = len(images)
    else:
        my_cols = 6
    my_rows = ceil(len(images) / my_cols)

    fig_width = my_cols * 10 / 6
    fig_height = my_rows * 10 / 4

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(my_rows, my_cols)

    axes = []
    matrix_dimension = (my_rows, my_cols)
    for n, image in enumerate(images):
        subplot_position = np.unravel_index(n, matrix_dimension)

        axes.append(fig.add_subplot(gs[subplot_position]))

    for ax in axes:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    for ax, image, target in zip(axes, images, targets):
        ax.set_title(target)
        ax.imshow(image, cmap=plt.cm.gray_r)
    plt.tight_layout()
    plt.show()


# In[ ]:


n_img = 18
images = X[:n_img].to_numpy().reshape(n_img,28,28)
targets = y[:n_img]

show_multiple_img(images, targets)


# In[ ]:


# Vediamo se ci sono dati mancanti:
X.isna().sum().sum(), y.isna().sum()


# In[ ]:





# In[ ]:


# Data ugmentation
def roll_images(X_train, y_train, shift=1):
    X_augmentated = X_train.copy()
    get_logger(VERSION).info(X_augmentated.shape)
    X_train_array = X_train.to_numpy().reshape(X_train.shape[0], 28, 28)

    new_trains1 = np.roll(X_train_array,  shift, axis=1)
    new_trains2 = np.roll(X_train_array, -shift, axis=1)
    new_trains3 = np.roll(X_train_array,  shift, axis=2)
    new_trains4 = np.roll(X_train_array, -shift, axis=2)
#     new_trains5 = np.roll(new_trains1,  shift, axis=2)
#     new_trains6 = np.roll(new_trains1, -shift, axis=2)
#     new_trains7 = np.roll(new_trains2,  shift, axis=2) 
#     new_trains8 = np.roll(new_trains2, -shift, axis=2)

    X_augmentated = np.concatenate((X_augmentated,
                                  new_trains1.reshape(new_trains1.shape[0],28*28),
                                  new_trains2.reshape(new_trains1.shape[0],28*28),
                                  new_trains3.reshape(new_trains1.shape[0],28*28),
                                  new_trains4.reshape(new_trains1.shape[0],28*28),
#                                   new_trains5.reshape(new_trains1.shape[0],28*28),
#                                   new_trains6.reshape(new_trains1.shape[0],28*28),
#                                   new_trains7.reshape(new_trains1.shape[0],28*28),
#                                 new_trains8.reshape(new_trains1.shape[0],28*28)
                                   ))
    y_augmentated = np.tile(y_train, X_augmentated.shape[0]//y_train.shape[0])
    get_logger(VERSION).info(X_augmentated.shape)
    get_logger(VERSION).info(y_augmentated.shape)
    return X_augmentated, y_augmentated

X, y = roll_images(X, y)


# In[ ]:



if debug:
    X = X.head(1000)
    y = y[:1000]


# In[ ]:


X.shape, y.shape


# In[ ]:


splits = 10
kfold = KFold(n_splits=splits, shuffle=True, random_state=SEED)

oof_preds = np.zeros((X.shape[0],))
preds = 0
total_mean_rmse = 0

for n_, (train_index, eval_index)in enumerate(kfold.split(X)):
    
    X_train, X_eval = X[train_index,:], X[eval_index,:]
    y_train, y_eval = y[train_index], y[eval_index]
    
    # Creo il modello
    get_logger(VERSION).info(f"Fold n.{n_} - Creating and fitting Model")
    model = KNeighborsClassifier(n_jobs=-1) 
    model.fit(X_train, y_train)
    
    get_logger(VERSION).info("Model Fitted")
    
    preds += model.predict(X_test) / splits
    get_logger(VERSION).info("X_Test Predicted")
    get_logger(VERSION).info(preds[:15])
    oof_preds[eval_index] += model.predict(X_eval)
    
    get_logger(VERSION).info("X_eval Predicted")
    
    fold_rmse = np.sqrt(mean_squared_error(y_eval, oof_preds[eval_index]))
    get_logger(VERSION).info(f"Fold {n_} RMSE: {fold_rmse}")
#         print(f"Trees: {model.tree_count_}")
    total_mean_rmse += fold_rmse / splits
    
get_logger(VERSION).info(f"\nOverall RMSE: {total_mean_rmse}")


# In[ ]:


dig_test_df = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[ ]:


X_test = dig_test_df


# In[ ]:


submissions = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")


# In[ ]:





# In[ ]:


preds = np.round(preds).astype(int)


# In[ ]:


submissions.Label = preds


# In[ ]:


submissions.to_csv("submission.csv", index = False)


# In[ ]:




