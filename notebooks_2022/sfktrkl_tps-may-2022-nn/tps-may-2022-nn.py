#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Loading datasets

# In[ ]:


import os
import random
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(6)
tf.config.threading.set_inter_op_parallelism_threads(2)

from tensorflow import keras
from tensorflow.keras import layers, callbacks

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv', index_col=0)
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv', index_col=0)
sub = pd.read_csv('../input/tabular-playground-series-may-2022/sample_submission.csv')


# # Explore Data

# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


print("Columns: \n{0}".format(list(train.columns)))


# # Basic Data Check

# In[ ]:


print('Train data shape:', train.shape)
print('Test data shape:', test.shape)


# ## Missing values

# In[ ]:


missing_values_train = train.isna().any().sum()
print('Missing values in train data: {0}'.format(missing_values_train[missing_values_train > 0]))

missing_values_test = test.isna().any().sum()
print('Missing values in test data: {0}'.format(missing_values_test[missing_values_test > 0]))


# ## Duplicates

# In[ ]:


duplicates_train = train.duplicated().sum()
print('Duplicates in train data: {0}'.format(duplicates_train))

duplicates_test = test.duplicated().sum()
print('Duplicates in test data: {0}'.format(duplicates_test))


# # Target Distribution

# In[ ]:


plt.figure(figsize=(10, 6))
plt.title('Target distribution')
ax = sns.countplot(x=train['target'], data=train)


# # Feature Engineering

# In[ ]:


# Credits to https://www.kaggle.com/code/ambrosm/tpsmay22-keras-quickstart/notebook
for df in [train, test]:
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
    # Next feature is from https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model
    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))
    df.drop('f_27', axis=1, inplace=True)
print("Columns: \n{0}".format(list(train.columns)))
train.head()


# # Modelling

# In[ ]:


X = train.drop('target', axis=1).copy()
y = train.target.copy()
test_X = test.copy()

# Scaling and Nomalization
transformer = make_pipeline(
    StandardScaler()
)
columns = X.columns[:-11]

transformer_new = make_pipeline(
    StandardScaler()
)
new_columns = X.columns[-11:]

preprocessor = make_column_transformer(
    (transformer, columns),
    (transformer_new, new_columns),
)


# ## Model

# In[ ]:


N_SPLITS = 5
EPOCHS = 200
BATCH_SIZE = 4096
ACTIVATION = 'swish'

my_seed = 1
def seedAll(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
seedAll(my_seed)

def load_model():
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",     # Quantity to be monitored
        patience=20,                # How many epochs to wait before stopping
        restore_best_weights=True)
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,                # Factor by which the learning rate will be reduced
        patience=5)                # Number of epochs with no improvement
    
    model = keras.Sequential([
        layers.Dense(108, activation=ACTIVATION, input_shape=[X.shape[1]]),      
        layers.Dense(64, activation=ACTIVATION), 
        layers.Dense(32, activation=ACTIVATION),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['AUC'])
    
    return model, [early_stopping, reduce_lr]


# ## Training

# In[ ]:


scores = []
f_scores = []
test_predictions = []
cv = StratifiedKFold(n_splits=N_SPLITS, random_state=my_seed, shuffle=True)
for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    train_X, val_X = X.iloc[train_idx], X.iloc[test_idx]
    train_y, val_y = y.iloc[train_idx], y.iloc[test_idx]
    
    train_X = preprocessor.fit_transform(train_X)
    val_X = preprocessor.transform(val_X)

    model, CALLBACKS = load_model()
    history = model.fit(
        train_X, train_y,
        validation_data=(val_X, val_y),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=CALLBACKS,        # Put your callbacks in a list
        verbose=0)                  # Turn off training log

    predictions = model.predict(val_X)
    score = roc_auc_score(val_y, predictions)
    scores.append(score)
    print(f"Fold {fold + 1} \t\t AUC: {score}")

    test_predictions.append(model.predict(preprocessor.transform(test_X)))

    # Saving history to plot at the end
    hist = pd.DataFrame(history.history)
    hist['folds'] = fold + 1
    f_scores = hist if fold == 0 else pd.concat([f_scores, hist], axis=0)
print('Overall AUC: ', np.mean(scores))


# ## Outcomes

# In[ ]:


for fold in range(f_scores['folds'].nunique()):
    fold = fold + 1
    history_f = f_scores[f_scores['folds'] == fold]

    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(14,4))
    fig.suptitle('Fold : ' + str(fold), fontsize=14)
        
    plt.subplot(1,2,1)
    plt.plot(history_f.loc[:, ['loss', 'val_loss']], label= ['loss', 'val_loss'])
    plt.legend(fontsize=15)
    plt.grid()
    
    plt.subplot(1,2,2)
    plt.plot(history_f.loc[:, ['auc', 'val_auc']],label= ['auc', 'val_auc'])
    plt.legend(fontsize=15)
    plt.grid()
    
    print("Validation Loss: {:0.4f}".format(history_f['val_loss'].min()));


# # Submission

# In[ ]:


sub['target'] = np.mean(test_predictions, axis=0)
sub.to_csv('submission.csv', index=False)
sub

