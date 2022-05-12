#!/usr/bin/env python
# coding: utf-8

# # MAY 2022
# work in progress ...

# In[ ]:


import time
from datetime import datetime

#measure notebook running time
start_time = time.time()

get_ipython().run_line_magic('matplotlib', 'inline')

import os, warnings
import numpy as np 
from numpy.random import seed
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import metrics
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix, precision_score,recall_score, f1_score, classification_report, accuracy_score

sns.set(style='white', context='notebook', palette='deep', rc={'figure.figsize':(10,8)})
print("loaded ...")


# In[ ]:


# Reproducibility
def set_seed(sd):
    seed(sd)
    np.random.seed(sd)
    tf.random.set_seed(sd)
    os.environ['PYTHONHASHSEED'] = str(sd)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
RandomSeed = 13
set_seed(RandomSeed)


# # EDA

# In[ ]:


train_data = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
test_data = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')
test_data['target'] = -1
train_data['Set'] = "Train"
test_data['Set'] = "Test"
DATA = train_data.append(test_data)
DATA.reset_index(inplace=True)
DATA.info()


# In[ ]:


features = [f for f in DATA.columns if "f_" in f]
float_features = [f for f in features if DATA[f].dtype == "float64"]
int_features = [f for f in features if DATA[f].dtype == "int64"]
str_features = [f for f in features if DATA[f].dtype == "object"]


# In[ ]:


DATA[int_features].describe()


# In[ ]:


DATA[float_features].describe()


# ### f_27

# In[ ]:


DATA[str_features].head(10)


# In[ ]:


def numSeq(x):
    prev = x[0]
    seq = 1
    for i in range(1, len(x)):
        if x[i] == prev:
            continue
        else:
            seq += 1
            prev = x[i]
    return seq


# In[ ]:


DATA['NumUnique'] = DATA['f_27'].apply(lambda row: len(set(row)))
#number of defferent sequences (excluded, worsens score)
DATA['NumSeq'] = DATA['f_27'].apply(numSeq)
features_from_string = ['NumUnique']
for c in range(10):
    DATA[f'ch{c}'] = DATA['f_27'].str.get(c).apply(ord) - ord('A')
    features_from_string.append(f'ch{c}')


# In[ ]:


DATA[[*str_features, *features_from_string]].head(10)


# Letter counts decrease LB score a little

# In[ ]:


# import string
# letters = list(string.ascii_uppercase)[:20]

# for L in letters:
#     DATA['count_'+L] = DATA['f_27'].str.count(L)

# features_from_string.extend([c for c in DATA.columns if c.startswith("count_")])


# In[ ]:


scaler = MinMaxScaler()
DATA[[*float_features,*int_features,*features_from_string]] = scaler.fit_transform(DATA[[*float_features,*int_features,*features_from_string]])


# In[ ]:


def plot_hist(features, title):
    N_cols = 8
    col_width = 4
    N_rows = round(len(features) / N_cols + 0.49)
    fig, axs = plt.subplots(nrows = N_rows, ncols=N_cols, figsize=(col_width * N_cols, N_rows * col_width))
    for i,f in enumerate(features):
        axs[i//N_cols, i%N_cols].hist(DATA[DATA.Set == 'Train'][f], bins=50);
        axs[i//N_cols, i%N_cols].set_title(f)
        axs[i//N_cols, i%N_cols].legend();


# In[ ]:


get_ipython().run_cell_magic('time', '', "plot_hist(float_features, 'float_features')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "plot_hist(int_features, 'int_features')")


# In[ ]:


MELT = pd.melt(DATA[DATA.Set == 'Train'][[*float_features,*int_features,*features_from_string,'target']], 
               value_vars = [*float_features,*int_features,*features_from_string],
               id_vars= 'target')


# In[ ]:


get_ipython().run_cell_magic('time', '', "ax = sns.displot(MELT, x='value', hue='target', col='variable', kind='kde',col_wrap= 5);\nax.set(xlim = (0,1), ylim = (0, 0.15));")


# In[ ]:


fig, ax = plt.subplots(figsize=(20,20)) 
ax = sns.heatmap(DATA[DATA.Set == 'Train'][[*int_features, *features_from_string,'target']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm");
ax.set_title("Target -  correlation to int, string features");


# In[ ]:


fig, ax = plt.subplots(figsize=(20,20)) 
ax = sns.heatmap(DATA[DATA.Set == 'Train'][[*float_features,'target']].corr(),annot=True, fmt = ".2f", cmap = "coolwarm");
ax.set_title("Target -  correlation to float features");


# In[ ]:


get_ipython().run_cell_magic('time', '', "#thanks for idead to: https://www.kaggle.com/code/ambrosm/tpsmay22-eda-which-makes-sense\nN_cols = 6\nN_rows = round((len([*float_features, *int_features]) / N_cols) + 0.49)\ncol_width = 4\nfig, axs = plt.subplots(N_rows, N_cols, figsize=(col_width * N_cols + 2, N_rows * col_width))\nfor f, ax in zip([*float_features, *int_features], axs.ravel()):\n    temp = pd.DataFrame({f: DATA[DATA.Set == 'Train'][f].values,'target': DATA[DATA.Set == 'Train'].target.values})\n    temp = temp.sort_values(f)\n    temp.reset_index(inplace=True)\n    ax.scatter(temp[f], temp.target.rolling(15000, center=True).mean(), s=2)\n    ax.set_xlabel(f'{f}')\nplt.suptitle('Target probability on single features')\nplt.show()")


# # Split

# In[ ]:


TRAIN = DATA[DATA.Set == 'Train']
TEST = DATA[DATA.Set == 'Test'][[*float_features, *int_features,*features_from_string]]
X = TRAIN[[*float_features, *int_features, *features_from_string]]
y = TRAIN.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = RandomSeed, stratify=y)


# # Models

# ### Auxilliary functions

# In[ ]:


def CM(y_test, val_pred, title):
    labels = [0,1]
    cm = confusion_matrix(y_test, val_pred, normalize = 'pred')
    cm_train = confusion_matrix(y_train, train_pred, normalize = 'pred')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,8))
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels= labels);
    disp_train.plot(ax=ax1, values_format='.1%', xticks_rotation='horizontal');
    disp_train.ax_.set_title('Train set', {'fontsize':20});

    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= labels);
    disp_test.plot(ax=ax2, values_format='.1%', xticks_rotation='horizontal');
    disp_test.ax_.set_title('Validation set',{'fontsize':20});
    fig.suptitle(title, fontsize=16);
    
def IMP(model, label, columns = X_train.columns):
    features = {}
    for feature, importance in zip(columns, model.feature_importances_):
        features[feature] = importance

    importances = pd.DataFrame({label:features})
    importances.sort_values(label, ascending = True, inplace=True)
    importances[:10].plot.barh()


# ## DNN

# In[ ]:


dnn_model = Sequential()
n_cols = X.shape[1]
dnn_model.add(Input(shape = (n_cols,), name = 'input'))
dnn_model.add(Dense(128, activation="swish", use_bias = True))
dnn_model.add(Dense(64, activation="swish", use_bias = True))
dnn_model.add(Dense(32, activation="swish", use_bias = True))
dnn_model.add(Dense(16, activation="swish", use_bias = True))
dnn_model.add(Dense(16, activation="swish", use_bias = True)) 
dnn_model.add(Dense(8, activation="swish", use_bias = True))
dnn_model.add(Dense(1, activation="sigmoid", name='out', use_bias = True))            
dnn_model.summary()


# In[ ]:


tf.keras.utils.plot_model(dnn_model, show_shapes=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'dnn_model.compile(loss=\'binary_crossentropy\', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-03), metrics=[\'binary_accuracy\',\'AUC\'])\nearly_stopping_monitor = EarlyStopping(patience=25, monitor=\'val_binary_accuracy\')\ncheckpoint = ModelCheckpoint("weights.hdf5", monitor = \'val_binary_accuracy\', save_best_only = True)\n# early_stopping_monitor = EarlyStopping(patience=25, monitor=\'AUC\')\n# checkpoint = ModelCheckpoint("weights.hdf5", monitor = \'AUC\', save_best_only = True)\ndnn_model.fit(X_train,y_train, \n              validation_data=(X_test,y_test), \n              callbacks=[checkpoint, early_stopping_monitor], \n              epochs=300, \n              batch_size=512, \n              verbose=0, \n              validation_split=0.25)\ndnn_model.load_weights("weights.hdf5")')


# In[ ]:


mtrcs = ['loss','binary_accuracy','auc']
fig, axs = plt.subplots(1, len(mtrcs), figsize=(30,10))
for i,ax in enumerate(axs.flatten()):
    train = mtrcs[i]
    test = "val_"+mtrcs[i];
    ax.plot(dnn_model.history.history[train], label='train')
    ax.plot(dnn_model.history.history[test], label = 'test')
    ax.set_title(train)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(train)
    ax.legend();


# In[ ]:


_, train_dnn_accuracy,train_dnn_auc = dnn_model.evaluate(X_train, y_train)
_, dnn_accuracy, dnn_auc = dnn_model.evaluate(X_test, y_test)
print('Train accuracy: {:.2f} %'.format(train_dnn_accuracy*100))
print('Accuracy: {:.2f} %'.format(dnn_accuracy*100))
print('Overfit: {:.2f} % '.format((train_dnn_accuracy - dnn_accuracy)*100))
print("Train AUC:",train_dnn_auc,"Test AUC:",dnn_auc)
# 96.64, 0.995


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_pred = np.rint(dnn_model.predict(X_train))\nval_pred = np.rint(dnn_model.predict(X_test))\nprint(classification_report(y_test, val_pred))')


# In[ ]:


CM(y_test, val_pred, 'DNN Classifier')


# In[ ]:


class DNN_wrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, df):
        pred = np.rint(self.model.predict(df))[:,0]
        return pred.astype(np.int32)
    def predict_proba(self, df):
        probs = self.model.predict(df)
        probs2 = np.ones_like(probs) - probs
        packed = np.concatenate((probs2, probs), axis=1)        
        return packed
    
DNN_MODEL = DNN_wrapper(dnn_model)


# # Submission

# In[ ]:


output = pd.DataFrame({'id': DATA[DATA.Set == 'Test']['id'], 'target': np.round(DNN_MODEL.predict_proba(TEST)[:,1],3)})
output.head(10)


# In[ ]:


#output
output.to_csv('submission.csv', index=False)
print("Submission was successfully saved!")


# In[ ]:


end_time = time.time()
print("Notebook run time: {:.1f} seconds. Finished at {}".format(end_time - start_time, datetime.now()) )

