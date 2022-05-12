#!/usr/bin/env python
# coding: utf-8

# 
# <h1> This notebooks aims to present different callbacks with learning rate function and customized early stopping

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from cycler import cycler
from IPython.display import display
import datetime
import scipy.stats

from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping, Callback
from tensorflow.keras.layers import Dense, Input, InputLayer, Add, Dropout, Embedding, Conv1D, Flatten, Concatenate,BatchNormalization,Reshape,Activation
from tensorflow.keras.utils import get_custom_objects
import tensorflow_addons as tfa
from keras import backend as K

import math
from math import pi
from math import cos
from math import floor

import gc


# In[ ]:


# Configuration :
BATCH_SIZE = 2048 
EPOCHS = 150
VERBOSE = 0


# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')

for df in [train, test]:
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
    # Next feature is from https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model
    df["unique_caracters"] = df.f_27.apply(lambda s: len(set(s)))
features = [f for f in test.columns if f != 'id' and f != 'f_27']


# In[ ]:


#features = [f for f in test.columns if f != 'id' and f != 'f_27']
rob = StandardScaler()
train[features] = pd.DataFrame(rob.fit_transform(train[features]),columns = features)
test[features] = pd.DataFrame(rob.transform(test[features]),columns = features)
train[features].shape,test[features].shape


# <h1> History display function

# In[ ]:


# from https://www.kaggle.com/code/ambrosm/tpsmay22-keras-quickstart
# Plot training history
def plot_history(history, *, n_epochs=None, plot_lr=False, title=None, bottom=None, top=None):
    """Plot (the last n_epochs epochs of) the training history
    
    Plots loss and optionally val_loss and lr."""
    plt.figure(figsize=(12, 6))
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


# <h1> Model function

# In[ ]:


def NN_model():
    
    inputs = Input(shape=(len(features)))
    
    A = 64
    REGUL = 6e-5
    activation = 'swish'
    INIT = "glorot_uniform"
    BIAS = True

    x1 =  Dense(A, 
                kernel_regularizer=tf.keras.regularizers.l2(REGUL),
                use_bias=BIAS,
                kernel_initializer = INIT,
                activation=activation
                )(inputs)

    x2 = Dense(A, 
                kernel_regularizer=tf.keras.regularizers.l2(REGUL),
                use_bias=BIAS,
                kernel_initializer = INIT,
                activation=activation
                )(x1)
    
    x21 = Concatenate()([x1,x2])
    x21 = Dropout(0.1)(x21)
    x21 = BatchNormalization()(x21)
    
    x3 = Dense(A, 
                kernel_regularizer=tf.keras.regularizers.l2(REGUL),
                use_bias=BIAS,
                kernel_initializer = INIT,
                activation=activation
                )(x21)

    x4 = Dense(16, 
                kernel_regularizer=tf.keras.regularizers.l2(REGUL),
                use_bias=BIAS,
                kernel_initializer = INIT,
                activation=activation
                )(x3)

    x5 = Dense(1, 
                use_bias=BIAS,
                activation='sigmoid',
                )(x4)
    
    model = Model(inputs, x5)
    
    LOSS = tf.keras.losses.BinaryCrossentropy()
    METRIC = tf.keras.metrics.AUC(name = 'auc')
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                          metrics= METRIC,
                          loss=tf.keras.losses.BinaryCrossentropy())
    
    return model


# <h1> Training function

# In[ ]:


def training(callbacks,BREAK = True, Cosine_Annealing = False):
    
    LOSS = tf.keras.losses.BinaryCrossentropy()
    METRIC = tf.keras.metrics.AUC(name = 'auc')
    nn_oof = np.zeros(train.shape[0])

    split = 5
    cv = StratifiedKFold(n_splits=split, shuffle=True, random_state=2)

    for fold, (idx_train, idx_valid) in enumerate(cv.split(train, train.target)):

        X_train, y_train = train.loc[idx_train][features], train.target.iloc[idx_train]
        X_valid, y_valid = train.loc[idx_valid][features], train.target.iloc[idx_valid]
        
        #---------  Model instantiation ------------------------

        model = NN_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                          metrics= METRIC,
                          loss=tf.keras.losses.BinaryCrossentropy())
        
        # Model Training :
        
        history = model.fit(X_train, y_train, 
                            validation_data=(X_valid,y_valid),
                            epochs=EPOCHS,
                            verbose=VERBOSE,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            callbacks=[callbacks]
                           )
                
        history_list = []
        history_list.append(history.history)

        print(f"\nTraining loss    ", np.round((history_list[0].get('loss')[-1]),5))
        print(f"Training val_loss", np.round((history_list[0].get('val_loss')[-1]),5))
        print(f"Training delta   ", np.round(100 * (history_list[0].get('loss')[-1] - history_list[0].get('val_loss')[-1])/history_list[0].get('loss')[-1],2),'%')
        
        if Cosine_Annealing :
            pred_list =[]
            for i in range(1, int(n_cycles+1)):
                file_name ='snapshot_model_'+str(i)+'.h5'
                model.load_weights(file_name)
                pred = model.predict(train[features]).squeeze()
                pred_list.append(pred)
                score = roc_auc_score(train.target, pred)
                print(f"cycle {i} Score: {score}")
        else : 
            model.load_weights('best_model.h5')
            pred_oof_nn = model.predict(X_valid)
            score = roc_auc_score(y_valid, pred_oof_nn)
            nn_oof[idx_valid] = pred_oof_nn.squeeze()
            print(f"\nFold: {fold + 1} NN1 Score: {score}")
        
        tf.keras.backend.clear_session()
        
        if BREAK :
            break

    if not BREAK :
        print(f"\n TOTAL oof NN1 = {roc_auc_score(train.target,nn_oof)}")
    
    return history_list, score


# <h1> Early Stopping customized

# The usual earlystopping is based upon loss or val_loss (better) or the metric choiced (auc..).
# The idea was to introduce an earlystopping based upon the difference between loss and val_loss. There is a loss limit value when start the patience calculation.
# Below the limit, no patience is recorded and when the loss has reached the limit (small value) the patience calculation starts : every val_loss - loss > 0 is recorded and after the number of patience has been reached : earlystopping.
# The goal is to avoid an overfitting risk when the val_loss increases or decreases very slowly and the loss still decreases quickly.
# The earlystopping can (verbose = 1) display the difference (val_loss - loss) at each epoch end.

# In[ ]:


# From Keras documentation :
class Early_Stopping_the_war(tf.keras.callbacks.Callback):

    def __init__(self,patience = 0,verbose = 1,loss_limit = 0.075 ):
        super(Early_Stopping_the_war, self).__init__()
        self.patience = patience
        self.best_weights = None
        self.verbose = verbose
        self.loss_limit = loss_limit

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_loss = np.Inf
        self.best_val_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        
        current_loss = logs.get("loss")
        current_val_loss = logs.get("val_loss")
        
        if np.less(current_loss, self.best_loss):
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()

        if self.verbose == 1 :
            print(15*' ','val_loss-loss:{}'.format(np.round((current_val_loss - current_loss),5)))

        if np.less(current_loss,current_val_loss) and current_loss < self.loss_limit :
            self.wait += 1
            self.best_val_loss = current_val_loss
            if self.verbose == 1 :
                print(15*' ',"val_loss > loss => patience :",self.wait) 

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
                    
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


# <h1> Training with Early_stopping_The_War monitored by val_loss-loss

# In[ ]:



# plateau monitored by val_loss if X_val only
plateau_val_loss = tf.keras.callbacks.ReduceLROnPlateau(
                                        monitor = 'val_loss', 
                                        factor = 0.9, 
                                        patience = 3, 
                                        verbose = VERBOSE, 
                                        mode = 'auto')
# plateau monitored by loss 
plateau_loss = tf.keras.callbacks.ReduceLROnPlateau(
                                        monitor ='loss', 
                                        factor = 0.95, 
                                        patience = 4, 
                                        verbose = VERBOSE, 
                                        mode ='max')

ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
                                        'best_model.h5', 
                                        monitor = 'val_loss',
                                        save_weights_only = True,
                                        save_best_only = True,
                                        mode = 'auto')


es = Early_Stopping_the_war(patience = 5,verbose = VERBOSE, loss_limit = 0.074)
callbacks = [es, plateau_val_loss,ModelCheckpoint]
history_list_plateau, score_plateau = training(callbacks)
gc.collect()


# <h1> Cosine learning rate decay

# In[ ]:


# from https://www.kaggle.com/code/ambrosm/tpsmay22-keras-quickstart
epochs = EPOCHS
lr_start=0.01
lr_end=0.0002
def cosine_decay(epoch):
    if epochs > 1:
        w = (1 + math.cos(epoch / (epochs-1) * math.pi)) / 2
    else:
        w = 1
    return w * lr_start + (1 - w) * lr_end

Learning_Rate_Scheduler = LearningRateScheduler(cosine_decay, verbose=0)
callbacks = [Learning_Rate_Scheduler, tf.keras.callbacks.TerminateOnNaN(),ModelCheckpoint]
history_list_cosine, score_cosine = training(callbacks)
gc.collect()


# <h1> Cosine annealing learning rate

# In[ ]:


# From Jason Brownlee : https://machinelearningmastery.com/snapshot-ensemble-deep-learning-neural-network/

class Cosine_Annealing(Callback):

    def __init__(self, n_epochs, n_cycles, lrate_max):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.lrates = list()
 
    # calculate learning rate for epoch
    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        epochs_per_cycle = floor(n_epochs/n_cycles)
        cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max/2 * (cos(cos_inner) + 1)

    # calculate and set learning rate at the start of the epoch
    def on_epoch_begin(self, epoch, logs={}):
        # calculate learning rate
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
        K.set_value(self.model.optimizer.lr, lr)
        # set learning rate
        #backend.set_value(self.model.optimizer.lr, lr)
        #log value
        self.lrates.append(lr)
 
    # save models at the end of each cycle
    def on_epoch_end(self, epoch, logs={}):
        # check if we can save model
        epochs_per_cycle = floor(self.epochs / self.cycles)
        if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
            # save model to file
            filename = "snapshot_model_%d.h5" % int((epoch + 1) / epochs_per_cycle)
            self.model.save(filename)
            
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = 'snapshot_model_' + str(i + 1) + '.h5'
        # load model from file
        MODEL = model.load_weights(filename)
        # add to list of members
        all_models.append(MODEL)
    print('>loaded %s' % filename)
    return all_models
            


# In[ ]:


EPOCHS = 150
n_cycles = EPOCHS /50 
ca = Cosine_Annealing(EPOCHS, n_cycles, 0.01)
callbacks = [ca]
history_list_Cosine_Annealing, score_Cosine_Annealing = training(callbacks,Cosine_Annealing = True)


# <h1> Let's compare the training history

# In[ ]:


plot_history(
    history_list_plateau[0],
    title=f"Early_stopping_The_War is monitored by val_loss-loss, AUC  = {score_plateau:.5f}",
    plot_lr=True, n_epochs=EPOCHS) 

plot_history(
    history_list_cosine[0],
    title=f"Cosine learning rate decay, AUC  = {score_cosine:.5f}",
    plot_lr=True, n_epochs=EPOCHS) 

plot_history(
    history_list_Cosine_Annealing[0],
    title=f"Cosine Annealing, AUC  = {score_Cosine_Annealing:.5f}",
    plot_lr=True, n_epochs=EPOCHS)


# <h3> Each strategy is effective, but depending on each dataset, there is sometimes a better one.
#     Cosine annealing is my favorite but takes a lot of time when you select many cycles...

# <h1> Training for submission

# In[ ]:


RUN = 3

EPOCHS = 400

blend_train_best = np.zeros((train.shape[0],1))
blend_test = np.zeros((test.shape[0],1))
blend_train = np.zeros((train.shape[0],1))

for i in range(RUN): 
    
    print('\n_______________ RUN {}  _____________\n'.format(i+1))
    
    #--------- NN1 -------------------------
    model = NN_model()
    n_cycles = EPOCHS / 100
    ca = Cosine_Annealing(EPOCHS, n_cycles, 0.01)
    
    model.fit(train[features], train.target,  
                epochs=EPOCHS,
                verbose=0,
                batch_size=BATCH_SIZE,
                shuffle=True,
                 callbacks=[ca]
             )
    
    pred_list =[]
    for i in range(1, int(n_cycles+1)):
        file_name ='snapshot_model_'+str(i)+'.h5'
        model.load_weights(file_name)
        pred = model.predict(train[features]).squeeze()
        pred_list.append(pred)
        score = roc_auc_score(train.target, pred)
        print(f"cycle {i} Score: {score}")
        
        # --------- test prediction --------------
        pred_test = model.predict(test[features]).squeeze()
        blend_test[:,0] += (pred_test/n_cycles)/RUN
        
        # ------ train prediction (no oof) -------
        pred_train = model.predict(train[features]).squeeze()
        blend_train[:,0] += (pred_train/n_cycles)/RUN
        
    score = roc_auc_score(train.target, np.mean(pred_list,axis=0))
    print(f"All cycles Score: {score}")
    
print('\nFINAL non oof AUC = ',roc_auc_score(train.target,blend_train[:,0]))


# In[ ]:


sub = pd.read_csv('../input/tabular-playground-series-may-2022/sample_submission.csv')
sub['target'] = blend_test
sub.to_csv('submission.csv', index = False)
pd.read_csv('submission.csv')


# In[ ]:




