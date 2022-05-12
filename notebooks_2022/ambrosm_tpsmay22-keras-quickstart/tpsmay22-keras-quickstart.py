#!/usr/bin/env python
# coding: utf-8

# # Keras Quickstart for TPSMAY22
# 
# This notebook shows how to train a Keras model with minimal feature engineering. For the corresponding EDA, see the [separate EDA notebook](https://www.kaggle.com/code/ambrosm/tpsmay22-eda-which-makes-sense).
# 
# Release notes:
# - V2: Input scaling, more hidden layers

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
import math
import random

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import CalibrationDisplay
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense, Input, InputLayer, Add
from tensorflow.keras.utils import plot_model

plt.rcParams['axes.facecolor'] = '#0057b8' # blue
plt.rcParams['axes.prop_cycle'] = cycler(color=['#ffd700'] +
                                         plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])


# In[ ]:


# Plot training history
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
    


# # Feature engineering
# 
# We read the data and apply minimal feature engineering: We only split the `f_27` string into ten separate features as described in the [EDA](https://www.kaggle.com/code/ambrosm/tpsmay22-eda-which-makes-sense), and we count the unique characters in the string.

# In[ ]:


train = pd.read_csv('../input/tabular-playground-series-may-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-may-2022/test.csv')
for df in [train, test]:
    # Extract the 10 letters from f_27 into individual features
    for i in range(10):
        df[f'ch{i}'] = df.f_27.str.get(i).apply(ord) - ord('A')
        
    # unique_characters feature is from https://www.kaggle.com/code/cabaxiom/tps-may-22-eda-lgbm-model
    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))
    
features = [f for f in test.columns if f != 'id' and f != 'f_27']
test[features].head(2)


# # The model
# 
# The model in version 1 of this notebook had only two hidden layers (because of a bug) and underfitted. In version 2, the model has four hidden layers and could overfit. To counter overfitting, I added a kernel_regularizer to all hidden layers.
# 

# In[ ]:


def my_model():
    """Simple sequential neural network with three hidden layers.
    
    Returns a (not yet compiled) instance of tensorflow.keras.models.Model.
    """
    activation = 'swish'
    inputs = Input(shape=(len(features)))
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              activation=activation,
             )(inputs)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              activation=activation,
             )(x)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              activation=activation,
             )(x)
    x = Dense(16, kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              activation=activation,
             )(x)
    x = Dense(1, #kernel_regularizer=tf.keras.regularizers.l2(1e-6),
              activation='sigmoid',
             )(x)
    model = Model(inputs, x)
    return model

plot_model(my_model(), show_layer_names=False, show_shapes=True)


# # Cross-validation
# 
# For cross-validation, we use a simple KFold with five splits. It has turned out that the scores of the five splits are very similar so that I usually run only the first split. This one split is good enough to evaluate the model.
# 
# I like to first train the model with early stopping to see what are good initial and final learning rates and the number of epochs, and then I switch to cosine learning rate decay. You can switch back to early stopping anytime by setting the parameter `USE_PLATEAU`.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Cross-validation of the classifier\n\nEPOCHS = 200\nEPOCHS_COSINEDECAY = 100\nVERBOSE = 0 # set to 0 for less output, or to 2 for more output\nDIAGRAMS = True\nUSE_PLATEAU = False\nBATCH_SIZE = 4096\n\n# see https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development\nnp.random.seed(1)\nrandom.seed(1)\ntf.random.set_seed(1)\n\ndef fit_model(X_tr, y_tr, X_va=None, y_va=None, run=0):\n    """Scale the data, fit a model, plot the training history and optionally validate the model\n    \n    Returns a trained instance of tensorflow.keras.models.Model.\n    \n    As a side effect, updates y_va_pred, history_list and score_list.\n    """\n    global y_va_pred\n    start_time = datetime.datetime.now()\n    \n    scaler = StandardScaler()\n    X_tr = scaler.fit_transform(X_tr)\n    \n    if X_va is not None:\n        X_va = scaler.transform(X_va)\n        validation_data = (X_va, y_va)\n    else:\n        validation_data = None\n\n    # Define the learning rate schedule and EarlyStopping\n    lr_start=0.01\n    if USE_PLATEAU and X_va is not None: # use early stopping\n        epochs = EPOCHS\n        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.7, \n                               patience=4, verbose=VERBOSE)\n        es = EarlyStopping(monitor="val_loss",\n                           patience=12, \n                           verbose=1,\n                           mode="min", \n                           restore_best_weights=True)\n        callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]\n\n    else: # use cosine learning rate decay rather than early stopping\n        epochs = EPOCHS_COSINEDECAY\n        lr_end=0.0002\n        def cosine_decay(epoch):\n            if epochs > 1:\n                w = (1 + math.cos(epoch / (epochs-1) * math.pi)) / 2\n            else:\n                w = 1\n            return w * lr_start + (1 - w) * lr_end\n\n        lr = LearningRateScheduler(cosine_decay, verbose=0)\n        callbacks = [lr, tf.keras.callbacks.TerminateOnNaN()]\n        \n    # Construct and compile the model\n    model = my_model()\n    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_start),\n                  #metrics=\'acc\',\n                  loss=tf.keras.losses.BinaryCrossentropy())\n    #model.compile(optimizer=tf.keras.optimizers.SGD(), loss=\'mse\')\n\n    # Train the model\n    history = model.fit(X_tr, y_tr, \n                        validation_data=validation_data, \n                        epochs=epochs,\n                        verbose=VERBOSE,\n                        batch_size=BATCH_SIZE,\n                        shuffle=True,\n                        callbacks=callbacks)\n\n    history_list.append(history.history)\n    callbacks, es, lr, history = None, None, None, None\n    print(f"Training loss:   {history_list[-1][\'loss\'][-1]:.3f}")\n    \n    if X_va is not None:\n        # Inference for validation\n        y_va_pred = model.predict(X_va, batch_size=BATCH_SIZE, verbose=VERBOSE)\n        #oof_list[run][val_idx] = y_va_pred\n        \n        # Evaluation: Execution time and AUC\n        score = roc_auc_score(y_va, y_va_pred)\n        print(f"Fold {run}.{fold} | {str(datetime.datetime.now() - start_time)[-12:-7]}"\n              f" | AUC: {score:.5f}")\n        score_list.append(score)\n        \n        if DIAGRAMS and fold == 0 and run == 0:\n            # Plot training history\n            plot_history(history_list[-1], \n                         title=f"Learning curve (validation AUC = {score:.5f})",\n                         plot_lr=True, n_epochs=110)\n\n            # Plot y_true vs. y_pred\n            plt.figure(figsize=(10, 4))\n            plt.hist(y_va_pred[y_va == 0], bins=np.linspace(0, 1, 21),\n                     alpha=0.5, density=True)\n            plt.hist(y_va_pred[y_va == 1], bins=np.linspace(0, 1, 21),\n                     alpha=0.5, density=True)\n            plt.xlabel(\'y_pred\')\n            plt.ylabel(\'density\')\n            plt.title(\'OOF Predictions\')\n            plt.show()\n\n    return model, scaler\n\n\nprint(f"{len(features)} features")\nhistory_list = []\nscore_list = []\nkf = KFold(n_splits=5)\nfor fold, (idx_tr, idx_va) in enumerate(kf.split(train)):\n    X_tr = train.iloc[idx_tr][features]\n    X_va = train.iloc[idx_va][features]\n    y_tr = train.iloc[idx_tr].target\n    y_va = train.iloc[idx_va].target\n    \n    fit_model(X_tr, y_tr, X_va, y_va)\n    break # we only need the first fold\n\nprint(f"OOF AUC:                       {np.mean(score_list):.5f}")')


# # Three diagrams for model evaluation
# 
# We plot the ROC curve just because it looks nice. The area under the red curve is the score of our model.
# 

# In[ ]:


# Plot the roc curve for the last fold
def plot_roc_curve(y_va, y_va_pred):
    plt.figure(figsize=(8, 8))
    fpr, tpr, _ = roc_curve(y_va, y_va_pred)
    plt.plot(fpr, tpr, color='r', lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.gca().set_aspect('equal')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.show()

plot_roc_curve(y_va, y_va_pred)


# Second, we plot a histogram of the out-of-fold predictions. Many predictions are near 0.0 or near 1.0; this means that in many cases the classifier's predictions have high confidence:

# In[ ]:


plt.figure(figsize=(12, 4))
plt.hist(y_va_pred, bins=25, density=True)
plt.title('Histogram of the oof predictions')
plt.show()


# Finally, we plot the calibration curve. The curve here is almost a straight line, which means that the predicted probabilities are almost exact: 

# In[ ]:


plt.figure(figsize=(12, 4))
CalibrationDisplay.from_predictions(y_va, y_va_pred, n_bins=50, strategy='quantile', ax=plt.gca())
plt.title('Probability calibration')
plt.show()


# # Submission
# 
# For the submission, we re-train the model on the complete training data with several different seeds and then submit the mean of the predicted ranks.

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Create submission\nprint(f"{len(features)} features")\n\nX_tr = train[features]\ny_tr = train.target\n\npred_list = []\nfor seed in range(10):\n    # see https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development\n    np.random.seed(seed)\n    random.seed(seed)\n    tf.random.set_seed(seed)\n    model, scaler = fit_model(X_tr, y_tr, run=seed)\n    pred_list.append(scipy.stats.rankdata(model.predict(scaler.transform(test[features]),\n                                                        batch_size=BATCH_SIZE, verbose=VERBOSE)))\n    print(f"{seed:2}", pred_list[-1])\nprint()\nsubmission = test[[\'id\']].copy()\nsubmission[\'target\'] = np.array(pred_list).mean(axis=0)\nsubmission.to_csv(\'submission.csv\', index=False)\nsubmission')


# # What next?
# 
# Now it's your turn! Try to improve this model by
# - Changing the network architecture 
# - Engineering more features
# - Tuning hyperparameters, optimizers, learning rate schedules and so on...
# 
# Or, if you prefer gradient boosting, you can have a look at the [Gradient Boosting Quickstart](https://www.kaggle.com/ambrosm/tpsmay22-gradient-boosting-quickstart).
# 
