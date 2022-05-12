#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Hey, thanks for viewing my Kernel!
# 
# If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! :)
# 
# 🎉 My 100. notebook! 🎉

# In[ ]:


import numpy as np
import pandas as pd
import warnings

warnings.simplefilter("ignore")
train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
sub = pd.read_csv("../input/tabular-playground-series-may-2022/sample_submission.csv")
display(train.head())
display(test.head())
display(sub.head())


# # Feature Engineering

# In[ ]:


def create_features(data):
    object_data_cols = [f"f_27_{i+1}" for i in range(10)]
    object_data = pd.DataFrame(data['f_27'].apply(list).tolist(), columns=object_data_cols)
    for feature in object_data_cols:
        object_data[feature] = object_data[feature].apply(ord) - ord('A')
    
    data = pd.concat([data, object_data], 1)
    data["unique_characters"] = data.f_27.apply(lambda s: len(set(s)))
    
    ## sum
    # float
    data['f_sum_2'] = (data['f_21']+data['f_22'])
    data['f_sum_3'] = (data['f_23']-data['f_20'])
    
    continuous_feat = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_19', 'f_20', 'f_21', 'f_22', 
                       'f_23', 'f_24', 'f_25', 'f_26', 'f_28']
    
    data['f_sum']  = data[continuous_feat].sum(axis=1)
    data['f_min']  = data[continuous_feat].min(axis=1)
    data['f_max']  = data[continuous_feat].max(axis=1)
    data['f_std']  = data[continuous_feat].std(axis=1)    
    data['f_mad']  = data[continuous_feat].mad(axis=1)
    data['f_mean'] = data[continuous_feat].mean(axis=1)
    data['f_kurt'] = data[continuous_feat].kurt(axis=1)
    data['f_count_pos']  = data[continuous_feat].gt(0).count(axis=1)
    
    # int
    data['f_sum_10'] = (data['f_07']-data['f_10'])
    data['f_sum_13'] = (data['f_08']-data['f_10'])
    
    return data


# In[ ]:


train_fe = create_features(train.copy())
test_fe = create_features(test.copy())
train_fe.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_tr = pd.DataFrame(scaler.fit_transform(train_fe.drop(['id', 'f_27', "target"], 1)), columns=train_fe.drop(['id', 'f_27', "target"], 1).columns)
test_tr = pd.DataFrame(scaler.transform(test_fe.drop(['id', 'f_27'], 1)), columns=train_fe.drop(['id', 'f_27', "target"], 1).columns)
train_tr.head()


# # Modeling

# In[ ]:


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense, Input, InputLayer, Add, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import clone_model

initial_learning_rate = 0.01
n_epoch = 200

features = train_fe.drop(['id', 'f_27', 'target'], 1).columns
def tf_model():
    activation = 'swish'
    inputs = Input(shape=(len(features)))
    x = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              activation=activation,
             )(inputs)
    #x = Dropout(0.25)(x)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              activation=activation,
             )(x)
    #x = Dropout(0.25)(x)
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              activation=activation,
             )(x)
    #x = Dropout(0.25)(x)
    x = Dense(32, kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              activation=activation,
             )(x)
    #x = Dropout(0.25)(x)
    x = Dense(16, kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              activation=activation,
             )(x)
    x = Dense(8, kernel_regularizer=tf.keras.regularizers.l2(30e-6),
              activation=activation,
             )(x)
    #x = Dropout(0.25)(x)
    x = Dense(1, #kernel_regularizer=tf.keras.regularizers.l2(1e-6),
              activation='sigmoid',
             )(x)
    model = Model(inputs, x)
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate,
                    decay_steps=n_epoch/2,
                    decay_rate=0.995,
                    staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.1)
    lr_metric = get_lr_metric(optimizer)
    
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(), metrics=[lr_metric])
    return model


# In[ ]:


plot_model(tf_model(), show_layer_names=False, show_shapes=True)


# # Automated Blending Function

# In[ ]:


def automated_blending(model_ori, X, y, X_test, nfold=10, plot=True, figsize=(16, 8), tf=False, verbose=0):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    
    train_preds = []
    train_targets = []
    auc = []
    skf = StratifiedKFold(n_splits=nfold)
    test_preds = 0
    if tf:
        ncols=5
        nrows=round(nfold/ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, round(nrows*16/ncols)))
        col_i, row_i = 0, 0
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold: {fold+1},", end=' ')
        X_train, X_valid = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_valid = y.iloc[train_idx, :], y.iloc[test_idx, :]
        
        if tf:
            model = model_ori()
        else:
            model = clone(model_ori)
            
        try:
            if tf:
                es = EarlyStopping(monitor="val_loss",
                                   patience=24, 
                                   verbose=verbose,
                                   mode="min", 
                                   restore_best_weights=True)
                callbacks = [es]
                history = model.fit(X_train, y_train, 
                                    validation_data=(X_valid, y_valid), 
                                    epochs=n_epoch,
                                    verbose=verbose,
                                    batch_size=4096,
                                    shuffle=True,
                                    callbacks=callbacks)
                preds = model.predict(X_valid)
                pd.DataFrame(history.history, columns=["loss", "val_loss"]).plot(ax=axes[row_i][col_i])
                col_i += 1
                if col_i == ncols:
                    col_i = 0
                    row_i += 1
            else:
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=20, verbose=verbose)
                preds = model.predict_proba(X_valid)
        except:
            print("Warnings, model has not eval func...")
            model.fit(X_train, y_train)
            preds = model.predict_proba(X_valid)
        
        if tf:
            auc_score = roc_auc_score(y_valid.values, preds)

            auc.append(auc_score/nfold)
            train_preds.extend(preds[:, 0])
            train_targets.extend(y_valid.values)

            tpreds = model.predict(X_test)
            test_preds += tpreds/nfold
        else:
            auc_score = roc_auc_score(y_valid.values, preds[:, 1])

            auc.append(auc_score/nfold)
            train_preds.extend(preds[:, 1])
            train_targets.extend(y_valid.values)

            tpreds = model.predict_proba(X_test)
            test_preds += tpreds[:, 1]/nfold
        print(f'auc: {round(auc_score, 5)}')
    
    print(f"auc mean: {sum(auc)}")
    df_preds = pd.DataFrame()
    df_preds['pred'] = train_preds
    df_preds['label'] = np.array(train_targets).reshape(-1)
    
    zero_mean = df_preds.loc[df_preds['label']==0, 'pred'].mean()
    zero_median = df_preds.loc[df_preds['label']==0, 'pred'].median()
    
    one_mean = df_preds.loc[df_preds['label']==1, 'pred'].mean()
    one_median = df_preds.loc[df_preds['label']==1, 'pred'].median()
    
    
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        palette ={0: "blue", 1: "red"}
        sns.kdeplot(data=df_preds, x='pred', hue='label', ax=ax, palette=palette)
        plt.axvline(x=zero_mean, color='blue', label=f'mean-zero-{round(zero_mean, 3)}', ls='--')
        plt.axvline(x=zero_median, color='blue', label=f'median-zero-{round(zero_median, 3)}', ls=':')
        
        plt.axvline(x=one_mean, color='red', label=f'mean-one-{round(one_mean, 3)}', ls='--')
        plt.axvline(x=one_median, color='red', label=f'median-one-{round(one_median, 3)}', ls=':')
        plt.legend()
        plt.show()
    
    return test_preds


# In[ ]:


from lightgbm import LGBMClassifier

test_preds = automated_blending(tf_model, 
                                train_tr, train_fe[['target']], 
                                test_tr, nfold=15, tf=True, verbose=0)


# In[ ]:


pre_sub1 = pd.read_csv("../input/tpsmay22-keras-quickstart/submission.csv")
pre_sub2 = pd.read_csv("../input/tps-may22-eda-neuronal-nets/my_submission_050722.csv")


# In[ ]:


import scipy
pre_sub2['target'] = scipy.stats.rankdata(pre_sub2['target'])
sub['target'] = scipy.stats.rankdata(test_preds)
sub['target'] = sub['target'] * 0.3 + pre_sub1['target'] * 0.3 + pre_sub2['target'] * 0.4
sub.to_csv("submission.csv", index=False)


# In[ ]:


sub.head()


# # References
# 1. [notebook](https://www.kaggle.com/code/ambrosm/tpsmay22-keras-quickstart/notebook?scriptVersionId=94617937)
# 2. [notebook](https://www.kaggle.com/code/cv13j0/tps-may22-eda-neuronal-nets/notebook?scriptVersionId=95032660)
