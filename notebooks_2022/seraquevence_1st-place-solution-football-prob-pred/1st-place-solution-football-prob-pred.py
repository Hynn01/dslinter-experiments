#!/usr/bin/env python
# coding: utf-8

# ## First Place Solution: Football Match Probability Prediction
# 
# Firstly, I would like to thank **Octosport** and **Sportmonks** for organizing a very interesting competition. I have never analyzed sports data before, but thanks to the tutorial notebooks, I was able to learn a lot trying to predict the results of this near random game.
# 
# My solution is based on the notebooks that I made public before, no difference at all. The only change was the inclusion of features A to C and X to Z which were also public (LGB model https://www.kaggle.com/code/seraquevence/top-x-football-prob-prediction-lgbm-v01). They are very powerful reducing the loss by almost 0.001 (0.996 to 0.995). The LGBM helped me a lot to figure out what were the best features to include.
# 
# Not being able to reduce the log loss more, I started to stack the same LSTM models with small modifications up to the point that the log-loss started to increanse again. In the end, the final result was the average of four LSTM models.
# 
# I have tried many things that didn't work like many feature engineering (multiplications, divisions, difference, sum, moving average, EWMA, etc), various Neural Network Architectures, league id aggregations, data augmentation, target engineering and so on. I would like to explore correlations between teams/results/leagues or different models for first/second/... division but I didn't have time.

# ## Import modules and loading the data
# First let's import modules, the training and the test set. The test set contains the same columns than the training set without the target.

# In[ ]:


# Import libraries
# import gc
import numpy as np
import pandas as pd
import datetime as dt
#
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss
#
import tensorflow as tf
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import layers


# In[ ]:


#loading the data
train = pd.read_csv('/kaggle/input/football-match-probability-prediction/train.csv')
test = pd.read_csv('/kaggle/input/football-match-probability-prediction/test.csv')
submission = pd.read_csv('/kaggle/input/football-match-probability-prediction/sample_submission.csv')


# ## Parameters
# Set some parameters that will be used later.

# In[ ]:


# Set seed
SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)

#Some parameters
MASK = -666 # fill NA with -666 (the number of the beast)
T_HIST = 10 # time history, last 10 games
CLASS = 3 #number of classes (home, draw, away)

DEBUG = False
# Run on a small sample of the data
if DEBUG:
    train = train[:10000]


# In[ ]:


# exclude matches with no history at date 1 - full of NA (1159 rows excluded)
train.dropna(subset=['home_team_history_match_date_1'], inplace = True)


# In[ ]:


print(f"Train: {train.shape} \n Submission: {submission.shape}")
train.head()


# ## Feature Engineering
# The **most** important part of Machine Learning.

# In[ ]:


# for cols "date", change to datatime 
for col in train.filter(regex='date', axis=1).columns:
    train[col] = pd.to_datetime(train[col])
    test[col] = pd.to_datetime(test[col])

# Some feature engineering
def add_features(df):
    for i in range(1, 11): # range from 1 to 10
        # Feat. difference of days
        df[f'home_team_history_match_DIFF_day_{i}'] = (df['match_date'] - df[f'home_team_history_match_date_{i}']).dt.days
        df[f'away_team_history_match_DIFF_days_{i}'] = (df['match_date'] - df[f'away_team_history_match_date_{i}']).dt.days
    # Feat. difference of scored goals
        df[f'home_team_history_DIFF_goal_{i}'] = df[f'home_team_history_goal_{i}'] - df[f'home_team_history_opponent_goal_{i}']
        df[f'away_team_history_DIFF_goal_{i}'] = df[f'away_team_history_goal_{i}'] - df[f'away_team_history_opponent_goal_{i}']
    # Feat dummy winner x loser
        df[f'home_winner_{i}'] = np.where(df[f'home_team_history_DIFF_goal_{i}'] > 0, 1., 0.) 
        df[f'home_loser_{i}'] = np.where(df[f'home_team_history_DIFF_goal_{i}'] < 0, 1., 0.)
        df[f'away_winner_{i}'] = np.where(df[f'away_team_history_DIFF_goal_{i}'] > 0, 1., 0.)
        df[f'away_loser_{i}'] = np.where(df[f'away_team_history_DIFF_goal_{i}'] < 0, 1., 0.)
    # Results: multiple nested where # away:0, draw:1, home:2
        df[f'home_team_result_{i}'] = np.where(df[f'home_team_history_DIFF_goal_{i}'] > 0., 2.,
                         (np.where(df[f'home_team_history_DIFF_goal_{i}'] == 0., 1,
                                   np.where(df[f'home_team_history_DIFF_goal_{i}'].isna(), np.nan, 0))))
        df[f'away_team_result_{i}'] = np.where(df[f'away_team_history_DIFF_goal_{i}'] > 0., 2.,
                         (np.where(df[f'away_team_history_DIFF_goal_{i}'] == 0., 1.,
                                   np.where(df[f'away_team_history_DIFF_goal_{i}'].isna(), np.nan, 0.))))
    # Feat. difference of rating ("modified" ELO RATING)
        df[f'home_team_history_ELO_rating_{i}'] = 1/(1+10**((df[f'home_team_history_opponent_rating_{i}']-df[f'home_team_history_rating_{i}'])/10))
        df[f'away_team_history_ELO_rating_{i}'] = 1/(1+10**((df[f'away_team_history_opponent_rating_{i}']-df[f'away_team_history_rating_{i}'])/10))
        df[f'home_away_team_history_ELO_rating_{i}'] = 1/(1+10**((df[f'away_team_history_rating_{i}']-df[f'home_team_history_rating_{i}'])/10))
        # df[f'away_team_history_DIFF_rating_{i}'] =  - df[f'away_team_history_opponent_rating_{i}']
    # Feat. same coach id
        df[f'home_team_history_SAME_coaX_{i}'] = np.where(df['home_team_coach_id']==df[f'home_team_history_coach_{i}'],1,0)
        df[f'away_team_history_SAME_coaX_{i}'] = np.where(df['away_team_coach_id']==df[f'away_team_history_coach_{i}'],1,0) 
    # Feat. same league id
        df[f'home_team_history_SAME_leaG_{i}'] = np.where(df['league_id']==df[f'home_team_history_league_id_{i}'],1,0)
        df[f'away_team_history_SAME_leaG_{i}'] = np.where(df['league_id']==df[f'away_team_history_league_id_{i}'],1,0) 
    # more features
        df[f'feature_A_{i}'] = df[f'home_team_history_ELO_rating_{i}'] * df[f'home_team_history_is_play_home_{i}']* df[f'home_team_history_SAME_leaG_{i}']
        df[f'feature_B_{i}'] = df[f'away_team_history_ELO_rating_{i}'] * df[f'away_team_history_is_play_home_{i}']* df[f'away_team_history_SAME_leaG_{i}']
        df[f'feature_C_{i}'] = df[f'home_away_team_history_ELO_rating_{i}'] * df[f'home_team_history_is_play_home_{i}'] * df[f'home_team_history_SAME_leaG_{i}']
        df[f'feature_X_{i}'] = df[f'home_team_history_ELO_rating_{i}'] * df[f'home_team_history_SAME_leaG_{i}']
        df[f'feature_Y_{i}'] = df[f'away_team_history_ELO_rating_{i}'] * df[f'away_team_history_SAME_leaG_{i}']
        df[f'feature_Z_{i}'] = df[f'home_away_team_history_ELO_rating_{i}'] * df[f'home_team_history_SAME_leaG_{i}']
    # Fill NA with -666
    # df.fillna(MASK, inplace = True)
    return df

train = add_features(train)
test = add_features(test)


# ## Scaling and Reshape
# The input/output of the lstm is very trick. It is expected an array of dimensions (Matches/Batch, Time history, Features). I hope I did it right.

# In[ ]:


# save targets
# train_id = train['id'].copy()
train_y = train['target'].copy()
#keep only some features
train_x = train.drop(['target', 'home_team_name', 'away_team_name'], axis=1) #, inplace=True) # is_cup EXCLUDED
# Exclude all date, league, coach columns
train_x.drop(train.filter(regex='date').columns, axis=1, inplace = True)
train_x.drop(train.filter(regex='league').columns, axis=1, inplace = True)
train_x.drop(train.filter(regex='coach').columns, axis=1, inplace = True)
#
# Test set
# test_id = test['id'].copy()
test_x = test.drop(['home_team_name', 'away_team_name'], axis=1)#, inplace=True) # is_cup EXCLUDED
# Exclude all date, league, coach columns
test_x.drop(test.filter(regex='date').columns, axis=1, inplace = True)
test_x.drop(test.filter(regex='league').columns, axis=1, inplace = True)
test_x.drop(test.filter(regex='coach').columns, axis=1, inplace = True)


# In[ ]:


# Target, train and test shape
print(f"Target: {train_y.shape} \n Train shape: {train_x.shape} \n Test: {test_x.shape}")
print(f"Column names: {list(train_x.columns)}")


# In[ ]:


train_x.head()


# In[ ]:


# Store feature names
# feature_names = list(train.columns)
# Pivot dataframe to create an input array for the LSTM network
feature_groups = ["home_team_history_is_play_home", "home_team_history_is_cup",
    "home_team_history_goal", "home_team_history_opponent_goal",
    "home_team_history_rating", "home_team_history_opponent_rating",  
    "away_team_history_is_play_home", "away_team_history_is_cup",
    "away_team_history_goal", "away_team_history_opponent_goal",
    "away_team_history_rating", "away_team_history_opponent_rating",  
    "home_team_history_match_DIFF_day", "away_team_history_match_DIFF_days",
    "home_team_history_DIFF_goal","away_team_history_DIFF_goal",
    "home_team_history_ELO_rating","away_team_history_ELO_rating",
    "home_away_team_history_ELO_rating",
    "home_team_history_SAME_coaX", "away_team_history_SAME_coaX",
    "home_team_history_SAME_leaG", "away_team_history_SAME_leaG",
    "home_team_result", "away_team_result",
    "home_winner", "home_loser", "away_winner", "away_loser",
    "feature_A", "feature_B", "feature_C",
    "feature_X", "feature_Y", "feature_Z"]      
# Pivot dimension (id*features) x time_history
train_x_pivot = pd.wide_to_long(train_x, stubnames=feature_groups, 
                i=['id','is_cup'], j='time', sep='_', suffix='\d+')
test_x_pivot = pd.wide_to_long(test_x, stubnames=feature_groups, 
                i=['id','is_cup'], j='time', sep='_', suffix='\d+')
#
print(f"Train pivot shape: {train_x_pivot.shape}")  
print(f"Test pivot shape: {test_x_pivot.shape}") 


# In[ ]:


# create columns based on index
train_x_pivot = train_x_pivot.reset_index()
test_x_pivot = test_x_pivot.reset_index()
# Deal with the is_cup feature
# There are NA in 'is_cup'
train_x_pivot=train_x_pivot.fillna({'is_cup':False})
train_x_pivot['is_cup'] = pd.get_dummies(train_x_pivot['is_cup'], drop_first=True)
#
test_x_pivot=test_x_pivot.fillna({'is_cup':False})
test_x_pivot['is_cup']= pd.get_dummies(test_x_pivot['is_cup'], drop_first=True)


# In[ ]:


train_x_pivot.head(20)


# In[ ]:


test_x_pivot.head(20)


# In[ ]:


# Feature engineering again!!! Group by id, stats over time
def add_features_II(df):
    # goals
    # df['home_team_history_DIFF_goal_csum'] = df.groupby('id')['home_team_history_DIFF_goal'].cumsum()
    # df['away_team_history_DIFF_goal_csum'] = df.groupby('id')['away_team_history_DIFF_goal'].cumsum()
    # df['home_team_hist_goal_csum'] = df.groupby('id')['home_team_history_goal'].cumsum()
    # df['home_team_hist_opp_goal_csum'] = df.groupby('id')['home_team_history_opponent_goal'].cumsum()
    # df['away_team_hist_goal_csum'] = df.groupby('id')['away_team_history_goal'].cumsum()
    # df['away_team_hist_opp_goal_csum'] = df.groupby('id')['away_team_history_opponent_goal'].cumsum()
    # rating
    # df['home_team_hist_rat_mean'] = df.groupby('id')['home_team_history_rating'].transform('mean')
    # df['away_team_hist_rat_mean'] = df.groupby('id')['away_team_history_rating'].transform('mean')
    # df['away_team_hist_rat_mean'] = df.groupby('id')['away_team_history_rating'].mean()
    # df['home_away_rat_elo'] = 1/(1+10**((df['away_team_hist_rat_mean']-df['home_team_hist_rat_mean'])/10))
    # Result (%)
    # df['home_team_result_mean'] = df.groupby('id')['home_team_result'].transform('mean')
    # df['away_team_result_mean'] = df.groupby('id')['away_team_result'].transform('mean')
    # df['home_team_result_perc'] = df.groupby('id')['home_team_result'].cumsum()
    # df['away_team_result_perc'] = df.groupby('id')['away_team_result'].cumsum()
    # df['home_team_result_perc'] = df['home_team_result_perc']/(df['time']*2)
    # df['away_team_result_perc'] = df['away_team_result_perc']/(df['time']*2)
    # Lags result if INV = True
    df['home_away_team_history_ELO_rating_lag1'] = df.groupby('id')['home_away_team_history_ELO_rating'].shift(-1)
    df['home_away_team_history_ELO_rating_lag2'] = df.groupby('id')['home_away_team_history_ELO_rating'].shift(-2)
    df['home_away_team_history_ELO_rating_lag3'] = df.groupby('id')['home_away_team_history_ELO_rating'].shift(-3)
    # df['away_team_result_lag2'] = df.groupby('id')['away_team_result'].shift(-2)
    # dummies results
    # df = pd.get_dummies(df['home_team_result'])
    # df = pd.get_dummies(df['away_team_result'])
    # df['one'] = 1
    # df['count'] = df.groupby('id').isna()
    
    return df
# INCREASE the LOSS -  Score: 0.99559
# train_x_pivot = add_features_II(train_x_pivot)
# test_x_pivot = add_features_II(test_x_pivot)


# In[ ]:


# train_x_pivot.head(20)
# test_x_pivot.head(20)


# In[ ]:


# Changing the sequence of time from 1...10 to 10...1 improve the model?
# bidirectional LSTM is used

INV = True
if INV:
    # Trying to keep the same id order
    train_x_pivot.sort_values(by=['time'], inplace = True, ascending=False)
    # Merge and drop columns
    train_x_pivot = pd.merge(train_x['id'], train_x_pivot, on="id").drop(['id', 'time'], axis = 1)
    # Test
    test_x_pivot.sort_values(by=['time'], inplace = True, ascending=False)
    test_x_pivot = pd.merge(test_x['id'], test_x_pivot, on="id").drop(['id', 'time'], axis = 1)
    
    # x_test_pivot = x_test_pivot.reset_index()
    # x_test_pivot['time'] = (T_HIST + 1) - x_test_pivot['time']
    # x_test_pivot.sort_values(by=['time'], inplace = True)
    # x_test = pd.merge(test_id, x_test_pivot, on="id").drop(['id', 'time'], axis = 1)
    # x_test = x_test.to_numpy().reshape(-1, T_HIST, x_test.shape[-1])


# In[ ]:


x_train = train_x_pivot.copy() #drop(['id', 'time'], axis=1)
x_test = test_x_pivot.copy() #drop(['id', 'time'], axis=1)
# Fill NA with median ( I tried mean as well, no improvement)
fill_median = True
if fill_median:
    x_train = np.where(np.isnan(x_train), np.nanmedian(x_train, axis=0), x_train)
    x_test = np.where(np.isnan(x_test), np.nanmedian(x_test, axis=0), x_test)

# Scale features using statistics that are robust to outliers
RS = RobustScaler()
x_train = RS.fit_transform(x_train)
x_test = RS.transform(x_test)
# Reshape 
x_train = x_train.reshape(-1, T_HIST, x_train.shape[-1])
x_test = x_test.reshape(-1, T_HIST, x_test.shape[-1])

if False:
    # Fill NA with MASK
    x_train = np.nan_to_num(x_train, nan=MASK)
    x_test = np.nan_to_num(x_test, nan=MASK)

# Back to pandas.dataframe
# x_train = pd.DataFrame(train, columns=feature_names)
# x_train = pd.concat([train_id, x_train], axis = 1)
#
# x_test = pd.DataFrame(test, columns=feature_names)
# x_test = pd.concat([test_id, x_test], axis = 1)


# In[ ]:


print(f"Train array shape: {x_train.shape} \nTest array shape: {x_test.shape}")


# In[ ]:


# Deal with targets
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(train_y)
encoded_y = encoder.transform(train_y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_y)
# 
print(encoded_y.shape)
print(dummy_y.shape)


# In[ ]:


# encoding away: 0 draw: 1 home: 2 
print(encoded_y[:10,])
# Order: away, draw, home
print(dummy_y[:10,])


# ## Setup GPU
# Below declare whether to use 1 GPU or multiple GPU. (Change CUDA_VISIBLE_DEVICES to use more GPUs). I am not aware of the impact on speed.

# In[ ]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# USE MULTIPLE GPUS
if os.environ["CUDA_VISIBLE_DEVICES"].count(',') == 0:
    gpu_strategy = tf.distribute.get_strategy()
    print('single strategy')
else:
    gpu_strategy = tf.distribute.MirroredStrategy()
    print('multiple strategy')


# ## Make a model
# The model is a very simple LSTM. Why LSTM? Because we have the time history of the football team. I'm not sure whether it has any influence on the result at least for my football team, losing and winning at random. 
# I would try to make a XGBOOST model with aggregated time features and compare the results.

# In[ ]:


# Huge LSTM model -> No good (loss ~ 1.02)
# RNN : A recurrent model can learn to use a long history of inputs, if it's relevant to the predictions the model is making. 
# Here the model will accumulate internal state for 10 matches, before making a single prediction for the next match.
def model_1():
    x_input = layers.Input(shape=(x_train.shape[-2:]))
    # x1 = layers.Masking(mask_value=MASK)(x_input)
    #
    # x1 = layers.Bidirectional(layers.LSTM(units=512, return_sequences=True))(x_input)
    # x2 = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(x1)
    # x3 = layers.Bidirectional(layers.LSTM(units=128, return_sequences=True))(x2)
    #
    # x4 = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(x1)
    # x5 = layers.Bidirectional(layers.LSTM(units=32, return_sequences=True))(x4)
    #
    # z2 = layers.Bidirectional(layers.GRU(units=256, return_sequences=True))(x2)
    # z3 = layers.Bidirectional(layers.GRU(units=128, return_sequences=True))(Add()([x3, z2]))
    # x = layers.Concatenate(axis=2)([x3, z2, z3])
    # x = layers.Bidirectional(layers.LSTM(units=192, return_sequences=True))(x)
    # Optiver
    x = layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(128, return_sequences=True))(x_input)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    conc = layers.concatenate([avg_pool, max_pool])
    out = layers.Flatten()(conc)
    x = layers.Dense(units=16, activation='selu')(out)
    # Output layer must create 3 output values, one for each class.
    # Activation function is softmax for multi-class classification.
    x_output = layers.Dense(units=CLASS, activation='softmax')(x)
    model = Model(inputs=[x_input], outputs=[x_output])
    return model


# In[ ]:


# This is similar to the model of "igorkf" (Good, loss ~ 0.999)
def model_2():
    x_input = layers.Input(shape=x_train.shape[1:])
    # x = layers.Masking(mask_value=MASK, input_shape=(x_train.shape[1:]))(x_input)
    x = layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(16, return_sequences=True))(x_input) #(x)
    x = layers.Dropout(0.5)(x)  
    x = layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(8, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    # output
    output = layers.Dense(CLASS, activation='softmax')(x)
    model = Model(inputs=[x_input],outputs=[output])

    return model


# In[ ]:


def model_2a():
    x_input = layers.Input(shape=x_train.shape[1:])
    # x = layers.Masking(mask_value=MASK, input_shape=(x_train.shape[1:]))(x_input)
    x = layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(8, return_sequences=True))(x_input) #(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(16, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(8, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    # output
    output = layers.Dense(CLASS, activation='softmax')(x)
    model = Model(inputs=[x_input],outputs=[output])
    
    return model 


# In[ ]:


def model_2b():
    x_input = layers.Input(shape=x_train.shape[1:])
    # x = layers.Masking(mask_value=MASK, input_shape=(x_train.shape[1:]))(x_input)
    x = layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(8, return_sequences=True))(x_input) #(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(32, return_sequences=True))(x)
    x = layers.Dropout(0.75)(x)
    x = layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(8, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    # output
    output = layers.Dense(CLASS, activation='softmax')(x)
    model = Model(inputs=[x_input],outputs=[output])
    
    return model 


# In[ ]:


# standard LSTM model -> Not bad
def model_3():
    x_input = layers.Input(shape=x_train.shape[1:])
    x = layers.Masking(mask_value=MASK, input_shape=(x_train.shape[1:]))(x_input)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x) 
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(16))(x)   #ATTENTION: return sequences False, no Flatten layer
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(8, activation='relu')(x)
    # Output layer must create 3 output values, one for each class.
    # Activation function is softmax for multi-class classification.
    output = layers.Dense(CLASS, activation='softmax')(x)
    model = Model(inputs=[x_input],outputs=[output])

    return model


# In[ ]:



# CNN A convolutional model makes predictions based on a fixed-width history, 
# which may lead to better performance than the dense model since it can see how things are changing over time:
CONV_WIDTH = 3

def model_4():
    x_input = layers.Input(shape=x_train.shape[1:])
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    # tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    x = layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH))(x_input)
    x = layers.Dense(16, activation = 'relu')(x)
    # Output layer must create 3 output values, one for each class.
    # Activation function is softmax for multi-class classification.
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(CLASS, activation='softmax')(x)
    model = Model(inputs=[x_input],outputs=[output])

    return model
#


# In[ ]:


'''#Simple convnet
MAX_POOL = 3
CONV_WIDTH = 3
def model_5():
    x_input = layers.Input(shape=x_train.shape[1:])
    x = layers.Conv1D(32, kernel_size = CONV_WIDTH, padding='same', activation="relu")(x_input)
    # x = layers.MaxPooling1D(pool_size = MAX_POOL, padding='same')(x)
    x = layers.Conv1D(16, kernel_size = CONV_WIDTH, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size = MAX_POOL, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(CLASS, activation='softmax')(x)
    model = Model(inputs=[x_input],outputs=[output])

    return model
'''


# In[ ]:


# Convolution LSTM1D (no good loss ~ 1.01)
CONV_WIDTH = 3
# first add an axis to your data
# x_train = np.expand_dims(x_train, 2)   

def model_6():
    x_input = layers.Input(shape= x_train.shape[1:]) # 4D tensor with shape: (samples, time, channels, rows) # (109779, 10, 1, 30)
    x = layers.ConvLSTM1D(32,kernel_size=CONV_WIDTH,padding="same",return_sequences=True,activation="relu")(x_input)
    x = layers.BatchNormalization()(x)
    x = layers.ConvLSTM1D(16,kernel_size=CONV_WIDTH,padding="same",return_sequences=True,activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(CLASS, activation='softmax')(x)
    model = Model(inputs=[x_input],outputs=[output])

    return model


# In[ ]:


# x_train.shape


# In[ ]:


# Choose your model
model = model_2()
model.summary()


# In[ ]:


plot_model(
    model, 
    to_file='Football_Prob_Model.png', 
    show_shapes=True,
    show_layer_names=True
)


# ## Fit the model and a make submission
# Once the model is fitted we make the **probabilities prediction**. Then we make the submission dataframe with **4 columns**. The columns home, away and draw contain probability while the column id contains the match id.

# In[ ]:


# Parameters
EPOCH = 200
BATCH_SIZE = 512
N_SPLITS = 5 # N_SPLITS of the traning set for validation using KFold
SEED = 123
VERBOSE = 0
PATIENCE = EPOCH // 10

test_preds = []

with gpu_strategy.scope():
    kf = KFold(n_splits=N_SPLITS, random_state=SEED) # shuffle=True
    # Model 1 #
    for fold, (train_idx, test_idx) in enumerate(kf.split(x_train, dummy_y)):
        print('-'*15, '>', f'Fold {fold+1}/{N_SPLITS}', '<', '-'*15)
        X_train, X_valid = x_train[train_idx], x_train[test_idx]
        Y_train, Y_valid = dummy_y[train_idx], dummy_y[test_idx]
        ######### Model: CHANGE HERE TOO ################
        model = model_2()
        # It is a multi-class classification problem, categorical_crossentropy is used as the loss function.
        model.compile(optimizer="adam", loss="categorical_crossentropy",
                     metrics=["accuracy"])
        #
        es = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0, mode='min',
                           restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=0)
        #
        model.fit(X_train, Y_train, 
                  validation_data=(X_valid, Y_valid), 
                  epochs=EPOCH,
                  verbose=VERBOSE,
                  batch_size=BATCH_SIZE,  
                  callbacks=[lr, es])
        # Model validation    
        y_true = Y_valid.squeeze()
        y_pred = model.predict(X_valid, batch_size=BATCH_SIZE).squeeze()
        score1 = log_loss(y_true, y_pred)
        print(f"Fold-{fold+1} | OOF LogLoss Score: {score1}")
        # Predictions
        test_preds.append(model.predict(x_test).squeeze())
        # test_preds.append(model.predict(x_test, batch_size=BATCH_SIZE).squeeze().reshape(-1, 1).squeeze())
    ####
    kf = KFold(n_splits=(N_SPLITS - 1), random_state=(SEED*2), shuffle=True)
    # Model 2 # 
    for fold, (train_idx, test_idx) in enumerate(kf.split(x_train, dummy_y)):
        print('-'*15, '>', f'Fold {fold+1}/{N_SPLITS}', '<', '-'*15)
        X_train, X_valid = x_train[train_idx], x_train[test_idx]
        Y_train, Y_valid = dummy_y[train_idx], dummy_y[test_idx]
        ######### Model: CHANGE HERE TOO ################
        model = model_2()
        # It is a multi-class classification problem, categorical_crossentropy is used as the loss function.
        model.compile(optimizer="adam", loss="categorical_crossentropy",
                     metrics=["accuracy"])
        #
        es = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0, mode='min',
                           restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=0)
        #
        model.fit(X_train, Y_train, 
                  validation_data=(X_valid, Y_valid), 
                  epochs=EPOCH,
                  verbose=VERBOSE,
                  batch_size=BATCH_SIZE,  
                  callbacks=[lr, es])
        # Model validation    
        y_true = Y_valid.squeeze()
        y_pred = model.predict(X_valid, batch_size=BATCH_SIZE).squeeze()
        score1 = log_loss(y_true, y_pred)
        print(f"Fold-{fold+1} | OOF LogLoss Score: {score1}")
        # Predictions
        test_preds.append(model.predict(x_test).squeeze())
        # test_preds.append(model.predict(x_test, batch_size=BATCH_SIZE).squeeze().reshape(-1, 1).squeeze())
    # Model 3 # 
    kf = KFold(n_splits=N_SPLITS - 1) # , random_state=(SEED*2), shuffle=True)

    for fold, (train_idx, test_idx) in enumerate(kf.split(x_train, dummy_y)):
        print('-'*15, '>', f'Fold {fold+1}/{N_SPLITS}', '<', '-'*15)
        X_train, X_valid = x_train[train_idx], x_train[test_idx]
        Y_train, Y_valid = dummy_y[train_idx], dummy_y[test_idx]
        ######### Model: CHANGE HERE TOO ################
        model = model_2a()
        # It is a multi-class classification problem, categorical_crossentropy is used as the loss function.
        model.compile(optimizer="adam", loss="categorical_crossentropy",
                     metrics=["accuracy"])
        #
        es = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0, mode='min',
                           restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=0)
        #
        model.fit(X_train, Y_train, 
                  validation_data=(X_valid, Y_valid), 
                  epochs=EPOCH,
                  verbose=VERBOSE,
                  batch_size=BATCH_SIZE,  
                  callbacks=[lr, es])
        # Model validation    
        y_true = Y_valid.squeeze()
        y_pred = model.predict(X_valid, batch_size=BATCH_SIZE).squeeze()
        score1 = log_loss(y_true, y_pred)
        print(f"Fold-{fold+1} | OOF LogLoss Score: {score1}")
        # Predictions
        test_preds.append(model.predict(x_test).squeeze())
        # test_preds.append(model.predict(x_test, batch_size=BATCH_SIZE).squeeze().reshape(-1, 1).squeeze())
    # Model 4 # 
    kf = KFold(n_splits=N_SPLITS * 2) # , random_state=(SEED*2), shuffle=True)

    for fold, (train_idx, test_idx) in enumerate(kf.split(x_train, dummy_y)):
        print('-'*15, '>', f'Fold {fold+1}/{N_SPLITS}', '<', '-'*15)
        X_train, X_valid = x_train[train_idx], x_train[test_idx]
        Y_train, Y_valid = dummy_y[train_idx], dummy_y[test_idx]
        ######### Model: CHANGE HERE TOO ################
        model = model_2a()
        # It is a multi-class classification problem, categorical_crossentropy is used as the loss function.
        model.compile(optimizer="adam", loss="categorical_crossentropy",
                     metrics=["accuracy"])
        #
        es = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0, mode='min',
                           restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=0)
        #
        model.fit(X_train, Y_train, 
                  validation_data=(X_valid, Y_valid), 
                  epochs=EPOCH,
                  verbose=VERBOSE,
                  batch_size=BATCH_SIZE,  
                  callbacks=[lr, es])
        # Model validation    
        y_true = Y_valid.squeeze()
        y_pred = model.predict(X_valid, batch_size=BATCH_SIZE).squeeze()
        score1 = log_loss(y_true, y_pred)
        print(f"Fold-{fold+1} | OOF LogLoss Score: {score1}")
        # Predictions
        test_preds.append(model.predict(x_test).squeeze())
        # test_preds.append(model.predict(x_test, batch_size=BATCH_SIZE).squeeze().reshape(-1, 1).squeeze())


# In[ ]:


# Mean is better than median for predictions.
predictions = np.mean(test_preds, axis = 0) # sum(test_preds)/N_SPLITS 

# away, draw, home
submission = pd.DataFrame(predictions,columns=['away', 'draw', 'home'])

# Round
round_num = False
if round_num:
    submission = submission.round(2)
    submission['draw'] = 1 - (submission['home'] + submission['away'])  
    
#do not forget the id column
submission['id'] = test[['id']]

#submit!
submission[['id', 'home', 'away', 'draw']].to_csv('submission.csv', index=False)


# In[ ]:


submission[['id', 'home', 'away', 'draw']].head()


# In[ ]:


predictions = np.median(test_preds, axis = 0) # sum(test_preds)/N_SPLITS 

# away, draw, home
submission = pd.DataFrame(predictions,columns=['away', 'draw', 'home'])

# Round
round_num = False
if round_num:
    submission = submission.round(2)
    submission['draw'] = 1 - (submission['home'] + submission['away'])  
    
#do not forget the id column
submission['id'] = test[['id']]

#submit!
submission[['id', 'home', 'away', 'draw']].to_csv('submission_median.csv', index=False)


# ## Conclusion
# **Good luck!**
# 
# The best accuracy of the validation set was 0.5015 so far, it is pretty low. You won't make money using this model.
# 
# A huge LSTM makes worse predictions than a small LSTM. Need to work on this!
# 
# Report any error that you will probably find.
