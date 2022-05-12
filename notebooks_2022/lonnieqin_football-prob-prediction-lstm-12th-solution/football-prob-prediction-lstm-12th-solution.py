#!/usr/bin/env python
# coding: utf-8

# ## Football Match Probability Prediction
# 
# **Competition Description**
# 
# Football has been at the heart of data science for more than a decade. If today's algorithms focus on event detection, player style, or team analysis, predicting the results of a match stays an open challenge.
# 
# Predicting the outcomes of a match between two teams depends mostly (but not only) on their current form. The form of a team can be viewed as their recent sequence of results versus the other teams. So match probabilities between two teams can be different given their calendar.
# 
# This competition is about predicting the probabilities of more than 150000 match outcomes using the recent sequence of 10 matches of the teams.
# 
# **Introduction**
# 
# The goal of this notebook is to develop a LSTM model.
# It may be improved greatly by more feature engineering and other stuffs.
# 
# > **In case you fork/copy/like this notebook:**
# > **Upvote it s'il te plaît/Per favore/Please**
# 
# This notebook was inspired (not copied) by the work of: 
# https://www.kaggle.com/igorkf/football-match-probability-prediction-lstm-starter

# ## Import modules and loading the data
# First let's import modules, the training and the test set. The test set contains the same columns than the training set without the target.

# In[ ]:


# Import libraries
# import gc
import numpy as np
import pandas as pd
import datetime as dt
#
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
#
import tensorflow as tf
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import layers
# from tensorflow.keras.layers import Bidirectional, LSTM, MaxPooling2D, Conv2D
# from tensorflow.keras.layers import Dense, Dropout, Input, Flatten, Masking
# from tensorflow.keras.layers import Concatenate, Add, GRU
# from tensorflow.keras.callbacks import ModelCheckpoint


# In[ ]:


#loading the data
train = pd.read_csv('/kaggle/input/football-match-probability-prediction/train.csv')
test = pd.read_csv('/kaggle/input/football-match-probability-prediction/test.csv')
submission = pd.read_csv('/kaggle/input/football-match-probability-prediction/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


train[train.columns[0:10]].head()


# In[ ]:


train[train.columns[10:20]].head()


# In[ ]:


train[train.columns[20:30]].head()


# In[ ]:


train[train.columns[30:40]].head()


# In[ ]:


train[train.columns[40:50]].head()


# In[ ]:


len(train.home_team_name.unique())


# In[ ]:


train.home_team_name.value_counts()[0:50].plot(kind="bar")


# In[ ]:


train.away_team_name.value_counts()[0:50].plot(kind="bar")


# In[ ]:


len(train.away_team_name.unique())


# In[ ]:


train.columns


# In[ ]:


test.head()


# In[ ]:


train.shape


# In[ ]:


print(train.columns)


# ## Parameters
# Set some parameters that will be used later.

# In[ ]:


# Set seed
np.random.seed(123)
tf.random.set_seed(123)

#Some parameters
MASK = -666 # fill NA with -666 (the number of the beast)
T_HIST = 10 # time history, last 10 games
CLASS = 3 #number of classes (home, draw, away)

DEBUG = False
# Run on a small sample of the data
if DEBUG:
    train = train[:10000]


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
    # Results: multiple nested where # away:0, draw:1, home:2
        df[f'home_team_result_{i}'] = np.where(df[f'home_team_history_DIFF_goal_{i}'] > 0, 2,
                         (np.where(df[f'home_team_history_DIFF_goal_{i}'] == 0, 1,
                                   np.where(df[f'home_team_history_DIFF_goal_{i}'].isna(), np.nan, 0))))
        df[f'away_team_result_{i}'] = np.where(df[f'away_team_history_DIFF_goal_{i}'] > 0, 2,
                         (np.where(df[f'away_team_history_DIFF_goal_{i}'] == 0, 1,
                                   np.where(df[f'away_team_history_DIFF_goal_{i}'].isna(), np.nan, 0))))
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
    "home_team_result", "away_team_result"]      
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
# There are na in 'is_cup'
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


x_train = train_x_pivot.drop(['id', 'time'], axis=1)
x_test = test_x_pivot.drop(['id', 'time'], axis=1)
# Scale features using statistics that are robust to outliers
RS = RobustScaler()
x_train = RS.fit_transform(x_train)
x_test = RS.transform(x_test)
# Fill NA with MASK
x_train = np.nan_to_num(x_train, nan=MASK)
x_test = np.nan_to_num(x_test, nan=MASK)
# Reshape 
x_train = x_train.reshape(-1, T_HIST, x_train.shape[-1])
x_test = x_test.reshape(-1, T_HIST, x_test.shape[-1])

# Back to pandas.dataframe
# x_train = pd.DataFrame(train, columns=feature_names)
# x_train = pd.concat([train_id, x_train], axis = 1)
#
# x_test = pd.DataFrame(test, columns=feature_names)
# x_test = pd.concat([test_id, x_test], axis = 1)


# In[ ]:


# Input array for the LSTM network
# Dimension (MATCHES, TIME, FEATURES)

# INV = False
'''if False:
    # Trying to keep the same id order
    x_train = pd.merge(train_id, x_train_pivot, on="id")
    x_train = x_train.drop(['id'], axis = 1).to_numpy().reshape(-1, T_HIST, x_train_pivot.shape[-1])
    # Test
    x_test = pd.merge(test_id, x_test_pivot, on="id")
    x_test = x_test.drop(['id'], axis = 1).to_numpy().reshape(-1, T_HIST, x_test_pivot.shape[-1])
'''


# In[ ]:


# Changing the sequence of time from 1...10 to 10...1 improve the model?
# bidirectional LSTM is used
'''
if False:
    # Trying to keep the same id order
    x_train_pivot = x_train_pivot.reset_index()
    x_train_pivot['time'] = (T_HIST + 1) - x_train_pivot['time']
    x_train_pivot.sort_values(by=['time'], inplace = True)
    # Merge and drop columns
    x_train = pd.merge(train_id, x_train_pivot, on="id").drop(['id', 'time'], axis = 1)
    x_train = x_train.to_numpy().reshape(-1, T_HIST, x_train.shape[-1])
    # Test
    x_test_pivot = x_test_pivot.reset_index()
    x_test_pivot['time'] = (T_HIST + 1) - x_test_pivot['time']
    x_test_pivot.sort_values(by=['time'], inplace = True)
    x_test = pd.merge(test_id, x_test_pivot, on="id").drop(['id', 'time'], axis = 1)
    x_test = x_test.to_numpy().reshape(-1, T_HIST, x_test.shape[-1])
'''


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


# ## Make a model
# The model is a very simple LSTM. Why LSTM? Because we have the time history of the football team. I'm not sure whether it has any influence on the result at least for my football team, losing and winning at random. 
# I would try to make a XGBOOST model with aggregated time features and compare the results.

# In[ ]:


# Huge LSTM model -> No good
# RNN : A recurrent model can learn to use a long history of inputs, if it's relevant to the predictions the model is making. 
# Here the model will accumulate internal state for 10 matches, before making a single prediction for the next match.
def model_1():
    x_input = Input(shape=(x_train.shape[-2:]))
    x1 = layers.Masking(mask_value=MASK)(x_input)
    #
    # x1 = Bidirectional(LSTM(units=512, return_sequences=True))(x_input)
    # x2 = Bidirectional(LSTM(units=256, return_sequences=True))(x1)
    # x3 = Bidirectional(LSTM(units=128, return_sequences=True))(x2)
    x4 = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(x1)
    x5 = layers.Bidirectional(layers.LSTM(units=32, return_sequences=True))(x4)
    # z2 = Bidirectional(GRU(units=256, return_sequences=True))(x2)
    # z3 = Bidirectional(GRU(units=128, return_sequences=True))(Add()([x3, z2]))
    # x = Concatenate(axis=2)([x3, z2, z3])
    # x = Bidirectional(LSTM(units=192, return_sequences=True))(x)
    x = layers.Flatten()(x5)
    x = layers.Dense(units=16, activation='selu')(x)
     # Output layer must create 3 output values, one for each class.
    # Activation function is softmax for multi-class classification.
    x_output = layers.Dense(units=CLASS, activation='softmax')(x)
    model = Model(inputs=[x_input], outputs=[x_output])
    return model


# In[ ]:


# This is similar to the model of "igorkf"
def model_2():
    x_input = layers.Input(shape=x_train.shape[1:])
    x = layers.Masking(mask_value=MASK, input_shape=(x_train.shape[1:]))(x_input)
    x = layers.Bidirectional(layers.LSTM(16, return_sequences=True))(x)
    #x = layers.Conv1D(8, activation='relu', kernel_size=(10))(x)
    x = layers.Dropout(0.3)(x)  
    x = layers.Bidirectional(layers.LSTM(8, return_sequences=True))(x)
    #x = layers.Conv1D(8, activation='relu', kernel_size=(10))(x)
    x = layers.Dropout(0.3)(x)
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
    x = layers.Bidirectional(layers.LSTM(16, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(8))(x) #ATTENTION: return sequences False, no Flatten layer
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation='relu')(x)
    # Output layer must create 3 output values, one for each class.
    # Activation function is softmax for multi-class classification.
    output = layers.Dense(CLASS, activation='softmax')(x)
    model = Model(inputs=[x_input],outputs=[output])

    return model


# In[ ]:



# CNN A convolutional model makes predictions based on a fixed-width history, 
# which may lead to better performance than the dense model since it can see how things are changing over time:
CONV_WIDTH = 10

def model_4():
    x_input = layers.Input(shape=x_train.shape[1:])
    x = layers.Masking(mask_value=MASK, input_shape=(x_train.shape[1:]))(x_input)
    x = layers.Conv1D(128, activation='relu', kernel_size=(CONV_WIDTH))(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)  
    x = layers.Bidirectional(layers.LSTM(16, return_sequences=False))(x)
    x = layers.Dense(128, activation = 'swish')(x)
    x = layers.Dropout(0.5)(x) 
    x = layers.Dense(64, activation = 'swish')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(32, activation = 'swish')(x)
    # Output layer must create 3 output values, one for each class.
    # Activation function is softmax for multi-class classification.
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(CLASS, activation='softmax')(x)
    model = Model(inputs=[x_input],outputs=[output])

    return model
#


# In[ ]:


# Choose your model
model = model_4()
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


# N_SPLITS of the traning set for validation using KFold
# Parameters
EPOCH = 200
BATCH_SIZE = 512
N_SPLITS = 15
SEED = 123
VERBOSE = 1
PATIENCE = EPOCH // 10

test_preds = []

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (train_idx, test_idx) in enumerate(kf.split(x_train, dummy_y)):
    print('-'*15, '>', f'Fold {fold+1}/{N_SPLITS}', '<', '-'*15)
    X_train, X_valid = x_train[train_idx], x_train[test_idx]
    Y_train, Y_valid = dummy_y[train_idx], dummy_y[test_idx]
    # Model
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
    model.save_weights(f"model_{fold}.tf") 
    # Model validation    
    y_true = Y_valid.squeeze()
    y_pred = model.predict(X_valid, batch_size=BATCH_SIZE).squeeze()
    score1 = log_loss(y_true, y_pred)
    print(f"Fold-{fold+1} | OOF LogLoss Score: {score1}")
    # Predictions
    test_preds.append(model.predict(x_test).squeeze())
    # test_preds.append(model.predict(x_test, batch_size=BATCH_SIZE).squeeze().reshape(-1, 1).squeeze())


# In[ ]:


predictions = sum(test_preds)/N_SPLITS 

# away, draw, home
submission = pd.DataFrame(predictions,columns=['away', 'draw', 'home'])

#do not forget the id column
submission['id'] = test[['id']]

#submit!
submission[['id', 'home', 'away', 'draw']].to_csv('submission.csv', index=False)


# In[ ]:


submission[['id', 'home', 'away', 'draw']].head()


# ## Conclusion
# **Good luck!**
# 
# The best accuracy of the validation set was 0.5015 so far, it is pretty low. You won't make money using this model.
# 
# A huge LSTM makes worse predictions than a small LSTM. Need to work on this!
# 
# Report any error that you will probably find.
