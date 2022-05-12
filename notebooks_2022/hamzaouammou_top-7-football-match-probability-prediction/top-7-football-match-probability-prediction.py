#!/usr/bin/env python
# coding: utf-8

# ## Football Match Probability Prediction - Run on GPU
# 
# **Competition Description**
# 
# Football has been at the heart of data science for more than a decade. If today's algorithms focus on event detection, player style, or team analysis, predicting the results of a match stays an open challenge.
# 
# Predicting the outcomes of a match between two teams depends mostly (but not only) on their current form. The form of a team can be viewed as their recent sequence of results versus the other teams. So match probabilities between two teams can be different given their calendar.
# 
# This competition is about predicting the probabilities of more than 150000 match outcomes using the recent sequence of 10 matches of the teams.
# 
# > **Upvote Please**

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


# In[ ]:


#loading the data
train = pd.read_csv('/kaggle/input/football-match-probability-prediction/train.csv')
test = pd.read_csv('/kaggle/input/football-match-probability-prediction/test.csv')
submission = pd.read_csv('/kaggle/input/football-match-probability-prediction/sample_submission.csv')


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


# exclude matches with no history at date 1 - full of NA (1159 rows excluded)
#train.dropna(subset=['home_team_history_match_date_1'], inplace = True)
# exclude leagues with less than 15 matches: This change has no effect on the final log loss
if False:
    train['league_id_count'] = train.groupby('league_id')['id'].transform('count')
    train = train.loc[train['league_id_count'] > 15]


# In[ ]:


print(f"Train: {train.shape} \n Submission: {submission.shape}")
train


# In[ ]:


train.home_team_history_match_date_2.isnull().value_counts()


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
        
    # Feat dummy winner x equality x loser 
        df[f'home_winner_{i}'] = np.where(df[f'home_team_history_DIFF_goal_{i}'] > 0, 1., 0.) 
        df[f'home_equality_{i}'] = np.where(df[f'home_team_history_DIFF_goal_{i}'] == 0, 1., 0.) 
        df[f'home_loser_{i}'] = np.where(df[f'home_team_history_DIFF_goal_{i}'] < 0, 1., 0.)
        df[f'away_winner_{i}'] = np.where(df[f'away_team_history_DIFF_goal_{i}'] > 0, 1., 0.)
        df[f'away_equality_{i}'] = np.where(df[f'away_team_history_DIFF_goal_{i}'] == 0, 1., 0.) 
        df[f'away_loser_{i}'] = np.where(df[f'away_team_history_DIFF_goal_{i}'] < 0, 1., 0.)
     
    # Scores Classification 
        df[f'home_team_history_DIFF_goal_classification_{i}'] = pd.cut((df[f'home_team_history_goal_{i}'] - df[f'home_team_history_opponent_goal_{i}']).fillna(0), [-100, 0, 3, 100], labels = False)
        df[f'away_team_history_DIFF_goal_classification_{i}'] = pd.cut((df[f'away_team_history_goal_{i}'] - df[f'away_team_history_opponent_goal_{i}']).fillna(0), [-100, 0, 3, 100], labels = False)
        
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
        
    # Feat. same coach id
        df[f'home_team_history_SAME_coaX_{i}'] = np.where(df['home_team_coach_id']==df[f'home_team_history_coach_{i}'],1,0)
        df[f'away_team_history_SAME_coaX_{i}'] = np.where(df['away_team_coach_id']==df[f'away_team_history_coach_{i}'],1,0) 
        
    # Feat. same league id
        df[f'home_team_history_SAME_leaG_{i}'] = np.where(df['league_id']==df[f'home_team_history_league_id_{i}'],1,0)
        df[f'away_team_history_SAME_leaG_{i}'] = np.where(df['league_id']==df[f'away_team_history_league_id_{i}'],1,0) 

    return df

train = add_features(train)
test = add_features(test)


# ## Scaling and Reshape
# The input/output of the lstm is very trick. It is expected an array of dimensions (Matches/Batch, Time history, Features). I hope I did it right.

# In[ ]:


# save targets
train_y = train['target'].copy()

#keep only some features
train_x = train.drop(['target', 'home_team_name', 'away_team_name'], axis=1) # is_cup EXCLUDED

# Exclude all date, league, coach columns
train_x.drop(train.filter(regex='date').columns, axis=1, inplace = True)
train_x.drop(train.filter(regex='league').columns, axis=1, inplace = True)
train_x.drop(train.filter(regex='coach').columns, axis=1, inplace = True)

# Test set
# test_id = test['id'].copy()
test_x = test.drop(['home_team_name', 'away_team_name'], axis=1) # is_cup EXCLUDED

# Exclude all date, league, coach columns
test_x.drop(test.filter(regex='date').columns, axis=1, inplace = True)
test_x.drop(test.filter(regex='league').columns, axis=1, inplace = True)
test_x.drop(test.filter(regex='coach').columns, axis=1, inplace = True)


# In[ ]:


# Target, train and test shape
print(f"Target: {train_y.shape} \n Train shape: {train_x.shape} \n Test: {test_x.shape}")


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
    "home_away_team_history_ELO_rating", "home_team_history_DIFF_goal_classification",
    "away_team_history_DIFF_goal_classification",
    "home_team_history_SAME_coaX", "away_team_history_SAME_coaX",
    "home_team_history_SAME_leaG", "away_team_history_SAME_leaG",
    "home_team_result", "away_team_result",
    "home_winner","home_equality", "home_loser", "away_winner","away_equality", "away_loser"         
                 ]    
# Pivot dimension (id*features) x time_history
train_x_pivot = pd.wide_to_long(train_x, stubnames=feature_groups, 
                i=['id','is_cup'], j='time', sep='_', suffix='\d+')
test_x_pivot = pd.wide_to_long(test_x, stubnames=feature_groups, 
                i=['id','is_cup'], j='time', sep='_', suffix='\d+')
#
print(f"Train pivot shape: {train_x_pivot.shape}")  
print(f"Test pivot shape: {test_x_pivot.shape}") 


# In[ ]:


train_x_pivot


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


# ### Add more important features

# In[ ]:


# Feature engineering again!!! Group by id, stats over time
def add_features_II(df):
    
    # goals
    df['home_team_history_DIFF_goal_csum'] = df.groupby('id')['home_team_history_DIFF_goal'].cumsum()
    df['away_team_history_DIFF_goal_csum'] = df.groupby('id')['away_team_history_DIFF_goal'].cumsum()

    # rating
    df['home_team_hist_rat_mean'] = df.groupby('id')['home_team_history_rating'].transform('mean')
    df['home_team_hist_rat_min'] = df.groupby('id')['home_team_history_rating'].transform('min')
    df['home_team_hist_rat_max'] = df.groupby('id')['home_team_history_rating'].transform('max')
    df['home_team_hist_rat_sum'] = df.groupby('id')['home_team_history_rating'].transform('sum')
    df['home_team_hist_rat_std'] = df.groupby('id')['home_team_history_rating'].transform('std')
    df['away_team_hist_rat_mean'] = df.groupby('id')['away_team_history_rating'].transform('mean')
    df['away_team_hist_rat_min'] = df.groupby('id')['away_team_history_rating'].transform('min')
    df['away_team_hist_rat_max'] = df.groupby('id')['away_team_history_rating'].transform('max')
    df['home_team_hist_rat_sum'] = df.groupby('id')['home_team_history_rating'].transform('sum')
    df['home_team_hist_rat_std'] = df.groupby('id')['home_team_history_rating'].transform('std')

    df['home_away_rat_elo'] = 1/(1+10**((df['away_team_hist_rat_mean']-df['home_team_hist_rat_mean'])/10))
    
    # Result (%)
    df['home_team_result_mean'] = df.groupby('id')['home_team_result'].transform('mean')
    df['home_team_result_min'] = df.groupby('id')['home_team_result'].transform('min')
    df['home_team_result_max'] = df.groupby('id')['home_team_result'].transform('max')
    df['home_team_result_sum'] = df.groupby('id')['home_team_result'].transform('sum')
    df['home_team_result_std'] = df.groupby('id')['home_team_result'].transform('std')
    df['away_team_result_mean'] = df.groupby('id')['away_team_result'].transform('mean')
    df['away_team_result_min'] = df.groupby('id')['away_team_result'].transform('min')
    df['away_team_result_max'] = df.groupby('id')['away_team_result'].transform('max')
    df['home_team_result_sum'] = df.groupby('id')['home_team_result'].transform('sum')
    df['home_team_result_std'] = df.groupby('id')['home_team_result'].transform('std')

    df['home_team_result_lag1'] = df.groupby('id')['home_team_result'].shift(-1)
    df['away_team_result_lag1'] = df.groupby('id')['away_team_result'].shift(-1)
    df['home_team_result_lag2'] = df.groupby('id')['home_team_result'].shift(-2)
    df['away_team_result_lag2'] = df.groupby('id')['away_team_result'].shift(-2)
    
    return df

train_x_pivot = add_features_II(train_x_pivot)
test_x_pivot = add_features_II(test_x_pivot)


# In[ ]:


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


# In[ ]:


x_train = train_x_pivot.copy() 
x_test = test_x_pivot.copy() 

# Fill NA with median
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
# The model is LSTM.

# In[ ]:


def model():
    x_input = layers.Input(shape=x_train.shape[1:])

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


# Choose your model
model = model()
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


# Just one fold, validation split = 20%
if False:
   EPOCH = 200
   BATCH_SIZE = 512 
   # N_SPLITS = 5
   SEED = 123
   VERBOSE = 1
   PATIENCE = EPOCH // 10
   VAL_SPLIT = 0.2
   # Model
   # It is a multi-class classification problem, categorical_crossentropy is used as the loss function.
   model.compile(optimizer="adam", loss="categorical_crossentropy",
                    metrics=["accuracy"])
   #
   es = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0, mode='min',
                          restore_best_weights=True)
   lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=0)
   #
   model.fit(x_train, dummy_y, 
                 # validation_data=(x_train, y_train), 
                 validation_split=VAL_SPLIT,
                 epochs=EPOCH,
                 verbose=VERBOSE,
                 batch_size=BATCH_SIZE,
                 callbacks=[lr, es])


# In[ ]:


# N_SPLITS of the traning set for validation using KFold
# Parameters
EPOCH = 200
BATCH_SIZE = 512
N_SPLITS = 20
SEED = 123
VERBOSE = 1
PATIENCE = EPOCH // 10

test_preds = []

with gpu_strategy.scope():
    kf = KFold(n_splits=N_SPLITS, random_state=SEED) 

    for fold, (train_idx, test_idx) in enumerate(kf.split(x_train, dummy_y)):
        print('-'*15, '>', f'Fold {fold+1}/{N_SPLITS}', '<', '-'*15)
        X_train, X_valid = x_train[train_idx], x_train[test_idx]
        Y_train, Y_valid = dummy_y[train_idx], dummy_y[test_idx]
        
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


# In[ ]:


predictions = sum(test_preds)/N_SPLITS 

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




