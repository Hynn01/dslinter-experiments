#!/usr/bin/env python
# coding: utf-8

# # > **Competition Description**
# > Football has been at the heart of data science for more than a decade. If today's algorithms focus on event detection, player style, or team analysis, predicting the results of a match stays an open challenge.
# > 
# > Predicting the outcomes of a match between two teams depends mostly (but not only) on their current form. The form of a team can be viewed as their recent sequence of results versus the other teams. So match probabilities between two teams can be different given their calendar.
# > 
# > This competition is about predicting the probabilities of more than 150000 match outcomes using the recent sequence of 10 matches of the teams.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Import Dataset

# In[ ]:


import numpy as np
import pandas as pd
import datetime as dt
#
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
#
import tensorflow as tf
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import layers

import lightgbm as lgb


# # Read Dataset

# In[ ]:


train = pd.read_csv('/kaggle/input/football-match-probability-prediction/train.csv')
test = pd.read_csv('/kaggle/input/football-match-probability-prediction/test.csv')
submission = pd.read_csv('/kaggle/input/football-match-probability-prediction/sample_submission.csv')


# In[ ]:


## Note :: Feature Engg and some code snippets in this notebook has been inpired by https://www.kaggle.com/code/seraquevence/ensemble-football-prob-prediction-lstm-lgbm-v01


# In[ ]:


# for cols "date", change to datatime 
for col in train.filter(regex='date', axis=1).columns:
    train[col] = pd.to_datetime(train[col])
    test[col] = pd.to_datetime(test[col])


# In[ ]:


train.head()


# In[ ]:


train.league_name.nunique()


# # Feature Engineering

# * Calculate historical match day difference
# * Calculate historical match goal difference
# * Check Winner
# * Check Coach
# * Check League
# * Calculate ELO rating

# In[ ]:


def add_features(df):
    for i in range(1, 11): # range from 1 to 10
        # Feat. difference of days
        df[f'home_team_history_match_DIFF_days_{i}'] = (df['match_date'] - df[f'home_team_history_match_date_{i}']).dt.days
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
        df[f'home_team_history_ELO_rating_{i}'] = 1/(1+10**((df[f'home_team_history_opponent_rating_{i}']-df[f'home_team_history_rating_{i}'])/400))
        df[f'away_team_history_ELO_rating_{i}'] = 1/(1+10**((df[f'away_team_history_opponent_rating_{i}']-df[f'away_team_history_rating_{i}'])/400))
        df[f'home_away_team_history_ELO_rating_{i}'] = 1/(1+10**((df[f'away_team_history_rating_{i}']-df[f'home_team_history_rating_{i}'])/400))
    # Feat. same coach id
        df[f'home_team_history_SAME_coaX_{i}'] = np.where(df['home_team_coach_id']==df[f'home_team_history_coach_{i}'],1,0)
        df[f'away_team_history_SAME_coaX_{i}'] = np.where(df['away_team_coach_id']==df[f'away_team_history_coach_{i}'],1,0) 
    # Feat. same league id
        df[f'home_team_history_SAME_leaG_{i}'] = np.where(df['league_id']==df[f'home_team_history_league_id_{i}'],1,0)
        df[f'away_team_history_SAME_leaG_{i}'] = np.where(df['league_id']==df[f'away_team_history_league_id_{i}'],1,0) 
    return df
train = add_features(train)
test = add_features(test)


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


#train_x.shape


# In[ ]:


#train_x.head()


# In[ ]:


feature_groups = ["home_team_history_is_play_home", "home_team_history_is_cup",
    "home_team_history_goal", "home_team_history_opponent_goal",
    "home_team_history_rating", "home_team_history_opponent_rating",  
    "away_team_history_is_play_home", "away_team_history_is_cup",
    "away_team_history_goal", "away_team_history_opponent_goal",
    "away_team_history_rating", "away_team_history_opponent_rating",  
    "home_team_history_match_DIFF_days", "away_team_history_match_DIFF_days",
    "home_team_history_DIFF_goal","away_team_history_DIFF_goal",
    "home_team_history_ELO_rating","away_team_history_ELO_rating",
    "home_away_team_history_ELO_rating",
    "home_team_history_SAME_coaX", "away_team_history_SAME_coaX",
    "home_team_history_SAME_leaG", "away_team_history_SAME_leaG",
    "home_team_result", "away_team_result",
    "home_winner", "home_loser", "away_winner", "away_loser"]      

train_x_pivot = pd.wide_to_long(train_x, stubnames=feature_groups, 
                i=['id','is_cup'], j='time', sep='_', suffix='\d+')
test_x_pivot = pd.wide_to_long(test_x, stubnames=feature_groups, 
                i=['id','is_cup'], j='time', sep='_', suffix='\d+')


# In[ ]:



train_x_pivot = train_x_pivot.reset_index()
test_x_pivot = test_x_pivot.reset_index()


# Encode Is_cup column
train_x_pivot=train_x_pivot.fillna({'is_cup':False})
train_x_pivot['is_cup'] = pd.get_dummies(train_x_pivot['is_cup'], drop_first=True)

test_x_pivot=test_x_pivot.fillna({'is_cup':False})
test_x_pivot['is_cup']= pd.get_dummies(test_x_pivot['is_cup'], drop_first=True)


# In[ ]:


get_ipython().system('pip install miceforest')


# # Data Imputation and Encoding

# In[ ]:


x_train = train_x_pivot.drop(['id', 'time'], axis=1).copy()
x_test = test_x_pivot.drop(['id', 'time'], axis=1).copy()
from sklearn.preprocessing import RobustScaler, LabelEncoder
import miceforest as mf
X_train = mf.ampute_data(x_train)
X_test = mf.ampute_data(x_test)
RS = RobustScaler()
X_train = RS.fit_transform(X_train)
X_test = RS.transform(X_test)
# Reshape 
X_train = X_train.reshape(-1, 10, X_train.shape[-1])
X_test = X_test.reshape(-1, 10, X_test.shape[-1])


# In[ ]:


X_train = np.where(np.isnan(X_train), np.nanmedian(X_train, axis=0), X_train)
X_test = np.where(np.isnan(X_test), np.nanmedian(X_test, axis=0), X_test)


# In[ ]:


#X_train


# In[ ]:


encoder = LabelEncoder()
encoder.fit(train_y)
encoded_y = encoder.transform(train_y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_y)
# 
print(encoded_y.shape)
print(dummy_y.shape)


# In[ ]:


# Using GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# USE MULTIPLE GPUS
if os.environ["CUDA_VISIBLE_DEVICES"].count(',') == 0:
    gpu_strategy = tf.distribute.get_strategy()
    print('single strategy')
else:
    gpu_strategy = tf.distribute.MirroredStrategy()
    print('multiple strategy')


# # Transformer Model
# > https://keras.io/examples/timeseries/timeseries_transformer_classification/

# In[ ]:


from tensorflow import keras
from tensorflow.python.keras import backend as K

# adjust values to your needs
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} )
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res
def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    return keras.Model(inputs, outputs)
input_shape = X_train.shape[1:]

modelt = build_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25
)

modelt.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
modelt.summary()


# In[ ]:


callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

modelt.fit(
    X_train,
    encoded_y,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=callbacks
)


# In[ ]:


test_preds = []
test_preds.append(modelt.predict(X_test).squeeze())


# In[ ]:


predictions = sum(test_preds)
sub = pd.DataFrame(predictions,columns=['away', 'draw', 'home'])
#do not forget the id column
sub['id'] = test[['id']]


# In[ ]:


# modelt.predict(X_train).squeeze()
# print(accuracy_score(encoded_y,[np.argmax(x) for x in modelt.predict(X_train).squeeze()]))
# confusion_matrix(encoded_y,[np.argmax(x) for x in modelt.predict(X_train).squeeze()])


# # LGBM Model
# > https://www.analyticsvidhya.com/blog/2021/08/complete-guide-on-how-to-use-lightgbm-in-python/#:~:text=The%20main%20features%20of%20the,Parallel%20Learning%20support.

# In[ ]:


# LGB MODEL

# Feature engineering again!!! Group by id, stats over time
def agg_features(df):
    # vol_cols = ['log_return1_realized_volatility', 'log_return2_realized_volatility']
    # Group by match id
    df_agg = df.groupby(['id'])[feature_groups].agg(['mean', 'median', 'std', 'max', 'min', 'sum',]).reset_index()
    # Rename columns joining suffix
    df_agg.columns = ['_'.join(col) for col in df_agg.columns]
    return df_agg
    
train_x_agg = agg_features(train_x_pivot)
test_x_agg = agg_features(test_x_pivot)

# Taking Latest 3 matches data to predict outcome

train_x_last = train_x_pivot.loc[train_x_pivot['time']==10]
test_x_last = test_x_pivot.loc[test_x_pivot['time']==10]
train_x1_last = train_x_pivot.loc[train_x_pivot['time']==9]
test_x1_last = test_x_pivot.loc[test_x_pivot['time']==9]
train_x2_last = train_x_pivot.loc[train_x_pivot['time']==8]
test_x2_last = test_x_pivot.loc[test_x_pivot['time']==8]


train_x_last=(train_x_last.merge(train_x1_last,on='id',suffixes=('_10','_9'))).merge(train_x2_last,left_on='id',right_on='id',suffixes=('','_8'))
test_x_last=(test_x_last.merge(test_x1_last,on='id',suffixes=('_10','_9'))).merge(test_x2_last,left_on='id',right_on='id',suffixes=('','_8'))

x_train = pd.merge(train_x_last, train_x_agg, left_on="id", right_on="id_").drop(['time_10','time_9','id','time'], axis = 1).copy()
x_test = pd.merge(test_x_last, test_x_agg, left_on="id", right_on="id_").drop(['time_10','time_9','id','time'], axis = 1).copy()


# In[ ]:


target2int = {'away': 0, 'draw': 1, 'home': 2}
# encode target
dummy_y = train_y.map(target2int)


# In[ ]:


from sklearn.metrics import accuracy_score
num_boost_round = 1000
SEED = 123
seed = SEED
N_SPLITS = 3
FIRST_OOF_ONLY= False
lgbm_params = {
    "objective":"multiclass" #"binary"
    , "boosting_type":"gbdt"
    , 'num_class':3
    , 'metric': "multi_logloss" #''binary_logloss
    ,        'learning_rate': 0.05,        
            'lambda_l1': 2,
            'lambda_l2': 7,
            'num_leaves': 400,
            'min_sum_hessian_in_leaf': 20,
            'feature_fraction': 0.5,
            'feature_fraction_bynode': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 42,
            'min_data_in_leaf': 700,
            'max_depth': 5,
            'seed': seed,
            'feature_fraction_seed': seed,
            'bagging_seed': seed,
            'drop_seed': seed,
            'data_random_seed': seed,
            'verbosity': -1
        }

test_preds = []

kf = kf = StratifiedKFold(n_splits=N_SPLITS, random_state=SEED, shuffle=True) 

for fold, (train_idx, test_idx) in enumerate(kf.split(x_train, dummy_y)):
    print('-'*15, '>', f'Fold {fold+1}/{N_SPLITS}', '<', '-'*15)
    X_train, X_valid = x_train.iloc[train_idx], x_train.iloc[test_idx]
    Y_train, Y_valid = dummy_y.iloc[train_idx], dummy_y.iloc[test_idx]
    # 
    train_dataset = lgb.Dataset(X_train, Y_train) 
    val_dataset = lgb.Dataset(X_valid, Y_valid)
    model = lgb.train(params = lgbm_params, 
                        train_set = train_dataset, 
                        valid_sets = [train_dataset, val_dataset], 
                        num_boost_round = num_boost_round, 
                        callbacks=[lgb.early_stopping(stopping_rounds=50)],
                        verbose_eval = 50)
    # Model validation    
    y_true = Y_valid.squeeze()
    y_pred = model.predict(X_valid).squeeze()
    score1 = log_loss(y_true, y_pred)
    print(f"Fold-{fold+1} | OOF LogLoss Score: {score1}")
    
    # Predictions
    test_preds.append(model.predict(x_test).squeeze())
    #lgb.plot_importance(model,max_num_features=20)
    if FIRST_OOF_ONLY: break


# In[ ]:


from sklearn.metrics import confusion_matrix
#validation_accuracy_score
print(accuracy_score(y_true,[np.argmax(x) for x in model.predict(X_valid).squeeze()]))
confusion_matrix(y_true,[np.argmax(x) for x in model.predict(X_valid).squeeze()])


# In[ ]:


## Draw matches are not being predicted correctly. This maybe due to class imbalance !


# In[ ]:


X_train = np.where(np.isnan(x_train), np.nanmedian(x_train, axis=0), x_train)
X_test = np.where(np.isnan(x_test), np.nanmedian(x_test, axis=0), x_test)
X_train=pd.DataFrame(X_train,columns=x_train.columns)
X_test=pd.DataFrame(X_test,columns=x_test.columns)
X_train=X_train.drop((X_train).columns[pd.DataFrame(X_train).isna().sum().values>0],axis=1)
X_test=X_test.drop((X_test).columns[pd.DataFrame(X_test).isna().sum().values>0],axis=1)


# # SMOTE Upsampling

# In[ ]:


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_train, Y_train = oversample.fit_resample(X_train, dummy_y)
Y_train.value_counts()


# In[ ]:


X_train_master=X_train.copy()
Y_train_master=Y_train.copy()
from sklearn.metrics import accuracy_score
num_boost_round = 1000
SEED = 123
seed = SEED
N_SPLITS = 3 # 5
FIRST_OOF_ONLY= False
lgbm_params = {
    "objective":"multiclass" #"binary"
    , "boosting_type":"gbdt"
    , 'num_class':3
    , 'metric': "multi_logloss" #''binary_logloss
    ,        'learning_rate': 0.05,        
            'lambda_l1': 2,
            'lambda_l2': 7,
            'num_leaves': 400,
            'min_sum_hessian_in_leaf': 20,
            'feature_fraction': 0.5,
            'feature_fraction_bynode': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 42,
            'min_data_in_leaf': 700,
            'max_depth': 5,
            'seed': seed,
            'feature_fraction_seed': seed,
            'bagging_seed': seed,
            'drop_seed': seed,
            'data_random_seed': seed,
            'verbosity': -1
        }

test_preds = []

kf = kf = StratifiedKFold(n_splits=N_SPLITS, random_state=SEED, shuffle=True) # KFold(n_splits=N_SPLITS, random_state=SEED, shuffle=True)

for fold, (train_idx, test_idx) in enumerate(kf.split(X_train_master, Y_train_master)):
    print('-'*15, '>', f'Fold {fold+1}/{N_SPLITS}', '<', '-'*15)
    X_train, X_valid = X_train_master.iloc[train_idx], X_train_master.iloc[test_idx]
    Y_train, Y_valid = Y_train_master.iloc[train_idx], Y_train_master.iloc[test_idx]
    # 
    train_dataset = lgb.Dataset(X_train, Y_train) #, categorical_feature = ['is_cup'])
    val_dataset = lgb.Dataset(X_valid, Y_valid)# , categorical_feature = ['is_cup'])
    model = lgb.train(params = lgbm_params, 
                        train_set = train_dataset, 
                        valid_sets = [train_dataset, val_dataset], 
                        num_boost_round = num_boost_round, 
                        callbacks=[lgb.early_stopping(stopping_rounds=50)],
                        verbose_eval = 50)
    # Model validation    
    y_true = Y_valid.squeeze()
    y_pred = model.predict(X_valid).squeeze()
    score1 = log_loss(y_true, y_pred)
    print(f"Fold-{fold+1} | OOF LogLoss Score: {score1}")
    # print(roc_auc_score(y_true, y_pred))
    # Predictions
    test_preds.append(model.predict(X_test).squeeze())
    #lgb.plot_importance(model,max_num_features=20)
    if FIRST_OOF_ONLY: break


# In[ ]:


print(accuracy_score(y_true,[np.argmax(x) for x in model.predict(X_valid).squeeze()]))
confusion_matrix(y_true,[np.argmax(x) for x in model.predict(X_valid).squeeze()])


# In[ ]:


predictions = sum(test_preds)/N_SPLITS 

# away, draw, home
sub_2 = pd.DataFrame(predictions,columns=['away', 'draw', 'home'])

#do not forget the id column
sub_2['id'] = test[['id']]


# # Combining Predictions of both models

# In[ ]:


def draw(x):
    if x[1]>x[0]:
        if x[1]>x[2]:
            return 1
    return -1
    
sub_2['draw_check']=sub_2.apply(draw,axis=1)
            


# In[ ]:


final=sub.merge(sub_2,on='id',suffixes=('_1','_2'))


# In[ ]:


# For draw predictions by LGBM, assigning its more weightage to LGBM.
# For Away and Home, taking mean.
def combine_models_draw(x):
    if x[-1]==1:
        return  [(x[0]*0.3+x[4]*0.7),(x[1]*0.3+x[5]*0.7),(x[2]*0.3+x[6]*0.7)]
    else:
        return [(x[0]*0.5+x[4]*0.5),(x[1]*0.5+x[5]*0.5),(x[2]*0.5+x[6]*0.5)]
    


# In[ ]:


final['result']=final.apply(combine_models_draw,axis=1)


# In[ ]:


final=final[['id','result']]
#away-draw-home


# # Submission

# In[ ]:


submission=pd.concat([final[['id']],pd.DataFrame(final['result'].to_list(), columns = ['away', 'draw', 'home'])],axis=1)


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




