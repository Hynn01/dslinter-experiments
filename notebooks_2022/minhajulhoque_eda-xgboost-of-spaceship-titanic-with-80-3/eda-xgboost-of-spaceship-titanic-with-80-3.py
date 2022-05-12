#!/usr/bin/env python
# coding: utf-8

# # Introduction 
# `V1.0.0`
# ### Who am I
# Just a fellow Kaggle learner. I was creating this Notebook as practice and thought it could be useful to some others 
# ### Who is this for
# This Notebook is for people that learn from examples. Forget the boring lectures and follow along for some fun/instructive time :)
# ### What can I learn here
# You learn all the basics needed to create a rudimentary XGBoost model with hyperparameter tuning. I go over a multitude of steps with explanations. Hopefully with these building blocks,you can go ahead and build much more complex models.
# 
# ### Things to remember
# + Please Upvote/Like the Notebook so other people can learn from it
# + Feel free to give any recommendations/changes. 
# + I will be continuously updating the notebook. Look forward to many more upcoming changes in the future.
# 
# ### You can also refer to these notebooks that have helped me as well:
# + https://www.kaggle.com/code/sanjaylalwani/spaceship-titanic-eda-ensemble-with-80-5#Feature-Selection

# # Imports

# In[ ]:


# Python Imports
import os
import numpy as np   # Library for n-dimensional arrays
import pandas as pd  # Library for dataframes (structured data)
from pathlib import Path
from datetime import datetime

# ML imports
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# I like to disable my Notebook Warnings.
import warnings
warnings.filterwarnings('ignore')

# Set seeds to make the experiment more reproducible.
from numpy.random import seed
seed(1)

# Allows us to see more information regarding the DataFrame
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)


# # Importing Data
# 1. Since data is in form of csv file we have to use pandas read_csv to load the data
# 2. After loading it is important to check the complete information of data. It is important to get a general feel of the data that we are going to be using.

# In[ ]:


DATA_DIR = Path("../input/spaceship-titanic")

TRAIN_DATA_PATH = DATA_DIR / "train.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"


# In[ ]:


train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)


# <div class="alert alert-block alert-info">
# <b>Tip:</b> We can use the .head() method to obtain the first 5 rows of the DataFrame.
# </div>

# In[ ]:


train_df.head(5)


# <div class="alert alert-block alert-info">
# <b>Tip:</b> We can use the .sample() method to obtain 5 random rows in the DataFrame.
# </div>

# In[ ]:


test_df.sample(5)


# # EDA/Visualizations
# The goal is to try and gain insights from the data prior to modeling

# ## Explorating the Dataframe
# It is useful to use .info() method to quickly have a glance on the general information about the DataFrame. It displays info such as the type of the columnd and also the # of non-null count. In this case there is 8693 entries. Some columns have less values than 8693 which mean they have some missing values that we will have to take care of.

# In[ ]:


train_df.info()


# The describe() method gives a quick summary of the statistical information of the numerical columns. We get descriptions for the mean, standard deviation and max value for example.

# In[ ]:


train_df.describe()


# In[ ]:


train_df.isna().sum()


# Here we are defining a function that returns are categorical, numerical and feature columns. We will be using it consistenly across the notebook.

# In[ ]:


def get_all_cols(df, target_col, exclude=None):
        
    if exclude is None:
        exclude = []
        
    # Select categorical columns
    object_cols = [cname for cname in df.columns 
                   if df[cname].dtype == "object"]

    # Select numerical columns
    num_cols = [cname for cname in df.columns 
                if df[cname].dtype in ['int64', 'float64', 'uint8']]
    
    all_cols = object_cols + num_cols
    exclude_cols = exclude + [target_col]
    feature_cols = [col for col in all_cols if col not in exclude_cols]
    
    return object_cols, num_cols, feature_cols


# In[ ]:


TARGET = "Transported"
object_cols, num_cols, feature_cols = get_all_cols(train_df, TARGET)


# <div class="alert alert-block alert-warning">  
# <b>Note:</b> We assign a constant variable TARGET that we can refer to throughout the notebook as the target variable. Makes it much easier than always typing "Transported".
# </div>

# We can also explore unique values for our feature columns using the unique() method.

# In[ ]:


for object_col in object_cols:
    train_df_unique_list = train_df[object_col].unique()
    print(f'{object_col}:{train_df_unique_list}\n')


# The value_counts() method allows us to get unique value counts that exist in a specific column. In this case, we will get the unique values and count of the three feature columns.

# In[ ]:


for object_col in object_cols:
    obj_val_counts = train_df[object_col].value_counts()
    print(f'LENGTH: {len(obj_val_counts)}\n',f'{obj_val_counts}\n')


# Analyzing the categorical columns, we notice that some of the colums (True/False) we can transform into a boolean column, some we can one-hot encode (the ones that have less than 7 unique values).

# # Feature Engineering / Prepare the data

# ## Handling Missing Values
# In this section, we will take care of the missing values before starting to train the model. 

# Let's start with the categorical columns. We notice that for HomePlanet, the most common category is Earth and for Destination it is TRAPPIST-1e. We fill the missing values with these common value.

# In[ ]:


train_df['HomePlanet']= train_df['HomePlanet'].fillna('Earth')
test_df['HomePlanet']= test_df['HomePlanet'].fillna('Earth')

train_df['Destination']= train_df['Destination'].fillna('TRAPPIST-1e')
test_df['Destination']= test_df['Destination'].fillna('TRAPPIST-1e')


# We then take care of the boolean columns and transform the True/False into 1/0.

# In[ ]:


train_df['CryoSleep']= train_df['CryoSleep'].fillna(False).astype(int)
test_df['CryoSleep']= test_df['CryoSleep'].fillna(False).astype(int)

train_df['VIP']= train_df['VIP'].fillna(False).astype(int)
test_df['VIP']= test_df['VIP'].fillna(False).astype(int)


# In[ ]:


train_df.sample(5)


# Now we will take care of the numerical columns. We fill the missing values with the mean.

# In[ ]:


train_df["Age"] = train_df["Age"].fillna(train_df["Age"].mean())
train_df["RoomService"] = train_df["RoomService"].fillna(train_df["RoomService"].mean())
train_df["FoodCourt"] = train_df["FoodCourt"].fillna(train_df["FoodCourt"].mean())
train_df["ShoppingMall"] = train_df["ShoppingMall"].fillna(train_df["ShoppingMall"].mean())
train_df["Spa"] = train_df["Spa"].fillna(train_df["Spa"].mean())
train_df["VRDeck"] = train_df["VRDeck"].fillna(train_df["VRDeck"].mean())

test_df["Age"] = test_df["Age"].fillna(test_df["Age"].mean())
test_df["RoomService"] = test_df["RoomService"].fillna(test_df["RoomService"].mean())
test_df["FoodCourt"] = test_df["FoodCourt"].fillna(test_df["FoodCourt"].mean())
test_df["ShoppingMall"] = test_df["ShoppingMall"].fillna(test_df["ShoppingMall"].mean())
test_df["Spa"] = test_df["Spa"].fillna(test_df["Spa"].mean())
test_df["VRDeck"] = test_df["VRDeck"].fillna(test_df["VRDeck"].mean())


# In[ ]:


train_df.isna().sum()


# There seems to only be two columns with missing values. These are two columns that we will transform first before taking care of the missing valuues.

# ## Treating the outliers
# Let's first define a helper function that will allow us to quickly plot box plots for our numerical columns.

# In[ ]:


def box_plots(df):
    plt.figure(figsize=(10,5))
    plt.title("Box Plot")
    sns.boxplot(df)
    plt.show()


# In[ ]:


TARGET = "Transported"
object_cols, num_cols, feature_cols = get_all_cols(train_df, TARGET)

# Remove CryoSleep and VIP column because they are binary columns
num_cols.remove('CryoSleep')
num_cols.remove('VIP')
continous_num_cols = num_cols.copy()
continous_num_cols


# In[ ]:


for col in continous_num_cols:
    box_plots(train_df[col])


# We notice some outliers outside of our whisker range. Let's remove them by calculating our upper_limit and lower_limit. These are calculated using the interquartile range.

# In[ ]:


train_df.loc[train_df['FoodCourt'] > 20000, 'FoodCourt'] = train_df.loc[train_df['FoodCourt'] < 20000, 'FoodCourt'].mean()
train_df.loc[train_df['ShoppingMall'] > 10000, 'ShoppingMall'] = train_df.loc[train_df['ShoppingMall'] < 10000, 'ShoppingMall'].mean()
train_df.loc[train_df['Spa'] > 20000, 'Spa'] = train_df.loc[train_df['Spa'] < 20000, 'Spa'].mean()
train_df.loc[train_df['VRDeck'] > 20000, 'VRDeck'] = train_df.loc[train_df['VRDeck'] < 20000, 'VRDeck'].mean()
train_df.loc[train_df['RoomService'] > 10000, 'RoomService'] = train_df.loc[train_df['RoomService'] < 10000, 'RoomService'].mean()


# In[ ]:


for col in continous_num_cols:
    box_plots(train_df[col])


# In[ ]:


# for col in continous_num_cols:
#     percentile25 = train_df[col].quantile(0.25)
#     percentile75 = train_df[col].quantile(0.75)
#     if (percentile25 != 0) or (percentile75 != 0):
#         iqr = percentile75 - percentile25

#         upper_limit = percentile75 + 1.5 * iqr
#         lower_limit = percentile25 - 1.5 * iqr

#         print("BEFORE", train_df.shape, "\n")
#         if lower_limit != 0:
#             train_df.loc[train_df[col] < lower_limit, col] = train_df.loc[train_df[col] < upper_limit, col].mean()
#         if upper_limit != 0:
#             train_df.loc[train_df[col] > upper_limit, col] = train_df.loc[train_df[col] > lower_limit, col].mean()
            
#         box_plots(train_df[col])
        
#         print("AFTER",train_df.shape, "\n")
        
#     else:
#         print(col, percentile25, percentile75)


# ## Prepare the Data
# In this subsection, we look into preparing the feature columns. That can be done by transforming the type of the column to a proper one, creating datetime features from our date column or even adding more valuable feature column (such as holidays) to our dataframe. This is the first step before going to other feature engineering steps.

# Let's create three new columns derived from the `Cabin` column. Dividing this column in three would potentially allow us to get more information that can help us achieve better accuracy. We can then take care of the missing values of the three new columns.

# In[ ]:


train_df[['Deck', 'Num', 'Side']] = train_df['Cabin'].str.split('/', expand=True)   
test_df[['Deck', 'Num', 'Side']] = test_df['Cabin'].str.split('/', expand=True)   

train_df


# In[ ]:


object_cols, num_cols, feature_cols = get_all_cols(train_df, TARGET, ["Cabin"])

for object_col in object_cols:
    obj_val_counts = train_df[object_col].value_counts()
    print(f'LENGTH: {len(obj_val_counts)}\n',f'{obj_val_counts}\n')


# We now take care of the missing values of the three new columns introduced: `Deck` :Categorical, `Num` :Numerical, `Side` :Categorical

# In[ ]:


train_df['Deck']= train_df['Deck'].fillna('F')
train_df['Num'] = train_df['Num'].astype(float)
train_df['Num']= train_df['Num'].fillna(train_df['Num'].mean())
train_df['Side']= train_df['Side'].fillna('S')

test_df['Deck']= test_df['Deck'].fillna('F')
test_df['Num'] = test_df['Num'].astype(float)
test_df['Num']= test_df['Num'].fillna(train_df['Num'].mean())
test_df['Side']= test_df['Side'].fillna('S')


# In[ ]:


train_df.isna().sum()


# Let's now create a new column `group_id` that will be derived from the `Passenger_ID`. We don't really care about the passenger ID, but we care more about the Group_ID because common group ids can represent a family or a common group of friends. This column has no missing values, therefore no missing value handling is required.

# In[ ]:


def create_group_id(passenger_id):
    splitted_id = passenger_id.split("_")
    group_id = splitted_id[1]
    return group_id


# In[ ]:


train_df["group_id"] = train_df["PassengerId"].apply(create_group_id)
train_df["group_id"] = train_df["group_id"].astype(int)

test_df["group_id"] = test_df["PassengerId"].apply(create_group_id)
test_df["group_id"] = test_df["group_id"].astype(int)


# <div class="alert alert-block alert-danger">  
# Don't forget to do the same for our test data!
# </div>

# Let's now remove the unwanted columns. We are removing PassengerId and Cabin because they have been transformed into other columns. We are removing Name because it contains too many unique values and we deduce it would have not much impact in the prediction. 
# 
# If you have more time, you can keep Name or transform it into another columns and see the real impact it has in the predictions.

# In[ ]:


DROP_COLS = ['PassengerId', 'Name', 'Cabin']
PassengerId = test_df['PassengerId']
train_df.drop(DROP_COLS,axis=1, inplace=True)
test_df.drop(DROP_COLS,axis=1, inplace=True)


# In[ ]:


train_df.info()


# ## Handling Categorical Data
# So that the model can understand categorical data, we must transform them in a numerical form. There is various ways to do that. 

# Some of them categorical data are,
# <div class="alert alert-block alert-info">
# <b>Nominal Data</b> --> data are not in any order --> OneHotEncoder or pandas.get_dummies() is used in this case
# </div>
# <div class="alert alert-block alert-info">
# <b>Ordinal data </b> --> data are in order --> LabelEncoder is used in this case
# </div>

# We can one hot encode Homeplanete, Destination and Side because they have less than 8 unique values. We will label encode deck.

# In[ ]:


object_cols, num_cols, feature_cols = get_all_cols(train_df, TARGET)

for object_col in object_cols:
    obj_val_counts = train_df[object_col].value_counts()
    print(f'LENGTH: {len(obj_val_counts)}\n',f'{obj_val_counts}\n')


# In[ ]:


ONE_HOT_CATEGORICAL = ['HomePlanet', 'Destination', 'Side']
def create_one_hot(df, categ_colums = ONE_HOT_CATEGORICAL):
    """
    Creates one_hot encoded fields for the specified categorical columns...
    Args
        df
        categ_colums
    Returns
        df
    """
    df = pd.get_dummies(df, columns=categ_colums)
    return df

LABEL_CATEGORICAL = ['Deck']
def encode_categ_features(df, categ_colums = LABEL_CATEGORICAL):
    """
    Use the label encoder to encode categorical features...
    Args
        df
        categ_colums
    Returns
        df
    """
    le = LabelEncoder()
    for col in categ_colums:
        df['enc_'+col] = le.fit_transform(df[col])
    df.drop(categ_colums, axis=1, inplace=True)
    return df

train_df = encode_categ_features(train_df)
test_df = encode_categ_features(test_df)

train_df = create_one_hot(train_df)
test_df = create_one_hot(test_df)


# In[ ]:


CATEGORICAL = ONE_HOT_CATEGORICAL + LABEL_CATEGORICAL

object_cols, num_cols, feature_cols = get_all_cols(train_df, TARGET)
object_cols


# In[ ]:


train_df.info(), test_df.info()


# No more object columns... we have done our job :)

# ## Feature Selection
# 
# Finding out the best feature which will contribute and have good relation with target variable.
# Following are some of the feature selection methods,

# 
# <div class="alert alert-block alert-info">
# <b>1. heatmap</b> 
# </div>
# <div class="alert alert-block alert-info">
# <b>2. feature_importance_</b> 
# </div>
# <div class="alert alert-block alert-info">
# <b>3. SelectKBest</b> 
# </div>

# ### Correlation 
# To see the correlation between the various features and also with the target value, we will use a heatmap.

# In[ ]:


plt.figure(figsize = (18,18))
sns.heatmap(train_df.corr(), annot = True, cmap = "RdYlGn")

plt.show()


# **We notice some features are heavily correlated. We will remove two to reduce the dimensionality of our model:**
# 1.  enc_Deck and HomePlanet_Europa are heavily negatively correlated. I decide to remove enc_Deck since it has more unique values than HomePlanet_Europa.

# In[ ]:


train_df.drop(["enc_Deck", "Side_S"], axis=1, inplace=True)
test_df.drop(["enc_Deck", "Side_S"], axis=1, inplace=True)


# In[ ]:


train_df['Under15'] = train_df['Age'].apply(lambda x: 1 if x < 15 else 0)
test_df['Under15'] = test_df['Age'].apply(lambda x: 1 if x < 15 else 0)

train_df = train_df.drop(['Age'], axis=1)
test_df = test_df.drop(['Age'], axis=1)


# ## Splitting the data
# In this section, we will split the data in train and test set. Do not confuse test set with our test data. Test set is just a subsample of train_df.

# In[ ]:


X = train_df.drop(TARGET, axis=1)
y = train_df[TARGET]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)


# # Models
# In this section, we will explore one model:
# 
# 1. XGBClassifier

# ## Training
# We've prepared the food (data), time to... FEED THE MACHINE.

# In[ ]:


xgb_model = XGBClassifier()
model = xgb_model.fit(X_train, y_train, eval_metric='logloss')

print("Performance on train data:", model.score(X_train, y_train))


# ## Predicting & Evaluating
# In this subsection, we evaluate using plots and metrics to see if our predictions are good or not.

# In[ ]:


y_pred_v = model.predict(X_valid)

print("Performance on validation data:", f1_score(y_valid, y_pred_v, average='micro'))

cm = confusion_matrix(y_valid, y_pred_v) 
print ("Confusion Matrix : \n", cm)


# # Hyperparameter Tuning
# 
# 
# * Choose following method for hyperparameter tuning
#     1. **RandomizedSearchCV**: Faster when there are many combinations of hyperparameter
#     2. **GridSearchCV**: Tries all combinations
# * Assign hyperparameters in form of dictionary
# * Fit the model
# * Check best paramters and best score

# ## Search for best hyperparameters
# I have already run the hyperparameter optimization and found an optimized model. The code has been commented out, but you can use it to then find an even more optimal model with different hyperparameters.

# In[ ]:


# # A parameter grid for XGBoost
# params = {
#         'min_child_weight': [1, 5, 10],
#         'gamma': [0.5, 1, 1.5, 2, 5],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0],
#         'max_depth': [3, 4, 5]
#         }

# xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='logloss',
#                     silent=True, nthread=1)

# def timer(start_time=None):
#     if not start_time:
#         start_time = datetime.now()
#         return start_time
#     elif start_time:
#         thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
#         tmin, tsec = divmod(temp_sec, 60)
#         print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
        
# folds = 5
# param_comb = 50

# skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

# random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,y), verbose=2, random_state=42 )

# # Here we go
# start_time = timer(None) # timing starts from this point for "start_time" variable
# random_search.fit(X, y)
# timer(start_time) # timing ends here for "start_time" variable

# print('\n All results:')
# print(random_search.cv_results_)
# print('\n Best estimator:')
# print(random_search.best_estimator_)
# print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
# print(random_search.best_score_ * 2 - 1)
# print('\n Best hyperparameters:')
# print(random_search.best_params_)
# results = pd.DataFrame(random_search.cv_results_)
# results.to_csv('xgb-random-grid-search-results-01.csv', index=False)


# Here is the model with the optimized hyperparameters

# ## Predicting with tuned model
# Let us used our tuned model to predict the Target price and see if it does better than our untuned model.

# In[ ]:


optimized_xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.8,
              enable_categorical=False, gamma=5, gpu_id=-1,
              importance_type=None, interaction_constraints='',
              learning_rate=0.02, max_delta_step=0, max_depth=5,
              min_child_weight=5, monotone_constraints='()',
              n_estimators=600, n_jobs=1, nthread=1, num_parallel_tree=1,
              predictor='auto', random_state=0, reg_alpha=0, reg_lambda=1,
              scale_pos_weight=1, silent=True, subsample=0.8,
              tree_method='exact', validate_parameters=1, verbosity=None)

optimized_model = optimized_xgb.fit(X_train, y_train, eval_metric='logloss')

print("Performance on train data:", optimized_model.score(X_train, y_train))


# ## Predicting & Evaluating tuned model

# In[ ]:


y_pred_v = optimized_model.predict(X_valid)

print("Performance on validation data:", f1_score(y_valid, y_pred_v, average='micro'))

cm = confusion_matrix(y_valid, y_pred_v) 
print ("Confusion Matrix : \n", cm)


# Performance on validation data: 0.7912593444508338
# 
# Confusion Matrix : 
# 
#  [[648 213]
#  
#  [150 728]]

# # Save the model to reuse it again
# There's various ways to save the model. We decided to go forward with pickling. It is very easy and straighforward. 

# In[ ]:


import pickle
# open a file, where you ant to store the data
with open('xgboost_tuned.pkl', 'wb') as file:
    pickle.dump(optimized_model, file)


# In[ ]:


with open('xgboost_tuned.pkl', 'rb') as model:
    xgboost_loaded = pickle.load(model)


# # Submitting
# In this section, we get the predictions for the TEST DATA and save our dataframe into a csv file to then be submitted.

# In[ ]:


y_pred = xgboost_loaded.predict(test_df)


# In[ ]:


#Create a  DataFrame with the passengers ids and our prediction
submission_df = pd.read_csv(SUBMISSION_PATH)
submission_df["Transported"] = y_pred
submission_df.to_csv('submission.csv', index=False)


# # Final Remarks
# Thank you for going through this notebook. Please feel free to show support and comment on the notebooks with advice or improvements. If you found it useful, please let me know as well :)
