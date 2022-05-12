#!/usr/bin/env python
# coding: utf-8

# # Predicting States of Manufacturing Control Data 🏭

# **Objective:** Build a powerfull GBDT Model that can provide a good estimation.
# 
# **Strategy:** I think I will follow this strategy:
# 
# **Level 1 Getting Started**
# 
# * Quick EDA to identify potential opportunities.
# * Simple pre-processing step to encode categorical features.
# * A basic CV strategy using 90% for TRaining and 10% for Testing.
# * Looking at the feature importances.
# * Creating a submission file.
# * Submit the file to Kaggle.
# 
# **Level 2 Feature Engineering**
# * Feature engineering using text information. (Massive boost in the score)
# * Cross validation loop (**Work in Progress...**)
# 
# ---

# **Data Description**
# 
# For this challenge, you are given (simulated) manufacturing control data and are tasked to predict whether the machine is in state 0 or state 1. 
# The data has various feature interactions that may be important in determining the machine state.
# 
# Good luck!
# 
# **Files**
# * train.csv - the training data, which includes normalized continuous data and categorical data
# * test.csv - the test set; your task is to predict binary target variable which represents the state of a manufacturing process
# * sample_submission.csv - a sample submission file in the correct format

# ---

# **Notebooks Ideas and Credits**
# 
# I took ideas or inspiration from the following notebooks, if you enjoy my work, please take a look to the notebooks that inspire my work.
# 
# **TPSMAY22 Gradient-Boosting Quickstart:** https://www.kaggle.com/code/ambrosm/tpsmay22-gradient-boosting-quickstart/notebook
# 
# 
# 

# ---

# # 1. Loading the Requiered Libraries

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


# ---
# 

# # 2. Setting the Notebook

# In[ ]:


get_ipython().run_cell_magic('time', '', "# I like to disable my Notebook Warnings.\nimport warnings\nwarnings.filterwarnings('ignore')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Notebook Configuration...\n\n# Amount of data we want to load into the Model...\nDATA_ROWS = None\n# Dataframe, the amount of rows and cols to visualize...\nNROWS = 50\nNCOLS = 15\n# Main data location path...\nBASE_PATH = '...'")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Configure notebook display settings to only use 2 decimal places, tables look nicer.\npd.options.display.float_format = '{:,.2f}'.format\npd.set_option('display.max_columns', NCOLS) \npd.set_option('display.max_rows', NROWS)")


# ---

# # 3. Loading the Information (CSV) Into A Dataframe

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Load the CSV information into a Pandas DataFrame...\ntrn_data = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')\ntst_data = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')\n\nsub = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv')")


# ---

# # 4. Exploring the Information Available

# ## 4.1. Analysing the Trian Dataset

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Explore the shape of the DataFrame...\ntrn_data.shape')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Display simple information of the variables in the dataset...\ntrn_data.info()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Display the first few rows of the DataFrame...\ntrn_data.head()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Generate a simple statistical summary of the DataFrame, Only Numerical...\ntrn_data.describe()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Calculates the total number of missing values...\ntrn_data.isnull().sum().sum()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Display the number of missing values by variable...\ntrn_data.isnull().sum()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Display the number of unique values for each variable...\ntrn_data.nunique()')


# In[ ]:


# Display the number of unique values for each variable, sorted by quantity...
trn_data.nunique().sort_values(ascending = True)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Check some of the categorical variables\ncateg_cols = ['f_29','f_30','f_13', 'f_18','f_17','f_14','f_11','f_10','f_09','f_15','f_07','f_12','f_16','f_08','f_27']\ntrn_data[categ_cols].sample(5)")


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Generate a quick correlation matrix to understand the dataset better\ncorrelation = trn_data.corr()')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Diplay the correlation matrix\ncorrelation')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Check the most correlated variables to the target\ncorrelation['target'].sort_values(ascending = False)[:5]")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Check the least correlated variables to the target\ncorrelation['target'].sort_values(ascending = True)[:5]")


# ---

# ## 4.2. Analysing the Trian Labels Dataset

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Check how well balanced is the dataset\ntrn_data['target'].value_counts()")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Check some statistics on the target variable\ntrn_data['target'].describe()")


# ---

# # 5. Feature Engineering

# ## 5.1 Text Base Features

# In[ ]:


get_ipython().run_cell_magic('time', '', '# The idea is to create a simple funtion to count the amount of letters on feature 27.\n# feature 27 seems quite important \n\ndef count_sequence(df, field):\n    \'\'\'\n    For each letter of the provided suquence it return new feature with the number of occurences.\n    \'\'\'\n    alphabet = [\'A\',\'B\',\'C\',\'D\',\'E\',\'F\',\'G\',\'H\',\'I\',\'J\',\'K\',\'L\',\'M\',\'N\',\'O\',\'P\',\'Q\',\'R\',\'S\',\'T\',\'U\',\'V\',\'W\',\'X\',\'Y\',\'Z\']    \n    \n    for letter in alphabet:\n        df[letter + \'_count\'] = df[field].str.count(letter)\n    \n    df["unique_characters"] = df[\'f_27\'].apply(lambda s: len(set(s)))\n    return df')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Utilizes the new created funtions to generate more features.\n# trn_data = count_sequence(trn_data, 'f_27')\n# tst_data = count_sequence(tst_data, 'f_27')")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'def count_chars(df, field):\n    \'\'\'\n    Describes something...\n    \'\'\'\n    \n    for i in range(10):\n        df[f\'ch_{i}\'] = df[field].str.get(i).apply(ord) - ord(\'A\')\n        \n    df["unique_characters"] = df[field].apply(lambda s: len(set(s)))\n    return df')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Utilizes the new created funtions to generate more features.\ntrn_data = count_chars(trn_data, 'f_27')\ntst_data = count_chars(tst_data, 'f_27')")


# ---

# # 7. Pre-Processing Labels

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Define a label encoding function\nfrom sklearn.preprocessing import LabelEncoder\nencoder = LabelEncoder()\ndef encode_features(df, cols = ['f_27']):\n    for col in cols:\n        df[col + '_enc'] = encoder.fit_transform(df[col])\n    return df\n\ntrn_data = encode_features(trn_data)\ntst_data = encode_features(tst_data)")


# In[ ]:


# Check the results of the transformation
trn_data.head()


# ---

# # 8. Feature Selection for Baseline Model

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Define what will be used in the training stage\nignore = ['id', 'target', 'f_27',  'f_27_enc'] # f_27 has been label encoded...\n\nfeatures = [feat for feat in trn_data.columns if feat not in ignore]\ntarget_feature = 'target'")


# ---

# # 9. Creating a Simple Train / Test Split Strategy

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Creates a simple train split breakdown for baseline model\nfrom sklearn.model_selection import train_test_split\ntest_size_pct = 0.20\nX_train, X_valid, y_train, y_valid = train_test_split(trn_data[features], trn_data[target_feature], test_size = test_size_pct, random_state = 42)')


# ---

# # 10. Building a Baseline GBT Model, Simple Split

# ## 10.1 XGBoost Model

# In[ ]:


get_ipython().run_cell_magic('time', '', '%%script false --no-raise-error\n# Import the model libraries\nfrom xgboost  import XGBClassifier')


# In[ ]:


get_ipython().run_cell_magic('time', '', "%%script false --no-raise-error\n# Define the model parameters to get started we use default values to a certain degree\nxgb_params = {'n_estimators'     : 8192,\n              'min_child_weight' : 96,\n              #'max_depth'        : 6,\n              #'learning_rate'    : 0.15,\n              #'subsample'        : 0.95,\n              #'colsample_bytree' : 0.95,\n              #'reg_lambda'       : 1.50,\n              #'reg_alpha'        : 1.50,\n              #'gamma'            : 1.50,\n              'max_bin'          : 512,\n              'random_state'     : 46,\n              'objective'        : 'binary:logistic',\n              'tree_method'      : 'gpu_hist',\n             }")


# In[ ]:


get_ipython().run_cell_magic('time', '', "%%script false --no-raise-error\n# Instanciate the XGBoost model using the previous parameters\nxgb = XGBClassifier(**xgb_params)\nxgb.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], eval_metric = ['auc'], early_stopping_rounds = 256, verbose = 250)")


# In[ ]:


get_ipython().run_cell_magic('time', '', '%%script false --no-raise-error\n# Check the model performance in the validation dataset\nfrom sklearn.metrics import roc_auc_score\nval_preds = xgb.predict_proba(X_valid[features])[:, 1]\nroc_auc_score(y_valid, val_preds)')


# In[ ]:


# Record some of the model results for future improvement
# Local Score = 0.9454953628406088 First Model Run >>> LB Score = 0.93147
# Local Score = 0.9448767329168479 First Model Run >>> LB Score = 0.93205

# 0.9816418086418166


# ---

# ## 10.2 LGMB Model

# In[ ]:


get_ipython().run_cell_magic('time', '', '%%script false --no-raise-error\n# Import the model libraries\nfrom lightgbm import LGBMClassifier')


# In[ ]:


get_ipython().run_cell_magic('time', '', "%%script false --no-raise-error\n# Define the model parameters to get started we use default values to a certain degree\nlgb_params = {'n_estimators'      : 8192,\n              'min_child_samples' : 96,\n              'max_bins'          : 512,\n              'random_state'      : 46,\n             }")


# In[ ]:


get_ipython().run_cell_magic('time', '', "%%script false --no-raise-error\n# Instanciate the XGBoost model using the previous parameters\nlgb = LGBMClassifier(**lgb_params)\nlgb.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], eval_metric = ['auc'], early_stopping_rounds = 256, verbose = 250)")


# In[ ]:


get_ipython().run_cell_magic('time', '', '%%script false --no-raise-error\n# Check the model performance in the validation dataset\nfrom sklearn.metrics import roc_auc_score\nval_preds = lgb.predict_proba(X_valid[features])[:, 1]\nroc_auc_score(y_valid, val_preds)')


# ---

# # 11. Building a Baseline GBT Model, Kfold Loop

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from lightgbm import LGBMClassifier\nfrom xgboost  import XGBClassifier\nfrom sklearn.model_selection import KFold\nfrom sklearn.metrics import roc_auc_score, roc_curve\nimport math')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Define the model parameters to get started we use default values to a certain degree\nlgb_params = {'n_estimators'      : 8192, # Was 8192...\n              'min_child_samples' : 96,\n              'max_bins'          : 512,\n              'random_state'      : 46,\n             }\n\nxgb_params = {'n_estimators'     : 8192,\n              'min_child_weight' : 96,\n              'max_depth'        : 6,\n              'learning_rate'    : 0.15,\n              'subsample'        : 0.95,\n              'colsample_bytree' : 0.95,\n              'reg_lambda'       : 1.50,\n              'reg_alpha'        : 1.50,\n              'gamma'            : 1.50,\n              'max_bin'          : 512,\n              'random_state'     : 46,\n              'objective'        : 'binary:logistic',\n              'tree_method'      : 'gpu_hist',\n             }")


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Create empty lists to store NN information...\n\nscore_list   = []\npredictions  = [] \n# Define kfolds for training purposes...\nkf = KFold(n_splits = 5)\n\nfor fold, (trn_idx, val_idx) in enumerate(kf.split(trn_data)):\n    print(f\'Training Fold {fold} ...\')\n    X_train, X_valid = trn_data.iloc[trn_idx][features], trn_data.iloc[val_idx][features]\n    y_train, y_valid = trn_data.iloc[trn_idx][target_feature], trn_data.iloc[val_idx][target_feature]\n    \n    # LGBM (Uncomment to use, and Comment the XGBoost Part... LGBM Takes forever)\n    # model = LGBMClassifier(**lgb_params)\n    # model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], eval_metric = [\'auc\'], early_stopping_rounds = 256, verbose = 0)\n    \n    # XGBoost\n    model = XGBClassifier(**xgb_params)\n    model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], eval_metric = [\'auc\'], early_stopping_rounds = 256, verbose = 0)\n    \n    y_valid_pred = model.predict_proba(X_valid.values)[:,1]\n    score = roc_auc_score(y_valid, y_valid_pred)\n\n    score_list.append(score)\n    print(f"Fold {fold}, AUC = {score:.3f}")\n    print((\'\'))\n    \n    tst_pred = model.predict_proba(tst_data[features].values)[:,1]\n    predictions.append(tst_pred)\n\nprint(f\'OOF AUC: {np.mean(score_list):.3f}\')\nprint(\'.........\')')


# # 11. Undertanding Model Behavior, Feature Importance

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Define a funtion to plot the feature importance properly\ndef plot_feature_importance(importance, names, model_type, max_features = 10):\n    #Create arrays from feature importance and feature names\n    feature_importance = np.array(importance)\n    feature_names = np.array(names)\n\n    #Create a DataFrame using a Dictionary\n    data={'feature_names':feature_names,'feature_importance':feature_importance}\n    fi_df = pd.DataFrame(data)\n\n    #Sort the DataFrame in order decreasing feature importance\n    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)\n    fi_df = fi_df.head(max_features)\n\n    #Define size of bar plot\n    plt.figure(figsize=(8,6))\n    \n    #Plot Searborn bar chart\n    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])\n    #Add chart labels\n    plt.title(model_type + 'FEATURE IMPORTANCE')\n    plt.xlabel('FEATURE IMPORTANCE')\n    plt.ylabel('FEATURE NAMES')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Utilize the feature importance function to visualize the most valueable features\nimport seaborn as sns\nimport matplotlib.pyplot as plt\nplot_feature_importance(model.feature_importances_,X_train.columns,'LGBM ', max_features = 25)")


# ---

# # 12. Baseline Model Submission File Generation

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Review the format of the submission file\nsub.head()')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Populated the prediction on the submission dataset and creates an output file\nsub['target'] = np.array(predictions).mean(axis=0)\nsub.to_csv('my_submission_043022.csv', index = False)")


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Review the submission file as a final step to upload to Kaggle.\nsub.head()')


# ---

# In[ ]:




