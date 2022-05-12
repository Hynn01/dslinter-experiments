#!/usr/bin/env python
# coding: utf-8

# ## 2019 Data Science Bowl
# #### Uncover the factors to help measure how young children learn
# [Crislânio Macêdo](https://www.linkedin.com/in/crislanio/) -  January, 02th, 2020
# 
# [Github](https://github.com/crislanio)  ________________[Sapere Aude Tech](https://medium.com/sapere-aude-tech) ________________[AboutMe](https://crislanio.wordpress.com/about)________________[Twitter](https://twitter.com/crs_macedo)________________[Ensina.AI](https://medium.com/ensina-ai/an%C3%A1lise-dos-dados-abertos-do-governo-federal-ba65af8c421c)________________[Quora](https://www.quora.com/profile/Crislanio)________________[Hackerrank](https://www.hackerrank.com/crislanio_ufc)
# 
# 📒 EDA: [📒👦👧 DS Bowl - Start here: A GENTLE Introduction](https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction)
# 
# ----------
# ----------

# ![](http://www.gpb.org/sites/www.gpb.org/files/styles/hero_image/public/blogs/images/2018/08/07/maxresdefault.jpg?itok=gN6ErLyU)
# ## [📒👦👧 DS Bowl - Start here: A GENTLE Introduction](https://www.kaggle.com/caesarlupum/ds-bowl-start-here-a-gentle-introduction)

# # Content

# ### 📒In this kernel I want to show a Exhaustive Analysis of  DS Bowl, with:
# 
# - <a href='#1'>1. Where does the data for the competition come from?</a>
# - <a href='#2'>2. Data Description</a>
# - <a href='#3'>3. Read in Data </a>
# - <a href='#4'>4. Glimpse of Data </a>
# - <a href='#5'>5. Insights </a>
# - <a href='#6'>6. Exploratory Data Analysis </a>
# - <a href='#7'>7. Column Types </a>
#     - <a href='#7-1'>7.1 Number of each type of column - Train </a>
#     - <a href='#7-2'>7.2 Number of unique classes in each object column  - Train </a>
#     - <a href='#7-3'>7.3 Number of each type of column - Train_labels </a>
#     - <a href='#7-4'>7.4 Number of each type of column - Train_labels </a>
#     - <a href='#7-5'>7.5 Number of unique classes in each object column  - Train_labels </a>
#     - <a href='#7-6'>7.6 Number of each type of column - Specs </a>
#     - <a href='#7-7'>7.7 Number of unique classes in each object column  - Specs </a>
# - <a href='#8'>8. Examine Missing Values </a>
# - <a href='#9'>9. Correlations </a>
# - <a href='#10'>10. Ploting </a>
# - <a href='#11'>11. Simple Baseline </a>
# - <a href='#12'>12. Simple LGBM aggregated data with CV </a>
# - <a href='#13'>13. Submission </a>
# - <a href='#14'>14. Evaluation </a>
# - <a href='#15'>15. Links for the Game </a>
# - <a href='#16'>General findings </a>
# 

# ### Illuminate Learning. Ignite Possibilities.
# Uncover new insights in early childhood education and how media can support learning outcomes. Participate in our fifth annual Data Science Bowl, presented by Booz Allen Hamilton and Kaggle.
# 
# PBS KIDS, a trusted name in early childhood education for decades, aims to gain insights into how media can help children learn important skills for success in school and life. In this challenge, you’ll use anonymous gameplay data, including knowledge of videos watched and games played, from the PBS KIDS Measure Up! app, a game-based learning tool developed as a part of the CPB-PBS Ready To Learn Initiative with funding from the U.S. Department of Education. Competitors will be challenged to predict scores on in-game assessments and create an algorithm that will lead to better-designed games and improved learning outcomes. Your solutions will aid in discovering important relationships between engagement with high-quality educational media and learning processes.
# 
# ##### Data Science Bowl is the world’s largest data science competition focused on social good. Each year, this competition gives Kagglers a chance to use their passion to change the world. Over the last four years, more than 50,000+ Kagglers have submitted over 114,000+ submissions, to improve everything from lung cancer and heart disease detection to ocean health.
# 
# > For more information on the Data Science Bowl, please visit [DataScienceBowl.com](DataScienceBowl.com)
# 

# In[ ]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from IPython.display import HTML
HTML('<iframe width="1100" height="619" src="https://www.youtube.com/embed/45Da3eqQKXQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


# ### <a id='1'>Where does the data for the competition come from?</a>
# <a href= '#1'> Top</a>
# 
# The data used in this competition is anonymous, tabular data of interactions with the PBS KIDS Measure Up! app. Select data, such as a user’s in-app assessment score or their path through the game, is collected by the PBS KIDS Measure Up! app, a game-based learning tool.
# 
# PBS KIDS is committed to creating a safe and secure environment that family members of all ages can enjoy. The PBS KIDS Measure Up! app does not collect any personally identifying information, such as name or location. All of the data used in the competition is anonymous. To view the full PBS KIDS privacy policy, please visit: [pbskids.org/privacy](pbskids.org/privacy).
# 
# No one will be able to download the entire data set and the participants do not have access to any personally identifiable information about individual users. The Data Science Bowl and the use of data for this year’s competition has been reviewed to ensure that it meets requirements of applicable child privacy regulations by PRIVO, a leading global industry expert in children’s online privacy.
# 
# ### What is the PBS KIDS Measure Up! app?
# In the PBS KIDS Measure Up! app, children ages 3 to 5 learn early STEM concepts focused on length, width, capacity, and weight while going on an adventure through Treetop City, Magma Peak, and Crystal Caves. Joined by their favorite PBS KIDS characters, children can also collect rewards and unlock digital toys as they play. To learn more about PBS KIDS Measure Up!, please click here.
# 
# PBS KIDS and the PBS KIDS Logo are registered trademarks of PBS. Used with permission. The contents of PBS KIDS Measure Up! were developed under a grant from the Department of Education. However, those contents do not necessarily represent the policy of the Department of Education, and you should not assume endorsement by the Federal Government. The app is funded by a Ready To Learn grant (PR/AWARD No. U295A150003, CFDA No. 84.295A) provided by the Department of Education to the Corporation for Public Broadcasting.
# ![image](http://www.wics.jp/data/editor/1902/thumb-962c8c816b10ed5f6586ece6e7a5f63d_1550554761_2866_600x400.jpg)
# 

# ### <a id='2'>Data Description</a>
# <a href= '#1'> Top</a>
# 
# In this dataset, you are provided with game analytics for the PBS KIDS Measure Up! app. In this app, children navigate a map and complete various levels, which may be activities, video clips, games, or assessments. Each assessment is designed to test a child's comprehension of a certain set of measurement-related skills. There are five assessments: Bird Measurer, Cart Balancer, Cauldron Filler, Chest Sorter, and Mushroom Sorter.
# 
# The intent of the competition is to use the gameplay data to forecast how many attempts a child will take to pass a given assessment (an incorrect answer is counted as an attempt). Each application install is represented by an installation_id. This will typically correspond to one child, but you should expect noise from issues such as shared devices. In the training set, you are provided the full history of gameplay data. In the test set, we have truncated the history after the start event of a single assessment, chosen randomly, for which you must predict the number of attempts. Note that the training set contains many installation_ids which never took assessments, whereas every installation_id in the test set made an attempt on at least one assessment.
# 
# #####  The outcomes in this competition are grouped into 4 groups (labeled accuracy_group in the data):
# 
# - 3: the assessment was solved on the first attempt
# - 2: the assessment was solved on the second attempt
# - 1: the assessment was solved after 3 or more attempts
# - 0: the assessment was never solved
# > The file train_labels.csv has been provided to show how these groups would be computed on the assessments in the training set. Assessment attempts are captured in event_code 4100 for all assessments except for Bird Measurer, which uses event_code 4110. If the attempt was correct, it contains "correct":true.
# 
# > Note that this is a synchronous rerun code competition and the private test set has approximately 8MM rows. You should be mindful of memory in your notebooks to avoid submission errors.
# 
# ### Files
# 
# - train.csv & test.csv
# 
# These are the main data files which contain the gameplay events.
# 
# - event_id - Randomly generated unique identifier for the event type. Maps to event_id column in specs table.
# - game_session - Randomly generated unique identifier grouping events within a single game or video play session.
# - timestamp - Client-generated datetime
# - event_data - Semi-structured JSON formatted string containing the events parameters. Default fields are: event_count, event_code, and - - game_time; otherwise fields are determined by the event type.
# - installation_id - Randomly generated unique identifier grouping game sessions within a single installed application instance.
# - event_count - Incremental counter of events within a game session (offset at 1). Extracted from event_data.
# - event_code - Identifier of the event 'class'. Unique per game, but may be duplicated across games. E.g. event code '2000' always - - - - identifies the 'Start Game' event for all games. Extracted from event_data.
# - game_time - Time in milliseconds since the start of the game session. Extracted from event_data.
# - title - Title of the game or video.
# - type - Media type of the game or video. Possible values are: 'Game', 'Assessment', 'Activity', 'Clip'.
# - world - The section of the application the game or video belongs to. Helpful to identify the educational curriculum goals of the media. Possible values are: 'NONE' (at the app's start screen), TREETOPCITY' (Length/Height), 'MAGMAPEAK' (Capacity/Displacement), 'CRYSTALCAVES' (Weight).
# - specs.csv
# 
# ##### This file gives the specification of the various event types.
# 
# - event_id - Global unique identifier for the event type. Joins to event_id column in events table.
# - info - Description of the event.
# - args - JSON formatted string of event arguments. Each argument contains:
# - name - Argument name.
# - type - Type of the argument (string, int, number, object, array).
# - info - Description of the argument.
# - train_labels.csv
# 
# ###### This file demonstrates how to compute the ground truth for the assessments in the training set.
# 
# - sample_submission.csv
# A sample submission in the correct format.

# 
# <html>
# <body>
# 
# <p><font size="5" color="Purple">If you find this kernel useful or interesting, please don't forget to upvote the kernel =)
# 
# </body>
# </html>
# 
# 

# ### Imports
# 
# > We are using a typical data science stack: `numpy`, `pandas`, `sklearn`, `matplotlib`. 
# 

# In[ ]:


import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
pd.set_option('max_columns', 100)


py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from tqdm import tqdm_notebook

from IPython.display import HTML

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rc('figure', figsize=(15.0, 8.0))


# # <a id='3'>3. Read in Data</a>
# <a href= '#1'> Top</a>
# 
# First, we can list all the available data files. There are a total of 6 files: 1 main file for training (with target) 1 main file for testing (without the target), 1 example submission file, and 4 other files containing additional information about energy types based on historic usage rates and observed weather. . 

# In[ ]:


import os
print(os.listdir("../input/data-science-bowl-2019/"))


# In[ ]:


get_ipython().run_cell_magic('time', '', "root = '../input/data-science-bowl-2019/'\n\n# Only load those columns in order to save space\nkeep_cols = ['event_id', 'game_session', 'installation_id', 'event_count', 'event_code', 'title', 'game_time', 'type', 'world']\ntrain = pd.read_csv(root + 'train.csv',usecols=keep_cols)\ntest = pd.read_csv(root + 'test.csv', usecols=keep_cols)\n\ntrain_labels = pd.read_csv(root + 'train_labels.csv')\nspecs = pd.read_csv(root + 'specs.csv')\nsample_submission = pd.read_csv(root + 'sample_submission.csv')")


# # <a id='4'>4. Glimpse of Data</a>
# <a href= '#1'> Top</a>
# 

# In[ ]:


print('Size of train data', train.shape)
print('Size of train_labels data', train_labels.shape)
print('Size of specs data', specs.shape)
print('Size of test data', test.shape)


# # <a id='5'>5. Insigths</a>
# <a href= '#1'> Top</a>
# 
# 

# **Train**

# In[ ]:


train.head()


# **Train_labels**

# In[ ]:


train_labels.head()


# **Specs**

# In[ ]:


specs.head()


# # <a id='6'>6. Exploratory Data Analysis</a>
# <a href= '#1'> Top</a>
# 
# 
# Exploratory Data Analysis (EDA) is an open-ended process where we calculate statistics and make figures to find trends, patterns, or relationships within the data. 

# > Still in progress

# # <a id='7'>7. Column Types</a>
# <a href= '#1'> Top</a>
# 
# 
# Let's look at the number of columns of each data type. `int64` and `float64` are numeric variables ([which can be either discrete or continuous](https://stats.stackexchange.com/questions/206/what-is-the-difference-between-discrete-data-and-continuous-data)). `object` columns contain strings and are  [categorical features.](http://support.minitab.com/en-us/minitab-express/1/help-and-how-to/modeling-statistics/regression/supporting-topics/basics/what-are-categorical-discrete-and-continuous-variables/) . 

# ### <a id='7-1'>7.1 Number of each type of column - Train</a>
# <a href= '#1'> Top</a>
# 

# train.dtypes

# In[ ]:


train.dtypes.value_counts()


# ### <a id='7-2'>7.2 Number of unique classes in each object column  - Train</a>
# <a href= '#1'> Top</a>
# 

# In[ ]:


train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# ### <a id='7-3'>7.3 Number of each type of column - Train_labels</a>
# <a href= '#1'> Top</a>
# 
# 

# train_labels.dtypes

# In[ ]:


train_labels.dtypes.value_counts()


# ### <a id='7-4'>7.4 Number of unique classes in each object column  - Train_labels</a>
# <a href= '#1'> Top</a>
# 
# 

# In[ ]:


train_labels.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# ### <a id='7-5'>7.5 Number of each type of column - Specs</a>
# <a href= '#1'> Top</a>
# 
# 

# specs.dtypes

# In[ ]:


specs.dtypes.value_counts()


# ### <a id='7-6'>7.6 Number of unique classes in each object column  - Specs</a>
# <a href= '#1'> Top</a>
# 
# 

# In[ ]:


specs.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# # <a id='8'>8. Examine Missing Values</a>
# <a href= '#1'> Top</a>
# 
# 
# 
# Next we can look at the number and percentage of missing values in each column. 

# ### checking missing data for train

# In[ ]:


total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing__train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing__train_data.head(10)


# ### checking missing data for Train_labels

# In[ ]:


total = train_labels.isnull().sum().sort_values(ascending = False)
percent = (train_labels.isnull().sum()/train_labels.isnull().count()*100).sort_values(ascending = False)
missing__train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing__train_data.head(10)


# ### checking missing data for Specs

# In[ ]:


total = specs.isnull().sum().sort_values(ascending = False)
percent = (specs.isnull().sum()/specs.isnull().count()*100).sort_values(ascending = False)
missing__train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing__train_data.head(10)


# # <a id='9'>9. Correlations</a>
# <a href= '#1'> Top</a>
# 
# 
# 
# Now that we have dealt with the categorical variables and the outliers, let's continue with the EDA. One way to try and understand the data is by looking for correlations between the features and the target. We can calculate the Pearson correlation coefficient between every variable and the target using the `.corr` dataframe method.
# 
# The correlation coefficient is not the greatest method to represent "relevance" of a feature, but it does give us an idea of possible relationships within the data. Some [general interpretations of the absolute value of the correlation coefficent](http://www.statstutor.ac.uk/resources/uploaded/pearsons.pdf) are:
# 
# 
# * .00-.19 “very weak”
# *  .20-.39 “weak”
# *  .40-.59 “moderate”
# *  .60-.79 “strong”
# * .80-1.0 “very strong”
# 

# > ### Train Correlation

# In[ ]:


corrs = train.corr()
corrs


# In[ ]:


plt.figure(figsize = (20, 8))

# Heatmap of correlations
sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');


# > ### Train_labels Correlation

# In[ ]:


corrs2 = train_labels.corr()
corrs2


# In[ ]:


plt.figure(figsize = (20, 8))

# Heatmap of correlations
sns.heatmap(corrs2, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');


# # <a id='10'>10. Ploting </a>
# <a href= '#1'> Top</a>
# 

# >  ### accuracy_group

# In[ ]:


plt.figure(figsize=(8, 6))
sns.countplot(x="accuracy_group",data=train_labels, order = train_labels['accuracy_group'].value_counts().index)
plt.title('Accuracy Group Count Column')
plt.tight_layout()
plt.show()


# In[ ]:


train_labels.groupby('accuracy_group')['game_session'].count()     .plot(kind='barh', figsize=(15, 5), title='Target (accuracy group)')
plt.show()


# In[ ]:


train.head()


# **Log(Count) of Observations by installation_id**

# In[ ]:



palete = sns.color_palette(n_colors=10)


# In[ ]:


train.groupby('installation_id')     .count()['event_id']     .apply(np.log1p)     .plot(kind='hist',
          bins=40,
          color=palete[1],
         figsize=(15, 5),
         title='Log(Count) of Observations by installation_id')
plt.show()


# #### Count of Observation by Game/Video title

# In[ ]:


train.groupby('title')['event_id']     .count()     .sort_values()     .plot(kind='barh',
          title='Count of Observation by Game/Video title',
         color=palete[1],
         figsize=(15, 15))
plt.show()


# #### Count by World

# In[ ]:


train.groupby('world')['event_id']     .count()     .sort_values()     .plot(kind='bar',
          figsize=(15, 4),
          title='Count by World',
          color=palete[1])
plt.show()


# # <a id='11'>11. Simple Baseline </a>
# <a href= '#1'> Top</a>
# 
# 

# See this noteboks [simple lgbm](https://www.kaggle.com/xhlulu/ds-bowl-2019-simple-lgbm-using-aggregated-data),[simple lgbm CV](https://kaggle.com/tanreinama/ds-bowl-2019-simple-lgbm-aggregated-data-with-cv
# ).(Simple and Helpful)
# 

# *     group1 and group2 are intermediary "game session" groups,
# *     which are reduced to one record by game session. group1 takes
# *     the max value of game_time (final game time in a session) and 
# *     of event_count (total number of events happened in the session).
# *     group2 takes the total number of event_code of each type

# In[ ]:


def group_and_reduce(df):
    # group1 and group2 are intermediary "game session" groups,
    # which are reduced to one record by game session. group1 takes
    # the max value of game_time (final game time in a session) and 
    # of event_count (total number of events happened in the session).
    # group2 takes the total number of event_code of each type
    group1 = df.drop(columns=['event_id', 'event_code']).groupby(
        ['game_session', 'installation_id', 'title', 'type', 'world']
    ).max().reset_index()

    group2 = pd.get_dummies(
        df[['installation_id', 'event_code']], 
        columns=['event_code']
    ).groupby(['installation_id']).sum()

    # group3, group4 and group5 are grouped by installation_id 
    # and reduced using summation and other summary stats
    group3 = pd.get_dummies(
        group1.drop(columns=['game_session', 'event_count', 'game_time']),
        columns=['title', 'type', 'world']
    ).groupby(['installation_id']).sum()

    group4 = group1[
        ['installation_id', 'event_count', 'game_time']
    ].groupby(
        ['installation_id']
    ).agg([np.sum, np.mean, np.std])

    return group2.join(group3).join(group4)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_small = group_and_reduce(train)\ntest_small = group_and_reduce(test)\n\nprint(train_small.shape)\ntrain_small.head()')


# # <a id='12'>12. Simple LGBM aggregated data with CV </a>
# <a href= '#1'> Top</a>
# 
# 

# [Kfold](https://machinelearningmastery.com/k-fold-cross-validation/)
# 
# Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.
# The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation. When a specific value for k is chosen, it may be used in place of k in the reference to the model, such as k=10 becoming 10-fold cross-validation.
# 
# ![](https://datavedas.com/wp-content/uploads/2018/04/image001-1.jpg)

# In[ ]:


get_ipython().run_cell_magic('time', '', "from sklearn.model_selection import KFold\nsmall_labels = train_labels[['installation_id', 'accuracy_group']].set_index('installation_id')\ntrain_joined = train_small.join(small_labels).dropna()\nkf = KFold(n_splits=5, random_state=2019)\nX = train_joined.drop(columns='accuracy_group').values\ny = train_joined['accuracy_group'].values.astype(np.int32)\ny_pred = np.zeros((len(test_small), 4))\nfor train, test in kf.split(X):\n    x_train, x_val, y_train, y_val = X[train], X[test], y[train], y[test]\n    train_set = lgb.Dataset(x_train, y_train)\n    val_set = lgb.Dataset(x_val, y_val)\n\n    params = {\n        'learning_rate': 0.01,\n        'bagging_fraction': 0.9,\n        'feature_fraction': 0.9,\n        'num_leaves': 50,\n        'lambda_l1': 0.1,\n        'lambda_l2': 1,\n        'metric': 'multiclass',\n        'objective': 'multiclass',\n        'num_classes': 4,\n        'random_state': 2019\n    }\n\n    model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50, valid_sets=[train_set, val_set], verbose_eval=50)\n    y_pred += model.predict(test_small)")


# # <a id='13'>13. Submission </a>
# <a href= '#1'> Top</a>
# 
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "y_pred = y_pred.argmax(axis=1)\ntest_small['accuracy_group'] = y_pred\ntest_small[['accuracy_group']].to_csv('submission.csv')")


# # <a id='14'>14. Evaluation </a>
# <a href= '#1'> Top</a>
# 
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'val_pred = model.predict(x_val).argmax(axis=1)\nprint(classification_report(y_val, val_pred))')


# # <a id='15'>15. Links for the Game </a>
# <a href= '#1'> Top</a>
# 
# 
# 

# **Mushroom Sorter**

# In[ ]:



HTML('<iframe width="1106" height="622" src="https://www.youtube.com/embed/1ejHigxuR2Q" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')


#   >  ⚡ Please, you can use parts of this notebook in your own scripts or kernels, no problem, but please give credit (for example link back to this, see this...)

# # <a id='16'>General findings</a>
# <a href='#1'>Top</a>
# 

# - **QWK Computation**
# 
#     https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-670484
# 
#     https://www.kaggle.com/c/data-science-bowl-2019/discussion/114138#latest-667441
# 
#     https://www.kaggle.com/c/data-science-bowl-2019/discussion/114135#latest-656785
# 
#     https://www.kaggle.com/c/data-science-bowl-2019/discussion/114472#latest-658804
# 
# - **Baseline**
#    
#    https://www.kaggle.com/c/data-science-bowl-2019/discussion/114376#latest-659424
#    
#    https://www.kaggle.com/c/data-science-bowl-2019/discussion/114783#latest-671345
# 

# # <a id='17'>Top Notebooks</a>
# <a href='#1'>Top</a>
# 
# 
# 

# ### XGBoost & Feature Selection DSBowl 🥣 🥣 by [@shahules](https://www.kaggle.com/shahules/xgboost-feature-selection-dsbowl)
# * You will learn
#     - Data Preparation and Evaluation
#     - Using XGBoost with StratifiedKFold
#     - Interpreting a ML model with Confidence using [SHAP](https://shap.readthedocs.io/en/latest/)
# 
# Souce: https://www.kaggle.com/shahules/xgboost-feature-selection-dsbowl 
# ### 📒👦👧 DS Bowl - Start here: A GENTLE Introduction 🥣 🥣 by [@caesarlupum](https://www.kaggle.com/caesarlupum/ds-bowl-start-here-a-gentle-introduction)
# * You will learn
#     - Where does the data for the competition come from?
#     - Data Description
#     - Exploratory Data Analysis
#     - Simple Baseline
#     - Simple LGBM aggregated data with CV
# 
# Souce:  https://www.kaggle.com/caesarlupum/ds-bowl-start-here-a-gentle-introduction
# 
#      
# ### OOP approach to FE and models by [@artgor ](https://www.kaggle.com/artgor/oop-approach-to-fe-and-models)    
# * You will learn
#     - Data Preparation and learn/use a class for generating features
#     - Training xgb / Training catboost
# 
# Source: https://www.kaggle.com/artgor/oop-approach-to-fe-and-models
# 
# ### Data Science Bowl 2019 EDA and Baseline by [@erikbruin ](https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline)    
# * You will learn
#     - Understanding the train data
#     - Understanding the test set
#     - Understanding and visualizing the train labels
#     - Feature engineering
# 
# Source: https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline
# 
# ### A new baseline for DSB 2019 - Catboost model by [@mhviraf ](https://www.kaggle.com/mhviraf/a-new-baseline-for-dsb-2019-catboost-model)    
# * You will learn
#     - Data Preparation 
#     - Baseline model with Catboost 
# 
# Source: https://www.kaggle.com/mhviraf/a-new-baseline-for-dsb-2019-catboost-model
# 
# ### 🚸 2019 Data Science Bowl - An Introduction by [@robikscube ](https://www.kaggle.com/robikscube/2019-data-science-bowl-an-introduction)    
# * You will learn
#     - Data Preparation
#     - Understanding the target
#     - Data Visualization  
#     - Baseline model 
# Source: https://www.kaggle.com/robikscube/2019-data-science-bowl-an-introduction
# 
# ### A baseline for DSB 2019 by [@mhviraf ](https://www.kaggle.com/mhviraf/a-baseline-for-dsb-2019)    
# * You will learn  
#     - Baseline model 
# Source: https://www.kaggle.com/mhviraf/a-baseline-for-dsb-2019
# 
# ### 2019 Data Science Bowl EDA by [@gpreda ](https://www.kaggle.com/gpreda/2019-data-science-bowl-eda)    
# * You will learn  
#     - Prepare the data analysis
#     -Data exploration
#     -Glimpse the data
#     -Missing data
#     -Unique values
#     -Most frequent values
#     -Values distribution
#     -Extract features from train/event_data
#     -Extract features from specs/args
#     -Merged data distribution
# Source: https://www.kaggle.com/gpreda/2019-data-science-bowl-eda
# 
# ### Convert to Regression by [@braquino ](https://www.kaggle.com/braquino/convert-to-regression)    
# * You will learn  
#     - Baseline model including a feature selection part
#     - Cohen cappa score of 0.456 (lb) with a local cv score of 0.529
#     - Add/remove features to improve local cv
# Source: https://www.kaggle.com/braquino/convert-to-regression
# 
# ### Quick and dirty regression by [@artgor ](https://www.kaggle.com/artgor/quick-and-dirty-regression)    
# * You will learn  
#     - Baseline model including a feature selection part
#     - Add/remove features to improve local cv
#     - Finding optimal coefficients for thresholds
# Source: https://www.kaggle.com/artgor/quick-and-dirty-regression
# 
# 
# 

# 
# <html>
# <body>
# 
# <p><font size="5" color="Blue">Remember the upvote button is next to the fork button, and it's free too! ;)</font></p>
# <p><font size="4" color="Purple">Don't hesitate to give your suggestions in the comment section</font></p>
# 
# </body>
# </html>
# 

# ## Final
