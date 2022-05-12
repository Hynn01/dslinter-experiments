#!/usr/bin/env python
# coding: utf-8

# ### Summary:
# * [Problem definition](#problem_definition)
# * [Data Analysis](#data_analysis)
#     * [Numeric Variables](#numeric_variables)
#     * [Categoric Variables](#categoric_variables)
#     * [Attribute Selection](#attribute_selection)
# * [Model](#model)
#     * [Tuning Hyperparameters](#tuning_hyper)
# * [Submission](#submission)

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind
from sklearn import ensemble
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn import model_selection
pd.set_option('display.max_columns', 500)


# <a id='problem_definition'></a>
# ## Problem Definition:
# For this challenge, you are given (simulated) manufacturing control data and are tasked to predict whether the machine is in state 0 or state 1. The data has various feature interactions that may be important in determining the machine state.    

# In[ ]:


train_data = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/train.csv')
test_data = pd.read_csv('/kaggle/input/tabular-playground-series-may-2022/test.csv')


# In[ ]:


features = train_data.columns.tolist()
features.remove('target')
target = 'target'


# In[ ]:


X_train, X_test, y_train, y_test = model_selection.train_test_split(train_data[features],
                                                                    train_data[target],
                                                                    random_state=42,
                                                                    test_size=0.2)


# In[ ]:


X_train_analysis = pd.concat([X_train, y_train],axis=1)


# <a id='data_analysis'></a>
# ## Data Analysis:
# As we can see, we only have 1 categorical column and the other columns are numeric and we don't have any missing values. Also, 15 of the columns are whole numbers and the others have been floated or negative.

# In[ ]:


def get_type_column_and_na(data, column):
    print(f"{column}, {data[column].dtype}, {data[column].isna().mean()}")


# In[ ]:


print("column_name, dtype, na_values")
for column in X_train_analysis.columns:
    get_type_column_and_na(train_data,column)
print(X_train_analysis.shape)


# In[ ]:


X_train_analysis.sample(5,random_state=50)


# <a id='numeric_variables'></a>
# As we can see, we have a balanced data set 51.3% for target 0 and 48.7% for 1. When we did a variable description we see:
# 
# **Floating Point Variables:**
# 
# **f_00, f_01, f_02 and f_05:** Target 0 has a negative mean while the other target has a positive mean and the standard deviation is close
# 
# **f_03:** Both targets with negative mean and with target 0 being lower than the other target, standard deviation close.
# 
# **f_04:** Both targets with negative mean and with target 0 being higher relative to the other target, standard deviation close.
# 
# **f_06:** Target 1 has a negative mean, standard deviation close in relation to the other target.
# 
# **f_19:** Both averages are positive, however target 0 shows a higher value in relation to target 1
# 
# **f_20:** Both targets with negative average and with target 0 being inferior in relation to the other target, different standard deviation.
# 
# **f_21 and f_22:** Target 0 average is negative while target 1 is positive
# 
# **f_23:** Both averages are negative, however, target 0 has a lower average and smaller dispersion. 75% of the data for target 0 is below 0.97 while for another target it is below 1.55
# 
# **f_24:** Both averages are negative however, target 0 has a lower average and a larger dispersion. 75% of the data for target 0 is below 1.04 while for another target it is below 1.47
# 
# **f_25:** Both means are positive however target 0 has a higher mean and std compared to the other target. 50% of the data for target 0 is below 0.23 while for the other target it is below 0.09
# 
# **f_26:** Averages are positive and target 0 has a lower mean and std value compared to the other target.
# 
# **fh_28:** Mean of target 0 is negative and lower than that of target 1, furthermore this is a variable that shows high std for both targets relative to other variables.
# 
# **Integer numeric variables:**
# 
# **f_08:** The mean of target 0 is lower and std is higher relative to that of target 1
# 
# **f_09, f_15, f_29 and f_30:** Target 0 average and std is lower relative to target 1
# 
# **f_10:** Target 0's mean is higher and std is lower relative to target 1
# 
# **f_11, f_13, f_14 and f_16:** Target 0 average and std is higher relative to target 1
# 
# **f_17:** Target 0 has lower mean and std is higher relative to target 1
# 
# **f_18:** Target 0 has higher mean and std is close
# 
# We have variables with the same logic or pattern, perhaps there is a relationship between these variables and it won't be necessary to include some of them in the model.

# In[ ]:


X_train_analysis.groupby('target').describe()


# <a id='categoric_variables'></a>
# **Categorical Variables**:
# 
# We have 283.678 unique strings for target 1 and 294.088 strings for target 0. 19% of the strings are for both target 1 and target 0

# In[ ]:


f_27_analysis = X_train_analysis.groupby(['f_27','target']).size().reset_index(name='count')
f_27_analysis = pd.pivot_table(f_27_analysis, values='count', index=['f_27'],columns=['target'], aggfunc=np.sum,fill_value=0).reset_index()
f_27_analysis.rename(columns={1:'target_1',0:'target_0'},inplace=True)


# In[ ]:


exclusive_1 = f_27_analysis[(f_27_analysis['target_1'] > 0) & (f_27_analysis['target_0'] == 0)].shape[0]
exclusive_0 = f_27_analysis[(f_27_analysis['target_0'] > 0) & (f_27_analysis['target_1'] == 0)].shape[0]
print(f'exclusive target 1: {exclusive_1}')
print(f'exclusive target 0: {exclusive_0}')


# As we can see, all strings have the same length regardless of the target variable.

# In[ ]:


f_27_length = X_train_analysis[['target','f_27']].copy()
f_27_length['string_length'] = X_train_analysis['f_27'].apply(len)
f_27_length.groupby(['target'])['string_length'].describe()


# We will apply a count term to measure the importance of a character to f_27 string for each target.

# In[ ]:


def get_count_vectorizer_df(data):
    countvectorizer = CountVectorizer(analyzer='char')
    cv_fit = countvectorizer.fit_transform(data)
    cv_tokens = countvectorizer.get_feature_names_out()
    return pd.DataFrame(data = cv_fit.toarray(),index = data, columns = cv_tokens)


# In[ ]:


tokens_f_27 = list(X_train_analysis['f_27'])

cv = get_count_vectorizer_df(tokens_f_27).reset_index()
cv = cv.drop_duplicates()
cv.rename(columns={'index':'f_27'},inplace=True)


# In[ ]:


X_train_analysis = X_train_analysis.merge(cv,on=['f_27'],how='left')


# As we can see, we have a difference of 8.2% for character B, whose target 0 is more present, and 9% for character G, whose target 1 is more present. Characters M,N,O,P,Q,R are less frequent in both targets, but more present in target 1. Finally, characters S and T have a high difference between the datasets.  

# In[ ]:


mean_count = X_train_analysis.groupby('target')[X_train_analysis.columns[33:]].mean().reset_index()
mean_count = mean_count.T.reset_index()
mean_count['difference'] = ((mean_count[0] - mean_count[1]) / mean_count[0]) * 100
mean_count.sort_values('difference',ascending=False)


# <a id='attribute_selection'></a>
# ## Attribute Selection:
# 
# It seeks to select attributes that are correlated with the target variable and that are relevant for distinguishing between groups. The statistical significance value used was 0.05 (p-value), that is, if the p-value is less than 0.05, it indicates that there is a difference between that attribute for the target variable.

# In[ ]:


def get_p_value_depedence(data,column):
    return ttest_ind(data[data['target']==0][column], data[data['target']==1][column])[1]


# In[ ]:


numeric_columns = ['f_00', 'f_01', 'f_02', 'f_03', 'f_04', 'f_05', 'f_06', 'f_07',
                  'f_08', 'f_09', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 
                  'f_16','f_17', 'f_18', 'f_19','f_20', 'f_21', 'f_22', 'f_23', 
                  'f_24', 'f_25','f_26', 'f_28', 'f_29', 'f_30']
ttest_result = []
for column in numeric_columns:
    ttest_result.append({'column':column,'p-value':get_p_value_depedence(X_train_analysis,column)})


# We can remove the columns f_03, f_04, f_06, f_12 and f_17

# In[ ]:


ttest_result = pd.DataFrame(ttest_result)
drop_numeric_columns = ttest_result[ttest_result['p-value'] > 0.05]['column'].to_list()
ttest_result.sort_values('p-value',ascending=False)


# <a id='model'></a>
# ### Model:

# In[ ]:


X_train = X_train.merge(cv,on='f_27',how='left')
X_train.drop(drop_numeric_columns,axis=1,inplace=True)
X_train.drop(['id','f_27'],axis=1,inplace=True)


# We use a random forest model, which is an ensemble. Ensemble methods use the combination of several models, which in this case are decision trees, to get a single result that is given by the majority vote for that class.

# In[ ]:


def report_model(X, y, model):
    y_pred = model.predict(X)
    print("Accuracy: {:.2f}".format(metrics.accuracy_score(y,y_pred) * 100))
    print("roc auc score: {:.2f}".format(metrics.roc_auc_score(y,y_pred)))
    print(metrics.classification_report(y, y_pred))


# In[ ]:


rf_clf = ensemble.RandomForestClassifier(n_estimators=200,
                                         min_samples_leaf=20,                                        
                                         n_jobs=-1,
                                         random_state=42)


# In[ ]:


rf_clf.fit(X_train, y_train)


# As we can see, we have a good model with high Accuracy and recall, but with a small problem with target 1, where the model makes no difference between the targets compared to recall target 0.

# In[ ]:


report_model(X_train, y_train, rf_clf)


# **Testing Model in Test data:**

# When we test our model on test data, we see a drop in recall for both targets and especially for target 1. We will improve this performance by tuning the hyperparameters.

# In[ ]:


f_27_tokens = list(X_test['f_27'])
cv = get_count_vectorizer_df(f_27_tokens)
cv = cv.reset_index().drop_duplicates()
cv.rename(columns={'index':'f_27'},inplace=True)


# In[ ]:


X_test = X_test.merge(cv,on='f_27',how='left')
X_test.drop(drop_numeric_columns,axis=1,inplace=True)
X_test.drop(['id','f_27'],axis=1,inplace=True)


# In[ ]:


report_model(X_test, y_test, rf_clf)


# <a id='tuning_hyper'></a>
# **Tuning the hyperparameters:**
# 
# We get a score of 89.8% using min_samples_leaf = 5 and n_estimators = 250 the best value of hyperparameters.

# In[ ]:


params = {"n_estimators":[50,100,200,250],
          "min_samples_leaf": [5,10,20,50,100] }

grid_search = model_selection.GridSearchCV(rf_clf,params,cv=2,scoring='roc_auc',verbose=3,refit=True)

grid_search.fit(X_train, y_train)
pd.DataFrame(grid_search.cv_results_).sort_values('rank_test_score',ascending=True)


# In[ ]:


rf_clf = ensemble.RandomForestClassifier(n_estimators=250,
                                         min_samples_leaf=5,                                        
                                         n_jobs=-1,
                                         random_state=42)
rf_clf.fit(X_train, y_train)


#  When we tested with train data we got a 98% roc auc score, a 9% improvement.

# In[ ]:


report_model(X_train, y_train, rf_clf)


# When we tested the model with the test data, we see a 1% of improvement.

# In[ ]:


report_model(X_test, y_test, rf_clf)


# As we can see, we have good importance of feature distribution. The first feature importance is f_26 and the value count of character A and B has more importance than 11 features where it already existed in the data set.

# In[ ]:


feature_importances = pd.DataFrame(rf_clf.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)
feature_importances.head()


# <a id='submission'></a>
# ### Submission:
# Creating the forecast for the competition test data

# In[ ]:


f_27_tokens = list(test_data['f_27'])
cv = get_count_vectorizer_df(f_27_tokens)
cv = cv.reset_index().drop_duplicates()
cv.rename(columns={'index':'f_27'},inplace=True)

X_test_data = test_data.merge(cv,on='f_27',how='left')
X_test_data.drop(drop_numeric_columns,axis=1,inplace=True)
X_test_data.drop(['id','f_27'],axis=1,inplace=True)


# In[ ]:


y_test_data = rf_clf.predict_proba(X_test_data)


# In[ ]:


result = pd.concat([test_data['id'],pd.Series(y_test_data[:,1])],axis=1)
result.rename(columns={0:'target'},inplace=True)
result.to_csv('/kaggle/working/submission.csv',index=False)


# And that’s it! It has been a pleasure to make this kernel, I have learned a lot! Thank you for reading and if you like it, please upvote it!
