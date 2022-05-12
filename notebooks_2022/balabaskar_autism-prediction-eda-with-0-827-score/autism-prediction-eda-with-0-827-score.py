#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


# In[ ]:


# Load the dataset
train_df = pd.read_csv('../input/autismdiagnosis/Autism_Prediction/train.csv')
test_df = pd.read_csv('../input/autismdiagnosis/Autism_Prediction/test.csv')
sample_df = pd.read_csv('../input/autismdiagnosis/Autism_Prediction/sample_submission.csv')

train_df.shape, test_df.shape, sample_df.shape


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


# Identify the target column
[x for x in train_df.columns if x not in test_df.columns]


# In[ ]:


# Using info() to find the non null rows and data type across columns
train_df.info()


# In[ ]:


# Function to extract the categorical columns and numerical columns in separate list for ease of EDA
def get_num_cat_cols(df):
    
    num_cls = [x for x in df.columns if df[x].dtype != 'object']
    cat_cls = [x for x in df.columns if x not in num_cls]

    print(f'Numerical columns \n',num_cls)
    print(f'Categorical columns \n',cat_cls)
    
    return num_cls, cat_cls


# In[ ]:


num_cols, cat_cols = get_num_cat_cols(train_df)


# In[ ]:


# Descriptive statistics for numerical columns
train_df[num_cols].describe()


# **Inference**
# 1. Columns - A1_Score to A10_Score has value either 0 or 1, to be considered as binary variable
# 2. Age starts from min 2 yrs to max 89 yrs with median value in 24.8 yrs and average value in 28.45 yrs
# 3. Result value ranges from -6.13 to +15.85
# 4. Target column indicates there is 20% have Autism and 80% dont have autism

# In[ ]:


# Find the unique values with counts in categorical variables, and check for special characters to do cleaning
for c in cat_cols:
    print('-'*100)
    print(f'Column Name - {c}')
    print('-'*100)
    print(train_df[c].value_counts())


# For columns **relation,ethinicity** we have special character **'?'**, which we will replace with **others**

# In[ ]:


train_df['relation'] = train_df['relation'].replace('?','Others')
train_df['ethnicity'] = train_df['ethnicity'].replace('?','others')

test_df['relation'] = test_df['relation'].replace('?','Others')
test_df['ethnicity'] = test_df['ethnicity'].replace('?','others')


# Column - **age_desc** contains single value across all rows, so it is good to drop this column

# In[ ]:


# checking the value in test data, to drop the column in both train and test dataset
test_df['age_desc'].value_counts()


# In[ ]:


train_df.drop('age_desc',axis=1,inplace=True)
test_df.drop('age_desc',axis=1,inplace=True)


# In[ ]:


# Recreating the numerical and categorical columns list
num_cols, cat_cols = get_num_cat_cols(train_df)


# The values in column with Score is either 0 or 1, so it has to be considered as nominal categorical variable. Adding these columns to cat_cols list

# In[ ]:


# Moving the binary variables in numerical columns to categorical columns list, for EDA
cat_cols = cat_cols + [x for x in num_cols if '_Score' in x]
print(cat_cols)
num_cols = [x for x in num_cols if x not in cat_cols]
print(num_cols)


# # EDA

# We have 2 only numerical columns - **age, result**

# In[ ]:


# Feature - age
sns.boxplot(x = train_df['Class/ASD'], y = train_df['age'])


# **Inference** 
# 1. Median Age of persons having autism is more than who dont have
# 2. Range for both values of target class is same, thus age cannot be good classifier of target classm below histogram indicates the same

# In[ ]:


sns.histplot(x = train_df['age'], hue = train_df['Class/ASD'])


# In[ ]:


# Feature - result
sns.boxplot(x = train_df['Class/ASD'], y = train_df['result'])


# **Inference** 
# 1. Median result of persons having autism is more than who dont have
# 2. Range of result data for person having autism is shorter and higher when compared to person not having autism, this is could be potential classifier of target class

# In[ ]:


sns.histplot(x = train_df['result'], hue = train_df['Class/ASD'])


# In[ ]:


# Function to conduct chi square test between categorical feature and target feature 
def chi_sq_test(ct):
    # input crosstab of 2 categorical variables
    stat, p, dof, expected = chi2_contingency(ct)

    # interpret p-value
    alpha = 0.05
    print("p value is " + str(p))
    if p <= alpha:
        print('Both variables are Dependent (reject H0)')
    else:
        print('Both variables are Independent (H0 holds true)')


# In[ ]:


# Function to plot stacked bar graph between categorical feature and target variable, 
# also helps us know the dependent variable using chisquare function - this step is feature selection
def cat_col_test(df, cat_colname, target_colname):
    
    print(f"Column name - {cat_colname}")
    ct = pd.crosstab(df[cat_colname],df[target_colname])
    chi_sq_test(ct)
    ax = ct.plot(kind='bar',stacked=True, rot = 45)
    ax.legend(title='mark', bbox_to_anchor=(1, 1.02), loc='upper left')

    for c in ax.containers:
        ax.bar_label(c, label_type='center')
    plt.show()


# In[ ]:


for c in cat_cols:
    cat_col_test(train_df, c, 'Class/ASD')


# **Inference**
# 
# 1. **used_app_before, gender** are independent from target variable, in chisquare test. Hence, dropping these variable for model building stage.

# In[ ]:


train_df.drop(['used_app_before','gender'],axis=1,inplace=True)
test_df.drop(['used_app_before','gender'],axis=1,inplace=True)


# In[ ]:


train_df.columns


# In[ ]:


# one hot encoding for categorical variables with only 2 unique values
train_df['jaundice'] = np.where(train_df['jaundice'] == 'yes',1,0)
test_df['jaundice'] = np.where(test_df['jaundice'] == 'yes',1,0)

train_df['austim'] = np.where(train_df['austim'] == 'yes',1,0)
test_df['austim'] = np.where(test_df['austim'] == 'yes',1,0)


# In[ ]:


# Label encoding based on frequency values of categorical variable
dict_ethnicity = dict(zip(train_df['ethnicity'].value_counts().index, range(1,train_df['ethnicity'].nunique()+1)))
dict_ethnicity


# In[ ]:


train_df['ethnicity'] = train_df['ethnicity'].map(dict_ethnicity)
test_df['ethnicity'] = test_df['ethnicity'].map(dict_ethnicity)


# In[ ]:


col_name = 'contry_of_res'
dict_country = dict(zip(train_df[col_name].value_counts().index, range(1,train_df[col_name].nunique()+1)))
train_df[col_name] = train_df[col_name].map(dict_country)
test_df[col_name] = test_df[col_name].map(dict_country)


# In[ ]:


col_name = 'relation'
dict_relation = dict(zip(train_df[col_name].value_counts().index, range(1,train_df[col_name].nunique()+1)))
train_df[col_name] = train_df[col_name].map(dict_relation)
test_df[col_name] = test_df[col_name].map(dict_relation)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


# Fillna with mode value for categorical variable
test_df['contry_of_res'].fillna(1.0,inplace=True)


# # Model building

# In[ ]:


X = train_df.drop(['ID','Class/ASD'],axis=1)
y = train_df['Class/ASD']

X.shape, y.shape


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=123)
scores = cross_val_score(model, X, y, scoring = 'roc_auc', cv = 5)
print(scores)
print(np.mean(scores))


# **CV score of ROC_AUC looks high - 0.90 score**
# 
# Lets train the model directly with train data and make prediction in test data

# In[ ]:


model = RandomForestClassifier(random_state=123)
model.fit(X,y)


# In[ ]:


y_pred = pd.DataFrame(model.predict_proba(test_df.drop('ID',axis=1)))[1].values


# In[ ]:


submission = pd.DataFrame({'ID':test_df['ID'],
                           'Class/ASD':y_pred})
submission.head()


# In[ ]:


submission.to_csv("submission.csv",index=False)

