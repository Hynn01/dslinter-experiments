#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from pandas.api.types import CategoricalDtype
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings("ignore")


# # Libraries:

# In[ ]:


from sklearn.metrics import roc_auc_score,matthews_corrcoef,balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif,f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPClassifier 
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from category_encoders import MEstimateEncoder
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier,VotingClassifier


# In[ ]:


train=pd.read_csv('/kaggle/input/autismdiagnosis/Autism_Prediction/train.csv')
test=pd.read_csv('/kaggle/input/autismdiagnosis/Autism_Prediction/test.csv')


# # Copy train and test sets:

# In[ ]:


train_set=train.copy()
test_set=test.copy()

#age_desc doesn't change and ID is irrelevant
train_set.drop(['age_desc','ID'],axis=1,inplace=True)

test_set.drop(['age_desc','ID'],axis=1,inplace=True)


# In[ ]:


y=train_set.pop('Class/ASD')


# In[ ]:


#Dropping useless features using chi 2, in each fold, these two were irrelevant

train_set.drop(['gender','used_app_before'],axis=1,inplace=True)


# # WHEN USING CHI SQUARE TO FIND RELATIONS:
# 
# <p style='font-size:18px;color:#16AF7C;'><i>Note: we <b>DO NOT</b> need to convert columns to any specific data type, the fact that they are discrete features make chi2 test work just fine</i></p>

# In[ ]:


def chi2_calc(df,target):
    scores=[]
    for col in df.columns:
        ct=pd.crosstab(df[col],target)
        stat,p,dof,expected=chi2_contingency(ct)
        scores.append(p)
    return pd.DataFrame(scores, index=df.columns, columns=['P value']).sort_values(by='P value')


# In[ ]:


train.head()


# In[ ]:


np.random.seed(1) #I'm using this because there's some
#randomness in how the selectors work, without this, in each run we get different results
kf = StratifiedKFold(n_splits=10, random_state=None,shuffle=False) #for cross validation/ random_state
# is None because shuffle is False
score=[]
model=lr=LogisticRegression(random_state=0,C=0.5,solver='liblinear')
for train_index, val_index in kf.split(train_set,y):
    
    #indices for train and validation sets
    X_train, X_val =train_set.iloc[train_index,:], train_set.iloc[val_index,:]
    y_train, y_val = y[train_index], y[val_index]
    
    #******************************* CLEANING ***********************************

    #for train set
    X_train.ethnicity=X_train.ethnicity.str.replace('others','Others',regex=False)
    X_train.ethnicity=X_train.ethnicity.str.replace('?','Others',regex=False)
    X_train.relation=X_train.relation.str.replace('?','Others',regex=False)
    X_train.relation=X_train.relation.str.replace('Health care professional','Others',regex=False)
    
    
    #for validation set:
    X_val.ethnicity=X_val.ethnicity.str.replace('others','Others',regex=False)
    X_val.ethnicity=X_val.ethnicity.str.replace('?','Others',regex=False)
    X_val.relation=X_val.relation.str.replace('?','Others',regex=False)
    X_val.relation=X_val.relation.str.replace('Health care professional','Others',regex=False)

    
    
    #******************************CHI2 SCORES*********************************
    
    #to see chi2 scores for each training split (A6,A4 in all of them are the top two then A9 in 6 of
    #them is #3 in the list)
    
    #print(chi2_calc(X_train[X_train.columns.difference(['age', 'result'])],y_train))
               
        
    #***************************************ENCODING****************************************** 
    
    #FOR ENCODING USE THE TRAINING VALUES, DO NOT CALCULATE THEM AGAIN FOR THE TEST SET!
    
    le=LabelEncoder()
    for col in ['jaundice','austim']:
        
        #for the training set:
        X_train[col]=le.fit_transform(X_train[col])
        
        #for the validation set:
        X_val[col]=le.transform(X_val[col])
         

    #*********************Encoding Relation Column***************************
    
    #create an encoding map, using the training set, then implementing it on val and test sets
    rel=X_train.relation.value_counts()
    rel=dict(zip(rel.index,range(len(rel))))
    
    #for the training set:
    X_train.relation=X_train.relation.map(rel)
    
    #for the validation set: if there's a category not present in the map, we'll assign sth. to it
    X_val.relation=X_val.relation.map(rel)
    X_val.relation[X_val.relation.isna()]=len(rel)
    
    
    
    #*********************Encoding Ethnicity Column***************************
    
    #create an encoding map, using the training set, then implementing it on val and test sets
    eth=X_train.ethnicity.value_counts()
    eth=dict(zip(eth.index,range(len(eth))))
    
    #for the training set:
    X_train.ethnicity=X_train.ethnicity.map(eth)
    
    #for the validation set: if there's a category not present in the map, we'll assign sth. to it
    X_val.ethnicity=X_val.ethnicity.map(eth)
    X_val.ethnicity[X_val.ethnicity.isna()]=len(eth)
    
    
    
    #*****************************Encoding Country Of Res******************************
    
    #create an encoding map, using the training set, then implementing it on val and test sets
    cont=X_train.contry_of_res.value_counts()
    cont=dict(zip(cont.index,range(len(cont))))
    
    #for the training set:
    X_train.contry_of_res=X_train.contry_of_res.map(cont)
    
    #for the validation set: if there's a category not present in the map, we'll assign sth. to it
    X_val.contry_of_res=X_val.contry_of_res.map(cont)
    X_val.contry_of_res[X_val.contry_of_res.isna()]=len(cont)
    
    #****************************Permutation Importance*********************

    
#     lr.fit(X_train, y_train)
#     result = permutation_importance(lr,X_val, y_val, n_repeats=5,
#                                 random_state=0)
#     print(pd.Series(result['importances_mean'],index=X_train.columns).sort_values(ascending=False))




   #*****************************Feature Engineering*************************
    #IF NOTHING IS DONE HERE, IN TRANSFORMING TEST SET COMMENT OUT THE LAST PART
    
#     kbest=SelectKBest(mutual_info_classif, k=17)
#     bst=kbest.fit(X_train,y_train).get_support()
#     cols=list(X_train.columns[bst])
#     print(cols)
#     X_train=kbest.fit_transform(X_train,y_train)
#     X_val=X_val.iloc[:,bst]
    mix_46=["A4_Score","A6_Score"]
    X_train['mix_46'] = X_train[mix_46].gt(0).sum(axis=1)
    X_val['mix_46'] = X_val[mix_46].gt(0).sum(axis=1)

    

     #******************************OverSampling*****************************
    ros = RandomOverSampler(random_state=0)
    # fit predictor and target variablex_ros
    X_train, y_train = ros.fit_resample(X_train, y_train)
    
    
                #*******************************************#
#     smote = SMOTE()

# #     fit predictor and target variable
#     X_train, y_train = smote.fit_resample(X_train, y_train)


    
    #*************************** Model Selection ***************************
    rf=RandomForestClassifier(n_estimators=50,random_state=0,criterion='gini')
    lr=LogisticRegression(random_state=0,C=0.5,solver='liblinear')
    svc=SVC(probability=True)
    nb=GaussianNB()
    
    estimators=[
    ('lr',lr),
    ('svc',svc),
    ('nb',nb)]
    
    stck=StackingClassifier(estimators=estimators,stack_method='predict_proba')
    vt=VotingClassifier(estimators=estimators,voting='soft',weights=[.5,.2,.1])
    
    model=lr
    
    #fit the model
    model.fit(X_train,y_train)
    
    #prediction
    y_pred=pd.DataFrame(model.predict_proba(X_val))[1].values
    
    #scoring
    score.append(roc_auc_score(y_val,y_pred))
    
display(np.array(score).mean())
display(np.array(score).std())


# # Fitting the model to the whole trainset:

# In[ ]:


#Cleaning:
train_set.ethnicity=train_set.ethnicity.str.replace('?','Others',regex=False)
train_set.relation=train_set.relation.str.replace('?','Others',regex=False)
train_set.relation=train_set.relation.str.replace('?','Others',regex=False)
train_set.relation=train_set.relation.str.replace('Health care professional','Others',regex=False)

#Encoding:

train_set['jaundice']=le.transform(train_set['jaundice'])
train_set['austim']=le.transform(train_set['austim'])


train_set.relation=train_set.relation.map(rel)
train_set.relation[train_set.relation.isna()]=len(rel)

train_set.ethnicity=train_set.ethnicity.map(eth)
train_set.ethnicity[train_set.ethnicity.isna()]=len(eth)

train_set.contry_of_res=train_set.contry_of_res.map(cont)
train_set.contry_of_res[train_set.contry_of_res.isna()]=len(cont)

#Feature Engineering
train_set['mix_46'] = train_set[mix_46].gt(0).sum(axis=1)

#Oversampling
ros = RandomOverSampler(random_state=0)
# fit predictor and target variablex_ros
train_set, y = ros.fit_resample(train_set, y)

#fitting the model
model.fit(train_set,y)


# # Transforming Test Set

# In[ ]:


#dropping irrelevant cols:
test_set.drop(['gender','used_app_before'],axis=1,inplace=True)

#Cleaning:
test_set.ethnicity=test_set.ethnicity.str.replace('?','Others',regex=False)
test_set.relation=test_set.relation.str.replace('?','Others',regex=False)
test_set.relation=test_set.relation.str.replace('?','Others',regex=False)
test_set.relation=test_set.relation.str.replace('Health care professional','Others',regex=False)

#Encoding:

test_set['jaundice']=le.transform(test_set['jaundice'])
test_set['austim']=le.transform(test_set['austim'])


test_set.relation=test_set.relation.map(rel)
test_set.relation[test_set.relation.isna()]=len(rel)

test_set.ethnicity=test_set.ethnicity.map(eth)
test_set.ethnicity[test_set.ethnicity.isna()]=len(eth)

test_set.contry_of_res=test_set.contry_of_res.map(cont)
test_set.contry_of_res[test_set.contry_of_res.isna()]=len(cont)

#result of FE:
test_set['mix_46'] = test_set[mix_46].gt(0).sum(axis=1)
# test_set=test_set[cols]


# # Feature Engineering:

# In[ ]:


# components = [ "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score","A6_Score","A7_Score","A8_Score","A9_Score","A10_Score"]
# train_set["Scores"] = train_set[components].gt(0).sum(axis=1)
# test_set["Scores"] = test_set[components].gt(0).sum(axis=1)


# # Over Sampling:
# 
# <p style= 'font-size:20px;color:powderblue;'><b><i>Few ways for over-sampling</i></b></p>

# In[ ]:


#OPTION 1
# import library

# from imblearn.over_sampling import RandomOverSampler

# ros = RandomOverSampler(random_state=0)

# # fit predictor and target variablex_ros

# train_set, y = ros.fit_resample(train_set, y)


# In[ ]:


# #OPTION 2: Doesn't really work here because it messes up the indices and have to fix it also need to move
#the dropping of target col to after the separation
# from sklearn.utils import resample

# # Separate Target Classes
# train_set_1= train_set[train_set['Class/ASD']==0]
# train_set_2 = train_set[train_set['Class/ASD']==1]

# # Upsample minority class
# train_set_2_upsampled = resample(train_set_2, replace=True,     # sample with replacement
#                           n_samples=639,    # to match majority class
#                           random_state=0) # reproducible result
# # Combine majority class with upsampled minority class
# train_set = pd.concat([train_set_1, train_set_2_upsampled])
 
# # Display new class counts
# train_set['Class/ASD'].value_counts()


# In[ ]:


# OPTION 3
# from imblearn.over_sampling import SMOTE

# smote = SMOTE()

# # fit predictor and target variable
# train_set, y = smote.fit_resample(train_set, y)


# # Feature_Importance:

# In[ ]:


# res=mutual_info_classif(train_set, y, discrete_features=train_set.dtypes == int, random_state=0)
# scores=pd.Series(res,index=train_set.columns,name='scores').sort_values(ascending=False)
# scores


# In[ ]:


# from sklearn.inspection import permutation_importance

# rf=RandomForestClassifier(n_estimators=20,random_state=0,criterion='gini')
# rf.fit(train_set, y)
# result = permutation_importance(rf,train_set, y, n_repeats=10,
#                                 random_state=0)
# pd.Series(result['importances_mean'],index=train_set.columns).sort_values(ascending=False)


# # Learning Curve:

# In[ ]:


from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(model, train_set, y, cv=5,
                                                        scoring='roc_auc', n_jobs=-1, train_sizes=np.linspace(0.1, 1, 5))


# In[ ]:


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


# In[ ]:


plt.subplots(1, figsize=(10,10))
plt.plot(train_sizes, train_mean, '--', color="blue",  label="Training score")
plt.plot(train_sizes, test_mean, color="yellow", label="Val score")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="green")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="green")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("neg-logloss Score"), plt.legend(loc="best")
plt.tight_layout()
plt.show()


# # Create Submission:

# In[ ]:


predictions = pd.DataFrame(model.predict_proba(test_set))[1].values

output = pd.DataFrame({'ID': test['ID'], 'Class/ASD': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# In[ ]:


output.head(20)


# In[ ]:




