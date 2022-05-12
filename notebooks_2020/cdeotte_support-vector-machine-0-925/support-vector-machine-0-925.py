#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machine scores LB 0.925
# In our previous kernel [here][1], we built 512 **linear** models using logistic regression in conjuction with the **magic feature** and scored LB 0.808. In this kernel we will build 512 **nonlinear** models using support vector machine with polynomial degree=4 kernel and score LB 0.926. Previously we performed feature selection with Lasso, aka LR's L1-penalty. Now we will perform feature selection with sklearn's VarianceThreshold selector (which will select more or less the same features). 
# 
# The success of this kernel demonstrates the nature of "Instant Gratification" competition data. It appears that the data is actually 512 datasets combined together. Each dataset has rougly 512 observations. Thus the total training data has `262144 = 512 * 512` observations. Each partial dataset is identified by a unique `wheezy-copper-turtle-magic` value. This kernel shows that each dataset's target is a nonlinear function of approximately 40 important features (and each dataset uses a different 40 important features).  
#   
# The next thing to investigate is whether there are interactions between the partial datasets that can improve prediction. If that is the case then instead of building 512 separate models, we need to build a single model that allows interactions. (Possibly NN with interesting architecture). 
# 
# Also each model (in this kernel) only uses about 40 features. Within each partial dataset, are the other 215 features really useless, or can we use them to improve prediction?
# 
# # Load Data
# 
# [1]: https://www.kaggle.com/cdeotte/logistic-regression-0-800

# In[1]:


import numpy as np, pandas as pd, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()


# # Build 512 Support Vector Models
# Using cross validation, we determined that SVC's polynomial kernel with degree=4 achieves the best CV.

# In[ ]:


# LOAD LIBRARIES
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# In[ ]:


# INITIALIZE VARIABLES
oof = np.zeros(len(train))
preds = np.zeros(len(test))
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

# BUILD 512 SEPARATE NON-LINEAR MODELS
for i in range(512):
    
    # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    
    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
        
    # STRATIFIED K FOLD (Using splits=25 scores 0.002 better but is slower)
    skf = StratifiedKFold(n_splits=11, random_state=42)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        # MODEL WITH SUPPORT VECTOR MACHINE
        clf = SVC(probability=True,kernel='poly',degree=4,gamma='auto')
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
    #if i%10==0: print(i)
        
# PRINT VALIDATION CV AUC
auc = roc_auc_score(train['target'],oof)
print('CV score =',round(auc,5))


# # Submit Predictions

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv',index=False)

import matplotlib.pyplot as plt
plt.hist(preds,bins=100)
plt.title('Test.csv predictions')
plt.show()


# # Conclusion
# In conclusion, the success of using 512 separate models suggests that the 262144 rows of "Instant Gratification" dataset may actually be 512 partial datasets that were combined. Is this competition actually 512 models (competitions) in one? The appendix below investigates this further.
# 
# Three suggestions to improve this kernel's accuracy are (1) identify and model interactions between partial datasets (2) extract information from the approx 215 variables that are not used in each model (3) build better partial models than SVC polynomial degree=4 via NN or LGBM.
# 
# # Appendix
# ## Variables are not Gaussian
# Each variable appears to be a nice bell shaped curve. However the curves are too tall and narrow. This is seen below by comparing the distribution of some variables with a Gaussian of the same mean and standard deviation. One can also verify that the variables are not Gaussian by making normality plots (pictured below).

# In[10]:


# LOAD LIBRARIES
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# PLOT FIRST 8 VARIABLES
plt.figure(figsize=(15,15))
for i in range(8):
    plt.subplot(3,3,i+1)
    #plt.hist(train.iloc[:,i+1],bins=100)
    sns.distplot(train.iloc[:,i+1],bins=100)
    plt.title( train.columns[i+1] )
    plt.xlabel('')
    
# PLOT GAUSSIAN FOR COMPARISON
plt.subplot(3,3,9)
std = round(np.std(train.iloc[:,8]),2)
data = np.random.normal(0,std,len(train))
sns.distplot(data,bins=100)
plt.xlim((-17,17))
plt.ylim((0,0.37))
plt.title("Gaussian with m=0, std="+str(std))

plt.subplots_adjust(hspace=0.3)
plt.show()


# ## Normality Plots
# Normality plots indicate that the variables are not Gaussian. If they were Gaussian, then we would see straight lines below. Instead we see piecewise straight lines indicating that we may have Gaussian mixture models. (Each variable is the sum of multiple Gaussians).

# In[12]:


# NORMALITY PLOTS FOR FIRST 8 VARIABLES
plt.figure(figsize=(15,15))
for i in range(8):
    plt.subplot(3,3,i+1)
    stats.probplot(train.iloc[:,1], plot=plt)
    plt.title( train.columns[i+1] )
    
# NORMALITY PLOT FOR GAUSSIAN
plt.subplot(3,3,9)
stats.probplot(data, plot=plt)   
plt.title("Gaussian with m=0, std="+str(std))

plt.subplots_adjust(hspace=0.4)
plt.show()


# ## Variables within partial datasets are Gaussian
# If we only look at the partial datasets where `wheezy-copper-turtle-magic = k` for `0 <= k <= 511`, then the variables are Gaussian. This can be seen by the plots below. This suggests how the full dataset was made. Perhaps Kaggle made 512 different datasets and then combined them for this competition.

# In[4]:


train0 = train[ train['wheezy-copper-turtle-magic']==0 ]


# In[14]:


# PLOT FIRST 8 VARIABLES
plt.figure(figsize=(15,15))
for i in range(8):
    plt.subplot(3,3,i+1)
    #plt.hist(train0.iloc[:,i+1],bins=10)
    sns.distplot(train0.iloc[:,i+1],bins=10)
    plt.title( train.columns[i+1] )
    plt.xlabel('')
    
# PLOT GAUSSIAN FOR COMPARISON
plt.subplot(3,3,9)
std0 = round(np.std(train0.iloc[:,8]),2)
data0 = np.random.normal(0,std0,2*len(train0))
sns.distplot(data0,bins=10)
plt.xlim((-17,17))
plt.ylim((0,0.1))
plt.title("Gaussian with m=0, std="+str(std0))
    
plt.subplots_adjust(hspace=0.3)
plt.show()


# ## Partial dataset normality plots

# In[15]:


# NORMALITY PLOTS FOR FIRST 8 VARIABLES
plt.figure(figsize=(15,15))
for i in range(8):
    plt.subplot(3,3,i+1)
    stats.probplot(train0.iloc[:,1], plot=plt)
    plt.title( train.columns[i+1] )
    
# NORMALITY PLOT FOR GAUSSIAN
plt.subplot(3,3,9)
stats.probplot(data0, plot=plt)   
plt.title("Gaussian with m=0, std="+str(std0))

plt.subplots_adjust(hspace=0.4)
plt.show()

