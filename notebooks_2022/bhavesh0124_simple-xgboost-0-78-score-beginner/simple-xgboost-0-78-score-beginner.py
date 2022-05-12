#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import LabelEncoder


# In[ ]:


df_train=pd.read_csv("../input/titanic/train.csv")
df_test=pd.read_csv("../input/titanic/test.csv")
df_train['Title'] = df_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df_test['Title'] = df_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


combined=pd.concat([df_train,df_test])


# In[ ]:


combined['Family_members'] = combined['SibSp'] + combined['Parch']
combined.drop(["Name","SibSp","Parch","Cabin","Ticket"],axis=1,inplace=True)


# In[ ]:


combined["Title"]=combined["Title"].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
combined["Title"]=combined["Title"].replace(['Mlle','Ms'],'Miss')
combined["Title"]=combined["Title"].replace(['Mme'],'Mrs')


# Missing Values handle

# In[ ]:


combined['Age'] = combined['Age'].fillna(combined.groupby(["Title"])['Age'].transform('mean'))
combined['Fare'] = combined['Fare'].fillna(combined.groupby(["Pclass"])['Fare'].transform('mean'))
combined['Embarked']=combined['Embarked'].fillna("S")


# In[ ]:


combined.loc[(combined['Pclass'] == 1),'Pclass_Band'] = 3
combined.loc[(combined['Pclass'] == 3),'Pclass_Band'] = 1
combined.loc[(combined['Pclass'] == 2),'Pclass_Band'] = 2


# In[ ]:


combined = pd.get_dummies(combined,columns=["Sex","Embarked","Title"])
combined.drop(["Sex_male","Embarked_S"],axis=1,inplace=True)
combined["Title_Mr"]=0


# In[ ]:


combined["Pclass_Band"]=combined["Pclass_Band"].astype(int)
combined.drop(["Pclass"],axis=1,inplace=True)


# In[ ]:


df_train=combined.iloc[:891,]
df_test=combined.iloc[891:,]


# In[ ]:


df_train.drop(["PassengerId"],axis=1,inplace=True)
df_train["Survived"]=df_train["Survived"].astype(int)


# In[ ]:


Y_train = df_train['Survived']
X_train = df_train.drop('Survived', axis=1)


# In[ ]:


Y_test = df_test['Survived']
X_test = df_test.drop('Survived', axis=1)


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from operator import itemgetter
import xgboost as xgb
import random
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc,recall_score,precision_score
import datetime as dt


# In[ ]:


def run_single(train, test, features, target, random_state=0):    
    eta = 0.4
    max_depth= 6
    subsample = 1
    colsample_bytree =1
    n_estimators=400
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "n_estimators":n_estimators,
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        #"min_chil_weight":min_chil_weight,
        "seed": random_state,
        #"gamma":gamma
        #"num_class" : 22,
    }
    num_boost_round = 3000
    early_stopping_rounds = 100
    test_size = 0.1

    
    X_train, X_valid = train_test_split(train, test_size=test_size, random_state=random_state)
    print('Length train:', len(X_train.index))
    print('Length valid:', len(X_valid.index))
    
    y_train = X_train[target]
    y_valid = X_valid[target]
    
    dtrain = xgb.DMatrix(X_train[features], y_train, missing=-99)
    dvalid = xgb.DMatrix(X_valid[features], y_valid, missing =-99)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid[features]), ntree_limit=gbm.best_iteration+1)
    
    #area under the precision-recall curve
    score = average_precision_score(X_valid[target].values, check)
    print('area under the precision-recall curve: {:.6f}'.format(score))

    
    check2=check.round()
    score = precision_score(X_valid[target].values, check2)
    print('precision score: {:.6f}'.format(score))

    score = recall_score(X_valid[target].values, check2)
    print('recall score: {:.6f}'.format(score))
    #xgb.plot_importance(model, height=0.4, ax=ax)
    #imp = get_importance(gbm, features)
    #print('Importance array: ', imp)

    print("Trainin and prediction again test set... ")
    
    d_final = xgb.DMatrix(train[features], train[target], missing =-99)
    
    gbm = xgb.train(params, d_final)
    test_prediction = gbm.predict(xgb.DMatrix(test[features],missing = -99), ntree_limit=gbm.best_iteration+1)
    
    df_test["Survived"]=pd.DataFrame(test_prediction)
    
    xgb.plot_importance(gbm, height=0.4)
    
    fpr, tpr, _ = roc_curve(X_valid[target].values, check)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()


    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction,gbm


# In[ ]:


features = list(df_train.columns.values)
features.remove('Survived')

train, test = train_test_split(df_train, test_size=.1, random_state=random.seed(42))

preds,num_boost_rounds = run_single(df_train, df_test.drop(["PassengerId"],axis=1), features,'Survived',42)


# In[ ]:


df_test["Survived"]=df_test["Survived"].apply(lambda x: round(x))


# In[ ]:


submission=df_test[["PassengerId","Survived"]]


# In[ ]:


submission.to_csv("submission-2",index=False)


# In[ ]:


submission


# In[ ]:




