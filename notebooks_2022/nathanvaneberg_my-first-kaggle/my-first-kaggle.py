#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import xgboost as xgb
import sklearn 
import seaborn as sns
import hyperopt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
SEED = 1

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


trainData = pd.read_csv("/kaggle/input/titanic/train.csv")
testData = pd.read_csv("/kaggle/input/titanic/test.csv")
genderSubmission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
trainData.head()


# In[ ]:


testData.head()


# In[ ]:


trainData.hist(figsize=(20,20),bins=100)
plt.show()


# In[ ]:


print(trainData.columns)
print("Size of the dataset")
print(len(trainData))
for col in trainData.columns:
    print("Col : ", col)
    if(col == 'Name' or col == 'PassengerId'):
        continue
    print('map ', np.array(map(str, trainData[col].unique())))
    print(np.sort(np.array(list(map(str, trainData[col].unique())))))


# ## Cabin extraction

# In[ ]:


def CabinLeter(x):
    if(pd.isnull(x)):
        return np.nan
    else:
        return x[0]
    
def CabinNumber(x):
    if(pd.isnull(x)):
        return np.nan
    else:
        #Trait the case where we have more than 1 x
        x = x.split()
        if(len(x[0]) < 2):
            return np.nan
        return int(x[0][1:])
    
def CabinLen(x):
    if(pd.isnull(x)):
        return np.nan
    else:
        x = x.split()
        return len(x)

def CabinExtraction(data):
    data['CabinLetter'] = data['Cabin'].apply(CabinLeter)
    data['CabinNumber'] = data['Cabin'].apply(CabinNumber)
    data['CabinLen'] = data['Cabin'].apply(CabinLen)
    
CabinExtraction(testData)
CabinExtraction(trainData)


# In[ ]:


print(trainData['CabinLetter'].unique())
print(trainData['CabinNumber'].unique())
print(trainData['CabinLen'].unique())


# # Name extraction

# In[ ]:


def familyName(x):
    if(pd.isnull(x)):
        return np.nan
    else:
        return x.split()[0][:-1]
    
trainData['familyName'] = trainData['Name'].apply(familyName)
testData['familyName'] = testData['Name'].apply(familyName)

print(trainData['familyName'].value_counts())
valueCountRed = trainData['familyName'].value_counts()[trainData['familyName'].value_counts()>5]
print(valueCountRed[:10])
print('Number value count red', len(valueCountRed))


# In[ ]:


#Let's look at the Andersson :
trainData[trainData['familyName'] == 'Andersson']


# ## Ticket extraction

# In[ ]:


def ticketNumber(x):
    if(pd.isnull(x)):
        return np.nan
    else:
        x = x.split()
        if(x[-1] == 'LINE'):
            return np.nan
        return int(x[-1])

def ticketText(x):
    if(pd.isnull(x)):
        return np.nan
    else:
        x = x.split()
        if(len(x)==1):
            return " "
        else:
            return x[0]
        
def uniqueTicket(x):
    return x in non_unique_tickets.index


def ticketExtraction(data):
    data['ticketNumber'] = data['Ticket'].apply(ticketNumber)
    data['ticketText'] = data['Ticket'].apply(ticketText)
    data['uniqueTicket'] = data['Ticket'].apply(uniqueTicket)
    

non_unique_tickets = trainData['Ticket'].value_counts()[trainData['Ticket'].value_counts().values>1]
    
ticketExtraction(trainData)
ticketExtraction(testData)

print('Non unique tickets')
print(non_unique_tickets.index)


plt.show()
plt.figure(figsize=(40,10))
plt.hist(trainData['ticketText'],bins=len(trainData['ticketText'].unique()),align='mid',log=True)
plt.show()

valueCountTicketRed = trainData['ticketText'].value_counts()[trainData['ticketText'].value_counts()>10]
print(valueCountTicketRed[:10])
print('Number value count red', len(valueCountTicketRed))


# ## Age extraction

# In[ ]:


#Look the influence of the age with the probability of surviving
fig, ax = plt.subplots()
print(trainData['Age'][trainData['Survived'] == 1].dropna())
a_heights, a_bins = np.histogram(trainData['Age'][trainData['Survived'] == 1].dropna(),bins=20)
b_heights, b_bins = np.histogram(trainData['Age'][trainData['Survived'] == 0].dropna(), bins=a_bins)
width = (a_bins[1] - a_bins[0])/3

ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue',label='survived')
ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen',label='dead')
plt.legend()
plt.show()


# In[ ]:


trainData['child'] = np.where(trainData['Age'] > 18,0,1)
testData['child'] = np.where(testData['Age'] > 18,0,1)
trainData['youngAdult'] = np.where(np.logical_and(trainData['Age'] > 18, trainData['Age'] < 25),1,0)
testData['youngAdult'] = np.where(np.logical_and(testData['Age'] > 18, testData['Age'] < 25),1,0)

sv_ya = trainData['Survived'][trainData['youngAdult'] == 1]
sv_nya = trainData['Survived'][trainData['youngAdult'] == 0]
print("Ratio young_men survived: ", np.sum(sv_ya)/len(sv_ya))
print("Ratio young_men dead: ", np.sum(sv_nya)/len(sv_nya))


# ## Fare analysis 

# We observe that when the fair is null they have all embarked at S and are all male their survival rate is also very low

# In[ ]:


print(trainData['Fare'].describe())
print("Fraction of nan: ", trainData['Fare'].isnull().sum()/len(trainData))
trainData[trainData['Fare']==0]


# In[ ]:


testData[testData['Fare']==0]


# In[ ]:


#Look the influence of the age with the probability of surviving
fig, ax = plt.subplots()
fig.set_figheight(2)
fig.set_figwidth(20)

a_heights, a_bins = np.histogram(trainData['Fare'][trainData['Survived'] == 1].dropna(),bins=100)
b_heights, b_bins = np.histogram(trainData['Fare'][trainData['Survived'] == 0].dropna(), bins=a_bins)
width = (a_bins[1] - a_bins[0])/3

ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue',label='survived')
ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen',label='dead')
plt.yscale('log')
plt.legend()
plt.show()


# In[ ]:


trainData['zeroFare'] = np.where(trainData['Fare']==0,1,0)
testData['zeroFare'] = np.where(testData['Fare']==0,1,0)


# # One-hot encoding

# In[ ]:


def one_hot_encoding(data,valueCountRed,valueCountTicketRed,ret):
    
    #FamilyNames
    dataDumFamily = pd.DataFrame()
    for name in valueCountRed.index:
        if(len(dataDumFamily) == 0):
            dataDumFamily = pd.DataFrame(np.where(data['familyName'] == name,1,0))
        else:
            dataDumFamily['Family_' + str(name)] = np.where(data['familyName'] == name,1,0)

    #Ticket
    dataDumTicket = pd.DataFrame()
    for name in valueCountTicketRed.index:
        if(len(dataDumTicket) == 0):
            dataDumTicket = pd.DataFrame(np.where(data['ticketText'] == name,1,0))
        else:
            dataDumTicket['Ticket_' + str(name)] = np.where(data['ticketText'] == name,1,0)

    #Embarked
    dataDum = pd.concat([dataDumFamily, pd.get_dummies(data.Embarked, prefix='Embarked')],axis=1)
    dataDum = pd.concat([dataDum, dataDumTicket],axis=1)
    #Cabin Letter
    dataDum = pd.concat([dataDum, pd.get_dummies(data.CabinLetter, prefix='CabinLetter')],axis=1)
    if(ret):
        #There is one feature that we need to add: CabinLetter_T
        dataDum['CabinLetter_T'] = 0
    #Pclass
    dataDum = pd.concat([dataDum, pd.get_dummies(data.Pclass, prefix='Pclass')],axis=1)
    dataDum['sex'] = np.where(data['Sex'] == 'male',0,1)
    dataDum = dataDum.drop([0],axis=1)
    return dataDum
    
trainDataDum = one_hot_encoding(trainData,valueCountRed,valueCountTicketRed,False)
testDataDum = one_hot_encoding(testData,valueCountRed,valueCountTicketRed,True)


# ## Merge features

# In[ ]:


def mergeFeatures(data,dataDum,ret,mode):
    X = data
    if(not ret):
        y = data['Survived']
        dropList = ['Cabin','PassengerId','Embarked','Survived','Name','Sex','familyName','CabinLetter','Ticket','Pclass', 'ticketText']
    else:
        y = None
        dropList = ['Cabin','PassengerId','Embarked','Name','Sex','familyName','CabinLetter','Ticket','Pclass', 'ticketText']
        
    if(mode=='dropNumber'):
        dropList.append("ticketNumber")
        dropList.append("CabinNumber")
        
    X = X.drop(dropList,axis=1)
    X = pd.concat([X,dataDum],axis=1)
    return X,y

X,y = mergeFeatures(trainData,trainDataDum,False,mode='dropNumber')
X_ret,_ = mergeFeatures(testData, testDataDum,True,mode='dropNumber')


# # Understand the dataset

# In[ ]:


#print(X.info())
#print(X_ret.info())


# In[ ]:


#print(X.describe().T)
#print(X_ret.describe().T)


# In[ ]:


pd.DataFrame.corrwith(X,y).sort_values()


# # Treat NAN values

# In[ ]:


def percent_missing_fun(data):
    percent_missing = data.isnull().sum() * 100 / len(data)
    return percent_missing[percent_missing>0].sort_values(ascending=False)
    
print("TRAIN X")
print(percent_missing_fun(X))
print("TEST X")
print(percent_missing_fun(X_ret))


# # Machine learning

# In[ ]:


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)   
def ACC_model(dtrain,y_train,dtest,y_test,model):
    
    y_pred = np.where(model.predict(dtest)>0.5,1,0)
    accuracyTest = accuracy_score(y_test, y_pred)
    y_pred = np.where(model.predict(dtrain)>0.5,1,0)
    accuracyTrain = accuracy_score(y_train, y_pred)
    return accuracyTest,accuracyTrain

def printACC(ACCTestList,ACCTrainList):
    print('---------------------------')
    print('Test set')
    print('Mean')
    print(np.mean(ACCTestList))
    print('Std')
    print(np.std(ACCTestList))
    print('Train set')
    print('Mean')
    print(np.mean(ACCTrainList))
    print('Std')
    print(np.std(ACCTrainList))
    print('--------------------------')
    
def printACCRed(ACCTestList,ACCTrainList):
     print(str(np.mean(ACCTestList))+ " "+ str(np.mean(ACCTrainList)))


# In[ ]:


def apply_cv(model,k=10):

    kf = KFold(n_splits=k)
    kf.get_n_splits(X)
    print(kf)
    ACCTestList = []
    ACCTrainList = []
    KFold(n_splits=2, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        ACCTest,ACCTrain = ACC_model(X_train,y_train,X_test,y_test, model)
        ACCTestList.append(ACCTest)
        ACCTrainList.append(ACCTrain)

    print('ACCTestList:')
    print(ACCTestList)
    print('ACCTrainList:')
    print(ACCTrainList)
    printACC(ACCTestList,ACCTrainList)
    
model = XGBClassifier()
apply_cv(model)


# In[ ]:


def score(params):
    """
    source: https://www.kaggle.com/code/yassinealouini/hyperopt-the-xgboost-model/script
    """
    ACCTestList = []
    ACCTrainList = []
    kf = KFold(n_splits=2, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dtrain = xgb.DMatrix(X_train, y_train)
        dtest = xgb.DMatrix(X_test, y_test)
        num_round = int(params['n_estimators'])
        evals = [(dtest, 'eval'), (dtrain, 'train')]
        params2 = params.copy()
        params2.pop('n_estimators', None)
        model = xgb.train(params2, dtrain, num_round,
                          evals=evals,
                          verbose_eval=True)
        ACCTest,ACCTrain = ACC_model(dtrain,y_train,dtest,y_test, model)
        ACCTestList.append(ACCTest)
        ACCTrainList.append(ACCTrain)

    printACCRed(ACCTestList,ACCTrainList)
    loss = 1-np.mean(ACCTestList)
    return {'loss': loss, 'status': STATUS_OK}


# def optimize(random_state=SEED):
#     """
#     This is the optimization function that given a space (space here) of 
#     hyperparameters and a scoring function (score here), finds the best hyperparameters.
#     source: https://www.kaggle.com/code/yassinealouini/hyperopt-the-xgboost-model/script
#     """
#     # To learn more about XGBoost parameters, head to this page: 
#     # https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
#     space = {
#         'n_estimators': hp.quniform('n_estimators', 1, 300, 1),
#         'eta': hp.quniform('eta', 0.025, 0.5, 0.025),
#         # A problem with max_depth casted to float instead of int with
#         # the hp.quniform method.
#         'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
#         'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
#         'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
#         'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
#         'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
#         'eval_metric': 'auc',
#         'objective': 'binary:logistic',
#         'seed': random_state
#     }
#     # Use the fmin function from Hyperopt to find the best hyperparameters
#     best = fmin(score, space, algo=tpe.suggest, 
#                 max_evals=2)
#     return best
# 
# optimize()

# Best hyperparams:
#  {'colsample_bytree': 0.75,
#  'eta': 0.125,
#  'gamma': 0.7000000000000001,
#  'max_depth': 5,
#  'min_child_weight': 2.0,
#  'n_estimators': 67.0,
#  'subsample': 0.8500000000000001}
#  0.15484394506866417

# In[ ]:


param = {'colsample_bytree': 0.55,
 'eta': 0.325,
 'gamma': 0.8500000000000001,
 'max_depth': 2,
 'min_child_weight': 4.0,
 'n_estimators': 2.0,
 'subsample': 0.8500000000000001}

model = XGBClassifier(n_estimators=int(param['n_estimators']),eta=param['eta'],gamma=param['gamma'],max_depth=param['max_depth'],
                      min_child_weight=param['min_child_weight'], subsample=param['subsample'],colsample_bytree=param['colsample_bytree'],eval_metric='auc',
                      objective= 'binary:logistic',seed=SEED)
apply_cv(model)


# In[ ]:


model = XGBClassifier(n_estimators=int(param['n_estimators']),eta=param['eta'],gamma=param['gamma'],max_depth=param['max_depth'],
                      min_child_weight=param['min_child_weight'], subsample=param['subsample'],colsample_bytree=param['colsample_bytree'],eval_metric='auc',
                      objective= 'binary:logistic',seed=SEED)

model.fit(X,y)
xgb.plot_importance(model)
predictions = model.predict(X_ret)
output = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")


# #Results
# - default : 0.75199
# - hyperparameters_tuning : 0.76555
# - drop ticket number : 0.7727
# - keep 4 best features : 0.74401
# - add zeroFare : 0.76555
# - add youngAdult + uniqueTicket: 0.75358
# - add hyperparameters selection: 0.7799
# - remove_numbers : 0.75837

# In[ ]:




