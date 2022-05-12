#!/usr/bin/env python
# coding: utf-8

# ## Introduction to Neural Networks - Fraud Detection
# 
# Nilson reports that U.S. card fraud (credit, debt, etc) was reportedly 9 billion in 2016 and expected to increase to 12 billion by 2020. For perspective, in 2017 both PayPal's and Mastercard's revenue was only $10.8 billion each.
# 
# 
# **Objective:** In this notebook, given the credit card transactions, we will build a simple neural network (i.e., Multilayer perceptrons) for Fraud Detection using Keras.

# This notebooks covers,
# 
# 1. Creating a Model
# 
# 2. Adding Layers
# 
# 3. Activations
# 
# 4. Optimizers and Loss functions
# 
# 5. Evaluation

# ### Dataset Description
# 
# The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, the original features and more background information about the data is not provided. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# 
# Source: https://www.kaggle.com/mlg-ulb/creditcardfraud

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, auc,classification_report
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers, metrics
import seaborn as sns
import tensorflow
from tensorflow.keras.metrics import Recall
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


dataset_file = '../input/credit-card-fraud-detection/creditcard.csv'


# In[ ]:


data = pd.read_csv(dataset_file)


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data = data.drop("Time", axis = 1)


# In[ ]:


data.info()


# In[ ]:


data = data.dropna()#not needed


# In[ ]:


data['Class'].value_counts()


# In[ ]:


# The classes are heavily skewed we need to solve this issue later.
print('No Frauds', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% of the dataset')
print('Frauds', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% of the dataset')


# Recall =  False Negative
# precision = False positive
# 
# * Recall, I have to minimize any fraudelent transaction being classified as genuine
# * Precision, I haver to minimize any genuine transaction being classfied as Fraud

# ```
# A = Ignore
# F1 = Balanced
# P =  TP/(TP+FP)
# R = TP(/TP+FN)
# ```

# ```
# TN =
# TP =
# FP = The actual value is that a transaction is genuine but model has predicted it to be Fraud Transaction
# FN = The actual value is that a transaction is Fraud but model has predicted it to be a Genuine Transaction
# ```

# In[ ]:


sns.countplot('Class', data=data)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)


# In[ ]:


X_data = data.iloc[:, :-1]# Predictors in x
y_data = data.iloc[:, -1]#Target columns


# In[ ]:


print(X_data.shape,y_data.shape)


# In[ ]:


X_data.head()


# In[ ]:


y_data.head()


# #### Split the data into train and test

# In[ ]:


def data_prepration(x,test_size = 0.2):
    """Splits the data into train and test set.
    Keyword arguments:
    x -- Dataframe 
    test_size -- Percentage to split(default 0.2)
    """
    x['Amount']  =x['Amount'].apply(lambda x :  np.log1p(x))# comment if you need to use log transform
    x_features= x.iloc[:,x.columns != "Class"]
    x_labels=x.iloc[:,x.columns=="Class"]
    x_features_train,x_features_test,x_labels_train,x_labels_test = train_test_split(x_features,x_labels,test_size=test_size,random_state=7)
    print("length of training data")
    print(len(x_features_train))
    print("length of test data")
    print(len(x_features_test))
    return(x_features_train,x_features_test,x_labels_train,x_labels_test)


# In[ ]:


X_train, X_test, y_train, y_test=data_prepration(data,test_size = 0.2)


# In[ ]:


X_train.describe()


# In[ ]:


#normalize and conver to array
sc = StandardScaler()
# Let us first do our amount normalised and other that we are doing above
X_train["Normalized Amount"] = sc.fit_transform(X_train['Amount'].values.reshape(-1, 1))
X_train.drop(["Amount"],axis=1,inplace=True)
X_test["Normalized Amount"] = sc.transform(X_test['Amount'].values.reshape(-1, 1))
X_test.drop(["Amount"],axis=1,inplace=True)


# In[ ]:


X_train.head()


# ### 1. Creating a model
# 
# Keras model object can be created with Sequential class
# 
# At the outset, the model is empty per se. It is completed by adding additional layers and compilation
# 
# Ref: https://keras.io/models/sequential/
# 

# In[ ]:


model = Sequential()


# ### 2. Adding layers [layers and activations]
# 
# Keras layers can be added to the model
# 
# Adding layers are like stacking lego blocks one by one
# 
# It should be noted that as this is a classification problem, sigmoid layer (softmax for multi-class problems) should be added
# 
# Ref: https://keras.io/layers/core/

# ```
# INPUT(29) -> HIDDEN(64) -> OUTPUT
# ```

# In[ ]:


model.add(Dense(64, input_shape = (29,), activation = 'tanh'))
model.add(Dense(1, activation = 'sigmoid'))


# ### 3. Model compile [optimizers and loss functions]
# 
# Keras model should be "compiled" prior to training
# 
# Types of loss (function) and optimizer should be designated
# 
# Ref (optimizers): https://keras.io/optimizers/
# 
# Ref (losses): https://keras.io/losses/
# 

# In[ ]:


opt = optimizers.SGD(lr = 0.005)


# In[ ]:


model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics=[tensorflow.keras.metrics.Recall()])


# ### 4. Summary of the model

# In[ ]:


model.summary()


# In[ ]:


def model_1():
    model = Sequential()
    model.add(Dense(64, input_shape = (29,), activation = 'sigmoid'))
    model.add(Dense(1, activation = 'sigmoid'))
    opt = optimizers.SGD(lr = 0.005)
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics=[tensorflow.keras.metrics.Recall()])
    return model


# ### 5.Training [Forward pass and Backpropagation]
# 
# Training the model

# In[ ]:


model.fit(X_train, np.array(y_train), epochs = 1, verbose = 1,validation_data=(X_test,y_test))


# In[ ]:


# model = model_1()
# model.fit(X_train, np.array(y_train), epochs = 1, verbose = 1,validation_data=(X_test,y_test),batch_size=32)


# ### 6. Evaluation
# Keras model can be evaluated with evaluate() function
# 
# Evaluation results are contained in a list
# 
# Ref (metrics): https://keras.io/metrics/

# In[ ]:


#X_test = preprocessing.normalize(X_test)
results = model.evaluate(X_test, y_test)
print(model.metrics_names)
print(results) 


# In[ ]:


model.predict(X_test, verbose=0)>0.5


# ### Confusion Matrix

# In[ ]:


Y_pred_cls = model.predict(X_test) 
Y_pred_cls = Y_pred_cls>0.5
print('Accuracy Model: '+ str(accuracy_score(y_test,Y_pred_cls)))
print('Recall_score: ' + str(recall_score(y_test,Y_pred_cls)))
print('Precision_score: ' + str(precision_score(y_test, Y_pred_cls)))
print('F-score: ' + str(f1_score(y_test,Y_pred_cls)))
confusion_matrix(y_test, Y_pred_cls)


# Lets Try a more complicated layer
# ```
# INPUT->HIDDEN->HIDDEN->OUTPUT
# ```

# In[ ]:


model_new = Sequential()
model_new.add(Dense(64, input_shape = (29,), activation = 'relu'))
model_new.add(Dense(128, activation = 'relu'))
#model_new.add(Dense(64, activation = 'relu'))
model_new.add(Dense(1, activation = 'sigmoid'))
opt = optimizers.Adam(lr = 0.001)
model_new.compile(optimizer = opt, loss = 'binary_crossentropy', metrics=[tensorflow.keras.metrics.Recall()])


# In[ ]:


def model_2():
    model_new = Sequential()
    model_new.add(Dense(64, input_shape = (29,), activation = 'relu'))
    model_new.add(Dense(128, activation = 'relu'))
    model_new.add(Dense(1, activation = 'sigmoid'))
    opt = optimizers.Adam(lr = 0.001)
    model_new.compile(optimizer = opt, loss = 'binary_crossentropy', metrics=[tensorflow.keras.metrics.Recall()])
    return model_new


# In[ ]:


model_new.summary()


# In[ ]:


model_new.fit(X_train, np.array(y_train), epochs = 5, verbose = 1,validation_data=(X_test,y_test))


# In[ ]:


results = model_new.evaluate(X_test, y_test)
print(model_new.metrics_names)
print(results) 


# In[ ]:


#Y_pred_cls = model_new.predict_classes(X_test, batch_size=200, verbose=0)
Y_pred_cls = model_new.predict(X_test) 
Y_pred_cls = Y_pred_cls>0.5
print('Accuracy Model: '+ str(accuracy_score(y_test, Y_pred_cls)))
print('Recall_score: ' +  str(recall_score(y_test,Y_pred_cls)))
print('Precision_score: ' + str(precision_score(y_test, Y_pred_cls)))
print('F-score: ' + str(f1_score(y_test,Y_pred_cls)))
confusion_matrix(y_test, Y_pred_cls)


# #### Saving model and re train 
# Uncomment if needed

# In[ ]:


# model_new.save('model_new.h5') 
# model_new_loaded = tensorflow.keras.models.load_model('model_new.h5')

# model_new_loaded.evaluate(X_test, y_test)
# #model_new_loaded.fit(X_train, np.array(y_train), epochs = 5, verbose = 1,validation_data=(X_test,y_test))


# ### Random Forest

# In[ ]:


rfcl = RandomForestClassifier(n_estimators = 10, random_state=2)
rfcl.fit(X_train, y_train)


# In[ ]:


pred_RF = rfcl.predict(X_test)
print('Accuracy Model: '+ str(accuracy_score(y_test, pred_RF)))
print('Recall_score: ' + str(recall_score(y_test,pred_RF)))
print('Precision_score: ' + str(precision_score(y_test, pred_RF)))
print('F-score: ' + str(f1_score(y_test,pred_RF)))
confusion_matrix(y_test, pred_RF)


# In[ ]:


# !pip install -U imbalanced-learn


# In[ ]:


# pip install delayed


# #### SMOTE

# In[ ]:


from imblearn.over_sampling import SMOTE
data = pd.read_csv('../input/credit-card-fraud-detection/creditcard.csv')
data_train_X,data_test_X,data_train_y,data_test_y=data_prepration(data)
columns = data_train_X.columns


# In[ ]:


data_train_y['Class'].value_counts(normalize=True)


# In[ ]:


data_test_y['Class'].value_counts(normalize=True)


# In[ ]:


data_train_y.iloc[0:,0].value_counts()


# In[ ]:


# now use SMOTE to oversample our train data which have features data_train_X and labels in data_train_y
os = SMOTE(sampling_strategy ='minority', random_state=0)
os_data_X,os_data_y=os.fit_resample(data_train_X,data_train_y)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=["Class"])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of normal transcation in oversampled data",len(os_data_y[os_data_y["Class"]==0]))
print("No.of fraud transcation",len(os_data_y[os_data_y["Class"]==1]))

## New upsampled
print("Proportion of Normal data in oversampled data is ",len(os_data_y[os_data_y["Class"]==0])/len(os_data_X))
print("Proportion of fraud data in oversampled data is ",len(os_data_y[os_data_y["Class"]==1])/len(os_data_X))

#Old
print("Proportion of Normal data in normal data is ",len(data_train_y[data_train_y["Class"]==0])/len(data_train_y))
print("Proportion of fraud data in normal data is ",len(data_train_y[data_train_y["Class"]==1])/len(data_train_y))


# In[ ]:


# Let us first do our amount normalised and other that we are doing above
os_data_X["Normalized Amount"] = StandardScaler().fit_transform(os_data_X['Amount'].values.reshape(-1, 1))
os_data_X.drop(["Time","Amount"],axis=1,inplace=True)
data_test_X["Normalized Amount"] = StandardScaler().fit_transform(data_test_X['Amount'].values.reshape(-1, 1))
data_test_X.drop(["Time","Amount"],axis=1,inplace=True)


# In[ ]:


#os_data_X = preprocessing.normalize(os_data_X)
#data_test_X = preprocessing.normalize(data_test_X)


# In[ ]:


# def model_2():
#     model_new = Sequential()
#     model_new.add(Dense(8, input_shape = (29,), activation = 'relu'))
#     model_new.add(Dense(16, activation = 'relu'))
#     model_new.add(Dense(1, activation = 'sigmoid'))
#     opt = optimizers.Adam(lr = 0.001)
#     model_new.compile(optimizer = opt, loss = 'binary_crossentropy', metrics=[tensorflow.keras.metrics.Recall()])
#     return model_new


# In[ ]:


np.random.seed(7)
model_new = model_2()
model_new.summary()


# In[ ]:


model_new.fit(os_data_X,os_data_y, epochs = 1, verbose = 1,validation_data=(data_test_X,data_test_y))


# In[ ]:


Y_pred_cls = model_new.predict(data_test_X) 
Y_pred_cls = Y_pred_cls>0.5
print('Accuracy Model: '+ str(accuracy_score(data_test_y, Y_pred_cls)))
print('Recall_score: ' +  str(recall_score(data_test_y,Y_pred_cls)))
print('Precision_score: ' + str(precision_score(data_test_y, Y_pred_cls)))
print('F-score: ' + str(f1_score(data_test_y,Y_pred_cls)))
confusion_matrix(data_test_y, Y_pred_cls)


# In[ ]:





# #### Random Forest usign SMOTE

# In[ ]:


rfcl = RandomForestClassifier(n_estimators = 10, random_state=2)
rfcl.fit(os_data_X, os_data_y)
pred_RF = rfcl.predict(data_test_X)
print('Accuracy Model: '+ str(accuracy_score(data_test_y, pred_RF)))
print('Recall_score: ' + str(recall_score(data_test_y,pred_RF)))
print('Precision_score: ' + str(precision_score(data_test_y, pred_RF)))
print('F-score: ' + str(f1_score(data_test_y,pred_RF)))
confusion_matrix(data_test_y, pred_RF)


# #### Cross Validation

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold,KFold
df = pd.read_csv('../input/credit-card-fraud-detection/creditcard.csv')
data_train_X,data_test_X,data_train_y,data_test_y=data_prepration(df)
columns = data_train_X.columns

X = df.drop(['Class'], axis=1)
y = df['Class']
columns = X.columns


# In[ ]:


def model_3():
    model_new = Sequential()
    model_new.add(Dense(64, input_shape = (29,), activation = 'relu'))
    model_new.add(Dense(32, activation = 'relu'))
    model_new.add(Dense(16, activation = 'relu'))
    model_new.add(Dense(1, activation = 'sigmoid'))
    opt = optimizers.Adam(lr = 0.005)
    model_new.compile(optimizer = opt, loss = 'binary_crossentropy', metrics=[tensorflow.keras.metrics.Recall()])
    return model_new
np.random.seed(7)
model_new = model_3()
model_new.summary()


# model 1
# model 2
# model 3
# model 4
# model 5
# 
# pred1[0.4]
# pred2[0.41]
# pred3[0.45]
# pred4[0.2]
# pred5[0.3]
# 
# 
# mean = 0.4 .0.5 = 0

# In[ ]:


sss = KFold(n_splits=5, random_state=7, shuffle=True)
import numpy
pred = numpy.zeros(data_test_X.shape[0])
pred = pred.reshape(-1,1)
data_test_X["Normalized Amount"] = StandardScaler().fit_transform(data_test_X['Amount'].values.reshape(-1, 1))
data_test_X.drop(["Time","Amount"],axis=1,inplace=True)
Fold= 1
for train_index, test_index in sss.split(data_train_X, data_train_y):
    print("Fold :",Fold)
    #print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = data_train_X.iloc[train_index], data_train_X.iloc[test_index]
    original_ytrain, original_ytest = data_train_y.iloc[train_index], data_train_y.iloc[test_index]

    # Turn into an array
    original_Xtrain = original_Xtrain.values
    original_Xtest = original_Xtest.values
    original_ytrain = original_ytrain.values
    original_ytest = original_ytest.values
    
    # now use SMOTE to oversample our train data which have features data_train_X and labels in data_train_y
    os = SMOTE(sampling_strategy ='minority', random_state=0)
    os_data_X,os_data_y= os.fit_resample(original_Xtrain, original_ytrain)
    os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
    os_data_y= pd.DataFrame(data=os_data_y,columns=["Class"])
    original_Xtest = pd.DataFrame(data=original_Xtest,columns=columns )
    original_ytest= pd.DataFrame(data=original_ytest,columns=["Class"])
 
    # Let us first do our amount normalised and other that we are doing above
    os_data_X["Normalized Amount"] = StandardScaler().fit_transform(os_data_X['Amount'].values.reshape(-1, 1))
    os_data_X.drop(["Time","Amount"],axis=1,inplace=True)
    
    original_Xtest["Normalized Amount"] = StandardScaler().fit_transform(original_Xtest['Amount'].values.reshape(-1, 1))
    original_Xtest.drop(["Time","Amount"],axis=1,inplace=True)
    
    #Model Train
    model_new.fit(os_data_X,os_data_y, epochs = 2, verbose = 1,validation_data=(original_Xtest,original_ytest),batch_size=512)
    Y_pred_cls = model_new.predict(original_Xtest, verbose=0)
    Y_pred_cls = Y_pred_cls>0.5
    print("-----------------------START-------------------------------------")
    print('Recall_score: ' +  str(recall_score(original_ytest,Y_pred_cls)))
    print('F-score: ' + str(f1_score(original_ytest,Y_pred_cls)))
    confusion_matrix(original_ytest, Y_pred_cls)
    Fold = Fold + 1
    print("-----------------------END---------------------------------------")

    pred+=model_new.predict(data_test_X)
    print("\n")
pred_new = pred/5


# In[ ]:


pred_val = pred_new>0.5


# In[ ]:


print('Cross Validation Recall_score: ' +  str(recall_score(data_test_y,pred_val)))
print('Cross Validation F-score: ' + str(f1_score(data_test_y,pred_val)))
print('Cross Validation Precision-score: ' + str(precision_score(data_test_y,pred_val)))
print(classification_report(data_test_y,pred_val))
confusion_matrix(data_test_y,pred_val)


# #### CV on RF

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold,KFold
df = pd.read_csv('../input/credit-card-fraud-detection/creditcard.csv')
data_train_X,data_test_X,data_train_y,data_test_y=data_prepration(df)
columns = data_train_X.columns

X = df.drop(['Class'], axis=1)
y = df['Class']
columns = X.columns


# In[ ]:


sss = KFold(n_splits=5, random_state=None, shuffle=False)
import numpy
pred = numpy.zeros(data_test_X.shape[0])
pred = pred.reshape(-1,1)
data_test_X["Normalized Amount"] = StandardScaler().fit_transform(data_test_X['Amount'].values.reshape(-1, 1))
data_test_X.drop(["Time","Amount"],axis=1,inplace=True)
Fold = 1
for train_index, test_index in sss.split(data_train_X, data_train_y):
    print("Fold :",Fold)
    #print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = data_train_X.iloc[train_index], data_train_X.iloc[test_index]
    original_ytrain, original_ytest = data_train_y.iloc[train_index], data_train_y.iloc[test_index]
    
    # Turn into an array
    original_Xtrain = original_Xtrain.values
    original_Xtest = original_Xtest.values
    original_ytrain = original_ytrain.values
    original_ytest = original_ytest.values
    
    # now use SMOTE to oversample our train data which have features data_train_X and labels in data_train_y
    os = SMOTE(sampling_strategy ='minority', random_state=0)
    os_data_X,os_data_y= os.fit_resample(original_Xtrain, original_ytrain)
    os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
    os_data_y= pd.DataFrame(data=os_data_y,columns=["Class"])
    original_Xtest = pd.DataFrame(data=original_Xtest,columns=columns )
    original_ytest= pd.DataFrame(data=original_ytest,columns=["Class"])
    # we can Check the numbers of our data

    # Let us first do our amount normalised and other that we are doing above
    os_data_X["Normalized Amount"] = StandardScaler().fit_transform(os_data_X['Amount'].values.reshape(-1, 1))
    os_data_X.drop(["Time","Amount"],axis=1,inplace=True)
    
    original_Xtest["Normalized Amount"] = StandardScaler().fit_transform(original_Xtest['Amount'].values.reshape(-1, 1))
    original_Xtest.drop(["Time","Amount"],axis=1,inplace=True)

    
    
    rfcl = RandomForestClassifier(n_estimators = 10, random_state=2)
    rfcl.fit(os_data_X, os_data_y)
    pred_RF = rfcl.predict(original_Xtest)
    print("-----------------------START-------------------------------------")
    print('Recall_score: ' + str(recall_score(original_ytest,pred_RF)))
    print('F-score: ' + str(f1_score(original_ytest,pred_RF)))
    confusion_matrix(original_ytest, pred_RF)
    Fold = Fold + 1
    print("-----------------------END---------------------------------------")

    pred+=pd.DataFrame(rfcl.predict_proba(data_test_X)).iloc[:,1].values.reshape(-1,1)
    print("\n")
pred_new = pred/5


# In[ ]:


pred_val = pred_new>=0.5


# In[ ]:


print('Cross Validation Recall_score: ' +  str(recall_score(data_test_y,pred_val)))
print('Cross Validation F-score: ' + str(f1_score(data_test_y,pred_val)))
print('Cross Validation Precision-score: ' + str(precision_score(data_test_y,pred_val)))
print(classification_report(data_test_y,pred_val))
confusion_matrix(data_test_y,pred_val)


# #### End of Notebook
