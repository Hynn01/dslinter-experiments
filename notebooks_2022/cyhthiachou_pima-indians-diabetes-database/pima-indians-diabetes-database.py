#!/usr/bin/env python
# coding: utf-8

# # 1. Import the modules that will do all the work

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # 2. Import the data

# We load in a dataset [pima indians diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). This dataset will allow us to predict if a pima indian will get diabetes based on their pregnancies, glucose, and a variety of other metrics. We will start from loading the data into a data frame called **df**.

# In[ ]:


df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()


# # 3. Identify and deal with missing data

# Let's start from inspecting the data by looking at the info of the dataframe.

# In[ ]:


df.info()


# We see a bunch of the variables collected for each person in this dataset. These columns are:
# * **Pregnancies**: Number of times pregnant
# * **Glucose**: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# * **BloodPressure**: Diastolic blood pressure (mm Hg)
# * **SkinThickness**: Triceps skin fold thickness (mm)
# * **Insulin**: 2-Hour serum insulin (mu U/ml)
# * **BMI**: Body mass index (weight in kg/(height in m)^2)
# * **DiabetesPedigreeFunction**: Diabetes pedigree function
# * **Age**: Ages in years
# * **Outcomes**: Class variable (0 or 1) 268 of 768 are 1, the others are 0
# 
# Questions emerged from the following table: Could Glucose, BloodPressure, SkinThickness, Insulin, BMI be zero?

# In[ ]:


df.describe()


# Further understand the data distribution. We can notice Insulin, DiabetesPedigreeFunction, Insulin and Ages have a right-skewed distribution pattern.

# In[ ]:


def show_hist(df):
    return df.hist(figsize=(20,20))
show_hist(df)


# In order to identify and deal with missing data, we can start from split the data into training, validating, and testing groups. In that way, we could compare whether simply drop the missing data or impute the missing data.

# In[ ]:


X_full = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
y = X_full.Outcome
features = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
X=X_full[features].copy()


# In[ ]:


x_temp,x_test,y_temp,y_test = train_test_split(X, y, train_size = 0.8, test_size=0.2,random_state=0)
x_train,x_valid,y_train,y_valid = train_test_split(x_temp, y_temp, train_size = 0.75, test_size=0.25,random_state=0)


# After splitting the data, we can define the way to evaluate the outcomes.

# In[ ]:


# get evaluation of the results

def confusion_matrix(pred,y):
    #true positive, false positive, actual positive, true negative, false negative, actual negative
    TP,FP,AP,TN,FN,AN=0,0,0,0,0,0
    pred=list(pred)
    y=list(y)
    
    for i in range(len(pred)):
        if y[i]==1:
            AP+=1
            if pred[i] == 1:
                TP+=1
            elif pred[i] == 0: 
                FN+=1
        elif y[i]==0:
            AN+=1
            if pred[i] == 1:
                FP+=1
            elif pred[i] == 0: 
                TN+=1
    

    #Recall rate = (true positive value) / (actual positive value)
    recall_rate = TP/AP
    #Specificity rate = (true negative value) / (actual negative value)
    specificity_rate = TN/AN
    #Accuracy rate = (true positive value + true negative value) / (total number of samples)
    accuracy_rate = (TP+TN)/len(y)
    #Misclassification (error) rate = (false positive value + false negative value) / (total number of samples)
    misclassification_rate = (FP+FN)/len(y)
    #Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. 
    precision = TP/(TP+FP)
    #F1 score is to find the harmonic mean of recall and precision. F1 Score = 2*(Recall * Precision) / (Recall + Precision)
    f1_score = 2*(recall_rate*precision) / (recall_rate+precision)
    
    print("recall_rate is {}, specificity_rate is {}, accuracy_rate is {}, misclassification_rate is {}, f1_score is {}".format(round(recall_rate,2),round(specificity_rate,2),round(accuracy_rate,2),round(misclassification_rate,2), round(f1_score,2)))
    
    return TP,FP,TN,FN,AP,AN,recall_rate,specificity_rate,accuracy_rate,misclassification_rate,f1_score

# Define a unified function to calculate the errors
def score_dataset(x_train, y_train, x_valid, y_valid, model):
    # Good practice usage scaling techniques:
        # 1.Fit the scaler using available training data.
        # 2.Apply the scale to training data.
        # 3.Apply the scale to data going forward.
    scaler = StandardScaler()
    scaled_x_train = scaler.fit_transform(x_train)
    model.fit(scaled_x_train, y_train.values.ravel())
    pred = model.predict(scaler.fit_transform(x_valid))
    return confusion_matrix(pred,y_valid)


# Before cleaning the data, we can calculate the error rate to understand the baseline.

# In[ ]:


# A group of models to compare
model1=LogisticRegression(max_iter=1000)
model2=RandomForestClassifier(n_estimators=63, random_state=0)
model3=tree.DecisionTreeClassifier(max_depth = 5, random_state=0)
model4 = SVC(random_state=42, C=1000, gamma=0.001)
model_group = [model1,model2, model3, model4]


# In[ ]:


score_dataset(x_train, y_train, x_valid, y_valid, model1)

# Result:
# recall_rate is 0.53, specificity_rate is 0.87, accuracy_rate is 0.74, misclassification_rate is 0.26, f1_score is 0.6
# Original performance without cleaning the data


# Option 1. Clean the data - Drop the zero data

# In[ ]:


# From Hist observation, following features contains non-helpful zero data: Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age
features_with_zero = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI","Age"]

# Try 1st way of cleaning the zero data,
# Drop the zero data
dropped_x_train = x_train.drop(features_with_zero, axis=1)
dropped_x_valid = x_valid.drop(features_with_zero, axis=1)
#printing the score dataset for dropping the zeros
print(score_dataset(dropped_x_train, y_train, dropped_x_valid, y_valid, model1))

# Result:
# recall_rate is 0.23, specificity_rate is 0.92, accuracy_rate is 0.66, misclassification_rate is 0.34, f1_score is 0.33
# (13, 8, 89, 44, 57, 97, 0.22807017543859648, 0.9175257731958762, 0.6623376623376623, 0.33766233766233766)
# Drop the zero data actually makes the prediction less accurate.


# Option 2. Clean the data - Impute the data with Sklearn SimpleImputer

# In[ ]:


# Try 2nd way of cleaning the zero data,
# SKLearn model replace the zero data with actual values

replaced_x_train = x_train.copy(deep=True)
replaced_x_valid = x_valid.copy(deep=True)

replaced_x_train[features_with_zero] = x_train[features_with_zero].replace(0,np.NaN)
replaced_x_valid[features_with_zero] = x_valid[features_with_zero].replace(0,np.NaN)

from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_x_train = pd.DataFrame(my_imputer.fit_transform(replaced_x_train))
imputed_x_valid = pd.DataFrame(my_imputer.fit_transform(replaced_x_valid))

imputed_x_train.columns = x_train.columns
imputed_x_valid.columns = x_valid.columns

print(score_dataset(imputed_x_train, y_train, imputed_x_valid, y_valid, model1))

# Original performance: recall_rate is 0.53, specificity_rate is 0.87, accuracy_rate is 0.74, misclassification_rate is 0.26,, f1_score is 0.6
# Result:
# recall_rate is 0.53, specificity_rate is 0.86, accuracy_rate is 0.73, misclassification_rate is 0.27, f1_score is 0.59
# Using Simple Imputer makes it Less accurate when predicting negative results.


# Option 3. Clean the data - Manually replace the missing data

# In[ ]:


# Try 3rd way of cleaning the zero data,
# Manually replace the zero data with actual values 

replaced_x_train = x_train.copy(deep=True)
replaced_x_valid = x_valid.copy(deep=True)

def manual_replace(x):
    x["Glucose"].fillna(x["Glucose"].mean(), inplace = True)
    x['BloodPressure'].fillna(x['BloodPressure'].mean(), inplace = True)
    x['SkinThickness'].fillna(x['SkinThickness'].median(), inplace = True)
    x['Insulin'].fillna(x['Insulin'].median(), inplace = True)
    x['BMI'].fillna(x['BMI'].median(), inplace = True)
    return x

replaced_x_train = manual_replace(replaced_x_train)
replaced_x_valid = manual_replace(replaced_x_valid)

print(score_dataset(replaced_x_train, y_train, replaced_x_valid, y_valid, model1))

# Original performance: recall_rate is 0.53, specificity_rate is 0.87, accuracy_rate is 0.74, misclassification_rate is 0.26,, f1_score is 0.6
# Result:
# recall_rate is 0.53, specificity_rate is 0.87, accuracy_rate is 0.74, misclassification_rate is 0.26, f1_score is 0.6
# After replaceing the data with mean or median, it remains the same performance.


# In[ ]:


# After trying the three ways to clean the data, result remains the same performance. What's missing? Need more investigation.


# # 4. Choose and optimize the model

# When we were at the step of cleaning data, we have listed several models we could try for this data set.
# * Support Vector Machines
# * Linear Regression
# * Random Forest Classifier
# * Decision Tree Classifier

# **1.Support vector machines**

# In[ ]:


# Try Support Vector Machines
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#     scaler = StandardScaler()
#     scaled_x_train = scaler.fit_transform(x_train)
#     model.fit(scaled_x_train, y_train.values.ravel())

scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(x_train)
pca = PCA()
pca_x_train = pca.fit_transform(scaled_x_train)

per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = [str(x) for x in range(1,len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1),height=per_var)
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False)
plt.ylabel('Percentage of Expllained Variance')
plt.xlabel('Principle Components')
plt.title('Scree Plot')
plt.show()


# The scree plot shows that the first principle component PC1 and PC2 account for a relatively large amount of variation in the raw data, which means it would be a good candidate for x-axis in the 2-dimensional graph.
# 
# 

# In[ ]:


# Questions: 
# what is first principle component?
# what is explained variance?
# why then it would be good candidates?


# In[ ]:


train_pc1_coords = pca_x_train[:,0]
train_pc2_coords = pca_x_train[:,1]

pca_train_scaled = preprocessing.scale(np.column_stack((train_pc1_coords, train_pc2_coords)))

param_grid = [
    {'C':[1,10,100,1000],
     'gamma':['scale',1,0.1, 0.01, 0.001, 0.0001],
     'kernel':['rbf']
    },
]

optimal_params = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    verbose=0
)

optimal_params.fit(pca_train_scaled, y_train)
print(optimal_params.best_params_)


# In[ ]:


clf_svm = SVC(random_state=42, C=1000, gamma=0.001)
score_dataset(x_train, y_train, x_valid, y_valid, clf_svm)


# **3. Random Forest Classifier**
# To find the optimal value for n_estimators for RandomForestClassifier

# In[ ]:


#decide the estimator for the random forest treee
group_estimators = list(range(60,67,1))
results = {}

for estimator in group_estimators:
    model2=RandomForestClassifier(n_estimators=estimator, random_state=0)
    f1_score = score_dataset(x_train, y_train, x_valid, y_valid, model2)[10]
    results[estimator] = f1_score

get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(list(results.keys()), list(results.values()))
plt.show()
# optimized RandomForestClassifier n_estimators 63


# **4.Decision Tree Classifier** To find the optimal value for max_depth for DecisionTreeClassifier

# In[ ]:


leaf_dep_list = [x for x in range(20) if x >1]

results ={}
for leaf_dep in leaf_dep_list:
  model3=tree.DecisionTreeClassifier(max_depth = leaf_dep, random_state=0)
  f1_score = score_dataset(x_train, y_train,x_valid, y_valid, model3)[10]
  results[leaf_dep]=f1_score

get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(list(results.keys()), list(results.values()))
plt.show()
# optimal max_depth=5


# **Compare the performance among all models**

# In[ ]:


for i in model_group:
    print(i)
    print(score_dataset(x_train, y_train, x_valid, y_valid, i))


# After comparing, the winning model is:
# **RandomForestClassifier**(n_estimators=63, random_state=0)
# recall_rate is 0.63, 
# specificity_rate is 0.9, 
# accuracy_rate is 0.8, 
# misclassification_rate is 0.2, f1_score is 0.7

# # 4. Making predictions

# In[ ]:


final_model = RandomForestClassifier(n_estimators=63, random_state=0)
final_model.fit(x_temp, y_temp)
preds_test=final_model.predict(x_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': x_test.index,
                       'Outcome': preds_test})
output.to_csv('submission.csv', index=False)

