#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression 

# ## Table of content 

# - [Task 1 - Preprocessing of data](#t1)
# - [Task 2 - Feature normalization](#t2)
# - [Task 3 - Logistic regression equation](#t3)
# - [Task 4 - Split dataset into train and test](#t4)
# - [Task 5 - SGDClassifier](#t5)
# - [Task 6 - Classification report for predictions from task 5](#t6)
# - [Task 7 & 8 - Regularization & classification reports](#t7_8)
# - [Task 9 - K-Nearest Neighbours (KNN)](#t9)
# - [References](#references)

# ## <a id="t1"> Task 1</a>

# In[ ]:


# Import required libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import plotly.express as px 
import seaborn as sns


# In[ ]:


# Ignore warnings 
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=ConvergenceWarning)


# In[ ]:


# Create header based on details provided by UCI website
header = ['variance', 'skewness', 'kurtosis', 'entropy', 'class']
# Load data from txt file (comma seperated) with header
df = pd.read_csv('../input/bank-note-authentication-uci-data/BankNote_Authentication.csv')
df.head()


# In[ ]:


# Last 5 rows of data set
df.tail()


# In[ ]:


# Shape of the data set
df.shape


# In[ ]:


# Information of data set
df.info()


# In[ ]:


# Statistical summary of the data set 
df.describe()


# In[ ]:


# Class count for unforged and forged bank notes.
df['class'].value_counts()


# Based on the information above, we can see that there is no missing data. The shape of the data is 1372 and it matches the non-null values of 1372 for all columns. As such, data imputation for missing values need not be conducted. 
# 
# All input features (i.e., variance, skewness, kurtosis, entropy) are numerical. Thus, encoding is not required to be performed for such case. Class is the target (or dependent) variable that is classified as 0 and 1 if bank note is not forged and forged respectively. 

# In[ ]:


# Distributions of numerical inputs
cols = ['variance', 'skewness', 'curtosis', 'entropy']

for col in cols: 
    fig = px.histogram(df, x=col, color='class', marginal='box', opacity=0.7, 
                        title=f"Distribution of input feature {col}")
    fig.show()


# With reference to the distribution plots above, we note the following observations. 
# 
# 1. There were a number of outliers for the following features based on the marginal box plot for variance, kurtosis, and entropy. However, they may contribute in patterns to identify if a bank note is forged. Hence, outliers need not be remove. 
# 2. We note that the variance and skewness of the forged notes have a rather different distribution as compared to the unforged notes in the first two histograms. 
# 3. The forged notes have the long right tail for kurtosis as shown in the third histogram.

# ## <a id='t2'> Task 2 - Feature Normalization </a>    

# In[ ]:


# Normalise the numercial input features using Sklearn MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
df_norm = scaler.fit_transform(df)
df_norm = pd.DataFrame(df_norm, columns=header)
df_norm.describe()


# In[ ]:


# Separate input features into variable X and target output to variable y
X = df_norm.drop(columns=['class'])
y = df_norm['class']
print(X.shape)
print(y.shape)


# As shown in the statistical summmary in task 1, the input variables are of different scale. Hence, normalisation will be done before model training to ensure equal importance/contribution of the features to the model. It may also help to reduces the training time depending on the algorithm used to train the model. 

# ## <a id='t3'>Task 3 - Logistic Regression Equation</a>

# __Logistic Regression__ is a statistical method for predicting binary (or two) classes. The target variable is dichotomous in nature. In this case, there are only two possible classes: whether the bank note is forged or not. 
# 
# The logistic regression graph is in the form of a sigmoid function such that the prediction above 0.5 (by default) would be classified as forged. Those below 0.5 would be classified as unforged. The threshold probability of 0.5 may be changed depending on the problem statement. 
# 
# __Logistic Regression Equation for Bank Notes Classification Problem (4 input features):__
# $ P(forged bank note) = P(x)= \Large\frac{1}{1 + e^{- (\beta_{0} + \beta_{1}x_{1} + \beta_{2}x_{2} + \beta_{3}x_{3} + \beta_{4}x_{4})}}$ 
# 
# where
# - $\beta_{0}$ - bias term, and
# - $\beta_{1}, \beta_{2}, \beta_{3}, \beta_{4}$ - weights for the input features
# 
# <br>
# The parameters that needs to be estimated are $\beta_{0}, \beta_{1}, \beta_{2}, \beta_{3}, \beta_{4}$ such that it  maximises the log-likelihood function as follows.
# $ l = \frac{1}{4} \sum \limits _{k=1} ^{4}y_{k} \log {P(x_{k})} + \sum \limits _{k=1} ^{4}(1-y_{k}) \log{(1-P(x_{k})}) $
# 
# where 
# - M = 4 as there are 4 inputs features,  
# - $y_{k}$ - categorical outcome of the prediction of k-th observation, and 
# - $x_{k}$ - input features / explanatory variables of k-th observation.

# ## <a id="t4"> Task 4 - Split dataset into train and test </a>

# In[ ]:


# Split dataset into train and test using Sklearn train_test_split
# As the API includes a shuffle attribute with a default value of True, additional manual shuffling of the 
# data is deemed unrequired.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)


# ## <a id='t5'>Task 5 - SGDClassifier</a>

# In[ ]:


# import required library
from sklearn.utils.fixes import loguniform
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# model and parameters
model = SGDClassifier(loss='log', learning_rate='optimal', eta0=0.00001, random_state=42)
space = dict()
space['alpha'] = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
space['max_iter'] = [500, 1000, 2500, 5000, 10000]
space['tol'] = [0.00001, 0.00005, 0.0005, 0.001, 0.005, 0.01]

# GridSearchCV for best params
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
search = GridSearchCV(model, space, scoring='accuracy', n_jobs=-1, cv=cv)
result = search.fit(X, y)


# In[ ]:


# Print best score and params for GridSearchCV
print(f'Best score: {result.best_score_}')
print(f'Best Hyperparameters: {result.best_params_}')


# #### Initial Grid Search Hyperparameter Tuning Findings
# 
# As shown above, we initial the random values in the hyperparameters *alpha*, *max_iter*, *tol*, *learning_rate* that we want to tuned in a dictionary and execute the GridSearchCV algorithm from Scikit Learn library. The difference between the initialised random values were intentionally set larger as further tuning of the values can be done after the initial best parameters are found. 
# 
# The inital best hyperparameters are summarised as follows. 
# * alpha: 5e-05
# * learning_rate = 'optimal'
# * max_iter = 500 
# * tol = 0.0005
# 
# Further tuning is conducted below with a lower range of values for each hyperparameter to check and verify if any further improvement is possible.

# In[ ]:


# Further tuning of GridSearchCV parameters based on initial findings
# model and parameters
model3 = SGDClassifier(loss='log', random_state=42)
space3 = dict()
space3['alpha'] = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007]
space3['max_iter'] = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
space3['tol'] = [0.00025, 0.0005, 0.00075, 0.001, 0.00125, 0.0025, 0.00275, 0.003]
space3['learning_rate'] = ['optimal']

# GridSearchCV for best params
cv3 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
search3 = GridSearchCV(model3, space3, scoring='accuracy', n_jobs=-1, cv=cv)
result3 = search3.fit(X, y)


# In[ ]:


# Print best score and params for RandomSearchCV
print(f'Best score: {result3.best_score_}')
print(f'Best Hyperparameters: {result3.best_params_}')


# #### Finalised Grid Search Hyperparameters 
# 
# The finalised hyperparameters used to train the model are as follows. 
# 
# * __alpha__ (val: 2e-05): Constant which multiplies with the regularization term and used to compute the learning_rate when learning_rate is set to optimal. The higher the value, the stronger the regularization.
# * __learning_rate__ (val: 'optimal'): Value that determine step the algorithm takes to optimise the loss function. Learning rate schedule includes 4 options: constant, optimal, invscaling and adpative. 
# * __max_iter__ (val: 50): Maximum number of passes over the training data. 
# * __tol__ (val: 0.00275): Stopping criterion for the training. If none is set, then the training will stop when loss > best_loss - tol. 

# In[ ]:


# Train model using parameters found by GridSearchCV 

sgdcls = SGDClassifier(loss='log', random_state=42, alpha=2e-05, 
                       learning_rate='optimal', max_iter=50, tol=0.00275)

sgdcls.fit(X_train, y_train)
y_pred = sgdcls.predict(X_test)


# Using the finalised hyperparameters as summarised above, we will build and train the model to predict if the bank notes is forged or not. Refer to task 6 for the classification report and confusion matrix. 

# #### Extra Random Search Hyperparameter Tuning
# 
# As the values are very close, we will use the parameters found using GridSearchCV in this practicum.

# In[ ]:


# model and parameters
from scipy.stats import randint as sp_randint
from scipy.stats import expon as sp_expon

model2 = SGDClassifier(loss='log', eta0=0.00001, random_state=42)
space2 = dict()
space2['alpha'] = loguniform(1e-6, 1e1)
space2['max_iter'] = sp_randint(100,2000)
space2['tol'] = loguniform(1e-5, 1e-1)
space2['learning_rate'] = ['optimal']

# RandomSearchCV for best params
cv2 = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
search2 = RandomizedSearchCV(model2, space2, n_iter=2000, scoring='accuracy', n_jobs=-1, cv=cv, random_state=42)
result2 = search2.fit(X, y)


# In[ ]:


# Print best score and params for RandomSearchCV
print(f'Best score: {result2.best_score_}')
print(f'Best Hyperparameters: {result2.best_params_}')


# ## <a id='t6'>Task 6: Classification report for above predictions</a> 

# In[ ]:


# import relevant metrics and print classification report
from sklearn.metrics import plot_confusion_matrix, classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


# confusion matrix
plot_confusion_matrix(sgdcls, X_test, y_test)
plt.show()


# The trained model has a high F1 score of 0.99. There were only 6 misclassified bank notes which were not forged but highlighted as forged by the model. As the impact of false negative (i.e., loss of profit/revenue by not identifying the forged bank notes) is greater than false positive, the impact of the misclassification above is deemed to be acceptable in this case. In the next task, we will look into the different regularization techniques. 

# ## <a id='t7_8'>Task 7 & 8 - Regularization and classification reports</a>

# #### L1-norm: Lasso Regression

# In[ ]:


# SGDclassifier with L1-norm (Lasso Regression) and classification report
sgdcls_l1 = SGDClassifier(loss='log', random_state=42, alpha=2e-05, 
                       learning_rate='optimal', max_iter=50, tol=0.00275, penalty='l1')

sgdcls_l1.fit(X_train, y_train)
y_pred_l1 = sgdcls_l1.predict(X_test)
print(classification_report(y_test, y_pred_l1))


# In[ ]:


# Confusion matrix of SGDclassifier with l1-norm regularization
plot_confusion_matrix(sgdcls_l1, X_test, y_test)
plt.show()


# In the above model, we add in the __penalty__ parameter and set to $l_{1}$, which is the lasso regression, which uses the $l_{1}norm$. 
# 
# * Lasso Regression cost function: $J(\theta) = MSE(\theta) + \alpha \sum \limits_{i=1}^{n} \left\lvert{\theta_{i}}\right\rvert$
# 
# where 
# - $\alpha$ is the hyperparameter to control how much to regularize the model. For $\alpha$ = 0, original regression model is used. For large $\alpha$, the result is close to a flat line at the dataset's mean.
# 
# Lasso regression tends to eliminate the weight of insignicant or non-important features (i.e., setting the weight to zero). In other words, it performs feature selection automatically and return a sparse model with only the features deemed as important. 
# 
# #### Lasso Regression Findings 
# Based on the classification report and confusion matrix, we note an improvement in accuracy as the number of misclassification decrease from 6 to 4. There were 2 forged bank notes predicted as not forged, and 2 original bank notes predicted as forged. 
# 
# However, this may not be ideal for this particular problem as the false negative (type 2 error) has more impact (i.e., result in potential loss of profit and revenue) as the forged bank note was not highlighted. 

# #### L2-norm: Ridge Regression

# In[ ]:


# SGDclassifier with L2-norm (Ridge Regression) and classification report
sgdcls_l2 = SGDClassifier(loss='log', random_state=42, alpha=2e-05, 
                       learning_rate='optimal', max_iter=50, tol=0.00275, penalty='l2')

sgdcls_l2.fit(X_train, y_train)
y_pred_l2 = sgdcls_l2.predict(X_test)
print(classification_report(y_test, y_pred_l2))


# In[ ]:


plot_confusion_matrix(sgdcls_l2, X_test, y_test)
plt.show()


# For ridge regression, we add the **penalty** parameter and set it to $l_{2}$, which uses the $l_{2} norm$. 
# * Ridge regression cost function: $J(\theta) = MSE(\theta) + \alpha(\frac{1}{2} \sum \limits_{i=1}^{n} \theta_{i}^{2})$
# 
# where 
# - $\alpha$ is the hyperparameter to control how much to regularize the model. For $\alpha$ = 0, original regression model is used. For large $\alpha$, the result is close to a flat line at the dataset's mean.
# 
# For ridge regression, it restricts the weights of the features that are not important or insignificant (by assigning values that are close to 0). Hence, the features contribution to the output is less significant. However, the features would need remain to contribute to the output in the final model. In other word, feature selection is not automatically conducted in ridge regression. 
# 
# #### Ridge regression findings
# 
# Based on the classification report and confusion matrix, we note that there is no difference from the initial model. 
# 

# #### L1-norm & L2-norm: Elastic Net

# In[ ]:


# SGDclassifier with L1-norm and L2-norm (Elastic Net) and classification report
sgdcls_en = SGDClassifier(loss='log', random_state=42, alpha=2e-05, 
                       learning_rate='optimal', max_iter=50, tol=0.00275, penalty='elasticnet')

sgdcls_en.fit(X_train, y_train)
y_pred_en = sgdcls_en.predict(X_test)
print(classification_report(y_test, y_pred_en))


# In[ ]:


plot_confusion_matrix(sgdcls_en, X_test, y_test)
plt.show()


# For elastic net, we add the **penalty** parameter and set it to `elasticnet`, which uses both $l_{1} norm$ and $l_{2} norm$. 
# 
# * Elastic net cost function: $J(\theta) = MSE(\theta) + \alpha \sum \limits_{i=1}^{n} \left\lvert{\theta_{i}}\right\rvert + \alpha(\frac{1}{2} \sum \limits_{i=1}^{n} \theta_{i}^{2})$
# 
# Elastic net is the combination of lasso and ridge regression. 
# 
# #### Elastic net regularization findings
# 
# Based on the classification report and confusion matrix, we note that there is no difference from the initial model.

# ## <a id='t9'>Task 9: K-Nearest Neighbour (KNN)</a>

# In[ ]:


# K-nearest neighbours and classification report
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier()
neigh.fit(X_train, y_train)
y_pred_knn = neigh.predict(X_test)
print(classification_report(y_test, y_pred_knn))


# In[ ]:


plot_confusion_matrix(neigh, X_test, y_test)
plt.show()


# #### Comparison of accuracy of KNN and SGD models
# 
# As shown in the classification report and confusion matrix, KNN model has a 100% accuracy for the test/validation set whereas SGD model has an accuracy of 99%. This could mean the the KNN model fits the dataset better than the SGD model where it's possible to accurately predict whether the bank note is forged by analysis the mode of the 5 nearest neighbour. However, the accuracy for SGD model is also considered respectable at 99%. 
# 

# ## <a id="references">References</a>
# - [Wikipedia: Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)
# - [Machine Learning Mastery: Hyperparameter optimization](https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/)
# - [Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)
