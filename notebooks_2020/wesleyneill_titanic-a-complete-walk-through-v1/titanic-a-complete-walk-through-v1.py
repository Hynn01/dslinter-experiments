#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# regular expressions
import re 

# math and data utilities
import numpy as np
import pandas as pd
import scipy.stats as ss
import itertools

# data and statistics libraries
import sklearn.preprocessing as pre
from sklearn import model_selection
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# visualization libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# Set-up default visualization parameters
mpl.rcParams['figure.figsize'] = [10,6]
viz_dict = {
    'axes.titlesize':18,
    'axes.labelsize':16,
}
sns.set_context("notebook", rc=viz_dict)
sns.set_style("whitegrid")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Initial Setup
# We can download the data from Kaggle to our data folder using the command line:
# 
# `kaggle competitions download -c titanic`
# 
# `unzip titanic.zip`
# 
# After that, let's get the data into some Pandas dataframes:

# In[ ]:


train_df = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')
test_df = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')


# # Exploratory Data Analysis:
# 
# Our next step will be to ask and answer the following questions:
# 
# 1. _Are we missing any data?_ 
# 2. _What form does our data take?_
# 3. _What additional information can we garner from what we already have?_
# 4. _What relationships can we find between our variables, especially between the input and output variables?_ 
# 5. _How can we use the answers to the first two question to add value to our data and the models that will use it?_
# 
# 
# ## Question 1: Missing Data
# Let's take a look at the number of entries in our training data, as well as those variables contain significant missing data. Below, we see that the training data contains 891 passenger samples, with 11 total variables describing each passenger. We see that there is a significant amount of missing data for the variables __Age__ and __Cabin__. We will have to deal with this missing data by either finding an intelligent way to fill the gaps, or perhaps dropping the features entirely. 

# In[ ]:


# Question 1: Are we missing any data?
train_df.info()


# ## Question 2: What is the form of our data:
# Taking a look at our `.info()` print out as well as the first few entries of our data frame below, we see that our data comes primarily in the form of categorical data, with the exception of __Age__ and __Fare__. These categories are described by Python strings, which is why the data type above is listed as 'object'. This is how Pandas deals with unidentified data types. We will later tell Pandas that these variables are strings. 

# In[ ]:


# Look at the first few entries
train_df.head()


# ## Question 3: What additional information can we garner from what we already have?
# 
# ### Passenger Title
#  
# A quick glance at the __Name__ variable shows us that each name comes with a title. A title is useful in telling us things like social status, marriage status career, and even rank within a specific career. Therefore, it may be useful to have this information on hand. Let's parse it out:

# In[ ]:


train_df['Title'] = train_df['Name'].str.extract(r'([A-Za-z]+)\.')
train_df.Title.value_counts()


# Next, we might notice that many of these titles are synonymous. For example, _Mme_ is the French equivalent to  'Mrs' and _Mlle_ is the equivalent to  'Miss'. Other titles imply varying levels of nobility like 'Sir', 'Countess' and 'Don'. Some titles infer a profession. Let's reduce our titles to their common denominators:

# In[ ]:


title_dict = {
    'Mrs': 'Mrs', 'Lady': 'Lady', 'Countess': 'Lady',
    'Jonkheer': 'Lord', 'Col': 'Officer', 'Rev': 'Rev',
    'Miss': 'Miss', 'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss', 'Dona': 'Lady',
    'Mr': 'Mr', 'Dr': 'Dr', 'Major': 'Officer', 'Capt': 'Officer', 'Sir': 'Lord', 'Don': 'Lord', 'Master': 'Master'
}

train_df.Title = train_df.Title.map(title_dict)


# In[ ]:


sns.countplot(train_df.Title).set_title("Histogram of Categorical Data: Title")


# In[ ]:


train_df.head(1)


# By again looking at our data set, we might notice the variables __SibSp__ and __Parch__. The first is the number of siblings and/or spouses that a passenger traveled with. The second is the number of parents and/or children a passenger traveled with. Combining these two variables we can get total family size. 

# In[ ]:


train_df['FamilySize'] = 1 + train_df.SibSp + train_df.Parch


# In[ ]:


sns.countplot(train_df.FamilySize)


# It appears that the Titanic's voyage was not necessarily a couple's or family affair. The majority of passengers traveled alone, and perhaps that is valuable information. Let's add the category __Alone__.

# In[ ]:


train_df['Alone'] = train_df.FamilySize.apply(lambda x: 1 if x==1 else 0)
plt.figure(figsize=(8,5))
sns.countplot(train_df.Alone)


# ### Last Name
# 
# A last name is a group identity. While we know that many passengers traveled alone, there were still a significant number of families onboard the Titanic. Perhaps survival among specific families was more common than others. This is all speculation, but perhaps worth a look.

# In[ ]:


train_df['LName'] = train_df.Name.str.extract(r'([A-Za-z]+),')


# ### Name Length
# 
# This one has a very simple explanation: While reviewing notebooks on Kaggle, I saw that one competitor found that the length of a person's name added to the performance of the model. So, why not try it out?

# In[ ]:


train_df['NameLength'] = train_df.Name.apply(len)
train_df


# ## Question 4: What statistical relationships does our data contain?
# 
# We now have a more robust data set that includes (possibly) valuable new insights into our passengers lives. But how helpful is this data, really? One way to find out is to look at the statistical relationships between our variables, especially between each input variable and our single output variable __Survived__. 
# 
# Correlation is a common go-to tool we would use to determine such relationships. However, it is important to note that we have mostly categorical data in our data set, and that throws a small wrench in our gears. 
# 
# First, our categorical data needs to be encoded into numeric format before we can do calculations of any kind. 
# 
# Next, we need to consider the types categorical we are studying:
# 
# - __Ordinal__ variables imply an underlying rank, or order. The classifications mild, moderate, severe would be an example. A common method of calculating the correlation is called _Kendall's Tau ($\tau$)_. 
# - __Nominal__ variables have no such rank or order. An example would be Male or Female. In this case we will use _Cramer's V_ correlation.

# In[ ]:


train_df.head(1)


# In[ ]:


# nominal variables (use Cramer's V)
nom_vars = ['Survived', 'Title', 'Embarked', 'Sex', 'Alone', 'LName']

# ordinal variables (nominal-ordinal, use Rank Biserial or Kendall's Tau)
ord_vars = ['Survived', 'Pclass', 'FamilySize', 'Parch', 'SibSp', 'NameLength']

# continuous variables (use Pearson's r)
cont_vars = ['Survived', 'Fare', 'Age']


# In the cell above, we separate our variables by their data types. The reason for this is that when considering the underlying associations between variables, there is not a "one size fits all" method. The most common mathematical method of calculating correlation is _Pearson's r_, which should typically only be used on continuous variables. In our case, the vast majority of variables are actually discrete/categorical. 
# 
# In order to perform calculations, we must convert any non numeric data into numbers. Let's get started:

# In[ ]:


# convert all string 'object' types to numeric categories
for i in train_df.columns:
    if train_df[i].dtype == 'object':
        train_df[i], _ = pd.factorize(train_df[i])


# In[ ]:


def cramers_v_matrix(dataframe, variables):
    
    df = pd.DataFrame(index=dataframe[variables].columns, columns=dataframe[variables].columns, dtype="float64")
    
    for v1, v2 in itertools.combinations(variables, 2):
        
        # generate contingency table:
        table = pd.crosstab(dataframe[v1], dataframe[v2])
        n     = len(dataframe.index)
        r, k  = table.shape
        
        # calculate chi squared and phi
        chi2  = ss.chi2_contingency(table)[0]
        phi2  = chi2/n
        
        # bias corrections:
        r = r - ((r - 1)**2)/(n - 1)
        k = k - ((k - 1)**2)/(n - 1)
        phi2 = max(0, phi2 - (k - 1)*(r - 1)/(n - 1))
        
        # fill correlation matrix
        df.loc[v1, v2] = np.sqrt(phi2/min(k - 1, r - 1))
        df.loc[v2, v1] = np.sqrt(phi2/min(k - 1, r - 1))
        np.fill_diagonal(df.values, np.ones(len(df)))
        
    return df


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(20,6))

# nominal variable correlation
ax1 = sns.heatmap(cramers_v_matrix(train_df, nom_vars), annot=True, ax=axes[0], vmin=0)

# ordinal variable correlation: 
ax2 = sns.heatmap(train_df[ord_vars].corr(method='kendall'), annot=True, ax=axes[1], vmin=-1)

# Pearson's correlation:
ax3 = sns.heatmap(train_df[cont_vars].corr(), annot=True, ax=axes[2], vmin=-1)

ax1.set_title("Cramer's V Correlation")
ax2.set_title("Kendall's Tau Correlation")
ax3.set_title("Pearson's R Correlation")


# The above heatmaps show our strength of association between each variable. While there is no rigid standard for "Highly Associated" or "Weakly Associated", we will use a cut-off value of |0.1| between our independent variables and survival. We will likely drop features whose association is lower than |0.1|. This is an entirely arbitrary guess, and I may return to raise or lower the bar later. 
# 
# For now, the features that meet the criteria for dropping are __SibSp__ and __Age__ (I actually end up leaving __Age__ in, as the model performed better with it). Additionally, I am choosing to drop __Name__, __Ticket__ and __Cabin__, mostly on a hunch that they don't add much. 
# 

# In[ ]:


todrop = ['SibSp', 'Ticket', 'Cabin', 'Name']
train_df = train_df.drop(todrop, axis=1)
train_df


# # Setup for Machine Learning:
# 
# During this phase, we will begin to format our data for feeding into a machine learning algorithm. We will then use this formatted data to get a picture of what a few different models can do for us, and pick the best one. This phase is broken into the following parts:
# 
# 1. __Train/Test Split__
# 2. __Normalize Data of each split__ 
# 3. __Impute missing values__ (I feel that __Cabin__ is a lost cause, as so many entries are missing, but we will look at __Age__.)
# 
# Let's go.
# 
# ## Train/Test Split
# 
# We will split our data once into training and testing sets. Within the training set, we will use stratified k-fold cross validation to find average performance of our models. 
# 
# The test set will not be touched until after we have fully tuned each of our candidate models using the training data and k-fold cross validation. Once training and tuning is complete, we will compare the results of each model on the held-out test set. The one that performs the best will be used for the competition.

# In[ ]:


X = train_df.drop(['Survived'], axis = 1)
Y = train_df.loc[:, 'Survived']

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=333)


# ## Normalizing the Data
# 
# Some Machine Learning models require all of our predictors to be on the same scale, while others do not. Most notably, models like Logistic Regression and SVM will probably benefit from scaling, while decision trees will simply ignore scaling. Because we are going to be looking at a mixed bag of algorithms, I'm going to go ahead and scale our data.

# In[ ]:


# We normalize the training and testing data separately so as to avoid data leaks.

x_train = pd.DataFrame(pre.scale(x_train), columns=x_train.columns, index=x_train.index)
x_test = pd.DataFrame(pre.scale(x_test), columns=x_test.columns, index=x_test.index)
x_train


# ## Imputing Missing Data
# 
# You might recall that there were a significant amount of missing __Age__ values in our data. Let's fill this in with the median age:

# In[ ]:


x_train.loc[x_train.Age.isnull(), 'Age'] = x_train.loc[:, 'Age'].median()
x_test.loc[x_test.Age.isnull(), 'Age'] = x_test.loc[:, 'Age'].median()
x_train.info()


# Now, all of the data in our training `DataFrame` is non-null. Later we will have to repeat this process on the test data.

# # Model Selection
# 
# Now that we have prepared our data, we want to look at different options available to us for solving classification problems. Some common ones are:
# 
# - K-Nearest Neighbors
# - Support Vector Machines
# - Decision Trees
# - Logistic Regression
# 
# We will train and tune each of these models on our training data by way of k-fold cross-validation. When complete, we will compare the tuned models' performance on a held out test set. 
# 
# ## Training and Comparing Base Models:
# 
# First, we want to get a feel model's performance before tuning. We will write two functions to help us describe our results. The first will evaluate the model several times over random splits in the data, and return the average performance as a dictionary. The second will simply nicely print our dictionary.

# In[ ]:


def kfold_evaluate(model, folds=5):
    eval_dict = {}
    accuracy = 0
    f1       = 0
    AUC      = 0
    
    skf = model_selection.StratifiedKFold(n_splits=folds)
    
    # perform k splits on the training data. Gather performance results.
    for train_idx, test_idx in skf.split(x_train, y_train):
        xk_train, xk_test = x_train.iloc[train_idx], x_train.iloc[test_idx]
        yk_train, yk_test = y_train.iloc[train_idx], y_train.iloc[test_idx]
    
        model.fit(xk_train, yk_train)
        y_pred = model.predict(xk_test)
        report = metrics.classification_report(yk_test, y_pred, output_dict=True)
        
        prob_array = model.predict_proba(xk_test)
    
        fpr, tpr, huh = metrics.roc_curve(yk_test, model.predict_proba(xk_test)[:,1])
        auc = metrics.auc(fpr, tpr)
        accuracy   += report['accuracy']
        f1         += report['macro avg']['f1-score']
        AUC        += auc
        
    # Average performance metrics over the k folds
    measures = np.array([accuracy, f1, AUC])
    measures = measures/folds

    # Add metric averages to dictionary and return.
    eval_dict['Accuracy']  = measures[0]
    eval_dict['F1 Score']  = measures[1]
    eval_dict['AUC']       = measures[2]  
    eval_dict['Model']     = model
    
    return eval_dict

# a function to pretty print our dictionary of dictionaries:
def pprint(web, level):
    for k,v in web.items():
        if isinstance(v, dict):
            print('\t'*level, f'{k}: ')
            level += 1
            pprint(v, level)
            level -= 1
        else:
            print('\t'*level, k, ": ", v)


# In[ ]:


evals = {}
evals['KNN'] = kfold_evaluate(KNeighborsClassifier())
evals['Logistic Regression'] = kfold_evaluate(LogisticRegression(max_iter=1000))
evals['Random Forest'] = kfold_evaluate(RandomForestClassifier())
evals['SVC'] = kfold_evaluate(SVC(probability=True))


# In[ ]:


result_df = pd.DataFrame(evals)
result_df.drop('Model', axis=0).plot(kind='bar', ylim=(0.7, 0.9)).set_title("Base Model Performance")
plt.xticks(rotation=0)
plt.show()
result_df


# ### Base Model Summary
# 
# It appears that we have a clear winner in our Random Forest classifier.  
# 
# ## Hyper-parameter Tuning: 
# 
# Let's tune up our current champion's hyper-parameters in hopes of eking out a little bit more performance. We will use scikit-learn's `RandomizedSearchCV` which has some speed advantages over using an exhaustive `GridSearchCV`. Our first step is to create our grid of parameters over which we will randomly search for the best settings:

# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators, 
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid, 0)


# Next, we want to create our `RandomizedSearchCV` object which will use the grid we just created above. It will randomly sample 10 combinations of parameters, test them over 3 folds and return the set of parameters that performed the best on our training data.

# In[ ]:


# create RandomizedSearchCV object
searcher = model_selection.RandomizedSearchCV(estimator = RandomForestClassifier(),
                                            param_distributions = random_grid,
                                            n_iter = 10, # Number of parameter settings to sample (this could take a while)
                                            cv     = 3,  # Number of folds for k-fold validation 
                                            n_jobs = -1, # Use all processors to compute in parallel
                                            random_state=0) 
search = searcher.fit(x_train, y_train)


# In[ ]:


params = search.best_params_
params


# After performing our parameter tuning, we can verify whether or not the parameters provided by the search actually improve the base model or not. Let's compare the performance of the two models before and after tuning.

# In[ ]:


tuning_eval = {}
tuned_rf = RandomForestClassifier(**params)
basic_rf = RandomForestClassifier()

tuning_eval['Tuned'] = kfold_evaluate(tuned_rf)
tuning_eval['Basic'] = kfold_evaluate(basic_rf)

result_df = pd.DataFrame(tuning_eval)
result_df.drop('Model', axis=0).plot(kind='bar', ylim=(0.7, 0.9)).set_title("Tuning Performance")
plt.xticks(rotation=0)
plt.show()
result_df


# # Final Steps: 
# 
# Now that we have chosen and tuned a Random Forest classifier, we want to test it on data it has never before seen.  This will tell us how we might expect the model to perform in the future, on new data. It's time to use that held out test set. 
# 
# Then, we will combine the test and training data, and re-fit our model to the combined data set, hopefully giving it the greatest chance of success on the unlabeled data from the competition. 
# 
# Finally, we will make our predictions on the unlabeled data for submission to the competition. 
# 
# ### Final Test on Held Out Data

# In[ ]:


y_pred = tuned_rf.predict(x_test)


# In[ ]:


results = metrics.classification_report(y_test, y_pred,
                                        labels = [0, 1],
                                        target_names = ['Died', 'Survived'],
                                        output_dict = True)

pprint(results, 0)


# It looks like we may have experienced some overfitting. Our model's performance on the test data is roughly 8-9% lower across the board. 

# ### Combine Training and Testing Datasets for Final Model Fit
# 
# Now that we have ascertained that our tuned model performs with about 76% accuracy and has an f1-score of 0.74 on new data, we can proceed to train our model on the entire labeled training set. 

# In[ ]:


X = pd.concat([x_train, x_test], axis=0).sort_index()
Y = pd.concat([y_train, y_test], axis=0).sort_index()
tuned_rf.fit(X, Y)


# ### Format and Standardize Unlabeled Data
# 
# Next we need to transform our unlabeled data in the same manner as when we were formatting our training data. This includes encoding categorical variables, dropping the same features and normalization. This should ensure consistent results on the never before seen competition data. 

# In[ ]:


# Feature Engineering:
test_df['Title'] = test_df.Name.str.extract(r'([A-Za-z]+)\.')
test_df['LName'] = test_df.Name.str.extract(r'([A-Za-z]+),')
test_df['NameLength'] = test_df.Name.apply(len)
test_df['FamilySize'] = 1 + test_df.SibSp + test_df.Parch
test_df['Alone'] = test_df.FamilySize.apply(lambda x: 1 if x==1 else 0)
test_df.Title = test_df.Title.map(title_dict)

# Feature Selection
test_df = test_df.drop(todrop, axis=1)

# Imputation of missing age and fare data
test_df.loc[test_df.Age.isna(), 'Age'] = test_df.Age.median()
test_df.loc[test_df.Fare.isna(), 'Fare'] = test_df.Fare.median()

# encode categorical data
for i in test_df.columns:
    if test_df[i].dtype == 'object':
        test_df[i], _ = pd.factorize(test_df[i])
        
# center and scale data 
test_df = pd.DataFrame(pre.scale(test_df), columns=test_df.columns, index=test_df.index)

# ensure columns of unlabeled data are in same order as training data.
test_df = test_df[x_test.columns]
test_df


# ### Make Final Predictions
# 
# Roughly 32 percent of the passengers aboard the Titanic died. We will do a last, common sense check to see if our algorithm predicts roughly the same distribution of survivals.

# In[ ]:


final = tuned_rf.predict(test_df)


# In[ ]:


final.sum()/len(final)


# In[ ]:


submission = pd.DataFrame({'PassengerId':test_df.index,
                           'Survived':final})
submission


# In[ ]:


submission.to_csv('submission2.csv', index=False)

