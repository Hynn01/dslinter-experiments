#!/usr/bin/env python
# coding: utf-8

# ## Intro
# 
# As some kind of entry point I wanted to start with the classical Titanic dataset, I’ll try to cover different stages of modelling from EDA to ensembling suitable models. I’ll omit some details to make this notebook much easier to scroll and navigate. Hope you’ll like it. Maybe it can be a good tutorial for beginners. I might add some more references and more details of every aspect.

# ## Importing libraries
# 

# In[ ]:



# Essential
import numpy as np
import pandas as pd

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# Models
import xgboost as xgb
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB


# Model evaluation and tuning
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import classification_report
import shap


# Imputing and scaling 
from sklearn.impute._knn import KNNImputer
from sklearn.preprocessing import StandardScaler






# ## Importing dataset
# We will need to concatenate train and test data for future feature engineering. It might be not recommended in some cases and can cause a data leakage. But we will discuss some caveats later.
# 
# Let's see what features we have

# In[ ]:


train_data = pd.read_csv('../input/titanic/train.csv')
# train_data['Survived'] = train_data['Survived'].astype(int)
test_data = pd.read_csv('../input/titanic/test.csv')
full_data =  train_data.append(test_data)

train_data.head()


# ## Exploring data 
# We have 3 categorical features:
#  - `PClass`
#  - `Sex`
#  - `Embarked`
# 
# We also have 4 numerical features:
#  - `Age`
#  - `SibSp`
#  - `Parch`
#  - `Fare`
# 
# And 3 nominal features:
#  - `Name`
#  - `Ticket`
#  - `Cabin`
# 
#    

# Let's see stats of numerical features in train dataset

# In[ ]:


train_data.describe()


# There’s not much interesting data we can see now, but we can see that `count` in column `Age` varies from other features, which means that we have some missing values. Let’s see how many null values we have in the train dataset:

# In[ ]:


print('Number of rows ',len(train_data))
print(train_data.isnull().sum())


# We have 177 rows with missing `Age` and 687 rows with missing `Cabin`

# I can see `PClass`,`Sex`,`Age`,`SibSp`,`Parch`,`Fare`,`Cabin` as potential important variables. 
# Also, we might combine some of those features (For example SibSp and Parch as they might be considered 'family')

# In[ ]:


full_data['FamilyMembers'] = full_data['SibSp'] + full_data['Parch']
train_data['FamilyMembers'] = train_data['SibSp'] + train_data['Parch']


# After combining `SibSp` and `Parch` into the new feature `FamilyMembers` counting number of family members for each passenger, we will visualize everything we have for know. First, let’s see how many passengers survived based on how big is the family passenger is travelling with (Chart on the left). And how many passengers survived based on `Sex` and `Pclass` (Chart on the right).

# In[ ]:



fig,ax = plt.subplots(1,2,figsize=(10,6))
sns.countplot(x='FamilyMembers',hue='Survived',data=train_data,ax=ax[0]).set(title='How many passengers survived? \nGrouped by number of family members on board',ylabel='Survived',xlabel='Family members')
sns.barplot(x='Sex',y='Survived',hue='Pclass',data=train_data,ax=ax[1]).set(title='How many passengers survived? (in percents)',ylabel='Percentage')
ax[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.00))


# Ok, so we can see that solo travellers died more often compared to the ones with family.
# Also, there’s a strong sign that females have a higher chance to survive.
# And we can see that males from 1st class had a higher chance to survive than males from 2nd and 3rd class.

# We also need to answer other questions info about the dataset:
# - How important is info about the port where passengers embarked?
# - Does `Age` distribution vary in groups of passengers who died and lived? Do we need to impute `Age` for the `177` passengers?
# 
# We can visualize those questions and try to answer them

# In[ ]:


fig,ax = plt.subplots(1,2,figsize=(10,6))


sns.histplot(x='Age',hue='Survived',data=train_data,ax=ax[0],kde=True).set(title='How many passengers survived? (Age,Pclass) ')
sns.barplot(x='Sex',y='Survived',hue='Embarked',data=train_data,ax=ax[1]).set(title='How many passengers survived? (in percents)',ylabel='Percentage')
ax[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.00))


# There's a small bump for passengers aged < 10 years. It is because children were prioritized during the evacuation.
# 
# There's no strong evidence if embark port affect the result since confidence intervals (black lines on bars) overlap with each other.

# But what about imputing `177` rows for `Age` feature? Do we need to fill missing values or this feature is not that important?
# Overall, it is not that important, we can see it on distribution plots, only very young passengers had a higher chance to survive.
# We can make `Age` feature a categorical feature and divide it in year bins.
# 
# However, there’s a small detail about the dataset however - if we would examine `Name` feature, we might see that there is a title of the passenger (e.g. Mr,Mrs,Dr,etc.).

# And one of the important titles is 'Master' whichб according to wikipedia, is used for boys:
# >  ... in the United States, unlike the UK, a boy can be addressed as Master only until age 12, then is addressed only by his name with no title until he turns 18, when he takes the title of Mr.
# 
# Therefore, we can consider passengers with the title ‘Master’ as young boys for age feature, which would most likely fit in that orange bump on the left chart with distribution.
# 
# I won't do it in this notebook, but it might be a good point for feature engineering or just to fill missing age values with consideration of the title of passenger.
# 

# ##  Feature engineering

# Additionally, to newly created combined feature `FamilyMembers` we also need to consider adding feature which identifies solo travellers (`IsSolo`). Also we'll fill empty values and change categorical string values to integer.  
# We will ignore `Age` feature in training, since it is difficult to correctly predict how old is the passenger (except those with 'Master' title of course)

# In[ ]:


#Passenger considered solo if he has no family members on board
full_data['IsSolo'] = (full_data['FamilyMembers'] == 0).astype(int)

# test_data['FamilyMembers'] = test_data['SibSp']+test_data['Parch']


# Replace string value to numbers. There's 2 nan values in test data, we will change them to value of most common port ('S') 
full_data['Embarked'] = full_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2,np.nan:0} ).astype(int)

# Replace string value of sex to numbers 1 - female, 0 - male
full_data['Sex'] = (full_data['Sex'] == 'female').astype(int)



#There's 1 missing value for Fare in test dataset, let's fill it with mean value
full_data['Fare'].fillna(full_data['Fare'].mean(), inplace=True)


# 
# If we examine the dataset more carefully, we will see interesting details considering a group of travellers:
# - Families usually pay equal fare and obviously have the same last name. 
# - Group of friends/relatives with different last names usually have the same ticket number
# 
# We can use those facts as a new feature that represents the chance of survival of this family/group, let's call it `Family_Survival`.
# I've changed the method used by [S.Xu's](https://www.kaggle.com/shunjiangxu/blood-is-thicker-than-water-friendship-forever), but changed his approach of minmaxing the family survival to average score, it seemed more precise.  

# In[ ]:


# Extracting last name from Name feature
full_data['Last_Name'] = full_data['Name'].apply(lambda x: str.split(x, ",")[0])

# Filling default value of family/group survival as mean of individual survival 
full_data['Family_Survival'] = train_data['Survived'].mean()


# for loop to find family members (family with same surname)
for grp, grp_df in full_data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            #check if whole family doesn't have 'Survived' value  
            if (np.isnan(grp_df['Survived']).all()):
                continue
            average_family_score = (grp_df.drop(ind)['Survived'].mean())
            #check if average_family_score is nan, it happens when only current passenger has 'Survived' value
            if np.isnan(average_family_score):
                average_family_score = row['Survived']
            average_family_score = round(average_family_score)   
            passID = row['PassengerId']
            full_data.loc[full_data['PassengerId'] == passID, 'Family_Survival'] = average_family_score

# for loop to find group of passengers who bought ticket together (friends,relatives)
for _, grp_df in full_data.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival'] == 0.5):
                if (np.isnan(grp_df['Survived']).all()):
                    continue
                average_family_score = (grp_df.drop(ind)['Survived'].mean())
                if np.isnan(average_family_score):
                    average_family_score = row['Survived']
                average_family_score = round(average_family_score)   
                passID = row['PassengerId']
                full_data.loc[full_data['PassengerId'] == passID, 'Family_Survival'] = average_family_score


# Good, now we can look at the updated dataset

# In[ ]:


full_data.head()


# ## Inspecting the models and features 

# After adding new features, we can start trying to choose the best model to fit the data.
# Let's add new features to train and test data.

# In[ ]:


train_data = full_data[:len(train_data)]
train_data.head()


# As previously mentioned, we wanted to use only certain features to train. 
# It's time to start preparing our data to train our models

# In[ ]:


features = ['Pclass','Sex','Fare','FamilyMembers','IsSolo','Family_Survival','Embarked']
y = train_data['Survived'].ravel()
X_train,X_val,y_train,y_val = train_test_split(train_data[features],y,test_size=0.20,random_state=111)


# We will test multiple types of models, such as:
# - Logistic regression
# - Support Vector Machine
# - Random Forest
# - Naive Bayes
# - KNN
# - XGBoosting 

# To correctly choose the right model for our task, we need to evaluate each model. We will use cross-validation during training models with default parameters.
# Hyperparameter tuning will be performed after we chose the most effective models.
# 
# Here’s a basic wrapper to make this process easier

# In[ ]:


def test_models(model,X,y_train):
    key = type(model).__name__
    model.fit(X,y_train)
    model_score =model.score(X,y_train)
    model_score=cross_val_score(model,X,y_train,cv=5).mean()
    if key not in summary:
        summary[key] = []
    summary[key].append(model_score)
    return summary


# We need to standardize our training data since some models are very sensitive to unscaled data.
# We'll do an experiment to showcase this: 
# 

# In[ ]:


scaler = StandardScaler()

features = ['Pclass','Sex','Fare','FamilyMembers','IsSolo','Family_Survival','Embarked']
summary={}
models_to_check= [SVC(),KNeighborsClassifier(),xgb.XGBClassifier(use_label_encoder=False,eval_metric='logloss'),LogisticRegression(solver='liblinear'),GaussianNB(),RandomForestClassifier()]

for item in models_to_check:
    summary = test_models(item,X_train[features],y_train)

print(X_train[features].columns)
X = scaler.fit_transform(X_train[features])


for item in models_to_check:
    summary = test_models(item,X,y_train)

summary = pd.DataFrame.from_dict(summary,orient='index',columns=['Without scaler','With scaler'])
print(summary)
("")


# As we can see, all models increased score with scaled data.
# Solver also failed to converge on non-scaled training data, so there is an undoubted need to scale data for this dataset.

# Ok, what about features, we can examine the importance of features. We will inspect the best classifier from our test - `XGBoost`.

# We will describe model's features importance in bar chart

# In[ ]:


model = xgb.XGBClassifier(use_label_encoder=False,eval_metric='logloss')
model.fit(X_train,y_train)
feature_importances = model.feature_importances_


plt.yticks(range(len(feature_importances)), features[:len(feature_importances)])
plt.xlabel('Relative Importance')
plt.barh(range(len(feature_importances)), feature_importances[:len(feature_importances)], color='b', align='center')
plt.title('Feature Importances')


# As we probably expected, `Sex` is the most important feature, after that we have `Pclass`, `Family_Survival` and `FamilyMembers`.
# Surprisingly, the new feature, `IsSolo` is practically useless.
# 
# Okay,let's see how features affect our model's output.
# One of the most useful and beautiful ways to plot the feature's output is in SHAP library in `summary_plot` method, by calculating Shapley values to explain the impact of the features on prediction.

# In[ ]:


model = model.fit(X_val,y_val)
explainer = shap.Explainer(model)
shap_values = explainer(X_val)

shap.summary_plot(shap_values)


# Just interpret this plot as - closer to the right side means more impact of that feature on prediction being 'Survived', closer to the left side 'Died'. 
# Red means closer to the higher value of the described feature, blue means closer to the low value of the described feature (Pclass for example: 3 - red, 2 - purple, 1 - blue).
# 
# 
# We'll break it down one by one:
# - `Sex` affects the chance of surviving, hence why red(bigger value, 1.0 in this case) are skewed to the right side.
# - `Fare` doesn't affect the output based on its value.
# - `Family_survival` does affect the output, passengers which families/group had bigger chance to live expected to survive, passengers which  families/groups had medium chance to live had smaller chance to live, and passengers with families/groups with smallest chance of survival expected to die with them(I guess...).
# - `Pclass` does  slightly affect the output, 1st class passengers are more likely to survive, than 2nd and 3rd class
# - `Embarked` is strange, but passengers from 'S'-Southampton are more likely to die
# - `Family members` does not affect that much
# - `IsSolo` does not affect at all
# 

# ## Choosing the best model
# Now we need to get the best hyperparameters for our models. Different methods can help in fine tuning the hyperparameters. We will use Grid Search and combine it with Stratified KFold cross validation. We will try to find the best parameters for each model and stack them later.  
# So I will omit the details to save valuable compiling time and set smaller parameter grids just to show the process.
# I will also set only 4 important features in training data, since I tried different combinations of them and these seemed most useful.

# Here's another small wrapper for Grid Search

# In[ ]:


def GridSearchCVWrapper(model,parameters, X_train,X_val,y_train,y_val):

    clf = GridSearchCV(estimator=model, param_grid=parameters,n_jobs=-1,
                    cv=StratifiedKFold(n_splits=5), 
                    scoring=['accuracy','recall','f1','roc_auc'],
                    verbose=1,refit='roc_auc')
    clf.fit(X_train,y_train)          
    preds = clf.best_estimator_.predict(X_val)
    print(classification_report(preds,y_val))
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring = "roc_auc")
    print("Scores:", scores)
    print(f"Mean:{scores.mean()} ± {scores.std()}")

    return clf


# Prepare our data for training

# In[ ]:


features = ['Pclass','Sex','FamilyMembers','Family_Survival']

test_data = full_data[len(train_data):]
test_data_x = test_data[features].copy(deep=True)
train_data = full_data[:len(train_data)]

scaler = StandardScaler()
X = train_data[features].copy(deep=True)
X= scaler.fit_transform(X)

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.20,random_state=111)


# In[ ]:


Logistic_model_params= {'penalty' : ['l1', 'l2'],
                        'C' : np.logspace(-4, 4, 20),
                        'solver' : ['liblinear']}

Logistic_model = GridSearchCVWrapper(LogisticRegression(),Logistic_model_params,X_train,X_val,y_train,y_val)
print(f"\nBest params for Logistic Regression are:")
print(Logistic_model.best_estimator_)


# In[ ]:


SVM_model_params = {'C':np.logspace(-2,1,4),
                    'gamma':np.logspace(-2,1,4),}
                    
SVM_model = GridSearchCVWrapper(SVC(),SVM_model_params, X_train,X_val,y_train,y_val)
print(f"\nBest params for SVM are:")
print(SVM_model.best_estimator_)


# In[ ]:



RF_model_params = { 'n_estimators': [200,350,500],
               'max_features': ['auto'],
               'max_depth': [2,5,None],
               'min_samples_split': [5, 10],
               'min_samples_leaf': [2, 4],
               'bootstrap': [True],
               'random_state':[1]}
RF_model = GridSearchCVWrapper(RandomForestClassifier(),RF_model_params,X_train,X_val,y_train,y_val)

print(f"\nBest params for Random Forest are:")
print(RF_model.best_estimator_)


# In[ ]:


Gaussian_model_params = {'var_smoothing':np.logspace(0,-9,100)}
Gaussian_model = GridSearchCVWrapper(GaussianNB(),Gaussian_model_params, X_train,X_val,y_train,y_val)

print(f"\nBest params for Naive Bayes are:")
print(Gaussian_model.best_estimator_)


# In[ ]:


#! for some reason this cell runs horribly slow in kaggle, so I truncated most of the parameters. I used params which i got from run on my pc.
Xgb_model_parameters = {
            'n_estimators': [200],
            'colsample_bytree': [0.7],
            'max_depth': [15],
            'reg_alpha': [1.1],
            'reg_lambda': [1.2],
            'n_jobs':[-1]}

Xgb_model = GridSearchCVWrapper(xgb.XGBClassifier(use_label_encoder=False,eval_metric='logloss'),Xgb_model_parameters,X_train,X_val,y_train,y_val)
print(f"\nBest params for XGBoost are:")
print(Xgb_model.best_estimator_)


# In[ ]:


KNN_model_params= {'n_neighbors':np.arange(1,30,2),
                    'leaf_size':np.arange(1,15,2),
                    'p':[1,2]}
KNN_model = GridSearchCVWrapper(KNeighborsClassifier(),KNN_model_params,X_train,X_val,y_train,y_val)

print(f"\nBest params for K Neighbors are:")
print(KNN_model.best_estimator_)


# ## Stacking models and getting results
# After completing training our models, it’s time to evaluate them and compare them one by one. We’ll do the last comparison and visualize the ROC AUC score of models on the heatmap to find a suitable combination of models for stacking.
# 
# The point is to combine the most accurate models to get a better score. We need to find models which have a little correlation between each other’s predictions.

# In[ ]:


data_of_classifier = pd.DataFrame()
classifiers = [SVM_model.best_estimator_,Xgb_model.best_estimator_, Logistic_model.best_estimator_, Gaussian_model.best_estimator_,RF_model.best_estimator_,KNN_model.best_estimator_]
for i in classifiers:
    fit_classifier = i.fit(X_train,y_train)
    data_of_classifier[type(i).__name__] = i.predict(X_val)
    print('Score of',type(i).__name__,':')
    print(cross_val_score(fit_classifier, X_train, y_train, cv=5, scoring = "roc_auc").mean())
sns.heatmap(data_of_classifier.astype(float).corr(),annot=True)


# Results are pretty even, but we will consider the models with the highest ROC AUC scores and combine 3 of them.
# 
# The good idea is to find models with less correlation between each other and high scores.
# 
# After some testing, the best combination I found was the Random Forest as a final estimator and will use KNeighbors and XGBoost as base estimators. I commented the different models in stacked classifier if you would want to test them.

# In[ ]:


data_to_test = scaler.transform(test_data[features])


# In[ ]:



estimators = [#('SVM',SVM_model.best_estimator_),
              ('XGB',Xgb_model.best_estimator_),
              ('Logistic',Logistic_model.best_estimator_)
               # ('Random Forest',Gaussian_model.best_estimator_),
               #('KNN',KNN_model.best_estimator_)
]

stacking_clf = StackingClassifier(estimators = estimators,final_estimator=RF_model.best_estimator_)

stacking_clf.fit(X,y)

predictions =  stacking_clf.predict(data_to_test)
predictions =predictions.astype(int)
final_results = pd.DataFrame({ 'PassengerId':test_data.PassengerId ,'Survived':predictions })
final_results.to_csv('../working/submission.csv',index=False)


# ## Conclusion
# I tried to do a little bit of everything on this notebook, so there's a lot of details I omitted, but I do appreciate your feedback
