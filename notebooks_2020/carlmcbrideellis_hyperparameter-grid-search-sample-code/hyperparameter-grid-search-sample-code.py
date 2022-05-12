#!/usr/bin/env python
# coding: utf-8

# ### Hyperparameter grid search sample code
# This is a sample code for performing a hyperparameter grid search using the [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) routine from scikit-learn. We shall use the default 5-fold [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics&#41;). Finally, for the classifier we shall use the [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), also from scikit-learn.
# 

# In[ ]:


#!/usr/bin/python3
# coding=utf-8
#===========================================================================
# This is a simple script to perform a classification on the kaggle 
# 'Titanic' data set using a grid search, in conjunction with a 
# random forest classifier
# Carl McBride Ellis (1.V.2020)
#===========================================================================
#===========================================================================
# load up the libraries
#===========================================================================
import pandas as pd
import numpy  as np

#===========================================================================
# read in the data
#===========================================================================
train_data = pd.read_csv('../input/titanic/train.csv')
test_data  = pd.read_csv('../input/titanic/test.csv')

#===========================================================================
# select some features of interest ("ay, there's the rub", Shakespeare)
#===========================================================================
features = ["Pclass", "Sex", "SibSp", "Parch", "Embarked"]

#===========================================================================
# for the features that are categorical we use pd.get_dummies:
# "Convert categorical variable into dummy/indicator variables."
#===========================================================================
X_train       = pd.get_dummies(train_data[features])
y_train       = train_data["Survived"]
final_X_test  = pd.get_dummies(test_data[features])

#===========================================================================
# hyperparameter grid search using scikit-learn GridSearchCV
# we use the default 5-fold cross validation
#===========================================================================
from sklearn.model_selection import GridSearchCV
# we use the random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='gini', max_features='auto')
gs = GridSearchCV(cv=5, error_score=np.nan, estimator=classifier,
# dictionaries containing values to try for the parameters
param_grid={'min_samples_leaf':  [10, 15, 20],
            'max_depth':         [3, 4, 5, 6],
            'n_estimators':      [10, 20, 30]})
gs.fit(X_train, y_train)

# grid search has finished, now echo the results to the screen
print("The best score is %.5f"  %gs.best_score_)
print("The best parameters are ",gs.best_params_)
the_best_parameters = gs.best_params_

#===========================================================================
# now perform the final fit, using the best values from the grid search
#===========================================================================
classifier = RandomForestClassifier(criterion='gini', max_features='auto',
             min_samples_leaf  = the_best_parameters["min_samples_leaf"],
             max_depth         = the_best_parameters["max_depth"],
             n_estimators      = the_best_parameters["n_estimators"])
classifier.fit(X_train, y_train)

#===========================================================================
# use the model to predict 'Survived' for the test data
#===========================================================================
predictions = classifier.predict(final_X_test)

#===========================================================================
# write out CSV submission file
#===========================================================================
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 
                       'Survived': predictions})
output.to_csv('submission.csv', index=False)


# Now let us compare our score with the final leaderboard score

# In[ ]:


from sklearn.metrics import accuracy_score
solution   = pd.read_csv('../input/submission-solution/submission_solution.csv')
print("The test (i.e. leaderboard) score is %.5f" % accuracy_score(solution['Survived'],predictions))


# We can see that our score is significantly higher than the final leaderboard score. This is a clear symptom of overfitting, which is *very* easy to do on the Titanic dataset. For more about this see ["*Overfitting and underfitting the Titanic*"](https://www.kaggle.com/carlmcbrideellis/overfitting-and-underfitting-the-titanic). Part of the reason for this is simply that the Titanic dataset is very small, and splitting it up makes it even smaller, with each small piece being even less representative of the whole. Another reason is due to the technique of cross-validation itself, whose variance estimate is a bit too small due to the correlation between the error estimates in different folds. This in turn means that the confidence intervals for prediction error are too small, leading to an overly-optimistic hyperparameter selection.
# 
# A better solution, although more computationally expensive, is to use **nested cross-validation**. To learn more about nested-CV see:
# 
# * ["*Nested versus non-nested cross-validation*"](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) on scikit-learn
# * [Stephen Bates, Trevor Hastie and Robert Tibshirani "*Cross-validation: what does it estimate and how well does it do it?*", arXiv:2104.00673 (2021)](https://arxiv.org/pdf/2104.00673.pdf)
# * ["*Nested Cross-Validation for Machine Learning with Python*"](https://machinelearningmastery.com/nested-cross-validation-for-machine-learning-with-python/) on *Machine Learning Mastery*
# * ["*A step by step guide to Nested Cross-Validation*"](https://www.analyticsvidhya.com/blog/2021/03/a-step-by-step-guide-to-nested-cross-validation/) on *Analytics Vidhya*
