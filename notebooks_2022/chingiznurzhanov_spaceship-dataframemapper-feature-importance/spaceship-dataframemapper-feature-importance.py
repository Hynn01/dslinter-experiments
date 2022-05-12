#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category= DataConversionWarning)
from IPython.core.display import HTML
from collections import Counter

# Data split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn_pandas import DataFrameMapper

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

#Feature engineering
from sklearn.inspection import permutation_importance
import shap
from sklearn.feature_selection import mutual_info_classif


# # In this work I want to show how to use DataFrameMapper and how well it shows the processes that we do during preprocessing. Also i find not many works with ShuffleSplit and StratifiedShuffleSplit. 

# ### p.s. some cell have import this is to make it easier to copy

# ## In this competition your task is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. To help you make these predictions, you're given a set of personal records recovered from the ship's damaged computer system.

# In[ ]:


df = pd.read_csv('../input/spaceship-titanic/train.csv')
test = pd.read_csv('../input/spaceship-titanic/test.csv')


# ## train.csv - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
# - PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
# - HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
# - CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
# - Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
# Destination - The planet the passenger will be debarking to.
# - Age - The age of the passenger.
# - VIP - Whether the passenger has paid for special VIP service during the voyage.
# RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
# - Name - The first and last names of the passenger.
# Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

# ### From https://www.kaggle.com/code/arootda/pycaret-visualization-optimization-0-81/notebook

# In[ ]:


from IPython.core.display import HTML

def multi_table(table_list):
    return HTML(
        f"<table><tr> {''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list])} </tr></table>")

multi_table([pd.DataFrame(df[i].value_counts()) for i in df.columns if i != 'Age'])


# In[ ]:


# miss value everywhere
plt.figure(figsize=(15,10))
sns.heatmap(df.isna(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})
plt.show()


# In[ ]:


df.describe().T


# In[ ]:


df[df.dtypes[(df.dtypes=="float64")|(df.dtypes=="int64")]
                        .index.values].hist(figsize=[24,24]);


# In[ ]:


# Ok now outlier
df[df.dtypes[(df.dtypes=="float64")|(df.dtypes=="int64")]
                        .index.values].boxplot(figsize=[15,10]);


# ### From https://www.kaggle.com/code/yassineghouzam/titanic-top-4-with-ensemble-modeling

# In[ ]:


from collections import Counter
def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = df[col].quantile(0.25)
        # 3rd quartile (75%)
        Q3 = df[col].quantile(0.75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers  

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(df,2,["RoomService","FoodCourt","ShoppingMall",'Spa',"VRDeck"])


# In[ ]:


df.loc[Outliers_to_drop]


# In[ ]:


# Drop outliers
df = df.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)


# In[ ]:


k = 100 #number of variables for heatmap
cols = df.corr().nlargest(k, 'Transported')['Transported'].index
cm = df[cols].corr()
plt.figure(figsize=(20,10))
sns.heatmap(cm, annot=True, cmap = 'viridis')


# In[ ]:


df[['PasId_group', 'PasId_number']] = df.PassengerId.str.split('_', expand=True)
test[['PasId_group', 'PasId_number']] = test.PassengerId.str.split('_', expand=True)

df[['Cabin_deck', 'Cabin_num','Cabin_side']] = df.Cabin.str.split('/', expand=True)
test[['Cabin_deck', 'Cabin_num','Cabin_side']] = test.Cabin.str.split('/', expand=True)

df['Age_type']=pd.cut(df.Age, bins=[0, 18, 99], labels=['child', 'adult'])
test['Age_type']=pd.cut(test.Age, bins=[0, 18, 99], labels=['child', 'adult'])


# In[ ]:


df['PasId_group'] = df['PasId_group'].astype('float64')
test['PasId_group'] = test['PasId_group'].astype('float64')

df['PasId_number'] = df['PasId_group'].astype('float64')
test['PasId_number'] = test['PasId_group'].astype('float64')

df['Cabin_num'] = df['Cabin_num'].astype('float64')
test['Cabin_num'] = test['Cabin_num'].astype('float64')

df['CryoSleep'] = df['CryoSleep'].astype('bool')
test['CryoSleep'] = test['CryoSleep'].astype('bool')

df['VIP'] = df['VIP'].astype('bool')
test['VIP'] = test['VIP'].astype('bool')


# In[ ]:


df = df.drop('Name',axis=1)
test = test.drop('Name',axis=1)
df = df.drop('PassengerId',axis=1)
test = test.drop('PassengerId',axis=1)
df = df.drop('Cabin',axis=1)
test = test.drop('Cabin',axis=1)
df = df.drop('PasId_number',axis=1)
test = test.drop('PasId_number',axis=1)


# In[ ]:


k = 100 #number of variables for heatmap
cols = df.corr().nlargest(k, 'Transported')['Transported'].index
cm = df[cols].corr()
plt.figure(figsize=(20,10))
sns.heatmap(cm, annot=True, cmap = 'viridis')


# In[ ]:


X = df.copy()
X = X.drop('Transported', axis=1)

y = df['Transported']


# # Data split

# In[ ]:


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, shuffle=True,random_state=0,stratify=y)


# ### Interesting train_test_split = StratifiedShuffleSplit with n_splits=1, and StratifiedShuffleSplit we can use in grid search

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=0)

train_index, val_index = next(iter(sss.split(X, y)))
X_train, X_test = X.iloc[train_index], X.iloc[val_index]
y_train, y_test = y.iloc[train_index], y.iloc[val_index]


# In[ ]:


len(X_train),len(X_test),len(y_train),len(y_test)


# In[ ]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn_pandas import DataFrameMapper


# In[ ]:


categorical_features = X_train.select_dtypes(include=['object','category']).columns
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
boolean_features = X_train.select_dtypes(include=['bool']).columns
cat = [([c], [SimpleImputer(strategy='most_frequent', fill_value='UNK'),
              LabelEncoder()]) for c in categorical_features]
num = [([n], [SimpleImputer(strategy='median'),StandardScaler()]) for n in numerical_features]

boolean = [([b], [OneHotEncoder(sparse=False, handle_unknown='ignore')]) for b in boolean_features]
mapper_for_scale = DataFrameMapper(num + cat + boolean, input_df=True,df_out=True)
mapper_for_pipe = DataFrameMapper(num + cat + boolean, input_df=True,df_out=True)


# ## When fit you need use X_train!!!

# In[ ]:


mapper_for_scale.fit(X_train);


# In[ ]:


X_train_scaled = mapper_for_scale.transform(X_train)
X_test_scaled = mapper_for_scale.transform(X_test)
test_scaled = mapper_for_scale.transform(test)
#if you want work with X
X_scaled = mapper_for_scale.transform(X)


# In[ ]:


#How X_train looks after preprocessing. Amazing you can see all old columns that you use
X_train_scaled


# In[ ]:


k = 100 #number of variables for heatmap
cols = X_scaled.corr().index
cm = X_scaled[cols].corr()
plt.figure(figsize=(20,10))
sns.heatmap(cm, annot=True, cmap = 'viridis')


# # With mapper and scaled X_train, y_train

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier( n_jobs=-1,
                       random_state=0)


# In[ ]:


# With scaled
model.fit(X_train_scaled,y_train)

model.score(X_train_scaled,y_train)


# In[ ]:


model.score(X_test_scaled,y_test)


# In[ ]:


## If you want search best parameters
#from sklearn.model_selection import RandomizedSearchCV
#
## Create the parameter grid
#rf_param_grid = {
#    'max_depth': np.arange(10,  25, 1),
#    'n_estimators': np.arange(50, 200, 2),
#    'min_samples_leaf': np.arange(2, 8, 1),
#    'min_samples_split': np.arange(2, 8, 1),
#    'criterion' : ("gini", "entropy")
#}
#
## Perform RandomizedSearchCV, we use cv=sss, we use StratifiedShuffleSplit
#gridsearch_roc_auc = RandomizedSearchCV(estimator=model, param_distributions=rf_param_grid,n_iter=100,
#                                        scoring='roc_auc', cv=sss, verbose=1,n_jobs=-1,random_state=0)
#
## Fit the estimator
#gridsearch_roc_auc.fit(X_scaled, y)
#
## Compute metrics
#print('Score: ', gridsearch_roc_auc.best_score_)
#print('Estimator: ', gridsearch_roc_auc.best_estimator_)


# In[ ]:


y_true = model.predict(test_scaled)


# # Pipeline

# ### Now pipeline  with mapper

# In[ ]:


clf = RandomForestClassifier(n_jobs=-1,
                       random_state=0)

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ("mapper", mapper_for_pipe),    
    ("clf", clf)
])


# In[ ]:


#With Pipeline
pipeline.fit(X_train,y_train)

pipeline.score(X_train,y_train)


# In[ ]:


pipeline.score(X_test,y_test)


# In[ ]:


y_true_pipeline = pipeline.predict(test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


# Let's compare model with pipeline and without. Accuracy 1 they are the same.
# I love DataFrameMapper you can see how your data change in procces
accuracy_score(y_true_pipeline,y_true_pipeline)


# In[ ]:


# with pipeline you need using this for param_grid
#sorted(pipeline.get_params().keys())


# In[ ]:


#rf_param_grid_pipe= {
#    'clf__max_depth': np.arange(3,  10, 1),
#    'clf__n_estimators': np.arange(90, 200, 2),
#    'clf__min_samples_leaf': np.arange(2, 4, 1)
#
#}
#
## Perform RandomizedSearchCV
#gridsearch_roc_auc_pipe = GridSearchCV(estimator=pipeline, param_grid=rf_param_grid_pipe,
#                                        scoring='roc_auc', cv=sss, verbose=1,n_jobs=-1)
## Fit the estimator
#gridsearch_roc_auc_pipe.fit(X, y)
#
## Compute metrics
#print('Score: ', gridsearch_roc_auc_pipe.best_score_)
#print('Estimator: ', gridsearch_roc_auc_pipe.best_estimator_)


# In[ ]:


y_preds = pipeline.predict(test)
sub = pd.read_csv('../input/spaceship-titanic/sample_submission.csv')
sub['Transported'] = y_preds.astype('bool')
sub.to_csv('submission.csv', index=False)

plt.figure(figsize=(6,6))
sub['Transported'].value_counts().plot.pie(explode=[0.1,0.1], autopct='%1.1f%%', shadow=True, textprops={'fontsize':16}).set_title("Prediction distribution")


# ### Score: 0.79378 not bad

# # Feature engineering

# In[ ]:


model = RandomForestClassifier(n_jobs=-1, random_state=0)


# In[ ]:


model.fit(X_train_scaled,y_train)


# # RandomForestClassifier importances

# In[ ]:


plt.rcParams['figure.figsize'] = (20,10)

# do code to support model
#"data" is the X dataframe and model is the SKlearn object

feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(X_train_scaled.columns, model.feature_importances_):
    feats[feature] = importance #add the name/value pair 

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=90)


# In[ ]:


importance_score = importances.sort_values(by='Gini-importance')

# I want to see >= 0.035
feature_importance_score_rf = importance_score[importance_score['Gini-importance'] >= 0.035]

feature_importance_score_rf.reset_index(inplace=True)

feature_importance_score_rf


# # Permutation importance

# In[ ]:


from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_train_scaled, y_train, n_repeats=30,random_state=0,n_jobs=-1)


# In[ ]:


permutation_importance_score_columns = []
permutation_importance_score = []
for i in result.importances_mean.argsort()[::-1]:
    if result.importances_mean[i] - 2 * result.importances_std[i]:
        print(f"{X_train_scaled.columns[i]}"
            f"{result.importances_mean[i]:.3f}"
            f" +/- {result.importances_std[i]:.3f}")
        permutation_importance_score_columns.append(X_train_scaled.columns[i])
        permutation_importance_score.append(result.importances_mean[i])


# In[ ]:


permutation_importance_score = pd.DataFrame(permutation_importance_score, index=permutation_importance_score_columns,columns=['permutation_importance_score'])

# I want to see 0.019
feature_permutation_importance_score = permutation_importance_score[permutation_importance_score['permutation_importance_score'] >= 0.019]

feature_permutation_importance_score.reset_index(inplace=True)

feature_permutation_importance_score


# # Shap feature importance

# In[ ]:


import shap


# In[ ]:


# Create object that can calculate shap values
explainer = shap.TreeExplainer(model)

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(X_test_scaled)
feature_names = list(X_test_scaled.columns.values)


# In[ ]:


# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values[0], feature_names,plot_type='bar')


# In[ ]:


# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values[0], X_test_scaled)


# In[ ]:


feature_names = list(X_test_scaled.columns.values)
vals = np.abs(shap_values[0]).mean(0)
feature_importance_shap = pd.DataFrame(list(zip(feature_names, vals)), columns=['index','feature_importance_vals'])
feature_importance_shap.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
feature_importance_shap.reset_index(drop=True,inplace=True)
feature_importance_shap


# # mi_score

# In[ ]:


# mutual information
from sklearn.feature_selection import mutual_info_classif

discrete = []
for col in X_train_scaled.columns:
    discrete.append(X_train_scaled[col].dtype == int)

mi_score = pd.DataFrame(mutual_info_classif(X_train_scaled,y_train,discrete_features=discrete), index=X_train_scaled.columns, columns=['Mutual information'])
plt.rcParams["figure.figsize"] = (10,15)
plt.barh(np.arange(len(discrete)), mi_score['Mutual information'])
plt.yticks(ticks=np.arange(len(discrete)),labels=mi_score.index)
plt.xlabel('Mutual information')


# In[ ]:


# I want to see >= 0.019
feature = mi_score[mi_score['Mutual information'] >= 0.025]
feature_importance_mi_score = feature.reset_index()
feature_importance_mi_score


# # Conclusion

# In[ ]:


dfs = [feature_importance_score_rf,feature_permutation_importance_score,feature_importance_shap,feature_importance_mi_score]
feature_importance_conclusion = pd.concat(dfs, join='outer',axis=1)


# In[ ]:


# DataFrame with feature importance
feature_importance_conclusion

