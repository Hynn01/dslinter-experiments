#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# # READ DATASET AND BASIC ANALYSIS

# In[ ]:


df = pd.read_csv("../input/autismdiagnosis/Autism_Prediction/train.csv", encoding='windows-1252')
test_df = pd.read_csv('../input/autismdiagnosis/Autism_Prediction/test.csv')


# In[ ]:


df.head()


# In[ ]:


df.isna().sum()


# **No Null datapoints are present**

# In[ ]:


ProfileReport(df)


# # EDA

# In[ ]:


sns.set(style="darkgrid")


f, (ax_1, ax_2) = plt.subplots(2,1, sharex=True, gridspec_kw={"height_ratios": (.2,.8)})
sns.boxplot(df[df['Class/ASD']==1]['age'], ax=ax_1)
# sns.boxplot(df[df['Class/ASD']==0]['age'], ax=ax_2)
# sns.histplot(df[df['Class/ASD']==1], x=df['age'], ax=ax_2, kde=True, color="#ea4335")
sns.histplot(x=df['age'], ax=ax_2, kde=True, color="#4285f4", hue=df['Class/ASD'])
plt.legend(title='', loc=2, labels=['Heart Disease', 'No HeartDisease'],bbox_to_anchor=(1.02, 1), borderaxespad=0.)
plt.tight_layout()
plt.show()


# In[ ]:


sns.histplot(x=df['austim'] , kde=True, color="#4285f4", hue=df['Class/ASD'])


# In[ ]:


df.groupby('contry_of_res')['Class/ASD'].count().nlargest(10).plot.bar()


# In[ ]:


fig = plt.figure(figsize=(16,12))

cat_features=df.select_dtypes(include='object').columns.to_list()
cat_features_1 = [item for item in cat_features if df[item].nunique() < 10]
size=len(cat_features_1)
n_rows=int(np.round(size/2)+1)

for i,feature in zip(np.arange(1,size), cat_features_1):
    ax = fig.add_subplot(n_rows,2,i)
    sns.countplot(df[feature], ax=ax, linewidth= 3, alpha= 0.5, hue=df['Class/ASD'])
    
plt.tight_layout()


# **Chances are high when the person has autism for target variable to be 1**

# In[ ]:


sns.histplot(data=df, x="age", hue= 'Class/ASD')


# **People between 20 and 40 have higher chances of being positive though the datapoints for higher ages are less.**

# In[ ]:


plt.figure(figsize=(10,6))
sns.scatterplot(x='age', y='result', data=df, hue='Class/ASD', alpha=0.7, size='austim', palette='Set2')


# **The result variable have higher value when Class/ASD is 1**

# In[ ]:


plt.figure(figsize=(14,8))

sns.set_theme(style="whitegrid")

plt.subplot(2,1,1)
sns.violinplot(x=df['contry_of_res'], y=df['result'], order=df['contry_of_res'].value_counts().iloc[:10].index, hue=df['Class/ASD'])
plt.xticks(rotation=90)
plt.title('Distribution of Result feature across most featured countries')

plt.subplot(2,1,2)
sns.violinplot(x=df['ethnicity'], y=df['result'], order=df['ethnicity'].value_counts().iloc[:10].index, hue=df['Class/ASD'])
plt.xticks(rotation=90)
plt.title('Distribution of Result feature across ethnicities')
plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(8,14))

num_features=df.select_dtypes(exclude='object').columns.to_list()[1:]

corr = df[num_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, annot=True, cmap="Blues",
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# **The A score featues are highly correlated with each other.**
# 
# **Class/ASD have high correlation with A4,A6**

# In[ ]:


fig, axs = plt.subplots(3,4, figsize=(16, 8), facecolor='w', edgecolor='k')
plt.suptitle(f'Distribution of Class/ASD wrt Numerical Features')

for feature,ax in zip(num_features,axs.ravel()):
    sns.violinplot(data=df,x=feature, y='Class/ASD', palette="light:g", inner="points", orient="h",ax=ax, split=True)
    plt.subplots_adjust(hspace = 1,wspace=.001)
    sns.despine(offset=10, trim=True);

    
plt.tight_layout()


# In[ ]:


fig, axs = plt.subplots(2,3, figsize=(16, 8), facecolor='w', edgecolor='k')
plt.suptitle(f'Distribution of Result wrt Categorical Features')

for feature,ax in zip(cat_features_1,axs.ravel()):
    sns.boxplot(data=df, x=feature, y='result', palette="light:g",  ax=ax)
    plt.subplots_adjust(hspace = 1,wspace=.051)
    sns.despine(offset=10)
    plt.xticks(rotation=90)

    
plt.tight_layout()


# # PREPROCESSING OF DATA

# **Removal of the outliers for numerical columns age and result using IQR**

# In[ ]:


# feat_to_remove_outl=['age','result']
# for feat in feat_to_remove_outl:
#     Q1 = df[feat].quantile(0.25)
#     Q3 = df[feat].quantile(0.75)
#     IQR = Q3 - Q1
#     df = df[~((df[feat] < (Q1 - 1.5 * IQR)) |(df[feat] > (Q3 + 1.5 * IQR)))]


# **Outliers removed for the features age and result**

# **38 rows have relation marked as ? and 193 rows have ethnicity marked as ?. Rather than droping these rows, we fill fill these with the most frequent occuring values in each of the respective features**

# In[ ]:


df['ethnicity'] = df['ethnicity'].str.replace('others','Others')
df['ethnicity'] = df['ethnicity'].str.replace('?','Others')


# In[ ]:


test_df['ethnicity'] = test_df['ethnicity'].str.replace('?','Others')


# In[ ]:


df['relation'] = df['relation'].str.replace('?','Others')


# In[ ]:


test_df['relation'] = test_df['relation'].str.replace('?','Others')


# In[ ]:


columns=['gender', 'jaundice', 'austim', 'used_app_before','age_desc', 'ethnicity','relation']

for col in columns:
    dif= df[col].nunique() - test_df[col].nunique()
    print(f'for column {col} : {dif}')


# **Values are replaced with the most commonly occuring values in featues relation and ethnicity**
# 
# **Next up, we have to encode the categorical features using one hot encoding for the model to perform better.**

# In[ ]:



df = pd.get_dummies(df, columns=[ 'ethnicity','relation', 'jaundice','gender','austim', 'used_app_before'])
test_df = pd.get_dummies(test_df, columns=[ 'ethnicity','relation','jaundice','gender','austim', 'used_app_before'])


# In[ ]:





# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df.contry_of_res = le.fit_transform(df.contry_of_res)

test_df.contry_of_res = le.fit_transform(test_df.contry_of_res)


# In[ ]:


df=df.drop(['ID', 'age_desc'], axis=1)
test_df=test_df.drop(['ID'], axis=1)


# In[ ]:


x_train=df.drop('Class/ASD', axis=1)
y_train=df['Class/ASD']
test_df=test_df.drop(['age_desc'], axis=1)


# In[ ]:


x_test=test_df


# In[ ]:


# from sklearn.preprocessing import StandardScaler 

# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# test_df =sc.transform(test_df)


# In[ ]:


from imblearn.over_sampling import SMOTE


sm = SMOTE(random_state=2)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())


# # MODEL COMPARISONS

# In[ ]:


from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier


from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay,RocCurveDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score


# In[ ]:


models=[]
model_score=[]
models.append(['Logistic Regression', LogisticRegressionCV(cv=5, random_state=0)])
models.append(['KNeighborsClassifier', KNeighborsClassifier(n_neighbors=25, weights='distance')])
models.append(['GaussianNB', GaussianNB()])
models.append(['BernoulliNB', BernoulliNB()])
models.append(['DecisionTreeClassifier', DecisionTreeClassifier()])
models.append(['RandomForestClassifier', RandomForestClassifier(random_state=123)])
models.append(['XGBClassifier', XGBClassifier()])
models.append(['SVC',SVC(kernel='linear', degree=10)])
models.append(['Extra Tree Classifier',ExtraTreeClassifier(random_state=0)])
models.append(['Ada Boost Classifier',AdaBoostClassifier(random_state=0)])
models.append(['XGBClassifier',XGBClassifier()])


# In[ ]:


for model in models:
        model_data=[]
        model[1].fit(x_train_res, y_train_res)
        y_pred = model[1].predict(x_test)
#         acc_score = accuracy_score(y_test, y_pred)
#         report = classification_report(y_test,y_pred, output_dict=True)
#         conf_matrix = confusion_matrix(y_test,y_pred)
#         f1= f1_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         roc = roc_auc_score(y_test, y_pred)
        accuracies = cross_val_score(estimator = model[1], X = x_train, y = y_train, cv = 10)   #K-Fold Validation
        print('----------------------------------------------------------------------------------------------')
        print(f'{model[0]} :')
        print('---------------------------------------------')
#         print(f'Accuracy Score : {acc_score}')
#         print('')
        print(f'K Fold Accuracy : {accuracies}')
        print('')
#         print(f'F1 Score : {f1}')
#         print('')
#         print(f'Recall : {recall}')
#         print('')
#         print(f'ROC :{roc}')
#         print('')
#         print(f'Confusion Matrix : ')
#         print('')
#         print(f'{conf_matrix}')
#         print('')
        model_data.append(model[0])
        model_data.append(accuracies)
        model_score.append(model_data)
#         ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
#         RocCurveDisplay.from_predictions(y_test, y_pred)
#         plt.show()


# **Logistic Regression, RandomForestClassifier and SVC performed the best for the given dataset. Now we aim to find the best parameters for the mentioned regressors using GridSearchCV**

# # GRID SEARCH FOR THE BEST MODELS

# In[ ]:


# from sklearn.model_selection import GridSearchCV


# gridmodels=[(AdaBoostClassifier(),[{'n_estimators': [25,50,100,200,250], 'random_state':[0,10,50,100,200]}]),
#             (BernoulliNB(),[{'alpha': [0.25,0.50,1.00], 'n_features_':[3,5,10,20,25]}])
# ]


# In[ ]:


# for gridmodel in gridmodels:
#     grid = GridSearchCV(estimator=gridmodel[0],param_grid = gridmodel[1], scoring = 'accuracy',cv = 5)
#     grid.fit(x_train, y_train)
#     best_accuracy = grid.best_score_
#     best_param = grid.best_params_
#     print('{}:\nBest Accuracy : {:.2f}%'.format(gridmodel[0],best_accuracy*100))
#     print('Best Parameters : ',best_param)
#     print('')
#     print('----------------')
#     print('')


# # PREDICTION AND SUBMISSION

# In[ ]:


best_model= BernoulliNB()
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)


# In[ ]:


pd.DataFrame({'ID': np.arange(1,201), 'Class/ASD': y_pred})['Class/ASD'].value_counts()


# In[ ]:


submission = pd.DataFrame({'ID': np.arange(1,201), 'Class/ASD': y_pred})
submission.to_csv('submission.csv', index=False)


# In[ ]:





# In[ ]:




