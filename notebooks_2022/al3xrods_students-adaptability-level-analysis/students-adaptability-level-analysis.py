#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msgno
import warnings
warnings.filterwarnings('ignore')

sns.set()


# In[ ]:


df = pd.read_csv('../input/students-adaptability-level-in-online-education/students_adaptability_level_online_education.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


for col in df.columns :
    df[col] = df[col].astype('category')


# In[ ]:


df.dtypes


# In[ ]:


df.isna().sum()


# In[ ]:


df.Gender.value_counts()
## The gender column seems pretty balanced.


# In[ ]:


df['Adaptivity Level'].value_counts(normalize=True)
# The target column is very imbalaced. I'll take care of it later on.


# In[ ]:


df['Education Level'].value_counts()


# In[ ]:


df['Institution Type'].value_counts()


# In[ ]:


df.Device.value_counts(normalize=True)


# In[ ]:


df.groupby(['Education Level', 'Financial Condition'])['Age'].count().unstack()


# In[ ]:


test1 = df.groupby(['Education Level', 'Financial Condition'])['Age'].count().unstack()


# ##### It would be interesting to test if there exists a statistical significance between Level of Education & Financial Condition. To test this, I will be importing the chi2_contingency function from scipy.stats 
# * Null Hypotesis : There is no statistical significance between these two categories.
# * Alternative Hypotesis : These two categories are in fact dependent
#  
# ##### I will chose the value of 0.05 for the p-value for this test.

# In[ ]:


from scipy import stats
_, p_val, _, _ = stats.chi2_contingency(test1)

if p_val < 0.05 :
    print('We can reject the null hypotesis')
else:
    print('We failed to reject the null hypotesis')


# In[ ]:


test2 = df.groupby(['Gender', 'Education Level'])['Age'].count().unstack()


# In[ ]:


_, p_val = stats.power_divergence(test2)

p_val.round(5)


# In[ ]:


# For online Education, studying in a PC is more comfortable, so here some reasonings :
# Mobiles are way more affordable and easy to use than PCs but are more adapted to online education.
# If the Financial Condition is Low, does that affects in what device are they stuying on ?
# Let's check.

test2 = df.groupby(['Device', 'Financial Condition'])['Age'].count().unstack()

_, p_val, _, _ = stats.chi2_contingency(test2)

if p_val < 0.05 :
    print('We can reject the null hypotesis')
else:
    print('We failed to reject the null hypotesis')


# In[ ]:


# Let's do the chi2_contingency for all columns.
for i in df.columns.tolist() :
    test = df.groupby([i, 'Adaptivity Level'])['Age'].count().unstack()
    _, p_val, _, _ = stats.chi2_contingency(test)
    if p_val.round(5) <=0.5:
        print(i, '- Adaptivity Level p-value:', p_val )


# In[ ]:


# How is Age range's boys & girls reflecting the Adaptivity Level ?
df.groupby(['Gender', 'Age', 'Adaptivity Level'])['Age'].count().unstack()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV as cv
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler 


# In[ ]:


X = df.drop('Adaptivity Level', axis=1)
y = df['Adaptivity Level']


# In[ ]:


rus = RandomOverSampler(random_state=10)
X_rus, y_rus = rus.fit_resample(X, y)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, stratify=y_rus, test_size= 0.3, random_state=0)


# In[ ]:


print('X_train shape:''\n' 'Rows:', X_train.shape[0],'\n' 'Columns:', X_train.shape[1])


# In[ ]:


lb = LabelEncoder()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)


# In[ ]:


ohe = OneHotEncoder(drop='first', sparse=False)
X_train = ohe.fit_transform(X_train)
X_test = ohe.transform(X_test)


# In[ ]:


logreg = LogisticRegression()
log_params = {'penalty':['l1', 'l2'], 'solver':['lbfgs', 'liblinear']}
cv = GridSearchCV(logreg, log_params, cv=5)
cv.fit(X_train, y_train)
cv.best_score_, cv.best_params_


# In[ ]:


y_pred = cv.predict(X_test)
print(classification_report(y_pred, y_test))


# In[ ]:


svc = SVC()
svc_params = {'kernel':['linear', 'poly', 'rbf']}
cv = GridSearchCV(svc, svc_params, cv=5)
cv.fit(X_train, y_train)


# In[ ]:


y_pred = cv.predict(X_test)
print(classification_report(y_pred, y_test))


# In[ ]:


knc = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_pred = knc.predict(X_test)
print(classification_report(y_pred, y_test))


# In[ ]:


'''Neighborhood Components Analysis (NCA) aims to improve the accuracy of nearest neighbors classification compared to the standard Euclidean distance.

Combined with a nearest neighbors classifier, NCA is attractive for classification because it can naturally handle multi-class problems 
without any increase in the model size, and does not introduce additional parameters that require fine-tuning .

Let's check if there is really a substantially improvement'''

knc = KNeighborsClassifier(n_neighbors=3)
nca = NeighborhoodComponentsAnalysis(random_state=10)
pipeline = Pipeline([
    ('nca', nca), 
    ('knc', knc)
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_pred, y_test))


# In[ ]:


bagg = BaggingClassifier(base_estimator= knc)
params = {'max_features':np.linspace(0.1, 1, 10), 'max_samples':np.linspace(0.1, 0.2, 10)}
cv = GridSearchCV(bagg, params, cv=5)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print(classification_report(y_pred, y_test))


# In[ ]:


ada = AdaBoostClassifier()
params = {'base_estimator':[svc, knc, logreg], 'n_estimators':[50, 60, 70, 80], 'learning_rate':np.linspace(0.1, 0.2, 10)}
cv = cv(ada, params, cv=3)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)
print(classification_report(y_pred, y_test))


# In[ ]:


# HistGradientBoostingClassifier can be orders of magnitude faster than GradientBoostingClassifier when the number of samples is larger than tens of thousands of samples.
hgbc = HistGradientBoostingClassifier(loss='categorical_crossentropy')
hgbc.fit(X_train, y_train)
y_pred = hgbc.predict(X_test)
print(classification_report(y_pred, y_test))

