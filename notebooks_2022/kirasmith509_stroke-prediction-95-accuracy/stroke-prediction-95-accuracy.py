#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data = pd.read_csv('../input/stroke-prediction-dataset/healthcare-dataset-stroke-data.csv')
data


# In[ ]:


data.drop(data[data['gender']=='Other'].index,inplace=True)


# In[ ]:


data.shape


# In[ ]:


data.smoking_status.value_counts()


# In[ ]:


data.shape


# In[ ]:


data.isnull().sum()


# In[ ]:


data.describe()


# In[ ]:


data.gender.replace({'Male':1,'Female':0},inplace = True)
data.ever_married.replace({'Yes':1,'No':0},inplace = True)
data.Residence_type.replace({'Urban':1,'Rural':0},inplace = True)


# In[ ]:


data


# In[ ]:


data.gender.value_counts().plot(kind = 'bar' )


# In[ ]:


data = pd.get_dummies(data, columns = ['work_type','smoking_status'])


# In[ ]:


data


# In[ ]:


cols = ['id','gender','age','hypertenstion','heart_disease','ever_marries','Residence_type','avg_glucose_level','bmi','stroke','work_type_Govt_job','work_type_Never_worked','work_type_Private','work_type_Self-employed','work_type_children','smoking_status_Unknown','smoking_status_formerly_smoked','smoking_status_never_smoked','smoking_status_smokes']


# In[ ]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.NaN,strategy='mean')
imp = imp.fit(data)
data = imp.transform(data)


# In[ ]:


data 


# In[ ]:


data = pd.DataFrame(data,columns = cols)


# In[ ]:


data


# In[ ]:





# In[ ]:


data.groupby('gender')['stroke'].sum().plot(kind = 'bar')


# In[ ]:


data.groupby('heart_disease')['stroke'].sum().plot(kind = 'bar')


# In[ ]:


data.heart_disease.value_counts()


# In[ ]:


plt.figure(figsize = (16,10))
sns.heatmap(data.corr(),annot = True,cmap='BuPu')
plt.show()


# In[ ]:


plt.figure(figsize=(15,6))
n = 0
for x in ['age','bmi','heart_disease']:
  n+=1
  plt.subplot(1,3,n)
  plt.subplots_adjust(hspace = 0.5,wspace = 0.5)
  sns.distplot(data[x],bins = 20)
  plt.title('distribution of {}'.format(x))


# In[ ]:


Age_18_25 = data.age[(data.age >= 18)& (data.age<=25)]
Age_26_35 = data.age[(data.age >= 26)& (data.age<=35)]
Age_36_45 = data.age[(data.age >= 36)& (data.age<=45)]
Age_46_55 = data.age[(data.age >= 46)& (data.age<=55)]
Age_above_55 = data.age[data.age >= 56]


# In[ ]:


agex = ['Age_18_25','Age_26_35','Age_36_45','Age_46_55','Age_above_55']
agey = [len(Age_18_25.values),len(Age_26_35.values),len(Age_36_45.values),len(Age_46_55.values),len(Age_above_55.values)]


# In[ ]:


plt.figure(figsize=(15,6))
sns.barplot(x = agex,y = agey , palette='mako')
plt.title = (" Range of age ")
plt.xlable = (" Range of age ")
plt.ylabel = ('No of Customers')
plt.show()


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix
from xgboost import XGBClassifier


# In[ ]:


y = data['stroke']
X = data.drop('stroke',axis = 1)


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


score = {}
pipe = make_pipeline(StandardScaler(),LogisticRegression())
pipe.fit(X_train,y_train)
score['logistic_Regression'] = pipe.score(X_test,y_test)


# In[ ]:


pipe = make_pipeline(StandardScaler(),KNeighborsClassifier())
pipe.fit(X_train,y_train)
score['KNN'] = pipe.score(X_test,y_test)


# In[ ]:


pipe = make_pipeline(StandardScaler(),RandomForestClassifier())
pipe.fit(X_train,y_train)
score['RandomForest'] = pipe.score(X_test,y_test)


# In[ ]:


pipe = make_pipeline(StandardScaler(),XGBClassifier())
pipe.fit(X_train,y_train)
score['XGB'] = pipe.score(X_test,y_test)


# In[ ]:


score


# In[ ]:


scaler = StandardScaler()
scaled_X_test = scaler.fit_transform(X_test)


# In[ ]:


scaled_X_train = scaler.fit_transform(X_train)


# In[ ]:


X_train


# In[ ]:


model = RandomForestClassifier()
model.fit(X_train,y_train)


# In[ ]:


y_pred = model.predict(scaled_X_test)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


plot_confusion_matrix(model,scaled_X_test,y_test);


# In[ ]:


print(classification_report(y_test,y_pred))

