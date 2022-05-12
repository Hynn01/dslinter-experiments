#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


Humor = pd.read_csv('../input/200k-short-texts-for-humor-detection/dataset.csv')
Humor.head()


# In[ ]:


Humor.info()


# In[ ]:


Humor['Length'] = Humor['text'].apply(lambda x: len(x.split(' ')))
Humor['Length'].describe()


# In[ ]:


#EDA 


# In[ ]:


plt.figure(figsize=(10,5))
sns.barplot(x='humor',y='Length',data=Humor)
plt.show()


# In[ ]:


Word_list = []
Word_dic = dict()
for i in list(Humor['text']):
    for j in i.split(' '):
        Word_list.append(j.upper())
len(Word_list)


# In[ ]:


from collections import Counter


# In[ ]:


Word_dic = Counter(Word_list)
Word_df = pd.DataFrame({'Word':list(Word_dic.keys()),'Frequency':list(Word_dic.values())})
Word_df.sort_values('Frequency',ascending=False,inplace=True)
Word_df.head()


# In[ ]:


sync = list(Word_df['Word'].head(60))
sync.remove('TRUMP')
sync.remove('PEOPLE')


# In[ ]:


len(Word_df)


# In[ ]:


Word_df['Frequency'].describe()


# In[ ]:


for i in [0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99]:
    print(Word_df['Frequency'].quantile(i))


# In[ ]:


for i in [0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.998,0.999]:
    print(Word_df['Frequency'].quantile(i))


# In[ ]:


#Filtering out only the top 0.8% of words based on frequency.


# In[ ]:


Word_df = Word_df[Word_df.Frequency>=333.0]
len(Word_df)


# In[ ]:


Word_df.head()


# In[ ]:


#As well as removing syncategoromic words.


# In[ ]:


Final_Words = list(Word_df.Word)
Final = [x for x in Final_Words if x not in sync]
len(Final)


# In[ ]:


#Clearing up other useless characters/terms in the final list.


# In[ ]:


for i in ['','-','--','&','3','7','2','A:','6','Q:','...','?','5','10','!','1','9','4','8']:
    Final.remove(i)


# In[ ]:


len(Final)


# In[ ]:


def upp(lis):
    return [j.upper() for j in lis]
    
for i in Final:
    Humor[i] = Humor['text'].apply(lambda x: 1 if i in upp(x.split(' ')) else 0)
Humor.drop(['text'],axis=1,inplace=True)
Humor.head()


# In[ ]:


len(Humor.columns)


# In[ ]:


Humor['humor'] = Humor['humor'].apply(lambda x: 1 if x==True else 0)
Humor['humor'].value_counts()


# In[ ]:


#Now we will go ahead and model the target variable humor based on the words.


# In[ ]:


Humor.head()


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X = Humor.drop('humor',axis=1)
y = Humor['humor']
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.01,random_state=100)
X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


#Taking a small sample for training and rest for testing.


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score


# In[ ]:


Model = RandomForestClassifier(random_state=100,n_jobs=-1)

params = {'n_estimators':[200],
          'max_depth':[10,15,20,25,30,35],
          'max_features':[0.2,0.25,0.3,0.35,0.4,0.45],
          'criterion':['gini','entropy']}

grid_search = GridSearchCV(estimator=Model,param_grid=params,verbose=1,n_jobs=-1,scoring='accuracy')
grid_search.fit(X_train,y_train)


# In[ ]:


Model_best = grid_search.best_estimator_


# In[ ]:


y_train_pred = Model_best.predict(X_train)
y_test_pred = Model_best.predict(X_test)

print('Train accuracy score :',accuracy_score(y_train,y_train_pred))
print('Test accuracy score :',accuracy_score(y_test,y_test_pred))
print('Train recall score :',recall_score(y_train,y_train_pred))
print('Test recall score :',recall_score(y_test,y_test_pred))


# In[ ]:


Feature_Importance = pd.DataFrame({'Feature':X_train.columns,'Importance':Model_best.feature_importances_})
Feature_Importance.sort_values(by='Importance',ascending=False,inplace=True)
Feature_Importance.set_index('Feature',inplace=True)
Feature_Importance.head()


# In[ ]:


#Highest Importances


# In[ ]:


temp = Feature_Importance.head(20)
plt.figure(figsize=(15,5))
sns.barplot(x=temp.Importance,y=temp.index)
plt.title('Features vs Importances')
plt.show()


# In[ ]:


#Lowest Importances


# In[ ]:


temp = Feature_Importance.tail(20)
plt.figure(figsize=(15,5))
sns.barplot(x=temp.Importance,y=temp.index)
plt.title('Features vs Importances')
plt.show()


# In[ ]:


# As we can see the accuracy and recall scores for the model are mediorcre with accuracy doing abit better. However form the plot of
#feature importances we can see that none of the words have an importance with respect to predicting humor. 


# In[ ]:


#Now we will try gradient boosting.


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


Model = GradientBoostingClassifier(random_state=100)

params = {'learning_rate':[0.1,0.2,0.3],
          'max_depth':[10,15,20,25,30],
          'max_features':[0.1,0.2,0.25,0.3]}

grid_search = GridSearchCV(estimator=Model,param_grid=params,verbose=1,n_jobs=-1,scoring='accuracy')
grid_search.fit(X_train,y_train)


# In[ ]:


Model_best = grid_search.best_estimator_


# In[ ]:


y_train_pred = Model_best.predict(X_train)
y_test_pred = Model_best.predict(X_test)

print('Train accuracy score :',accuracy_score(y_train,y_train_pred))
print('Test accuracy score :',accuracy_score(y_test,y_test_pred))
print('Train recall score :',recall_score(y_train,y_train_pred))
print('Test recall score :',recall_score(y_test,y_test_pred))


# In[ ]:


Feature_Importance = pd.DataFrame({'Feature':X_train.columns,'Importance':Model_best.feature_importances_})
Feature_Importance.sort_values(by='Importance',ascending=False,inplace=True)
Feature_Importance.set_index('Feature',inplace=True)
Feature_Importance.head()


# In[ ]:


#Highest Importances


# In[ ]:


temp = Feature_Importance.head(20)
plt.figure(figsize=(15,5))
sns.barplot(x=temp.Importance,y=temp.index)
plt.title('Features vs Importances')
plt.show()


# In[ ]:


#Lowest Importances


# In[ ]:


temp = Feature_Importance.tail(20)
plt.figure(figsize=(15,5))
sns.barplot(x=temp.Importance,y=temp.index)
plt.title('Features vs Importances')
plt.show()


# In[ ]:


#The scores obtained using gradient boosting are better than RFs however there is some overfitting. 


# In[ ]:


# END FOR NOW #

