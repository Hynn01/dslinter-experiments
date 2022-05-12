#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')


# In[ ]:


df.head()


# In[ ]:


# One review
df1['review'][0]


# ## Text Cleaning
# 
# 1. Sample 10000 rows
# 2. Remove html tags
# 3. Remove special characters
# 4. Converting every thing to lower case
# 5. Removing Stop words
# 6. Stemming

# In[ ]:


df=df.sample(10000)


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df['sentiment'].replace({'positive':1,'negative':0},inplace=True)


# In[ ]:


df.head()


# In[ ]:


import re
clean = re.compile('<.*?>')
re.sub(clean, '', df.iloc[2].review)


# In[ ]:


# Function to clean html tags
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


# In[ ]:


df['review']=df['review'].apply(clean_html)


# In[ ]:


# converting everything to lower

def convert_lower(text):
    return text.lower()


# In[ ]:


df['review']=df['review'].apply(convert_lower)


# In[ ]:


# function to remove special characters

def remove_special(text):
    x=''
    
    for i in text:
        if i.isalnum():
            x=x+i
        else:
            x=x + ' '
    return x


# In[ ]:


remove_special(' th%e @ classic use of the word.it is called oz as that is the nickname given to the oswald maximum security state penitentary. it focuses mainly on emerald city, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. em city is home to many..aryans, muslims, gangstas, latinos, christians, italians, irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.i would say the main appeal of the show is due to the fact that it goes where other shows wouldnt dare. forget pretty pictures painted for mainstream audiences, f')


# In[ ]:


df['review']=df['review'].apply(remove_special)


# In[ ]:


# Remove the stop words
import nltk


# In[ ]:


from nltk.corpus import stopwords


# In[ ]:


stopwords.words('english')


# In[ ]:


df


# In[ ]:



def remove_stopwords(text):
    x=[]
    for i in text.split():
        
        if i not in stopwords.words('english'):
            x.append(i)
    y=x[:]
    x.clear()
    return y


# In[ ]:


df['review']=df['review'].apply(remove_stopwords)


# In[ ]:


df


# In[ ]:


# Perform stemming

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[ ]:


y=[]
def stem_words(text):
    for i in text:
        y.append(ps.stem(i))
    z=y[:]
    y.clear()
    return z
        


# In[ ]:


stem_words(['I','loved','loving','it'])


# In[ ]:


df['review']=df['review'].apply(stem_words)


# In[ ]:


df


# In[ ]:


# Join back

def join_back(list_input):
    return " ".join(list_input)
    


# In[ ]:


df['review']=df['review'].apply(join_back)


# In[ ]:


df['review']


# In[ ]:


X=df.iloc[:,0:1].values


# In[ ]:


X.shape


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1000)


# In[ ]:


X=cv.fit_transform(df['review']).toarray()


# In[ ]:


X.shape


# In[ ]:


X[0].mean()


# In[ ]:


y=df.iloc[:,-1].values


# In[ ]:


y.shape


# In[ ]:


# X,y
# Training set
# Test Set(Already know the result)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# In[ ]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB


# In[ ]:


clf1=GaussianNB()
clf2=MultinomialNB()
clf3=BernoulliNB()


# In[ ]:


clf1.fit(X_train,y_train)
clf2.fit(X_train,y_train)
clf3.fit(X_train,y_train)


# In[ ]:


y_pred1=clf1.predict(X_test)
y_pred2=clf2.predict(X_test)
y_pred3=clf3.predict(X_test)


# In[ ]:


y_test.shape


# In[ ]:


y_pred1.shape


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


print("Gaussian",accuracy_score(y_test,y_pred1))
print("Multinomial",accuracy_score(y_test,y_pred2))
print("Bernaulli",accuracy_score(y_test,y_pred3))


# In[ ]:




