#!/usr/bin/env python
# coding: utf-8

# # YouTube Trending Videos Analysis
# 
# This notebook covers a small part of exploration. Includes a way to merge the csv files with the category json files.   
# With all the data together in the same dataframe, the idea is to practice different tasks, like classification, regression, clustering and recommendation, that can be done with this dataset, and testing different models for each problem.
# 
# ## Table of contents:
#  
# 1. Imports
# 2. Joining files
# 3. Feature Exploration
# 4. Category Classification 

# ## 1. Prepare Data and Import Libraries

# In[ ]:


import os
import re
import string
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import datetime
import nltk
from collections import Counter
#Procesado
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline


#Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization,                                    LeakyReLU, Add, GlobalAveragePooling1D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import RootMeanSquaredError
import xgboost

from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,roc_auc_score,                            r2_score, mean_squared_error, recall_score, precision_recall_curve


# In[ ]:


get_ipython().system('mkdir json/ videos/')
get_ipython().system('cp ../input/youtube-new/*.json json/')
get_ipython().system('cp ../input/youtube-new/*videos.csv videos/')


# ## 2. Join video and category

# In[ ]:


path='videos/'
videos = pd.DataFrame()
for file in os.listdir(path=path):
    try:
        video = pd.read_csv(path+file,encoding='utf-8',index_col='video_id')
        video['country'] = file[:2]
        videos=videos.append(video)

    except:
        ## Hay csv que no se pueden leer con uft-8
        video = pd.read_csv(path+file,encoding='latin-1',index_col='video_id')
        video['country']=file[:2]
        videos=videos.append(video)

videos['videos_id'] = videos.index.values.reshape(-1,1)


# In[ ]:


path = 'json/'
categories=pd.DataFrame()
for file in os.listdir(path=path):
    with open(path+file) as f:
        data = pd.DataFrame(json.load(f)['items'])
        id = pd.Series(data['id'],dtype=int)
        assignable = data['snippet'].apply(lambda x: x['assignable'])
        title = data['snippet'].apply(lambda x: x['title'])

        category = pd.concat([id,assignable,title],axis=1)
        category.columns = ['category_id', 'assignable', 'snippet']
        category['country'] = file[:2]

        categories = categories.append(category)


# In[ ]:


df = videos.merge(categories,on=['category_id','country'])


# ## 3. Feature Exploration

# ### Correlations and Class Balance
# 

# In[ ]:


print("Duplicated Rows: {}".format(df.shape[0]-df.drop_duplicates().shape[0]))

df = df.drop_duplicates()


# In[ ]:


unique_videos = df[['title','videos_id']].drop_duplicates()
print("Size of the original dataset: {}".format(len(df)))
print("Number of unique videos in the dataset: {}".format(len(unique_videos)))

#Some videos are repeated, but with different trending dates

titles_count = df['title'].value_counts()
df['trending_time'] = df.apply(lambda x: titles_count[x['title']],axis=1)
# Trending_time is the number of days in trending.


# In[ ]:


df.isna().sum()
# 48325 missing values in 'description'


# In[ ]:


print(df.columns)

columns_floats = ['views', 'likes', 'dislikes','comment_count','comments_disabled', 'ratings_disabled']
columns_dates = ['trending_date','publish_time']


# In[ ]:


#Float variables correlations
sns.set(rc={"figure.figsize":(10, 8)})
sns.heatmap(df[columns_floats].corr(),annot=True)


# In[ ]:


sns.countplot(data=df,x='snippet')
plt.xticks(rotation= 90)


# In[ ]:


# Some categories has low representation, could be eliminated.

#categories = df['snippet'].value_counts()[:13].index
#print(categories)

#df = df.query('snippet in @categories')


# ### Feature Engineer

# In[ ]:


def DefineDatetimeVariables(DF):
    DF['Month'] = DF['trending_date'].apply(lambda x:datetime.datetime.strptime(x,'%y.%d.%m').month)
    DF['DifferTime'] = DF.apply(lambda x:(datetime.datetime.strptime(x.trending_date,'%y.%d.%m')-datetime.datetime.strptime(x.publish_time,'%Y-%m-%dT%H:%M:%S.000Z')).days,axis=1)
    DF['Hora'] = DF['publish_time'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%dT%H:%M:%S.000Z').hour)

DefineDatetimeVariables(df)

# Month as a ordinal veriable, not categorical.


# I change likes and dislikes for it's ratio and total the number of 'interactions': likes+dislikes

# In[ ]:


def TransformLikes(DF):
    DF['liked'] = DF.apply(lambda x: x.likes/(x.likes+x.dislikes+1e-10),axis=1)
    DF['ratings']= DF.apply(lambda x: x.likes+x.dislikes,axis=1)

TransformLikes(df)


# In[ ]:


columns_floats = ['views',
 #'likes',
 #'dislikes',
 'liked',
 'ratings',
 'comments_disabled',
 'comment_count',
 'ratings_disabled',
 'Month',
 'DifferTime',
 'Hora']

df[columns_floats].corr()


# ### Tokenizer
# 
# Remove punctuation.
# Remove Stopwords

# In[ ]:


def remove_punc(text):
    text = text.replace('|',' ')
    text = "".join([chart for chart in text if chart not in string.punctuation])

    return text

df['tags_clean'] = df['tags'].apply(lambda x: remove_punc(x))


# In[ ]:


def tokenizer(text):
    tokens = re.split('\W+',text)
    return tokens

df['tokens'] = df['tags_clean'].apply(lambda x: tokenizer(x.lower()))


# #### Remove Stopwords and numbers

# In[ ]:


nltk.download('stopwords')
stopwords_en = nltk.corpus.stopwords.words('english')
stopwords_sp = nltk.corpus.stopwords.words('spanish')

stopwords = stopwords_en + stopwords_sp


# In[ ]:


def remove_stopwords(tokens):
    tokens = [token for token in tokens if token not in stopwords]
    return tokens

df['tokens'] = df['tokens'].apply(lambda x: remove_stopwords(x))


# In[ ]:


def remove_numb(tokens):
    tokens = [token for token in tokens if token.isalpha()]
    return tokens

df['tokens'] = df['tokens'].apply(lambda x: remove_numb(x))


# #### Count words

# In[ ]:


unique_videos = df[['title','videos_id']].drop_duplicates()

c = Counter()
def counter(text):
      c.update(text)
df.loc[unique_videos.index,'tokens'].apply(lambda x: counter(x))

common_words_1 = [word[0] for word in c.most_common()[:300]]


# In[ ]:


mono_words = list()
for word in common_words_1:
    if len(word)==1:
        mono_words.append(word)

for word in mono_words:
    common_words_1.remove(word)


# ### Create features
# 
# For each most common word, I create a new feature in the dataset, that counts the how many times this word appears in tag. 

# In[ ]:


for word in common_words_1:
    df[word] = df['tokens'].apply(lambda x: x.count(word)) 


# # 4. Category Classification Model
# Neural Network

# ### General Preprocessing
# 

# In[ ]:


unique_videos = df[['title','snippet']].drop_duplicates()

X_train,X_test = train_test_split(unique_videos,test_size=0.2,stratify=unique_videos['snippet'])

columns  = ['views',
 'ratings',
 'liked',
 'trending_time',
 'comment_count',
 'comments_disabled',
 'ratings_disabled',
 'Month',
 #'DifferTime',
 'Hora'] + common_words_1

#train_titles = X_train['title']
#test_titles = X_test['title']
#X_train = df.query('title in @train_titles')[columns]
#X_test = df.query('title in @test_titles')[columns]
X_train = df.loc[X_train.index,columns]
X_test = df.loc[X_test.index,columns]

countries_train = df.loc[X_train.index,'country']
countries_test = df.loc[X_test.index,'country']

oh_country = OneHotEncoder(sparse=False)

countries_train = pd.DataFrame(oh_country.fit_transform(countries_train.values.reshape(-1,1)),index = countries_train.index)
countries_test = pd.DataFrame(oh_country.transform(countries_test.values.reshape(-1,1)),index = countries_test.index)
X_train = pd.concat([X_train,countries_train],axis=1)
X_test = pd.concat([X_test,countries_test],axis=1)

y_train = df.loc[X_train.index,'snippet']
y_test = df.loc[X_test.index,'snippet']


# In[ ]:


lab_encoder = OneHotEncoder(sparse=False)

y_train_oh = lab_encoder.fit_transform(y_train.values.reshape(-1,1))
y_test_oh = lab_encoder.transform(y_test.values.reshape(-1,1))


# In[ ]:


columns_scaled  = ['views',
 'ratings',
 'liked',
 'trending_time',
 'comment_count',
 'Month',
 #'DifferTime',
 'Hora'] #+ common_words_1

scaler = StandardScaler()

X_train[columns_scaled] = scaler.fit_transform(X_train[columns_scaled])
X_test[columns_scaled] = scaler.transform(X_test[columns_scaled])

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

X_train[common_words_1] = minmax.fit_transform(X_train[common_words_1])
X_test[common_words_1] = minmax.transform(X_test[common_words_1])


# In[ ]:


def ShowScores(y_test,y_pred,tikslabels='auto'):

    plt.figure(figsize = (20,6))
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,linewidths=.3,yticklabels=tikslabels,xticklabels=tikslabels)

    print(f"Accuracy Score: {accuracy_score(y_test,y_pred)}")


#  ### Random Forest Classifier

# In[ ]:


rf = RandomForestClassifier(n_estimators=500,max_depth=25,max_features='log2',verbose=1,criterion='gini')

rf.fit(X_train,y_train)


# In[ ]:


pred= rf.predict(X_test)
ShowScores(y_test,pred,np.unique(y_test))


# In[ ]:


depths = [tree.tree_.max_depth for tree in rf.estimators_]
print(f"Mean tree depth in the Random Forest: {np.round(np.mean(depths))}")
print(f"Number Features Seen: {rf.estimators_[0].n_features_in_}")


# ### Neural Network (Keras)
# 

# In[ ]:


def build_model(optimizer='adam', lr=0.001):
    inp = Input(shape=(X_train.shape[-1]))
    x = Dense(256,activation='relu')(inp)
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.2)(x)
    #x=Dense(512,activation='relu')(x)
    #x=Dropout(0.2)(x)
    x=Dense(1024,activation='relu')(x)
    x=Dropout(0.2)(x)
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.2)(x)
    x=Dense(128,activation='relu')(x)
    x=Dropout(0.2)(x)
    x=Dense(64,activation='relu')(x)
    x=Dense(18,activation='softmax')(x)

    model = Model(inputs=inp,outputs=x)

    if optimizer=='adam':
        optimizer_model = Adam(learning_rate=lr)
    elif optimizer == 'rmsprop':
        optimizer_model = RMSprop(learning_rate=lr)

    model.compile(loss='categorical_crossentropy',optimizer=optimizer_model,metrics=['accuracy'])

    return model


# In[ ]:


model = build_model(optimizer='adam',lr=0.001)


# In[ ]:


model.fit(x=X_train*1,
          y=y_train_oh,
          batch_size=128,
          epochs=80,
          validation_data=(X_test*1,y_test_oh))

model.fit(x=X_train*1,
          y=y_train_oh,
          batch_size=512,
          epochs=5,
          validation_data=(X_test*1,y_test_oh))


# In[ ]:


pred= model.predict(X_test*1)
ShowScores(np.argmax(y_test_oh,axis=1),np.argmax(pred,axis=1),np.unique(y_test))


# In[ ]:




