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
# 5. Comment Classification (unbalanced problem)
# 6. Likes Regression
# 7. Ratio Likes/Dislikes Regression
# 8. Clustering
# 9. Video Recommendation

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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

#Regression
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#Clustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.neighbors import NearestNeighbors

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization,                                    LeakyReLU, Add, GlobalAveragePooling1D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import RootMeanSquaredError
import xgboost

from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,roc_auc_score,                            r2_score, mean_squared_error, recall_score, precision_recall_curve


# In[ ]:


get_ipython().system('mkdir json/ videos/')
get_ipython().system('cp ../input/youtube-trending-video-dataset/*.json json/')
get_ipython().system('cp ../input/youtube-trending-video-dataset/*data.csv videos/')


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


videos['category_id'] = videos['categoryId']


# In[ ]:


df = videos.merge(categories,on=['category_id','country'])


# ## 3. Feature Exploration

# ### Correlations and Class Balance
# 

# In[ ]:


print(df.shape[0]-df.drop_duplicates().shape[0])
# 1222 duplicated rows

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

columns_floats = ['view_count', 'likes', 'dislikes','comment_count','comments_disabled', 'ratings_disabled']
columns_dates = ['trending_date','publish_time']


# In[ ]:


#Float variables correlations
sns.heatmap(df[columns_floats].corr(),annot=True)


# In[ ]:


sns.countplot(data=df,x='snippet')
plt.xticks(rotation= 90)


# In[ ]:


# Algunas categorías tienen muy poca representación.
# Se eliminan del dataset.
# Otras opciones: Juntar categorías (ej: Trailers+Movies, Shows+Entertainment) para aumentar la representación.
#categories = df['snippet'].value_counts()[:13].index
#print(categories)

#df = df.query('snippet in @categories')


# ### Feature Engineer

# In[ ]:


def DefineDatetimeVariables(DF):
    DF['Month'] = DF['trending_date'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%dT%H:%M:%SZ').month)
    DF['DifferTime'] = DF.apply(lambda x:(datetime.datetime.strptime(x.trending_date,'%Y-%m-%dT%H:%M:%SZ')-datetime.datetime.strptime(x.publishedAt,'%Y-%m-%dT%H:%M:%SZ')).days,axis=1)
    DF['Hora'] = DF['publishedAt'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%dT%H:%M:%SZ').hour)

DefineDatetimeVariables(df)

# Month as a ordinal veriable, not categorical.


# I change likes and dislikes for it's ratio and total the number of 'interactions': likes+dislikes

# In[ ]:


def TransformLikes(DF):
    DF['liked'] = DF.apply(lambda x: x.likes/(x.likes+x.dislikes+1e-10),axis=1)
    DF['ratings']= DF.apply(lambda x: x.likes+x.dislikes,axis=1)

TransformLikes(df)


# In[ ]:


columns_floats = ['view_count',
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

columns  = ['view_count',
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


columns_scaled  = ['view_count',
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
    x=Dense(15,activation='softmax')(x)

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
          epochs=20,
          validation_data=(X_test*1,y_test_oh))

model.fit(x=X_train*1,
          y=y_train_oh,
          batch_size=512,
          epochs=2,
          validation_data=(X_test*1,y_test_oh))


# In[ ]:


pred= model.predict(X_test*1)
ShowScores(np.argmax(y_test_oh,axis=1),np.argmax(pred,axis=1),np.unique(y_test))


# # 5. Coment Classification Model
# 
# Random Forest

# #### Utils

# In[ ]:


def plot_results(X_test,y_test,model,umbral=0.5,plot=True,proba=False):
    print('Resultados con Umbral {}: '.format(umbral))
    if proba:
        probas = model.predict(X_test)
        pred = probas>umbral
    else:
        probas = model.predict_proba(X_test)[:,1]
        pred = probas>umbral

    print(confusion_matrix(y_test,pred*1))

    print('Precisión: ', precision_score(y_test,pred))
    print('Recall: ', recall_score(y_test,pred))
    print('AUC: ', roc_auc_score(y_test,pred))

    if plot:
        print('---'*20)
        precision, recall, umbral = precision_recall_curve(y_test,probas)

        print('Precison-Recall Curve: ')
        plt.figure(figsize=(15,8))
        plt.plot(precision,recall)
        for i in range(0,len(umbral),1000):
            plt.annotate("%.2f" % umbral[i],(precision[i],recall[i]))
        plt.ylabel('Recall')
        plt.xlabel('Precision')

        plt.show()


# ### Preprocessing

# In[ ]:


unique_videos = df[['title','comments_disabled','ratings_disabled']].drop_duplicates()

disabled = unique_videos.apply(lambda x: x['comments_disabled']+x['ratings_disabled'] != 0,axis=1)

print(disabled.value_counts())
print('Proportion: %.3f' %(disabled.value_counts()[1]/disabled.value_counts().sum()))

columns  = ['view_count',
 #'liked',
 #'ratings',
 'trending_time',
 #'ratings_disabled',
 'Month',
 #'DifferTime',
 'Hora'] + common_words_1

X = df.loc[unique_videos.index,columns]

snippet = df.loc[X.index,'snippet']

oh = OneHotEncoder(sparse=False)
oh.fit(snippet.values.reshape(-1,1))
snippet = pd.DataFrame(data = oh.transform(snippet.values.reshape(-1,1)),index=snippet.index,columns=oh.get_feature_names_out())

X = pd.concat([X,snippet],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,disabled,test_size=0.2,stratify=disabled)

#train_titles = X_train['title']
#test_titles = X_test['title']
#X_train = df.query('title in @train_titles')[columns]
#X_test = df.query('title in @test_titles')[columns]


# In[ ]:


columns_scaled  = ['view_count',
 'trending_time',
 'Month',
 #'DifferTime',
 'Hora'] #+ common_words_1

scaler = StandardScaler()

X_train[columns_scaled] = scaler.fit_transform(X_train[columns_scaled])
X_test[columns_scaled] = scaler.transform(X_test[columns_scaled])


# ### RandomForest

# In[ ]:


rf = RandomForestClassifier(n_estimators=1000,
                            max_depth=None,
                            criterion='entropy',
)


# In[ ]:


rf.fit(X_train,y_train)
plot_results(X_test,y_test,rf,umbral=0.5,plot=True)


# In[ ]:


plot_results(X_test,y_test,rf,umbral=0.33,plot=False)


# #### Grid Search

# In[ ]:


estimator = RandomForestClassifier()

params = {'n_estimators':(400,600,800),
          'max_depth':(5,10,25)}

grid_search = GridSearchCV(estimator=estimator,param_grid=params,cv=2,verbose=2)


# In[ ]:


grid_search.fit(X_train,y_train)


# ### Logistic Classification

# In[ ]:


clf = LogisticRegression(C=1,random_state=0, class_weight='balanced',solver='newton-cg')

clf.fit(X_train,y_train)


# In[ ]:


plot_results(X_test,y_test,clf,plot=True)


# ### SVC

# In[ ]:


svc = LinearSVC(class_weight='balanced',loss="squared_hinge",penalty="l2")

svc.fit(X_train,y_train)


# In[ ]:


pred = svc.predict(X_test)
confusion_matrix(y_test,pred*1)


# ### Near Miss

# In[ ]:


nm = NearMiss()

X_nm, y_nm = nm.fit_resample(X_train, y_train)

print('Original dataset shape:', Counter(y_train))
print('Resample dataset shape:', Counter(y_nm))


# In[ ]:


rf.fit(X_nm,y_nm)
plot_results(X_test,y_test,rf,umbral=0.5,plot=True)


# In[ ]:


svc.fit(X_nm,y_nm)
pred = svc.predict(X_test)
confusion_matrix(y_test,pred*1)


# ### SMOTE

# In[ ]:


smote = SMOTE()

X_smote, y_smote = smote.fit_resample(X_train, y_train)

print('Original dataset shape', Counter(y_train))
print('Resample dataset shape', Counter(y_smote))


# In[ ]:


rf = RandomForestClassifier(n_estimators=800,
                            max_depth=25,
                            class_weight={0:.9, 1:.1}
)

rf.fit(X_smote,y_smote)
plot_results(X_test,y_test,rf,umbral=0.5,plot=True)

#### Peores resultados que sin aplicar SMOTE


# In[ ]:


svc.fit(X_smote,y_smote)
pred = svc.predict(X_test)
confusion_matrix(y_test,pred*1)


# ### Neural Network

# In[ ]:


def build_model():
    inp = Input(shape=(X_train.shape[-1]))
    x = Dense(256,activation='relu')(inp)
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.2)(x)
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.2)(x)
    x=Dense(1024,activation='relu')(x)
    x=Dropout(0.2)(x)
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.2)(x)
    x=Dense(128,activation='relu')(x)
    x=Dropout(0.2)(x)
    x=Dense(64,activation='relu')(x)
    x=Dense(1,activation='sigmoid')(x)

    return Model(inputs=inp,outputs=x)


# In[ ]:


model = build_model()

optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy',optimizer=optimizer)

class_weight={0: 0.05, 1: 0.95}

model.fit(X_train*1,y_train*1,epochs=5,batch_size=64,validation_split=0.2,class_weight=class_weight)


# In[ ]:


plot_results(X_test*1,y_test*1,model,umbral=0.5,proba=True,plot=True)


# In[ ]:


model = build_model()

optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy',optimizer=optimizer)

class_weight={0: 0.9, 1: 0.1}

model.fit(X_nm*1,y_nm*1,epochs=5,batch_size=16,validation_split=0.1,class_weight=class_weight)
plot_results([X_test.iloc[:,:5]*1,X_test.iloc[:,5:]],y_test,model,umbral=0.5,proba=True,plot=True)


# In[ ]:


### SMOTE improve the results of the neural network by balancing the weights of the classes.

model = build_model()

optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy',optimizer=optimizer)

class_weight={0: 0.9, 1: 0.1}

model.fit(X_smote*1,y_smote*1,epochs=10,batch_size=128,validation_split=0.2,class_weight=class_weight)
plot_results(X_test*1,y_test,model,umbral=0.5,proba=True,plot=True)


# # 6. Likes Regression Model
# Best: XGBoost

# ### General Preprocesing

# In[ ]:


unique_videos = df[['title','videos_id']].drop_duplicates()

X_train,X_test = train_test_split(unique_videos,test_size=0.2)

columns  = ['view_count',
 'comment_count',
 'comments_disabled',
 'ratings_disabled',
 'Month',
 'trending_time',
 'Hora'] + common_words_1

#train_titles = X_train['title']
#test_titles = X_test['title']
#X_train = df.query('title in @train_titles')[columns]
#X_test = df.query('title in @test_titles')[columns]
X_train = df.loc[X_train.index,columns]
X_test = df.loc[X_test.index,columns]

y_train = df.loc[X_train.index,'likes']
y_test = df.loc[X_test.index,'likes']


# In[ ]:


columns_scaled  = ['view_count',
 'trending_time',
 'comment_count',
 'Month',
 #'DifferTime',
 'Hora'] #+ common_words_1

X_scaler = StandardScaler()

X_train[columns_scaled]=X_scaler.fit_transform(X_train[columns_scaled])
X_test[columns_scaled]=X_scaler.transform(X_test[columns_scaled])

y_scaler = StandardScaler()

y_train_=y_scaler.fit_transform(y_train.values.reshape(-1,1))
y_test_=y_scaler.transform(y_test.values.reshape(-1,1))


# ## Linear Regression

# In[ ]:


regr = LinearRegression()
# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_test_pred = regr.predict(X_test)
y_train_pred = regr.predict(X_train)


print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# ## Lasso Regression

# In[ ]:


lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_train_pred = lasso.predict(X_train)
y_test_pred= lasso.predict(X_test)
#print(lasso.coef_)
print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# ## XGBoost

# In[ ]:


xgb = xgboost.XGBRegressor(#colsample_bytree=0.4,
                # gamma=0,                 
                 learning_rate=0.15,
                 max_depth=10,
                # min_child_weight=1.5,
                 n_estimators=1000,                                                                    
                # reg_alpha=0.75,
                # reg_lambda=0.45,
                # subsample=0.6,
                 seed=42)
xgb.fit(X_train,y_train)
y_test_pred = xgb.predict(X_test)
y_train_pred= xgb.predict(X_train)

print('RMSE train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(y_train, y_train_pred)),
        np.sqrt(mean_squared_error(y_test, y_test_pred))))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# ## RandomForest

# In[ ]:


rfreg = RandomForestRegressor(max_depth=10,
                              n_estimators=700,
                              #max_features='log2',
                              random_state=0)

rfreg.fit(X_train,y_train_)
y_test_pred = rfreg.predict(X_test)
y_train_pred= rfreg.predict(X_train)

print('MSE train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(y_train, y_scaler.inverse_transform(y_train_pred.reshape(-1,1)))),
        np.sqrt(mean_squared_error(y_test, y_scaler.inverse_transform(y_test_pred.reshape(-1,1))))))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train_, y_train_pred),
        r2_score(y_test_, y_test_pred)))


# ## Neural Network

# In[ ]:


def build_model():
  inp = Input(shape=(X_train.shape[-1]))
  x = Dense(256,activation='relu')(inp)
  x=Dense(512,activation='relu')(x)
  x=Dropout(0.2)(x)
  x=Dense(512,activation='relu')(x)
  x=Dropout(0.2)(x)
  x=Dense(1024,activation='relu')(x)
  x=Dropout(0.2)(x)
  x=Dense(512,activation='relu')(x)
  x=Dropout(0.2)(x)
  x=Dense(128,activation='relu')(x)
  x=Dropout(0.2)(x)
  x=Dense(64,activation='relu')(x)
  x=Dense(1)(x)

  return Model(inputs=inp,outputs=x)

model = build_model()
model.summary()


# In[ ]:


optimizer = Adam(learning_rate=1e-4)
model.compile(loss='mse',optimizer=optimizer)


# In[ ]:


model.fit(x=X_train*1,
          y=y_train_,
          epochs=20,
          batch_size=128,
          validation_data=(X_test*1,y_test_))


# In[ ]:


y_test_pred = model.predict(X_test*1)
y_train_pred= model.predict(X_train*1)

print('RMSE train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(y_train, y_scaler.inverse_transform(y_train_pred.reshape(-1,1)))),
        np.sqrt(mean_squared_error(y_test, y_scaler.inverse_transform(y_test_pred.reshape(-1,1))))))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train_, y_train_pred),
        r2_score(y_test_, y_test_pred)))


# 
# # 7. Ratio Likes/Dislikes Regression
# 
# Best: Neural Network

# ### Preprocessing

# In[ ]:


unique_videos = df[['title','liked']].drop_duplicates()

X_train,X_test = train_test_split(unique_videos,test_size=0.2)

columns  = ['view_count',
 'comment_count',
 'comments_disabled',
 'ratings_disabled',
 'Month',
 'trending_time',
 'Hora'] + common_words_1

#train_titles = X_train['title']
#test_titles = X_test['title']
#X_train = df.query('title in @train_titles')[columns]
#X_test = df.query('title in @test_titles')[columns]
X_train = df.loc[X_train.index,columns]
X_test = df.loc[X_test.index,columns]

y_train = df.loc[X_train.index,'liked']
y_test = df.loc[X_test.index,'liked']


# In[ ]:


columns_scaled  = ['view_count',
 'comment_count',
 'Month',
 'trending_time',
 'Hora'] #+ common_words_1

X_scaler = StandardScaler()

X_train[columns_scaled]=X_scaler.fit_transform(X_train[columns_scaled])
X_test[columns_scaled]=X_scaler.transform(X_test[columns_scaled])

y_scaler = StandardScaler()

y_train_=y_scaler.fit_transform(y_train.values.reshape(-1,1))
y_test_=y_scaler.transform(y_test.values.reshape(-1,1))


# ### XGBoost

# In[ ]:


xgb = xgboost.XGBRegressor(#colsample_bytree=0.4,
                # gamma=0,
                 learning_rate=0.07,
                 max_depth=5,
                # min_child_weight=1.5,
                 n_estimators=700,
                # reg_alpha=0.75,
                # reg_lambda=0.45,
                # subsample=0.6,
                 seed=42)
xgb.fit(X_train,y_train)
y_test_pred = xgb.predict(X_test)
y_train_pred= xgb.predict(X_train)

print('RMSE train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(y_train, y_scaler.inverse_transform(y_train_pred.reshape(-1,1)))),
        np.sqrt(mean_squared_error(y_test, y_scaler.inverse_transform(y_test_pred.reshape(-1,1))))))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# ### Neural Network

# In[ ]:


def build_model(optimizer = 'adam',lr=1e-4):
      inp = Input(shape=(X_train.shape[-1]))
      x=Dense(256,kernel_initializer="he_uniform",kernel_regularizer=None)(inp)
      x_1=LeakyReLU(0)(x)
      #x_1=BatchNormalization()(x)

      x=Dense(512,kernel_initializer="he_uniform",kernel_regularizer=None)(x_1)
      x=LeakyReLU(0)(x)
      x=Dropout(0.2)(x)
      #x=BatchNormalization()(x)
      x=Dense(512,kernel_initializer="he_uniform",kernel_regularizer=None)(x)
      x=LeakyReLU(0)(x) 
      x=Dropout(0.2)(x)

      y=Dense(512,kernel_initializer="he_uniform",kernel_regularizer=None)(x_1)
      y=LeakyReLU(0)(y)
      y=Dropout(0.2)(y)

      x=Add()([x,y])
      #x=BatchNormalization()(x)
      x=Dense(1024,kernel_initializer="he_uniform",kernel_regularizer=None)(x)
      x=LeakyReLU(0)(x)
      x=Dropout(0.2)(x)
      #x=BatchNormalization()(x)
      x=Dense(512,kernel_initializer="he_uniform",kernel_regularizer=None)(x)
      x=LeakyReLU(0)(x)
      x=Dropout(0.2)(x)

      x1=Dense(512,kernel_initializer='he_uniform')(x_1)
      x1=LeakyReLU(0)(x1)
      x1=Dropout(0.2)(x1)

      x=Add()([x,x1])
      #x=BatchNormalization()(x)
      x=Dense(128,kernel_initializer="he_uniform")(x)
      x=LeakyReLU(0)(x)
      x=Dropout(0.2)(x)
      #x=BatchNormalization()(x)
      x=Dense(64,kernel_initializer="he_uniform")(x)
      x=Reshape((64,1))(x)
      #x=LeakyReLU(0.2)(x)
      #x=BatchNormalization()(x)
      #x=Dense(1)(x)
      x=GlobalAveragePooling1D(keepdims=False)(x)

      model = Model(inputs=inp,outputs=x)

      model.summary()
      if optimizer == 'adam':
        model_opt = Adam(learning_rate=lr)
      elif optimizer == 'rmsprop':
        model_opt = RMSprop(learning_rate=lr)
      else:
        return print('[!]Error: Choose a valid optimizer (adam, rmsprop)')

      model.compile(loss='mse',optimizer=model_opt,metrics=[RootMeanSquaredError()])
      return model


# In[ ]:


model = build_model(lr=0.0001)


# In[ ]:


model.fit(x=X_train*1,
          y=y_train_,
          epochs=30,
          batch_size=128,
          validation_data=(X_test*1,y_test_))

model.fit(x=X_train*1,
          y=y_train_,
          epochs=5,
          batch_size=512,
          validation_data=(X_test*1,y_test_))


# In[ ]:


y_test_pred = model.predict(X_test*1)
y_train_pred= model.predict(X_train*1)

print('RMSE train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(y_train, y_scaler.inverse_transform(y_train_pred.reshape(-1,1)))),
        np.sqrt(mean_squared_error(y_test, y_scaler.inverse_transform(y_test_pred.reshape(-1,1))))))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train_, y_train_pred),
        r2_score(y_test_, y_test_pred)))


# In[ ]:


print(y_train.mean(),y_train.std())
print(y_train_.mean(),y_train_.std())


# ### Random Forest

# In[ ]:


rf = RandomForestRegressor(n_estimators = 600,
                           max_depth = 25)


# In[ ]:


rf.fit(X_train,y_train)


# In[ ]:


y_test_pred = rf.predict(X_test)
y_train_pred= rf.predict(X_train)

print('RMSE train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(y_train, y_train_pred)),
        np.sqrt(mean_squared_error(y_test, y_test_pred))))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# In[ ]:


y_test_pred = rf.predict(X_test)
y_train_pred= rf.predict(X_train)

print('RMSE train: %.3f, test: %.3f' % (
        np.sqrt(mean_squared_error(y_train, y_train_pred)),
        np.sqrt(mean_squared_error(y_test, y_test_pred))))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# # 8. Clustering

# ### Preprocessing

# In[ ]:


unique_videos = df[['title','videos_id']].drop_duplicates()

columns  = ['view_count',
 'ratings',
 'liked',
 'comment_count',
 'comments_disabled',
 'ratings_disabled',
 'Month',
 'trending_time',
 'Hora'] + common_words_1


X = df.loc[unique_videos.index,columns]

countries = df.loc[X.index,'country']
oh = OneHotEncoder(sparse=False)
countries = pd.DataFrame(oh.fit_transform(countries.values.reshape(-1,1)),index=countries.index,columns=oh.get_feature_names())

X = pd.concat([X,countries],axis=1)

y = df.loc[X.index,['country','snippet']]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# ### KMeans

# In[ ]:


pca_kmeans = make_pipeline(StandardScaler(),
                           PCA(n_components=2),
                           KMeans(n_clusters=7, init='k-means++', random_state=0))


# In[ ]:


pca_kmeans.fit(X_train)


# In[ ]:


pred = pca_kmeans.predict(X_test)

X_test_pca = pca_kmeans[0].transform(X_test)
X_test_pca = pca_kmeans[1].transform(X_test_pca)

X_test_pca.shape


# In[ ]:


plt.scatter(x=X_test_pca[:,0],y=X_test_pca[:,1],c=pred)


# In[ ]:


enc = LabelEncoder()
y_test_enc = enc.fit_transform(y_test['country'])

plt.scatter(x=X_test_pca[:,0],y=X_test_pca[:,1],c=y_test_enc)


# ### Gaussian Mixture

# #### PCA

# In[ ]:



pca_gmm = make_pipeline(StandardScaler(),
                           PCA(n_components=2),
                           GaussianMixture(n_components=7))


# In[ ]:


pca_gmm.fit(X_train)


# In[ ]:


pred = pca_gmm.predict(X_test)

X_test_pca = pca_gmm[0].transform(X_test)
X_test_pca = pca_gmm[1].transform(X_test_pca)

X_test_pca.shape


# In[ ]:


plt.scatter(x=X_test_pca[:,0],y=X_test_pca[:,1],c=pred)


#  #### LDA

# In[ ]:


from sklearn.mixture import GaussianMixture
lda_gmm = make_pipeline(StandardScaler(),
                           LDA(n_components=2),
                           GaussianMixture(n_components=13))


# In[ ]:


lda_gmm.fit(X_train,y_train.loc[:,'snippet'])


# In[ ]:


pred = lda_gmm.predict(X_test)

X_test_lda = lda_gmm[0].transform(X_test)
X_test_lda = lda_gmm[1].transform(X_test_lda)

enc = LabelEncoder()
y_test_enc = enc.fit_transform(y_test['snippet'])

fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,8))

ax1.scatter(x=X_test_lda[:,0],y=X_test_lda[:,1],c=pred)
ax1.set_title('GMM prediction')

ax2.scatter(x=X_test_lda[:,0],y=X_test_lda[:,1],c=y_test_enc)
ax2.set_title('Real categories labels')

plt.show()


# ### t-SNE

# In[ ]:


from sklearn.manifold import TSNE

tsne_gmm = make_pipeline(StandardScaler(),
                           TSNE(n_components=2))
gmm = GaussianMixture(n_components=13)


# In[ ]:


#X_train_tsne = tsne_gmm.fit_transform(X_train)

gmm = GaussianMixture(n_components=13)
gmm.fit(X_train_tsne)


# In[ ]:



pred = gmm.predict(X_train_tsne)

plt.scatter(x=X_train_tsne[:,0],y=X_train_tsne[:,1],c=pred)


# In[ ]:


enc = LabelEncoder()
y_train_enc = enc.fit_transform(y_train['snippet'])

plt.scatter(x=X_train_tsne[:,0],y=X_train_tsne[:,1],c=y_train_enc)


# # 9. Recommendation System

# ### Preprocessing

# In[ ]:


codes = {'FR':'0001','DE':'0010','GB':'0011','US':'0100','CA':'0101','MX':'0110','IN':'0111','RU':'1000','JP':'1001','KR':'1010'}

def codcountry(x):
    number = codes[x]
    list_codes = list()
    for num in number:
        list_codes.append(int(num))
    return list_codes



df['country_'] = df['country'].apply(lambda x: codcountry(x))

df['country_1'] = df['country_'].apply(lambda x: x[0])
df['country_2'] = df['country_'].apply(lambda x: x[1])
df['country_3'] = df['country_'].apply(lambda x: x[2])
df['country_4'] = df['country_'].apply(lambda x: x[3])

df_ = df.query('video_error_or_removed == 0')


# In[ ]:


unique_videos = df_[['title','videos_id']].drop_duplicates()

columns  = ['view_count',
 'ratings',
 'liked',
 'comment_count',
 'comments_disabled',
 'ratings_disabled',
 'trending_time'] + common_words_1 + ['country_{}'.format(i+1) for i in range(4)]

X = df.loc[unique_videos.index,columns]


# ## KNN

# In[ ]:


kNN = NearestNeighbors(n_neighbors=2, algorithm="auto", metric='cosine')
kNN.fit(X.values*1)


# In[ ]:


video_id = 'VONVWm2X4JI'
video_indx = unique_videos.query('videos_id == @video_id').index[0]
print(video_indx)
indices = kNN.kneighbors(X.loc[video_indx,:].values.reshape(1,-1), return_distance=False,n_neighbors=2)
id1,id2 = indices[0]
print(df.loc[id1,:]['videos_id'])
print(df.loc[id2,:]['videos_id'])

