#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Importing data and libraries

# In[ ]:


from sklearn.model_selection import GridSearchCV,cross_val_score,train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score,recall_score,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rs = 42


# In[ ]:


url = '../input/tabular-playground-series-may-2022/'
df = pd.read_csv(url+"train.csv")
test = pd.read_csv(url+'test.csv')


# In[ ]:


df.info()


# In[ ]:


df.set_index("id",inplace=True)


# In[ ]:


test.set_index("id",inplace=True)


# In[ ]:


df.describe().loc[['min','max'],:]


# # Memory Reduction

# Since the number of training instances is large, model training and other processes will be slowed down. So it is best to reduce their sizes to the lowest level possible **without losing precision** of digits.

# In[ ]:


def reduce_memory(data,memory_size_int=8,memory_size_float=16):
    for col in data.columns:
        if str(data[col].dtype)[:1] == 'i':
            data[col] = data[col].astype(np.int8)
        elif str(data[col].dtype)[:1] == 'f':
            data[col] = data[col].astype(np.float16)
    return data
reduce_memory(data=df).info()


# In[ ]:


df = reduce_memory(data=df)


# In[ ]:


test = reduce_memory(data=test)


# # EDA

# In[ ]:


plt.figure(figsize=(10,8),dpi=200)
sns.heatmap(df.corr())


# In[ ]:


int_features = [col for col in df.columns if df[col].dtype==np.int8]
int_features.remove('target')
print(int_features)


# In[ ]:


float_features = [col for col in df.columns if df[col].dtype==np.float16]
print(float_features)


# In[ ]:


def plot_dists(data,cols,hue=None,bins=20,plot='hist'):
    n_rows = int(np.ceil(len(cols)/2))
    col = 0
    sns.set_style("whitegrid")
    fig,axes = plt.subplots(nrows=n_rows,ncols=2,figsize=(20,15))
    for i in range(n_rows):
        for j in range(2):
            if col > len(cols):
                axes[i][j].axis("off")
                break
            if plot=='hist':
                sns.histplot(x=cols[col],data=data,hue=hue,ax=axes[i][j],bins=bins)
                axes[i][j].set_title(f'Distribution of {cols[col]}',size=10)
            elif plot=='count':
                sns.countplot(x=cols[col],data=data,hue=hue,ax=axes[i][j])
                axes[i][j].set_title(f'Count of {cols[col]}',size=10)
            elif plot == 'kde':
                sns.kdeplot(x=cols[col],data=data,hue=hue,ax=axes[i][j])
                axes[i][j].set_title(f'Distribution of {cols[col]}',size=10)
            fig.tight_layout()
            col += 1


# In[ ]:


plot_dists(df,float_features,plot='hist',hue='target')


# In[ ]:


plot_dists(data=df,cols=int_features,plot='count',hue='target')


# In[ ]:


plot_dists(data=df,cols=float_features,plot='kde',hue='target')


# In[ ]:


df_corr = df.corr()


# In[ ]:


plt.figure(figsize=(8,6),dpi=150)
df_corr['target'].sort_values(ascending=False)[1:].plot(kind='bar')


# In[ ]:


pd.DataFrame(df_corr['target']).style.applymap(lambda x : f"color: {'red' if abs(x)<0.001 else 'green'}")


# *f_03, f_06, f_12* have very little correlation with target. Let's drop them

# In[ ]:


df.drop(['f_03','f_06','f_12'],axis=1,inplace=True)
test.drop(['f_03','f_06','f_12'],axis=1,inplace=True)


# In[ ]:


int_features = [col for col in df.columns if df[col].dtype==np.int8]
int_features.remove('target')
float_features = [col for col in df.columns if df[col].dtype==np.float16]


# # Feature Engineering and Data Preparation

# In[ ]:


df["f_27"].value_counts()


# In[ ]:


df['f_27'].str.split('',expand=True)


# In[ ]:


# adapted from https://www.kaggle.com/code/kotrying/tps22-05
df_split = df["f_27"].str.split('',expand=True).iloc[:,1:11]
df_split.columns = [f'f_27_{i}' for i in range(10)]
df_concat = pd.concat([df,df_split],axis=1)


# In[ ]:


test_split = test["f_27"].str.split('',expand=True).iloc[:,1:11]
test_split.columns = [f'f_27_{i}' for i in range(10)]
test_concat = pd.concat([test,test_split],axis=1)


# In[ ]:


df_concat.columns,test_concat.columns


# In[ ]:


def most_common(x):
    x = x.upper()
    set_x = set(x)
    count_high = 0
    letter_high = ''
    for letter in set_x:
        cnt = x.count(letter)
        if cnt>count_high:
            letter_high = letter
            count_high = cnt
    return letter_high
df['f_27'].apply(most_common).unique()
    


# In[ ]:


df = df_concat.copy()
df['most_common'] = df['f_27'].apply(most_common)


# In[ ]:


test = test_concat.copy()
test['most_common'] = test['f_27'].apply(most_common)


# In[ ]:


obj_feats = [col for col in df.columns if df[col].dtype==object]
obj_feats.remove('f_27')
obj_feats


# In[ ]:


plot_dists(data=df,plot='count',hue='target',cols=obj_feats)


# In[ ]:


df.shape


# In[ ]:


df.drop('f_27',axis=1,inplace=True)


# In[ ]:


test.drop('f_27',axis=1,inplace=True)


# In[ ]:


for col in df.columns:
    if df[col].nunique()<=10:
        print(col)


# In[ ]:


df.loc[:,'f_27_0':]


# In[ ]:


df.loc[:,'f_27_0':].info()


# In[ ]:


for col in df.loc[:,'f_27_0':].columns:
    df[col] = df[col].apply(lambda x:ord(x)-64)
df.loc[:,'f_27_0':]


# In[ ]:


for col in test.loc[:,'f_27_0':].columns:
    test[col] = test[col].apply(lambda x:ord(x)-64)
test.loc[:,'f_27_0':]


# In[ ]:


df.columns


# In[ ]:


X,y = df.drop(['target'],axis=1),df["target"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=rs)


# In[ ]:


X_train.shape


# In[ ]:


X_train = X_train.values
X_test = X_test.values


# # Modelling

# ## Classical ML models

# ### DecisionTree and RandomForest

# In[ ]:


dtr = DecisionTreeClassifier(max_depth=5)
scores = cross_val_score(estimator=dtr,X=X_train,y=y_train,cv=5)


# In[ ]:


print(scores)


# In[ ]:


rf = RandomForestClassifier(n_estimators=20)
scores = cross_val_score(estimator=rf,X=X_train,y=y_train,cv=5)
print(scores)


# In[ ]:


rf.get_params()


# ### RandomForest GridSearch

# In[ ]:


param_grid = {
    'criterion' : ['gini','entropy'],
    'max_depth' : [4,5,6],
    'n_estimators' : [100,150]
}

gs = GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_grid,cv=5)
gs = gs.fit(X_train,y_train)


# In[ ]:


gs.best_score_


# In[ ]:


gs.best_estimator_


# In[ ]:


gs.best_params_


# ## Monitoring Overfitting

# In[ ]:


def plot_2(x,y,z,label_y,label_z,title):
    plt.plot(x,y,'b-',label=label_y)
    plt.plot(x,z,'r--',label=label_z)
    plt.title(title)
    plt.legend()
def train_validation_curve_for_rf(X,y,val_size=0.3,rs=42,epochs=10,n_estimators=False,max_depth=True,
                                 max_depth_start=6,n_estimators_start=100,criterion='gini'):
    from sklearn.metrics import accuracy_score,precision_score,recall_score
    X_tr,X_val,y_tr,y_val = train_test_split(X,y,test_size=val_size,random_state=rs)
    acc_tr,acc_val = [],[]
    pr_tr,pr_val = [],[]
    rec_tr,rec_val = [],[]
    if max_depth:
        for depth in range(max_depth_start,max_depth_start+epochs):
            rf = RandomForestClassifier(criterion=criterion,max_depth=depth,n_estimators=100)
            rf = rf.fit(X_tr,y_tr)
            pred_tr = rf.predict(X_tr)
            pred_val = rf.predict(X_val)
            acc_tr.append(accuracy_score(y_tr,pred_tr))
            acc_val.append(accuracy_score(y_val,pred_val))
            pr_tr.append(precision_score(y_tr,pred_tr))
            pr_val.append(precision_score(y_val,pred_val))
            rec_tr.append(recall_score(y_tr,pred_tr))
            rec_val.append(recall_score(y_val,pred_val))
        fig,ax = plt.subplots(nrows=3,figsize=(10,8))
        plt.sca(ax[0])
        plot_2([x for x in range(max_depth_start,max_depth_start+epochs)],acc_tr,acc_val,'train','val','train/val accuracy')
        plt.sca(ax[1])
        plot_2([x for x in range(max_depth_start,max_depth_start+epochs)],pr_tr,pr_val,'train','val','train/val precision')
        plt.sca(ax[2])
        plot_2([x for x in range(max_depth_start,max_depth_start+epochs)],rec_tr,rec_val,'train','val','train/val recall')
    elif n_estimators:
        for n_estimator in range(n_estimators_start,n_estimators_start+epochs,int(np.ceil(epochs/10))):
            rf = RandomForestClassifier(criterion=criterion,max_depth=6,n_estimators=n_estimator)
            rf = rf.fit(X_tr,y_tr)
            pred_tr = rf.predict(X_tr)
            pred_val = rf.predict(X_val)
            acc_tr.append(accuracy_score(y_tr,pred_tr))
            acc_val.append(accuracy_score(y_val,pred_val))
            pr_tr.append(precision_score(y_tr,pred_tr))
            pr_val.append(precision_score(y_val,pred_val))
            rec_tr.append(recall_score(y_tr,pred_tr))
            rec_val.append(recall_score(y_val,pred_val))
        fig,ax = plt.subplots(nrows=3,figsize=(10,8))
        plt.sca(ax[0])
        plot_2([x for x in range(n_estimators_start,n_estimators_start+epochs,int(np.ceil(epochs/10)))],
               acc_tr,acc_val,'train','val','train/val accuracy')
        plt.sca(ax[1])
        plot_2([x for x in range(n_estimators_start,n_estimators_start+epochs,int(np.ceil(epochs/10)))],
               pr_tr,pr_val,'train','val','train/val precision')
        plt.sca(ax[2])
        plot_2([x for x in range(n_estimators_start,n_estimators_start+epochs,int(np.ceil(epochs/10)))],
               rec_tr,rec_val,'train','val','train/val recall')


# In[ ]:


train_validation_curve_for_rf(X=X_train,y=y_train,epochs=20)


# In[ ]:


train_validation_curve_for_rf(X=X_train,y=y_train,n_estimators=True,n_estimators_start=90,
                       epochs=100,max_depth=False)


# From the above plots **max_depth of 14** and **n_estimators of 160** should be the ideal parameters for the baseline model

# ## DL model

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[ ]:


def build_model(units=128,lower_thresh=16,activation="selu",kernel_initializer="lecun_normal",
               optimizer="adam",metrics="accuracy"):
    model = tf.keras.models.Sequential()
    while units>=lower_thresh:
        model.add(layers.Dense(units=units,activation=activation,kernel_initializer=kernel_initializer))
        units = int(units/2)
    model.add(layers.Dense(1,activation="sigmoid"))
    
    model.compile(loss="binary_crossentropy",optimizer=optimizer,
                 metrics=metrics)
    
    return model


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# In[ ]:


model_1 = build_model()


# In[ ]:


history_1 = model_1.fit(X_train_sc,y_train,epochs=20,validation_split=0.2)


# In[ ]:


model_1_df = pd.DataFrame(history_1.history)


# In[ ]:


model_1_df.plot(figsize=(10,6))
plt.title("epochs=20");


# In[ ]:


model_2 = build_model(units=256)
early_cb = tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
history_2 = model_2.fit(X_train_sc,y_train,epochs=40,
                       validation_split=0.2,
                       callbacks=[early_cb])


# In[ ]:


model_2_df = pd.DataFrame(history_2.history)


# In[ ]:


model_2_df.plot(figsize=(10,5))
plt.title("epochs=40, stopped at 27",size=20)


# In[ ]:


model_3 = build_model(units=256,optimizer="nadam")
history_3 = model_3.fit(X_train_sc,y_train,epochs=20,validation_split=0.2)


# In[ ]:


model_3_df = pd.DataFrame(history_3.history)


# In[ ]:


model_3_df.plot(figsize=(10,5))
plt.title("nadam",size=20)


# In[ ]:


model_1_df.val_accuracy.max(),model_2_df.val_accuracy.max(),model_3_df.val_accuracy.max()


# In[ ]:


model_4 = build_model(units=256,optimizer="nadam",kernel_initializer="he_normal",activation="elu")
history_4 = model_4.fit(X_train_sc,y_train,epochs=20,validation_split=0.2)


# In[ ]:


model_4_df = pd.DataFrame(history_4.history)
model_4_df.plot(figsize=(10,5))
plt.title("nadam+elu",size=20)


# In[ ]:


sc = StandardScaler()
X_train_sc = sc.fit_transform(np.r_[X_train,X_test])


# In[ ]:


nn_model = build_model(units=256,optimizer="nadam")
early_cb = tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
nn_model.fit(X_train_sc,np.r_[y_train,y_test],epochs=30,
            callbacks=[early_cb])


# In[ ]:


test


# In[ ]:


final_test = test.values


# In[ ]:


final_test = sc.transform(final_test)


# In[ ]:


pred = nn_model.predict(final_test)


# In[ ]:


test["target"] = pred


# In[ ]:


sub = test.loc[:,"target":]


# In[ ]:


sub.to_csv("submission.csv")


# In[ ]:




