#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


train=pd.read_csv("../input/spaceship-titanic/train.csv")
test=pd.read_csv("../input/spaceship-titanic/test.csv")


# In[ ]:


train.head(3)


# In[ ]:


train=train.drop(["PassengerId","Name"],axis=1)
test=test.drop(["PassengerId","Name"],axis=1)


# In[ ]:


test.shape


# In[ ]:


train.shape


# In[ ]:


train.info()


# In[ ]:


test.info()


# In[ ]:


train[['deck', 'num', 'side']] = train['Cabin'].str.split('/', expand=True)
test[['deck', 'num', 'side']] = test['Cabin'].str.split('/', expand=True)
train=train.drop(["Cabin","num"],axis=1)
test=test.drop(["Cabin","num"],axis=1)


# In[ ]:


pd.crosstab(train.Transported,train.deck)


# In[ ]:


pd.crosstab(train.Transported,train.side)


# In[ ]:


pd.crosstab(train.Transported,train.HomePlanet)


# In[ ]:


pd.crosstab(train.Transported,train.Destination)


# In[ ]:


pd.crosstab(train.Transported,train.VIP)


# In[ ]:


pd.crosstab(train.Transported,train.Destination)


# In[ ]:


pd.crosstab(train.Transported,train.CryoSleep)


# In[ ]:


train_df=pd.get_dummies(train,drop_first=True)
test_df=pd.get_dummies(test,drop_first=True)


# In[ ]:


test_df.shape
#test.shape


# In[ ]:


from sklearn import preprocessing
 
label_encoder1 = preprocessing.LabelEncoder()
label_encoder2 = preprocessing.LabelEncoder()
train_df['Transported']= label_encoder1.fit_transform(train_df['Transported'])


# In[ ]:


train_df.head(5)


# In[ ]:


train_df.isna().sum()


# In[ ]:


test_df.isna().sum()


# In[ ]:


df=train_df.copy()


# In[ ]:


from scipy.stats import chi2_contingency 
import numpy as np
chisqt = pd.crosstab(df.VIP_True,df.Transported, margins=True)
value = np.array([chisqt.iloc[0][0:5].values,
                  chisqt.iloc[1][0:5].values])
print(chi2_contingency(value)[0:3])
print("x2 , p value , degree of freedom.")


# H o red aralarında ilişki var

# In[ ]:


from scipy.stats import chi2_contingency 
import numpy as np
chisqt = pd.crosstab(df.CryoSleep_True,df.Transported, margins=True)
value = np.array([chisqt.iloc[0][0:5].values,
                  chisqt.iloc[1][0:5].values])
print(chi2_contingency(value)[0:3])
print("x2 , p value , degree of freedom.")


# H o red aralarında ilişki var

# In[ ]:


from scipy.stats import chi2_contingency 
import numpy as np
chisqt = pd.crosstab(df.HomePlanet_Europa,df.Transported, margins=True)
value = np.array([chisqt.iloc[0][0:5].values,
                  chisqt.iloc[1][0:5].values])
print(chi2_contingency(value)[0:3])
print("x2 , p value , degree of freedom.")


# H o red aralarında ilişki var

# In[ ]:


from scipy.stats import chi2_contingency 
import numpy as np
chisqt = pd.crosstab(df.HomePlanet_Mars,df.Transported, margins=True)
value = np.array([chisqt.iloc[0][0:5].values,
                  chisqt.iloc[1][0:5].values])
print(chi2_contingency(value)[0:3])
print("x2 , p value , degree of freedom.")


# In[ ]:


train_df.info()


# In[ ]:


test_df.info()


# In[ ]:


train_df.shape


# In[ ]:


test_df.shape


# In[ ]:


a=train_df.select_dtypes(["float64"]).fillna(train_df.select_dtypes(["float64"]).mean())
b=train_df.select_dtypes(["uint8"]).fillna(train_df.select_dtypes(["uint8"]).mode())
kümelemetrain=pd.concat([a,b],axis=1)
c=test_df.select_dtypes(["float64"]).fillna(test_df.select_dtypes(["float64"]).mean())
d=test_df.select_dtypes(["uint8"]).fillna(test_df.select_dtypes(["uint8"]).mode())
kümelemetest=pd.concat([c,d],axis=1)


# In[ ]:


test_df.head()


# In[ ]:





# In[ ]:


kümelemetrain.shape


# In[ ]:


kümelemetrain.isna().sum()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler,StandardScaler

mntrain=StandardScaler()
mntest=StandardScaler()


# In[ ]:


x1=kümelemetrain.select_dtypes("float64")
x2=kümelemetest.select_dtypes("float64")


# In[ ]:


x1.head(2)


# In[ ]:


x2.head(2)


# In[ ]:


x1=mntrain.fit_transform(x1)
x2=mntest.fit_transform(x2)


# In[ ]:


from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=4,init="k-means++")
kmeans.fit(x1)
predicttrain = kmeans.predict(x1)


# In[ ]:


kmeans=KMeans(n_clusters=4,init="k-means++")
kmeans.fit(x2)
predicttest = kmeans.predict(x2)


# In[ ]:


predicttrain=pd.DataFrame(predicttrain)
predicttest=pd.DataFrame(predicttest)


# In[ ]:


train_df["küme"]=predicttrain
test_df["küme"]=predicttest


# In[ ]:


train_df.küme.value_counts()


# In[ ]:


test_df.küme.value_counts()


# In[ ]:


train_df.isna().sum()


# In[ ]:


test_df.isna().sum()


# In[ ]:


küme0=dict(train_df[train_df.küme==0].mean())
küme1=dict(train_df[train_df.küme==1].mean())
küme2=dict(train_df[train_df.küme==2].mean())
küme3=dict(train_df[train_df.küme==3].mean())


# In[ ]:


tküme0=dict(test_df[test_df.küme==0].mean())
tküme1=dict(test_df[test_df.küme==1].mean())
tküme2=dict(test_df[test_df.küme==2].mean())
tküme3=dict(test_df[test_df.küme==3].mean())


# In[ ]:


train_df[train_df.küme==0]=train_df[train_df.küme==0].fillna(value=küme0)
train_df[train_df.küme==1]=train_df[train_df.küme==1].fillna(value=küme1)
train_df[train_df.küme==2]=train_df[train_df.küme==2].fillna(value=küme2)
train_df[train_df.küme==3]=train_df[train_df.küme==3].fillna(value=küme3)


# In[ ]:


test_df[test_df.küme==0]=test_df[test_df.küme==0].fillna(value=tküme0)
test_df[test_df.küme==1]=test_df[test_df.küme==1].fillna(value=tküme1)
test_df[test_df.küme==2]=test_df[test_df.küme==2].fillna(value=tküme2)
test_df[test_df.küme==3]=test_df[test_df.küme==3].fillna(value=tküme3)


# In[ ]:


train_df.isna().sum()


# In[ ]:


test_df.isna().sum()


# In[ ]:


x=train_df.iloc[:,:-1]
x=x.drop("Transported",axis=1)
y=train_df["Transported"]


# In[ ]:


train_df


# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import MinMaxScaler

mn=MinMaxScaler()

x_train = mn.fit_transform(x_train)
x_test = mn.fit_transform(x_test)


# In[ ]:


X_train, X_test,y_train_s,y_tes_t = train_test_split(x,y,test_size=0.33, random_state=0)


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.callbacks import EarlyStopping
model=Sequential()
model.add(Dense(units=30,activation="relu"))
model.add(Dropout(0.6)) #koyulan unit lerin %60 ını rastgele açıp kapatıyor
model.add(Dense(units=15,activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(units=15,activation="relu"))
model.add(Dropout(0.6))
model.add(Dense(units=1,activation="sigmoid")) #çıkış layerı

#model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
model.compile(optimizer='sgd',loss='mse',metrics=[tf.keras.metrics.BinaryAccuracy()])
earlyStopping=EarlyStopping(monitor="val_accuracy",mode="max",verbose=1,patience=25)
model.fit(x=x_train,y=y_train,epochs=700,validation_data=(x_test,y_test),verbose=1,callbacks=[earlyStopping])
"""


# In[ ]:


from xgboost import XGBClassifier
classifier = XGBClassifier()


# In[ ]:


y_train.value_counts()


# In[ ]:


y_test.value_counts()


# In[ ]:


classifier.fit(X_train,y_train)


# In[ ]:


xgb_pred=classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(xgb_pred,y_test)


# In[ ]:


#ysa_pred=pd.DataFrame(ysa_pred)
#ysa_pred=ysa_pred[0].map(lambda  x : 1 if (x >0.5) else 0)


# In[ ]:


#accuracy_score(ysa_pred,y_test)


# In[ ]:


#model.save("my_model")


# In[ ]:


#import keras


# In[ ]:


#model.save_weights("model.h5")


# In[ ]:


#model.summary()


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=4,metric="minkowski")


# In[ ]:


knn.fit(X_train,y_train)


# In[ ]:


knn_pred=knn.predict(X_test)


# In[ ]:


accuracy_score(knn_pred,y_test)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
naive_pred = gnb.fit(X_train, y_train).predict(X_test)


# In[ ]:


accuracy_score(naive_pred,y_test)


# In[ ]:


from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X_train,y_train)


# In[ ]:


r_pred=r_dt.predict(X_test)


# In[ ]:


r_pred=pd.DataFrame(r_pred)
r_pred=r_pred[0].map(lambda  x : 1 if (x >0.5) else 0)
accuracy_score(r_pred,y_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
logr= LogisticRegression(random_state=0)
logr.fit(X_train,y_train)


# In[ ]:


log_pred=logr.predict(X_test)


# In[ ]:


accuracy_score(log_pred,y_test)


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay
print("logit MODEL\n")
print(classification_report(y_test,log_pred))
cm = confusion_matrix(y_test,log_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# In[ ]:


print("XGB MODEL\n")
print(classification_report(y_test,xgb_pred))
cm = confusion_matrix(y_test,xgb_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


# In[ ]:


import plotly.express as px
df = px.data.tips()
# Here we use a column with categorical data
fig = px.histogram(train, x="Transported")
fig.show()


# In[ ]:


test_df.shape


# In[ ]:


test_df=test_df.drop("küme",axis=1)


# In[ ]:


#mn=MinMaxScaler()
#test_df=mn.fit_transform(test_df)


# In[ ]:


#y_predxgb=classifier.predict(test_df)
y_predlogit=logr.predict(test_df)
#y_predlogit=logr.predict(testdf)
#y_predknn=knn.predict(testdf) 
#_predr_dt=r_dt.predict(testdf)


# In[ ]:


y_predlogit


# In[ ]:





# In[ ]:


#Create a  DataFrame with the passengers ids and our prediction
submission_df = pd.read_csv("../input/spaceship-titanic/sample_submission.csv")
submission_df["Transported"] = y_predlogit
a={1:True,0:False}
submission_df["Transported"]=submission_df["Transported"].map(a)

submission_df.to_csv('submission.csv', index=False)


# In[ ]:


submission_df


# In[ ]:




